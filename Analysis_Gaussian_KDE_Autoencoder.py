from alabtools import analysis
from alabtools import geo
import sys
import pickle
import numpy as np
import scipy.stats as stat
import scipy.spatial.distance as dist
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage import maximum_filter
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib import path
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.neighbors import KDTree


def gaussian_kde(vector):
    kernel = stat.gaussian_kde(vector)
    kernel_pdf = kernel.evaluate(vector)

    return kernel, kernel_pdf
# Perform kernel density estimation#


def grid_construction(vector, kernel):
    xmin = vector[0].min()
    xmax = vector[0].max()
    ymin = vector[1].min()
    ymax = vector[1].max()

    x, y = np.mgrid[xmin:xmax:1000j, ymin:ymax:1000j]
    grid = np.vstack((x.ravel(), y.ravel()))

    z = np.reshape(kernel(grid).T, (1000, 1000))

    return x, y, z
# Construct grids and corresponding density values#


def structure_construction(peak, coord):

    return coord[:, peak, :]
# Obtain structure#


def detecting_maxima(x, y, z, e):
    maxima = []
    maximad = []
    local_structure = maximum_filter(z, size=(5, 5))
    binary_structure = (local_structure == z) ^ (z == 0)
    detected_maxima = np.where(binary_structure == True)
    
    for i in range(len(detected_maxima[0])):
        x0 = detected_maxima[0][i]
        y0 = detected_maxima[1][i]
        xd = x[x0][0]
        yd = y.T[y0][0]
        if z[x0][y0] >= e:
            maxima.append([x0, y0])
            maximad.append([xd, yd])
    
    return np.array(maxima), np.array(maximad)
# Detect local maximas among the constructed grids#


def detecting_boundary(vector, maxima, maximad):
    pa = []
    ind = []
    temp = np.copy(maxima)
    tempd = np.copy(maximad)
    cs = sns.kdeplot(vector[0], vector[1], n_levels=100)

    for i in range(len(maximad)):
        tag = 0
        for e in range(1, len(cs.collections)):
            p = cs.collections[e].get_paths()
            for p0 in p:
                t = p0.contains_points(tempd)
                if t.tolist().count(True) <= 1 and t[i] == True:
                    tag = 1
                    break
            if tag == 1:
                break
        if tag == 1:
            ind.append(i)

    temp = temp[ind]
    tempd = tempd[ind]
    
    for i in range(len(tempd)):
        tag = 0
        for e in range(1, len(cs.collections)):
            p = cs.collections[e].get_paths()
            for p0 in p:
                t = p0.contains_points(tempd)
                if t.tolist().count(True) <= 1 and t[i] == True:
                    pa.append(p0)
                    tag = 1
                    break
            if tag == 1:
                break
        if tag == 0:
            pa.append(path.Path([[0, 0]]))
            
    return pa, temp, tempd
# Detect cluster boundary#


def detecting_peaks(maxima, x, y, vector, kernel_pdf):
    peak = []
    n = 0
    tree = KDTree(vector.T)

    for grid in maxima:
        xi = x[grid[0]][0]
        yi = y[0][grid[1]]
        distance, ind = tree.query([[xi, yi]], k=1)
        peak.append(ind[0][0])
        n += 1
    
    return peak, n
# Detect peak for each cluster#


def detecting_occupancy(pa, peak, vector):
    tag = np.where(pa.contains_points(vector.T) == True)[0]
    if len(tag) > 0:
        sub_vector = vector[:, tag]
        tree = KDTree(sub_vector.T)
        _, ind = tree.query([vector.T[peak]], k=len(tag))
        new_tag = tag[ind[0]]
    else:
        new_tag = tag
    
    return vector[:, new_tag], new_tag
# Detect cluster members#


def self_minmax_scaler(vector):
    scaled_vector = vector / np.max(vector)
    
    return scaled_vector
# Apply minmax normalization#

    
def contact_matrix_construction(occ, coord, radius):
    beads = len(coord)
    contact_m = np.zeros((beads, beads), dtype=int)
    distance_m = np.zeros((beads, beads), dtype=float)
    np.fill_diagonal(contact_m, len(occ))

    for i in occ:
        matrix = dist.pdist(coord[:, i, :])
        scaled_matrix = self_minmax_scaler(matrix)
        scaled_matrix = dist.squareform(scaled_matrix)
        member_in = np.where(matrix <= 3 * radius)
        member_out = np.where(matrix > 3 * radius)
        matrix[member_in] = 1
        matrix[member_out] = 0
        matrix = dist.squareform(matrix.astype(int))
        
        distance_m += scaled_matrix
        contact_m += matrix
        
    return contact_m, distance_m / len(occ)
# Construct contact frequency matrix and average distance matrix#


def radial_profile(coord, occ, beads, n_radius):
    rp = []

    for index in occ:
        subrp = []
        for j in range(beads):
            ratio = coord[j, index, :] / n_radius
            subrp.append(np.linalg.norm(ratio))
        rp.append(subrp)
    rp = np.array(rp)

    ave_rp = np.mean(rp, axis=0)
    std_rp = np.std(rp, axis=0)

    return ave_rp, np.log2(std_rp / np.mean(std_rp)), rp
# Compute radial position for GM or H1#


def ellipse_radial_profile(coord, occ, beads, x_radius, y_radius, z_radius):
    rp = []

    for index in occ:
        subrp = []
        for j in range(beads):
            ratio = (coord[j, index, :][0] / x_radius)**2 + (coord[j, index, :][1] / y_radius)**2 + (coord[j, index, :][2] / z_radius)**2
            subrp.append(np.sqrt(ratio))
        rp.append(subrp)
    rp = np.array(rp)

    ave_rp = np.mean(rp, axis=0)
    std_rp = np.std(rp, axis=0)

    return ave_rp, np.log2(std_rp / np.mean(std_rp)), rp
# Compute radial position for HFF#


def radius_gyration(coord, radius, occ, beads):
    all_gyr = []
    radii = np.full(5, radius)
    for index in occ:
        gyr = []
        for i in range(beads - 4):
            gyr.append(geo.RadiusOfGyration(coord[i:i + 5, index, :], radii))
        all_gyr.append([np.nan, np.nan] + gyr + [np.nan, np.nan])

    return np.mean(np.array(all_gyr), axis=0)
# Calculate radius of gyration#


def refinement(vector):
    dmatrix = dist.pdist(vector.T)
    dvector = np.sum(dist.squareform(dmatrix), axis=0)
    refined_vector = vector.T[abs(dvector - np.mean(dvector)) < 3 * np.std(dvector)].T

    return refined_vector
# Filter outliers#


def TSA_intensity(dist):

    return np.exp(-4 * dist / 1000)
# Convert TSA-seq signals#


def TSA_prediction(f, cell, allclusters, statesInfo, states, tag, coord, radii, occ):
    res = []
    finalminlist = []

    for j in occ:
        str_crd = coord[:, j, :]
        signal = np.zeros(len(statesInfo))
        distlist = []

        if j >= 10000:
            k = j - 10000
            new_str_crd = coord[:, k, :]
        else:
            k = j
            new_str_crd = str_crd

        for cluster in allclusters[tag][k]:
            if len(cluster) <= 3:
                continue
                    
            xyz = new_str_crd[np.array(cluster)]
            r = radii[np.array(cluster)]
            com = geo.CenterOfMass(xyz, r**3)   
            dists = np.linalg.norm(str_crd - com, axis=1)         
            signal += TSA_intensity(dists)
            distlist.append(dists)
        
        res.append(signal)
        finalminlist.append(np.min(distlist, axis=0))    

    ares = np.array(res).mean(axis=0)
    finalTSA = ares
    if tag == "NOR":
        nucleoli = np.genfromtxt("./Models/NOR_200kb.bed", dtype=str).transpose()[3]
        not_nor = np.where(nucleoli != "NOR")[0]
        finalTSA = np.log2(finalTSA[:len(states)] / np.mean(finalTSA[not_nor]))
    else:
        finalTSA = np.log2(finalTSA[:len(states)] / finalTSA.mean())
    
    return finalTSA, finalminlist
# Calculate TSA-seq#


def speckle_association(mat, d, radius):
    saf = []
    mat = np.transpose(mat)
    
    for m in mat:
        saf.append(len(m[m < d + radius]) / len(m))
    
    return np.array(saf)
# Calculate speckle association frequency#


def main():
    cell = sys.argv[1]
    index = int(sys.argv[2]) - 1
    start = int(sys.argv[3])
    end = int(sys.argv[4])

    if cell == "GM":
        f = analysis.HssFile("./Models/GM_igm-model.hss", "r")
    elif cell == "H1":
        f = analysis.HssFile("./Models/Control_H1_igm-model.hss", "r")
    elif cell == "HFF":
        f = analysis.HssFile("./Models/Control_HFF_igm-model.hss", "r")
    else:
        print("Unknown Cell Type.")
    vector = np.load("Encoded_" + cell + "_chr" + str(index + 1) + "_" + str(start) + "_" + str(end) + ".npy")
    org_coordinates = f.get_coordinates()
    new_coordinates = np.concatenate((org_coordinates[len(org_coordinates) // 2:, :, :], org_coordinates[:len(org_coordinates) // 2, :, :]), axis=0)
    coordinates = np.concatenate((org_coordinates, new_coordinates), axis=1)
    radii = f.get_radii()
    radius = f.get_radii()[0]
    e = 0.0

    length = f.index.chrom_sizes
    starting = [0]
    acc = 0
    for sublength in length:
        acc += sublength
        starting.append(acc)
    starting = np.array(starting)
    starting1 = np.sum(length[0:index])
    ending1 = np.sum(length[0:index + 1])
    if cell == "GM":
        starting2 = np.sum(length[0:index + 23])
        ending2 = np.sum(length[0:index + 24])
    else:
        starting2 = np.sum(length[0:index + 24])
        ending2 = np.sum(length[0:index + 25])
    coord = np.concatenate((org_coordinates[starting1:ending1, :, :], org_coordinates[starting2:ending2, :, :]), axis=1)
    beads = ending1 - starting1

    print("Fitting kernel density and detecting peaks...")
    kernel, kernel_pdf = gaussian_kde(vector)
    refined_vector = refinement(vector)
    x, y, z = grid_construction(refined_vector, kernel)
    maxima, maximad = detecting_maxima(x, y, z, e)
    pa, maxima, maximad = detecting_boundary(refined_vector, maxima, maximad)
    peak, n = detecting_peaks(maxima, x, y, vector, kernel_pdf)

    print("Constructing contact matrices and radial profiles...")
    cmap = LinearSegmentedColormap.from_list("rg", ["w", "r"], N=256)
    gs = GridSpec(11, 1, height_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5])
    gsn = GridSpec(6, 1, height_ratios=[1, 1, 1, 1, 1, 5])

    cs = open("./Models/" + cell + '_speckle_clusters.dat', 'rb')
    allclusters_speckle = pickle.load(cs)
    cs.close()
    cs = open("./Models/" + cell + '_nucleoli_clusters.dat', 'rb')
    allclusters_nucleoli = pickle.load(cs)
    cs.close()

    if cell == "GM":
        states = np.genfromtxt("./Models/" + cell + '_subcompartments.bed', dtype=None, encoding=None)
        statesInfo = np.array(states['f3'].tolist() + states['f3'][:f.index.copy.sum()].tolist())
        tag1 = "A1"
        tag2 = "NOR"
    elif cell == "H1" or cell == "HFF":
        states = np.genfromtxt("./Models/" + cell + '_SPINstates.bed', dtype=None, encoding=None)
        statesInfo = np.array(states['f3'].tolist() + states['f3'][:f.index.copy.sum()].tolist())
        tag1 = "Speckle"
        tag2 = "NOR"

    dom_data = np.genfromtxt("./Models/" + cell + '_domains_200kb.bed', dtype=str).transpose()[3]
    dom_data = dom_data[starting[index]:starting[index + 1]]
    domains = np.where(dom_data == 'domain')[0]
    dd = domains

    full_occ = np.array([])
    for i in range(n):
        _, occ = detecting_occupancy(pa[i], peak[i], vector)
        m = len(occ)
        if m < 100:
            continue
        full_occ = np.append(full_occ, occ)
    full_occ = full_occ.astype(int)

    if cell == "HFF":
        rp, rp_var, rp_profile = ellipse_radial_profile(coord, full_occ, beads, 7840.0, 6470.0, 2450.0)
    else:
        rp, rp_var, rp_profile = radial_profile(coord, full_occ, beads, 5000.0)
    gyration = radius_gyration(coord, radius, full_occ, beads)
    tsa_seq_speckle, finalminlist = TSA_prediction(f, cell, allclusters_speckle, statesInfo, states, tag1, coordinates, radii, full_occ)
    tsa_seq_speckle = tsa_seq_speckle[starting[index]:starting[index + 1]]
    tsa_seq_surface, finallist = TSA_surface_prediction(f, cell, states, coordinates, full_occ)
    tsa_seq_surface = tsa_seq_surface[starting[index]:starting[index + 1]]
    saf_seq = speckle_association(finalminlist, 1000, radius)
    saf_seq = saf_seq[starting[index]:starting[index + 1]]
    
    matrix, dmatrix = contact_matrix_construction(full_occ, coord, radius)
    
    features = []
    fig = plt.figure(figsize=(10, 30))
    mat = fig.add_subplot(gsn[5])
    im = mat.imshow(np.log2(matrix + 1), cmap=cmap, interpolation="nearest", aspect=1, vmin=0.0, vmax=np.log2(matrix + 1).max())
    plt.xlim(start, end - 1)
    mat.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: ('%g') % (x * 0.2)))
    mat.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: ('%g') % (x * 0.2)))
    plt.ylim(end - 1, start)
    
    tsa_speckle = fig.add_subplot(gsn[0])
    neg_p = np.where(tsa_seq_speckle < 0)
    pos_p = np.where(tsa_seq_speckle >= 0)
    tsa_speckle.bar(np.arange(len(tsa_seq_speckle))[neg_p], tsa_seq_speckle[neg_p], color='navy')
    tsa_speckle.bar(np.arange(len(tsa_seq_speckle))[pos_p], tsa_seq_speckle[pos_p], color='red')
    tsa_speckle.xaxis.set_tick_params(labelbottom=False)
    plt.xlim(start, end - 1)
    features.append(tsa_seq_speckle)
    
    saf = fig.add_subplot(gsn[1])
    saf.plot(np.arange(len(saf_seq)), saf_seq, color='black')
    saf.xaxis.set_tick_params(labelbottom=False)
    plt.xlim(start, end - 1)
    features.append(saf_seq)
    
    gyr = fig.add_subplot(gsn[2])
    gyr.plot(np.arange(len(gyration)), gyration, color='black')
    gyr.xaxis.set_tick_params(labelbottom=False)
    plt.xlim(start, end - 1)
    features.append(gyration)
    
    rp_x = fig.add_subplot(gsn[3])
    rp_x.plot(np.arange(len(rp)), rp, color='black')
    rp_x.xaxis.set_tick_params(labelbottom=False)
    plt.xlim(start, end - 1)
    features.append(rp)
    
    rp_var_x = fig.add_subplot(gsn[4])
    neg_p = np.where(rp_var < 0)
    pos_p = np.where(rp_var >= 0)
    rp_var_x.bar(np.arange(len(rp_var))[neg_p], rp_var[neg_p], color='steelblue')
    rp_var_x.bar(np.arange(len(rp_var))[pos_p], rp_var[pos_p], color='palevioletred')
    rp_var_x.xaxis.set_tick_params(labelbottom=False)
    plt.xlim(start, end - 1)
    features.append(rp_var)
    
    plt.savefig("./" + cell + "_Structural_Features/Chr" + str(index + 1) + "_Occ_Contact_Matrix_Ensemble.pdf", dpi=600)
    plt.close()
    
    np.save("./" + cell + "_Structural_Features/Chr" + str(index + 1) + "_Features.npy", np.array(features))
    
    occupancy = []
    s = []
    j = 1
    occ_matrix = []
    occ_dmatrix = []
    occ_neighbors = []
    occ_features = []
    for i in range(n):
        _, occ = detecting_occupancy(pa[i], peak[i], vector)
        m = len(occ)
        if m < 100:
            continue
            
        s.append(m)
        occupancy.append(occ)
        matrix, dmatrix = contact_matrix_construction(occ, coord, radius)
        if cell == "HFF":
            sub_rp, sub_rp_var, sub_rp_profile = ellipse_radial_profile(coord, occ, beads, 7840.0, 6470.0, 2450.0)
        else:
            sub_rp, sub_rp_var, sub_rp_profile = radial_profile(coord, occ, beads, 5000.0)
        rp_ratio = np.log2(sub_rp / rp)
        rp_var_ratio = np.log2(sub_rp_var / rp_var)
        sub_gyration = radius_gyration(coord, radius, occ, beads)
        gyration_ratio = np.log2(sub_gyration / gyration)
        sub_tsa_seq_speckle, finalminlist = TSA_prediction(f, cell, allclusters_speckle, statesInfo, states, tag1, coordinates, radii, occ)
        sub_tsa_seq_speckle = sub_tsa_seq_speckle[starting[index]:starting[index + 1]]
        tsa_seq_speckle_ratio = np.log2(sub_tsa_seq_speckle / tsa_seq_speckle)
        sub_saf_seq = speckle_association(finalminlist, 1000, radius)
        sub_saf_seq = sub_saf_seq[starting[index]:starting[index + 1]]
        saf_seq_ratio = np.log2(sub_saf_seq / saf_seq)

        occ_matrix.append(matrix)
        occ_dmatrix.append(dmatrix)
        occ_neighbors.append(occ)
        sub_features = []

        fig = plt.figure(figsize=(10, 40))
        mat = fig.add_subplot(gs[10])
        im = mat.imshow(np.log2(matrix + 1), cmap=cmap, interpolation="nearest", aspect=1, vmin=0.0, vmax=np.log2(matrix + 1).max())
        plt.xlim(start, end - 1)
        mat.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: ('%g') % (x * 0.2)))
        mat.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: ('%g') % (x * 0.2)))
        plt.ylim(end - 1, start)

        sub_tsa_speckle = fig.add_subplot(gs[0])
        neg_p = np.where(sub_tsa_seq_speckle < 0)
        pos_p = np.where(sub_tsa_seq_speckle >= 0)
        sub_tsa_speckle.bar(np.arange(len(sub_tsa_seq_speckle))[neg_p], sub_tsa_seq_speckle[neg_p], color='navy')
        sub_tsa_speckle.bar(np.arange(len(sub_tsa_seq_speckle))[pos_p], sub_tsa_seq_speckle[pos_p], color='red')
        sub_tsa_speckle.xaxis.set_tick_params(labelbottom=False)
        plt.xlim(start, end - 1)
        sub_features.append(sub_tsa_seq_speckle)

        tsa_speckle = fig.add_subplot(gs[1])
        tsa_speckle.bar(np.arange(len(tsa_seq_speckle_ratio)), tsa_seq_speckle_ratio, color='gray')
        tsa_speckle.xaxis.set_tick_params(labelbottom=False)
        plt.xlim(start, end - 1) 
        sub_features.append(tsa_seq_speckle_ratio)     

        sub_saf = fig.add_subplot(gs[2])
        sub_saf.plot(np.arange(len(sub_saf_seq)), sub_saf_seq, color='black')
        sub_saf.xaxis.set_tick_params(labelbottom=False)
        plt.xlim(start, end - 1)
        sub_features.append(sub_saf_seq)

        saf = fig.add_subplot(gs[3])
        neg_p = np.where(saf_seq_ratio < 0)
        pos_p = np.where(saf_seq_ratio >= 0)
        saf.bar(np.arange(len(saf_seq_ratio))[neg_p], saf_seq_ratio[neg_p], color='darkgreen')
        saf.bar(np.arange(len(saf_seq_ratio))[pos_p], saf_seq_ratio[pos_p], color='darkgoldenrod')
        saf.xaxis.set_tick_params(labelbottom=False)
        plt.xlim(start, end - 1) 
        sub_features.append(saf_seq_ratio)

        sub_gyr = fig.add_subplot(gs[4])
        sub_gyr.plot(np.arange(len(sub_gyration)), sub_gyration, color='black')
        sub_gyr.xaxis.set_tick_params(labelbottom=False)
        plt.xlim(start, end - 1)
        sub_features.append(sub_gyration)

        gyr = fig.add_subplot(gs[5])
        neg_p = np.where(gyration_ratio < 0)
        pos_p = np.where(gyration_ratio >= 0)
        gyr.bar(np.arange(len(gyration_ratio))[neg_p], gyration_ratio[neg_p], color='royalblue')
        gyr.bar(np.arange(len(gyration_ratio))[pos_p], gyration_ratio[pos_p], color='darkseagreen')
        gyr.xaxis.set_tick_params(labelbottom=False)
        gyr.xaxis.set_tick_params(labelbottom=False)
        plt.xlim(start, end - 1)
        sub_features.append(gyration_ratio)
        
        sub_rp_x = fig.add_subplot(gs[6])
        sub_rp_x.plot(np.arange(len(sub_rp)), sub_rp, color='black')
        sub_rp_x.xaxis.set_tick_params(labelbottom=False)
        plt.xlim(start, end - 1)
        sub_features.append(sub_rp)

        rp_x = fig.add_subplot(gs[7])
        neg_p = np.where(rp_ratio < 0)
        pos_p = np.where(rp_ratio >= 0)
        rp_x.bar(np.arange(len(rp_ratio))[neg_p], rp_ratio[neg_p], color='darkcyan')
        rp_x.bar(np.arange(len(rp_ratio))[pos_p], rp_ratio[pos_p], color='lightsteelblue')
        rp_x.xaxis.set_tick_params(labelbottom=False)
        plt.xlim(start, end - 1)
        sub_features.append(rp_ratio)
        
        sub_rp_var_x = fig.add_subplot(gs[8])
        neg_p = np.where(sub_rp_var < 0)
        pos_p = np.where(sub_rp_var >= 0)
        sub_rp_var_x.bar(np.arange(len(sub_rp_var))[neg_p], sub_rp_var[neg_p], color='steelblue')
        sub_rp_var_x.bar(np.arange(len(sub_rp_var))[pos_p], sub_rp_var[pos_p], color='palevioletred')
        sub_rp_var_x.xaxis.set_tick_params(labelbottom=False)
        plt.xlim(start, end - 1)
        sub_features.append(sub_rp_var)

        rp_var_x = fig.add_subplot(gs[9])
        rp_var_x.bar(np.arange(len(rp_var_ratio)), rp_var_ratio, color='gray')
        rp_var_x.xaxis.set_tick_params(labelbottom=False)
        plt.xlim(start, end - 1)
        sub_features.append(rp_var_ratio)

        plt.savefig("./" + cell + "_Structural_Features/Chr" + str(index + 1) + "_Occ_Contact_Matrix_" + str(j) + ".pdf", dpi=600)
        plt.close()
        
        occ_features.append(sub_features)
        j += 1
        print(str(j - 1) + "/" + str(n) + " finished.")
        
    np.save("./" + cell + "_Structural_Features/Chr" + str(index + 1) + "_Occ_Matrix.npy", np.array(occ_matrix))
    np.save("./" + cell + "_Structural_Features/Chr" + str(index + 1) + "_Occ_DMatrix.npy", np.array(occ_dmatrix))
    np.save("./" + cell + "_Structural_Features/Chr" + str(index + 1) + "_Occ_Neighbors.npy", np.array(occ_neighbors))
    np.save("./" + cell + "_Structural_Features/Chr" + str(index + 1) + "_Peaks.npy", np.array(peak))
    np.save("./" + cell + "_Structural_Features/Chr" + str(index + 1) + "_Occ_Features.npy", np.array(occ_features))

    plt.figure(figsize=(8, 8))
    plt.scatter(refined_vector[0], refined_vector[1], c="navy", s=10.0, alpha=0.5)    
    i = 1
    for j in range(len(peak)):        
        occ_vector, occ = detecting_occupancy(pa[j], peak[j], vector)
        m = len(occ)
        if m < 100:
            continue
        plt.scatter(occ_vector[0], occ_vector[1], c="orange", s=20.0, alpha=0.5)    
        plt.plot(vector[0][peak[j]], vector[1][peak[j]], marker="o", c="r", markersize=10.0, alpha=0.5)
        plt.annotate(str(i), (vector[0][peak[j]], vector[1][peak[j]]))
        i += 1  
    plt.axis("tight")
    plt.savefig("./" + cell + "_Structural_Features/Chr" + str(index + 1) + "_peak_200kb.pdf", dpi=600)
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.imshow(z.T, interpolation='nearest', origin='low')
    plt.axis('tight')
    plt.savefig("./" + cell + "_Structural_Features/Chr" + str(index + 1) + "_hist_200kb.pdf", dpi=600)
    plt.close()

    sns.jointplot(x=refined_vector[0], y=refined_vector[1], kind="kde", n_levels=15, cbar=True)
    plt.axis('tight')
    plt.savefig("./" + cell + "_Structural_Features/Chr" + str(index + 1) + "_density_200kb.pdf", dpi=600)
    plt.close()

    plt.figure(figsize=(8, 8))
    barwidth = 0.25
    r = np.arange(len(s))
    plt.bar(r, s, color='red', width=barwidth, edgecolor='white', alpha=0.75)
    plt.xticks([r for r in range(len(s))], np.arange(1, len(s) + 1))
    plt.axis('tight')
    plt.savefig("./" + cell + "_Structural_Features/Chr" + str(index + 1) + "_occupancy_200kb.pdf", dpi=600)
# Main#


if __name__ == '__main__':
    main()