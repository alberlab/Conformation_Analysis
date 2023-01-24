from alabtools import analysis
from alabtools import geo
import sys
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
from sklearn.ensemble import IsolationForest


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
# Find local maximas among the constructed grids#


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
    
    
def contact_matrix_construction(occ, coord, radius, j, index):
    beads = len(coord)
    contact_m = np.zeros((beads, beads), dtype=int)
    distance_m = np.zeros((beads, beads), dtype=float)
    np.fill_diagonal(contact_m, len(occ))
    cmap = LinearSegmentedColormap.from_list("rg", ["w", "r"], N=256)
    dcmap = LinearSegmentedColormap.from_list("rg", ["r", "w", "b"], N=256)

    for i in range(len(occ)):
        matrix = dist.pdist(coord[:, occ[i], :])
        scaled_matrix = self_minmax_scaler(matrix)
        distance_m += dist.squareform(scaled_matrix)
        member_in = np.where(matrix <= 3 * radius)
        member_out = np.where(matrix > 3 * radius)
        matrix[member_in] = 1
        matrix[member_out] = 0
        matrix = dist.squareform(matrix.astype(int))
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

    return np.mean(rp, axis=0)
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

    return np.mean(rp, axis=0)
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
    refined_index = np.where(abs(dvector - np.mean(dvector)) < 3 * np.std(dvector))[0]
    outlier_index = np.where(abs(dvector - np.mean(dvector)) >= 3 * np.std(dvector))[0]
    refined_vector = vector.T[refined_index].T

    return refined_vector, refined_index, outlier_index
# Filter outliers#


def sliding_window(matrix, pos, step):
    matrix_1 = matrix[pos:pos + step, pos:pos + step]
    np.fill_diagonal(matrix_1, 0)
    matrix_1 = dist.squareform(matrix_1)
    matrix_2 = matrix[pos + step:pos + 2*step, pos + step:pos + 2*step]
    np.fill_diagonal(matrix_2, 0)
    matrix_2 = dist.squareform(matrix_2)
    matrix_3 = matrix[pos:pos + step, pos + step:pos + 2*step]

    return np.count_nonzero(matrix_1) + np.count_nonzero(matrix_2) - np.count_nonzero(matrix_3)
# Sliding window#


def domain_boundary(matrix, size):
    score = []
    s0 = []
    for i in range(40):
        s0.append(0.0)

    for i in range(len(matrix) - 2 * size):
        border = sliding_window(matrix, i, size)
        score.append(border)

    return np.array(s0 + score + s0)
# Calculate domain boundary profile#


def main():
    cell = sys.argv[1]
    index = int(sys.argv[2]) - 1
    start = int(sys.argv[3])
    end = int(sys.argv[4])

    if cell == "GM":
        f = analysis.HssFile("../../GM_igm-model.hss", "r")
    elif cell == "H1":
        f = analysis.HssFile("../../Control_H1_igm-model.hss", "r")
    elif cell == "HFF":
        f = analysis.HssFile("../../Control_HFF_igm-model.hss", "r")
    else:
        print("Unknown Cell Type.")
    vector = np.load("Encoded_" + cell + "_chr" + str(index + 1) + "_" + str(start) + "_" + str(end) + ".npy")
    coord = f.get_coordinates()
    radius = f.get_radii()[0]
    e = 0.0

    length = f.index.chrom_sizes
    starting1 = np.sum(length[0:index])
    ending1 = np.sum(length[0:index + 1])
    if cell == "GM":
        starting2 = np.sum(length[0:index + 23])
        ending2 = np.sum(length[0:index + 24])
    else:
        starting2 = np.sum(length[0:index + 24])
        ending2 = np.sum(length[0:index + 25])
    coord = np.concatenate((coord[starting1:ending1, :, :], coord[starting2:ending2, :, :]), axis=1)
    beads = ending1 - starting1

    print("Fitting kernel density and detecting peaks...")
    kernel, kernel_pdf = gaussian_kde(vector)
    refined_vector, _, _ = refinement(vector)
    x, y, z = grid_construction(refined_vector, kernel)
    maxima, maximad = detecting_maxima(x, y, z, e)
    pa, maxima, maximad = detecting_boundary(refined_vector, maxima, maximad)
    peak, n = detecting_peaks(maxima, x, y, vector, kernel_pdf)

    print("Constructing contact matrices and radial profiles...")
    cmap = LinearSegmentedColormap.from_list("rg", ["w", "r"], N=256)
    gs = GridSpec(6, 2, height_ratios=[1, 1, 1, 1, 1, 5], width_ratios=[20, 1], wspace=0.1)
    
    full_occ = np.array([])
    for i in range(n):
        _, occ = detecting_occupancy(pa[i], peak[i], vector)
        m = len(occ)
        if m < 100:
            continue
        full_occ = np.append(full_occ, occ)
    full_occ = full_occ.astype(int)

    if cell == "HFF":
        all_rp = ellipse_radial_profile(coord, full_occ , beads, 7840.0, 6470.0, 2450.0)
    else:
        all_rp = radial_profile(coord, full_occ , beads, 5000.0)
    if cell == "GM":
        a1, a2, b1, b2, b3, b4, null = subcompartment_state(cell, index)
    else:
        sp, ia1, ia2, ia3, ir1, ir2, nlm1, nlm2, lm, null = spin_state(cell, index)
    gyration = radius_gyration(coord, radius, full_occ, beads)

    occupancy = []
    s = []
    j = 1
    occ_matrix = []
    occ_dmatrix = []
    occ_neighbors = []
    for i in range(n):
        _, occ = detecting_occupancy(pa[i], peak[i], vector)
        m = len(occ)
        if m < 100:
            continue

        s.append(m)
        occupancy.append(occ)
        matrix, dmatrix = contact_matrix_construction(occ, coord, radius, j, index + 1)
        new_matrix = np.copy(matrix)
        sub_boundary = domain_boundary(new_matrix, 40)
        if cell == "HFF":
            sub_rp = ellipse_radial_profile(coord, occ, beads, 7840.0, 6470.0, 2450.0)
        else:
            sub_rp = radial_profile(coord, occ, beads, 5000.0)
        rp_ratio = np.log2(sub_rp / all_rp)
        sub_gyration = radius_gyration(coord, radius, occ, beads)
        gyration_ratio = np.log2(sub_gyration / gyration)

        occ_matrix.append(matrix)
        occ_dmatrix.append(dmatrix)
        occ_neighbors.append(occ)

        fig = plt.figure(figsize=(10, 25))
        mat = fig.add_subplot(gs[10])
        im = mat.imshow(np.log2(matrix + 1), cmap=cmap, interpolation="nearest", aspect=1, vmin=0.0, vmax=np.log2(matrix + 1).max())
        plt.xlim(start, end - 1)
        mat.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: ('%g') % (x * 0.2)))
        plt.ylim(end - 1, start)
        mat.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: ('%g') % (y * 0.2)))

        db = fig.add_subplot(gs[0])
        db.plot(np.arange(len(sub_boundary)), sub_boundary, color='black')
        db.xaxis.set_tick_params(labelbottom=False)
        plt.ylim(-50, 500)
        plt.xlim(start, end - 1)

        sub_gyr = fig.add_subplot(gs[2])
        sub_gyr.plot(np.arange(len(sub_gyration)), sub_gyration, color='black')
        sub_gyr.xaxis.set_tick_params(labelbottom=False)
        plt.ylim(200.0, 400.0)
        plt.xlim(start, end - 1)

        gyr = fig.add_subplot(gs[4])
        neg_p = np.where(gyration_ratio < 0)
        pos_p = np.where(gyration_ratio >= 0)
        gyr.bar(np.arange(len(gyration_ratio))[neg_p], gyration_ratio[neg_p], color='royalblue')
        gyr.bar(np.arange(len(gyration_ratio))[pos_p], gyration_ratio[pos_p], color='darkseagreen')
        gyr.xaxis.set_tick_params(labelbottom=False)
        plt.ylim(-0.1, 0.15)
        plt.xlim(start, end - 1)
        
        sub_rp_x = fig.add_subplot(gs[6])
        sub_rp_x.plot(np.arange(len(sub_rp)), sub_rp, color='black')
        sub_rp_x.xaxis.set_tick_params(labelbottom=False)
        plt.ylim(0.4, 1.0)
        plt.xlim(start, end - 1)

        rp_x = fig.add_subplot(gs[8])
        neg_p = np.where(rp_ratio < 0)
        pos_p = np.where(rp_ratio >= 0)
        rp_x.bar(np.arange(len(rp_ratio))[neg_p], rp_ratio[neg_p], color='darkcyan')
        rp_x.bar(np.arange(len(rp_ratio))[pos_p], rp_ratio[pos_p], color='lightsteelblue')
        rp_x.xaxis.set_tick_params(labelbottom=False)
        plt.ylim(-0.25, 0.2)
        plt.xlim(start, end - 1)

        cbar = fig.add_subplot(gs[11])
        fig.colorbar(ax=mat, mappable=im, cax=cbar)
        plt.axis("tight")
        plt.savefig("./" + cell + "/Chr" + str(index + 1) + "_Occ_Contact_Matrix_" + str(j) + ".pdf", dpi=600)
        plt.close()
        
        j += 1
        print(str(j - 1) + "/" + str(n) + " finished.")
        
    np.save("./" + cell + "/Chr" + str(index + 1) + "_Occ_Matrix.npy", np.array(occ_matrix))
    np.save("./" + cell + "/Chr" + str(index + 1) + "_Occ_DMatrix.npy", np.array(occ_dmatrix))
    np.save("./" + cell + "/Chr" + str(index + 1) + "_Occ_Neighbors.npy", np.array(occ_neighbors))
    np.save("./" + cell + "/Chr" + str(index + 1) + "_Peaks.npy", np.array(peak))

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
    plt.savefig("./" + cell + "/Chr" + str(index + 1) + "_peak_200kb.pdf", dpi=600)
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.imshow(z.T, interpolation='nearest', origin='low')
    plt.axis('tight')
    plt.savefig("./" + cell + "/Chr" + str(index + 1) + "_hist_200kb.pdf", dpi=600)
    plt.close()

    sns.jointplot(x=refined_vector[0], y=refined_vector[1], kind="kde", n_levels=15, cbar=True)
    plt.axis('tight')
    plt.savefig("./" + cell + "/Chr" + str(index + 1) + "_density_200kb.pdf", dpi=600)
    plt.close()

    plt.figure(figsize=(8, 8))
    barwidth = 0.25
    r = np.arange(len(s))
    plt.bar(r, s, color='red', width=barwidth, edgecolor='white', alpha=0.75)
    plt.xticks([r for r in range(len(s))], np.arange(1, len(s) + 1))
    plt.axis('tight')
    plt.savefig("./" + cell + "/Chr" + str(index + 1) + "_occupancy_200kb.pdf", dpi=600)
# Main#


if __name__ == '__main__':
    main()