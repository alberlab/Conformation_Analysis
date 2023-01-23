import sys
from alabtools import analysis
import numpy as np
import scipy.spatial.distance as dist
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D 
from keras.models import Model
from tensorflow import image
from sklearn import manifold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_data(X):
    X = X.astype("float32")
    N = len(X)
    X = np.reshape(X, (N, N, 1))
    N = N // 50 * 50
    X = image.resize(X, [N, N])

    return X, N
# Transform matrices#


def Autoencoder(X_train, input_x, latent_x, output_x):
    autoencoder = Model(input_x, output_x)
    autoencoder.summary()

    encoder = Model(input_x, latent_x)
    encoder.summary()

    autoencoder.compile(optimizer="adadelta", loss="mean_squared_error")
    autoencoder.fit(X_train, X_train, epochs=15, batch_size=200, shuffle=True)

    return encoder, autoencoder
# Fit dataset#


def build_model(N):
    input_x = Input(shape=(N, N, 1))
    x = Conv2D(16, (10, 10), activation='relu', strides=(1, 1), padding='same')(input_x)
    x = MaxPooling2D((5, 5), strides=None, padding='same')(x)
    x = Conv2D(8, (10, 10), activation='relu', strides=(1, 1), padding='same')(x)
    x = MaxPooling2D((5, 5), strides=None, padding='same')(x)
    x = Conv2D(4, (10, 10), activation='relu', strides=(1, 1), padding='same')(x)
    latent_x = MaxPooling2D((2, 2), strides=None, padding='same')(x)

    x = Conv2D(4, (10, 10), activation='relu', strides=(1, 1), padding='same')(latent_x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (10, 10), activation='relu', strides=(1, 1), padding='same')(x)
    x = UpSampling2D((5, 5))(x)
    x = Conv2D(16, (10, 10), activation='relu', strides=(1, 1), padding='same')(x)
    x = UpSampling2D((5, 5))(x)
    output_x = Conv2D(1, (10, 10), activation='sigmoid', padding='same')(x)

    return input_x, latent_x, output_x
# Build autoencoder#


def plot_example(X_train, output_X, n, cell, index):
    plt.figure(figsize=(20, 4))

    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X_train[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(output_X[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig("result_autoencoder_" + cell + "_chr" + str(index) + ".pdf", dpi=600)
# Plot examples#


def self_minmax_scaler(vector):
    vector = vector / np.max(vector)
    
    return vector
# Apply minmax normalization#


def distance_matrix(coord, radius):
    distance_m = []

    for i in range(len(coord[0, :, 0])):
        matrix = dist.pdist(coord[:, i, :])
        matrix = self_minmax_scaler(matrix - 2 * radius)
        matrix = dist.squareform(matrix)
        matrix, N = load_data(matrix)
        distance_m.append(matrix)

        if i % 1000 == 0:
            print("=", end="")
    print("")

    return np.array(distance_m), N
# Construct distance matrices from coordinates#


def tSNE_embedding(vector, perplexity):
    tsne = manifold.TSNE(n_components=2, init='random', perplexity=perplexity, learning_rate=1000.0, random_state=0,
                         n_iter=10000)
    tsnevector = tsne.fit_transform(vector).T

    return tsnevector
# Perform t-SNE embedding#


def standard_scaler(vector):
    scaler = StandardScaler()
    parameter = scaler.fit_transform(vector)

    return parameter
# Apply standard normalization#


def main():
    cell = sys.argv[1]
    index = int(sys.argv[2])
    tag = int(sys.argv[3])
    start = int(sys.argv[4])
    end = int(sys.argv[5])
 
    if tag == 0:
        if cell == "GM":
            f = analysis.HssFile("./Models/GM_igm-model.hss", "r")
        elif cell == "H1":
            f = analysis.HssFile("./Models/H1_igm-model.hss", "r")
        elif cell == "HFF":
            f = analysis.HssFile("./Models/HFF_igm-model.hss", "r")
        else:
            print("Unknown cell type.")
            sys.exit()
        
        coord = f.get_coordinates()
        radius = f.get_radii()[0]
        dom_data = np.genfromtxt("./Models/{}_domains_200kb.bed".format(cell), dtype=str).transpose()[3]

        length = f.index.chrom_sizes
        starting1 = np.sum(length[0:index - 1])
        ending1 = np.sum(length[0:index])
        if cell == "GM":
            starting2 = np.sum(length[0:index + 22])
            ending2 = np.sum(length[0:index + 23])
        else:
            starting2 = np.sum(length[0:index + 23])
            ending2 = np.sum(length[0:index + 24])
        dom_data = dom_data[starting1:ending1]
        dom_data = dom_data[start:end]
        dd = np.where(dom_data == 'domain')[0]
        coord = np.concatenate((coord[starting1:ending1, :, :], coord[starting2:ending2, :, :]), axis=1)
        coord = coord[np.arange(start, end)[dd], :, :]

        print("Calculating distance matrices...")
        X_train, N = distance_matrix(coord, radius)
        
        print("Feature vector size:")
        print(X_train.shape)

        print("Building model...")
        input_x, latent_x, output_x = build_model(N)

        print("Fitting model...")
        encoder, autoencoder = Autoencoder(X_train, input_x, latent_x, output_x)
        latent_X = encoder.predict(X_train)
        output_X = autoencoder.predict(X_train)
        vector = []
        for x in latent_X:
            vector.append(x.flatten())
        vector = np.array(vector)
        np.save("encoder_predicted_" + cell + "_chr" + str(index) + "_" + str(start) + "_" + str(end) + ".npy", vector)
        X_train = np.reshape(X_train, (len(X_train), N, N))
        output_X = np.reshape(output_X, (len(output_X), N, N))
        plot_example(X_train, output_X, 10, cell, index)
    
    vector = np.load("encoder_predicted_" + cell + "_chr" + str(index) + "_" + str(start) + "_" + str(end) + ".npy")
    vector = standard_scaler(vector)

    print("Feature vector size:")
    print(vector.shape)
    
    print("Performing tSNE...")
    tsnevector = tSNE_embedding(vector, 200)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    plt.scatter(tsnevector[0], tsnevector[1], c="b", s=1.0, alpha=0.5)
    plt.axis('tight')    
        
    print("Saving data and figure...")
    plt.savefig("Encoded_" + cell + "_chr" + str(index) + "_" + str(start) + "_" + str(end) + ".pdf")
    np.save("Encoded_" + cell + "_chr" + str(index) + "_" + str(start) + "_" + str(end) + ".npy", tsnevector)
# Main#


if __name__ == '__main__':
    main()
