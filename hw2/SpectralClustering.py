from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn import datasets as ds
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from itertools import cycle, islice
import warnings

np.random.seed(0)
n_samples = 1500

# Compare the Spectral clustering with GMM

def test_2d():
    # generate dataset
    circles, _ = ds.make_circles(
        n_samples=n_samples, factor=0.5, noise=0.05)
    moons, _ = ds.make_moons(n_samples=n_samples, noise=0.05, random_state=1)
    blobs, _ = ds.make_blobs(n_samples=n_samples, random_state=1)
    s_curve, _ = ds.samples_generator.make_s_curve(n_samples=n_samples, random_state=1,noise=0.05)
    s_curve = np.hstack((s_curve[:,0].reshape(-1,1),s_curve[:,2].reshape(-1,1)))
    no_structure = np.random.rand(
        n_samples, 2)

    # Anisotropicly distributed data
    X, _ = ds.make_blobs(n_samples=n_samples, random_state=1)
    transformation = [[1, -0.2], [-0.8, 0.3]]
    aniso = np.dot(X, transformation)

    # blobs with varied variances
    varied, _ = ds.make_blobs(n_samples=n_samples,
                              cluster_std=[0.2, 1, 3],
                              random_state=1)

    plt.figure(figsize=(18, 12.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                        hspace=.01)

    plot_num = 1

    default_base = {'quantile': .3,
                    'eps': .3,
                    'damping': .9,
                    'preference': -200,
                    'n_neighbors': 10,
                    'n_clusters': 3}

    datasets = [
        (circles, {'damping': .77, 'preference': -240,
                   'quantile': .2, 'n_clusters': 2}),
        (moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
        (s_curve, {'n_clusters': 2}),
        (varied, {'eps': .18, 'n_neighbors': 2}),
        (aniso, {'eps': .15, 'n_neighbors': 2}),
        (blobs, {}),
        (no_structure, {})]

    # iterate the datasets to verify spectral clustering
    for i_dataset, (dataset, algo_params) in enumerate(datasets):
        # update parameters with dataset-specific values
        params = default_base.copy()
        params.update(algo_params)
        X = dataset

        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)

        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(
            X, n_neighbors=params['n_neighbors'], include_self=False)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)

        # spectral
        # create cluster objects
        spectral = SpectralClustering(
            n_clusters=params['n_clusters'], eigen_solver='arpack',
            affinity="nearest_neighbors")

        # filter the warning
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding may not work as expected.",
                category=UserWarning
            )
            spectral.fit(X)

        y_pred = spectral.labels_.astype(np.int)

        # plot
        plt.subplot(len(datasets), 2, plot_num)
        if i_dataset == 0:
            plt.title("Spectral Clustering", size=18)

        colors = np.array(list(islice(cycle(['#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        # GMM
        gmm = GaussianMixture(
            n_components=params['n_clusters'], covariance_type='full')

        gmm.fit(X)
        y_pred = gmm.predict(X)

        # plot
        plt.subplot(len(datasets), 2, plot_num+1)
        if i_dataset == 0:
            plt.title("GMM", size=18)

        colors = np.array(list(islice(cycle(['#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        plot_num += 2
    plt.show()


def test_3d():
    swiss, _ = ds.make_swiss_roll(n_samples=n_samples, noise=0.05)
    blobs, _ = ds.make_blobs(n_samples=n_samples, n_features=3, random_state=8)

    # Anisotropicly distributed data
    random_state = 170
    X, _ = ds.make_blobs(n_samples=n_samples, n_features=3,
                         random_state=random_state)
    transformation = [[0.6, -0.6, -0.2], [-0.4, 0.8, -0.5], [-0.3, 0.5, -0.7]]
    aniso = np.dot(X, transformation)

    # blobs with varied variances
    varied, _ = ds.make_blobs(n_samples=n_samples,
                              n_features=3,
                              cluster_std=[0.2, 1, 3],
                              random_state=random_state)

    no_structure = np.random.rand(n_samples, 3)

    default_base = {'quantile': .3,
                    'eps': .3,
                    'damping': .9,
                    'preference': -200,
                    'n_neighbors': 10,
                    'n_clusters': 3}

    datasets = [
        (swiss, {'damping': .77, 'preference': -240,
                 'quantile': .2}),
        (varied, {'eps': .18, 'n_neighbors': 2}),
        (aniso, {'eps': .15, 'n_neighbors': 2}),
        (blobs, {}),
        (no_structure, {})]

    fig = plt.figure(figsize=(18, 12.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                        hspace=.01)
    plot_num = 1

    for i_dataset, (dataset, algo_params) in enumerate(datasets):
        # update parameters with dataset-specific values
        params = default_base.copy()
        params.update(algo_params)
        X = dataset
        X[:, 1] *= 0.5
        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)

        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(
            X, n_neighbors=params['n_neighbors'], include_self=False)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)

        # spectral
        # create cluster objects
        spectral = SpectralClustering(
            n_clusters=params['n_clusters'], eigen_solver='arpack',
            affinity="nearest_neighbors")

        # filter the warning
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding may not work as expected.",
                category=UserWarning
            )
            spectral.fit(X)

        y_pred = spectral.labels_.astype(np.int)

        # plot
        ax = fig.add_subplot(len(datasets), 2, plot_num, projection='3d')
        if i_dataset == 0:
            plt.title("Spectral Clustering", size=18)

        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=10, color=colors[y_pred])

        # GMM
        gmm = GaussianMixture(
            n_components=params['n_clusters'], covariance_type='full')
        gmm.fit(X)
        y_pred = gmm.predict(X)

        # plot
        ax = fig.add_subplot(len(datasets), 2, plot_num+1, projection='3d')
        if i_dataset == 0:
            plt.title("GMM", size=18)

        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=10, color=colors[y_pred])

        plot_num += 2
    plt.show()


if __name__ == "__main__":
    test_2d()
    # test_3d()
