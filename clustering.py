from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

def flattent_waveforms(waveforms):
    return waveforms.reshape(waveforms.shape[0], -1)

def reconstruct_waveforms(X, n_samples, n_channels):
    return X.reshape(-1, n_samples, n_channels)

def waveforms_pca(waveforms, var):
    samples, n_samples, n_channels = waveforms.shape
    X = flattent_waveforms(waveforms)

    initial_number_of_features = X.shape[1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_.cumsum()
    n_components = np.argmax(explained_variance >= var) + 1
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X_scaled)

    reduced_ratio = X_reduced.shape[1] / initial_number_of_features
    principal_components = pca.components_

    principal_components = reconstruct_waveforms(principal_components, n_samples, n_channels)

    return X_reduced, reduced_ratio, principal_components


def cluster(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels, kmeans.inertia_