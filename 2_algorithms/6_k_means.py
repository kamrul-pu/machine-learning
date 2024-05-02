import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        n_samples, n_features = X.shape

        # Initialize centroids randomly
        centroids = X[np.random.choice(n_samples, self.n_clusters, replace=False)]

        # Main loop
        for _ in range(self.max_iter):
            # Assign samples to nearest centroid
            distances = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=0)

            # Update centroids based on cluster means
            for i in range(self.n_clusters):
                centroids[i] = X[labels == i].mean(axis=0)

        self.labels_ = labels
        self.cluster_centers_ = centroids

    def predict(self, X):
        distances = np.sqrt(
            ((X - self.cluster_centers_[:, np.newaxis]) ** 2).sum(axis=2)
        )
        return np.argmin(distances, axis=0)


# Example usage:
np.random.seed(0)
# Create synthetic data with 3 clusters
X1 = np.random.normal(loc=[0, 0], scale=1, size=(100, 2))
X2 = np.random.normal(loc=[5, 5], scale=1, size=(100, 2))
X3 = np.random.normal(loc=[-5, 5], scale=1, size=(100, 2))
X = np.vstack([X1, X2, X3])

# Initialize and fit KMeans model
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Plot clusters and centroids
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap="viridis", alpha=0.5)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    c="red",
    marker="x",
    s=200,
)
plt.title("K-means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label="Cluster")
plt.show()
