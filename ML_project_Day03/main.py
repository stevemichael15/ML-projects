import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.datasets import make_moons
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.metrics import silhouette_score
warnings.filterwarnings("ignore")

# Dataset
x, y = make_moons(2000, noise=0.1, random_state=1)

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Actual classes
axes[0, 0].scatter(x[:, 0], x[:, 1], c=y, cmap='viridis')
axes[0, 0].set_title("Actual")

# Plot 2: Agglomerative Clustering
model_algo = AgglomerativeClustering(n_clusters=2, metric="euclidean", linkage="ward")
algo_pred = model_algo.fit_predict(x)
axes[0, 1].scatter(x[:, 0], x[:, 1], c=algo_pred, cmap='viridis')
axes[0, 1].set_title("Agglomerative Clustering")

# Plot 3: KMeans Clustering
model_kmeans = KMeans(n_clusters=2, random_state=1)
kmeans_pred = model_kmeans.fit_predict(x)
axes[1, 0].scatter(x[:, 0], x[:, 1], c=kmeans_pred, cmap='viridis')
axes[1, 0].set_title("KMeans Clustering")

# Plot 4: DBSCAN Clustering
model_dbscan = DBSCAN(eps=0.1, metric="euclidean")
dbscan_pred = model_dbscan.fit_predict(x)
axes[1, 1].scatter(x[:, 0], x[:, 1], c=dbscan_pred, cmap='viridis')
axes[1, 1].set_title("DBSCAN Clustering")


plt.tight_layout()
plt.show()


# Evaluating each Clustering method to know how accurately each of them analysis the spread
print(f"Silhoutte Score of Algomerative Clustering: {silhouette_score(x, algo_pred)}")
print(f"Silhoutte Score of Kmeans Clustering: {silhouette_score(x, kmeans_pred)}")
print(f"Silhoutte Score of DBSCAN Clustering: {silhouette_score(x, dbscan_pred)}")