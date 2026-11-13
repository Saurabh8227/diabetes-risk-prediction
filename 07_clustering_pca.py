"""STEP 7 - PATIENT GROUPS (UNSUPERVISED)  (Unit V)
Group patients (without using the Outcome) using K-Means, view them with PCA.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from config import *

df = pd.read_csv(CLEAN_FILE)
X = StandardScaler().fit_transform(df.drop(columns=TARGET))

# 1. Choose number of clusters (elbow + silhouette)
inertia, sil = [], []
ks = range(2, 9)
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X, km.labels_))
    print(f"k={k}  silhouette={sil[-1]:.3f}")
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(ks, inertia, "o-"); ax[0].set_title("Elbow Method")
ax[1].plot(ks, sil, "o-"); ax[1].set_title("Silhouette Score")
plt.savefig(f"{OUT_DIR}/cluster_k_selection.png"); plt.close()

# 2. Final clustering with k=3
km = KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE).fit(X)
df["Cluster"] = km.labels_

# 3. PCA to 2D for plotting
pcs = PCA(n_components=2).fit_transform(X)
plt.scatter(pcs[:, 0], pcs[:, 1], c=df["Cluster"], cmap="viridis", s=15)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Patient Clusters (PCA view)")
plt.savefig(f"{OUT_DIR}/clusters_pca.png"); plt.close()

# 4. Profile each cluster - this is the real insight
print("\nCluster profile (average values):")
print(df.groupby("Cluster").mean().round(2))
print("\nDiabetes rate per cluster:\n", df.groupby("Cluster")[TARGET].mean().round(2))
