import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ==========================================
# STEP 1: Load Data & Preprocessing
# ==========================================
df = pd.read_csv("cleaned_dataset.csv")

# صرف عددی (Numeric) کالمز کو الگ کرنا
numeric_df = df.select_dtypes(include=[np.number]).dropna()

# 1. Scaling (StandardScaler) - تمام کالمز کو ایک ہی اسکیل پر لانا
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df)

print("=== Data Scaled Successfully ===")
print("Scaled Shape:", X_scaled.shape)

# ==========================================
# STEP 2: PCA (Dimensionality Reduction)
# ==========================================
# ڈیٹا کو نچوڑ کر 2 اہم ابعاد (Components) میں تبدیل کرنا
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"\nExplained Variance Ratio (2 Components): {pca.explained_variance_ratio_}")
print(f"Total Variance Retained: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")

# ==========================================
# STEP 3: Elbow Method & Silhouette Score (Proving Optimal K)
# ==========================================
wcss = []
silhouette_scores = []
K_range = range(2, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_pca)
    wcss.append(kmeans.inertia_)
    score = silhouette_score(X_pca, kmeans.labels_)
    silhouette_scores.append(score)

# 1. Elbow Method Graph
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(K_range, wcss, 'bo-', markersize=8)
plt.title('Elbow Method (WCSS vs K)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Inertia)')

# 2. Silhouette Score Graph
plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, 'ro-', markersize=8)
plt.title('Silhouette Score vs K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')

plt.tight_layout()
plt.savefig('cluster_validation.png')
plt.show()

# ==========================================
# STEP 4: K-Means Clustering (Optimal K = 3)
# ==========================================
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
numeric_df['Cluster'] = kmeans.fit_predict(X_pca)

# ==========================================
# STEP 5: Visualize PCA Clusters
# ==========================================
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=numeric_df['Cluster'], palette='Set1', s=70)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='black', marker='X', label='Centroids')
plt.title('Customer Segments in PCA Space')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.tight_layout()
plt.savefig('pca_customer_clusters.png')
plt.show()

# ==========================================
# STEP 6: Reverse-Engineering & Business Personas
# ==========================================
# ہر کلسٹر کی اوسط خصوصیات معلوم کرنا
cluster_summary = numeric_df.groupby('Cluster').mean()
print("\n================ Business Personas Summary ================")
print(cluster_summary[['Quantity', 'TotalPrice', 'ItemsInCart', 'Has_Coupon']].head())

# نتائج کو نئی CSV میں محفوظ کریں
numeric_df.to_csv("customer_segments.csv", index=False)
print("\n=== Customer Segmentation Finished & Saved ===")

