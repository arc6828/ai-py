import pandas as pd
from sklearn.impute import SimpleImputer

# โหลดข้อมูล
df = pd.read_csv("data.csv")

# แทนค่าที่หายไปด้วยค่าเฉลี่ย
imputer = SimpleImputer(strategy="mean")
df["feature"] = imputer.fit_transform(df[["feature"]])


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)



from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.1)
df_selected = selector.fit_transform(df)


import seaborn as sns
import matplotlib.pyplot as plt

correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.show()


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
df_pca = pca.fit_transform(df)


import matplotlib.pyplot as plt

plt.scatter(df["feature1"], df["feature2"])
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Scatter Plot of Features")
plt.show()


import seaborn as sns

sns.pairplot(df)
plt.show()


from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30)
df_tsne = tsne.fit_transform(df)

plt.scatter(df_tsne[:, 0], df_tsne[:, 1])
plt.title("t-SNE Visualization")
plt.show()



from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# สร้างโมเดล K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# คำนวณ Silhouette Score
score = silhouette_score(X, clusters)
print(f"Silhouette Score: {score}")



import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

wcss = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)  # ค่า WCSS

# วาดกราฟ Elbow Method
plt.plot(k_range, wcss, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal k')
plt.show()


from sklearn.decomposition import PCA
import numpy as np

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# ดูว่าแต่ละ Principal Component อธิบายข้อมูลได้กี่เปอร์เซ็นต์
explained_variance = pca.explained_variance_ratio_
print(f"Explained Variance: {explained_variance}")
print(f"Total Variance Explained: {np.sum(explained_variance) * 100:.2f}%")


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap="viridis")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE Visualization of Clustering")
plt.show()


plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=100, label='Centroids')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Clustering Results with K-Means")
plt.legend()
plt.show()


import seaborn as sns

correlation_matrix = pd.DataFrame(X).corr()

plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

