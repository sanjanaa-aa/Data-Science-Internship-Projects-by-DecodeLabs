import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Week3/online_shoppers_intention.csv")

print(df.head())

print("\nShape:")
print(df.shape)
print("\nColumns:")
print(df.columns)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nData Types:")
print(df.dtypes)
# Remove Revenue column
df = df.drop("Revenue", axis=1)

print("\nNew Shape:")
print(df.shape)
# Encode categorical columns
df["Month"] = df["Month"].astype("category").cat.codes
df["VisitorType"] = df["VisitorType"].astype("category").cat.codes
df["Weekend"] = df["Weekend"].astype(int)

print("\nData Types After Encoding:")
print(df.dtypes)
from sklearn.preprocessing import StandardScaler

# Scale data
scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)

print("\nScaled Data Shape:")
print(scaled_data.shape)
from sklearn.decomposition import PCA

# Apply PCA
pca = PCA(n_components=2)

pca_data = pca.fit_transform(scaled_data)

print("\nPCA Shape:")
print(pca_data.shape)
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("\nTotal Variance Retained:")
print(pca.explained_variance_ratio_.sum())
from sklearn.cluster import KMeans

wcss = []

for k in range(1, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(pca_data)
    wcss.append(kmeans.inertia_)

print("\nWCSS Values:")
print(wcss)
plt.figure(figsize=(8,5))

plt.plot(range(1, 11), wcss, marker='o')

plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")

plt.show()
from sklearn.metrics import silhouette_score

for k in range(2, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(pca_data)

    score = silhouette_score(
        pca_data,
        labels
    )

    print(f"K={k} Silhouette Score: {score:.4f}")
# Final K-Means Model
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(pca_data)

# Add cluster labels to dataframe
df["Cluster"] = clusters

print("\nCluster Distribution:")
print(df["Cluster"].value_counts())
print("\nCluster Summary:")
print(df.groupby("Cluster").mean())
print("\nCustomer Personas")

print("""
Cluster 0:
Regular Visitors
- Medium browsing activity
- Average engagement
- Largest customer group
""")

print("""
Cluster 1:
Highly Engaged Customers
- High administrative and product-related activity
- Spend more time on the website
- Most valuable customer segment
""")

print("""
Cluster 2:
Low Engagement Customers
- Very low activity
- Visit fewer pages
- Least engaged segment
""")
plt.figure(figsize=(8,6))

plt.scatter(
    pca_data[:, 0],
    pca_data[:, 1],
    c=clusters
)

plt.title("Customer Segments using PCA and K-Means")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.show()