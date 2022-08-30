import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
from common import make_df, process_merchant_avg_cents_hour_of_day, filter_weekday_hour

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.manifold import TSNE

df_feature_vectors = pd.read_pickle('feature_vectors_fourier_day_week_month_normalized_to_dc_harmonic_top200_trans_sum.pkl')
print(df_feature_vectors.shape)

id_merchant = int('005e8bb6fb', 16)
df_feature_vectors = df_feature_vectors.set_index('merchant')
df_feature_vectors = df_feature_vectors.dropna()
print(df_feature_vectors)

feature_vectors = np.array(df_feature_vectors['vector'].tolist())
print(feature_vectors)

feature_vectors = feature_vectors[:, :8]
print(feature_vectors.shape)

# print(np.nonzero(np.isnan(feature_vectors)))
# print(np.min(feature_vectors))
# feature_vectors = feature_vectors[:, 8:]
# print(feature_vectors.shape)

# feature_vectors = feature_vectors[:, :8]
# to_be_scaled_features = feature_vectors[:, :8]

# Scale the feature vectors for average daily sales
# print("scaling")
# scaler = MinMaxScaler()
# scaler.fit(feature_vectors)
# print(scaler.data_max_)
# feature_vectors = scaler.transform(feature_vectors)


# drop people with only one transaction.
# print("msk_fv")
# msk_fv = np.array(feature_vectors[:, 8] > 1000)
# print(msk_fv.shape)
# feature_vectors = feature_vectors[msk_fv, :]

# print(msk_fv.shape)

# pca = PCA(2)  # project from 9 to 2 dimensions
# projected = pca.fit_transform(feature_vectors)
# print(feature_vectors)
# print(projected)

# plt.scatter(projected[:, 0], projected[:, 1],
#              edgecolor='none', alpha=0.5,
#             cmap=plt.cm.get_cmap('rainbow', 10))
# plt.xlabel('component 1')
# plt.ylabel('component 2')
# plt.colorbar();
# plt.show()
# pca = PCA().fit(feature_vectors)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance');

# Elbow Plot
wcss = [] 
for i in range(1, 11): 
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(feature_vectors) 
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

n_clusters = 4
n_rows = 1

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(feature_vectors)
print("kmeans clusters")
print(kmeans.cluster_centers_)

print(clusters)

msk_cluster_1 = clusters == 1
msk_cluster_2 = clusters == 2
msk_cluster_3 = clusters == 3
msk_cluster_4 = clusters == 4
msk_cluster_5 = clusters == 5

# cluster_1_merchants = df_feature_vectors.iloc[msk_cluster_1].reset_index()['merchant'].tolist()

# cluster_1_merchants_rnd = np.random.choice(cluster_1_merchants, 3)
# print("Cluster 1 Merchants")
# print(cluster_1_merchants_rnd)

# cluster_2_merchants = df_feature_vectors.iloc[msk_cluster_2].reset_index()['merchant'].tolist()

# cluster_2_merchants_rnd = np.random.choice(cluster_2_merchants, 3)
# print("Cluster 2 Merchants")
# print(cluster_2_merchants_rnd)

# cluster_3_merchants = df_feature_vectors.iloc[msk_cluster_3].reset_index()['merchant'].tolist()

# cluster_3_merchants_rnd = np.random.choice(cluster_3_merchants, 3)
# print("Cluster 3 Merchants")
# print(cluster_3_merchants_rnd)

# cluster_4_merchants = df_feature_vectors.iloc[msk_cluster_4].reset_index()['merchant'].tolist()

# cluster_4_merchants_rnd = np.random.choice(cluster_4_merchants, 3)
# print("Cluster 4 Merchants")
# print(cluster_4_merchants_rnd)

# print(msk_cluster_5)
# cluster_5_merchants = df_feature_vectors.iloc[msk_cluster_5].reset_index()['merchant'].tolist()

# cluster_5_merchants_rnd = np.random.choice(cluster_5_merchants, 3)
# print("Cluster 5 Merchants")
# print(cluster_5_merchants_rnd)

for fv in kmeans.cluster_centers_:
    plt.plot(fv, '.-')

plt.title("Kmeans Cluster Centers")
plt.xlabel("Feature Vector")
plt.ylabel("Magnitude")
# fig, ax = plt.subplots(n_rows, n_clusters // n_rows, figsize=(8, 3))
# centers = kmeans.cluster_centers_
# for axi, center in zip(ax.flat, centers):
#     axi.set(xticks=[], yticks=[])
#     axi.plot(center, 'o-')

plt.show()


# pca = PCA(2)  # project from 64 to 2 dimensions
# projected = pca.fit_transform(kmeans.cluster_centers_)

# plt.scatter(projected[:, 0], projected[:, 1],
#              edgecolor='none', alpha=0.5,
#             cmap=plt.cm.get_cmap('rainbow', 10))
# plt.xlabel('component 1')
# plt.ylabel('component 2')
# plt.colorbar();
plt.show()



# # plt.style.use("fivethirtyeight")
# plt.figure(figsize=(8, 8))

# scat = sns.scatterplot(
#     "component_1",
#     "component_2",
#     s=50,
#     data=pcadf,
#     hue="predicted_cluster",
#     palette="Set2",
# )

# scat.set_title(
#     "Average Daily Sales by 3 Hour Blocks"
# )
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

plt.show()