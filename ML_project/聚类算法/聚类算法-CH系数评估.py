from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

#
x, y = make_blobs(
    n_samples=1000,
    n_features=2,
    centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
    # todo 标准差和方差都是查看数据的离散程度
    cluster_std=[0.4, 0.2, 0.2, 0.2],
    random_state=22
)

# kmeans = KMeans(n_clusters=4)
# model = kmeans.fit(x)
# # 计算sse值，误差平方和
# print("sse:", model.inertia_)

sse_list = []
ch_list = []
for i in range(2, 100):
    kmeans = KMeans(n_clusters=i)
    model = kmeans.fit(x)
    y_pred = model.predict(x)
    sse_list.append(kmeans.inertia_)
    ch_list.append(calinski_harabasz_score(x, y_pred))

plt.figure(figsize=(8, 8), dpi=100)
plt.xticks(range(0, 100, 3), labels=range(0, 100, 3))
plt.grid()
plt.title("CH")
plt.plot(range(2, 100), ch_list, 'or-')

plt.show()