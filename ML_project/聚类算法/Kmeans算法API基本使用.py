from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

#
x, y = make_blobs(
    n_samples=10,
    n_features=2,
    centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
    # todo 标准差和方差都是查看数据的离散程度
    cluster_std=[0.4, 0.2, 0.2, 0.2],
    random_state=22
)
# 打印特征，也就是每个样本点的坐标
print(f'x: {x}')
print(f'y: {y}')

plt.figure(figsize=(8, 8))
plt.scatter(x[:, 0], x[:, 1], marker='x')
plt.show()

# 创建模型
kmeans = KMeans(n_clusters=4, random_state=22)
model = kmeans.fit(x)
# 预测每个点属于哪个簇
y_pred = model.predict(x)
print(f'y_pred: {y_pred}')
plt.figure(figsize=(8, 8))
plt.scatter(x[:, 0], x[:, 1], c=y_pred,marker='x')
plt.show()

# 模型评估
print(f'模型评估：{calinski_harabasz_score(x, y_pred)}')
