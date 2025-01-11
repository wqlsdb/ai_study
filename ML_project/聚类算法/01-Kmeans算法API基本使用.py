from sklearn.cluster import KMeans  # 算法
from sklearn.metrics import calinski_harabasz_score  # ch系数 评估指标
import matplotlib.pyplot as plt  # 绘图
from sklearn.datasets import make_blobs  # 生成数据集
import warnings

warnings.filterwarnings('ignore')

# todo: 1-生成1000个样本数据集, 每个样本有2个特征, 四类样本(4个中心点)
x, y = make_blobs(n_samples=1000,
				  n_features=2,
				  centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
				  cluster_std=[0.4, 0.2, 0.2, 0.2],
				  random_state=22)
print(x)
print(y)
# todo: 2-绘制样本数据集的散点图
plt.figure(figsize=(8, 8))
# x[:, 0]: 1000个样本的第一个特征值
plt.scatter(x[:, 0], x[:, 1], marker='o')
plt.show()
# todo: 3-调用KMeans算法, 实现聚类
# 创建kmeans对象
# n_clusters: 质心数, 类别数
kmeans = KMeans(n_clusters=4)
# y_pred = kmeans.fit_predict(x)
# 调用kmeans对象的fit方法, 训练模型
model = kmeans.fit(x)
# 模型进行聚类
y_pred = model.predict(x)
print('y_pred->', y_pred)
print('每类中心点的值->', kmeans.cluster_centers_)

plt.figure(figsize=(8, 8))
# x[:, 0]: 1000个样本的第一个特征值
plt.scatter(x[:, 0], x[:, 1], c=y_pred)
plt.show()

# todo: 4-模型评估
print('ch-score->', calinski_harabasz_score(x, y_pred))