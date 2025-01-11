from sklearn.cluster import KMeans  # 算法
from sklearn.metrics import calinski_harabasz_score  # ch系数 评估指标
from sklearn.metrics import silhouette_score  # sc系数
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

# todo: 2-使用kmeans算法聚类
# 创建空列表, 存储每个k值对应的sse
sse_list = []
# sc列表
sc_list = []
# ch列表
ch_list = []
# 循环遍历k值, 计算每个模型的sse值
for i in range(2, 100):
	# i是k值
	kmeans = KMeans(n_clusters=i, max_iter=100)
	model = kmeans.fit(x)
	# 聚类, 预测y值
	y_pred = model.predict(x)
	# 计算sse值 误差平方和 存储到列表中
	sse_list.append(kmeans.inertia_)
	# sc系数
	sc_list.append(silhouette_score(x, y_pred))
	# ch系数
	ch_list.append(calinski_harabasz_score(x, y_pred))

# 绘制肘部图
plt.figure(figsize=(18, 8), dpi=100)
# 设置x刻度值
plt.xticks(range(0, 100, 3), labels=range(0, 100, 3))
plt.grid()
# plt.title('sse')
# plt.title('sc')
plt.title('ch')
# 绘图 x轴的值是k值, y轴的值是sse值
# plt.plot(range(2, 100), sse_list, 'or-')
# plt.plot(range(2, 100), sc_list, 'or-')
plt.plot(range(2, 100), ch_list, 'or-')
plt.show()