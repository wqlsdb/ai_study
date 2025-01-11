import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# todo: 1-加载csv文件数据集
data = pd.read_csv('data/customers.csv')
print(data.head())
print(data.shape)
data.info()
# todo: 2-获取进行聚类的特征列
# 根据收入和消费指数进行聚类
X = data.iloc[:, [3, 4]]
print(X)
# 实际工作中还需要进行特征处理 标准化/删除异常值等
print('==============确定最佳K值==============')
# 最佳的k=5
# todo: 3-使用KMeans算法进行聚类, 绘制肘部图/sc系数图
sse_list = []
sc_list = []
for i in range(2, 11):
	# 创建kmeans对象
	kmeans = KMeans(n_clusters=i, random_state=22)
	# 调用fit方法进行模型训练
	kmeans.fit(X)
	# 预测, 聚类
	y_pred = kmeans.predict(X)
	# 计算sse值
	sse_list.append(kmeans.inertia_)
	# 计算sc值
	sc_list.append(silhouette_score(X, y_pred))

# print(sse_list)
plt.plot(range(2, 11), sse_list)
plt.title('the elbow method')
plt.xlabel('number of clusters')
plt.ylabel('mysse')
plt.grid()
plt.show()
plt.title('sc')
plt.plot(range(2, 11), sc_list)
plt.grid(True)
plt.show()

print('==============根据最佳k值进行聚类==============')
kmeans = KMeans(n_clusters=4, random_state=22)
# kmeans = MiniBatchKMeans(n_clusters=4, random_state=22)
y_pred = kmeans.fit_predict(X)
print('y_pred->', y_pred)
# 查看每类的质心点值
print('cluster_centers->', kmeans.cluster_centers_)
# 通过可视化图形分析聚类结果
print(X.iloc[y_pred == 0])
print(X.iloc[y_pred == 1])
print(X.iloc[y_pred == 2])
# y_pred == 0:行数据根据布尔值取值
plt.scatter(X.iloc[y_pred == 0, 0], X.iloc[y_pred == 0, 1], s=100, c='red', label='Standard')
plt.scatter(X.iloc[y_pred == 1, 0], X.iloc[y_pred == 1, 1], s=100, c='blue', label='Traditional')
plt.scatter(X.iloc[y_pred == 2, 0], X.iloc[y_pred == 2, 1], s=100, c='green', label='Normal')
plt.scatter(X.iloc[y_pred == 3, 0], X.iloc[y_pred == 3, 1], s=100, c='cyan', label='Youth')
plt.scatter(X.iloc[y_pred == 4, 0], X.iloc[y_pred == 4, 1], s=100, c='magenta', label='TA')
# plt.scatter(X.iloc[:, 0], X.iloc[:, 1], s=100, c=y_pred)
# kmeans.cluster_centers_:聚类后每类的中心点
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='black', label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()