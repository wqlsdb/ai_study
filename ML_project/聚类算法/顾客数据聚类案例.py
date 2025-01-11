import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

data = pd.read_csv('./data/customers.csv')
data.info()
print(data.head(10))
print(data.shape)

X = data.iloc[:, [3, 4]]
# print(X.head(10))
sse_list = []
sc_list = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i,random_state=22)
    # 训练模型
    model = kmeans.fit(X)
    y_pred = model.predict(X)
    # 计算sse值，误差平方和
    sse_list.append(kmeans.inertia_)
    # 计算score值，轮廓系数
    sc_list.append(silhouette_score(X, y_pred))

print(sse_list)

plt.plot(range(2, 11), sse_list)
plt.xlabel('number of clusters')
plt.ylabel('SSE')
plt.title('SSE')
plt.grid()
plt.show()

plt.title('SC')
plt.plot(range(2, 11), sc_list)
plt.grid(True)
plt.show()


KMeans = KMeans(n_clusters=5,random_state=22)
y_pred = KMeans.fit_predict(X)
# print('y_pred',y_pred)

# print('聚类中心质心点',KMeans.cluster_centers_)

plt.scatter(X.iloc[y_pred == 0, 0], X.iloc[y_pred == 0, 1], s=100, c='red', label='Standard')
