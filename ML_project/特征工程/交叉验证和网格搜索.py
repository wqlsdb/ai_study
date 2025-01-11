from sklearn.datasets import load_iris          # 加载鸢尾花测试集的.
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV    # 分割训练集和测试集的
from sklearn.preprocessing import StandardScaler        # 数据标准化的
from sklearn.neighbors import KNeighborsClassifier      # KNN算法 分类对象
from sklearn.metrics import accuracy_score              # 模型评估的, 计算模型预测的准确率
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False



# 1. 加载数据集.
iris_data = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=22)
# 3. 特征工程.
# 特征提取: 无需手动实现, 数据已经提取完毕后, 分别是: 花萼的长度, 花萼的宽度, 花瓣的长度, 花瓣的宽度.
# 字段名如下: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# 特征预处理(归一化, 标准化), 我们发现特征(共4列)差值都不大, 所以可以不做特征预处理.
# 但是加上这个步骤以后, 会让我们的代码更加的完整, 所以我给大家加上这个操作.
# 3.1 创建标准化对象.
transfer = StandardScaler()
# 3.2 对训练集的特征做: 训练(拟合) 及 转换的操作.
# fit_transform(): 先基于训练集的特征进行拟合, 例如: 获取均值, 标准差, 方差等信息, 在基于内置好的模型(标准化对象), 调整参数, 进行转换.
x_train = transfer.fit_transform(x_train)
# 3.3 对测试集的特征做:  转换的操作.
# transform(): 基于训练集的特征, 调用内置好的模型(标准化对象), (基于上一步已经拟合后的结果)调整参数, 进行转换.
x_test = transfer.transform(x_test)

estimator=KNeighborsClassifier()
param_grid = {'neighbors':list(range(1,21))}

estimator=GridSearchCV(estimator,param_grid,cv=4)

estimator.fit(x_train,y_train)

