import pandas as pd
import numpy as np

# 1. 获取数据
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])

target = raw_df.values[1::2, 2]

# 2.数据预处理
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=22)
# 3.特征工程
# 3.1 创建标准化对象
from sklearn.preprocessing import StandardScaler

transfer = StandardScaler()

# 3.2 对训练和测试集的特征进行标准化
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)
# 4. 模型训练
# 4.1 创建模型对象 线性回归只正规方程
# 参数：fit_intercept: 是否需要偏置值，默认为True
from sklearn.linear_model import LinearRegression

estimator = LinearRegression(fit_intercept=True)
# 4.2 模型训练
estimator.fit(x_train, y_train)

# 5. 模型预测
y_predict = estimator.predict(x_test)
print(f'预测值为：{y_predict}')
print(f'查看每个特征的权重：{estimator.coef_}')
print(f'偏置：{estimator.intercept_}')
print('-' * 88)
# 6. 模型评估
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error

print(f'平均绝对误差MAE：{mean_absolute_error(y_test, y_predict)}')
print(f'均方误差MSE：{mean_squared_error(y_test, y_predict)}')

print(f'均方根误差：{root_mean_squared_error(y_test, y_predict)}')
print(f'均方根误差numpy：{np.sqrt(mean_squared_error(y_test, y_predict))}')
