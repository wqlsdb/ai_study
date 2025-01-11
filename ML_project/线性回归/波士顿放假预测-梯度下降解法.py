import pandas as pd
import numpy as np

# 1. 获取数据.共计506条数据
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
# print(f'特征: {len(data)}, {data[:3]}')
# print(f'标签: {len(target)}, {target[:3]}')

# 2. 数据预处理
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=22)

# 3.特征工程，特征提取，特征预处理（标准化，归一化）,特征降维，特征选取，特征组合
# 3.1 创建标准化对象
from sklearn.preprocessing import StandardScaler

transfer = StandardScaler()

# 3.2 对训练和 测试集的 特征进行标准化处理
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 4. 模型训练
# 4.1 创建线性回归之 梯度下降对象
from sklearn.linear_model import SGDRegressor

# todo                      参1；是否计算偏置   参2：学习率固定成：常量    参3：学习率的值
estimator = SGDRegressor(fit_intercept=True, learning_rate='constant', eta0=0.001)
# 4.2 模型训练
estimator.fit(x_train, y_train)

# 5.模型预测
y_predict = estimator.predict(x_test)
print(f'查看每列特征数据的权重：{estimator.coef_}')
print(f'偏置：{estimator.intercept_}')
print('-' * 88)

# 6.模型评估
# 参1：真实值，参2：预测值
# MAE(平均绝对误差)和RMSE（均方根误差）都可以衡量模型的预测效果，如果更关注异常值，就用RMSE
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error      # 均方误差评估, 平均绝对值误差评估, 均方根误差评估
print(f'平均绝对误差：{mean_absolute_error(y_test,y_predict)}')
print(f'均方误差:{mean_squared_error(y_test,y_predict)}')

print(f'均方根误差：{root_mean_squared_error(y_test,y_predict)}')
print(f'均方根误差：{np.sqrt(mean_squared_error(y_test,y_predict))}')
