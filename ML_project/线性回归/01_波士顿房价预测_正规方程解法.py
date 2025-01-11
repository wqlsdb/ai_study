"""
案例:
    波士顿房价预测, 正规方程解法.

回顾:
    正规方程解法 和 梯度下降的区别:
        相同点: 都可以用来找 损失函数的极小值, 评估 回归模型.
        不同点:
            正规方程: 一次性求解, 资源开销较大, 适合于小批量干净的数据集, 如果数据没有逆, 也无法计算.
            梯度下降: 迭代求解, 资源开销相对较小, 适用于大批量的数据集, 实际开发 更推荐.
                分类:
                    全梯度下降
                    随机梯度下降
                    小批量梯度下降
                    随机平均梯度下降

    线性回归模型 评估的方案:
        方案1: 平均绝对误差, Mean Absolute Error,  误差绝对值的和 的 平均值.
        方案2: 均方误差, Mean Squared Error, 误差的平方和的平均值
        方案3: 均方根误差, Root Mean Squared Error, 误差的平方和的平均值的 平方根.

    机器学习项目的研发流程:
        1. 获取数据.
        2. 数据的预处理.
        3. 特征工程.
            特征提取, 特征预处理(标准化, 归一化), 特征降维, 特征选取, 特征组合.
        4. 模型训练.
        5. 模型评估.
        6. 模型预测.
"""

# 导包
# from sklearn.datasets import load_boston                # 数据
from sklearn.preprocessing import StandardScaler        # 特征处理
from sklearn.model_selection import train_test_split    # 数据集划分
from sklearn.linear_model import LinearRegression       # 正规方程的回归模型
from sklearn.linear_model import SGDRegressor           # 梯度下降的回归模型
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error      # 均方误差评估, 平均绝对值误差评估, 均方根误差评估
import pandas as pd
import numpy as np


# 1. 获取数据.
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
# print(f'特征: {len(data)}, {data[:5]}')
# print(f'目标: {len(target)}, {target[:5]}')

# 2. 数据的预处理.
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=22)

# 3. 特征工程, 特征提取, 特征预处理(标准化, 归一化), 特征降维, 特征选取, 特征组合.
# 3.1 创建标准化对象.
transfer = StandardScaler()
# 3.2 对训练和 和 测试集的 特征进行标准化处理.
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 4. 模型训练.
# 4.1 创建 线性回归之 正规方程对象.
# 参数: fit_intercept: 是否计算截距(bias, 偏置), 默认是: True
estimator = LinearRegression(fit_intercept=True)
# 4.2 模型训练.
estimator.fit(x_train, y_train)

# 5. 模型预测.
y_predict = estimator.predict(x_test)
# print(f'预测值为: {y_predict}')
print(f'查看每个特征的权重: {estimator.coef_}')
print(f'偏置: {estimator.intercept_}')
print('-' * 22)

# 6. 模型评估.
# 参1: 真实值, 参2: 预测值
# MAE(平均绝对误差) 和 RMSE(均方根误差) 都可以衡量模型的预测效果, 如果更关注异常值, 就用 RMSE
print(f'平均绝对误差: {mean_absolute_error(y_test, y_predict)}')  # 3.425181871853366
print(f'均方误差: {mean_squared_error(y_test, y_predict)}')      # 20.77068478427006

print(f'均方根误差: {root_mean_squared_error(y_test, y_predict)}')       # 4.557486674063903
print(f'均方根误差: {np.sqrt(mean_squared_error(y_test, y_predict))}')   # 4.557486674063903, 效果同上, sqrt全称是: square root
