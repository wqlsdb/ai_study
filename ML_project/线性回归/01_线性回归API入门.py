"""
线性回归解释:
    概述:
        线性回归是用来描述 1个或者多个自变量(特征)  和 因变量(标签) 之间关系 进行建模分析的 一种模型.
    分类:
        一元线性回归:
            1个特征, 1个标签.
        多元线性回归:
            多个特征, 1个标签.
    应用场景:
        有特征, 有标签, 且标签是连续的.
    线性回归的公式:
        一元线性回归: y = kx + b = wx + b
        多元线性回归: y = w的转置 * x + b
    名词解释:
        关于 y = kx + b,
        k: 斜率, 在机器学习中叫: 权重(Weight), 简称: w
        b: 截距, 在机器学习中叫: 偏置(Bias), 简称: b

回顾机器学习的 实现步骤:
    1. 获取数据.
    2. 数据预处理.
    3. 特征工程.
    4. 模型训练.
    5. 模型预测.
    6. 模型评估.
"""
# 案例: 基于身高, 预测体重.
# 导包.
from sklearn.linear_model import LinearRegression

# 1. 准备数据集.
x_train = [[160], [166], [172], [174], [180]]
y_train = [56.3, 60.6, 65.1, 68.5, 75]
x_test = [[176]]

# 2. 创建线性回归模型.
estimator = LinearRegression()

# 3. 模型训练.
estimator.fit(x_train, y_train)

# 4. 模型预测.
y_predict = estimator.predict(x_test)
print(f'预测值为: {y_predict}')

# 5. 额外给大家查看下, 关于上述 线性回归模型中的 权重 和 偏置结果.
print(f'权重(斜率): {estimator.coef_}')         # 0.92942177
print(f'偏置(截距): {estimator.intercept_}')    # -93.27346938775514