'''
一元线性回归：y=kx+b=wx+b
    1个特征1个标签
多元线性回归: y=w的转置 * x + b
    多个特征1个标签
应用场景：
    有特征，有标签，且标签连续的
线性回归公式：

'''

# 1. 准备数据
x_train = [[160], [166], [172], [174], [180]]
y_train = [56.3, 60.6, 65.1, 68.5, 75]
x_test = [176]

from sklearn.linear_model import LinearRegression

# 创建线性回归模型
estimator = LinearRegression()
# 3.训练模型
estimator.fit(x_train, y_train)
# 4.模型预测
y_predict = estimator.predict(x_test)
print(f'预测结果为: {y_predict}')

# 5。查看线性回归模型中的 权重 和 偏执结果
print(f'模型的权重(斜率)为: {estimator.coef_}')
print(f'模型的偏置(截距)为: {estimator.intercept_}')
