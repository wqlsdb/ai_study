"""
案例:
    演示正则化解决 拟合问题.

回顾:
    拟合: 指的是 模型和数据之间存在关系, 即: 预测值 和 真实值之间的关系.
    欠拟合: 模型在训练集, 测试集标签都不好.
        原因: 模型过于简单了.
    过拟合: 模型在训练集表现好, 在测试集表现不好.
        原因: 模型过于复杂了, 数据不纯, 数据量少.
    正好拟合(泛化): 模型在训练集, 测试集都好

    奥卡姆剃刀: 在误差相同(泛化程度一样)的情况下, 优先选择简单的模型.

正则化解释:
    概述:
        正则化: 是一种对模型复杂度的一种控制, 通过 降低特征的权重 实现.
    分类:
        L1正则化, 指的是: Lasso模块, 会 降低权重, 甚至可能降为0 -> 特征选取.
        L2正则化, 指的是: Ridge模块, 会 降低权重, 但不会降为0.
"""
# 导包
import numpy as np  # 主要是做数学相关运算操作等.
import matplotlib.pyplot as plt  # Matplotlib绘图的
from sklearn.linear_model import LinearRegression, Lasso, Ridge  # 线性回归的 正规方程解法, L1正则化, L2正则化.
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error  # 计算均方误差, 均方根误差, 平均绝对误差
from sklearn.model_selection import train_test_split  # 切割测试集和训练集.


# 1. 定义函数, 演示: 欠拟合情况.
def dm01_模型欠拟合():
    # 1. 生成数据.
    # 1.1 指定随机数种子, 种子一样, 则每次生成的随机数的规则都是相同的.
    np.random.seed(22)
    # 1.2 生成x轴的坐标.
    x = np.random.uniform(-3, 3, size=100)
    # 1.3 基于x轴的值, 结合线性公式, 生成y轴的值.
    # 回顾一元线性回归公式: y = kx + b, 这里为了数据效果更好, 加入噪声...
    # 即: y = 0.5x² + x + 2 + 噪声(正太分布生成的随机数).
    #  np.random.normal(0, 1, size=100)意思是: 生成正态分布的随机数, 均值是0, 标准差是1, 随机数个数是100.
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
    # 1.4 查看数据集.
    print(f'x轴的值: {x}')     #  [1, 2, 3]
    print(f'y轴的值: {y}')

    # 2. 数据的预处理.todo:,生成二维数组，作为特征列
    X1 = x.reshape(-1, 1)
    print(f'处理后, X: {X1}')   # [[1], [2], [3]]

    # 3. 创建模型(算法)对象.
    # 3.1 线性回归的 正规方程对象.
    estimator = LinearRegression()
    # 3.2 模型训练.
    estimator.fit(X1, y)

    # 4. 模型预测.
    y_predict = estimator.predict(X1)
    print(f'预测值: {y_predict}')

    # 5. 模型评估.
    print(f'均方误差: {mean_squared_error(y, y_predict)}')
    print(f'均方根误差: {root_mean_squared_error(y, y_predict)}')
    print(f'平均绝对误差: {mean_absolute_error(y, y_predict)}')

    # 6. 绘制图形.
    plt.scatter(x, y)                           # 真实值的x轴, y轴 绘制散点图.
    plt.plot(x, y_predict, color='red')   # 预测值的x轴, y轴 绘制折线图(充当拟合回归线)
    plt.show()


# 2. 定义函数, 演示: 正好拟合的情况.
def dm02_模型正好拟合():
    # 1. 生成数据.
    # 1.1 指定随机数种子, 种子一样, 则每次生成的随机数的规则都是相同的.
    np.random.seed(22)
    # 1.2 生成x轴的坐标.
    x = np.random.uniform(-3, 3, size=100)
    # 1.3 基于x轴的值, 结合线性公式, 生成y轴的值.
    # 回顾一元线性回归公式: y = kx + b, 这里为了数据效果更好, 加入噪声...
    # 即: y = 0.5x² + x + 2 + 噪声(正太分布生成的随机数).
    #  np.random.normal(0, 1, size=100)意思是: 生成正态分布的随机数, 均值是0, 标准差是1, 随机数个数是100.
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
    # 1.4 查看数据集.
    print(f'x轴的值: {x}')     # [1, 2, 3]
    print(f'y轴的值: {y}')

    # 2. 数据的预处理.todo:把一维数组x转换为二维数组X1
    X1 = x.reshape(-1, 1)
    # np.hstack()解释: 垂直合并, 即: 竖直方向合并.
    X2 = np.hstack([X1, X1 ** 2])
    print(f'处理后, X1: {X1}')   # [[1], [2], [3]]
    print(f'处理后, X2: {X2}')   # [[1, 1], [2, 4], [3, 9]]

    # 3. 创建模型(算法)对象.
    # 3.1 线性回归的 正规方程对象.
    estimator = LinearRegression()
    # 3.2 模型训练.
    estimator.fit(X2, y)

    # 4. 模型预测.
    y_predict = estimator.predict(X2)
    print(f'预测值: {y_predict}')

    # 5. 模型评估.
    print(f'均方误差: {mean_squared_error(y, y_predict)}')
    print(f'均方根误差: {root_mean_squared_error(y, y_predict)}')
    print(f'平均绝对误差: {mean_absolute_error(y, y_predict)}')

    # 6. 绘制图形.
    # 6.1 真实值的x轴, y轴 绘制散点图.
    plt.scatter(x, y)
    # 6.2 根据预测值的x轴(升序排序), y轴值, 绘制折线图(充当拟合回归线).
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='red')   # 预测值的x轴, y轴 绘制折线图(充当拟合回归线)
    plt.show()


# 3. 定义函数, 演示: 过拟合的情况.
def dm03_模型过拟合():
    # 1. 生成数据.
    # 1.1 指定随机数种子, 种子一样, 则每次生成的随机数的规则都是相同的.
    np.random.seed(22)
    # 1.2 生成x轴的坐标.
    x = np.random.uniform(-3, 3, size=100)
    # 1.3 基于x轴的值, 结合线性公式, 生成y轴的值.
    # 回顾一元线性回归公式: y = kx + b, 这里为了数据效果更好, 加入噪声...
    # 即: y = 0.5x² + x + 2 + 噪声(正太分布生成的随机数).
    #  np.random.normal(0, 1, size=100)意思是: 生成正态分布的随机数, 均值是0, 标准差是1, 随机数个数是100.
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
    # 1.4 查看数据集.
    print(f'x轴的值: {x}')     # [1, 2, 3]
    print(f'y轴的值: {y}')

    # 2. 数据的预处理.
    X1 = x.reshape(-1, 1)
    # np.hstack()解释: 垂直合并, 即: 竖直方向合并.
    X3 = np.hstack([X1, X1 ** 2, X1 ** 3, X1 ** 4, X1 ** 5, X1 ** 6, X1 ** 7, X1 ** 8, X1 ** 9, X1 ** 10])
    print(f'处理后, X1: {X1}')   # [[1], [2], [3]]
    print(f'处理后, X3: {X3}')   # [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], [3, 9, 27...]]

    # 3. 创建模型(算法)对象.
    # 3.1 线性回归的 正规方程对象.
    estimator = LinearRegression()
    # 3.2 模型训练.
    estimator.fit(X3, y)

    # 4. 模型预测.
    y_predict = estimator.predict(X3)
    print(f'预测值: {y_predict}')

    # 5. 模型评估.
    print(f'均方误差: {mean_squared_error(y, y_predict)}')
    print(f'均方根误差: {root_mean_squared_error(y, y_predict)}')
    print(f'平均绝对误差: {mean_absolute_error(y, y_predict)}')

    # 6. 绘制图形.
    # 6.1 真实值的x轴, y轴 绘制散点图.
    plt.scatter(x, y)
    # 6.2 根据预测值的x轴(升序排序), y轴值, 绘制折线图(充当拟合回归线).
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='red')   # 预测值的x轴, y轴 绘制折线图(充当拟合回归线)
    plt.show()


# 4. 定义函数, 演示: L1正则化 解决 过拟合.
def dm04_L1正则化():
    # 1. 生成数据.
    # 1.1 指定随机数种子, 种子一样, 则每次生成的随机数的规则都是相同的.
    np.random.seed(22)
    # 1.2 生成x轴的坐标.
    x = np.random.uniform(-3, 3, size=100)
    # 1.3 基于x轴的值, 结合线性公式, 生成y轴的值.
    # 回顾一元线性回归公式: y = kx + b, 这里为了数据效果更好, 加入噪声...
    # 即: y = 0.5x² + x + 2 + 噪声(正太分布生成的随机数).
    #  np.random.normal(0, 1, size=100)意思是: 生成正态分布的随机数, 均值是0, 标准差是1, 随机数个数是100.
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
    # 1.4 查看数据集.
    print(f'x轴的值: {x}')     # [1, 2, 3]
    print(f'y轴的值: {y}')

    # 2. 数据的预处理.
    X1 = x.reshape(-1, 1)
    # np.hstack()解释: 垂直合并, 即: 竖直方向合并.
    X3 = np.hstack([X1, X1 ** 2, X1 ** 3, X1 ** 4, X1 ** 5, X1 ** 6, X1 ** 7, X1 ** 8, X1 ** 9, X1 ** 10])
    print(f'处理后, X1: {X1}')   # [[1], [2], [3]]
    print(f'处理后, X3: {X3}')   # [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], [3, 9, 27...]]

    # 3. 创建模型(算法)对象.
    # 3.1 线性回归的 正规方程对象.
    # estimator = LinearRegression()

    # 创建 L1正则化对象.
    # 参数alpha: 正则化的(惩罚)系数, (惩罚)系数越大, 则正则化项的权重越小.
    estimator = Lasso(alpha=0.1)

    # 3.2 模型训练.
    estimator.fit(X3, y)

    # 4. 模型预测.
    y_predict = estimator.predict(X3)
    print(f'预测值: {y_predict}')

    # 5. 模型评估.
    print(f'均方误差: {mean_squared_error(y, y_predict)}')
    print(f'均方根误差: {root_mean_squared_error(y, y_predict)}')
    print(f'平均绝对误差: {mean_absolute_error(y, y_predict)}')

    # 6. 绘制图形.
    # 6.1 真实值的x轴, y轴 绘制散点图.
    plt.scatter(x, y)
    # 6.2 根据预测值的x轴(升序排序), y轴值, 绘制折线图(充当拟合回归线).
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='red')   # 预测值的x轴, y轴 绘制折线图(充当拟合回归线)
    plt.show()


# 5. 定义函数, 演示: L2正则化 解决 过拟合.
def dm05_L2正则化():
    # 1. 生成数据.
    # 1.1 指定随机数种子, 种子一样, 则每次生成的随机数的规则都是相同的.
    np.random.seed(22)
    # 1.2 生成x轴的坐标.
    x = np.random.uniform(-3, 3, size=100)
    # 1.3 基于x轴的值, 结合线性公式, 生成y轴的值.
    # 回顾一元线性回归公式: y = kx + b, 这里为了数据效果更好, 加入噪声...
    # 即: y = 0.5x² + x + 2 + 噪声(正太分布生成的随机数).
    #  np.random.normal(0, 1, size=100)意思是: 生成正态分布的随机数, 均值是0, 标准差是1, 随机数个数是100.
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
    # 1.4 查看数据集.
    print(f'x轴的值: {x}')     # [1, 2, 3]
    print(f'y轴的值: {y}')

    # 2. 数据的预处理.
    X1 = x.reshape(-1, 1)
    # np.hstack()解释: 垂直合并, 即: 竖直方向合并.
    X3 = np.hstack([X1, X1 ** 2, X1 ** 3, X1 ** 4, X1 ** 5, X1 ** 6, X1 ** 7, X1 ** 8, X1 ** 9, X1 ** 10])
    print(f'处理后, X1: {X1}')   # [[1], [2], [3]]
    print(f'处理后, X3: {X3}')   # [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], [3, 9, 27...]]

    # 3. 创建模型(算法)对象.
    # 3.1 线性回归的 正规方程对象.
    # estimator = LinearRegression()

    # 创建 L1正则化对象.
    # 参数alpha: 正则化的(惩罚)系数, (惩罚)系数越大, 则正则化项的权重越小.
    # estimator = Lasso(alpha=0.1)

    # 创建 L2正则化对象.
    estimator = Ridge(alpha=0.1)

    # 3.2 模型训练.
    estimator.fit(X3, y)

    # 4. 模型预测.
    y_predict = estimator.predict(X3)
    print(f'预测值: {y_predict}')

    # 5. 模型评估.
    print(f'均方误差: {mean_squared_error(y, y_predict)}')
    print(f'均方根误差: {root_mean_squared_error(y, y_predict)}')
    print(f'平均绝对误差: {mean_absolute_error(y, y_predict)}')

    # 6. 绘制图形.
    # 6.1 真实值的x轴, y轴 绘制散点图.
    plt.scatter(x, y)
    # 6.2 根据预测值的x轴(升序排序), y轴值, 绘制折线图(充当拟合回归线).
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='red')   # 预测值的x轴, y轴 绘制折线图(充当拟合回归线)
    plt.show()


# 6. 在main方法中测试.
if __name__ == '__main__':
    # dm01_模型欠拟合()
    # dm02_模型正好拟合()
    # dm03_模型过拟合()
    # dm04_L1正则化()
    dm05_L2正则化()
