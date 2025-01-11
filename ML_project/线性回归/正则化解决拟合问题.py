import numpy as np


def dm01_模型欠拟合():
    # 1.生成数据
    import numpy as np
    np.random.seed(22)
    # 1.2 生成x轴的值,100条数据 todo:特征列
    x = np.random.uniform(-3, 3, size=100)
    # 1.3 基于x轴的值，结合线性公式，生成y轴的数据 todo:标签列
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)

    # 2. 数据的预处理,生成二维数组 todo:特征列
    X1 = x.reshape(-1, 1)
    # print(f'生成的二维数组X1：{X1}')
    # 3.创建模型对象 线性回归算法
    from sklearn.linear_model import LinearRegression
    estimator = LinearRegression()
    # 3.2 模型训练
    estimator.fit(X1, y)

    # 4.模型预测
    y_predict = estimator.predict(X1)
    print(f'预测值：{y_predict}')

    # 5.模型评估
    from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error
    print(f'均方误差: {mean_squared_error(y, y_predict)}')
    print(f'均方根误差: {root_mean_squared_error(y, y_predict)}')
    print(f'平均绝对误差: {mean_absolute_error(y, y_predict)}')
    # 6.绘制图形
    import matplotlib.pyplot as plt
    plt.scatter(x, y)
    plt.plot(x, y_predict, color='red')
    plt.show()


def dm02_模型正好拟合():
    # 1.生成数据
    np.random.seed(22)
    x = np.random.uniform(-3, 3, size=100)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
    # 2. 数据预处理
    X1 = x.reshape(-1, 1)
    # X2 = np.hstack([X1, X1 ** 2, X1 ** 300])
    # 2.1 todo:增加特征列实现正好拟合
    X2 = np.hstack([X1, X1 ** 2])

    # 3.创建模型对象
    # 3.1 线性回归的 正规方程对象
    from sklearn.linear_model import LinearRegression
    estimator = LinearRegression()
    # 3.2 模型训练
    estimator.fit(X2, y)

    # 4. 模型预测
    y_predict = estimator.predict(X2)
    print(f'预测值：{y_predict}')
    from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error
    print(f'均方误差: {mean_squared_error(y, y_predict)}')
    print(f'均方根误差: {root_mean_squared_error(y, y_predict)}')
    print(f'平均绝对误差: {mean_absolute_error(y, y_predict)}')

    # 6.绘制图形
    import matplotlib.pyplot as plt
    plt.scatter(x, y)
    plt.plot(np.sort(x), y_predict[np.argsort(x)], color='red')
    plt.show()




if __name__ == '__main__':
    # dm01_模型欠拟合()
    dm02_模型正好拟合()