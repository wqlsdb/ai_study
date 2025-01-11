"""
案例:
    演示线性回归 和 回归决策树对比.

结论:
    回归类问题, 既能用线性回归, 也能用决策树回归. 优先使用 线性回归, 因为 决策树回归可能会导致 过拟合.
"""

# 导包.
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor      # 回归决策树
from sklearn.linear_model import LinearRegression   # 线性回归
import matplotlib.pyplot as plt                     # 绘图


# 1. 获取数据.
x = np.array(list(range(1,11))).reshape(-1, 1)
y = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05])
print(x)
print(y)

# 2. 创建线性回归 和 决策树回归模型.
estimator1 = LinearRegression()                    # 线性回归
estimator2 = DecisionTreeRegressor(max_depth=1)    # 回归决策树, 层数=1
estimator3 = DecisionTreeRegressor(max_depth=3)    # 回归决策树, 层数=3

# 3. 训练模型.
estimator1.fit(x, y)
estimator2.fit(x, y)
estimator3.fit(x, y)

# 4. 准备测试数据, 用于测试.
# 起始, 结束, 步长.
x_test = np.arange(0.0, 10.0, 0.1).reshape(-1, 1)
print(x_test)

# 5. 模型预测.
y_predict1 = estimator1.predict(x_test)
y_predict2 = estimator2.predict(x_test)
y_predict3 = estimator3.predict(x_test)

# 6. 绘图
plt.figure(figsize=(10, 5))

# 散点图(原始的坐标)
plt.scatter(x, y, color='gray', label='data')
# 线性回归的预测结果
plt.plot(x_test, y_predict1, color='r', label='liner regression')
# 回归决策树, 层数=1
plt.plot(x_test, y_predict2, color='b', label='max depth=1')
# 回归决策树, 层数=3
plt.plot(x_test, y_predict3, color='g', label='max depth=3')
# 显示图例.
plt.legend()
# 设置x轴标签.
plt.xlabel('data')
# 设置y轴标签.
plt.ylabel('target')
# 设置标题
plt.title('Decision Tree Regression')

# 显示图片
plt.show()