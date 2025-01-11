import numpy as np

import pandas as pd

from sklearn.tree import DecisionTreeRegressor  # 回归决策树

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

x = np.array(list(range(1, 11))).reshape(-1, 1)
print(x)
y = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05])

# 2.创建逻辑回归 和 决策树回归模型
model1 = LinearRegression()  # 线性回归模型
model2 = DecisionTreeRegressor(max_depth=1)  # 回归决策树模型
model3 = DecisionTreeRegressor(max_depth=3)  # 回归决策树模型

# 3. 训练模型
model1.fit(x, y)
model2.fit(x, y)
model3.fit(x, y)

# 4.准备测试数据，用于猜测
# 起始，结束，步长
x_test = np.arange(0.0, 10.0, 0.1).reshape(-1, 1)

# 5. 模型预测
y_predict1 = model1.predict(x_test)
y_predict2 = model2.predict(x_test)
y_predict3 = model3.predict(x_test)

# 6. 绘图
plt.figure(figsize=(50, 30))
plt.scatter(x, y, color='green', label='原始数据')
plt.plot()
