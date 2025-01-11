import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 生成数据
np.random.seed(42)
x1 = np.random.rand(100, 1)  # 线性特征
x2 = np.random.rand(100, 1)  # 非线性特征
y = 3 * x1 + 5 * x2**2 + np.random.randn(100, 1) * 0.1  # 混合关系

# 组合数据
data = pd.DataFrame(np.hstack([x1, x2]), columns=['x1', 'x2'])
data['x2_squared'] = data['x2']**2

# 拟合模型
model = LinearRegression()
model.fit(data[['x1', 'x2_squared']], y)

# 预测值
y_pred = model.predict(data[['x1', 'x2_squared']])

# 绘图
plt.figure(figsize=(10, 6))

# 绘制真实值和拟合值
plt.scatter(x1, y, color='blue', label='True Data (Linear Component)', alpha=0.6)
plt.scatter(x2, y, color='green', label='True Data (Nonlinear Component)', alpha=0.6)
plt.scatter(x1, y_pred, color='red', label='Predicted (Linear + Nonlinear)', alpha=0.8)

# 添加标题和标签
plt.title("Linear and Nonlinear Relationship")
plt.xlabel("Features (x1 and x2)")
plt.ylabel("Target (y)")
plt.legend()
plt.grid()
plt.show()
