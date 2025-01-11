import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 生成一些数据
np.random.seed(42)
data = np.random.normal(loc=0, scale=1, size=1000)  # 一维数据

# 单变量 KDE 图
plt.figure(figsize=(8, 4))
sns.kdeplot(data, color='blue', fill=True, label='KDE', alpha=0.5)
plt.title("Univariate KDE Plot")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()