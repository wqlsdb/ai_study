from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# 1. 准备数据集.


data = {
    '电视广告数(x)': [1, 3, 2, 1, 3],
    '汽车销售数(y)': [14, 24, 18, 17, 27]
}
df = pd.DataFrame(data)
print(df)

X = df[['电视广告数(x)']]
y = df['汽车销售数(y)']
# 划分训练集和测试集.
x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=22)

# 2. 创建线性回归模型.
estimator = LinearRegression()

# 3. 模型训练.
estimator.fit(x_train, y_train)

# 4. 模型预测.
y_predict = estimator.predict(x_test)
print(f'预测值为: {y_predict}')

# 5. 额外给大家查看下, 关于上述 线性回归模型中的 权重 和 偏置结果.
print(f'权重(斜率): {estimator.coef_}')  # 0.92942177
print(f'偏置(截距): {estimator.intercept_}')

# plt.plot(x_train, y_train, 'o')
plt.plot(x_train, estimator.predict(x_train), color='red')

plt.scatter(x_train, y_train)
plt.legend()
plt.show()
