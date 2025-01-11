import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

data = pd.read_csv('./data/breast-cancer-wisconsin.csv')
# data.info()  # 699 hang , 11 lie,无空行有脏数据‘？’
# 2.特征预处理
# 2.1 缺失值处理
data = data.replace('?', np.nan)

# 2.2 缺失值不多可以直接删除，不影响实际结果
data.dropna(axis=0, inplace=True)

# 3. 特征工程
x = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

# 3.2 查看结果
# print(len(x), len(y))
# print(f'{x.head(5)}')
# print(f'{y.head(5)}')

# 3.3拆分训练集 和测试集
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)
# print(len(x_train), len(x_test), len(y_train), len(y_test))
# print(x_train, x_test, y_train, y_test)
# print('-'*88)
# 3.3 标准化处理，数据极相差不大，实际可以不做
from sklearn.preprocessing import StandardScaler

transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 4. 模型训练
# 4.1 创建模型
from sklearn.linear_model import LogisticRegression

estimator = LogisticRegression()

# 4.2 模型训练
estimator.fit(x_train, y_train)
# 5. 模型预测
y_predict = estimator.predict(x_test)
print("预测值是:\n", y_predict)
print("实际值是:\n", y_test)
# 6. 模型评估
# 6.1 准确率
# print("准确率是:\n", estimator.score(x_test, y_test))
# print("准确率是:\n", accuracy_score(y_test, y_predict))

# 至此，逻辑回归入门代码写完，但是
