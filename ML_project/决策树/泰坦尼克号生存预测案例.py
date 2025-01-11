import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# 1.读取数据
titanic_df = pd.read_csv('./data/train.csv')
# 2.数据预测处理
x = titanic_df[['Pclass', 'Sex', 'Age']]
y = titanic_df[['Survived']]
# 2.2 用Age平均值，填充空值
x['Age'] = x['Age'].fillna(x['Age'].mean())
# 2.3 对Sex进行独热编码,机器学习的模型不支持字符串，需要转换为数字
x = pd.get_dummies(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)
# 3.特征工程（预处理）
# 4.模型训练
estimator = DecisionTreeClassifier()
estimator.fit(x_train, y_train)
# 5.模型预测
p_predict = estimator.predict(x_test)
print(f'预测结果为：\n{p_predict}')
# 6.模型评估
# 7.绘图
# 7.1 设置布局大小
plt.figure(figsize=(90, 80))
# 7.2 绘制树形图
plot_tree(estimator, filled=True, max_depth=10)
plt.savefig('./data/demo3.png')
plt.show()