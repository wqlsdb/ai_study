# AdaBoost实战葡萄酒数据

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier  # 集成学习
from sklearn.metrics import accuracy_score
# 1. 读取数据.
df_wine = pd.read_csv('./data/wine0501.csv')

# 2 数据预处理
# 2.1 删除类别为1的样本，剩余的样本为2和3
y = df_wine[df_wine['Class label'] != 1]
# 2.1 提取特征 和 标签.
x = df_wine[['Alcohol', 'Hue']].values
y = df_wine['Class label'].values
# print(y)
# 2.3 标签编码器
le = LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)
# print(y)
# 3. 模型训练
# 3.1 测试 AdaB
ada = AdaBoostClassifier()
# 模型训练
ada.fit(x_train, y_train)
# 模型预测
y_pred_ada = ada.predict(x_test)
# 模型评估
print(f'AdaBoost准确率: {accuracy_score(y_test, y_pred_ada)}')

my_tree = DecisionTreeClassifier(criterion='entropy', max_depth=1)

my_ada = AdaBoostClassifier(estimator=my_tree,n_estimators=500,learning_rate=0.1,random_state=22)

my_ada.fit(x_train, y_train)
y_pred_my_ada = my_ada.predict(x_test)
print(f'自定义AdaBoost准确率: {accuracy_score(y_test, y_pred_my_ada)}')

