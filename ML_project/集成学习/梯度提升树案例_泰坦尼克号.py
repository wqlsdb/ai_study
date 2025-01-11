import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('./data/train.csv')

x = data[['Pclass', 'Sex', 'Age']].copy()
y = data[['Survived']].copy()
# 2.2 缺失特征处理
x['Age'] = x['Age'].fillna(x['Age'].mean())
# 2.3.
x = pd.get_dummies(x)
# 2.4
# 2.5 划分训练集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)
# 3. 构建模型
# 3.1 构建GBDT模型（梯度提升树）
model = GradientBoostingClassifier()
# 模型训练
model.fit(x_train, y_train)
# 模型预测
y_predict = model.predict(x_test)
# 模型评分
print(f'GBDT（梯度提升树）正确率:{model.score(x_test, y_test)}')
print(f'GBDT（梯度提升树预测结果为：{accuracy_score(y_test, y_predict)}')

# 4. 模型调优
gbc = GradientBoostingClassifier()

param_dict = {'n_estimators': [70, 80, 90, 100, 130, 150, 200], 'max_depth': [2, 3, 4, 5, 6], 'random_state': [22]}
gbc_model = GridSearchCV(gbc, param_dict, cv=5)