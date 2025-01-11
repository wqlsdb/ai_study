# 1. 数据读取
import pandas as pd

data = pd.read_csv('./data/train.csv')
# 2. 数据预处理
# 2.1 抽取特征列 和 标签列\
# copy():创建一个新的数据集，拷贝一份原始的数据集
x = data[['Pclass', 'Age', 'Sex']].copy()
y = data['Survived']
# 2.2 缺失值处理
x['Age'] = x['Age'].fillna(x['Age'].mean())
# 2.3 针对性别，进行热编码处理
x = pd.get_dummies(x)
# 2.4 划分训练集和测试集
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test \
    = train_test_split(x, y, test_size=0.2, random_state=22)
# 3. 模型训练 -> 单一决策树
# 创建模型对象
from sklearn.tree import DecisionTreeClassifier

moudel1 = DecisionTreeClassifier()
moudel1.fit(x_train, y_train)
# 模型评估
print(f'单决策树模型预测结果：{moudel1.score(x_test, y_test)}')  # 0.7821
# 4.模型训练 -> 随机森林训练
# 创建模型对象
from sklearn.ensemble import RandomForestClassifier

moudel2 = RandomForestClassifier()
moudel2.fit(x_train, y_train)
print(f'随机森林模型预测结果：{moudel2.score(x_test, y_test)}')
# 5. 模型训练-> 随机森林 + 网格搜索 + 交叉验证
# 创建模型对象
moudle3 = RandomForestClassifier()
# 创建参数字典，定义模型可选的 超参数的值
param_dict = {'n_estimators': [50, 60, 70, 80, 90, 200], 'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
# 网格搜索 + 交叉验证，获取最优组合的参数
from sklearn.model_selection import GridSearchCV

# gs_model = GridSearchCV(moudle3, param_grid=param_dict, cv=5)
gs_model = GridSearchCV(moudle3, param_grid=param_dict, cv=6)
# 模型训练
gs_model.fit(x_train, y_train)
# 模型评估
print('网格搜索 + 交叉验证后的模型预测结果：', gs_model.score(x_test, y_test))

# 7. 查看最优组合
print('最优参数组合：', gs_model.best_params_)
