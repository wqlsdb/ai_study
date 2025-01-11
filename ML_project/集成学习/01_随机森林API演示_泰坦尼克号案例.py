"""
案例:
    通过泰坦尼克号数据集, 基于特征列: Pclass(船舱等级), Age(年龄), Sex(性别, 需要热编码处理)来预测 是否生存(Survivor).

目的:
    演示随机森林算法的API, 即: RandomForestClassifier,
    随机森林算法属于 -> Bagging 算法, 即: 集成学习算法.
原理：
    有放回采样，平权投片
    每个弱学习机器是并行的没有依赖关系
"""

# 导包.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# 1. 加载数据.
data = pd.read_csv('./data/train.csv')

# 2. 数据预处理.
# 2.1 抽取特征列 和 标签列.
# copy(): 创建一个新数据集, 拷贝一份原始的数据集.
x = data[['Pclass', 'Age', 'Sex']].copy()
y = data['Survived']
# 2.2 缺失值处理.
x['Age'] = x['Age'].fillna(x['Age'].mean())
# 2.3 针对于性别, 进行热编码处理.
x = pd.get_dummies(x)
# 2.4 查看数据集.
# print(len(x), len(y))
# print(x)
# print(y)
# 2.5 划分训练集和测试集.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)

# 3. 模型训练 -> 单一决策树训练.
# 创建模型对象.
estimator1 = DecisionTreeClassifier()
# 模型训练.
estimator1.fit(x_train, y_train)
# 模型评估, 并打印.
print('单决策树模型预测结果:', estimator1.score(x_test, y_test))  # 0.7821

# 4. 模型训练 -> 随机森林训练.
# 创建模型对象.
estimator2 = RandomForestClassifier()
# 模型训练.
estimator2.fit(x_train, y_train)
# 模型评估, 并打印.
print('随机森林模型预测结果:', estimator2.score(x_test, y_test))  # 0.7598

# 5. 模型训练 -> 随机森林 + 网格搜索调优.
# 创建模型对象.
estimator3 = RandomForestClassifier()
# 创建参数字典, 定义模型可选的 超参数的值.
param_dict = {'n_estimators':[50, 60, 70, 80, 90, 200], 'max_depth':[2, 3, 4, 5, 6, 7, 8, 9, 10]}
# 网格搜索 + 交叉验证, 获取最优组合.
# gs_estimator: 网格搜索 + 交叉验证的模型对象 -> 优化后的模型对象.
gs_estimator = GridSearchCV(estimator3, param_grid=param_dict, cv=5)  # 3折交叉验证.
# 模型训练.
gs_estimator.fit(x_train, y_train)
# 模型评估, 并打印.
print('网格搜索 + 交叉验证后的模型预测结果:', gs_estimator.score(x_test, y_test))

# 7. 查看最优组合.
print('网格搜索 + 交叉验证后的模型最优参数组合:', gs_estimator.best_params_)