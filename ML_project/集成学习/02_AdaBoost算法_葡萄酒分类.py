"""
案例:
    演示葡萄酒分类案例, 即: 用于演示AdaBoost算法.

AdaBoost算法原理:
    1. 它属于 Boosting算法(提升算法), 会训练n课决策树.
    2. 数据集(全集)会全部交给第1个模型(弱学习器)来训练, 正确值 -> 权重下降, 错误值 -> 权重上升.
    3. 把上个弱学习器的处理结果, 交个第2个模型(弱学习器)来训练, 正确值 -> 权重下降, 错误值 -> 权重上升.
    4. 重复该操作, 串行化执行, 加权投票, 直至获取最终结果.
"""

# 导包
# AdaBoost实战葡萄酒数据
import pandas as pd
from sklearn.preprocessing import LabelEncoder          # 数据预处理, 标签编码器
from sklearn.model_selection import train_test_split    # 训练集测试集划分
from sklearn.tree import DecisionTreeClassifier         # 决策树分类器
from sklearn.ensemble import AdaBoostClassifier         # AdaBoost分类器, 机器学习算法
from sklearn.metrics import accuracy_score              # 模型评估, 准确率


# 1. 读取数据.
df_wine = pd.read_csv('./data/wine0501.csv')

# 2. 数据预处理.
# 2.1 删除类别为1的样本,从中过滤出2, 3类别, 即: Adaboost算法更适合做 二分法(二分类).
df_wine = df_wine[df_wine['Class label'] != 1]

# 2.1 提取特征 和 标签.
x = df_wine[['Alcohol', 'Hue']].values
y = df_wine['Class label'].values       # [2, 3]

# 2.2 打印结果.
# print(x)
# print(y)

# 2.3 标签编码器, 把标签值改为: [0, 1]
le = LabelEncoder()
y = le.fit_transform(y)     # [2, 3] => [0, 1]
# print(y)

# 2.4 训练集测试集划分.
x_train, x_test, y_train, y_test = train_test_split(x, y,
test_size=0.2, random_state=22)

# 3. 模型训练.
# 3.1 测试 AdaBoostClassifier类, 默认用的是什么树.
ada = AdaBoostClassifier()
# 模型训练
ada.fit(x_train, y_train)
# 模型预测
y_pred_ada = ada.predict(x_test)
# 模型评估.
print(f'AdaBoostClassifier的准确率: {accuracy_score(y_test, y_pred_ada)}')  # 0.875


# 3.2 手动创建 决策树对象, 将其传入 AdaBoostClassifier类, 获取预测结果.
# 创建 决策树对象, 参1: 熵, 参2: 树的深度.
my_tree = DecisionTreeClassifier(criterion='entropy', max_depth=1)

# 创建 决策树对象, 参1: 指定为基尼值, 参2: 树的深度.
# my_tree = DecisionTreeClassifier(criterion='gini', max_depth=1)

# 创建 AdaBoostClassifier类对象, 参1: 决策树对象, 参2: 迭代次数, 参3: 学习率, 参4: 随机种子.
my_ada = AdaBoostClassifier(estimator=my_tree, n_estimators=500, learning_rate=0.1, random_state=22)

# 把上述的 决策树对象, 传入 AdaBoostClassifier类, 获取预测结果.
# 模型训练
my_ada.fit(x_train, y_train)
# 模型预测
y_pred_my_ada = my_ada.predict(x_test)
# 模型评估.
print(f'手写AdaBoostClassifier的准确率: {accuracy_score(y_test, y_pred_my_ada)}') # 0.9166


