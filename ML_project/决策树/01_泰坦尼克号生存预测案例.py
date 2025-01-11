"""
案例:
    泰坦尼克号生存预测, 基于: 船舱等级, 性别, 年龄做预测, 是否: Died, Survivor
目的:
    演示决策树的分层.
"""

# 导包
import pandas as pd                                     # 数据处理
from sklearn.model_selection import train_test_split    # 训练集和测试集分割
from sklearn.preprocessing import StandardScaler        # 标准化
from sklearn.tree import DecisionTreeClassifier         # 决策树分类器
from sklearn.metrics import classification_report       # 分类报告
import matplotlib.pyplot as plt                         # 绘图
from sklearn.tree import plot_tree                      # 绘制决策树

# 1. 读取数据.
titanic_df = pd.read_csv("./data/train.csv")
# titanic_df.info()

# 2. 数据预测处理.
# 2.1 获取特征 和 标签.
x = titanic_df[["Pclass", "Sex", "Age"]]    # 船舱等级, 性别, 年龄.
y = titanic_df["Survived"]  # 是否: Died, Survivor.
# 2.2 用Age列的平均值, 来填充空值.
# x["Age"].fillna(x["Age"].mean(), inplace=True)    # 已过时, 会报错.
x['Age'] = x['Age'].fillna(x['Age'].mean())

# 2.3 查看数据集.
# print(len(x), len(y))
# x.info()
# y.info()
#
# print(x.head(5))
# print(y.head(5))

# 2.4 对Sex列做热编码处理.
x = pd.get_dummies(x)

# 2.5 查看预处理后的结果.
# print(x.head(5))
# print(y.head(5))
# x.info()
# y.info()

# 2.6 拆分训练集和测试集.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)

# 3. 特征工程(标准化), 这里不需要.
# transfer = StandardScaler()
# x_train = transfer.fit_transform(x_train)
# x_test = transfer.transform(x_test)

# 4. 模型训练.
estimator = DecisionTreeClassifier()
estimator.fit(x_train, y_train)

# 5. 模型预测.
y_predict = estimator.predict(x_test)
print("预测结果为:\n", y_predict)

# 6. 模型评估.
print("准确率: \n", estimator.score(x_test, y_test))
print(f'分类评估报告: \n {classification_report(y_test, y_predict, target_names=["Died", "Survivor"])}')

# 7. 绘图.
# 7.1 设置画布大小.
plt.figure(figsize=(50, 30))
# 7.2 绘制决策树.
# 参1: estimator: 决策树分类器,  参2: filled: 填充颜色,  参3: max_depth: 层数.
plot_tree(estimator, filled=True, max_depth=10)
# 保存图片.
plt.savefig("./data/titanic_tree.png")
# 7.3 具体的绘制.
plt.show()