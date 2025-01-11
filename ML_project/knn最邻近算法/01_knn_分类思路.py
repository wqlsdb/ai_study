"""
KNN(K 近邻算法, 全拼是: K Nearest Neighbors):
    概述:
        大白话解释: 近朱者赤, 近墨者黑.
        专业话述: 找离测试集最近的哪个K样本, 然后投票, 哪个标签值多, 就用它作为 测试集的 最终结果(标签).
    KNN算法 实现思路:
        思路1: 分类思路.      投票, 选最多的.
        思路2: 回归思路.      求均值.
    KNN算法 分类思路 原理如下:
        1. 基于欧氏距离计算 每个训练集 离 测试集的 距离.
            欧式距离: 对应维度差值的平方和, 开平方根.
        2. 按照距离值, 进行升序排列, 找到距离最小的那 K个样本值.
        3. 分类思路: 投票选取, 票数最多的哪个标签值 -> 作为 测试集的 标签.
            如果标签的票数一致, 参考: 最简单的模型(即: 最近距离的哪个标签结果), 奥卡姆剃刀原理.
    KNN算法 代码实现步骤:
        1. 导包.
            确保你已经安装了机器学习的库, 没装就装一下.   pip install scikit-learn
        2. 创建模型(算法)对象.
        3. 准备训练集(x_train, y_train)
        4. 准备测试集(x_test)
        5. 模型训练.
        6. 模型预测, 并打印预测结果.
"""

from sklearn.neighbors import KNeighborsClassifier      # KNN算法 -> 分类对象.

# 1. 创建模型(算法)对象.
# estimator单词的意思是: 评估, 预测, 评估器.
# Neighbor单词的意思是: 邻居.
# n_neighbors参数的意思是: (最近的)邻居的数量, 默认是: 5
# estimator = KNeighborsClassifier(n_neighbors=2)
estimator = KNeighborsClassifier(n_neighbors=3)

# 2. 准备训练集(x_train, y_train)
x_train = [[1], [2], [3], [4]]
# y_train = [0, 0, 0, 1]      # 分类: 二分法
y_train = [0, 0, 1, 0]      # 分类: 二分法

# 3. 准备测试集(x_test)
# x_test = [[5]]

# 4. 模型训练.
# 参1: 训练集的特征, 参2: 训练集的标签.
estimator.fit(x_train, y_train)

# 5. 模型预测, 并打印预测结果.
# 传入 测试集的特征, 基于模型, 获取 测试集的 预测标签.
# y_test = estimator.predict(x_test)

y_test = estimator.predict([[5]])
# todo 预测结果是根据标签列和定义的K值有关系，
'''
    例1：测试值5，选取的k=2,标签列y_train = [0, 0, 1, 0]
    根据knn算法分类思路：选取离测试集最近的k个样本，[1,0]，由于票数一致
    参考: 最简单的模型(即: 最近距离的哪个标签结果), 奥卡姆剃刀原理.

    例2：测试值5，选取的k=3,标签列y_train = [0, 0, 1, 0]
    根据knn算法分类思路：选取离测试集最近的k个样本，[1,0]，票数:0的出现两次，1的出现一次，票数多的胜
'''
# 打印结果.
print(f'预测结果是: {y_test}')   # 1