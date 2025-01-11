# -*- coding: UTF-8 -*-
import numpy as np
import operator
import collections

"""
函数说明:创建数据集

Parameters:
	无
Returns:
	group - 数据集
	labels - 分类标签
Modify:
	2017-07-13
"""


def createDataSet():
    # 四组二维特征
    group = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])
    # 四组特征的标签
    labels = ['爱情片', '爱情片', '动作片', '动作片']
    return group, labels


"""
函数说明:kNN算法,分类器

Parameters:
	inX - 用于分类的数据(测试集)
	dataSet - 用于训练的数据(训练集)
	labes - 分类标签
	k - kNN算法参数,选择距离最小的k个点
Returns:
	sortedClassCount[0][0] - 分类结果

Modify:
	2017-11-09 by Cugtyt 
		* GitHub(https://github.com/Cugtyt) 
		* Email(cugtyt@qq.com)
		Use list comprehension and Counter to simplify code
	2017-07-13
"""


def classify0_bak(inx, dataset, labels, k):
    # 计算距离
    dist = np.sum((inx - dataset) ** 2, axis=1) ** 0.5
    # k个最近的标签
    k_labels = [labels[index] for index in dist.argsort()[0: k]]
    # 出现次数最多的标签即为最终类别
    label = collections.Counter(k_labels).most_common(1)[0][0]
    return label


def classify0(inx, dataset, labels, k):
    # 计算输入样本与数据集中每个样本的距离
    # (inx - dataset) 计算的是输入样本与数据集每个样本之间的差值矩阵
    # (inx - dataset) ** 2 对上述差值矩阵进行平方运算
    # np.sum(..., axis=1) 按行求和，得到一个一维数组，表示输入样本与每个样本的距离平方和
    # ** 0.5 对距离平方和开方，得到欧氏距离
    dist = np.sum((inx - dataset) ** 2, axis=1) ** 0.5

    # 根据计算出的距离对索引进行排序
    # argsort() 返回的是按照从小到大排序后的索引数组
    # [0: k] 取出前k个最小距离对应的索引
    sorted_indices = dist.argsort()[0: k]

    # 获取这k个最近邻居的标签
    # 使用列表推导式根据索引从labels中取出对应的标签
    k_labels = [labels[index] for index in sorted_indices]

    # 统计这k个标签中出现次数最多的标签
    # collections.Counter(k_labels) 创建一个Counter对象，统计每个标签出现的次数
    # most_common(1) 返回出现次数最多的标签及其出现次数，返回的是一个列表，其中每个元素是一个元组
    # [0][0] 取出出现次数最多的标签
    label = collections.Counter(k_labels).most_common(1)[0][0]

    # 返回最终的分类结果
    return label


if __name__ == '__main__':
    # 创建数据集
    group, labels = createDataSet()
    # 测试集
    test = [101, 20]
    # kNN分类
    test_class = classify0(test, group, labels, 3)
    # 打印分类结果
    print(test_class)
