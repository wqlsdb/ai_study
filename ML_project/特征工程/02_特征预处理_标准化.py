"""
特征预处理解释:
    背景:
        实际开发中, 如果多个特征列因为单位(量纲)问题, 导致数值的差距过大, 则会导致 模型预测值偏差.
        例如:
            身高 -> 单位: 米
            体重 -> 单位: KG(公斤)
        为了保证每个特征列 对最终预测结果的 权重比都是相近的, 所以要进行特征预处理的操作.
    实现方式:
        归一化.
        标准化.

标准化介绍:
    概述:
        它是特征预处理的一种方案, 采用的是: sklearn.preprocessing.StandardScaler 类.
    计算公式:
        x'  = (x - mean) / 该特征列值的标准差
    公式解释:
        x    ->      特征列中, 某个具体要计算的值.
        mean ->      该特征列的平均值
    应用场景:
        比较适合大数据集的应用场景, 当数据量比较大的情况下, 受到最大值, 最小值的影响也会变得微乎其微.
        实际开发中, 一般用 标准化处理.
    总结:
        无论是归一化, 还是标准化, 目的都是为了避免因 特征列的量纲(单位)问题导致权重不同, 从而影响最终的预测结果.
        目的都是一样的, 数据集小可以用归一化, 数据集大可以用 标准化.
"""

# 导包.
from sklearn.preprocessing import StandardScaler

# 1. 创建数据集.
x_train = [[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]]

# 2. 创建标准化对象.
transfer = StandardScaler()

# 3. 具体的标准化操作.
# todo:fit_transform(): 针对于训练集的, 即: 训练 + 转换(标准化)
# todo:fit(): 针对于测试集的, 即: 只有转换(标准化)
x_train_new = transfer.fit_transform(x_train)   # 训练集: x_train, 返回 标准化后的数据
# x_train_new = transfer.fit(x_train)           # 测试集: x_test, 返回 标准化对象

# 4. 打印标准化的结果.
print(x_train_new)

# 5. 打印特征列的 平均值 和 标准差.
print(transfer.mean_)       # 平均值
print(transfer.var_)        # 标准差