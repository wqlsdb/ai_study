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

归一化介绍:
    概述:
        它是特征预处理的一种方案, 采用的是: sklearn.preprocessing.MinMaxScaler 类.
    计算公式:
        x'  = (x - min) / (max - min)
        x'' = x' * (mx - mi) * mi
    公式解释:
        x   ->      特征列中, 某个具体要计算的值.
        min ->      该特征列的最小值.
        max ->      该特征列的最大值.
        mx  ->      区间的最大值, 默认的区间是 [0, 1], 即: mx = 1
        mi  ->      区间的最小值, 默认的区间是 [0, 1], 即: mi = 0
    弊端:
        即使该特征列的数据再多, 也只会受到该列的最大值, 最小值的影响,
        如果最大值或者最小值又恰巧是异常值, 就会导致 计算结果偏差, 鲁棒性较差.
"""

# 导包.
from sklearn.preprocessing import MinMaxScaler

# 1. 创建数据集.
x_train = [[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]]

# 2. 创建归一化对象.
# transfer = MinMaxScaler(feature_range=(0, 1))
transfer = MinMaxScaler(feature_range=(3, 5))       # 效果同上, 默认的区间是 [0, 1]

# 3. 具体的归一化操作.
# fit_transform(): 针对于训练集的, 即: 训练 + 转换(归一化)
# fit(): 针对于测试集的, 即: 只有转换(归一化)
x_train_new = transfer.fit_transform(x_train)   # 训练集: x_train, 返回 归一化后的数据
# x_train_new = transfer.fit(x_train)               # 测试集: x_test, 返回 归一化对象

# 4. 打印归一化的记过.
print(x_train_new)