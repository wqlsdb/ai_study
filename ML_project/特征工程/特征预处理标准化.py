from sklearn.preprocessing import StandardScaler
# 1.创建数据集
x_train = [[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]]

# 2.创建标准化对象。
transfer = StandardScaler()
# 3.具体标准化操作
x_train_new = transfer.fit_transform(x_train)

# 4.打印，标准化后的特征结果
print(x_train_new)

# 5.打印特征列的 平均值 和 标准差
# print(transfer.mean_)

# print(transfer.var_)
