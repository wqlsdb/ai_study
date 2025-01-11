from sklearn.preprocessing import MinMaxScaler
# 1.创建数据集
x_train = [[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]]

# 2.创建归一化对象。
# transfer = MinMaxScaler(feature_range=(0,1))
transfer = MinMaxScaler()
# 3.具体归一化操作
x_train_new = transfer.fit_transform(x_train)

# 4.打印，归一化后的特征结果
print(x_train_new)
