from sklearn.preprocessing import LabelEncoder
# 创建 LabelEncoder 对象
le = LabelEncoder()

# 样本数据
departments = ['Sales', 'Marketing', 'Sales', 'Research & Development', 'Marketing']

# 将类别标签转换为数字标签
encoded_departments = le.fit_transform(departments)

# 输出转换后的结果
print(encoded_departments)
print(le.classes_)

