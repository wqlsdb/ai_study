import sys

import pandas as pd
from sklearn.metrics import confusion_matrix

# 1.定义数据集
y_train = ['恶性', '恶性', '恶性', '恶性', '恶性', '恶性', '良性', '良性', '良性', '良性']
# 2.定义标签名
label = ['恶性', '良性']
df_label = ['恶性(正例)', '良性(反例)']

# 3. 定义预测结果
y_pre_A = ['恶性', '恶性', '恶性', '良性', '良性', '良性', '良性', '良性', '良性', '良性']
# 4. 把上述的 预测结果A 转换成 混淆矩阵.
# 参1: 真实样本, 参2: 预测样本, 参3: 样本标签(正例, 反例)
cm_A = confusion_matrix(y_train, y_pre_A, labels=label)
print(f'混淆矩阵A: \n {cm_A}')

# 5. 把混淆矩阵 转换成 DataFrame
df_A = pd.DataFrame(cm_A, index=df_label, columns=df_label)
print(f'预测结果A对应的DataFrame对象: \n {df_A}')
print('-' * 88)

if __name__ == '__main__':
    import os
    print(sys.executable)
