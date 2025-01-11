from sklearn.datasets import load_iris  # 加载鸢尾花测试集的.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def dm01_load_irise():
    # 1. 加载数据集.
    iris_data = load_iris()

    # 2. 打印 iris_data 对象, 发现是: 字典.
    # print(iris_data)

    # 3. 查看数据集, 即: 特征列, 分别是: sepal length (cm), sepal width (cm), petal length (cm), petal width (cm).
    # print(f'查看特征列前5条数据{iris_data.data[:5]}')

    # 4. 查看数据集, 即: 标签列.
    # print(f'查看标签列前5条数据:{iris_data.target[:5]}')

    # 5. 查看特征列, 标签列, 即: 数据集的条数
    # print(f'特征列数据：{len(iris_data.data)},标签列数据：{len(iris_data.target)}')
    # print('_' * 33)


    # 6. 查看数据集的列名, 即: 特征列的列名.
    # print(iris_data.feature_names)  # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

    # 7. 查看各分类标签 对应的 具体的标签名.
    # print(iris_data.keys())   # ['setosa', 'versicolor', 'virginica'], 分类编号分别是: 0, 1, 2

    # 8. 查看iris数据集, 所有的 键的名字, 即: 它有哪些原始的数据.
    # dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])

# 4. 鸢尾花案例, KNN模型, 完成测试.
def dm04_模型评估和预测():
    # 1. 准备数据.
    iris_data = load_iris()     # 150条数据, 目前是: 字典的形式.

    # 2. 数据预处理
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data,iris_data.target,test_size=0.2,random_state=22)
    print(x_train)
    # 3.特征工程
    # 3.1 创建标准化对象
    transfer = StandardScaler()
    # 3.2 对训练集的特征做：训练（拟合） 及转换操作
    x_train = transfer.fit_transform(x_train)
    # 3.3 对测试集的特征，做转换操作
    x_test = transfer.transform(x_test)

    # 4. 模型训练
    # 4.1 创建模型对象 KNN算法的 分类对象
    estimator = KNeighborsClassifier(n_neighbors=5)
    # 4.2 具体的训练模型的动作
    estimator.fit(x_train,y_train)

    # 5. 模型预测
    # 场景 1 对上述切分后的测试集预测
    y_predict = estimator.predict(x_test)
    print(f'模型预测结果为：{y_predict}')
    print(f'测试集真实标签: {y_test}')
    print('-'*32)

    # 场景2：对新的数据做预测
    # step1:准备新的数据集做 标准化处理
    x_test_new = [[3.5,2.8,1.3,0.35]]
    # step 2 对上述的测试集做 标准化处理
    x_test_new = transfer.transform(x_test_new)
    # step 3 模型预测
    y_predict_new = estimator.predict(x_test_new)
    y_predict_proba_new = estimator.predict_proba(x_test_new)
    print(f'模型预测结果为: {y_predict_new}')
    # step4: 查看上述的特征 -> 各结果标签的 概率值.
    print(f'模型预测概率结果为: {y_predict_proba_new}')  # [[0.2 0.8 0. ]]
    print('-' * 30)








if __name__ == '__main__':
    # dm01_load_irise()
    dm04_模型评估和预测()
    # dict1 = {'id': '001', 'name': '张三', 'age': '28'}
    # print(dict1.keys())