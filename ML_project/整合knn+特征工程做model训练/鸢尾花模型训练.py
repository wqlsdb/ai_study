from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# 1. 查看鸢尾花数据集.
def dm01_load_iris():
    # 1.加载数据集
    iris_data = load_iris()
    # 2. 数据预处理
    # ，由于数据为官方提供，不需要做空值和非法制处理，
    # 只用切分训练集和测试集即可，比例为 8:2
    x_train, x_test, y_train, y_test = train_test_split(
        iris_data.data, iris_data.target, test_size=0.2, random_state=22)
    # 3. 特征工程
    # 3.1 特征提取，官方对数据已做处理
    # 3.2 特征预处理（标准化，归一化），通过数据发现4列差值不大，可以不做预处理
    # 3.2.1 创建标准化对象，
    transfer = StandardScaler()
    # 3.3 标准化操作，对训练集特征做：训练（拟合）及转换操作
    x_train = transfer.fit_transform(x_train)
    # 3.4 对测试集的特征做转换操作
    # transform():基于训练集的特征，调用内置好的模型（基于上一步已经拟合后的结果）调整参数。进行转换
    x_test = transfer.transform(x_test)
    # 3.5 特征降维，特征选取，特征组合，数据较为简单，这里不需要

    # 4.模型训练
    # 4.1 创建模型对象，KNN算法的分类模型对象
    estimator = KNeighborsClassifier(n_neighbors=5)
    # 4.2 具体的训练模型的动作
    estimator.fit(x_train, y_train)

    # 5.模型预测
    # todo:场景1: 对上述切分后的数据集做预测
    y_predict = estimator.predict(x_test)
    print(f'模型预测的结果为：{y_predict}')
    print(f'测试集真实标签:{y_test}')
    print('-' * 88)
    # todo:场景2：准备新的数据充当测试集的特征
    # step1:准备新的数据集
    x_test_new = [[3.6, 1.5, 2.2, 0.3]]
    # step2:特征工程,对上述测试集做标准化处理
    # 创建标准化对象
    x_test_new = transfer.transform(x_test_new)
    # step3:模型预测
    # todo:模型预测
    y_predict_new = estimator.predict(x_test_new)
    # todo:模型预测概率结果
    y_predict_proba_new = estimator.predict_proba(x_test_new)
    print(f'模型预测的结果为：{y_predict_new}')
    print(f'模型预测概率结果为：{y_predict_proba_new}')
    print('-' * 88)

    # 6.模型评估
    # KNN算法，模型评估主要参考 准确率，即：预测正确的数量 / 总的数量
    # 方式1: 针对于 测试集的特征 和 测试集的标签, 进行预测.  模型预测前后 都可以评估.
    print(f'模型的准确率：{estimator.score(x_test, y_test)}')

    # 方式2: 针对于 预测值(y_predict) 和 测试集的真实标签(y_test) 做评估.   模型预测后 评估.
    print(f'模型的准确率: {accuracy_score(y_test, y_predict)}')  # 0.93
    # print(f'模拟数据模型的准确率: {accuracy_score(y_test, y_predict_new)}') # 0.93


if __name__ == '__main__':
    dm01_load_iris()
