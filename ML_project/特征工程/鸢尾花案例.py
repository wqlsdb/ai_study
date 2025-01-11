# 导入类库
# 导入工具包
from sklearn.datasets import load_iris  # 加载鸢尾花测试集的.
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # 分割训练集和测试集的
from sklearn.preprocessing import StandardScaler  # 数据标准化的
from sklearn.neighbors import KNeighborsClassifier  # KNN算法 分类对象
from sklearn.metrics import accuracy_score  # 模型评估的, 计算模型预测的准确率
from pylab import mpl

# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]


def dm01_load_irise():
    # 1.加载数据集（150条样本数据）{'data':array([[5.1, 3.5, 1.4, 0.2],[4.9, 3. , 1.4, 0.2],。150条。])，'target': array(0,1,2..均摊150条)}
    iris_data = load_iris()
    # 2. 答应iris_data对象，字典
    # print(iris_data)
    print('_' * 33)
    # 3.查看数据集，即特征列，分别是
    # print(f'打印前五条样例：\n{iris_data.data[:5]}')
    print('_' * 33)
    # 4.查看数据集，即：标签列
    print(f'打印target标签列数据:\n {iris_data.target[:5]}')
    print('_' * 33)

    # 5.查看特征列，标签列，即：数据集的条数
    print(f'特征列数据：{len(iris_data.data)},标签列数据：{len(iris_data.target)}')
    print('_' * 33)

    # 6.查看数据集的列名，即特征列的列名
    # print(iris_data.feature_names)
    print('_' * 33)

    # 7.查看各分类标签 对应的 具体化的标签名
    # print(iris_data.target_names)
    print('_' * 33)

    # 8. 查看iris数据集 所有的 键 的名字，即 他有哪些原始额数据
    # print(iris_data.keys())


def dm02_iris_visualization():
    # 1.加载数据集
    iris_data = load_iris()
    # 3.把上述数据集封装成dataframe
    iris_df = pd.DataFrame(
        data=iris_data.data,
        columns=iris_data.feature_names
    )
    print('打印DataFrame的前五条')
    print(iris_df.head(5))

    # 4.新增1列，充当标签列
    iris_df['label'] = iris_data.target

    # 5.查看数据集
    # print(iris_data)

    # 6.具体可视化动作 x=花瓣长度, y=花瓣宽度,data=iris的df对象，hue：颜色区分，fit_reg=Fase不绘制拟合回归线
    sns.lmplot(data=iris_df, x='petal length (cm)', y='petal width (cm)', hue='label', fit_reg=True)
    plt.title('iris data')
    plt.tight_layout()
    plt.show()

    # 3. 定义函数 dm03_train_test_split(), 实现: 数据集划分
    def dm03_train_test_split():
        # 1. 加载数据集, 查看数据
        iris_data = load_iris()
        # 2. 划分数据集, 即: 特征工程(预处理-标准化)
        pass

    # 4. 定义函数 dm04_模型训练和预测(), 实现: 模型训练和预测
def dm04_model_train_and_predict():
    # 1. 加载数据集, 查看数据
    iris_data = load_iris()

    # 2. 划分数据集, 即: 数据基本处理
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2,
                                                        random_state=22)

    # 3. 特征工程
    # 3.1 创建标准化对象
    transfer = StandardScaler()
    # 3.2 对训练集的特征做：训练 及 转换的操作
    x_train = transfer.fit_transform(x_train)
    # print(f'---------------{x_train}')
    # 3.3 对测试集特征：转换操作
    x_test = transfer.transform(x_test)
    print(f'---------------{x_test}')

    # 4. 模型训练
    # 4.1 创建模型对象，KNN算法的分类对象
    estimator = KNeighborsClassifier(n_neighbors=3)
    # 4.2 具体的训练模型的动作
    estimator.fit(x_train, y_train)

    # 5. 模型预测
    # 场景1：
    y_predict = estimator.predict(x_test)
    print(f'模型的预测结果为: {y_predict}')
    print(f'测试集真实标签: {y_test}')

    # 场景2：对新的数据预测
    # step 1:准备新数据集
    x_test_new = [[3.6, 1.5, 2.2, 0.3]]

    # step 2:对上述测试集做标准化处理
    x_test_new = transfer.transform((x_test_new))
    # step 3 模型预测
    y_predict_new = estimator.predict(x_test_new)
    y_predict_proba_new = estimator.predict_proba(x_test_new)
    print(f'模型预测的结果为：{y_predict_new}')
    # step 4 查看上述特征 各结果标签的 概率值
    print(f'模型预测概率结果为：{y_predict_proba_new}')
    print('-' * 30)

    # 6. 模型评估.
    # KNN算法, 模型评估主要参考 准确率, 即: 预测正确的数量 / 总的数量.
    # 方式1: 针对于 测试集的特征 和 测试集的标签, 进行预测.  模型预测前后 都可以评估.
    print(f'模型的准确率: {estimator.score(x_test, y_test)}')  # 0.93

    # 方式2: 针对于 预测值(y_predict) 和 测试集的真实标签(y_test) 做评估.   模型预测后 评估.
    print(f'模型的准确率: {accuracy_score(y_test, y_predict)}') # 0.93
if __name__ == '__main__':
    dm04_model_train_and_predict()
# dm01_load_irise()
# dm02_iris_visualization()

