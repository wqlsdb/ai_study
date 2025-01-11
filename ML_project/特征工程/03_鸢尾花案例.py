"""
案例: 鸢尾花案例, 目的: 演示KNN算法.

回顾 机器学习项目的开发流程:
    1. 准备数据.
    2. 数据预处理.
    3. 特征工程.
        特征提取, 特征预处理(归一化, 标准化), 特征降维, 特征选取, 特征组合
    4. 模型训练.
    5. 模型预测.
    6. 模型评估.
"""

# 导入工具包
from sklearn.datasets import load_iris          # 加载鸢尾花测试集的.
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split    # 分割训练集和测试集的
from sklearn.preprocessing import StandardScaler        # 数据标准化的
from sklearn.neighbors import KNeighborsClassifier      # KNN算法 分类对象
from sklearn.metrics import accuracy_score              # 模型评估的, 计算模型预测的准确率
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 1. 查看鸢尾花数据集.
def dm01_load_iris():
    # 1. 加载数据集.
    iris_data = load_iris()

    # 2. 打印 iris_data 对象, 发现是: 字典.
    print(iris_data)

    # 3. 查看数据集, 即: 特征列, 分别是: sepal length (cm), sepal width (cm), petal length (cm), petal width (cm).
    print(iris_data.data[:5])

    # 4. 查看数据集, 即: 标签列.
    print(iris_data.target[:5])

    # 5. 查看特征列, 标签列, 即: 数据集的条数
    print(len(iris_data.data), len(iris_data.target))       # 150条

    # 6. 查看数据集的列名, 即: 特征列的列名.
    print(iris_data.feature_names)  # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

    # 7. 查看各分类标签 对应的 具体的标签名.
    print(iris_data.target_names)   # ['setosa', 'versicolor', 'virginica'], 分类编号分别是: 0, 1, 2

    # 8. 查看iris数据集, 所有的 键的名字, 即: 它有哪些原始的数据.
    # dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
    print(iris_data.keys())   #


# 2. 加载鸢尾花的数据集, 并可视化.
def dm02_iris_visualization():
    # 1. 加载数据集.
    iris_data = load_iris()

    # 2. 查看 特征名(充当列名), 数据列(充当列值), 标签列(新增列)
    # print(iris_data.feature_names)  # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    # print(iris_data.data[:5])
    # print(iris_data.target[:5])

    # 3. 把上述的数据集, 封装成DataFrame对象.
    iris_df = pd.DataFrame(
        data = iris_data.data,
        columns = iris_data.feature_names
    )
    # 4. 新增1列, 充当: 标签列.
    iris_df['label'] = iris_data.target
    # 5. 查看数据集.
    print(iris_df)

    # 6. 具体的可视化动作, 绘制散点图即可.
    # 参1: 要绘制的df对象, x: 花瓣的长度, y: 花瓣的宽度, hue: 标签分类, fit_reg: 是否绘制回归线.
    sns.lmplot(data=iris_df, x='petal length (cm)', y='petal width (cm)', hue='label', fit_reg=True)
    # 7. 设置标题 和 绘制图形.
    # plt.title('Iris DataSet')
    plt.title('鸢尾花数据集展示')
    # 设置紧密布局, 解决标题显示不全的问题.
    plt.tight_layout()
    plt.show()


# 3. 演示数据切分, 150条(总) -> 120条(训练集) + 30条(测试集).
def dm03_train_test_split():
    # 1. 加载数据集.
    iris_data = load_iris()

    # 2. 查看数据集.
    # print(iris_data.data)   # 特征
    # print(iris_data.target) # 标签

    # 3. 查看数据集的总数.
    # print(len(iris_data.data), len(iris_data.target))   # 150

    # 4. 切分训练集和测试集.
    # 参1: 要被切分的特征, 参2: 要被切分的标签, 参3: 测试集的比例(这里是: 20%)  参4: 随机种子, 种子一致, 每次获取的随机结果都是一样的.
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=22)

    # 5. 打印测试结果.
    print(f'训练集的特征: {x_train}')
    print(f'训练集的标签: {y_train}')
    print(f'训练集的数据条数: {len(x_train), len(y_train)}')

    print(f'测试集的特征: {x_test}')
    print(f'测试集的标签: {y_test}')
    print(f'测试集的数据条数: {len(x_test), len(y_test)}')


# 4. 鸢尾花案例, KNN模型, 完成测试.
def dm04_模型评估和预测():
    # 1. 准备数据.
    iris_data = load_iris()     # 150条数据, 目前是: 字典的形式.

    # 2. 数据预处理.
    # 这里不需要做空值, 非法值的处理, 只要切分训练集和测试集即可, 这里比例是 8:2
    # x_train:训练集（特征数据）x_test:测试集,y_train(训练集标签数据)，y_test:测试集(标签数据)
    x_train, x_test, y_train, y_test = train_test_split(
        iris_data.data, iris_data.target,
        test_size=0.2, random_state=22)

    # 3. 特征工程.
    # 特征提取: 无需手动实现, 数据已经提取完毕后, 分别是: 花萼的长度, 花萼的宽度, 花瓣的长度, 花瓣的宽度.
    # 字段名如下: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

    # 特征预处理(归一化, 标准化), 我们发现特征(共4列)差值都不大, 所以可以不做特征预处理.
    # 但是加上这个步骤以后, 会让我们的代码更加的完整, 所以我给大家加上这个操作.
    # 3.1 创建标准化对象.
    transfer = StandardScaler()
    # 3.2 对训练集的特征做: 训练(拟合) 及 转换的操作.
    # fit_transform(): 先基于训练集的特征进行拟合, 例如: 获取均值, 标准差, 方差等信息, 在基于内置好的模型(标准化对象), 调整参数, 进行转换.
    x_train = transfer.fit_transform(x_train)
    # 3.3 对测试集的特征做:  转换的操作.
    # transform(): 基于训练集的特征, 调用内置好的模型(标准化对象), (基于上一步已经拟合后的结果)调整参数, 进行转换.
    x_test = transfer.transform(x_test)
    # 特征降维, 特征选取, 特征组合, 这里不需要该操作, 因为数据比较简单.

    # 4. 模型训练.
    # 4.1 创建模型对象, KNN算法的 分类对象.
    estimator = KNeighborsClassifier(n_neighbors=5)
    # 4.2 具体的训练模型的动作.
    estimator.fit(x_train, y_train)

    # 5. 模型预测.
    # 场景1: 对上述的(切分后的)测试集做预测.
    y_predict = estimator.predict(x_test)
    print(f'模型预测结果为: {y_predict}')
    print(f'测试集真实标签: {y_test}')

    # 场景2: 对新的数据集做预测.
    # step1: 准备新的数据集, 充当: 测试集的 特征.
    x_test_new = [[3.6, 1.5, 2.2, 0.3]]
    # step2: 对上述的测试集做 标准化处理.
    x_test_new = transfer.transform(x_test_new)
    # step3: 模型预测.
    y_predict_new = estimator.predict(x_test_new)
    y_predict_proba_new = estimator.predict_proba(x_test_new)
    print(f'模型预测结果为: {y_predict_new}')  # 0, 1, 2
    # step4: 查看上述的特征 -> 各结果标签的 概率值.
    print(f'模型预测概率结果为: {y_predict_proba_new}')  # [[0.2 0.8 0. ]]
    print('-' * 30)

    # 6. 模型评估.
    # KNN算法, 模型评估主要参考 准确率, 即: 预测正确的数量 / 总的数量.
    # 方式1: 针对于 测试集的特征 和 测试集的标签, 进行预测.  模型预测前后 都可以评估.
    print(f'模型的准确率: {estimator.score(x_test, y_test)}')  # 0.93

    # 方式2: 针对于 预测值(y_predict) 和 测试集的真实标签(y_test) 做评估.   模型预测后 评估.
    print(f'模型的准确率: {accuracy_score(y_test, y_predict)}') # 0.93


# 5. 在main函数中测试.
if __name__ == '__main__':
    # dm01_load_iris()
    # dm02_iris_visualization()
    # dm03_train_test_split()
    dm04_模型评估和预测()