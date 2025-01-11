"""
交叉验证解释:
    概述:
        它是一种更加完善的, 可信度更高的模型预估方式, 思路是: 把数据集分成n份, 每次都取1份当做测试集, 其它的当做训练集, 然后计算模型的: 评分.
        然后再用下1份当做测试集, 其它当做训练集, 计算模型评分, 分成几份, 就进行几次计算, 最后计算: 所有评分的均值, 当做模型的最终评分.
    好处:
        交叉验证的结果 比 单一切分训练集和测试集 获取评分结果, 可信度要高.
    细节:
        1.把数据集分成N份, 就叫: N折交叉验证, 例如: 分成4份, 就叫4折交叉验证.
        2.交叉验证一般不会单独使用, 而是结合网格搜索一起使用.

网格搜索:
    概述:
        它是机器学习内置的API, 一般结合交叉验证一起使用, 指的是: GridSearchCV, 目的是: 寻找最优超参.
    格式:
        GridSearchCV(estimator模型对象, params = [超参可能出现的值1, 值2, 值3...], cv=折数)
    超参解释:
        模型中需要用户(程序员)手动传入的参数 -> 统称为: 超参.
    目的:
        网格搜索 + 交叉验证 目的都是为了 寻找模型的 最优的解决方案, 追求较高的效率.
"""
# 导入工具包
from sklearn.datasets import load_iris  # 加载鸢尾花测试集的.
from sklearn.model_selection import train_test_split, GridSearchCV  # 分割训练集和测试集的,  网格搜索 + 交叉验证.
from sklearn.preprocessing import StandardScaler  # 数据标准化的
from sklearn.neighbors import KNeighborsClassifier  # KNN算法 分类对象
from sklearn.metrics import accuracy_score  # 模型评估的, 计算模型预测的准确率

# 1. 加载鸢尾花数据集.
iris_data = load_iris()
# 2. 数据预处理.
x_train, x_test, y_train, y_test = train_test_split(
    iris_data.data, iris_data.target, test_size=0.2, random_state=22)
# 3. 特征工程, 数据标准化(特征预处理)
transfer = StandardScaler()
# 分别对训练集 和 测试集进行标准化处理.
x_train = transfer.fit_transform(x_train)  # 训练(拟合) + 转换
x_test = transfer.transform(x_test)  # 只进行转换, 不进行拟合.

# 4. 模型训练.
# 4.1 创建KNN 分类器对象 -> 模型对象.
estimator = KNeighborsClassifier()  # 注意: 不要传入超参, 即: K的数值.
# 4.2 定义字典, 记录: 超参可能出现的值.
param_dict = {'n_neighbors': list(range(1, 21))}    # K的范围: 1 ~ 21 包左不包右
# 4.3 创建网格搜索对象.
# 参1: 模型对象, 参2: 超参字典, 参3: 交叉验证的折数.
# 处理完之后, 会返回1个功能更加强大的模型对象.
estimator = GridSearchCV(estimator, param_dict, cv=4)

# 4.4 模型训练.
estimator.fit(x_train, y_train)

# 5. 模型预测.
y_predict = estimator.predict(x_test)
print(f'预测结果是: {y_predict}')

# 6. 打印下 网格搜索 和 交叉验证的结果.
print(estimator.best_score_)        # 最优组合的 平均分,    0.9666666666666666
print(estimator.best_estimator_)    # 最优组合的 模型对象
print(estimator.best_params_)       # 最优组合的 超参数(供参考) {'n_neighbors': 3}
print(estimator.cv_results_)        # 所有组合的 评分结果(过程)
print('-' * 22)

# 7. 结合上述的结果, 最模型再次做评估.
estimator = KNeighborsClassifier(n_neighbors=3)
estimator.fit(x_train, y_train)
y_predict = estimator.predict(x_test)
print(f'模型准确率为: {accuracy_score(y_test, y_predict)}')   # 0.9666666666666667
