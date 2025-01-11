from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 1.加载鸢尾花数据集
iris_data = load_iris()
# 2.数据预处理
x_train, x_test, y_train, y_test = train_test_split(
    iris_data.data, iris_data.target, test_size=0.2, random_state=22
)
# 3.特征工程
transfer = StandardScaler()
# 分别对训练集和测试集做标准化处理
# todo:训练（拟合）+转换
x_train = transfer.fit_transform(x_train)
# todo:只进行转换，不做拟合
x_test = transfer.transform(x_test)

# 4.模型训练
# todo 实际开发中4.1-4.3需要多次循环，拿到最优的参数，通过修改交叉验证的折数次数，把CV定义为变量，重复循环折叠的次数，拿到最优的
# 4.1 创建KNN分类器对象
estimator = KNeighborsClassifier()
# todo 4.2 定义字典，记录：超参可能出现的值
param_dict = {'n_neighbors': list(range(1, 21))}
# todo 4.3 创建网格搜索对象，
# 参1：模型对象，参2：超参字典，参3：交叉验证的折数
estimator = GridSearchCV(estimator, param_dict, cv=4)

# 4.4 模型训练
estimator.fit(x_train, y_train)

# 5. 模型预测
y_predict = estimator.predict(x_test)
print(f'预测的结果是：{y_predict}')

# 6.打印下，网格搜索 和 交叉验证的结果
print(f'最优组合的平均分：{estimator.best_estimator_}')
print(f'最优组合的模型对象：{estimator.best_estimator_}')
print(f'最优组合的超参：{estimator.best_params_}')
print(f'所有组合的评分结果：{estimator.cv_results_}')
print('-' * 88)

# 7.结合上诉的结果，对模型再次训练做评估
# todo：将上述训练出的最优参数，传入模型中。
estimator = KNeighborsClassifier(n_neighbors=estimator.best_params_.get('n_neighbors'))
estimator.fit(x_train, y_train)
y_predict = estimator.predict(x_test)
from sklearn.metrics import accuracy_score

print(f'模型的准确率为：{accuracy_score(y_test, y_predict)}')
