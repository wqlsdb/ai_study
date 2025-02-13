import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt  # 数据可视化
from IPython.core.pylabtools import figsize
import seaborn as sns  # Seaborn 可视化
# 特征值标准化和缺失值填充
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
# 机器学习模型
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
# 参数调整
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# %matplotlib inline
warnings.filterwarnings('ignore')
# pd.set_option('display.max_columns', 60)
# 设置默认字体
plt.rcParams['font.size'] = 24
sns.set(font_scale=2)


def load_data_features():
    print('=======================特征列缺失值填充和归一化/目标列形状修改=======================')
    # todo: 1-加载训练和测试数据集
    train_features = pd.read_csv('./data/training_features.csv')
    test_features = pd.read_csv('./data/testing_features.csv')
    train_labels = pd.read_csv('./data/training_labels.csv')
    test_labels = pd.read_csv('./data/testing_labels.csv')
    print('train_features->', train_features.shape)
    print('train_labels->', train_labels.shape)
    print('train_features缺失值数量->', train_features.isnull().sum())
    # todo: 2-特征列缺失值填充  中位数策略
    # 创建缺失值填充器
    # strategy:填充策略 中位数
    imputer = SimpleImputer(strategy='median')
    # 训练填充器
    imputer.fit(train_features)
    # 填充
    X = imputer.transform(train_features)
    X_test = imputer.transform(test_features)
    print('X->', np.sum(np.isnan(X)))
    print('X_test->', np.sum(np.isnan(X_test)))
    # todo: 3-特征列归一化
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_test = scaler.transform(X_test)
    print('X归一化的值->', X)
    # todo: 4-目标列修改形状  原形状(6622, 1) 新形状(6622,)
    # values: 获取值, 返回numpy数组
    # train_labels.values.reshape((len(train_labels),))
    # -1: 自动计算  (6622, 1)->(-1, 2)-> -1位置的值*2=6622*1 6622/2=3311
    # ()->元组有多少个值(轴数)就是多少维度, 当前位置的值大小就是当前数据个数
    y = train_labels.values.reshape((-1,))
    y_test = test_labels.values.reshape((-1,))
    print('y的形状为->', y.shape)
    print('y->', y)
    return X, X_test, y, y_test

# 计算模型MAE值 平均绝对误差
def mae(y_true, y_pred):
	return np.mean(abs(y_true - y_pred))

# 模型训练过程函数
def fit_and_evaluate(X, X_test, y, y_test, model):
    # todo: 1-模型训练
    model.fit(X, y)
    y_pred = model.predict(X_test)
    model_mae = mae(y_test,y_pred)
    return model_mae

# 训练模型调优
def train(X, X_test, y, y_test):
    print('=======================模型训练调优=======================')
    # 线性回归模型 lr
    lr = LinearRegression()
    lr_mae = fit_and_evaluate(X, X_test, y, y_test, lr)
    print('lr MAE值->', lr_mae)
    # 随机森林回归模型 rfr
    rfr = RandomForestRegressor(random_state=60)
    rfr_mae = fit_and_evaluate(X, X_test, y, y_test, rfr)
    print('rfr MAE值->', rfr_mae)
    # GBDT回归模型 gbdt
    gbdt = GradientBoostingRegressor(random_state=60)
    gbdt_mae = fit_and_evaluate(X, X_test, y, y_test, gbdt)
    print('gbdt MAE值->', gbdt_mae)
    # KNN回归模型 knn
    knn = KNeighborsRegressor(n_neighbors=10)
    knn_mae = fit_and_evaluate(X, X_test, y, y_test, knn)
    print('knn MAE值->', knn_mae)
    # 模型名称和MAE保存到df对象中
    model_comparison = pd.DataFrame(data={'model': ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'KNN'],
                                          'mae': [lr_mae, rfr_mae, gbdt_mae, knn_mae]})
    # 根据mae进行降序排序
    model_comparison.sort_values(by='mae', ascending=False)
    print('model_comparison->\n', model_comparison)
    # 绘制横向柱状图
    plt.figure(figsize=(40, 20))
    model_comparison.plot(kind='barh', x='model', y='mae', )
    plt.ylabel('')
    plt.yticks(size=8)
    plt.xlabel('Mean Absolute Error')
    plt.xticks(size=8)
    plt.title('Model Comparison on Test MAE', size=20)
    plt.legend()
    # plt.show()

# 调优模型(粗调优)
def model_param01(X,y):
    print('=======================模型调优(粗调优)=======================')
    # 调整GBDT模型
    # todo: 1-设置粗调参数, 字典类型
    n_estimators = [30, 60, 90, 100, 150]  # 有多少棵树
    max_depth = [1, 2, 3, 4, 5]  # 树的最大深度
    min_samples_leaf = [2, 4, 6, 8]  # 每个叶子节点的最少样本数量
    min_samples_split = [2, 4, 6, 10]  # 节点分裂需满足的最少样本数量
    max_features = [1.0, 'sqrt', 'log2', None]  # 建树使用的最大特征
    # 1-2 定义要搜索的参数
    params_grid = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'min_samples_leaf': min_samples_leaf,
                   'min_samples_split': min_samples_split,
                   'max_features': max_features}
    # todo: 2-创建GBDT模型对象
    gbdt = GradientBoostingRegressor(random_state=60)
    # todo: 3-参数粗调 随机参数搜索
    random_cv = RandomizedSearchCV(estimator=gbdt,  # 模型对象
                                   param_distributions=params_grid,  # 参数字典
                                   cv=4,  # 4折交叉验证
                                   n_iter=25,  # 25种参数组合
                                   scoring='neg_mean_absolute_error',  # 评估指标, 负的平均绝对误差(值越大模型越好)
                                   n_jobs=-1,  # 可用所有cpu资源
                                   verbose=1,  # 打印调优信息
                                   random_state=42)
    random_cv.fit(X, y)
    # todo: 4-粗调效果
    # 结果保存到文件中
    # pd.DataFrame(random_cv.cv_results_).to_csv('./cv_results.csv')
    print('调优结果展示->\n', random_cv.cv_results_)

# 参数精调, 当前拿树的棵树参数进行精调
# def model_param02(X, y):

if __name__ == '__main__':
    X, X_test, y, y_test = load_data_features()
    # train(X, X_test, y, y_test)
    model_param01(X, y)