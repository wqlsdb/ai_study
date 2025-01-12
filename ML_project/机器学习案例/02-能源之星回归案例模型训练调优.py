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
	train_features = pd.read_csv('training_features.csv')
	test_features = pd.read_csv('testing_features.csv')
	train_labels = pd.read_csv('training_labels.csv')
	test_labels = pd.read_csv('testing_labels.csv')
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
	# todo: 2-模型预测  y_pred
	y_pred = model.predict(X_test)
	# todo: 3-模型评估 MAE
	model_mae = mae(y_test, y_pred)
	return model_mae


# 训练函数
def train(X, X_test, y, y_test):
	print('=======================模型训练选择初版模型=======================')
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
	model_comparison = model_comparison.sort_values(by='mae', ascending=False)
	print('model_comparison->', model_comparison)
	# 绘制横向柱状图
	model_comparison.plot(kind='barh', x='model', y='mae', )
	plt.ylabel('')
	plt.yticks(size=14)
	plt.xlabel('Mean Absolute Error')
	plt.xticks(size=14)
	plt.title('Model Comparison on Test MAE', size=20)
	plt.show()


# 粗调参数
def modelparam01(X, y):
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
	print('调优结果展示->', random_cv.cv_results_)
	# max_depth=5, min_samples_leaf=4, n_estimators=150, min_samples_split=2, max_features=None
	print('最优的模型->', random_cv.best_estimator_)
	print('最优模型分值->', random_cv.best_score_)


# 参数精调, 当前拿树的棵树参数进行精调
def modelparam02(X, X_test, y, y_test):
	# todo: 1-设置精调参数
	trees_grid = {'n_estimators': [100, 150, 200, 250, 300, 350, 400]}
	# todo: 2-创建粗调后的GBDT模型对象
	model_gbdt = GradientBoostingRegressor(max_depth=5,
										   min_samples_leaf=4,
										   min_samples_split=2,
										   max_features=None)

	# todo: 3-参数精调
	grid_cv = GridSearchCV(estimator=model_gbdt,
						   param_grid=trees_grid,
						   cv=4,
						   scoring='neg_mean_absolute_error',
						   n_jobs=-1,
						   verbose=1,
						   return_train_score=True)
	grid_cv.fit(X, y)
	# todo: 4-绘制训练集和测试集误差曲线分析精调模型效果
	# 获取精调后的参数df对象
	results = pd.DataFrame(grid_cv.cv_results_)
	print('results->', results)
	# x: n_estimators参数值列表
	# y: 训练集和测试集误差值列表
	plt.plot(results['param_n_estimators'], -results['mean_train_score'], label='Training Error')
	plt.plot(results['param_n_estimators'], -results['mean_test_score'], label='Testing Error')
	plt.xlabel('Number of Trees')
	plt.ylabel('Mean Abosolute Error')
	plt.legend()
	plt.title('Performance vs Number of Trees')
	plt.show()
	# todo: 5-初版GBDT模型和精调版GBDT模型效果对比
	# 创建初版GBDT对象
	default_model = GradientBoostingRegressor(random_state=60)
	# 获取精调版GBDT模型
	final_model = grid_cv.best_estimator_
	print('final_model->', final_model)
	# 模型训练
	default_model.fit(X, y)
	# final_model.fit(X, y)
	# 模型预测
	default_pred = default_model.predict(X_test)
	final_pred = final_model.predict(X_test)
	print('默认模型 MAE', mae(y_test, default_pred))
	print('精调模型 MAE', mae(y_test, final_pred))
	# 绘制kde图 真实值和预测值
	sns.kdeplot(y_test, label='values')
	sns.kdeplot(final_pred, label='prediction')
	plt.xlabel('Energy Star Score')
	plt.ylabel('Density')
	plt.title('Test Values and Predictions')
	plt.show()
	# 绘制直方图  真实值和预测值误差
	res = y_test - final_pred
	plt.hist(res, bins=20)
	plt.xlabel('Error')
	plt.ylabel('Count')
	plt.title('Distribution of Residuals')
	plt.show()


# 模型特征解读
def modelfeatures(X, X_test, y, y_test):
	# todo: 1-获取所有特征列列名
	train_features = pd.read_csv('training_features.csv')
	features_cols = train_features.columns
	print('features_cols->', features_cols)
	# todo: 2-训练精调GBDT模型
	model_gbdt = GradientBoostingRegressor(max_depth=5,
										   min_samples_leaf=4,
										   min_samples_split=2,
										   max_features=None,
										   n_estimators=200,
										   random_state=42)
	model_gbdt.fit(X, y)
	gbdt_pred = model_gbdt.predict(X_test)
	print('GBDT全部特征模型 MAE->', mae(y_test, gbdt_pred))
	# todo: 3-获取GBDT模型特征重要分, 保存到df对象中
	feature_results = pd.DataFrame(
		data={'features': list(features_cols), 'importance': model_gbdt.feature_importances_})
	# 根据特征重要分进行降序排序, 重置行索引
	feature_results = feature_results.sort_values(by='importance', ascending=False).reset_index(drop=True)
	# 获取特征重要分前10的特征列名索引下标  features_cols获取列下标
	most_important_features = feature_results['features'][:10]
	# list.index(x) -> 获取列表中x元素对应的下标
	indices = [list(features_cols).index(col) for col in most_important_features]
	print('indices->', indices)
	# todo: 4-获取特征重要分前10数据子集(训练和测试集)
	X_indices = X[:, indices]
	X_test_indices = X_test[:, indices]
	# todo: 5-模型训练 lr gbdt模型
	lr = LinearRegression()
	lr.fit(X, y)
	lr_pred = lr.predict(X_test)
	lr.fit(X_indices, y)
	lr_pred_indices = lr.predict(X_test_indices)
	print('LR 全部特征模型 MAE->', mae(y_test, lr_pred))
	print('LR 前10特征模型 MAE->', mae(y_test, lr_pred_indices))
	# 选择10个特征训练的GBDT模型, 模型复杂度低, 过拟合少
	gbdt_indices = GradientBoostingRegressor(max_depth=5, max_features=None, min_samples_leaf=4, min_samples_split=2, n_estimators=200, random_state=42)
	gbdt_indices.fit(X_indices, y)
	gbdt_pred_indices = gbdt_indices.predict(X_test_indices)
	print('GBDT 前10特征模型 MAE->', mae(y_test, gbdt_pred_indices))


if __name__ == '__main__':
	X, X_test, y, y_test = load_data_features()
	# train(X, X_test, y, y_test)
	# modelparam01(X, y)
	# modelparam02(X, X_test, y, y_test)
	modelfeatures(X, X_test, y, y_test)
