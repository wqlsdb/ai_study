import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize  # 用来设置图片大小
import seaborn as sns
from sklearn.model_selection import train_test_split  # 训练集测试集划分
import warnings

warnings.filterwarnings('ignore')

print('=======================查看数据集=======================')
# 加载数据集
data = pd.read_csv('data/Energy_and_Water_Data_Disclosure_for_Local_Law_84_2017__Data_for_Calendar_Year_2016_.csv')
# 查看数据集信息
# 查看前五条数据
print('前5行数据->', data.head())
# 查看数据集形状 行*列
print('数据集形状->', data.shape)
# 查看数值列统计描述信息
print('统计描述信息->', data.describe())
data.info()
print('目标值列y->', data['ENERGY STAR Score'])
# 查看1-100分的数据条目数
print('1-100分数据条目数->', data['ENERGY STAR Score'].value_counts())
print('=======================数据清洗=======================')
# todo:1-替换缺失值  Not Available替换成nan值
data = data.replace(to_replace='Not Available', value=np.nan)
# data.info()
# todo:2-将原本为数值列的列(当前object列)转换成数值列
# df[列名] = df[列名].astype(dtype=)
for col in data.columns:
	# 判断, 列名中包含以下字符的列需要转换成float类型
	if ('ft²' in col or 'kBtu' in col or 'Metric Tons CO2e' in col or 'kWh' in
			col or 'therms' in col or 'gal' in col or 'Score' in col):
		data[col] = data[col].astype(float)
# data.info()
# todo:3-删除缺失值占比超过50%的列
# data = data.dropna(thresh=len(data)*0.5, axis=1)
# 统计每列的缺失个数
mis_val = data.isnull().sum()
# print('mis_val->', mis_val)
# 统计每类的缺失值占比
# data.shape[0]
mis_val_percent = mis_val / len(data) * 100
# print('mis_val_percent->', mis_val_percent)
# 缺失值个数s对象和缺失值占比s对象合并成df对象
mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
# print('mis_val_table->', mis_val_table)
# 修改合并df的列名
mis_val_table_ren_columns = mis_val_table.rename(columns={0: 'Missing Values', 1: '% of Total Values'})
# 先获取包含缺失值列的df子集,再根据缺失值占比进行降序排序
mis_val_table_ren_columns = (mis_val_table_ren_columns[mis_val_table_ren_columns['% of Total Values'] != 0].
							 sort_values('% of Total Values', ascending=False).round(1))
# print('mis_val_table_ren_columns->', mis_val_table_ren_columns)
# 获取缺失值占比超过50%的列名
drop_cols = mis_val_table_ren_columns[mis_val_table_ren_columns['% of Total Values'] > 50].index
# print('drop_cols->', drop_cols)
# 删除列名
data = data.drop(columns=drop_cols)
print(data.shape)
print('=======================探索性数据分析EDA=======================')
print('=======================EDA之异常值=======================')
# todo: 1-修改目标列的列名
data = data.rename(columns={'ENERGY STAR Score': 'score'})
# todo: 2-探索目标值列的异常值 hist直方图  describe()
print(data['score'].describe())
# data[data['score'] == 1].sample(120)  # 随机采样
# plt.style.use('fivethirtyeight')
# # bins: 100箱, 1-100分
# plt.hist(data['score'].dropna(), bins=100)
# plt.xlabel('Score')
# plt.ylabel('Number of Buildings')
# plt.title('Energy Star Score Distribution')
# plt.show()
# todo: 3-探索特征值列的异常值 拿一个特征举例 hist直方图  describe()
print(data['Site EUI (kBtu/ft²)'].describe())
# plt.style.use('fivethirtyeight')
# bins: 100箱, 1-100分
# plt.hist(data['Site EUI (kBtu/ft²)'].dropna(), bins=20)
# plt.xlabel('Site EUI')
# plt.ylabel('Count')
# plt.title('Site EUI Distribution')
# plt.show()
# todo: 4-删除异常值 IQR 四分位距方法
first_quartile = data['Site EUI (kBtu/ft²)'].quantile(q=0.25)
third_quartile = data['Site EUI (kBtu/ft²)'].quantile(q=0.75)
IQR = third_quartile - first_quartile
# 删除异常值, 过滤数据子集
data = data[(data['Site EUI (kBtu/ft²)'] > (first_quartile - 3 * IQR)) & (
		data['Site EUI (kBtu/ft²)'] < (third_quartile + 3 * IQR))]
print(data.shape)
print(data['Site EUI (kBtu/ft²)'].describe())
plt.hist(data['Site EUI (kBtu/ft²)'].dropna(), bins=20)
plt.xlabel('Site EUI')
plt.ylabel('Count')
plt.title('Site EUI Distribution')
plt.show()
print('=======================EDA之KDE图分析x和y关系=======================')
# 离散值特征列和y关系分析
# 最大物业特征列
# todo: 1-删除score列包含缺失值的样本
types = data.dropna(subset=['score'])
print(types)
# todo: 2-统计Largest Property Use Type列每个特征值出现的次数
types = types['Largest Property Use Type'].value_counts()
print(types)
# todo: 3-获取出现次数大于100的特征值
types = types[types.values > 100].index
print(types)
# todo: 4-绘制KDE核密度估计图
for b_type in types:
	# 获取每个特征值对应的数据子集
	subset = data[data['Largest Property Use Type'] == b_type]
	# 绘制kde图
	sns.kdeplot(subset['score'].dropna(), label=b_type)
plt.xlabel('Energy Star Score', size=20)
plt.ylabel('Density', size=20)
plt.title('Density Plot of Energy Star Scores by Building Type', size=28)
plt.legend()
plt.show()
# 自治市镇特征列
# todo: 1-删除score列包含缺失值的样本
boroughs = data.dropna(subset=['score'])
print(boroughs)
# todo: 2-统计Largest Property Use Type列每个特征值出现的次数
boroughs = boroughs['Borough'].value_counts()  # df.groupby('Borough')['Borough'].count()
print(boroughs)
# todo: 3-获取出现次数大于100的特征值
boroughs = boroughs[boroughs.values > 100].index
print(boroughs)
# todo: 4-绘制KDE核密度估计图
for borough in boroughs:
	# 获取每个特征值对应的数据子集
	subset = data[data['Borough'] == borough]
	# 绘制kde图
	sns.kdeplot(subset['score'].dropna(), label=borough)
plt.xlabel('Energy Star Score', size=20)
plt.ylabel('Density', size=20)
plt.title('Density Plot of Energy Star Scores by Building Type', size=28)
plt.legend()
plt.show()
print('=======================EDA之皮尔逊相关系数分析x和y关系=======================')
# 原始数值特征列和y关系分析
# todo: 1-获取数值列,计算相关系, 获取特征列和score列的series对象
corr_data = data.select_dtypes('number').corr()['score'].sort_values()
print(corr_data)
# todo: 2-结果保存到文件中
corr_data.to_csv('111.csv')
# 特征列和y是否存在非线性关系
# 将数值列进行平方根和对数计算, 衍生出两个新的特征
# todo: 1-获取数值列
numeric_subset = data.select_dtypes("number")  # df子集
# todo: 2-循环遍历对数值列进行平方根和对数计算, 衍生出两个新的特征
for col in numeric_subset.columns:
	if col != 'score':
		# 平方根新特征  df[new_col] = 新值
		numeric_subset['sqrt_' + col] = np.sqrt(numeric_subset[col])
		# 对数新特征
		numeric_subset['log_' + col] = np.log(numeric_subset[col])
# todo: 3-获取离散型特征列, 和第2步结果放到一起, 观察x和y的非线性关系
categorical_subset = data[['Largest Property Use Type', 'Borough']]
# todo: 4-离散型特征进行one-hot编码处理
categorical_subset = pd.get_dummies(categorical_subset)
# todo: 5-将离散型特征列和数值特征列合并到一起
features = pd.concat([numeric_subset, categorical_subset], axis=1)
# todo: 6-删除score列中包含缺失值的样本
features = features.dropna(subset=['score'])
# todo: 7-计算皮尔逊相关系数
# dropna(): log计算 如果x小于0 产生nan
corr_data2 = features.corr()['score'].dropna().sort_values()
print(corr_data2)
corr_data2.to_csv('222.csv')
# 通过图像关系x和y之间的关系
figsize(12, 12)  # 用量越大 分数越低 有明显的线性相关性
plt.scatter(features['Site EUI (kBtu/ft²)'], features['score'])
plt.show()

figsize(12, 12)  # 相关性不是太好 但总体分数变化不大
plt.scatter(features['Weather Normalized Site Electricity Intensity (kWh/ft²)'], features['score'])
plt.show()
print('=======================双变量图分析x和y关系=======================')
# 通过成对关系图, 观察x和x之间是否存在线性关系
print(types)
# 最大业务特征列和EUI能源使用量特征列以及y列关系
# todo: 1-在features df上添加最大物业特征列
features['Largest Property Use Type'] = data.dropna(subset=['score'])['Largest Property Use Type']
# todo: 2-获取条目数大于100的物业特征值对应的数据子集
# isin(values=[值1, 值2, 值3, ...])
features = features[features['Largest Property Use Type'].isin(types)]
# todo: 3-绘制散点图
figsize(12, 10)
sns.lmplot(x='Site EUI (kBtu/ft²)', y='score',
		   hue='Largest Property Use Type',
		   data=features,
		   scatter_kws={'alpha': 0.8, 's': 60},
		   fit_reg=False,
		   height=15,
		   aspect=1.2)
# size:高度 aspect:宽高比
# 调整坐标轴上的标签和标题
plt.xlabel('Site EUI', size=28)
plt.ylabel('Energy Star Score', size=28)
plt.title('Energy Star Score vs Site EUI', size=36)
plt.show()

"""
# 成对关系图
# 实际工作中取更多列绘制成对关系图
plot_data = features[['score', 'Site EUI (kBtu/ft²)',
					  'Weather Normalized Source EUI (kBtu/ft²)',
					  'log_Total GHG Emissions (Metric Tons CO2e)']]
# 2-2 将inf替换成nan
# np.log(0)->-inf np.log(x),x非常大,计算内存溢出->inf
plot_data = plot_data.replace({np.inf: np.nan, -np.inf: np.nan})
# 2-3重命名列
plot_data = plot_data.rename(columns={'Site EUI (kBtu/ft²)': 'Site EUI',
									  'Weather Normalized Source EUI (kBtu/ft²)': 'Weather Norm EUI',
									  'log_Total GHG Emissions (Metric Tons CO2e)': 'log GHG Emissions'})
# 2-4 去掉缺失值
plot_data = plot_data.dropna()


def corr_func(x, y, **kwargs):  # 计算相关系数
	r = np.corrcoef(x, y)[0][1]  # 根据数据格式把r提取出来 print('\n-->', np.corrcoef(x, y))
	'''
array([[1., -0.74116006],
	   [-0.74116006, 1.]])
'''
# Get Current Axes，获取当前坐标轴， 如果没有会创建新的坐标轴
ax = plt.gca()
# 在图形的指定位置添加文本注释
# xy：文本注释的位置， x轴位置是 20% 处， y轴位置是 80% 处
# xycoords：ax.transAxes 坐标系是基于轴的坐标系（轴坐标系），即 xy=(0, 0) 是坐标轴的左下角，xy=(1, 1) 是右上角
ax.annotate('r={:.2f}'.format(r),  # 数据2位小数
			xy=(.2, .8), xycoords=ax.transAxes,
			size=20)


# 2-5 绘制成对关系图
# 创建成对关系图对象
grid = sns.PairGrid(data=plot_data, height=3)
grid.map_upper(plt.scatter, color='red', alpha=0.6)  # 右上绘制散点图
grid.map_diag(plt.hist, color='red', edgecolor='black')  # 对角线绘制直方图
grid.map_lower(corr_func)  # 左下显示相关性绘制density plot
grid.map_lower(sns.kdeplot, cmap=plt.cm.Reds)
# 设置标题
plt.suptitle('Pairs Plot of Energy Data', size=36, y=1.02)
plt.show()
"""
print('=======================特征工程=======================')


def remove_collinear_features(features, threshold=0.6):
	# todo: 1-将目标值y列保存起来, 分析x和x之间的线性关系
	y = features['score']
	# todo: 2-获取所有特征列子集
	features = features.drop(columns=['score'])
	while True:  # 后续取最大系数的列名进行删除, 一列中有多个大于0.6的列
		# todo: 3-计算x和x之间的相关系数矩阵
		corr_matrix = features.corr()
		# print('corr_matrix->', corr_matrix)
		# todo: 4-将对角线的系数值1置为0, 自己和自己相关性1
		for i in range(len(corr_matrix)):
			# df.iloc[行下标, 列下标]
			corr_matrix.iloc[i, i] = 0
		# print('corr_matrix->', corr_matrix)
		# todo: 5-定义空列表, 存储待删除的特征列名
		drop_cols = []
		# todo: 6-遍历相关系数矩阵
		for col in corr_matrix.columns:
			# 如果特征列不在删除列表中
			if col not in drop_cols:
				# 进行绝对值处理, 对当前列的series对象处理
				# df[列名]
				v = np.abs(corr_matrix[col])
				# 并且判断最大值是否大于阈值, 如果大于
				if np.max(v) > threshold:
					# 获取最大值对应的列名保存到待删除列表中
					# np.argmax(v): 获取最大值对应的行索引下标
					idx = np.argmax(v)
					# print('idx->', idx)
					# v.index: 获取series对象的行索引值(列名) ['order', 'property id', ...]
					index_list = v.index
					# print('index_list->', index_list)
					name = index_list[idx]
					# print('name->', name)
					# 将name值保存到待删除列表中
					drop_cols.append(name)
		# todo: 7-如果待删除特征列表不为空, 进行删除操作, 否则结束死循环
		# 0,'',None->False, 其他所有值->True
		if drop_cols:
			features = features.drop(columns=drop_cols)
		else:
			break
	# print('features->', features.shape)
	# todo: 8-将目标值y列保存回特征列子集中
	# df[new_col] = 新值
	features['score'] = y
	return features


def get_features(data):
	print('=======================特征工程之特征选择和特征衍生=======================')
	# todo: 1-复制data->删除异常值后的data
	features = data.copy()
	# todo: 2-选择数值特征列 -> 特征选择
	numeric_subset = features.select_dtypes('number')
	print('numeric_subset形状->', numeric_subset.shape)
	# todo: 3-对数值特征列进行对数计算生成新特征 -> 特征衍生
	for col in numeric_subset.columns:
		if col != 'score':
			numeric_subset['log_' + col] = np.log(numeric_subset[col])
	print('numeric_subset形状->', numeric_subset.shape)
	# todo: 4-选择离散型特征列 -> 特征选择
	categorical_subset = features[['Borough', 'Largest Property Use Type']]
	print('categorical_subset形状->', categorical_subset.shape)
	# todo: 5-对离散型特征进行one-hot编码
	categorical_subset = pd.get_dummies(categorical_subset)
	print('categorical_subset形状->', categorical_subset.shape)
	print('categorical_subset->', categorical_subset.head())
	# todo: 6-合并数值特征列和离散特征列
	features = pd.concat([numeric_subset, categorical_subset], axis=1)
	print('features形状->', features.shape)
	print('=======================特征工程之删除共线特征=======================')

	# 验证特征列之间存在共线性
	plot_data = data[['Weather Normalized Site EUI (kBtu/ft²)', 'Site EUI (kBtu/ft²)']].dropna()
	print(plot_data.corr())
	plt.plot(plot_data['Site EUI (kBtu/ft²)'], plot_data['Weather Normalized Site EUI (kBtu/ft²)'], 'bo')
	plt.xlabel('Site EUI')
	plt.ylabel('Weather Norm EUI')
	plt.title('Weather Norm EUI vs Site EUI')
	plt.show()
	# def remove_collinear_features(features, threshold=0.6):
	# 	# todo: 1-将目标值y列保存起来, 分析x和x之间的线性关系
	# 	y = features['score']
	# 	# todo: 2-获取所有特征列子集
	# 	features = features.drop(columns=['score'])
	# 	while True:  # 后续取最大系数的列名进行删除, 一列中有多个大于0.6的列
	# 		# todo: 3-计算x和x之间的相关系数矩阵
	# 		corr_matrix = features.corr()
	# 		# print('corr_matrix->', corr_matrix)
	# 		# todo: 4-将对角线的系数值1置为0, 自己和自己相关性1
	# 		for i in range(len(corr_matrix)):
	# 			# df.iloc[行下标, 列下标]
	# 			corr_matrix.iloc[i, i] = 0
	# 		# print('corr_matrix->', corr_matrix)
	# 		# todo: 5-定义空列表, 存储待删除的特征列名
	# 		drop_cols = []
	# 		# todo: 6-遍历相关系数矩阵
	# 		for col in corr_matrix.columns:
	# 			# 如果特征列不在删除列表中
	# 			if col not in drop_cols:
	# 				# 进行绝对值处理, 对当前列的series对象处理
	# 				# df[列名]
	# 				v = np.abs(corr_matrix[col])
	# 				# 并且判断最大值是否大于阈值, 如果大于
	# 				if np.max(v) > threshold:
	# 					# 获取最大值对应的列名保存到待删除列表中
	# 					# np.argmax(v): 获取最大值对应的行索引下标
	# 					idx = np.argmax(v)
	# 					# print('idx->', idx)
	# 					# v.index: 获取series对象的行索引值(列名) ['order', 'property id', ...]
	# 					index_list = v.index
	# 					# print('index_list->', index_list)
	# 					name = index_list[idx]
	# 					# print('name->', name)
	# 					# 将name值保存到待删除列表中
	# 					drop_cols.append(name)
	# 		# todo: 7-如果待删除特征列表不为空, 进行删除操作, 否则结束死循环
	# 		# 0,'',None->False, 其他所有值->True
	# 		if drop_cols:
	# 			features = features.drop(columns=drop_cols)
	# 		else:
	# 			break
	# 	# print('features->', features.shape)
	# 	# todo: 8-将目标值y列保存回特征列子集中
	# 	# df[new_col] = 新值
	# 	features['score'] = y
	# 	return features

	features = remove_collinear_features(features, 0.6)
	print('features->', features.shape)
	print('=======================特征工程之数据集分割=======================')
	# todo: 1-删除都为NaN值的列 -> log(x<0)->nan 按列删
	features = features.dropna(axis=1, how='all')
	# todo: 2-删除score目标列为nan值的样本 按行删
	score = features.dropna(subset=['score'])
	# todo: 3-获取features和targets两份数据
	features = score.drop(columns=['score'])
	targets = pd.DataFrame(data=score['score'])
	# todo: 4-将features数据集中的-inf/inf替换成nan值  log(x=0)->-inf log(x>9999999)->inf
	features = features.replace({-np.inf: np.nan, np.inf: np.nan})
	# todo: 5-数据集分割
	X, X_test, y, y_test = train_test_split(features, targets, test_size=0.3, random_state=42)
	print('X->', X.shape)
	print('X_test->', X_test.shape)
	print('y->', y.shape)
	print('y_test->', y_test.shape)
	# todo: 6-数据集保存文件中
	X.to_csv("./data/training_features.csv", index=False)
	X_test.to_csv("./data/testing_features.csv", index=False)
	y.to_csv("./data/training_labels.csv", index=False)
	y_test.to_csv("./data/testing_labels.csv", index=False)
	return X, X_test, y, y_test


X, X_test, y, y_test = get_features(data)
print('===========================baseline模型=======================')


def mae(y_true, y_pred):
	# y_true:真实y值
	# y_pred:预测的y值
	return np.mean(abs(y_true - y_pred))


# 假设基线模型预测的y值都是66分
baseline_guess = np.median(y)
print('baseline_guess->', baseline_guess)
print('baseline模型的平均绝对误差->', mae(y, baseline_guess))
