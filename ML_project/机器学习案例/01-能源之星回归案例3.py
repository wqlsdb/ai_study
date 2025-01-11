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
data = pd.read_csv('./data/Energy_and_Water_Data_Disclosure_for_Local_Law_84_2017__Data_for_Calendar_Year_2016_.csv')
# 查看数据集信息
# 查看前五条数据
print('前5行数据->\n', data.head())
# 查看数据集形状 行*列
print('数据集形状->', data.shape)
# 查看数值列统计描述信息
print('统计描述信息->\n', data.describe())
# data.info()
print('目标值列y->\n', data['ENERGY STAR Score'])
# 查看1-100分的数据条目数
print('1-100分数据条目数->\n', data['ENERGY STAR Score'].value_counts())
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

print('=======================特征工程=======================')
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
print('打印数值列热编码后的所有列及数据->\n', categorical_subset.head())
# todo: 6-合并数值特征列和离散特征列
features = pd.concat([numeric_subset, categorical_subset], axis=1)
print('features形状->', features.shape)
