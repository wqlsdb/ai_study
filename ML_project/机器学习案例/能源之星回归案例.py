import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

sns.set(font_scale=2)
plt.rcParams['font.size'] = 24  # 设置默认字体大小


def dm01_业务数据(verbose=True):
    """
    读取CSV文件并根据verbose参数决定是否打印数据集信息。
    参数:
    - verbose: 如果为True，则打印数据集的信息；如果为False，则只返回数据。
    返回:
    - data: 从CSV文件中读取的数据。
    """
    data = pd.read_csv(
        './data/Energy_and_Water_Data_Disclosure_for_Local_Law_84_2017__Data_for_Calendar_Year_2016_.csv')
    if verbose:
        # 查看数据集
        print('=======================查看数据的行列数=======================')
        print('查看数据的行列数：\n', data.shape)
        print('=======================查看数据样例=======================')
        print('查看数据样例：\n', data.head(5))
        print('=======================查看数据每列信息=======================')
        print('查看数据每列信息：\n', data.describe())
        print('=======================数据的标签列=======================')
        print('数据的标签列：\n', data['ENERGY STAR Score'])
        print('=======================数据集的标签列的数值分布情况=======================')
        print('数据集的标签列的数值分布情况：\n', data['ENERGY STAR Score'].value_counts())
    return data


def dm02_数据清洗(data):
    # todo:1-替换缺失值  Not Available替换成nan值
    # print('替换缺失值前的nan值个数\n', data.isna().sum())
    data = data.replace(to_replace='Not Available', value=np.nan)
    print('=======================填补空数据后的数据信息=======================')
    # print('替换缺失值后的nan值个数\n', data.isna().sum())
    # todo:2-将原本为数值列的列(当前object列)转换成数值列,通过观察数据发现，部分数值列为object类型，需要转换成数值类型
    # df[列名] = df[列名].astype(dtype=)
    for col in data.columns:
        # 判断, 列名中包含以下字符的列需要转换成float类型
        if ('ft²' in col or 'kBtu' in col or 'Metric Tons CO2e' in col or 'kWh' in
                col or 'therms' in col or 'gal' in col or 'Score' in col):
            data[col] = data[col].astype(float)
    # todo:3-删除缺失值占比超过50%的列
    # 统计每列的缺失个数
    mis_val = data.isnull().sum()
    # 统计每类的缺失值占比
    mis_val_percent = mis_val / len(data) * 100
    # 缺失值个数s对象和缺失值占比s对象合并成df对象
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    # 对mis_val_table df对象修改列名，默认列名0,1
    # mis_val_table_ren_columns = mis_val_table.rename(columns={0: 'Missing Values', 1: '% of Total Values'})
    mis_val_table = mis_val_table.rename(columns={0: '缺失值个数', 1: '缺失值占比'})
    # print('缺失值个数s对象和缺失值占比s对象合并成df对象：\n', mis_val_table)
    # 先获取包含缺失值列的df子集,再根据缺失值占比进行降序排序
    mis_val_table = (mis_val_table[mis_val_table['缺失值占比'] != 0]
                     .sort_values('缺失值占比', ascending=False).round(1))
    # print(mis_val_table)
    # 获取缺失值占比超过50%的列.todo:index 属性获取列名
    mis_val_table_over50 = mis_val_table[mis_val_table['缺失值占比'] > 50].index
    # print('缺失值占比超过50%的列：\n', mis_val_table_over50)
    print('=======================删除缺失值占比超过50%的列=======================')
    # 删除前的数据集
    print('删除缺失值占比超过50%的列前的数据集：\n', data.shape)
    # 删除缺失值占比超过50%的列
    drop_cols = data.drop(columns=mis_val_table_over50)
    print('删除缺失值占比超过50%的列后的数据集：\n', drop_cols.shape)
    return data


def dm03_探索性数据分析_EDA之异常值(tf_data):
    print('=======================dm03_探索性数据分析_EDA之异常值=======================')
    # todo: 1-修改目标列的列名
    data = tf_data.rename(columns={'ENERGY STAR Score': 'score'})
    # 统计score这一列的数据分布情况
    print('=======================score这一列的数据分布情况=======================')
    print('score这一列的数据分布情况：\n', data['score'].value_counts())
    '''
    100.0    649
    1.0      299
    '''
    # print('score这一列的数据分布情况：\n',data['score'].hist())
    # todo: 2-探索目标值列的异常值 hist直方图  describe()
    print('查看score这一列的数据分布情况：\n', data['score'].describe())
    # 随机采样 todo,根据直方图可知分数为1的样本太多了
    # data = data[(data['score'] == 1 | data['score'] == 100)].sample(120)
    # data = data[(data['score'] == 1) | (data['score'] == 100)].sample(120)
    # print('随机采样：\n', data)
    # plt.figure(figsize=(10, 8))
    # plt.style.use('fivethirtyeight')
    # # bins: 100箱, 1-100分
    # plt.hist(data['score'].dropna(), bins=100)
    # plt.xlabel('Score')
    # plt.ylabel('Number of Buildings')
    # # todo 解决中文乱码问题
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # # todo 解决负号显示问题
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.title('能源之星评分分布')
    # plt.show()
    # todo: 4-删除异常值 IQR 四分位距方法
    first_quartile = data['Site EUI (kBtu/ft²)'].quantile(q=0.25)
    third_quartile = data['Site EUI (kBtu/ft²)'].quantile(q=0.75)
    IQR = third_quartile - first_quartile
    # 删除异常值，过滤数据子集
    print('查看四分位距法前的结果：\n', data.shape)
    data = data[(data['Site EUI (kBtu/ft²)'] > (first_quartile - 3 * IQR)) & (
            data['Site EUI (kBtu/ft²)'] < (third_quartile + 3 * IQR))]
    print('查看通过四分位距法后过滤的结果：\n', data.shape)
    print('查看通过四分位距法过滤后的数据信息\n', data['Site EUI (kBtu/ft²)'].describe())
    print('Site EUI (kBtu/ft²)统计这一列数据分布情况\n', data['Site EUI (kBtu/ft²)'].value_counts())
    # plt.figure(figsize=(10, 8))
    # plt.style.use('fivethirtyeight')
    # # bins: 100箱, 1-100分
    # plt.hist(data['Site EUI (kBtu/ft²)'].dropna(), bins=100)
    # plt.xlabel('Site EUI')
    # plt.ylabel('Count')
    # # todo 解决中文乱码问题
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # # todo 解决负号显示问题
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.title('能源使用面积分布')
    # plt.show()
    return data


def dm04_EDA之KDE图分析x和y关系(qx_data):
    # 离散值特征列和y关系分析
    # 最大物业特征列
    # todo: 1-删除score列包含缺失值的样本
    print('查看删除空行后的数据：\n', qx_data.shape)
    types = qx_data.dropna(subset=['score'])
    print('查看删除空行后的数据：\n', types.shape)
    # todo: 2-统计Largest Property Use Type列每个特征值出现的次数
    types = qx_data['Largest Property Use Type'].value_counts()
    print('统计Largest Property Use Type用于业务场景分布情况\n', types)
    # todo: 3-获取出现次数大于100的特征值
    types = types[types.values > 100].index
    print('查看大于100的办公业务场景：\n', types)
    # todo: 4-绘制KDE核密度估计图
    count = 0
    plt.figure(figsize=(12, 10))
    for b_type in types:
        # todo types中拿到的仅仅是列表column的名字且数据大于100的，因此需要通过column过去对应的数据
        subset = qx_data[qx_data['Largest Property Use Type'] == b_type]
        # count=count+1
        # print(subset)
        # todo
        #  遍历出的不同类型的建筑类型进行绘制KDE图
        sns.kdeplot(subset['score'].dropna(), label=b_type)
    # print('一共遍历了几次：',count)

    # plt.style.use('fivethirtyeight')
    plt.xlabel('Energy Star Score', size=20)
    plt.ylabel('Density', size=20)
    # todo 解决中文乱码问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # todo 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('能源之星建筑类型分布得分', size=20)
    # todo 显示图例，就是显示图例的标签知道每个线是干嘛的
    plt.legend()
    plt.show()

    # 自治市镇特征列
    print('统计score列删除空数据前行列：\n', qx_data.shape)
    # todo: 1-删除score列包含缺失值的样本
    boroughs = qx_data.dropna(subset=['score'])
    print('统计score列删除空数据后行列：\n', boroughs.shape)

    # todo: 2-统计Largest Property Use Type列每个特征值出现的次数
    boroughs = boroughs['Borough'].value_counts()  # df.groupby('Borough')['Borough'].count()
    print('统计城镇分数分布情况\n', boroughs)
    # todo: 3-获取出现次数大于100的特征值
    boroughs = boroughs[boroughs.values>100].index
    print('查看大于100的城镇：\n',boroughs)
    # todo: 4-绘制KDE核密度估计图
    plt.figure(figsize=(12, 10))
    for borough in boroughs:
        # 获取每个特征值对应的数据子集
        subset = qx_data[qx_data['Borough'] == borough]
        # 绘制kde图
        sns.kdeplot(subset['score'].dropna(), label=borough)
    # plt.style.use('fivethirtyeight')
    plt.xlabel('Energy Star Score', size=20)
    plt.ylabel('Density', size=20)
    # todo 解决中文乱码问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # todo 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('能源之星城镇密度分布', size=20)
    # todo 显示图例，就是显示图例的标签知道每个线是干嘛的
    plt.legend()
    plt.show()
    # 原始数值特征列和y关系分析
    # todo: 1-获取数值列,计算相关系, 获取特征列和score列的series对象
    # todo:代码解释 select_dtypes('number')获取数值列
    # todo:corr()计算数值列之间的相关系数输出的是一个对称矩阵，
    #  corr()['score']：从相关系数矩阵中提取与score列的相关系数
    # todo 最后返回的是一个series对象，排序后输出
    corr_data = qx_data.select_dtypes('number').corr()['score'].sort_values()
    print(corr_data)
    print('查看所有特征列组合的dataframe的行数和列数：\n',corr_data.shape)
    # todo: 2-结果保存到文件中
    corr_data.to_csv('./data/corr_data.csv', index=True)

    # 特征列和y是否存在非线性关系
    # 将数值列进行平方根和对数计算, 衍生出两个新的特征
    # todo: 1-获取数值列
    numeric_subset = qx_data.select_dtypes("number")
    print('查看数值列的形状：\n', numeric_subset.shape)
    # todo: 2-循环遍历对数值列进行平方根和对数计算, 衍生出两个新的特征
    for col in numeric_subset.columns:
        # todo 该列若不是分数列，全部开平方根和取对数
        if col != 'score':
            # 获取数值列的平方根
            numeric_subset[col + 'sqrt'] = np.sqrt(numeric_subset[col])
            # 获取数值列的对数
            numeric_subset[col + 'log'] = np.log(numeric_subset[col])
    print('数值列的平方根和取对数后的数据：\n', numeric_subset.shape)
    # todo: 3-获取离散型特征列, 和第2步结果放到一起, 观察x和y的非线性关系
    categorical_subset = qx_data[['Largest Property Use Type', 'Borough']]
    # todo: 4-离散型特征进行one-hot编码处理
    categorical_subset = pd.get_dummies(categorical_subset)
    print('离散型特征one-hot编码处理后的形状：\n', categorical_subset.shape)
    # todo: 5-将离散型特征列和数值特征列合并到一起
    features = pd.concat([numeric_subset,categorical_subset])
    print('离散型特征one-hot编码处理后的形状：\n', features.shape)
    # todo: 6-删除score列中包含缺失值的样本
    features = features.dropna(subset=['score'])
    print('删除score列中包含缺失值的样本后的形状：\n', features.shape)
    # todo: 7-计算皮尔逊相关系数
    corr_data2 = features.corr()['score'].dropna().sort_values()
    corr_data2.to_csv('./data/corr_data2.csv', index=True)
    plt.figure(figsize=(12, 12)) # 用量越大 分数越低 有明显的线性相关性
    plt.scatter(features['Site EUI (kBtu/ft²)'], features['score'])
    # plt.xlabel('Energy Star Score', size=20)
    # plt.ylabel('Density', size=20)
    # todo 解决中文乱码问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # todo 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('特征列之间的相关系数', size=20)
    # todo 显示图例，就是显示图例的标签知道每个线是干嘛的
    plt.legend()
    plt.show()

if __name__ == '__main__':
    data = dm01_业务数据(verbose=False)
    tf_data = dm02_数据清洗(data)
    qx_data = dm03_探索性数据分析_EDA之异常值(tf_data)
    dm04_EDA之KDE图分析x和y关系(qx_data)
