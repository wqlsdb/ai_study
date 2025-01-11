import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#                                               准确率           精确率            召回率        F1值        roc曲线          分类评估报告
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# 1. 定义函数, 用于实现: 数据预处理.
def dm01_数据预处理():
    # 1. 读取数据.
    data = pd.read_csv('./data/churn.csv')
    # data.info()
    # print(data.head(10))

    # 2. 因为上述的Churn, gender是字符串类型, 我们对其做热编码(one-hot)处理.
    data = pd.get_dummies(data)
    # data.info()
    # print(data.head(10))
    # 3. 删除列, 因为热编码之后, 会多出一个列, 我们删除掉.
    data.drop(['gender_Male', 'Churn_No'], axis=1, inplace=True)
    # data.info()
    # print(data.head(10))

    # 4. 修改列名.
    data.rename(columns={'Churn_Yes': 'flag'}, inplace=True)
    # print(data.head(10))
    # 5. 我们查看下数据集中, 标签 是否是 均衡的.
    # print(data.flag.value_counts())
    # print(data.columns)


# 2. 定义函数, 用于显示: 月度会员的流失情况.
def dm02_会员流失可视化情况():
    # 1. 读取数据.
    data = pd.read_csv('./data/churn.csv')
    # 2. 对上述的数据做 热编码处理.
    data = pd.get_dummies(data)
    # 3. 删除列, 因为热编码之后, 会多出一个列, 我们删除掉.
    data.drop(['gender_Male', 'Churn_No'], axis=1, inplace=True)
    # 4. 修改列名.
    data.rename(columns={'Churn_Yes': 'flag'}, inplace=True)
    # 5. 查看数据集的分布情况.
    print(data.flag.value_counts())
    # 6. 通过计数柱状图, 绘制(月度)会员的流失情况.
    # 参数x意思是: x轴的列名(是否是月度会员, 0 -> 不是会员, 1 -> 是会员)
    # 参数hue意思是: 根据hue的值, 将数据进行分类(False -> 不流失, True -> 流失)
    sns.countplot(data, x='Contract_Month', hue='flag')
    plt.show()


# 3. 定义函数, 用于实现: 逻辑回归模型的训练和评估.
def dm03_逻辑回归模型训练评估():
    # 1. 读取数据.
    data = pd.read_csv('./data/churn.csv')
    # print(data.head(10))
    # 2. 对上述的数据做 热编码处理.
    data = pd.get_dummies(data)
    # 3. 删除列, 因为热编码之后, 会多出一个列, 我们删除掉.
    data.drop(['gender_Male', 'Churn_No'],axis=1,inplace=True)
    # 4. 修改列名.
    data.rename(columns={'Churn_Yes':'flag'},inplace=True)
    print(data.head(10))
    # print(data.flag.head(10))
    # 5. 查看数据集, 从中筛除: 特征列 和 标签列.
    # print(data.head(10))    # 特征列: Contract_Month, PaymentElectronic, internet_other
    # print(data.columns)     # 标签列: flag

    # 6. 拆分训练集和测试集. ['Contract_Month', 'PaymentElectronic', 'internet_other']
    x=data[['Contract_Month', 'PaymentElectronic', 'internet_other']]
    y=data['flag']

    # print(y)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=22)
    # print(y_test, len(y_test))
    # print('-'*88)
    # 7. 创建逻辑回归模型, 并训练.
    estimator = LogisticRegression()
    estimator.fit(x_train, y_train)
    # 8. 模型预测.
    y_predict = estimator.predict(x_test)
    # print(f'预测值为: {y_predict, len(y_predict)}')

    # 9. 模型评估.
    # 9.1 准确率.
    # 真实值, 预测值.
    print(f'准确率：{estimator.score(x_test,y_test)}')
    print(f'准确率：{accuracy_score(y_test,y_predict)}')
    print('-' * 22)
    # todo 混淆矩阵 把预测结果 y_predict 转换成混淆矩阵
        # todo 参1：真实样本，参2：预测样本，参3：样本标签（正例，反例）
    # 1.定义标签
    label = [0, 1]  # 1正例 0反例

    df_label = ['True(正例)', 'False(反例)']
    # cm_A = confusion_matrix(y_test,y_predict)
    cm_A = confusion_matrix(y_test,y_predict,labels=label)
    print(f'混淆矩阵：\n {cm_A}')
    df_A = pd.DataFrame(cm_A,index=df_label,columns=df_label)
    print(f'混淆矩阵对应的DataFrame对象：\n {df_A}')

    # 9.2 精确率.

    print('-' * 22)
    # 9.3 召回率.

    print('-' * 22)
    # 9.4 F1值

    print('-' * 22)
    # 9.5 roc曲线

    print('-' * 22)
    # 9.6 分类评估报告
    # 参数macro avg意思是: 宏平均, 是指: 所有的分类器, 都按照 macro 的方式, 计算平均值/
    # 不考虑样本的权重, 直接平均, 跟样本的数量, 权重无关, 所有特征权重都一样, 适合于 数据集比较平衡的情况.

    # 参数weighted avg意思是: 权重平均, 是指: 所有的分类器, 都按照 weighted 的方式, 计算平均值/
    # 考虑样本的权重, 根据样本的权重, 计算平均值, 适合于 数据集比较不平衡的情况.


# 4. 在main函数中测试.
if __name__ == '__main__':
    # dm01_数据预处理()
    # dm02_会员流失可视化情况()
    dm03_逻辑回归模型训练评估()