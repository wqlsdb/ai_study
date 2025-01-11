import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import Counter
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

def dm01_数据预处理():
    data = pd.read_csv('./data/红酒品质分类.csv')
    # 2.抽取特征和标签
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1] - 3 #最后一行标签，原始范围【3到8】
    # 4. 查看数据分布情况，发现标签不均衡
    # print(Counter(y))
    # print(x.shape, y.shape)
    # print(y.head(10))
    # 5. 拆分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22,stratify=y)


    # 打印拆分后的 训练集和标签分布情况
    # print(Counter(y_train))
    # print(Counter(y_test))

    # 6.
    # print(pd.concat([x_train, y_train], axis=1))
    # pd.concat([x_train, y_train], axis=1).to_csv('./data/train.csv', index=False)
    # pd.concat([x_test, y_test], axis=1).to_csv('./data/test.csv',index=False)

# 2.定义函数，实现模型训练
def dm02_模型训练():
    train_data = pd.read_csv('./data/train.csv')
    test_data = pd.read_csv('./data/test.csv')
    # 2. 数据预处理，抽取特征和标签
    x_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    x_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]
    # 扩展：查看数据分布情况
    print(Counter(y_train))
    # todo:平衡权重问题
    from sklearn.utils import class_weight
    class_weights = class_weight.compute_sample_weight('balanced', y=y_train)
    # 3. 模型训练
    # 3.1 创建XGBoost模型，极限梯度
    # xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, min_child_weight=1, gamma=0, subsample=0.8,)
    estimator = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, objective='multi:softmax')
    estimator.fit(x_train, y_train)
    # 5. 模型预测
    y_pred = estimator.predict(x_test)
    print(f'预测结果为：{y_pred}')
    # 7. 模型保存
    joblib.dump(estimator, './model/xgb.pkl')

# 定义函数，实现XGBoost模型预测
def dm03_模型预测():
    train_data = pd.read_csv('./data/train.csv')
    test_data = pd.read_csv('./data/test.csv')
    # 2. 数据预处理，抽取特征和标签
    x_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    x_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]
    # 扩展：查看数据分布情况
    # print(Counter(y_train))
    # 3.分类采样，目的：避免过拟合
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=22)
    # 4. 读取模型对象
    estimator = joblib.load('./model/xgb.pkl')
    # 5. 定义超参字典
    param_dict = {'n_estimators':[100,200,300,400,500],'learning_rate':[0.1,0.2,0.3,0.4,0.5],'max_depth':[3,4,5,6,7]}
    # 6. 模型调优
    gs_model = GridSearchCV(estimator, param_grid=param_dict, cv=skf,n_jobs=-1)
    # 7.模型训练
    gs_model.fit(x_train, y_train)
    # 8.模型预测
    y_pred = gs_model.predict(x_test)
    print(f'预测结果为：{y_pred}')
    # 9. 打印最优组合
    print(f'最优参数组合为：{gs_model.best_params_}')
    print(f'最优模型得分为：{gs_model.best_score_}')
    print(f'模型准确率：{gs_model.score(x_test, y_test)}')

if __name__ == '__main__':
    # dm01_数据预处理()
    # dm02_模型训练()
    dm03_模型预测()