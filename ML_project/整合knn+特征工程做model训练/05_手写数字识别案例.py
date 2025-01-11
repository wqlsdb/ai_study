# 导入工具包
import joblib
from sklearn.datasets import load_iris  # 加载鸢尾花测试集的.
from sklearn.model_selection import train_test_split, GridSearchCV  # 分割训练集和测试集的,  网格搜索 + 交叉验证.
from sklearn.preprocessing import StandardScaler  # 数据标准化的
from sklearn.neighbors import KNeighborsClassifier  # KNN算法 分类对象
from sklearn.metrics import accuracy_score  # 模型评估的, 计算模型预测的准确率
import matplotlib.pyplot as plt     # 绘图.
import pandas as pd
from collections import Counter


# 需求1: 定义函数, 接收索引, 将该行的 手写数字 -> 识别为: 图片, 并绘制出来.
def dm01_show_digit(idx):
    # 1. 读取csv文件, 获取到df对象.
    data = pd.read_csv('./data/手写数字识别.csv')
    # 2. 判断用户传入的索引是否合法.
    if idx < 0 or idx >= len(data):
        print('传入的索引有误, 程序结束!')
        return
    # 3. 走这里, 说明索引没有问题, 查看下所有的数据集.
    x = data.iloc[:, 1:]    # 获取到所有的 像素点.
    y = data.iloc[:, 0]     # 获取到所有的 标签.
    # print(x)
    # print(y)
    # 查看下 每个数字一共有多少个, 即: 0数字多少个, 1数字多少个, 2数字多少个, ...
    print(f'数字的种类: {Counter(y)}')   # Counter({1: 4684, 7: 4401, 3: 4351, 9: 4188, 2: 4177, 6: 4137, 0: 4132, 4: 4072, 8: 4063, 5: 3795})
    print(f'像素的形状: {x.shape}')      # (42000, 784)

    # 4. 根据传入的索引, 获取到该行的数据.
    print(f'您传入的索引, 对应的数字是: {y[idx]}')

    # 5. 绘制图片.
    # 5.1 把图片的像素点, 转为: 28 * 28的图片.
    digit = x.iloc[idx].values.reshape(28, 28)
    # 5.2 绘制图片.
    plt.figure(figure=(2, 2), dpi=14)
    plt.imshow(digit, cmap='gray')  # 灰度图
    plt.axis('off') # 关闭坐标.
    plt.savefig('./data/demo3.png')
    plt.show()


# 需求2: 定义函数, 使用KNN算法, 用于: 识别手写数字.  保存模型.
def dm02_train_model():
    # 1. 读取csv文件, 获取数据.
    data = pd.read_csv('./data/手写数字识别.csv')
    # 2. 数据预处理.
    x = data.iloc[:, 1:]    # 像素点(特征)
    y = data.iloc[:, 0]     # 标签.
    # 参数: stratify 参考y轴数据的分布, 划分 训练集和测试集.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=22, stratify=y)

    # 3. 特征工程.
    # 特征的预处理, 因为: x是像素, 范围是: [0, 255], y是数值, 范围是: [0, 9]
    # x_train_new = transfer.fit_transform(x_train)
    # todo x_train是一个dataFrame,做数值运算每个元素都会除255
    x_train = x_train / 255

    # 4. 模型训练.
    estimator = KNeighborsClassifier(n_neighbors=9)
    estimator.fit(x_train, y_train)

    # 5. 模型评估.
    print(f'准确率: {estimator.score(x_test, y_test)}')

    # 6. 模型保存.
    joblib.dump(estimator, './model/knn.pkl')   # pickle: Pandas的独有的文件格式.

# 需求3: 定义函数, 使用KNN算法, 用于: 识别手写数字. 使用模型.
def dm03_use_model():
    # 1. 读取图片, 绘制图片, 看看图片是谁.
    img = plt.imread('./data/demo.png')
    # print(img)  #  [[像素点1, 像素点2, 点3... 点28], [像素点1, 像素点2, 点3... 点28], ...]  # 28 * 28
    plt.imshow(img, cmap='gray')    # 灰度图
    plt.show()

    # 2. 读取模型, 获取模型对象.
    knn = joblib.load('./model/knn.pkl')
    # print(knn)

    # 3. 模型预测.
    y_predict = knn.predict(img.reshape(1, -1))  # [像素点1, 点2... 点784]     # 1行, -1: 能转多少列, 转多少列.
    # y_predict = knn.predict(img.reshape(1, 784))  # [像素点1, 点2... 点784]      # 1行, 784列
    print(f'预测结果为: {y_predict}')



# 5. 在main函数中测试.
if __name__ == '__main__':
    # dm01_show_digit(20)
    # dm02_train_model()
    dm03_use_model()
