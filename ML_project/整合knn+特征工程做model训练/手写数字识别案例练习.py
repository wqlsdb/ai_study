import matplotlib.pyplot as plt  # 绘图.
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib  # 模型保存的包


# 需求1: 定义函数, 接收索引, 将该行的 手写数字 -> 识别为: 图片, 并绘制出来.
def dm01_show_digit(idx):
    # 1.读取csv文件，获取到DF对象
    data = pd.read_csv('./data/手写数字识别.csv')
    # 2.判断用户传入的索引是否合法化
    if idx < 0 or idx > len(data):
        print('传入的索引有误, 程序结束!')
        return

    # 3. 走这里, 说明索引没有问题, 查看下所有的数据集.
    x = data.iloc[:, 1:]  # 获取所有的训练集x_train，除了第一列索引
    y = data.iloc[:, 0]  # 获取所有标签y_train，第一列的索引
    # print(y)
    # 查看下 每个数字一共有多少个, 即: 0数字多少个, 1数字多少个, 2数字多少个
    print(f'数字的种类：{Counter(y)}')
    print(f'像素的形状：{x.shape}')

    # 4. 根据传入的索引,获取到该行的数据
    print(f'您传入的索引，对应的数字是：{y[idx]}')

    # 5.绘制图片
    # 5.1 把图片的像素点，转换为：28*28的图片
    digit = x.iloc[idx].values.reshape(28, 28)
    # print(digit)
    # 5.2 绘制图片
    # plt.figure(figure=(2, 2), dpi=14)
    # todo 图片必须是28*28 且是28字节
    plt.figure(figsize=(1, 1), dpi=28)
    plt.tight_layout()
    plt.imshow(digit, cmap='gray')
    plt.axis('off')
    plt.savefig('./data/demo061')
    plt.show()


# 需求2: 定义函数, 使用KNN算法, 用于: 识别手写数字.  保存模型.
def dem02_train_model():
    # 1.读取csv文件，获取数据
    data = pd.read_csv('./data/手写数字识别.csv')
    # 2.数据预处理，获取特征和标签
    x = data.iloc[:, 1:]  # 特征（像素点）
    y = data.iloc[:, 0]  # 标签
    # 参数: stratify 参考y轴数据分布(对标签值做匀散分布，防止多取漏取)，划分 训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=22, stratify=y
    )

    # 3.特征工程
    # todo:fit_transform(): 针对于训练集的, 即: 训练 + 转换(标准化)
    # todo:fit(): 针对于测试集的, 即: 只有转换(标准化)
    # 特征的预处理，因为：x 是像素，范围是：[0,255],y是数值，范围是：[0,9]
    # todo x_train是一个dataFrame,做数值运算每个元素都会除255
    x_train = x_train / 255   # 准确率：0.770
    x_test = x_test / 255
    # 3.1 创建标准化对象
    # transfer = StandardScaler()
    # x_train = transfer.fit_transform(x_train)  # 准确率：0.824

    # 4.模型训练。
    estimator = KNeighborsClassifier(n_neighbors=9)
    estimator.fit(x_train, y_train)

    # 5.模型评估。
    print(f'准确率：{estimator.score(x_test, y_test)}')

    # 6.模型保存
    joblib.dump(estimator, './model/knn.pkl')  # pickle: Pandas的独有的文件格式.


def dm03_use_model():
    # 1.读取图片，绘制图片，看看图片是哪个数字
    img = plt.imread('./data/demo061.png')
    # print(img)
    plt.imshow(img, cmap='gray')
    plt.show()

    # 2.读取模型，获取模型对象
    knn = joblib.load('./model/knn.pkl')
    print(knn)

    # 3. 模型预测 reshape(1,-1),从一开始，到-1（无穷大结束）
    y_predict = knn.predict(img.reshape(1, -1))
    print(f'预测结果为：{y_predict}')


def dm04_use_model():
    # 1.读取图片，绘制图片，看看图片是哪个数字
    img = plt.imread('./data/demo061.png')
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = img[..., :3].dot([0.2989, 0.5870, 0.1140])
    elif len(img.shape) == 3 and img.shape[2] == 3:
        img = img.dot([0.2989, 0.5870, 0.1140])
    plt.imshow(img, cmap='gray')
    plt.show()
    # 2.读取模型，获取模型对象
    knn = joblib.load('./model/knn.pkl')
    print(knn)

    # 3. 模型预测 reshape(1,-1),从一开始，到-1（无穷大结束）
    img_flat = img.flatten().reshape(1, -1)
    y_predict = knn.predict(img_flat)
    print(f'预测结果为：{y_predict}')


if __name__ == '__main__':
    # dm01_show_digit(20)
    # dem02_train_model()
    dm03_use_model()
    # dm04_use_model()
