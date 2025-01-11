import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB  # 多项分布朴素贝叶斯

# 1. 读取数据.
data = pd.read_csv('./data/书籍评价.csv', encoding='gbk')
# print(f'查看数据集：\n{data.head(5)}')
# 2. 数据预处理.
# 2.1 增加1列 labels充当标签列, 即: 0 -> 差评, 1 -> 好评.
# data['labels'] = data['labels'].map(lambda x: 1 if x == '好评' else 0)
data['labels'] = np.where(data['评价'] == '差评', 0, 1)
# print(f'查看数据集：\n{data.head(5)}')
# 2.2 把上述的 labels 充当: y列(标签列)
y = data['labels']
# print(y)

# print(data.head(5))

# 2.3
# 2.3 对用户录入内容进行切次
# comment_list = [jieba.lcut(line) for line in data['内容']]
# print(f'切词后的结果：\n{comment_list}')

comment_list = ['.'.join(jieba.lcut(line)) for line in data['内容']]
print(f'切词后的结果：\n{comment_list}')

# 2.5 把上述切词后的内容, 无效词给过滤掉, 例如: , 的 等之类的, 都是无效词.
# 读取 停用词文件, 获取到所有的停用词(对区分 好评 和 差评 没有帮助的词 -> 停用词)
with open('./data/stopwords.txt','r',encoding='utf-8') as f:
    # 一次读取所有行, 格式: ['行1\n', '行2\n', '行3\n'...]
    lines = f.readlines()
    # print(lines)
    # 删除上述每行后的\n, 然后把所有数据放到1个列表中.
    stopword_list = [line.strip() for line in lines]
    print(stopword_list)
    # 对上述列表中的停用词进行去重.
    stopword_list = list(set(stopword_list))
    # print(len(stopword_list))       # 1711


# 3. 向量化处理, 把切词后的内容, 无效词给过滤掉, 例如: , 的 等之类的, 都是无效词.
# 参1: 停用词列表
transfer = CountVectorizer(stop_words=stopword_list)
# print(transfer)
# 具体的向量化过程, 然后转成列表, 就可以知道: 共多少个切词, 每个语句中都出现了哪些切词.
# 例如: []
x = transfer.fit_transform(comment_list).toarray()
# print(x.shape)
# print(x)

# 查看上述37个切词的总的内容.
# print(transfer.get_feature_names_out())

# 4. 切分训练集 和 测试集, 共12条评论, 我们让前10条当训练集, 后3条当测试集.
x_train = x[:10]
y_train = y[:10]
x_test = x[10:]
y_test = y[10:]

# 5. 创建模型对象(贝叶斯), 进行训练.
estimator = MultinomialNB()
estimator.fit(x_train, y_train)

# 6. 模型预测.
y_predict = estimator.predict(x_test)
print(f'预测值:{y_predict}')

# 7. 模型评估.
print(f'准确率:{estimator.score(x_test, y_test)}')


