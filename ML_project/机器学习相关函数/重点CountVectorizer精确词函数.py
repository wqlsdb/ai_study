# 1. 定义你的停用词列表
import jieba

stopword_list = ['的', '是', '在', '和', '了', '上', '也', '有', '与', '对']

# 3. 输入文本数据
corpus = [
    '我喜欢学习编程因为它很有趣',
    '喜欢喜欢喜欢喜欢喜欢喜欢',
    '编程是一项非常有用的技能',
    '喜欢学习编程需要耐心和时间'
]
# print(corpus)
comment_list = []
for i in range(len(corpus)):
    corpus[i] = ' '.join(jieba.lcut(corpus[i]))
    # print(corpus[i])
    comment_list.append(corpus[i])
print(comment_list)
comment_list = list(set(comment_list))
# 2. 创建CountVectorizer实例，并传入停用词列表
from sklearn.feature_extraction.text import CountVectorizer

transfer = CountVectorizer(stop_words=stopword_list)

# 4. 使用fit_transfer方法来学习词汇表并返回文档的词频矩阵
X = transfer.fit_transform(comment_list)
# print(X)
# 5.输出词汇表
print('词汇表：', transfer.get_feature_names_out())
# # 6.输出文档-词频矩阵
print('文档-词频矩阵：\n', X.toarray())
