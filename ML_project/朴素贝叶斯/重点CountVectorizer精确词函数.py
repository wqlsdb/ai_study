# 1. 定义你的停用词列表
stopword_list = ['的', '是', '在', '和', '了', '上', '也', '有', '与', '对']
# 2. 创建CountVectorizer实例，并传入停用词列表
from sklearn.feature_extraction.text import CountVectorizer
transfer = CountVectorizer(stop_words=stopword_list)
# 3. 输入文本数据
corpus = [
    '我喜欢学习编程因为它很有趣',
    '编程是一项非常有用的技能',
    '学习编程需要耐心和时间'
]

# 4. 使用fit_transfer方法来学习词汇表并返回文档的词频矩阵
X = transfer.fit_transform(corpus)
# 5.输出词汇表
print('词汇表：',transfer.get_feature_names_out())
# 6.输出文档-词频矩阵
print('文档-词频矩阵：\n',X.toarray())
