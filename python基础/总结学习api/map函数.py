'''
map函数的一种简化设计
    只需要告诉map要处理的函数及要处理的数据，map会自动将数据中的每个元素传递给函数
'''
# eg.1
numbers = [1, 2, 3, 4, 5]
squared=list(map(lambda x:x**2,numbers))
print(squared)

words = ['apple', 'banana', 'cherry']
# eg.2
def to_upper(word):
    return word.upper()
# todo map函数的一种简化设计只需要告诉map要处理的函数及要处理的数据，map会自动将数据中的每个元素传递给函数
upper_words = list(map(to_upper, words))
print(upper_words)

# eg.3
lengths = list(map(len,words))
print(lengths)
