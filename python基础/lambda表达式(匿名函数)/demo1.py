'''
lambda 参数:操作(参数)
lambda [arg1[,arg2,arg3....argN]]:expression
'''
# eg.1
key = lambda x: x ** 2
print(key(2))
# eg.2
# # 使用 lambda 创建一个加法函数
add = lambda x,y : x+y
print(add(2,6))
# eg.3
# todo 定义一个列表嵌套元组
list_tuple = [(1, 'one'), (2, 'two'), (3, 'three'), (4, 'four')]
# todo 定义一个匿名函数：作用用于返回稍后遍历元组中的第二个元素
key = lambda item: list_tuple[1]

sort_result = sorted(list_tuple, key=lambda item: item[1])
print(sort_result)

for my_tuple in list_tuple:
    result = key(my_tuple)
    print(f'Tuple:{my_tuple},第二个元素：{result}')

# eg.4
def my_calculate(a,b,func):
    return func(a,b)
print(my_calculate(2,6,lambda a,b:a*b))
print(my_calculate(2,6,lambda c,d:c+d))
print(my_calculate(10,2,lambda e,f:e-f))