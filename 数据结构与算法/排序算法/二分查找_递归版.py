"""
二分查找解释:
    概述:
        二分查找也叫折半查找, 是一种效率比较高的 检索算法.
    前提:
        数据必须是有序的, 升序, 降序均可.
    原理: 假设数据是升序的.
        1. 获取中间索引的值, 然后和要查找的值进行比较.
        2. 如果相等, 则直接返回True即可.
        3. 如果要查找的值比 中间索引的值小, 就去 中值左 进行查找.
        3. 如果要查找的值比 中间索引的值大, 就去 中值右 进行查找.
"""
# step1: 定义函数, 表示二分查找.
def binary_search(my_list, item):
    """
    自定义的代码, 实现: 二分查找.
    :param my_list: 有序的数据 列表
    :param item: 要查找的元素.
    :return: True-> 找到了, False->没找到.
    """
    # 1. 获取列表的长度.
    n = len(my_list)
    # 2. 判断如果列表的长度为0, 则: return False
    if n == 0:
        return False
    # 3. 定义变量, 记录: 中间索引.
    mid = n // 2
    # 4. 判断要查找的值 和 中间索引值 的关系.
    if my_list[mid] == item:
        return True
        # return f'列表中存在元素：{item}'  todo 将返回值从字符串改为布尔值 True 或 False，以符合函数文档中的描述。
    elif item < my_list[mid]:
        # 要查找的值 小于 中间索引的值, 去 中值左 进行查找.
        return binary_search(my_list[:mid], item)
    else:
        # 要查找的值 大于 中间索引的值, 去 中值右 进行查找.
        return binary_search(my_list[mid + 1:], item)
    # 5. 走到这里, 说明整个列表查找完毕还没有找到, 返回False即可.
    return False
    # return f'列表中没有元素：{item}'  todo 将返回值从字符串改为布尔值 True 或 False，以符合函数文档中的描述。





# step2: 测试代码.
my_list = [2, 3, 5, 9, 13, 27, 31, 39, 55, 66, 99]
# my_list = []
print(binary_search(my_list, 31))

