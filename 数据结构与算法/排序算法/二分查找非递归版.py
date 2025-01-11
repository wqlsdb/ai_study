'''
思路：这种方法就是先拿到列表的开头和结尾索引，然后折中取中间值，判断输入的值和列表中的中间值，
    小于中间值则就是列表的左边，同时可以忽略中间值右边的数据必定不符合规则，所以可把end赋值给中间值，然后再次循环
'''
# 根据需求：方法需要有要被查的列表，以及你要查的元素
def binary_search(my_list, item):
    start = 0   #
    end = len(my_list) - 1
    while start <= end:
        mid = (start + end) // 2
        if item == my_list[mid]:
            return True
        elif item < my_list[mid]:
            end = mid - 1
        else:
            start = mid + 1

my_list = [2, 3, 5, 9, 13, 27, 31, 39, 55, 66, 99]
# my_list = []
print(binary_search(my_list, 39))
