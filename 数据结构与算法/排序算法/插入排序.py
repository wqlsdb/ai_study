"""
排序算法的稳定性介绍:
    概述:
        排序算法 = 把一串数据按照升序 或者 降序进行排列的 方式, 方法, 思维.
    分类:
        稳定排序算法:
            排序后, 相同元素的相对位置 不发生改变.
        不稳定排序算法:
            排序后, 相同元素的相对位置 发生改变.
    举例:
        不稳定排序算法: 选择排序, 快速排序...
        稳定排序算法:   冒泡排序, 插入排序...

插入排序介绍:
    原理:
        把列表分成有序和无序的两部分, 每次从无序数据中拿到第1个元素, 然后放到对应有序列表的位置即可.
    核心:
        1. 比较的总轮数.      列表的长度 - 1, 即: i的值 -> 1, 2, 3, 4
        2. 每轮比较的总次数.   i ~ 0
        3. 谁和谁比较.        索引j 和 j - 1 的元素比较.
    分析流程, 假设元素个数为5个, 具体如下:  [5, 3, 4, 7, 2]  -> 长度为: 5
        比较的轮数, i        每轮具体哪两个元素比较:        每轮比较的次数, j
        第1轮,索引:1            1和0                          1
        第2轮,索引:2            2和1, 2和0                    2
        第3轮,索引:3            3和2, 3和1, 3和0              3
        第4轮,索引:4            4和3, 4和2, 4和1, 4和0        4
总结:
    插入排序属于 稳定排序算法, 最优时间复杂度: O(n), 最坏时间复杂度: O(n²)
"""


class Insert_sort_Alogrithm:

    def insert_sort(self, my_list):
        n = len(my_list)
        for i in range(n):
            for j in range(i, 0, -1):
                if my_list[j] < my_list[j - 1]:
                    my_list[j], my_list[j - 1] = my_list[j - 1], my_list[j]
                else:
                    break
        return my_list


if __name__ == '__main__':
    my_list = [5, 8, 3, 6, 4, 7, 2]
    isa = Insert_sort_Alogrithm()
    result = isa.insert_sort(my_list)
    print(result)
