"""
案例: 自定义代码, 采用面向对象思维, 模拟: (单向)链表.

回顾:
    链表 由节点组成, 节点 = 数值域(元素域) + 地址域(链接域)

分析:
    节点类: SingleNode
        属性:
            item: 数值域(元素域)
            next: 地址域(链接域)

    链表类: SingleLinkedList
        属性:
            head: 指向头结点
        行为:
            is_empty(self) 链表是否为空
            length(self) 链表长度
            travel(self. ) 遍历整个链表
            add(self, item) 链表头部添加元素
            append(self, item) 链表尾部添加元素
            insert(self, pos, item) 指定位置添加元素
            remove(self, item) 删除节点
            search(self, item) 查找节点是否存在

顺序表 和 链表的总结:
    顺序表:
        更适合 查,改的操作, 相对更节省存储空间, 支持随机读取(索引).
        要求: 空间必须是连续的.
    链表:
        更适合 增,删的操作, 相对更占用存储空间, 不支持随机读取.
        要求: 空间无所谓连不连续, 只要有地儿就行.
    时间复杂度对比:
            操作      顺序表     链表
        查找某个元素    O(1)     O(n)
        头部加/删      O(n)     O(1)
        尾部加/删      O(1)     O(n)
        中间加/删      O(n)     O(n)

"""


# 定义节点类
class SingleNode:
    def __init__(self, item):  # 初始化属性，单项链表属性：节点要有元素和链接域
        self.item = item  # 元素
        self.next = None  # 链接域


class SingleLinkedList:
    def __init__(self, head=None):  # 初始化属性：头节点
        self.head = head

    def is_empty(self):  # 链表是否为空
        return True if self.head == None else False

    def length(self):  # 链表长度
        # todo 思考：遍历链表，首先要从头部遍历，一直到元素为空
        cur = self.head
        # 定义一个变量记录链表的长度
        count = 0
        # 遍历 链表
        while cur is not None:
            count += 1
            # todo cur.next代表下一个元素，cur=cur.next:就是节点后移，此时复制后的cur就已经是下一个元素了
            cur = cur.next
        return count

    def travel(self):  # 遍历整个链表
        if self.is_empty():
            print(f'链表无数据，请先添加:')
        else:
            # 1. 定义变量, 记录: 当前节点. 默认: 从头结点开始.
            cur = self.head
            while cur is not None:
                print(f'链表中的元素：{cur.item}')
                cur = cur.next

    def add(self, item):  # 链表头部添加元素
        # 1.创建一个节点对象
        new_node = SingleNode(item)
        # todo 2.头部添加：那么新的节点的地址值应该只向上一个节点的头部self.head
        new_node.next = self.head
        # 3.设置头结点为：新节点
        self.head = new_node

    def append(self, item):  # 链表尾部添加元素
        # 1.把要添加的节点封装成新元素
        new_node = SingleNode(item)
        # 2.判断节点是否为空
        if self.is_empty():
            self.add(item)
        else:
            # 3 说明节点不为空，就获取链表的最后一个节点，定义变量，记录最后一个节点，默认从头遍历
            cur = self.head
            # 4.循环遍历
            if cur.next is not None:
                # 5.节点后移
                cur = cur.next
            # 把遍历出的最后一个节点的链接域只向新的节点
            cur.next = new_node

    def insert(self, pos, item):  # 指定位置添加元素
        # 1. 如果要插入的位置小于等于0, 就往链表的头部添加.
        if pos <= 0:
            self.add(item)
        # 2. 如果要插入的位置大于等于链表的长度, 就往链表的头部添加.
        elif pos >= self.length():
            self.append(item)
        else:
            # 4. 定义cur变量, 记录: 插入位置前的哪个节点. 默认: 从头结点开始找.
            cur = self.head
            # 5. 定义变量count, 记录: 插入位置前的哪个元素的"索引, 位置"
            count = 0
            # 6. todo 只要 count < pos - 1, 就一直查找.
            # 因为我们写的代码是 cur = cur.next, 所以找到插入位置的 上上个节点, 该节点的next记录的就是: 插入前的那个节点.
            # 大白话, 插入位置为2, 则只要找到0位置的节点即可, 0位置节点的 next(地址域)记录的就是 1节点的地址.
            while count < pos - 1:
                count += 1
                cur = cur.next
            # 7.todo 走到这里, 说明cur就是插入位置前的那个节点, 把item数据封装成新节点, 添加即可.
            # 封装新节点
            new_node = SingleNode(item)
            # todo 设置新节点的地址域为: cur的下个节点
            new_node.next = cur.next
            # todo 设置cur的地址域为: 新节点的地址
            cur.next = new_node

    def remove(self, item):  # 删除节点
        # 1. 定义变量cur, 记录要被删除的元素, 默认从: 头结点开始.
        cur = self.head
        # 2. 定义变量pre(previous), 记录: 删除节点的前1个节点.
        pre = None
        # 3. 循环查找, 只要cur不为空, 就一直遍历.
        while cur.next is not None:
            # 4. 判断当前节点的 数值域(元素域) 和 要被删除的内容是否一致.
            if cur.item == item:
                # 4.1 走这里, 说明 cur 就是要被删除的节点.
                # 5.判断当前节点是否是头结点, 如果是: 直接用 head指向当前节点的链接域(地址域)
                if cur == self.head:
                    # 5.1 走到这里, 说明cur是头结点.
                    self.head = cur.next
                else:
                    # 5.2 走到这里, 说明cur不是头结点, 设置其 前驱节点的地址域(链接域) 指向 其后继节点即可.
                    pre.next = cur.next
                    # 核心: 删完记得break.
                break
            else:
                # 4.2 走到这, 说明 cur 不是要被删除的节点, 继续往后找.
                pre = cur
                cur = cur.next

    def search(self, item):
        # 1. 定义变量cur, 记录当前节点, 默认: 从头结点开始.
        cur = self.head
        # 2. 循环判断, 只要cur不为空, 就一直查找.
        while cur is not None:
            # 3. 判断当前节点的元素域是否等于要查找的元素值.
            if cur.item == item:
                return f'链表中存在元素:{item}'
            # 没有找到, cur就往后移动.
            cur = cur.next
        # 走到这里, while循环执行完毕, 说明没有找到.
        return f'链表中不存在元素:{item}'


if __name__ == '__main__':
    # 实例化一个节点
    node=SingleNode('鲁班-射手')
    print(f'节点的元素域：{node.item},节点的链接域：{node.next}')
    # 实例化链表
    sll = SingleLinkedList(node)
    # 链表添加一个节点
    sll.add('王者荣耀')
    # sll.remove('王者荣耀')
    print(f'链表的头结点元素:{sll.head.item},链表的链接域:{sll.head.next}')
    sll.append('伽罗-射手')
    sll.insert(1,'上官婉儿-法师')
    result = sll.is_empty()
    print(f'链表是否为空：{result}')
    lenth=sll.length()
    print(f'链表的长度：{lenth}')
    sll.travel()
    print('-'*32)

    sear_result=sll.search('狄仁杰-射手')
    print(f'链表中：{sear_result}')



