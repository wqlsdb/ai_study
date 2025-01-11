class SingleNode:
    def __init__(self, item):  # 初始化属性，单项链表属性：节点要有元素和链接域
        self.item = item  # 元素
        self.next = None  # 链接域


class SingleLinkedList:
    def __init__(self, head=None):  # 初始化属性：头节点
        self.head = head

    def is_empty(self):  # 判断链表是否为空
        return True if self.head == None else False

    def length(self):  # 计算链表长度
        cur = self.head
        count = 0
        while cur != None:  # 遍历链表
            count += 1
            cur = cur.next
        return count

    def travel(self):  # 遍历整个链表并打印每个节点的元素
        cur = self.head
        while cur != None:  # 遍历链表
            print(cur.item)
            cur = cur.next

    def add(self, item):  # 在链表头部添加元素
        node = SingleNode(item)  # 创建新节点
        node.next = self.head  # 新节点指向原头节点
        self.head = node  # 更新头节点为新节点

    def append(self, item):  # 在链表尾部添加元素
        node = SingleNode(item)  # 创建新节点
        if self.is_empty():  # 如果链表为空，则直接将头节点指向新节点
            self.head = node
        else:
            cur = self.head
            while cur.next != None:  # 遍历到链表尾部
                cur = cur.next
            cur.next = node  # 将尾节点指向新节点

    def insert(self, pos, item):  # 在指定位置插入元素
        if pos <= 0:
            self.add(item)  # 如果位置小于等于0，在头部插入
        elif pos > (self.length() - 1):
            self.append(item)  # 如果位置大于链表长度减1，在尾部插入
        else:
            pre = self.head
            count = 0
            while count < (pos - 1):  # 找到要插入位置的前一个节点
                count += 1
                pre = pre.next
            node = SingleNode(item)  # 创建新节点
            node.next = pre.next  # 新节点指向原位置节点
            pre.next = node  # 前一个节点指向新节点

    def remove(self, item):  # 删除指定元素的节点
        cur = self.head
        pre = None
        while cur != None:  # 遍历链表
            if cur.item == item:
                if not pre:
                    self.head = cur.next  # 如果删除的是头节点，更新头节点
                else:
                    pre.next = cur.next  # 否则，前一个节点指向当前节点的下一个节点
                break
            else:
                pre = cur
                cur = cur.next

    def search(self, item):  # 查找指定元素是否存在
        cur = self.head
        while cur != None:  # 遍历链表
            if cur.item == item:
                return True  # 找到返回True
            cur = cur.next
        return False  # 未找到返回False


if __name__ == '__main__':
    # 实例化一个节点
    node = SingleNode('鲁班-射手')
    print(f'节点的元素域：{node.item}, 节点的链接域：{node.next}')

    # 创建一个空的单链表
    ll = SingleLinkedList()
    print(ll.is_empty())  # 输出: True
    print(ll.length())  # 输出: 0

    # 添加元素到链表中
    ll.add('后羿-射手')  # 头部添加
    ll.append('孙尚香-射手')  # 尾部添加
    ll.insert(1, '黄忠-射手')  # 指定位置添加

    # 遍历链表
    ll.travel()  # 输出: 后羿-射手 黄忠-射手 鲁班-射手 孙尚香-射手

    # 查找元素
    print(ll.search('黄忠-射手'))  # 输出: True
    print(ll.search('李白-战士'))  # 输出: False

    # 删除元素
    ll.remove('鲁班-射手')
    ll.travel()  # 输出: 后羿-射手 黄忠-射手 孙尚香-射手