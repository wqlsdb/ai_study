# 1.定义节点类：节点的属性：元素，链接域（指针）
class Node:
    def __init__(self,item):
        self.item=item
        self.next=None

# 定义一个链表
class SingleLinkedList:
    def __init__(self, head=None):  #定义一个头结点
        self.head = head

    def add(self, item):
        new_node = Node(item)
        new_node.next = self.head
        self.head = new_node

    def remove(self, item):
        cur = self.head
        pre = None
        while cur.next is not None:
            if cur.item == item:
                if cur == self.head:
                    self.head = cur.next
                else:
                    pre.next = cur.next
                break
            else:
                pre = cur
                cur = cur.next

    def travel(self):
        cur = self.head
        while cur is not None:
            print(f'元素：{cur.item}')
            cur = cur.next

if __name__ == '__main__':
    n1 = Node('公孙离')
    sll = SingleLinkedList(n1)
    print(f'链表的头结点元素：{sll.head.item}')
    sll.add('鲁班')
    sll.add('后裔')
    sll.add('伽罗')
    sll.add('鲁班')
    sll.remove('鲁班')
    sll.travel()
