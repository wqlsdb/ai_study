class SingleNode:
    def __init__(self,item):
        self.item=item
        self.next=None


class SingleLinkedList:
    def __init__(self,head=None):
        self.head=head

    def add(self, item):
        n1 = SingleNode(item)
        n1.next = self.head
        self.head = n1

    def travel(self):
        cur = self.head
        while cur is not None:
            print(f'节点的元素：{cur.item}')
            cur = cur.next

    def length(self):
        cur = self.head
        count = 0
        while cur is not None:
            count += 1
            cur = cur.next
        return count

    def get_index(self, index):
        if index < 0 or index > self.length():
            return -1
        cur = self.head
        count = 0
        while count < index:
            count += 1
            cur = cur.next
        return cur.item


if __name__ == '__main__':
    n1 = SingleNode('公孙离')
    sll = SingleLinkedList(n1)
    # print(f'链表的头结点元素：{sll.head.item}')
    sll.add('鲁班')
    sll.add('后裔')
    sll.add('伽罗')
    sll.add('鲁班')
    result = sll.get_index(2)
    print(result)
