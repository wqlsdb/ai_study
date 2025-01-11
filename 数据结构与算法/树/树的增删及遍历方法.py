'''
案例:
    自定义代码 模拟 二叉树.

分析流程:
    定义节点类Node,
        属性: item(元素域), lchild(左子树), rchild(右子树)
    定义二叉树类BinaryTree:
        属性: root    -> 充当根节点.
        行为:
            add()    -> 往二叉树中, 添加元素的.
            breadth_travel() -> 广度优先, 逐层遍历.
'''
# 1. 定义节点类Node
class Node:
    # 初始化属性
    def __init__(self, item):
        self.item = item     # 元素域
        self.lchild = None   # 左子树
        self.rchild = None   # 右子树


# 2. 定义二叉树类BinaryTree
class BinaryTree:
    # 初始化属性.
    def __init__(self, root=None):
        self.root = root        # 二叉树的根节点.

    # 定义函数, 实现往二叉树中, 添加元素.
    def add(self, item):
        # 1. 判断根节点是否为空.
        if self.root is None:
            # 走这里, 根节点为空, 把 item封装成节点, 作为根节点即可.
            self.root = Node(item)
            # 添加完毕后, 记得结束 添加动作即可.
            return
        # 2. 定义列表 -> 充当队列(先进先出), 用于记录: 二叉树中已经存在的节点.
        # todo 思考 通过什么方式有序的去拿到上一轮遍历过的节点呢？
        # todo:添加元素就需要遍历二叉树的每个节点，一直遍历到空的子树，每遍历一个一遍后都需要去拿上一次遍历不为空的
        #   子树，通过什么方式有序的去拿到上一轮遍历过的节点呢？只能通过队列，从队列中有序取节点，判断节点的子树是否有空
        queue = []
        # 3. 把根节点添加到 队列 中.
        queue.append(self.root)
        # 4. 循环查找队列中的元素, 直至找到: 某个节点的左子树, 或者右子树为空的情况.
        while True:
            # 5. 获取(弹出) 队列中的第1个元素, 即: 从 根节点开始.
            node = queue.pop(0)
            # 6. 判断节点的左子树是否为空.
            if node.lchild is None:
                # 6.1 走到这里, 说明当前节点的左子树为空, 把新元素添加到这里即可, 并记得返回.
                node.lchild = Node(item)
                return
            else:
                # 6.2 走到这里, 说明当前节点的左子树不为空, 就把左子树添加到 队列中.
                queue.append(node.lchild)
            # 7. 走到这里, 说明当前节点的左子树不为空, 继续判断右子树是否为空.
            if node.rchild is None:
                node.rchild = Node(item)
                return
            else:
                queue.append(node.rchild)


    # 定义函数, 实现广度优先, 逐层遍历.
    def breadth_travel(self):
        # 1. 判断根节点是否为空, 如果为空, 直接返回即可.
        if self.root is None:
            return
        # 2. 走这里, 说明根节点不为空, 我们逐个获取节点, 打印信息即可.
        queue=[]
        queue.append(self.root)
        # 3. 开始循环, 直至队列为空.
        while len(queue)>0:
            # 4. 获取(弹出) 队列中的第1个元素, 即: 从 根节点开始.
            node=queue.pop(0)
            # 5. 打印当前节点的信息(元素域)
            print(f'节点元素：{node.item}')
        # 6. 判断当前节点的左子树是否为空, 不为空就添加到队列中.
            if node.lchild != None:
                queue.append(node.lchild)
        # 7. 判断当前节点的右子树是否为空, 不为空就添加到队列中.
            if node.rchild != None:
                queue.append(node.rchild)


    # 定义函数, 实现深度优先 -> 先序(前序), 根左右
    def preorder_travel(self, root):    # root是传入的节点
        if root is not None:
            print(f'节点元素：{root.item}',end=' ')
            self.preorder_travel(root.lchild)
            self.preorder_travel(root.rchild)

    # 定义函数, 实现深度优先 -> 中序, 左根右
    def inorder_travel(self, root):    # root是传入的节点
        pass

    # 定义函数, 实现深度优先 -> 后序, 左右根
    def postorder_travel(self, root):    # root是传入的节点
        pass

    # 3. 定义测试方法.
def dm01_测试节点类和二叉树类():
    # 1.实例化根节点
    n1 = Node('王者荣耀')
    # 2. 打印节点的: 元素域, 左子树, 右子树.
    print(f'元素域:{n1.item},左子树:{n1.lchild},右子树:{n1.rchild}')
    # 3. 创建二叉树对象.
    bt = BinaryTree(n1)
    # 4. 打印二叉树对象，二叉树根节点，左子树，右子树
    print(
        f'二叉树对象:{bt},\n二叉树根节点:{bt.root},\n二叉树根节点的元素域:{bt.root.item},二叉树根的左子树:{bt.root.lchild},二叉树根的右子树:{bt.root.rchild}')

def dm02_测试队列添加和弹出队头元素():
        # 1. 创建队列(先进先出), 其实: 还是列表模拟的.
        queue = []

        # 2. 演示队列添加元素.
        queue.append('a')
        queue.append('b')
        queue.append('c')

        # 3. 弹出队头元素.
        print(queue.pop(0))  # ['a', 'b', 'c'] -> a
        print(queue.pop(0))  # ['b', 'c'] -> b
        print(queue.pop(0))  # ['c'] -> c

        # 4. 打印队列.
        # print(queue)        # ['a', 'b', 'c']

def dm03_广度优先遍历():
    # 1. 定义二叉树.
    bt = BinaryTree()
    # 2. 往二叉树中添加元素.
    bt.add('a')
    bt.add('b')
    bt.add('c')
    bt.add('d')
    bt.add('e')
    bt.add('f')
    bt.add('g')
    bt.add('h')
    bt.add('i')
    bt.add('j')
    # 3. 遍历二叉树 -> 广度优先.
    bt.breadth_travel()

def dm04_深度优先遍历():
    # 1. 定义二叉树
    bt = BinaryTree()
    # 2. 添加元素.
    bt.add(0)
    bt.add(1)
    bt.add(2)
    bt.add(3)
    bt.add(4)
    bt.add(5)
    bt.add(6)
    bt.add(7)
    bt.add(8)
    bt.add(9)
    # 3. 打印遍历结果.
    print('先序遍历(根左右): ')
    bt.preorder_travel(bt.root) # 传入根节点, 即: 从根节点开始遍历.


if __name__ == '__main__':
    # dm01_测试节点类和二叉树类()
    # dm03_广度优先遍历()
    dm04_深度优先遍历()