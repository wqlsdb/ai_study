'''
斐波那契数，通常用 F(n) 表示，形成的序列称为 斐波那契数列 。
该数列由 0 和 1 开始，后面的每一项数字都是前面两项数字的和。
也就是：
    F(0) = 0，F(1) = 1 F(n) = F(n - 1) + F(n - 2)，
    其中 n > 1 给你n ，请计算 F(n) 。

示例 1：输入：2，输出：1
    解释：F(2)=F(1)+F(0)=1+0=1
示例 2：输入：3，输出：1
    解释：F(n) = F(n - 1) + F(n - 2)，将参数3带入函数得：F(3)=F(3-1)+F(3-2)
    结果为：F(3)=F(2)+F(1)
    根据递归思想：已知：F(2)=1，F(1)=1
    所以：F(3)=2

'''
import sys
import time


# todo 思路1：通过递归思想

class Solution:
    def fib(self, n: int) -> int:
        if n == 0:
            return 0
        if n in [1, 2]:
            return 1
        return self.fib(n - 1) + self.fib(n - 2)


class Solution2:
    def fib(self, n: int) -> int:

        # 排除 Corner Case
        if n == 0:
            return 0

        # 创建 dp table
        dp = [0] * (n + 1)

        # 初始化 dp 数组
        dp[0] = 0
        dp[1] = 1

        # 遍历顺序: 由前向后。因为后面要用到前面的状态
        for i in range(2, n + 1):
            # 确定递归公式/状态转移公式
            dp[i] = dp[i - 1] + dp[i - 2]

        # 返回答案
        return dp[n]


if __name__ == '__main__':
    start=time.time()
    sl = Solution()
    result = sl.fib(30)
    end=time.time()
    neicun=sys.getsizeof(sl)
    neicunlei = sys.getsizeof(result)
    print(f'开始{start}-------结束;{end-start:.2f},内存对象：{neicunlei},类：{neicun}')
    print(result)
    # start2 = time.time()
    # sl2=Solution2()
    # fib2=sl2.fib(30)
    # end2 = time.time()
    # print(fib2)
    # cache_lei = sys.getsizeof(sl2)
    # cache_obj=sys.getsizeof(fib2)
    # print(f'开始{start2}-------结束;{end2 - start2:.2f},内存对象：{cache_obj},类：{cache_lei}')

