'''
toarray() 方法通常在使用 Python 的 Scikit-learn 库或类似库处理数据时遇到。
它常用于将稀疏矩阵转换为密集的 numpy 数组。稀疏矩阵是一种大多数元素为零的矩阵，
使用稀疏表示法可以节省大量的内存空间。
'''

from scipy import sparse
import numpy as np

# 1. 创建一个3乘3的稀疏矩阵，只有对角线位置的元素为1，其他位置为0
sparse_matrix = sparse.eye(3)
# 打印稀疏矩阵,你会看到它并不是以常规的二维数组形式展示
print("稀疏矩阵 (Sparse Matrix):")
print(sparse_matrix)

# 2. 使用 toarray() 方法将稀疏矩阵转换为 numpy 数组
dense_array = sparse_matrix.toarray()
print("\n密集矩阵 (Dense Array):")
print(dense_array)