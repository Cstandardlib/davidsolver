import numpy as np

# 定义矩阵数据
h_mat_data = """
1 6 3 4 3 0 2 7 
6 2 7 5 6 5 2 1 
3 7 3 3 2 2 7 4 
4 5 3 4 0 3 0 2 
3 6 2 0 5 7 2 6 
0 5 2 3 7 6 5 4 
2 2 7 0 2 5 7 1 
7 1 4 2 6 4 1 8
"""

# 将矩阵数据字符串分割成行，然后转换成整数，并创建 NumPy 数组
h_mat = np.array([row.split() for row in h_mat_data.strip().split('\n')], dtype=int)

# 输出转换后的 NumPy 矩阵
print("转换后的 NumPy 矩阵:")
print(h_mat)

# 使用 numpy 的 eigh 函数计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eigh(h_mat)

# 输出特征值
print("特征值（升序排序）:")
for eigenvalue in sorted(eigenvalues):
    print(eigenvalue)

# 输出特征向量
print("\n特征向量（对应于升序的特征值）:")
for i, eigenvector in enumerate(eigenvectors.T):
    print(f"第 {i+1} 个特征向量:")
    print(eigenvector)