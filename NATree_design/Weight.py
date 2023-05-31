# 为广度优先遍历矩阵添加权重
import pandas as pd
import numpy as np


# 为正则化矩阵加入权值
def Addweight(X, weighted_file):
    # 权值矩阵
    w = pd.read_csv(weighted_file, sep=',', header=None)
    w = np.asarray(w)
    X1 = X
    # X1 = copy.deepcopy(X)

    # 求根节点,矩阵首行就代表的根节点
    for k in range(X.shape[0]):
        for row in range(1, X.shape[1]):
            for column in range(X.shape[2]):
                # 目的节点
                adj_x = X[k, row - 1, column]
                adj_y = X[k, row, column]
                # 目的节点为0表示不可达
                if (adj_x == 0) | (adj_y == 0):
                    continue
                # 替换对应矩阵上的值
                if float(w[int(adj_x - 1), int(adj_y - 1)]) == 0:
                    X1[k, row, column] = 1
                    continue
                X1[k, row, column] = 1 / float(w[int(adj_x - 1), int(adj_y - 1)])
    X1[:, 0, :] = 1
    return X1


if __name__ == "__main__":
    data = np.load('C:/Users/Yyyk/Desktop/mzbfile/工作/TreeCN/TreeMatrix.npy')
    matrix = data[0]
    print(matrix)
    matrix1 = Addweight(matrix)
    print(matrix1)
