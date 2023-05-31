# encoding utf-8
'''
@Author: william
@Description: 树建立特征矩阵
@time:2021/6/5 20:33
'''
from NATree_design.Weight import Addweight
from NATree_design.BFS_make_tree import BFS, InvertGraph
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # 读入矩阵文件
    # A = np.load('../data_set/output_data/Adj_Matrix.npy')
    A = pd.read_csv('../data_set/output_data/A_50.csv', delimiter=',', header=None).values
    # print(A)
    # 将邻接矩阵转化为邻接表
    graph = InvertGraph(A)
    nodes_number = A.shape[0]
    max_node_number = 0
    max_layer_number = 0
    max_node_child_number = 0

    for i in range(len(graph)):
        tmp_max_node_child_number = len(graph[i + 1])
        if tmp_max_node_child_number > max_node_child_number:
            max_node_child_number = tmp_max_node_child_number

    for i in range(nodes_number):
        output, n, layer, tmp_max_node_number = BFS(graph, i)
        if n > max_layer_number:
            max_layer_number = n
        if tmp_max_node_number > max_node_number:
            max_node_number = tmp_max_node_number

    y = np.zeros([1, max_layer_number, max_node_number * max_node_child_number])
    for i in range(nodes_number):
        output, n, layer, tmp_max_node_number = BFS(graph, i + 1)

        layer_count = 0
        current_total_number = 0
        x = np.ones([1, max_node_number * max_node_child_number])

        for item in layer:
            layer_number = item[layer_count]

            if layer_count == 0:
                x = x * output[0]
                layer_count += 1
                current_total_number += 1
                continue

            if layer_count != n:
                if (max_node_number * max_node_child_number) % layer_number != 0:
                    avg_node_number = \
                        int(
                            ((max_node_number * max_node_child_number) - ((max_node_number * max_node_child_number) % layer_number)
                             ) / layer_number)
                else:
                    avg_node_number = int((max_node_number * max_node_child_number) / layer_number)

                x_tmp = np.ones([1, avg_node_number]) * output[current_total_number]
                for j in range(current_total_number + 1, current_total_number + layer_number):
                    x_tmp_1 = np.ones([1, avg_node_number]) * output[j]
                    x_tmp = np.hstack((x_tmp, x_tmp_1))

                re_count = x_tmp.shape
                if (max_node_number * max_node_child_number) % layer_number != 0:
                    for q in range((max_node_number * max_node_child_number) - x_tmp.shape[1]):
                        x_tmp = np.hstack((x_tmp, np.zeros([1, 1])))
                x = np.vstack((x, x_tmp))

                layer_count += 1
                current_total_number += layer_number
            # else:
            #     x_tmp_2 = np.ones([1, 1]) * output[current_total_number]
            #
            #     for k in range(current_total_number + 1, current_total_number + layer_number):
            #         k_tmp_2 = np.ones([1, 1]) * output[k]
            #         x_tmp_2 = np.hstack(x_tmp_2, k_tmp_2)
            #     x = np.vstack(x, x_tmp_2)
            #     layer_count += 1

        if layer_count < max_layer_number:
            for item in range(max_layer_number - layer_count):
                x = np.vstack((x, np.zeros([1, (max_node_number * max_node_child_number)])))


        if i == 0:
            y = np.reshape(x, (1, max_layer_number, (max_node_number * max_node_child_number)))
        else:
            y = np.concatenate((y, np.reshape(x, (1, max_layer_number, (max_node_number * max_node_child_number)))), axis=0)

    print(max_node_number)
    print(max_layer_number)
    print(y)
    y_weight = Addweight(y, '../data_set/output_data/w_50.csv')
    np.save("../data_set/output_data/TreeMatrix_50.npy", y)
