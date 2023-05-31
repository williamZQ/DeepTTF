# encoding utf-8
'''
@Author: william
@Description: BFS建立树
@time:2021/6/5 18:44
'''
import queue
import numpy as np


# 将邻接矩阵转化为字典
def InvertGraph(A_matrix):
    graph = {}
    for i, row in enumerate(A_matrix):
        # i表示节点值，需要加1；row表示节点所在行的索引
        # print('i row',i,row)
        node = i + 1
        # 每行的非零元素的下标
        nonzero = np.nonzero(row)[0].tolist()
        nonzero1 = [index + 1 for index in nonzero]
        # 下标需要加1
        # print(i+1,nonzero1)
        # 向字典中加入数据
        graph[node] = nonzero1
    return graph


def MaxNodeNumber(layer):
    max_number = 0
    k = 0
    for i in layer:
        tmp_number = i[k]
        k += 1
        if tmp_number > max_number:
            max_number = tmp_number
    return max_number


def BFS(adj, start):
    visited = set()
    total = 0
    n = 0
    count = 0
    layer = []
    output = []
    q = queue.Queue()
    q.put(start)
    visited.add(start)
    while not q.empty():
        u = q.get()
        output.append(u)
        if u == start:
            layer.append({0: 1})
            total += 1
            current_layer = adj.get(start, [])
        else:
            count += 1

        if len(current_layer) > 0:
            if u == current_layer[-1]:
                n += 1
                layer.append({n: count})
                total += count
                count = 0
                # print("*")
                current_layer = adj.get(u, [])

        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v)
                q.put(v)

    if len(output) - total != 0:
        n += 1
        layer.append({n: len(output) - total})

    max_node_number_last_layer = MaxNodeNumber(layer)
    return output, n + 1, layer, max_node_number_last_layer


if __name__ == "__main__":
    graph = {1: [2, 3], 2: [1, 3], 3: [1, 2, 4], 4: [3, 5], 5: [4, 6], 6: [5, 27], 27: [6, 15, 16, 28],
             15: [14, 16, 27], 14: [13, 15], 13: [12, 11, 14], 12: [10, 11, 13], 11: [10, 12, 13], 10: [9, 11, 12],
             9: [8, 10], 8: [7, 9], 7: [8], 16: [15, 27, 17, 18], 17: [16, 18, 19], 18: [16, 17, 19], 19: [17, 18, 20],
             20: [19, 21], 21: [20, 26, 28], 22: [23], 23: [22, 26], 24: [25, 26], 25: [24], 26: [21, 23, 24],
             28: [21, 27, 29, 30], 29: [28, 30], 30: [28, 29]}
    graph = {1: [2, 4], 2: [1, 3, 4], 3: [2, 4], 4: [1, 2, 3, 5]}
    # graph = {
    #     1: [2, 3, 4, 5], 2: [1], 3: [1, 8], 4: [1, 6], 5: [1, 6, 7],
    #     6: [4, 5], 7: [5, 8], 8: [3, 7, 9], 9: [8]
    # }
    output, n, layer, max_node_number_last_layer = BFS(graph, 1)
    print(output)
    print(n)
    print(layer)
    print(max_node_number_last_layer)
