from math import inf
import operator

'''
图生成最小生成树
'''

class IndexMinPriorityQueue:
    def __init__(self, length):
        self.items = [None for _ in range(length)]
        # Ascendingly sort items and memorize the item's relative index in items
        self.pq = [None] + [i if self.items[i] else None for i in range(len(self.items))]
        # Its index is associate with elements in pq, and also syncs with indices of list items
        self.qp = [i if self.pq[i] else None for i in range(len(self.pq))]
        self.N = 0

    def size(self):
        return self.N

    def is_empty(self):
        return self.N == 0

    def less(self, i, j):
        """Compare the given two items in self.items"""
        return operator.lt(self.items[self.pq[i]], self.items[self.pq[j]])

    def swap(self, i, j):
        """But change the position of the two items in the subsidiary pq and qp lists"""
        self.pq[i], self.pq[j] = self.pq[j], self.pq[i]
        self.qp[self.pq[i]], self.qp[self.pq[j]] = i, j

    def min_elem_index(self):
        """Find the minimum element's index"""
        return self.pq[1]

    def is_index_exist(self, index):
        """Judge if the given index is exist in this queue"""
        return self.qp[index] is not None

    def insert(self, index, item):
        """Insert an element associated with the element's index in this queue"""
        if self.is_index_exist(index):
            return
        self.items[index] = item
        self.N += 1
        # Now it isn't a orderly queue
        self.pq[self.N] = index
        self.qp[index] = self.N

        # swim the last element to make list pq ordered
        self.swim(self.N)

    def delete_min_elem(self):
        """Delete the minimum element, and return its index"""
        min_index = self.pq[1]
        # print(f"min_ele: {self.items[min_index]}")
        self.swap(1, self.N)
        self.pq[self.N] = None
        self.qp[min_index] = None
        self.items[min_index] = None
        self.N -= 1
        self.sink(1)
        return min_index

    def change_item(self, idx, itm):
        """Substitute a item which index=idx with a new item which value=itm"""
        self.items[idx] = itm
        k = self.qp[idx]
        self.sink(k)
        self.swim(k)

    def swim(self, index):
        """Move the smaller element up; We should only change order in pq and qp"""
        while index > 1:
            # Compare the current node with its parent node, if smaller, swim up
            if self.less(index, int(index/2)):  # Compare values in items
                self.swap(index, int(index/2))  # But swap the mapping position in pq and qp
            index = int(index/2)

    def sink(self, index):
        """Move the bigger element down; We should only change order in pq and qp"""
        # print(f"SINK: idx:{index} N:{self.N}")
        while 2*index <= self.N:
            index_smaller = 2*index if 2*index+1 > self.N else \
                (2*index if self.less(2*index, 2*index+1) else 2*index+1)
            # print(f"index_smaller: {index_smaller}")
            # print(f"index: {index}")
            if self.less(index, index_smaller):
                break
            self.swap(index, index_smaller)
            index = index_smaller


class Edge:
    def __init__(self, vertex1, vertex2, weight):
        self.vertex1 = vertex1
        self.vertex2 = vertex2
        self.weight = weight

    def get_weight(self):
        return self.weight

    def either(self):
        """Return either vertex of this Edge"""
        return self.vertex1

    def opposite(self, v):
        return self.vertex1 if v == self.vertex2 else self.vertex2

    def compare(self, edge):
        return 1 if self.weight > edge.weight else (-1 if self.weight < edge.weight else 0)


class WeightedUndigraph:
    def __init__(self, v):
        self.num_vertices = v
        self.num_edges = 0
        self.adj_list = [[] for _ in range(v)]

    def get_num_vertices(self):
        return self.num_vertices

    def get_num_edges(self):
        return self.num_edges

    def add_edge(self, edge: Edge):
        v1 = edge.either()
        v2 = edge.opposite(v1)
        self.adj_list[v1].append(edge)
        self.adj_list[v2].append(edge)
        self.num_edges += 1

    def adjacent_edges(self, vertex):
        return self.adj_list[vertex]

    def all_edges(self):
        all_edges = []
        for i in range(self.num_vertices):
            for edge in self.adj_list[i]:
                if edge.opposite(i) < i:
                    all_edges.append(edge)
        return all_edges


class PrimMST:
    def __init__(self, graph):
        """MST here represent the Minimum Spanning Tree of the current loop"""
        self.graph = graph
        # Memorize the cheapest edge to MST of each vertex(index)
        self.min_edge_to_MST = [None for _ in range(self.graph.get_num_vertices())]
        # Store the smallest weight of each vertex(index)'s edge to MST;
        # Initialize it with infinite plus, we will compare out a minimum weight after
        self.min_weight_to_MST = [+inf for _ in range(self.graph.get_num_vertices())]
        # Mark a True if a vertex(index) has been visited
        self.marked = [False for _ in range(self.graph.get_num_vertices())]
        # Memorize the smaller weight of each vertex(index)'s edge connected to MST
        self.the_cut_edges = IndexMinPriorityQueue(self.graph.get_num_vertices())

        # Initialize a 0.0 as the minimum weight to weight_to_MST
        self.min_weight_to_MST[0] = 0.0
        self.the_cut_edges.insert(0, 0.0)
        while not self.the_cut_edges.is_empty():
            # Take out the minimum-weighted vertex, and make a visit(update) for it
            self.visit(self.the_cut_edges.delete_min_elem())
            # self.visit(self.the_cut_edges.delete_min_and_get_index())

    def visit(self, v):
        """Update the MST"""
        self.marked[v] = True
        for e in self.graph.adjacent_edges(v):
            w = e.opposite(v)
            # Check if the opposite vertex of v in edge e is marked, if did, skip this loop
            if self.marked[w]:
                continue
            # Find out the minimum-weighted-edge vertex opposite to this vertex(v)
            if e.get_weight() < self.min_weight_to_MST[w]:    # e.get_weight():Get weight of the edge between v and w
                # Update the minimum edge and weight
                # print(f"v: {v}, w: {w}, min_weight_edge: {e.get_weight()}")
                self.min_edge_to_MST[w] = e
                self.min_weight_to_MST[w] = e.get_weight()
                if self.the_cut_edges.is_index_exist(w):
                    # print(w, e.get_weight())
                    self.the_cut_edges.change_item(w, e.get_weight())
                else:
                    self.the_cut_edges.insert(w, e.get_weight())

    def min_weight_edges(self):
        return [edge for edge in self.min_edge_to_MST if edge]


if __name__ == '__main__':

    with open('../data_set/origin_data/from_weight_graph_to_tree/30_nodes/MST.txt', 'r') as f:
        num_vertices = int(f.readline())
        num_edges = int(f.readline())
        graph = WeightedUndigraph(num_vertices)

        for e in range(num_edges):
            v1, v2, w = f.readline().split()
            graph.add_edge(Edge(int(v1), int(v2), float(w)))
    P_MST = PrimMST(graph)

    for e in P_MST.min_weight_edges():
        v = e.either()
        w = e.opposite(v)
        weight = e.weight
        print(f"v: {v} w: {w} weight: {weight}")
