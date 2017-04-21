from copy import deepcopy
import numpy as np
import pickle
import os


class Node:
    def __init__(self, feature_vector, density=0, subclass_id=-1):
        self.density = density
        self.subclass_id = subclass_id
        self.feature_vector = np.array(feature_vector, dtype=float)
        self.acc_signals = 0
        self.total_points = float(0)

    def __repr__(self):
        return str(f"{float(self.density):<10.5}\t"
                   f"{str(self.feature_vector):^30}\t"
                   f"{self.subclass_id}")


class Graph:
    def __init__(self):
        self.nodes = {}
        self.neighbors = {}
        self.edges = {}

    def __repr__(self):
        max_id_width = len(str(max(self.nodes.keys()))) + 1
        representation = f"{'id':<{max_id_width}}\t" \
                         f"{'density':^10.10}\t" \
                         f"{'feature_vector':^30}\t" \
                         f"subclass_id\n{'='*80}\n"
        for node_id in self.nodes:
            representation += f"{node_id:<{max_id_width}}:" \
                              f"\t {self.nodes[node_id]}\n"
        return representation

    def current_state(self, deep=True) -> dict:
        nodes = self.nodes
        neighbors = self.neighbors
        edges = self.edges
        if deep:
            nodes = deepcopy(self.nodes)
            neighbors = deepcopy(self.neighbors)
            edges = deepcopy(self.edges)
        return {
            'count_neurons': len(self.nodes),
            'nodes': nodes,
            'neighbors': neighbors,
            'edges': edges,
        }


def load_mock(path=r"src/suspended_undirected_graph_enhanced"):
    if not os.path.exists(path):
        return
    with open(path, "rb") as loader:
        return pickle.load(loader)


def save_mock(graph, path=r"src/suspended_undirected_graph_enhanced"):
    if not os.path.exists(path):
        return
    with open(path, "wb") as writer:
        pickle.dump(graph, writer)


def change_mock(graph):
    graph.nodes[1].feature_vector = np.array([2.5, 2.5])
    graph.nodes[2].feature_vector = np.array([5.25, 2])
    graph.nodes[3].feature_vector = np.array([6, 1.5])
    graph.nodes[4].feature_vector = np.array([6, 0.5])
    graph.nodes[5].feature_vector = np.array([5.25, 1])
    graph.nodes[6].feature_vector = np.array([2.875, 0.5])
    graph.nodes[7].feature_vector = np.array([1, 3.375])
    graph.nodes[8].feature_vector = np.array([4.5, 3.75])
    graph.nodes[9].feature_vector = np.array([5.75, 4.25])
    graph.nodes[10].feature_vector = np.array([6.75, 3.75])
    graph.nodes[11].feature_vector = np.array([5.5, 1.125])
    graph.nodes[12].feature_vector = np.array([5.25, 1.375])
    graph.nodes[13].feature_vector = np.array([4.25, 1.625])
    graph.nodes[14].feature_vector = np.array([4.25, 1.25])
    graph.nodes[15].feature_vector = np.array([3.5, 0.5])
    graph.nodes[16].feature_vector = np.array([3, 2.25])
    graph.nodes[17].feature_vector = np.array([2.75, 1.75])
    graph.nodes[18].feature_vector = np.array([2, 1.25])
    graph.nodes[19].feature_vector = np.array([1.5, 1])
    graph.nodes[20].feature_vector = np.array([0.75, 1])
    graph.nodes[21].feature_vector = np.array([1, 1.5])
    graph.nodes[22].feature_vector = np.array([1.25, 2.5])
    graph.nodes[23].feature_vector = np.array([1, 2.75])
    graph.nodes[24].feature_vector = np.array([1.75, 3])
    graph.nodes[25].feature_vector = np.array([2, 3.25])
    graph.nodes[26].feature_vector = np.array([3, 4.25])
    graph.nodes[27].feature_vector = np.array([3.25, 3.625])
    graph.nodes[28].feature_vector = np.array([3, 3])
    graph.nodes[29].feature_vector = np.array([4.5, 4.25])
    graph.nodes[30].feature_vector = np.array([4.5, 3.375])
    graph.nodes[31].feature_vector = np.array([5.75, 3.25])
    graph.nodes[32].feature_vector = np.array([6.25, 3.25])
    graph.nodes[33].feature_vector = np.array([6.5, 2.5])
    graph.nodes[34].feature_vector = np.array([6.75, 2])
