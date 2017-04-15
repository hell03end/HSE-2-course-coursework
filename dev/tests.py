import numpy as np
import matplotlib.pyplot as plt
try:
    from dev import ESOINN
except ImportError as error:
    print(error.args)
    import ESOINN


def display_nodes(nodes: dict, plot=False, log=False):
    if not nodes:
        print("LOG: no nodes")
        return
    # plot
    x, y, mark = [], [], []
    for node_id in nodes:
        features = nodes[node_id].feature_vector
        x.append(features[0])
        y.append(features[1])
        mark.append(nodes[node_id].subclass_id)
    plt.scatter(x, y, c=mark, s=100)
    plt.title("Topology")
    for node_id in nodes:
        features = nodes[node_id].feature_vector
        plt.annotate(node_id, [features[0], features[1]])
    if plot:
        plt.show()
    # log
    if not log:
        return
    max_id_width = len(str(max(nodes.keys()))) + 1
    width_id = max_id_width if max_id_width > len(" id ") else 4
    width_class = max_id_width if max_id_width > len(" class ") else 7
    print(f"|{'—'*(width_id+1)}|{'—'*30}|{'—'*(width_class+1)}|{'—'*21}|")
    print(f"| {'id':^{width_id}}|{'weights':^30}|"
          f"{'class':^{width_class}} |{'density':^21}|")
    print(f"|{'—'*(width_id+1)}|{'—'*30}|{'—'*(width_class+1)}|{'—'*21}|")
    for node_id in nodes:
        node = nodes[node_id]
        print(f"| {node_id:<{width_id}}|{str(node.feature_vector):^30}|"
              f"{node.subclass_id:{width_class}} | {str(node.density):20.20}|")
    print(f"|{'—'*(width_id+1)}|{'—'*30}|{'—'*(width_class+1)}|{'—'*21}|\n")


def display_neighbors(neighbors: dict):
    if not neighbors:
        print("LOG: no neighbors")
        return
    max_id_width = len(str(max(neighbors.keys()))) + 1
    width_id = max_id_width if max_id_width > len(" id ") else 4
    print(f" {'id':^{width_id}}| neighbors ")
    for node_id in neighbors:
        print(f" {node_id:<{width_id}}| {neighbors.get(node_id, None)}")
    print()


def display_edges(edges: dict, nodes: dict, plot=False, log=False):
    if not edges:
        print("LOG: no edges")
        return
    # plot
    for edge in edges:
        x, y = [], []
        for node_id in edge:
            node_features = nodes[node_id].feature_vector
            x.append(node_features[0])
            y.append(node_features[1])
        plt.plot(x, y)
    if plot:
        plt.show()
    # log
    if not log:
        return
    print(f"|{'—'*20}|{'—'*5}|")
    print(f"|{'edge':^20}|{'age':^5}|")
    print(f"|{'—'*20}|{'—'*5}|")
    for edge in edges:
        print(f"|{str(edge):^20}| {edges[edge]:<4}|")
    print(f"|{'—'*20}|{'—'*5}|")


def train(nn, df=None) -> dict:
    if not df:
        pass
    for signal in df:
        nn.fit(signal)
    return nn.current_state()


if __name__ == "__main__":
    pass
