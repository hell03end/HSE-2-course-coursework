import matplotlib.pyplot as plt
from timeit import timeit
import numpy as np
try:
    from dev import ESOINN, mock
    from dev.commons import enable_logging
except ImportError as error:
    print(error.args)
    import ESOINN
    import mock
    from commons import enable_logging


# @TODO: add generators for training data
class TrainingSamples:
    def __init__(self, random_state):
        self.__seed = random_state


class BasicTest:
    """
    Base class for all tests.
    Link to training nn is stored in each object.
    Implements common train method for unsupervised learning.
    """
    def __init__(self, nn: ESOINN.EnhancedSelfOrganizingIncrementalNN):
        self._nn = nn
        self._state = nn.current_state(deep=True)
        self._logger = enable_logging(f"{__name__}.BasicTest")

    def train(self, df):
        self._logger.info("Start training")
        self._state = self._nn.fit(df, get_state=True)
        self._logger.info("Training complete")


# @FIXME: separate plotting and logging methods
# @TODO: unit tests
class CoreTest(BasicTest):
    def display_nodes(self, plot=False, show=False, log=False):
        if not self._state['nodes']:
            self._logger.warning("No nodes")
            return
        nodes = self._state['nodes']
        if plot:
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
        if plot and show:
            plt.show()

        if not log:
            return
        max_id_width = len(str(max(nodes.keys()))) + 1
        width_id = max_id_width if max_id_width > len(" id ") else 4
        width_class = max_id_width if max_id_width > len(" class ") else 7
        print(f"|{'—'*(width_id+1)}|{'—'*30}|"
              f"{'—'*(width_class+1)}|{'—'*21}|")
        print(f"| {'id':^{width_id}}|{'weights':^30}|"
              f"{'class':^{width_class}} |{'density':^21}|")
        print(f"|{'—'*(width_id+1)}|{'—'*30}|"
              f"{'—'*(width_class+1)}|{'—'*21}|")
        for node_id in nodes:
            node = nodes[node_id]
            print(f"| {node_id:<{width_id}}|"
                  f"{str(node.feature_vector):^30}|"
                  f"{node.subclass_id:{width_class}} | "
                  f"{str(node.density):20.20}|")
        print(f"|{'—'*(width_id+1)}|{'—'*30}|"
              f"{'—'*(width_class+1)}|{'—'*21}|\n")

    def display_neighbors(self):
        if not self._state['neighbors']:
            self._logger.warning("No neighbors")
            return
        neighbors = self._state['neighbors']
        max_id_width = len(str(max(neighbors.keys()))) + 1
        width_id = max_id_width if max_id_width > len(" id ") else 4
        print(f" {'id':^{width_id}}| neighbors ")
        for node_id in neighbors:
            print(f" {node_id:<{width_id}}| {neighbors.get(node_id, None)}")
        print()

    def display_edges(self, plot=False, show=False, log=False):
        if not self._state['edges']:
            self._logger.warning("No edges")
            return
        edges = self._state['edges']
        nodes = self._state['nodes']
        if plot:
            for edge in edges:
                x, y = [], []
                for node_id in edge:
                    node_features = nodes[node_id].feature_vector
                    x.append(node_features[0])
                    y.append(node_features[1])
                plt.plot(x, y, '#9f9fa3')
        if plot and show:
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

    def display_info(self, plot=False, separate_show=False, log=False,
                     show=True):
        self.display_nodes(plot=plot, show=separate_show, log=log)
        if log:
            self.display_neighbors()
        self.display_edges(plot=plot, show=separate_show, log=log)
        if show and not separate_show:
            self.plot()

    @staticmethod
    def plot():
        plt.show()


class TrainTest(BasicTest):
    pass


def load_mock_graph():
    pass


if __name__ == "__main__":
    pass
