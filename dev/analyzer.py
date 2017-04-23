import matplotlib.pyplot as plt
try:
    from dev.ESOINN import EnhancedSelfOrganizingIncrementalNN
    from dev.commons import enable_logging
except ImportError as error:
    print(error.args)
    from .ESOINN import EnhancedSelfOrganizingIncrementalNN
    from .commons import enable_logging


# @TODO: separate plotting and logging methods
class Plotter:
    """
    Base class for all tests.
    Link to training nn is stored in each object.
    Implements common train method for unsupervised learning.
    """
    def __init__(self, nn: EnhancedSelfOrganizingIncrementalNN,
                 logging_level="debug"):
        self._nn = nn
        # self._state = nn.current_state(deep=False)
        self._logger = enable_logging(f"{self.__class__}", logging_level)
        if not isinstance(nn, EnhancedSelfOrganizingIncrementalNN):
            self._logger.warning(f"{type(nn)} passed instead of NN, "
                                 f"so tests won't work")

    def display_nodes(self, plot=False, show=False, log=False) -> None:
        state = self._nn.current_state(deep=False)
        if not state['nodes']:
            self._logger.warning("No nodes")
            return
        nodes = state['nodes']
        if plot:
            scale_x, scale_y = 0.05, 0.05
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
                plt.annotate(node_id, [features[0]+scale_x,
                                       features[1]+2*scale_y])
                plt.annotate(f"{float(nodes[node_id].density):1.5}",
                             [features[0]+scale_x, features[1]-2*scale_y])
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

    def display_neighbors(self) -> None:
        state = self._nn.current_state(deep=False)
        if not state['neighbors']:
            self._logger.warning("No neighbors")
            return
        neighbors = state['neighbors']
        max_id_width = len(str(max(neighbors.keys()))) + 1
        width_id = max_id_width if max_id_width > len(" id ") else 4
        print(f" {'id':^{width_id}}| neighbors ")
        for node_id in neighbors:
            print(f" {node_id:<{width_id}}| {neighbors.get(node_id, None)}")
        print()

    def display_edges(self, plot=False, show=False, log=False) -> None:
        state = self._nn.current_state(deep=False)
        if not state['edges']:
            self._logger.warning("No edges")
            return
        edges = state['edges']
        nodes = state['nodes']
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
                     show=True) -> None:
        self.display_nodes(plot=plot, show=separate_show, log=log)
        if log:
            self.display_neighbors()
        self.display_edges(plot=plot, show=separate_show, log=log)
        if show and not separate_show:
            self.plot()

    @staticmethod
    def plot() -> None:
        plt.show()

    def get_state(self) -> dict:
        return self._nn.current_state(deep=True)
