import matplotlib.pyplot as plt
import logging
import re


# @TODO: use logging to file
# @TODO: use names of classes or manual names instead of __name__
# @FIXME: correct work of logging level
def enable_logging(name=None, level="info"):
    # @TODO: pass filename, filemode
    _level = logging.INFO
    if level[0] == 'd':
        _level = logging.DEBUG
    elif level[0] == 'w':
        _level = logging.WARNING
    elif level[0] == 'e':
        _level = logging.ERROR
    elif level[0] == 'c':
        _level = logging.CRITICAL
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s"
                               " - %(message)s")
    if name:
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger(__name__)
    logger.setLevel(_level)
    return logger


# @TODO: separate plotting and logging methods
class Plotter:
    def __init__(self, nn, logging_level="debug"):
        self._nn = nn
        logger_name = re.sub(r"[' <>]", '', str(self.__class__))
        logger_name = re.sub(r"^class", '', logger_name)
        self._logger = enable_logging(f"{logger_name}", logging_level)

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
                     show=True, equal=True) -> None:
        self.display_nodes(plot=plot, show=separate_show, log=log)
        if log:
            self.display_neighbors()
        self.display_edges(plot=plot, show=separate_show, log=log)
        if show and not separate_show:
            self.plot(equal)

    @staticmethod
    def plot(equal=False) -> None:
        if equal:
            plt.axis('equal')
        plt.show()

    def get_state(self) -> dict:
        return self._nn.current_state(deep=True)
