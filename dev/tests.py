import matplotlib.pyplot as plt
from timeit import timeit
from copy import deepcopy
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


# @TODO: separate plotting and logging methods
class BasicTest:
    """
    Base class for all tests.
    Link to training nn is stored in each object.
    Implements common train method for unsupervised learning.
    """
    def __init__(self, nn: ESOINN.EnhancedSelfOrganizingIncrementalNN,
                 logging_level="debug"):
        self._nn = nn
        # self._state = nn.current_state(deep=False)
        self._logger = enable_logging(f"{__name__}.BasicTest", logging_level)
        if not isinstance(nn, ESOINN.EnhancedSelfOrganizingIncrementalNN):
            self._logger.warning(f"{type(nn)} passed instead of NN, "
                                 f"so tests won't work")

    def display_nodes(self, plot=False, show=False, log=False):
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

    def display_neighbors(self):
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

    def display_edges(self, plot=False, show=False, log=False):
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

    def get_state(self) -> dict:
        return self._nn.current_state(deep=True)


# @TODO: unit tests
class CoreTest(BasicTest):
    # @CHECKME: is it necessary?
    def apply_function(self, function, n_times=0, **kwargs):
        result = function(self._nn, **kwargs)
        if n_times > 0:
            self._logger.info(str(timeit("function(self._nn, **kwargs)",
                                         globals=locals(), number=n_times)))
        return result

    def initialize_tests(self):
        g = mock.load_mock()
        self.configure_nn(g.nodes, g.neighbors, g.edges)

    def configure_nn(self, nodes=None, neighbors=None, edges=None):
        if nodes:
            self._nn.nodes.clear()  # for clean configuration
            for node_id in nodes:
                node = nodes[node_id]
                self._nn.nodes[node_id] = \
                    ESOINN.ESOINNNode(node.feature_vector)
                self._nn.nodes[node_id]._ESOINNNode__density = node.density
                self._nn.nodes[node_id].update_accumulate_signals()
                self._nn.nodes[node_id].subclass_id = node.subclass_id
                if node.subclass_id != -1:
                    self._nn.nodes[node_id].subclass_id = node.subclass_id
                else:
                    self._nn.nodes[node_id].subclass_id = 0
            self._nn.unique_id = len(nodes)  # set correct unique id
        if neighbors and edges:
            self._nn.neighbors.clear()
            self._nn.edges.clear()
            self._nn.neighbors = deepcopy(neighbors)
            self._nn.edges = deepcopy(edges)

    def report_error(self, test, name, **kwargs):
        res, *time = test(**kwargs)
        if not res:
            self._logger.error(f"TEST: {name}")
        # else:
        #     self._logger.info(f"TEST: {name}")
        if time and kwargs.get('n_times', None):
            self._logger.info(f"{name} for {kwargs['n_times']} iterations:\t"
                              f"{time}")

    # @TODO: use dists[] instead of dist0, dist1
    def test_find_winners(self, n_times=0):
        feature_vector = [5.5, 3]
        winners, dists = self._nn.find_winners(feature_vector)
        dist0 = dists[0] == self._nn.metrics([5.75, 3.25], feature_vector)
        dist1 = dists[1] == self._nn.metrics([6.25, 3.25], feature_vector)
        run_time = None
        if n_times > 0:
            run_time = timeit('self._nn.find_winners(feature_vector)',
                              number=n_times, globals=locals())
        return winners == [30, 31] and dist0 and dist1, run_time

    def test_find_neighbors(self, n_times=0):
        neighbors = self._nn.find_neighbors(20, depth=2)
        check = neighbors == {17, 18, 19, 21, 22, 23, 6}
        neighbors = self._nn.find_neighbors(20)
        run_time = None
        if n_times > 0:
            run_time = timeit('self._nn.find_neighbors(20)',
                              number=n_times, globals=locals())
        return check and neighbors == {18, 19, 21, 22}, run_time

    def test_calc_threshold(self, n_times=0):
        right_calc = True
        id0, id1 = 34, 19
        _, dist = self._nn.find_winners(self._nn.nodes[id0].feature_vector)
        rc = self._nn.rc

        for i in range(1, 3):
            self._nn.rc = i
            neighbors = self._nn.find_neighbors(id1, depth=i)
            max_dist = max([
                self._nn.metrics(
                    self._nn.nodes[neighbor_id].feature_vector,
                    self._nn.nodes[id1].feature_vector
                ) for neighbor_id in neighbors
            ])
            right_calc &= self._nn.calc_threshold(id0) == dist[1]
            right_calc &= self._nn.calc_threshold(id1) == max_dist
        self._nn.rc = rc

        run_time = None
        if n_times > 0:
            run_time = timeit('self._nn.calc_threshold(id0)',
                              number=n_times, globals=locals())
        return right_calc, run_time

    def test_create_node(self):
        last_id = self._nn.unique_id
        feature_vector = [1, 0.5]
        self._nn.create_node(feature_vector)
        return [
            last_id + 1 == self._nn.unique_id
            and len(self._nn.nodes) == self._nn.unique_id
            and list(self._nn.nodes[last_id].feature_vector) == feature_vector
        ]

    # @TODO: change literals to variables
    def test_create_edges(self):
        id1 = self._nn.unique_id - 1
        if not self._nn.nodes.get(id1, None):
            return
        self._nn.create_edges([id1, 18, 19, 20])
        successfully_add = True

        for i in range(18, 21):
            successfully_add &= self._nn.edges.get((i, id1), False) == 0
            successfully_add &= id1 in self._nn.neighbors.get(i, set())

        return [successfully_add]

    def test_remove_edges(self):
        id0, id1 = 20, self._nn.unique_id - 1
        if not self._nn.nodes.get(id1, None):
            return
        self._nn.remove_edges([id1, id0])
        successfully_remove = True
        successfully_remove &= self._nn.edges.get((id0, id1), True)
        successfully_remove &= id1 not in self._nn.neighbors.get(id0, set())
        return [successfully_remove]

    def test_remove_node(self):
        id1 = self._nn.unique_id - 1
        if not self._nn.nodes.get(id1, None):
            return
        self._nn.remove_node(id1)
        successfully_remove = True
        for i in range(18, 20):
            successfully_remove &= self._nn.edges.get((i, id1), True)
            successfully_remove &= id1 not in self._nn.neighbors.get(i, set())
        return [successfully_remove]

    def test_update_edges_age(self, n_times=0):
        step = 2
        successfully_age_update = True
        self._nn.update_edges_age(19, step=step)
        neighbors_id = self._nn.find_neighbors(19)
        for neighbor_id in neighbors_id:
            successfully_age_update &= \
                self._nn.edges.get((min(19, neighbor_id),
                                    max(19, neighbor_id)), False) == step
        run_time = None
        if n_times > 0:
            run_time = timeit('self._nn.update_edges_age(19, step=0)',
                              number=n_times, globals=locals())
        return successfully_age_update, run_time

    def test_update_node_points(self, n_times=0):
        ids = (34, 19)
        right_points = True
        for rc in range(1, 3):
            for i in ids:
                neighbors = self._nn.find_neighbors(i, depth=rc)
                points = self._nn.nodes[i].points
                self._nn.update_node_points(i, neighbors)
                if neighbors:
                    mean_dist = 1/len(neighbors)*np.sum([
                        self._nn.metrics(
                            self._nn.nodes[i].feature_vector,
                            self._nn.nodes[neighbor_id].feature_vector
                        ) for neighbor_id in neighbors
                    ])
                    right_points &= \
                        self._nn.nodes[i].points == points+1/(1+mean_dist)**2
                else:
                    right_points &= points + 1 == self._nn.nodes[i].points

        run_time = None
        if n_times > 0:
            neighbors = self._nn.find_neighbors(ids[1])
            run_time = timeit('self._nn.update_node_points(ids[1], neighbors)',
                              number=n_times, globals=locals())
        return right_points, run_time

    # @TODO: undone
    def test_update_node_density(self):
        return [True]

    # @TODO: undone
    def test_update_feature_vectors(self, n_times=0):
        id1 = 19
        neighbors = self._nn.find_neighbors(id1)
        self._nn.update_feature_vectors(id1, [0, 0], neighbors)
        run_time = None
        if n_times > 0:
            run_time = timeit('self._nn.update_feature_vectors(id1, [0, 0], '
                              'neighbors)', number=n_times, globals=locals())
        return True, run_time

    def test_remove_old_ages(self, n_times=0):
        ids = (1, 34)
        successfully_remove = True
        self._nn.create_edges(ids)
        self._nn.edges[ids] = self._nn.max_age + 1
        self._nn.remove_old_ages()
        successfully_remove &= self._nn.edges.get(ids, True)
        successfully_remove &= ids[1] not in self._nn.neighbors.get(ids[0], {})
        successfully_remove &= ids[0] not in self._nn.neighbors.get(ids[1], {})

        run_time = None
        if n_times > 0:
            run_time = timeit('self._nn.remove_old_ages()',
                              number=n_times, globals=locals())
        return successfully_remove, run_time

    # @TODO: undone
    def test_calc_mean_density_in_subclass(self, n_times=0):
        run_time = None
        if n_times > 0:
            run_time = timeit('self._nn.calc_mean_density_in_subclass(0)',
                              number=n_times, globals=locals())
        return True, run_time

    # @TODO: undone
    def test_calc_alpha(self):
        return [True]

    def test_merge_subclass_condition(self, n_times=0):
        run_time = None
        if n_times > 0:
            run_time = timeit('self._nn.merge_subclass_condition((0, 1))',
                              number=n_times, globals=locals())
        return True, run_time

    def run_unit_tests(self, n_times=0):
        self.report_error(self.test_find_winners, "find_winners()",
                          n_times=n_times)
        self.report_error(self.test_find_neighbors, "find_neighbors()",
                          n_times=n_times)
        self.report_error(self.test_calc_threshold, "calc_threshold()",
                          n_times=n_times)
        self.report_error(self.test_create_node, "create_node()")
        self.report_error(self.test_create_edges, "create_edges()")
        self.report_error(self.test_remove_edges, "remove_edges()")
        self.report_error(self.test_remove_node, "remove_node()")
        self.report_error(self.test_update_edges_age, "update_edges_age()",
                          n_times=n_times)
        self.report_error(self.test_update_node_points, "update_node_points()",
                          n_times=n_times)
        self.report_error(self.test_update_node_density,
                          "update_node_density()")
        self.report_error(self.test_update_feature_vectors,
                          "update_feature_vectors()", n_times=n_times)
        self.report_error(self.test_remove_old_ages, "remove_old_ages()",
                          n_times=n_times)
        self.report_error(self.test_calc_mean_density_in_subclass,
                          "calc_mean_density_in_subclass()", n_times=n_times)
        self.report_error(self.test_calc_alpha, "calc_alpha()")
        self.report_error(self.test_merge_subclass_condition,
                          "merge_subclass_condition()", n_times=n_times)


class TrainTest(BasicTest):
    pass


if __name__ == "__main__":
    pass
