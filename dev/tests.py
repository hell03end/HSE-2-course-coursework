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
    from .commons import enable_logging


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
        self._logger = enable_logging(f"{self.__class__}", logging_level)
        if not isinstance(nn, ESOINN.EnhancedSelfOrganizingIncrementalNN):
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


# @TODO: unit tests
class CoreTest(BasicTest):
    # @CHECKME: is it necessary?
    def apply_function(self, function, n_times=0, **kwargs):
        result = function(self._nn, **kwargs)
        if n_times > 0:
            self._logger.info(str(timeit("function(self._nn, **kwargs)",
                                         globals=locals(), number=n_times)))
        return result

    def initialize_tests(self) -> None:
        g = mock.load_mock()
        self.configure_nn(g.nodes, g.neighbors, g.edges)

    def configure_nn(self, nodes=None, neighbors=None, edges=None) -> None:
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

    def report_error(self, test, name, **kwargs) -> None:
        res, *time = test(**kwargs)
        if not res:
            self._logger.error(f"TEST: {name}")
        # else:
        #     self._logger.info(f"TEST: {name}")
        if time and kwargs.get('n_times', None):
            self._logger.debug(f"{time[0]:.5}\tfor {kwargs['n_times']} {name}")

    # @TODO: use dists[] instead of dist0, dist1
    def test_find_winners(self, n_times=0) -> tuple:
        feature_vector = [5.5, 3]
        winners, dists = self._nn.find_winners(feature_vector)
        dist0 = dists[0] == self._nn.metrics([5.75, 3.25], feature_vector)
        dist1 = dists[1] == self._nn.metrics([6.25, 3.25], feature_vector)
        run_time = None
        if n_times > 0:
            run_time = timeit('self._nn.find_winners(feature_vector)',
                              number=n_times, globals=locals())
        return winners == (30, 31) and dist0 and dist1, run_time

    def test_find_neighbors(self, n_times=0) -> tuple:
        neighbors = self._nn.find_neighbors(20, depth=2)
        check = neighbors == {17, 18, 19, 21, 22, 23, 6}
        neighbors = self._nn.find_neighbors(20)
        run_time = None
        if n_times > 0:
            run_time = timeit('self._nn.find_neighbors(20)',
                              number=n_times, globals=locals())
        return check and neighbors == {18, 19, 21, 22}, run_time

    def test_calc_threshold(self, n_times=0) -> tuple:
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

    # @FIXME: undone all
    def test_build_connection(self, n_times=0):
        pass

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

    def test_update_edges_age(self, n_times=0) -> tuple:
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

    def test_update_node_points(self, n_times=0) -> tuple:
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

    # @FIXME: undone all
    def test_update_node_density(self):
        return [True]

    # @FIXME: undone method testing
    def test_update_feature_vectors(self, n_times=0) -> tuple:
        id1 = 0
        run_time = None
        if n_times > 0:
            neighbors = self._nn.find_neighbors(id1)
            nodes = deepcopy(self._nn.nodes)
            run_time = timeit('self._nn.update_feature_vectors(id1, [0, 0], '
                              'neighbors)', number=n_times, globals=locals())
            self._nn.nodes = nodes
        return True, run_time

    def test_remove_old_ages(self, n_times=0) -> tuple:
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

    # @FIXME: undone method testing
    def test_calc_mean_density_in_subclass(self, n_times=0) -> tuple:
        run_time = None
        if n_times > 0:
            run_time = timeit('self._nn.calc_mean_density_in_subclass(0)',
                              number=n_times, globals=locals())
        return True, run_time

    # @FIXME: undone all
    def test_calc_alpha(self):
        return [True]

    # @FIXME: undone method testing
    def test_merge_subclass_condition(self, n_times=0) -> tuple:
        run_time = None
        if n_times > 0:
            run_time = timeit('self._nn.merge_subclass_condition((0, 1))',
                              number=n_times, globals=locals())
        return True, run_time

    def test_change_class_id(self, n_times=0) -> tuple:
        id1, class_id = 0, 0
        self._nn.change_class_id(id1, class_id)
        correct_marking = True
        for node_id in self._nn.nodes:  # node 34 has different class
            if node_id == 34:
                continue
            correct_marking &= self._nn.nodes[node_id].subclass_id == class_id
        run_time = None
        if n_times > 0:
            run_time = timeit('self._nn.change_class_id(id1, class_id)',
                              number=n_times, globals=locals())
        return correct_marking, run_time

    def test_combine_subclasses(self, n_times=0) -> tuple:
        ids = (0, 34)
        correct_marking = True

        self._nn.nodes[ids[1]].subclass_id = -1
        self._nn.combine_subclasses(ids)
        for node in self._nn.nodes.values():
            correct_marking &= node.subclass_id != -1

        self._nn.nodes[ids[1]].subclass_id += 100
        self._nn.combine_subclasses(ids)
        fix_class_id = self._nn.nodes[ids[1]].subclass_id
        for node in self._nn.nodes.values():
            correct_marking &= node.subclass_id == fix_class_id

        run_time = None
        if n_times > 0:
            run_time = timeit(
                'self._nn.nodes[ids[1]].subclass_id += 1; '
                'self._nn.combine_subclasses(ids)',
                number=n_times, globals=locals()
            )
        # rest class id values
        for node in self._nn.nodes.values():
            node.subclass_id = 0
        return correct_marking, run_time

    def test_find_neighbors_local_maxes(self, n_times=0) -> tuple:
        results = {}
        for node_id in self._nn.nodes:
            results[node_id] = self._nn.find_neighbors_local_maxes(node_id)
            # print(f"{node_id:<6}: {results[node_id]}")

        run_time = None
        if n_times > 0:
            run_time = timeit(
                'for node_id in self._nn.nodes: '
                'self._nn.find_neighbors_local_maxes(node_id)',
                number=n_times, globals=locals()
            )
        return results == mock.NEIGHBOR_LOCAL_MAXES, run_time

    # @FIXME: undone all
    def test_mark_subclasses(self, n_times=0):
        pass

    # @FIXME: undone all
    def test_calc_heavy_neighbor_min_dist(self, n_times=0):
        pass

    # @FIXME: undone all
    def test_separate_subclasses(self, n_times=0):
        pass

    def test_remove_noise(self, n_times=0) -> tuple:
        mean_density = np.sum([
            node.density for node in self._nn.nodes.values()
        ])/len(self._nn.nodes)

        self._nn.remove_noise()
        successfully_remove = True
        for node in self._nn.nodes.values():
            successfully_remove &= node.density > mean_density

        run_time = None
        if n_times > 0:
            run_time = timeit('self._nn.remove_noise()',
                              number=n_times, globals=locals())
        self.initialize_tests()  # reset nn state
        return True, run_time

    def test_predict(self, n_times=0) -> tuple:
        signal, test = (6, 3.25), (self._nn.nodes[30].subclass_id, 1)
        hat = self._nn.predict(signal)
        run_time = None
        if n_times > 0:
            run_time = timeit('self._nn.predict(signal)',
                              number=n_times, globals=locals())
        return hat == test, run_time

    def test_find_class_apex(self, n_times=0) -> tuple:
        test, ignore_id = (0, {i for i in range(34)}), 34
        correct_result = True
        for node_id in self._nn.nodes:
            if node_id == ignore_id:
                correct_result &= \
                    ignore_id == self._nn.find_class_apex(ignore_id)[0]
                continue
            correct_result &= test == self._nn.find_class_apex(node_id)
        run_time = None
        if n_times > 0:
            run_time = timeit('self._nn.find_class_apex(33)',
                              number=n_times, globals=locals())
        return correct_result, run_time

    def test_update(self, n_times=0) -> tuple:
        apex_id, subclass_ids = self._nn.find_class_apex(0)
        ignore_id = set(self._nn.nodes.keys()) - subclass_ids
        correct_mark = True
        self._nn.change_class_id(apex_id, -1)
        apexes = self._nn.update()
        for node_id in self._nn.nodes.keys() - ignore_id:
            correct_mark &= self._nn.nodes[node_id].subclass_id == apex_id
        run_time = None
        if n_times > 0:
            run_time = timeit('self._nn.update()',
                              number=n_times, globals=locals())
        return apexes == {apex_id}.union(ignore_id) and correct_mark, run_time

    def run_unit_tests(self, n_times=0):
        params = {
            'n_times': n_times
        }
        self.report_error(self.test_find_winners, "find_winners()", **params)
        self.report_error(self.test_find_neighbors, "find_neighbors()",
                          **params)
        self.report_error(self.test_calc_threshold, "calc_threshold()",
                          **params)
        self.report_error(self.test_create_node, "create_node()")
        self.report_error(self.test_create_edges, "create_edges()")
        self.report_error(self.test_remove_edges, "remove_edges()")
        self.report_error(self.test_remove_node, "remove_node()")
        self.report_error(self.test_update_edges_age, "update_edges_age()",
                          **params)
        self.report_error(self.test_update_node_points, "update_node_points()",
                          **params)
        self.report_error(self.test_update_node_density,
                          "update_node_density()")
        self.report_error(self.test_update_feature_vectors,
                          "update_feature_vectors()", **params)
        self.report_error(self.test_remove_old_ages, "remove_old_ages()",
                          **params)
        self.report_error(self.test_calc_mean_density_in_subclass,
                          "calc_mean_density_in_subclass()", **params)
        self.report_error(self.test_calc_alpha, "calc_alpha()")
        self.report_error(self.test_merge_subclass_condition,
                          "merge_subclass_condition()", **params)
        self.report_error(self.test_change_class_id, "change_class_id()",
                          **params)
        self.report_error(self.test_combine_subclasses, "combine_subclasses()",
                          **params)
        self.report_error(self.test_find_neighbors_local_maxes,
                          "find_neighbors_local_maxes()", **params)
        # self.report_error(self.test_mark_subclasses, "mark_subclasses",
        #                   **params)
        # self.report_error(self.test_calc_mean_density_in_subclass,
        #                   "calc_mean_density_in_subclass", **params)
        # self.report_error(self.test_separate_subclasses, "separate_subclasses",
        #                   **params)
        self.report_error(self.test_remove_noise, "remove_noise", **params)
        self.report_error(self.test_predict, "predict", **params)
        self.report_error(self.test_find_class_apex, "find_class_apex",
                          **params)
        self.report_error(self.test_update, "update", **params)


class TrainTest(BasicTest):
    pass
