from timeit import timeit
from copy import deepcopy
import numpy as np
try:
    from dev.ESOINN import ESOINNNode, EnhancedSelfOrganizingIncrementalNN
    from dev.mock import load_mock, NEIGHBOR_LOCAL_MAXES
    from dev.commons import Plotter
except ImportError as error:
    print(error.args)
    from .ESOINN import ESOINNNode, EnhancedSelfOrganizingIncrementalNN
    from .mock import load_mock, NEIGHBOR_LOCAL_MAXES
    from .commons import Plotter


class UnitTest(Plotter):
    def __init__(self, nn, logging_level="debug"):
        super().__init__(nn, logging_level)
        self.__success = True

    def reset_tests(self) -> None:
        if not isinstance(self._nn, EnhancedSelfOrganizingIncrementalNN):
            raise ValueError(f"Tests are compatible only with "
                             f"EnhancedSelfOrganizingIncrementalNN class, got:"
                             f" {type(self._nn)}")
        g = load_mock()
        self._nn.nodes.clear()  # for clean configuration
        for node_id in g.nodes:
            node = g.nodes[node_id]
            self._nn.nodes[node_id] = ESOINNNode(node.feature_vector)
            self._nn.nodes[node_id]._ESOINNNode__density = node.density
            self._nn.nodes[node_id].update_accumulate_signals()
            self._nn.nodes[node_id].subclass_id = node.subclass_id
            if node.subclass_id != -1:
                self._nn.nodes[node_id].subclass_id = node.subclass_id
            else:
                self._nn.nodes[node_id].subclass_id = 0
        self._nn.unique_id = len(g.nodes)  # set correct unique id

        self._nn.neighbors.clear()
        self._nn.edges.clear()
        self._nn.neighbors = deepcopy(g.neighbors)
        self._nn.edges = deepcopy(g.edges)

    def report_error(self, test, name, **kwargs) -> None:
        res, *time = test(**kwargs)
        if not res:
            self._logger.error(f"TEST: {name}")
            self.__success = False
        # else:
        #     self._logger.info(f"TEST: {name}")
        if time and kwargs.get('n_times', None):
            self._logger.debug(f"{time[0]:.5}\tfor {kwargs['n_times']} {name}")

    def calc_run_time(self, target_call: str, n_times=0, **kwargs):
        if n_times <= 0:
            return
        return timeit(f"{target_call}", number=n_times, globals=locals())

    # @TODO: use dists[] instead of dist0, dist1
    def test_find_winners(self, n_times=0) -> tuple:
        feature_vector = [5.5, 3]
        winners, dists = self._nn.find_winners(feature_vector)
        dist0 = dists[1] == self._nn.metrics([5.75, 3.25], feature_vector)
        dist1 = dists[1] == self._nn.metrics([6.25, 3.25], feature_vector)
        return winners == (30, 31) and dist0 and dist1, \
            self.calc_run_time(f"self._nn.find_winners({feature_vector})",
                               n_times)

    def test_find_neighbors(self, n_times=0) -> tuple:
        id1 = 20
        successfully_found = True
        neighbors = self._nn.find_neighbors(id1)
        successfully_found &= neighbors == self._nn.neighbors.get(id1, set())
        neighbors = self._nn.find_neighbors(id1, depth=2)
        successfully_found &= neighbors == {17, 18, 19, 21, 22, 23, 6}
        return successfully_found, \
            self.calc_run_time(f"self._nn.find_neighbors({id1})", n_times)

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
        return right_calc,\
            self.calc_run_time(f"self._nn.calc_threshold({id1}); "
                               f"self._nn.calc_threshold({id0})", n_times)

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

    def test_update_edges_age(self, n_times=0) -> tuple:
        id1 = 19
        step = 2
        successfully_update = True
        self._nn.update_edges_age(id1, step=step)
        neighbors_id = self._nn.find_neighbors(id1)
        for neighbor_id in neighbors_id:
            successfully_update &= \
                self._nn.edges.get((min(id1, neighbor_id),
                                    max(id1, neighbor_id)), False) == step
        return successfully_update, \
            self.calc_run_time(f"self._nn.update_edges_age({id1}, step=0)",
                               n_times)

    def test_update_node_points(self, n_times=0) -> tuple:
        ids = (34, 19)
        correct_points = True
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
                    correct_points &= \
                        self._nn.nodes[i].points == points+1/(1+mean_dist)**2
                else:
                    correct_points &= points + 1 == self._nn.nodes[i].points
        return correct_points, \
            self.calc_run_time(f"self._nn.update_node_points({ids[1]}, "
                               f"{neighbors})", n_times)

    # @CHECKME: Access to a private fields
    def test_update_node_density(self, n_times) -> tuple:
        Test_check = True
        id1 = 0
        self._nn.nodes[id1]._ESOINNNode__total_points = 0
        self._nn.update_node_points(id1, list(self._nn.neighbors[id1]))
        self._nn.nodes[id1]._ESOINNNode__acc_signals = 2
        val_density = self._nn.nodes[id1].points / \
                      self._nn.nodes[id1].accumulate_signals

        self._nn.nodes[id1]._ESOINNNode__total_points = 0
        self._nn.update_node_density(id1, list(self._nn.neighbors[id1]))
        Test_check &= val_density == self._nn.nodes[id1].density

        # Return to start state
        self.reset_tests()
        return Test_check, \
                self.calc_run_time(f"self._nn.update_node_density({id1}, "
                                   f"{list(self._nn.neighbors[id1])})",
                                   n_times)

    # @CHECKME: Access to a private fields
    def test_update_feature_vectors(self, n_times=0) -> tuple:
        id1 = 0
        input_signal = [2, 2.5]
        win_learn_step = lambda t: 1 / t
        neig_learn_step = lambda t: 1 / (100*t)

        self._nn.nodes[id1]._ESOINNNode__acc_signals = 3
        for neighbor_id in self._nn.neighbors[id1]:
            self._nn.nodes[neighbor_id]._ESOINNNode__acc_signals = 2

        winner_feature_vec = self._nn.nodes[id1].feature_vector + \
            (input_signal - self._nn.nodes[id1].feature_vector) * \
            win_learn_step(self._nn.nodes[id1].accumulate_signals)

        neighbors_feature_vec = []
        for neighbor_id in self._nn.neighbors[id1]:
            neighbors_feature_vec.append(
                self._nn.nodes[neighbor_id].feature_vector + \
                (input_signal - self._nn.nodes[neighbor_id].feature_vector) * \
                neig_learn_step(self._nn.nodes[id1].accumulate_signals))
        self._nn.update_feature_vectors(id1,
                                        input_signal,
                                        self._nn.neighbors[id1])
        Test_check = True
        Test_check &= list(winner_feature_vec) == list(self._nn.nodes[id1].feature_vector)
        for i, neighbor_id in enumerate(self._nn.neighbors[id1]):
            Test_check &= list(neighbors_feature_vec[i]) == \
                          list(self._nn.nodes[neighbor_id].feature_vector)

        # Return to start state
        self.reset_tests()
        return Test_check, \
               self.calc_run_time(f"self._nn.update_feature_vectors("
                                  f"{id1},"
                                  f"{input_signal},"
                                  f"{self._nn.neighbors[id1]})", n_times)

    def test_remove_old_ages(self, n_times=0) -> tuple:
        ids = (1, 34)
        successfully_remove = True
        self._nn.create_edges(ids)
        self._nn.edges[ids] = self._nn.max_age + 1
        self._nn.remove_old_ages()
        successfully_remove &= self._nn.edges.get(ids, True)
        successfully_remove &= ids[1] not in self._nn.neighbors.get(ids[0], {})
        successfully_remove &= ids[0] not in self._nn.neighbors.get(ids[1], {})
        return successfully_remove, \
            self.calc_run_time("self._nn.remove_old_ages()", n_times)

    def test_calc_mean_density_in_subclass(self, n_times=0) -> tuple:
        id0, id1 = 0, 1
        _, node_ids = self._nn.find_class_apex(id0)
        density = 0
        for node_id in node_ids:
            density += self._nn.nodes[node_id].density
        val_mean_density = density/len(node_ids)
        mean_density = self._nn.calc_mean_density_in_subclass(id1)
        return mean_density == val_mean_density, self.calc_run_time(
            f"self._nn.calc_mean_density_in_subclass({id1})", n_times)

    def test_calc_alpha(self, n_times) -> tuple:
        id1 = 1
        apex_density = 15
        mean_density = self._nn.calc_mean_density_in_subclass(id1)
        if 2*mean_density >= apex_density:
            val_alpha =  0
        elif 3*mean_density >= apex_density:
            val_alpha =  0.5
        else:
            val_alpha = 1
        alpha = self._nn.calc_alpha(id1, apex_density)
        return alpha == val_alpha, self.calc_run_time(
               f"self._nn.calc_alpha({id1}, {apex_density})", n_times)

    def test_merge_subclass_condition(self, n_times=0) -> tuple:
        self._nn.separate_subclass()
        Test_check = self._nn.merge_subclass_condition([1,11])
        Test_check &= self._nn.merge_subclass_condition([12,15]) != True

        # Return to start state
        self.reset_tests()
        return Test_check == True, self.calc_run_time(
            f"self._nn.merge_subclass_condition({[0, 1]})", n_times)

    def test_change_class_id(self, n_times=0) -> tuple:
        id1, class_id = 0, 1
        self._nn.change_class_id(id1, class_id)
        correct_marking = True
        for node_id in self._nn.nodes:  # node 34 has different class
            if node_id == 34:
                continue
            correct_marking &= self._nn.nodes[node_id].subclass_id == class_id
        return correct_marking, self.calc_run_time(
            f"self._nn.change_class_id({id1}, {class_id})", n_times)

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

        run_time = self.calc_run_time(
            f"self._nn.nodes[{ids[1]}].subclass_id += 1; "
            f"self._nn.combine_subclasses({ids})", n_times
        )
        # rest class id values
        for node in self._nn.nodes.values():
            node.subclass_id = 0
        return correct_marking, run_time

    def test_find_local_maxes(self, n_times=0) -> tuple:
        maxes = set(range(10))
        maxes.add(34)
        apexes_found = self._nn.find_local_maxes()
        return apexes_found == maxes, \
            self.calc_run_time("self._nn.find_local_maxes()", n_times)

    def test_continue_mark(self, n_times=0) -> tuple:
        id1, subclass_id = 1, 1
        val_overlap_ids = {11, 12, 29, 30, 32}
        val_visited = {1, 33}
        overlap_ids, visited = self._nn.continue_mark([id1], subclass_id, set())
        return val_overlap_ids == overlap_ids and val_visited == visited, \
               self.calc_run_time(f"self._nn.continue_mark({[id1]},"
                                  f"{subclass_id},"
                                  f"{set()})",
                                  n_times)

    def test_get_nearest_neighbor(self, n_times=0) -> tuple:
        id1 = 29
        val_nearest_id = 7
        nearest_id = self._nn.get_nearest_neighbor(id1, set())
        return val_nearest_id == nearest_id, \
               self.calc_run_time(f"self._nn.get_nearest_neighbor({id1}, "
                                  f"{set()})",
                                  n_times)

    def test_check_overlap(self, n_times=0) -> tuple:
        Test_check = True
        id1, id2 = 20, 9
        val_continue = {20}
        overlap_ids = {id1, id2}
        visited = {18, 19, 30}
        neighbors = self._nn.neighbors[id2]
        continue_id = self._nn.check_overlap(overlap_ids, visited)

        # For test with removing edges
        for neighbor_id in neighbors:
            if neighbor_id in visited:
                Test_check &= id2 not in self._nn.neighbors[neighbor_id]

        # For test without removing edges
        Test_check &= continue_id == val_continue

        return Test_check, \
               self.calc_run_time(f"self._nn.check_overlap({overlap_ids}, "
                                  f"{visited})",
                                  n_times)

    def test_separate_subclass(self, n_times=0) -> tuple:
        neighbors = {0: {15, 16, 27},
                     1: {12, 33},
                     4: {10, 11},
                     5: {14, 17},
                     6: {23, 24, 25},
                     7: {26, 28, 29},
                     8: {30},
                     9: {31},
                     10: {4},
                     11: {4},
                     12: {1, 13},
                     13: {12},
                     14: {5},
                     15: {0},
                     16: {0},
                     17: {5, 18},
                     18: {17, 19, 20},
                     19: {18, 20},
                     20: {18, 19, 21, 22},
                     21: {20, 22},
                     22: {20, 21},
                     23: {6, 24},
                     24: {6, 23, 25},
                     25: {6, 24},
                     26: {7, 28, 29},
                     27: {0},
                     28: {7, 26},
                     29: {7, 26},
                     30: {8},
                     31: {9},
                     32: {33},
                     33: {1, 32}}

        edges = {(0, 15): 0,
                 (0, 16): 0,
                 (0, 27): 0,
                 (1, 12): 0,
                 (1, 33): 0,
                 (4, 10): 0,
                 (4, 11): 0,
                 (5, 14): 0,
                 (5, 17): 0,
                 (6, 23): 0,
                 (6, 24): 0,
                 (6, 25): 0,
                 (7, 26): 0,
                 (7, 28): 0,
                 (7, 29): 0,
                 (8, 30): 0,
                 (9, 31): 0,
                 (12, 13): 0,
                 (17, 18): 0,
                 (18, 19): 0,
                 (18, 20): 0,
                 (19, 20): 0,
                 (20, 21): 0,
                 (20, 22): 0,
                 (21, 22): 0,
                 (23, 24): 0,
                 (24, 25): 0,
                 (26, 28): 0,
                 (26, 29): 0,
                 (32, 33): 0}

        subclasses = {
                    0: 0,
                    1: 1,
                    2: 2,
                    3: 3,
                    4: 4,
                    5: 5,
                    6: 6,
                    7: 7,
                    8: 8,
                    9: 9,
                    10: 4,
                    11: 4,
                    12: 1,
                    13: 1,
                    14: 5,
                    15: 0,
                    16: 0,
                    17: 5,
                    18: 5,
                    19: 5,
                    20: 5,
                    21: 5,
                    22: 5,
                    23: 6,
                    24: 6,
                    25: 6,
                    26: 7,
                    27: 0,
                    28: 7,
                    29: 7,
                    30: 8,
                    31: 9,
                    32: 1,
                    33: 1,
                    34: 34}

        self._nn.separate_subclass()
        Test_check = True

        # Zeroing out all edges age (because some edges age = 2)
        for edge in self._nn.edges:
            self._nn.edges[edge] = 0

        Test_check &= self._nn.neighbors == neighbors
        Test_check &= self._nn.edges == edges

        # Create subclass dict
        test_subclasses = {} # key - id, value - subclass_id
        for node_id in self._nn.nodes:
            test_subclasses[node_id] = self._nn.nodes[node_id].subclass_id

        Test_check &= test_subclasses == subclasses
        return Test_check, self.calc_run_time("self._nn.separate_subclass()",
                                              n_times)

    def test_remove_noise(self, n_times=0) -> tuple:
        mean_density = np.sum([
            node.density for node in self._nn.nodes.values()
        ])/len(self._nn.nodes)

        self._nn.remove_noise()
        successfully_remove = True
        for node in self._nn.nodes.values():
            successfully_remove &= node.density > mean_density

        run_time = self.calc_run_time("self._nn.remove_noise()", n_times)
        self.reset_tests()  # reset nn state
        return True, run_time

    def test_predict(self, n_times=0) -> tuple:
        signal, test = (6, 3.25), (self._nn.nodes[30].subclass_id, 1)
        hat = self._nn.predict(signal)
        return hat == test, self.calc_run_time(f"self._nn.predict({signal})",
                                               n_times)

    def test_find_class_apex(self, n_times=0) -> tuple:
        test, ignore_id = (0, {i for i in range(34)}), 34
        correct_result = True
        for node_id in self._nn.nodes:
            if node_id == ignore_id:
                correct_result &= \
                    ignore_id == self._nn.find_class_apex(ignore_id)[0]
                continue
            correct_result &= test == self._nn.find_class_apex(node_id)
        return correct_result, \
            self.calc_run_time("self._nn.find_class_apex(33)", n_times)

    def test_update(self, n_times=0) -> tuple:
        apex_id, subclass_ids = self._nn.find_class_apex(0)
        ignore_id = set(self._nn.nodes.keys()) - subclass_ids
        correct_mark = True
        self._nn.change_class_id(apex_id, -1)
        apexes = self._nn.update()
        for node_id in self._nn.nodes.keys() - ignore_id:
            correct_mark &= self._nn.nodes[node_id].subclass_id == apex_id
        return apexes.keys() == {apex_id}.union(ignore_id) and correct_mark, \
            self.calc_run_time("self._nn.update()", n_times)

    def run_tests(self, n_times=0) -> bool:
        self.reset_tests()
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
                          "update_node_density()", **params)
        self.report_error(self.test_update_feature_vectors,
                          "update_feature_vectors()", **params)
        self.report_error(self.test_remove_old_ages, "remove_old_ages()",
                          **params)
        self.report_error(self.test_calc_mean_density_in_subclass,
                          "calc_mean_density_in_subclass()", **params)
        self.report_error(self.test_calc_alpha, "calc_alpha()", **params)
        self.report_error(self.test_merge_subclass_condition,
                          "merge_subclass_condition()", **params)
        self.report_error(self.test_change_class_id, "change_class_id()",
                          **params)
        self.report_error(self.test_combine_subclasses, "combine_subclasses()",
                          **params)
        self.report_error(self.test_find_local_maxes, "find_local_maxes()",
                          **params)
        self.report_error(self.test_continue_mark, "continue_mark()", **params)
        self.report_error(self.test_get_nearest_neighbor,
                          "get_nearest_neighbor()",
                          **params)
        self.report_error(self.test_check_overlap, "check_overlap()", **params)
        self.report_error(self.test_separate_subclass, "separate_subclass()",
                          **params)
        self.report_error(self.test_remove_noise, "remove_noise", **params)
        self.report_error(self.test_predict, "predict", **params)
        self.report_error(self.test_find_class_apex, "find_class_apex",
                          **params)
        self.report_error(self.test_update, "update", **params)

        return self.__success

class TrainTest(Plotter):
    pass
