from timeit import timeit
from copy import deepcopy
import numpy as np
import re
try:
    from dev.ESOINN import _ESOINNNode, EnhancedSelfOrganizingIncrementalNN, \
        learning_rate_generator
    from dev.mock import load_mock, NEIGHBOR_LOCAL_MAXES
    from dev.commons import Plotter
except ImportError as error:
    print(error.args)
    from .ESOINN import _ESOINNNode, EnhancedSelfOrganizingIncrementalNN, \
        learning_rate_generator
    from .mock import load_mock, NEIGHBOR_LOCAL_MAXES
    from .commons import Plotter


class UnitTest(Plotter):
    def __init__(self, nn, logging_level="debug"):
        super().__init__(nn, logging_level)
        self.__success = True
        self.__state = self._nn.current_state(deep=False)

    def _reset_tests(self) -> None:
        if not isinstance(self._nn, EnhancedSelfOrganizingIncrementalNN):
            raise ValueError(f"Tests are compatible only with "
                             f"EnhancedSelfOrganizingIncrementalNN class, got:"
                             f" {type(self._nn)}")
        g = load_mock()
        self._nn._reset()  # for clean configuration
        self._nn._configure(
            nodes=[{
                'id': node_id,
                'features': g.nodes[node_id].features,
                'density': g.nodes[node_id].density,
                'subclass_id': g.nodes[node_id].subclass_id
            } for node_id in g.nodes],
            neighbors=deepcopy(g.neighbors),
            edges=deepcopy(g.edges)
        )
        self.__update_state()

    def __update_state(self) -> None:
        self.__state = self._nn.current_state(deep=False)

    def __report_error(self, test: classmethod, **kwargs) -> None:
        self.__update_state()
        res, *time = test(**kwargs)
        name = re.sub(r"\b_test_", '', str(test.__name__))
        if not res:
            self._logger.error(f"TEST: {name}")
            self.__success = False
        if time and kwargs.get('n_times', False):
            self._logger.debug(f"{str(time[0]):10.10}\t"
                               f"for {kwargs['n_times']} {name}")

    def _calc_run_time(self, target_call: str, n_times=0, **kwargs):
        if n_times <= 0:
            return
        return timeit(f"{target_call}", number=n_times, globals=locals())

    def _test_find_winners(self, n_times=0) -> tuple:
        feature_vector = [5.5, 3]
        found = True
        winners, dists = self._nn._find_winners(feature_vector)
        found &= dists[0] == self._nn.metrics([5.75, 3.25], feature_vector)
        found &= dists[1] == self._nn.metrics([6.25, 3.25], feature_vector)
        return winners == (30, 31) and found, \
            self._calc_run_time(f"self._nn._find_winners({feature_vector})",
                                n_times)

    def _test_find_neighbors(self, n_times=0) -> tuple:
        id1, id0 = 20, 34
        successfully_found = True
        neighbors = self._nn._find_neighbors(id1)
        successfully_found &= neighbors == \
            self.__state['neighbors'].get(id1, set())
        neighbors = self._nn._find_neighbors(id1, depth=2)
        successfully_found &= neighbors == {17, 18, 19, 21, 22, 23, 6}
        neighbors = self._nn._find_neighbors(id1, depth=-1)
        successfully_found &= neighbors == \
            set(self.__state['nodes'].keys()) - {id0, id1}
        return successfully_found, \
            self._calc_run_time(f"self._nn._find_neighbors({id1})", n_times)

    def _test_calc_threshold(self, n_times=0) -> tuple:
        right_calc = True
        id0, id1 = 34, 19
        _, dist = self._nn._find_winners(self.__state['nodes'][id0].features)
        rc = self.__state['configuration']['Rc']

        for i in range(1, 3):
            self._nn._set_rc(i)
            neighbors = self._nn._find_neighbors(id1, depth=i)
            max_dist = max([
                self._nn.metrics(
                    self.__state['nodes'][neighbor_id].features,
                    self.__state['nodes'][id1].features
                ) for neighbor_id in neighbors
            ])
            right_calc &= self._nn._calc_threshold(id0) == dist[1]
            right_calc &= self._nn._calc_threshold(id1) == max_dist
        self._nn._set_rc(rc)
        return right_calc,\
            self._calc_run_time(f"self._nn._calc_threshold({id1}); "
                               f"self._nn._calc_threshold({id0})", n_times)

    def _test_create_node(self):
        last_id = self.__state['last_node_id']
        feature_vector = [1, 0.5]
        self._nn._create_node(feature_vector)
        self.__update_state()
        return [
            last_id + 1 == self.__state['last_node_id']
            and len(self.__state['nodes']) == self.__state['last_node_id'] + 1
            and list(self.__state['nodes'][last_id+1].features) ==
            feature_vector
        ]

    def _test_create_edges(self):
        id1 = self.__state['last_node_id']
        if not self.__state['nodes'].get(id1, None):
            return
        self._nn._create_edges([id1, 18, 19, 20])
        successfully_add = True
        self.__update_state()
        for i in range(18, 21):
            successfully_add &= self.__state['edges'].get((i, id1), False) == 0
            successfully_add &= id1 in self.__state['neighbors'].get(i, set())

        return [successfully_add]

    def _test_remove_edges(self):
        id0, id1 = 20, self.__state['last_node_id']
        if not self.__state['nodes'].get(id1, None):
            return
        self._nn._remove_edges([id1, id0])
        self.__update_state()
        successfully_remove = True
        successfully_remove &= self.__state['edges'].get((id0, id1), True)
        successfully_remove &= id1 not in self.__state['neighbors'].get(id0,
                                                                        set())
        return [successfully_remove]

    def _test_remove_node(self):
        id1 = self.__state['last_node_id']
        if not self.__state['nodes'].get(id1, None):
            return
        self._nn._remove_node(id1)
        self.__update_state()
        successfully_remove = True
        for i in range(18, 20):
            successfully_remove &= self.__state['edges'].get((i, id1), True)
            successfully_remove &= id1 not in \
                self.__state['neighbors'].get(i,set())
        return [successfully_remove]

    def _test_update_edges_age(self, n_times=0) -> tuple:
        id1 = 19
        step = 2
        rc = self.__state['configuration']['Rc']
        successfully_update = True
        self._nn._update_edges_age(id1, step=step)
        neighbors_id = self._nn._find_neighbors(id1, depth=rc)
        for neighbor_id in neighbors_id:
            edge = (min(id1, neighbor_id), max(id1, neighbor_id))
            if self.__state['edges'].get(edge, None):
                successfully_update &= self.__state['edges'].get(edge) == step
        self._reset_tests()
        self._nn._set_rc(rc+1)
        self._nn._update_edges_age(id1, step=step)
        neighbors_id = self._nn._find_neighbors(id1, depth=rc)
        for neighbor_id in neighbors_id:
            edge = (min(id1, neighbor_id), max(id1, neighbor_id))
            if self.__state['edges'].get(edge, None):
                successfully_update &= self.__state['edges'].get(edge) == step
        self._nn._set_rc(rc)
        return successfully_update, \
            self._calc_run_time(f"self._nn._update_edges_age({id1}, step=0)",
                                n_times)

    def _test_update_node_points(self, n_times=0) -> tuple:
        ids = (34, 19)
        correct_points = True
        for rc in range(1, 3):
            for i in ids:
                neighbors = self._nn._find_neighbors(i, depth=rc)
                points = self.__state['nodes'][i].points
                self._nn._update_node_points(i, neighbors)
                if neighbors:
                    mean_dist = 1/len(neighbors)*np.sum([
                        self._nn.metrics(
                            self.__state['nodes'][i].features,
                            self.__state['nodes'][neighbor_id].features
                        ) for neighbor_id in neighbors
                    ])
                    correct_points &= \
                        self.__state['nodes'][i].points == \
                        points + 1/(1 + mean_dist)**2
                else:
                    correct_points &= points + 1 == \
                                      self.__state['nodes'][i].points
        return correct_points, \
            self._calc_run_time(f"self._nn._update_node_points({ids[1]}, "
                               f"{neighbors})", n_times)

    def _test_update_node_density(self, n_times) -> tuple:
        test_check = True
        id1 = 0
        neighbors = self._nn._find_neighbors(id1)
        self.__state['nodes'][id1].reset_points()
        self._nn._update_node_points(id1, neighbors)
        self.__state['nodes'][id1]._ESOINNNode__acc_signals = 2
        val_density = self.__state['nodes'][id1].points / \
                      self.__state['nodes'][id1].accumulate_signals
        self.__state['nodes'][id1].reset_points()
        self._nn._update_node_density(id1, neighbors)
        test_check &= val_density == self.__state['nodes'][id1].density
        self._reset_tests()
        return test_check, self._calc_run_time(
            f"self._nn._update_node_density({id1}, {neighbors})", n_times)

    def _test_update_feature_vectors(self, n_times=0) -> tuple:
        id1 = 0
        input_signal = np.array([2, 2.5])
        win_learn_step = learning_rate_generator()
        neig_learn_step = learning_rate_generator(100)
        neighbors = self.__state['neighbors'][id1]

        self.__state['nodes'][id1]._ESOINNNode__acc_signals = 3
        for neighbor_id in neighbors:
            self.__state['nodes'][neighbor_id]._ESOINNNode__acc_signals = 2

        winner_feature_vec = self.__state['nodes'][id1].features + \
            (input_signal - self.__state['nodes'][id1].features) * \
            win_learn_step(self.__state['nodes'][id1].accumulate_signals)

        neighbors_feature_vec = []
        for neighbor_id in neighbors:
            neighbors_feature_vec.append(
                self.__state['nodes'][neighbor_id].features +
                (input_signal - self.__state['nodes'][neighbor_id].features) *
                neig_learn_step(self.__state['nodes'][id1].accumulate_signals))
        self._nn._update_feature_vectors(id1, input_signal, neighbors)
        test_check = True
        test_check &= list(winner_feature_vec) == \
            list(self.__state['nodes'][id1].features)
        for i, neighbor_id in enumerate(neighbors):
            test_check &= list(neighbors_feature_vec[i]) == \
                          list(self.__state['nodes'][neighbor_id].features)

        self._reset_tests()
        return test_check, self._calc_run_time(
            f"self._nn._update_feature_vectors({id1}, {list(input_signal)}, "
            f"{neighbors})", n_times)

    def _test_remove_old_ages(self, n_times=0) -> tuple:
        ids = (1, 34)
        successfully_remove = True
        self._nn._create_edges(ids)
        self.__state['edges'][ids] = \
            self.__state['configuration']['max_age'] + 1
        self._nn._remove_old_edges()
        successfully_remove &= self.__state['edges'].get(ids, True)
        successfully_remove &= ids[1] not in \
            self.__state['neighbors'].get(ids[0], {})
        successfully_remove &= ids[0] not in \
            self.__state['neighbors'].get(ids[1], {})
        return successfully_remove, \
            self._calc_run_time("self._nn._remove_old_edges()", n_times)

    # @FIXME: undone
    def _test_calc_mean_density_in_subclass(self, n_times=0) -> tuple:
        id0, id1 = 0, 1
        _, node_ids = self._nn._find_class_apex(id0)
        density = 0
        for node_id in node_ids:
            density += self.__state['nodes'][node_id].density
        val_mean_density = density/len(node_ids)
        mean_density = self._nn._calc_mean_density_in_subclass(id1)
        return True, self._calc_run_time(
            f"self._nn._calc_mean_density_in_subclass({id1})", n_times)

    def _test_calc_alpha(self, n_times) -> tuple:
        id1 = 1
        apex_density = 15
        mean_density = self._nn._calc_mean_density_in_subclass(id1)
        if 2*mean_density >= apex_density:
            val_alpha = 0
        elif 3*mean_density >= apex_density:
            val_alpha = 0.5
        else:
            val_alpha = 1
        alpha = self._nn._calc_alpha(id1, apex_density)
        return alpha == val_alpha, self._calc_run_time(
               f"self._nn._calc_alpha({id1}, {apex_density})", n_times)

    def test_merge_subclass_condition(self, n_times=0) -> tuple:
        self._nn._separate_subclasses()
        test_check = self._nn._merge_subclass_condition([1, 11])
        test_check &= not self._nn._merge_subclass_condition([12, 15])

        # Return to start state
        self._reset_tests()
        return test_check, self._calc_run_time(
            f"self._nn.merge_subclass_condition({[0, 1]})", n_times)

    def _test_change_class_id(self, n_times=0) -> tuple:
        id1, class_id = 0, 1
        self._nn._change_class_id(id1, class_id)
        correct_marking = True
        for node_id in self.__state['nodes']:
            if node_id == 34:  # node 34 has different class
                continue
            correct_marking &= \
                self.__state['nodes'][node_id].subclass_id == class_id
        return correct_marking, self._calc_run_time(
            f"self._nn._change_class_id({id1}, {class_id})", n_times)

    def _test_combine_subclasses(self, n_times=0) -> tuple:
        ids = (0, 34)
        correct_marking = True

        self.__state['nodes'][ids[1]].subclass_id = -1
        self._nn._combine_subclasses(ids)
        for node in self.__state['nodes'].values():
            correct_marking &= node.subclass_id != -1

        self.__state['nodes'][ids[1]].subclass_id += 100
        self._nn._combine_subclasses(ids)
        fix_class_id = self.__state['nodes'][ids[1]].subclass_id
        for node in self.__state['nodes'].values():
            correct_marking &= node.subclass_id == fix_class_id

        run_time = self._calc_run_time(
            f"self._nn._EnhancedSelfOrganizingIncrementalNN__nodes[{ids[1]}]."
            f"subclass_id += 1; self._nn._combine_subclasses({ids})", n_times
        )
        self._reset_tests()
        return correct_marking, run_time

    def _test_find_local_maxes(self, n_times=0) -> tuple:
        maxes = set(range(10))
        maxes.add(34)
        apexes_found = self._nn._find_local_maxes()
        return apexes_found == maxes, \
            self._calc_run_time("self._nn._find_local_maxes()", n_times)

    def test_continue_mark(self, n_times=0) -> tuple:
        id1, subclass_id = 1, 1
        val_overlap_ids = {11, 12, 29, 30, 32}
        val_visited = {1, 33}
        overlap_ids, visited = self._nn._continue_mark([id1], subclass_id, set())
        return val_overlap_ids == overlap_ids and val_visited == visited, \
               self._calc_run_time(f"self._nn.continue_mark({[id1]},"
                                  f"{subclass_id},"
                                  f"{set()})",
                                   n_times)

    def test_get_nearest_neighbor(self, n_times=0) -> tuple:
        id1 = 29
        val_nearest_id = 7
        nearest_id = self._nn._get_nearest_neighbor(id1, set())
        return val_nearest_id == nearest_id, \
               self._calc_run_time(f"self._nn.get_nearest_neighbor({id1}, "
                                  f"{set()})",
                                   n_times)

    def test_check_overlap(self, n_times=0) -> tuple:
        Test_check = True
        id1, id2 = 20, 9
        val_continue = {20}
        overlap_ids = {id1, id2}
        visited = {18, 19, 30}
        neighbors = self._nn.neighbors[id2]
        continue_id = self._nn._check_overlap(overlap_ids, visited)

        # For test with removing edges
        for neighbor_id in neighbors:
            if neighbor_id in visited:
                Test_check &= id2 not in self._nn.neighbors[neighbor_id]

        # For test without removing edges
        Test_check &= continue_id == val_continue

        return Test_check, \
               self._calc_run_time(f"self._nn.check_overlap({overlap_ids}, "
                                  f"{visited})",
                                   n_times)

    def test_separate_subclasses(self, n_times=0) -> tuple:
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

        self._nn._separate_subclasses()
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
        return Test_check, self._calc_run_time("self._nn.separate_subclasses()",
                                               n_times)

    # @FIXME: undone
    def _test_remove_noise(self, n_times=0) -> tuple:
        mean_density = np.sum([
            node.density for node in self.__state['nodes'].values()
        ])/len(self.__state['nodes'])
        neighbors = deepcopy(self.__state['neighbors'])

        self._nn._remove_noise()
        self.__update_state()
        successfully_remove = True
        for node_id in self.__state['nodes']:
            neighbors_count = len(neighbors.get(node_id, ()))
            node_density = self.__state['nodes'][node_id].density
            if not neighbors_count or neighbors_count == 1 and node_density < \
                self.__state['configuration']['C2'] * mean_density \
                or neighbors_count == 2 and node_density < \
                    self.__state['configuration']['C1'] * mean_density:
                successfully_remove = False
                break

        run_time = self._calc_run_time("self._nn._remove_noise()", n_times)
        self._reset_tests()
        return True, run_time

    def _test_predict(self, n_times=0) -> tuple:
        signal, test = (6, 3.25), (self.__state['nodes'][30].subclass_id, 1)
        hat = self._nn.predict(signal)
        return hat == test, self._calc_run_time(f"self._nn.predict({signal})",
                                                n_times)

    def _test_find_class_apex(self, n_times=0) -> tuple:
        test, ignore_id = (0, {i for i in range(34)}), 34
        correct_result = True
        for node_id in self.__state['nodes']:
            if node_id == ignore_id:
                correct_result &= \
                    ignore_id == self._nn._find_class_apex(ignore_id)[0]
                continue
            correct_result &= test == self._nn._find_class_apex(node_id)
        return correct_result, \
            self._calc_run_time("self._nn._find_class_apex(33)", n_times)

    def _test_update(self, n_times=0) -> tuple:
        apex_id, subclass_ids = self._nn._find_class_apex(0)
        ignore_id = set(self.__state['nodes'].keys()) - subclass_ids
        correct_mark = True
        self._nn._change_class_id(apex_id, -1)
        apexes = self._nn.update(remove_noise=False)
        for node_id in self.__state['nodes'].keys() - ignore_id:
            correct_mark &= \
                self.__state['nodes'][node_id].subclass_id == apex_id
        return apexes.keys() == {apex_id}.union(ignore_id) and correct_mark, \
            self._calc_run_time("self._nn.update()", n_times)

    def run_tests(self, n_times=0) -> bool:
        self._reset_tests()
        params = {
            'n_times': n_times
        }
        self.__report_error(self._test_find_winners, **params)
        self.__report_error(self._test_find_neighbors, **params)
        self.__report_error(self._test_calc_threshold, **params)
        self.__report_error(self._test_create_node)
        self.__report_error(self._test_create_edges)
        self.__report_error(self._test_remove_edges)
        self.__report_error(self._test_remove_node)
        self.__report_error(self._test_update_edges_age, **params)
        self.__report_error(self._test_update_node_points, **params)
        self.__report_error(self._test_update_node_density, **params)
        self.__report_error(self._test_update_feature_vectors, **params)
        self.__report_error(self._test_remove_old_ages, **params)
        self.__report_error(self._test_calc_mean_density_in_subclass, **params)
        self.__report_error(self._test_calc_alpha, **params)
        # self.__report_error(self.test_merge_subclass_condition, **params)
        self.__report_error(self._test_change_class_id, **params)
        self.__report_error(self._test_combine_subclasses, **params)
        self.__report_error(self._test_find_local_maxes, **params)
        # self.__report_error(self.test_continue_mark, **params)
        # self.__report_error(self.test_get_nearest_neighbor, **params)
        # self.__report_error(self.test_check_overlap, **params)
        # self.__report_error(self.test_separate_subclasses, **params)
        self.__report_error(self._test_remove_noise, **params)
        self.__report_error(self._test_predict, **params)
        self.__report_error(self._test_find_class_apex, **params)
        self.__report_error(self._test_update, **params)
        self._reset_tests()

        return self.__success


class TrainTest(Plotter):
    pass
