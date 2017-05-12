from copy import deepcopy
import numpy as np
from time import time
try:
    from dev.commons import enable_logging
except ImportError as error:
    print(error.args)
    from .commons import enable_logging


def euclidean_distance(x, y) -> float:
    return np.sqrt(np.sum(np.square(np.array(x) - np.array(y))))


def learning_rate_generator(k=1, n=1) -> callable:
    return lambda t: n/(k*t)


class _ESOINNNode:
    def __init__(self, feature_vector):
        self.__weights = np.array(feature_vector, dtype=float)
        self.__acc_signals = 0
        self.__total_points = float(0)
        self.__density = float(0)
        self.__subclass_id = -1

    def __repr__(self) -> str:
        return f"{str(self.features):^30} | {self.subclass_id:10} | " \
               f"{float(self.density):13.10} | {self.points:13.10} | " \
               f"{self.accumulate_signals}"

    @property
    def subclass_id(self) -> int:
        return self.__subclass_id

    @subclass_id.setter
    def subclass_id(self, value: int) -> None:
        if value < -1 or not isinstance(value, int):
            raise ValueError("Wrong subclass id!")
        self.__subclass_id = value

    @property
    def density(self) -> float:
        return self.__density

    @property
    def points(self) -> float:
        return self.__total_points

    @property
    def accumulate_signals(self) -> int:
        return self.__acc_signals

    @property
    def features(self) -> np.array:
        return self.__weights

    def update_accumulate_signals(self, n: int=1) -> None:
        if n < 1 or not isinstance(n, int):
            raise ValueError("Wrong number to update accumulated signals!")
        self.__acc_signals += n

    def reset_points(self) -> None:
        self.__total_points = 0

    def update_points(self, points: float) -> None:
        if points < 0:
            raise ValueError("Wrong value of points!")
        self.__total_points += points
        
    def update_density(self, coeff: float=None) -> None:
        if coeff:
            self.__density = self.__total_points/coeff
        else:
            self.__density = self.__total_points/self.accumulate_signals

    def update_feature_vector(self, signal: np.array, coeff: float) -> None:
        self.__weights += coeff*(signal - self.__weights)


class EnhancedSelfOrganizingIncrementalNN:
    def __init__(self, c1: float=0.001, c2: float=1, learning_step: int=200,
                 max_age: int=50, forget: bool=False, rc: int=1,
                 mark_type: int=1, strong_condition=True, global_points=False,
                 adaptive_noise_removal=True, metrics=euclidean_distance,
                 learning_rate_winner=learning_rate_generator(),
                 learning_rate_winner_neighbor=learning_rate_generator(k=100),
                 logging_level: str="info", full_logging_info: bool=True):
        self.__C1 = c1
        self.__C2 = c2
        self.__lambd = learning_step
        self.__max_age = max_age
        self.__count = 0
        self.__metrics = metrics
        self.__forget = forget
        self.__id = 0
        self.__rc = rc
        self.__rate = learning_rate_winner
        self.__rate_neighbor = learning_rate_winner_neighbor
        self.__mark_type = mark_type
        self.__strong_condition = strong_condition
        self.__adaptive_noise = adaptive_noise_removal
        self.__global_points = global_points

        self._logger = enable_logging(f"{str(self.__class__)}.ESOINN",
                                      level=logging_level,
                                      full_info=full_logging_info)

        self.__nodes = {}  # key=id; value=ESOINNNode
        # @TODO: use {node_id: {neighbor_id: age}} instead of neighbors & edges
        self.__neighbors = {}  # key=id; value=set of neighbors' ids
        self.__edges = {}  # key=tuple(2), where t[0] < t[1]; value=age

    @property
    def metrics(self) -> callable:
        return self.__metrics

    def reset(self, c1: float=0.001, c2: float=1, learning_step: int=200,
              max_age: int=50, forget: bool=False, rc: int=1,
              mark_type: int=1, strong_condition=True, global_points=False,
              adaptive_noise_removal=True, metrics=euclidean_distance,
              learning_rate_winner=learning_rate_generator(),
              learning_rate_winner_neighbor=learning_rate_generator(k=100)
              ) -> None:
        self.__C1 = c1
        self.__C2 = c2
        self.__lambd = learning_step
        self.__max_age = max_age
        self.__metrics = metrics
        self.__forget = forget
        self.__rc = rc
        self.__rate = learning_rate_winner
        self.__rate_neighbor = learning_rate_winner_neighbor
        self.__mark_type = mark_type
        self.__strong_condition = strong_condition
        self.__adaptive_noise = adaptive_noise_removal
        self.__global_points = global_points
        self._reset()

    def _set_rc(self, rc: int) -> None:
        if rc > 1:
            self.__rc = rc

    def partial_fit(self, input_signal: np.array) -> None:
        self.__count += 1
        if self.__count < 3:
            self._create_node(input_signal)
            return

        winners_ids, distances = self._find_winners(input_signal)
        self._logger.debug(f">\twinners: {winners_ids} ({distances})")
        if distances[0] > self._calc_threshold(winners_ids[0]) \
                or distances[1] > self._calc_threshold(winners_ids[1]):
            self._logger.debug(">\tNEW NODE CREATION")
            self._create_node(input_signal)
            if not self.__strong_condition:
                return
        else:
            self._logger.debug(">\tINCLASS INSERTION")

            self._update_edges_age(winners_ids[0])
            self._build_connection(winners_ids)

            # do it one time only 'cause it takes long time
            winner_neighbors = self._find_neighbors(winners_ids[0],
                                                    depth=self.__rc)

            # original order was changed: density before update acc signal
            self.__nodes[winners_ids[0]].update_accumulate_signals()
            self._update_node_density(winners_ids[0], winner_neighbors)

            self._update_feature_vectors(node_id=winners_ids[0],
                                         input_signal=input_signal,
                                         neighbors=winner_neighbors)
            self._remove_old_edges()
        if self.__strong_condition:
            condition = not (self.__count - self.__id + 1) % self.__lambd
        else:
            condition = not self.__count % self.__lambd
        if condition:
            self._logger.debug(f">\tseparate classes on {self.__count}"
                               f" step with lambda={self.__lambd}")
            self._separate_subclasses()
            self._remove_noise()

    def fit(self, signals: list, get_state=False):
        self._logger.debug("Start training")
        start_t = time()
        self._reset()
        for idx, signal in enumerate(signals):
            self._logger.debug(f"$ proceed {signal}\t({idx})")
            self.partial_fit(signal)
        self.update()
        train_t = time() - start_t
        self._logger.info(f"Training complete by {train_t}")
        if get_state:
            return self.current_state(deep=True)

    # @TODO: remove inf coeff and separate variables for each winner, add Rc
    def _find_winners(self, input_signal: np.array) -> tuple:
        winner1 = float('inf')
        winner1_id = -1
        winner2 = float('inf')
        winner2_id = -1
        for node_id in self.__nodes:
            dist = self.__metrics(
                input_signal, self.__nodes[node_id].features
            )
            if dist <= winner1:
                winner1, winner2 = dist, winner1
                winner2_id = winner1_id
                winner1_id = node_id
            elif dist < winner2:
                winner2 = dist
                winner2_id = node_id
        return (winner1_id, winner2_id), (winner1, winner2)

    # @TODO: add filter for all neighbors found; use generator instead
    def _find_neighbors(self, node_id: int, depth=1) -> set:
        visited = {node_id}
        queue = set(self.__neighbors.get(node_id, set()))
        while depth and queue:
            depth -= 1
            for vertex in queue.copy():  # @CHECKME: do not use copy!
                visited.add(vertex)
                queue.remove(vertex)
                queue = queue.union([node for node in self.__neighbors[vertex]
                                     if node not in visited])
        return visited - {node_id}

    def _calc_threshold(self, node_id: int) -> float:
        neighbors = self.__neighbors.get(node_id, None)
        node_feature_vector = self.__nodes[node_id].features
        if neighbors:
            neighbors = self._find_neighbors(node_id, depth=self.__rc)
            threshold = np.max([self.__metrics(
                node_feature_vector, self.__nodes[neighbor_id].features)
                for neighbor_id in neighbors])
        else:
            threshold = np.min([self.__metrics(
                node_feature_vector, self.__nodes[node].features)
                for node in self.__nodes if node != node_id])
        self._logger.debug(f">\tthreshold for {node_id}: {threshold}")
        return threshold

    def _create_node(self, input_signal: np.array) -> None:
        self._logger.debug(f">\tnew node id: {self.__id}")
        self.__nodes[self.__id] = _ESOINNNode(input_signal)
        self.__id += 1  # to provide unique ids for each neuron

    def _remove_node(self, node_id: int) -> None:
        neighbors = self.__neighbors.get(node_id, set())
        for neighbor_id in neighbors.copy():
            self._remove_edges([node_id, neighbor_id])
        self._logger.debug(f">\tnode {node_id} removal ({neighbors})")
        del self.__nodes[node_id]

    def _update_edges_age(self, node_id: int, step=1) -> None:
        neighbors = self._find_neighbors(node_id, depth=self.__rc)
        for neighbor_id in neighbors:
            pair_id = min(node_id, neighbor_id), max(node_id, neighbor_id)
            if self.__edges.get(pair_id, None):
                self.__edges.get[pair_id] += step

    # algorithm 3.2
    def _build_connection(self, nodes_ids: "list of 2 ids") -> None:
        winners_classes = [self.__nodes[nodes_ids[0]].subclass_id,
                           self.__nodes[nodes_ids[1]].subclass_id]
        if winners_classes[0] == -1 \
                or winners_classes[1] == -1 \
                or winners_classes[0] == winners_classes[1] \
                or self._merge_subclass_condition(nodes_ids):
            self._logger.debug(f">\tCREATE CONNECTION {nodes_ids}")
            self._combine_subclasses(nodes_ids)
            self._create_edges(nodes_ids)
        else:
            self._logger.debug(f">\tREMOVE CONNECTION {nodes_ids}")
            self._divide_subclasses(nodes_ids)

    # @TODO: do it with indexes
    def _create_edges(self, nodes_ids: "list of 2 ids") -> None:
        for node_id in nodes_ids:
            if node_id not in self.__neighbors:
                self.__neighbors[node_id] = set()
            for insert_id in nodes_ids:
                if insert_id != node_id:
                    self.__neighbors[node_id] |= {insert_id}
                    nodes_pair = (min(node_id, insert_id),
                                  max(node_id, insert_id))
                    self._logger.debug(f">\tcreate edge {nodes_pair}")
                    self.__edges[nodes_pair] = 0

    # @TODO: do it with indexes
    def _remove_edges(self, nodes_ids: "list of 2 ids") -> None:
        for node_id in nodes_ids:
            for remove_id in nodes_ids:
                if remove_id != node_id \
                        and remove_id in self.__neighbors.get(node_id, {}):
                    self.__neighbors[node_id] -= {remove_id}
                    nodes_pair = (min(node_id, remove_id),
                                  max(node_id, remove_id))
                    self._logger.debug(f">\tremove edge {nodes_pair}")
                    if nodes_pair in self.__edges:
                        del self.__edges[nodes_pair]
            if node_id in self.__neighbors and not self.__neighbors[node_id]:
                del self.__neighbors[node_id]

    # @TODO: try to reset points for each learning period
    def _update_node_points(self, node_id: int, neighbors: set) -> None:
        node_feature_vector = self.__nodes[node_id].features
        points_update = 1
        if self.__global_points:
            mean_dist = 1/len(self.__nodes)*np.sum([
                self.__metrics(node_feature_vector, self.__nodes[vertex].features)
                for vertex in self.__nodes
            ])
            points_update = 1/(1 + mean_dist)**2
        elif neighbors:
            mean_neighbors_dist = 1/len(neighbors)*np.sum([
                self.__metrics(
                    node_feature_vector,
                    self.__nodes[neighbor_id].features
                ) for neighbor_id in neighbors
            ])
            points_update = 1/(1 + mean_neighbors_dist)**2
        self._logger.debug(f">\tadd {points_update} to points of {node_id}")
        self.__nodes[node_id].update_points(points_update)

    def _update_node_density(self, node_id: int, neighbors: set) -> None:
        self._update_node_points(node_id, neighbors)
        if self.__forget:
            self.__nodes[node_id].update_density(self.__count)
        else:
            self.__nodes[node_id].update_density()

    def _update_feature_vectors(self, node_id: int, input_signal: np.array,
                                neighbors: set) -> None:
        acc_signal = self.__nodes[node_id].accumulate_signals
        self.__nodes[node_id].update_feature_vector(
            signal=input_signal,
            coeff=self.__rate(acc_signal)
        )
        neighbors_rate = self.__rate_neighbor(acc_signal)
        for neighbor_id in neighbors:
            self.__nodes[neighbor_id].update_feature_vector(
                signal=input_signal,
                coeff=neighbors_rate
            )

    def _remove_old_edges(self) -> None:
        for edge in self.__edges.copy():
            if self.__edges[edge] > self.__max_age:
                self._logger.debug(f">\tremove edge {edge} "
                                   f"(>{self.__max_age})")
                self._remove_edges(edge)

    # @CHECKME: wrong: going throw all nodes in class
    # @TODO: update tests
    def _calc_mean_density_in_subclass(self, node_id: int) -> np.ndarray:
        if self.__nodes[node_id].subclass_id == -1:
            return np.ndarray([0], dtype=float)
        neighbors = set()
        node_class = self.__nodes[node_id].subclass_id
        visited = set()
        queue = [node_id]
        while queue:
            vertex = queue.pop(0)
            vertex_subclass = self.__nodes[vertex].subclass_id
            if vertex_subclass == node_class:
                neighbors.add(vertex)
            if vertex not in visited:
                visited.add(vertex)
                queue.extend([
                    node for node in self.__neighbors.get(vertex, set())
                    if node not in visited])

        return 1/len(neighbors)*np.sum([
            self.__nodes[node_id].density for node_id in neighbors
        ])
    
    def _calc_alpha(self, node_id: int, apex_density: float) -> float:
        mean_density = self._calc_mean_density_in_subclass(node_id)
        if 2*mean_density >= apex_density:
            return 0
        elif 3*mean_density >= apex_density:
            return 0.5
        return 1

    # @FIXME: class apex can be removed (WHY?)
    def _merge_subclass_condition(self, nodes_ids: "list of 2 ids") -> bool:
        min_density = min(self.__nodes[nodes_ids[0]].density,
                          self.__nodes[nodes_ids[1]].density)
        apex_a = self.__nodes.get(self.__nodes[nodes_ids[0]].subclass_id, ())
        if not apex_a:
            apex_a = self.__nodes[self._find_class_apex(nodes_ids[0])[0]]
        apex_b = self.__nodes.get(self.__nodes[nodes_ids[1]].subclass_id, ())
        if not apex_b:
            apex_b = self.__nodes[self._find_class_apex(nodes_ids[1])[0]]
        alphas = (self._calc_alpha(nodes_ids[0], apex_a.density),
                  self._calc_alpha(nodes_ids[1], apex_b.density))
        return min_density > alphas[0]*apex_a.density \
            and min_density > alphas[1]*apex_b.density

    def _change_class_id(self, node_id: int, class_id: int) -> None:
        visited = set()
        queue = [node_id]
        while queue:
            vertex = queue.pop(0)
            self.__nodes[vertex].subclass_id = class_id
            if vertex not in visited:
                visited.add(vertex)
                queue.extend([
                    node for node in self.__neighbors.get(vertex, set())
                    if node not in visited])

    def _combine_subclasses(self, nodes_ids: "list of 2 ids") -> None:
        nodes = [self.__nodes[nodes_ids[0]], self.__nodes[nodes_ids[1]]]
        subclass_ids = [nodes[0].subclass_id, nodes[1].subclass_id]
        if subclass_ids[0] == -1 and subclass_ids[1] == -1:
            for node_id in nodes_ids:
                # @TODO: use id of node with max density
                self._change_class_id(node_id, nodes_ids[0])
        elif subclass_ids[0] != subclass_ids[1]:
            if subclass_ids[0] == -1:
                self._change_class_id(nodes_ids[0], subclass_ids[1])
            else:
                self._change_class_id(nodes_ids[1], subclass_ids[0])

    # @TODO: tests
    def _divide_subclasses(self, nodes_ids: "list of 2 ids") -> None:
        self._remove_edges(nodes_ids)
        apexes = (self._find_class_apex(nodes_ids[0])[0],
                  self._find_class_apex(nodes_ids[1])[0])
        if apexes[0] == apexes[1]:
            return
        for idx in range(len(nodes_ids)):
            self._change_class_id(nodes_ids[idx], apexes[idx])

    # @TODO: tests
    def _find_local_maxes(self) -> set:
        apexes = set()
        for node_id in self.__nodes.keys():
            neighbors = self.__neighbors.get(node_id, {})
            if not neighbors:
                apexes.add(node_id)
                continue
            is_apex = True
            current_density = self.__nodes[node_id].density
            for neighbor_id in neighbors:
                is_apex &= self.__nodes[neighbor_id].density < current_density
                if not is_apex:
                    break
            if is_apex:
                apexes.add(node_id)
        return apexes

    # @TODO: tests
    def _continue_mark(self, node_ids: set, class_id: int, visited: set):
        predicted_overlap_ids = set()
        for node_id in node_ids:
            queue = {node_id}
            while queue:
                for vertex in queue.copy():
                    for_extend = set()
                    vertex_density = self.__nodes[vertex].density
                    queue.remove(vertex)
                    if vertex not in predicted_overlap_ids:
                        is_overlap = False
                        for neighbor_id in self.__neighbors.get(vertex, {}):
                            neighbor_density = self.__nodes[neighbor_id].density
                            if neighbor_density > vertex_density:
                                if neighbor_id not in visited and \
                                neighbor_id not in predicted_overlap_ids:
                                    is_overlap = True
                            else:
                                for_extend.add(neighbor_id)
                        if is_overlap:
                            predicted_overlap_ids.add(vertex)
                        else:
                            visited.add(vertex)
                            self.__nodes[vertex].subclass_id = class_id
                            queue = queue.union(for_extend)
        return predicted_overlap_ids, visited

    # @TODO: tests
    def _check_overlap(self, overlap_ids: set, visited: set) -> set:
        continue_ids = set()  # continue mark this nodes
        for node_id in overlap_ids:
            near_id = self._get_nearest_neighbor(node_id, overlap_ids)
            if near_id in visited:
                continue_ids.add(node_id)
                node_density = self.__nodes[node_id].density
                for neighbor_id in self.__neighbors.get(node_id, {}).copy():
                    if self.__nodes[neighbor_id].density > node_density \
                            and neighbor_id not in visited \
                            and neighbor_id not in overlap_ids:
                        self._remove_edges([neighbor_id, node_id])
            else:
                for neighbor_id in self.__neighbors.get(node_id, {}).copy():
                    if neighbor_id in visited:
                        self._remove_edges([node_id, neighbor_id])
        return continue_ids

    # @TODO: tests
    def _get_nearest_neighbor(self, node_id: int, overlap_ids=set()) -> int:
        min_dist = float('inf')
        near_id = -1
        node_density = self.__nodes[node_id].density
        for neighbor_id in self.__neighbors.get(node_id, set()):
            if self.__nodes[neighbor_id].density > node_density \
                    and neighbor_id not in overlap_ids:
                dist = self.__metrics(self.__nodes[neighbor_id].features,
                                      self.__nodes[node_id].features)
                if min_dist > dist:
                    min_dist = dist
                    near_id = neighbor_id
        return near_id

    # @TODO: tests
    def _separate_subclasses(self) -> None:
        apexes = self._find_local_maxes()
        if not self.__mark_type:  # @CHECKME: depreciated
            for apex in apexes:
                queue = {apex}
                visited = set()
                while queue:
                    predicted_overlap_ids, visit = \
                        self._continue_mark(queue, apex, visited)
                    visited = visited.union(visit)
                    queue = self._check_overlap(predicted_overlap_ids, visited)
        elif self.__mark_type == 1:
            self._stupid_mark(apexes)
            self._solve_overlap_conflicts()
            _ = self.update(remove_noise=False)
        else:  # @CHECKME: depreciated
            self._stupid_mark(apexes)
            overlap = self._find_overlap()
            self._remove_overlap(overlap)
            _ = self.update(remove_noise=False)

    # @TODO: tests
    def _solve_overlap_conflicts(self) -> None:
        for edge in self.__edges.copy():
            if self.__nodes[edge[0]].subclass_id == \
                    self.__nodes[edge[1]].subclass_id:
                continue
            if not self._merge_subclass_condition(edge):
                self._remove_edges(edge)

    # @TODO: tests
    def _find_neighbors_local_maxes(self, node_id):
        apexes = set()
        visited = set()
        queue = [node_id]
        for vertex in queue:
            is_local_max = True
            vertex_density = self.__nodes[vertex].density
            for neighbor_id in self.__neighbors.get(vertex, set()):
                if self.__nodes[neighbor_id].density > vertex_density:
                    if neighbor_id not in visited:
                        queue.append(neighbor_id)
                    is_local_max = False
                visited.add(neighbor_id)
            if is_local_max:
                apexes.add(vertex)
            visited.add(vertex)
        if not apexes:
            return {node_id}
        return apexes

    # @TODO: tests
    def _find_overlap(self, verify=True) -> set():
        overlap = set()
        for node_id in self.__nodes:
            neighbors = self.__neighbors.get(node_id, set())
            node_density = self.__nodes[node_id].density
            count_greater = 0
            greater_apexes = []
            for neighbor_id in neighbors:
                if node_density < self.__nodes[neighbor_id].density:
                    count_greater += 1
                    greater_apexes.append(
                        self._find_neighbors_local_maxes(neighbor_id))
            same_apexes = True
            for i in greater_apexes:
                for j in greater_apexes:
                    same_apexes &= i == j
            if count_greater > 1 and (not same_apexes or not verify):
                overlap.add(node_id)
        return overlap

    # @TODO: tests
    def _stupid_mark(self, apexes: set) -> None:
        for apex_id in apexes:
            visited = set()
            queue = [apex_id]
            while queue:
                vertex = queue.pop(0)
                self.__nodes[vertex].subclass_id = apex_id
                visited.add(vertex)
                vertex_density = self.__nodes[vertex].density
                for neighbor_id in self.__neighbors.get(vertex, set()):
                    if vertex_density > self.__nodes[neighbor_id].density:
                        if neighbor_id not in visited:
                            nearest_id = self._get_nearest_neighbor(neighbor_id)
                            if nearest_id == vertex:
                                queue.append(neighbor_id)

    # @TODO: tests
    def _remove_overlap(self, nodes_ids: set) -> None:
        for overlap_id in nodes_ids:
            neighbors = self.__neighbors.get(overlap_id, set())
            node_density = self.__nodes[overlap_id].density
            greater_nodes = []
            for neighbor_id in neighbors:
                if node_density < self.__nodes[neighbor_id].density:
                    greater_nodes.append(neighbor_id)
            for node_id in greater_nodes:
                edge = [overlap_id, node_id]
                if not self._merge_subclass_condition(edge):
                    self._remove_edges(edge)
            if not self.__neighbors.get(overlap_id, set()):
                self._remove_node(overlap_id)

    # @TODO: update tests
    def _remove_noise(self) -> None:
        if not self.__adaptive_noise:
            mean_density = np.sum([
                    node.density for node in self.__nodes.values()
                ])/len(self.__nodes)
        # copy of neighbors is used to solve problem with order sensitivity
        neighbors_copy = deepcopy(self.__neighbors)
        for node_id in self.__nodes.copy():
            if self.__adaptive_noise:
                mean_density = self._calc_mean_density_in_subclass(node_id)
            neighbors_count = len(neighbors_copy.get(node_id, set()))
            node_density = self.__nodes[node_id].density
            if not neighbors_count or neighbors_count == 1 \
                    and node_density < self.__C2*mean_density \
                    or neighbors_count == 2 \
                    and node_density < self.__C1*mean_density:
                self._logger.debug(f">\tremove {node_id} as NOISE "
                                   f"({node_density} : {mean_density})")
                self._remove_node(node_id)

    # @TODO: add Rc
    def predict(self, input_signal) -> tuple:
        winners, distances = self._find_winners(input_signal)
        chance1 = distances[1]/(distances[0] + distances[1])
        chance2 = distances[0]/(distances[0] + distances[1])
        win1class = self.__nodes[winners[0]].subclass_id
        win2class = self.__nodes[winners[1]].subclass_id
        return win1class, chance1 + chance2*(win1class == win2class)

    def _get_classes(self) -> dict:
        queue = set(self.__nodes.keys())
        apexes_ids = {}
        while queue:
            node_id = queue.pop()
            apex_id, visited = self._find_class_apex(node_id)
            queue -= visited
            apexes_ids[apex_id] = visited
        return apexes_ids

    # algorithm 3.3
    def update(self, remove_noise=True) -> dict:
        if remove_noise:
            self._remove_noise()
        queue = set(self.__nodes.keys())
        apexes_ids = {}
        while queue:
            node_id = queue.pop()
            apex_id, visited = self._find_class_apex(node_id)
            queue -= visited
            apexes_ids[apex_id] = visited
            for vertex in visited:
                self.__nodes[vertex].subclass_id = apex_id
        return apexes_ids

    def _find_class_apex(self, node_id: int) -> tuple:
        apex_id = node_id
        visited = {node_id}
        queue = list(self.__neighbors.get(node_id, set()))
        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                if self.__nodes[apex_id].density < \
                        self.__nodes[vertex].density:
                    apex_id = vertex
                visited.add(vertex)
                queue.extend([node for node in self.__neighbors[vertex]
                              if node not in visited])
        return apex_id, visited

    def current_state(self, deep=True) -> dict:
        nodes = self.__nodes
        neighbors = self.__neighbors
        edges = self.__edges
        if deep:
            nodes = deepcopy(self.__nodes)
            neighbors = deepcopy(self.__neighbors)
            edges = deepcopy(self.__edges)
        classes = self._get_classes()
        return {
            'count_signals': self.__count,
            'count_neurons': len(self.__nodes),
            'last_node_id': self.__id - 1,
            'nodes': nodes,
            'neighbors': neighbors,
            'edges': edges,
            'classes': classes,
            'configuration': {
                'C1': self.__C1,
                'C2': self.__C2,
                'Rc': self.__rc,
                'lambda': self.__lambd,
                'forget': self.__forget,
                'max_age': self.__max_age,
                'metrics': self.__metrics,
                'learning_rate_winner': self.__rate,
                'learning_rate_winner_neighbor': self.__rate_neighbor,
            }
        }

    def _reset(self) -> None:
        self._logger.debug("$ clearing")
        self.__nodes.clear()
        self.__edges.clear()
        self.__neighbors.clear()
        self.__id = 0
        self.__count = 0

    def _configure(self, nodes, neighbors: dict={}, edges: dict={},
                   non_classified=-1) -> None:
        if isinstance(nodes, dict):
            self.__nodes = deepcopy(nodes)
        elif isinstance(nodes, list):
            for node in nodes:
                node_id = node['id']
                self.__nodes[node_id] = _ESOINNNode(node['features'])
                self.__nodes[node_id].update_points(node['density'])
                self.__nodes[node_id].update_density(1)
                self.__nodes[node_id].update_accumulate_signals()
                if node['subclass_id'] != -1:
                    self.__nodes[node_id].subclass_id = node['subclass_id']
                else:
                    self.__nodes[node_id].subclass_id = non_classified
        else:
            raise ValueError(f"Wrong type of 'nodes': {type(nodes)}")
        self.__id = len(nodes)  # set correct unique id
        self.__neighbors = deepcopy(neighbors)
        self.__edges = deepcopy(edges)
