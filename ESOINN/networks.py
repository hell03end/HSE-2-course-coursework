from copy import deepcopy
import numpy as np
from time import time
try:
    from dev.commons import enable_logging
except ImportError as error:
    print(error.args)
    from .commons import enable_logging


def learning_rate_generator(k=1, n=1) -> callable:
    return lambda t: n / (k * t)


class ESOINN:
    def __init__(self, c1: float=0.001, c2: float=1, learning_step: int=200,
                 max_age: int=50, forget: bool=False,
                 strong_period_condition=True, strong_merge_condition=True,
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
        self.__rate = learning_rate_winner
        self.__rate_neighbor = learning_rate_winner_neighbor
        self.__strong_period_condition = strong_period_condition
        self.__adaptive_noise = adaptive_noise_removal
        self.__learning_period = 1
        self.__merge_AND = strong_merge_condition

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
            if not self.__strong_period_condition:
                return
        else:
            self._logger.debug(">\tINCLASS INSERTION")

            self._update_edges_age(winners_ids[0])
            self._build_connection(winners_ids)

            self._update_node_density(winners_ids[0])
            self.__nodes[winners_ids[0]].update_accumulate_signals()
            self._update_feature_vectors(node_id=winners_ids[0],
                                         input_signal=input_signal)
            self._remove_old_edges()
        if not self.__count % self.__lambd:
            self._logger.debug(f">\tseparate classes on {self.__count}"
                               f" step with lambda={self.__lambd}")
            self._separate_subclasses()
            self._remove_noise()
            self.__learning_period += 1

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
            threshold = np.max([
                self.__metrics(node_feature_vector,
                               self.__nodes[neighbor_id].features)
                for neighbor_id in neighbors
            ])
        else:
            threshold = np.min([
                self.__metrics(node_feature_vector,
                               self.__nodes[node].features)
                for node in self.__nodes if node != node_id
            ])
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
        neighbors = self.__neighbors.get(node_id, set())
        for neighbor_id in neighbors:
            edge = min(node_id, neighbor_id), max(node_id, neighbor_id)
            if not self.__edges.get(edge, None) is None:
                self.__edges[edge] += step

    # algorithm 3.2
    def _build_connection(self, nodes_ids: "list of 2 ids") -> None:
        classes = [self.__nodes[nodes_ids[0]].subclass_id,
                   self.__nodes[nodes_ids[1]].subclass_id]
        if classes[0] == -1 or classes[1] == -1 or classes[0] == classes[1] \
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
                    edge = (min(node_id, insert_id), max(node_id, insert_id))
                    self._logger.debug(f">\tcreate edge {edge}")
                    self.__edges[edge] = 0

    # @TODO: do it with indexes
    def _remove_edges(self, nodes_ids: "list of 2 ids") -> None:
        for node_id in nodes_ids:
            for remove_id in nodes_ids:
                if remove_id != node_id \
                        and remove_id in self.__neighbors.get(node_id, {}):
                    self.__neighbors[node_id] -= {remove_id}
                    edge = (min(node_id, remove_id), max(node_id, remove_id))
                    self._logger.debug(f">\tremove edge {edge}")
                    if edge in self.__edges:
                        del self.__edges[edge]
            if node_id in self.__neighbors and not self.__neighbors[node_id]:
                del self.__neighbors[node_id]

    def _update_node_points(self, node_id: int) -> None:
        neighbors = self.__neighbors.get(node_id, set())
        points_update = 1
        if neighbors:
            node_feature_vector = self.__nodes[node_id].features
            mean_neighbors_dist = 1 / len(neighbors) * np.sum([
                self.__metrics(
                    node_feature_vector,
                    self.__nodes[neighbor_id].features
                ) for neighbor_id in neighbors
            ])
            points_update = 1 / (1 + mean_neighbors_dist)**2
        self._logger.debug(f">\tadd {points_update} to points of {node_id}")
        self.__nodes[node_id].update_points(points_update)

    def _update_node_density(self, node_id: int) -> None:
        self._update_node_points(node_id)
        self.__nodes[node_id].update_win_period(self.__learning_period)
        if self.__forget:
            self.__nodes[node_id].update_density(self.__learning_period)
        else:
            self.__nodes[node_id].update_density()

    def _update_feature_vectors(self, node_id: int,
                                input_signal: np.array) -> None:
        neighbors = self.__neighbors.get(node_id, set())
        acc_signal = self.__nodes[node_id].accumulate_signals
        self.__nodes[node_id].update_feature_vector(input_signal,
                                                    self.__rate(acc_signal))
        neighbors_rate = self.__rate_neighbor(acc_signal)
        for neighbor_id in neighbors:
            self.__nodes[neighbor_id].update_feature_vector(input_signal,
                                                            neighbors_rate)

    def _remove_old_edges(self) -> None:
        for edge in self.__edges.copy():
            if self.__edges[edge] > self.__max_age:
                self._logger.debug(f">\tremove edge {edge} "
                                   f"(>{self.__max_age})")
                self._remove_edges(edge)

    # @CHECKME: wrong: going throw all nodes in class
    def _calc_mean_density_in_subclass(self, node_id: int) -> np.ndarray:
        if self.__nodes[node_id].subclass_id == -1:
            return float(self.__nodes[node_id].density)
        node_class = self.__nodes[node_id].subclass_id
        neighbors = set()
        visited = set()
        queue = [node_id]
        while queue:
            vertex = queue.pop(0)
            if self.__nodes[vertex].subclass_id == node_class:
                neighbors.add(vertex)
            if vertex not in visited:
                visited.add(vertex)
                queue.extend([
                    node for node in self.__neighbors.get(vertex, set())
                    if node not in visited])

        return 1 / len(neighbors) * np.sum([
            self.__nodes[node_id].density for node_id in neighbors
        ])

    def _calc_alpha(self, node_id: int, apex_density: float) -> float:
        mean_density = self._calc_mean_density_in_subclass(node_id)
        if 2 * mean_density >= apex_density:
            return 0
        elif 3 * mean_density >= apex_density:
            return 0.5
        return 1

    # @CHECKME: can be wrong
    def _merge_subclass_condition(self, nodes_ids: "list of 2 ids") -> bool:
        min_density = min(self.__nodes[nodes_ids[0]].density,
                          self.__nodes[nodes_ids[1]].density)
        apex_a = self.__nodes[self._find_subclass_apex(nodes_ids[0])]
        apex_b = self.__nodes[self._find_subclass_apex(nodes_ids[1])]
        alphas = (self._calc_alpha(nodes_ids[0], apex_a.density),
                  self._calc_alpha(nodes_ids[1], apex_b.density))
        if self.__merge_AND:
            condition = min_density > alphas[0] * apex_a.density \
                and min_density > alphas[1] * apex_b.density
        else:
            condition = min_density > alphas[0] * apex_a.density \
                or min_density > alphas[1] * apex_b.density
        return condition

    def _change_class_id(self, node_id: int, class_id: int) -> None:
        visited = set()
        node_class_id = self.__nodes[node_id].subclass_id
        queue = [node_id]
        while queue:
            vertex = queue.pop(0)
            if node_class_id == self.__nodes[vertex].subclass_id:
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

    def _divide_subclasses(self, nodes_ids: "list of 2 ids") -> None:
        self._remove_edges(nodes_ids)
        apexes = (self._find_class_apex(nodes_ids[0])[0],
                  self._find_class_apex(nodes_ids[1])[0])
        if apexes[0] == apexes[1]:
            return
        for idx in range(len(nodes_ids)):
            self._change_class_id(nodes_ids[idx], apexes[idx])

    def _find_local_maxes(self) -> set:
        apexes = set()
        for node_id in self.__nodes:
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

    def _separate_subclasses(self) -> None:
        apexes = self._find_local_maxes()
        self._mark_subclasses(apexes)
        self._remove_overlap()
        _ = self.update(remove_noise=False)

    def _remove_overlap(self) -> None:
        for edge in self.__edges.copy():
            if self.__nodes[edge[0]].subclass_id == \
                    self.__nodes[edge[1]].subclass_id:
                continue
            if not self._merge_subclass_condition(edge):
                self._remove_edges(edge)
            else:
                self._combine_subclasses(edge)

    def _mark_subclasses(self, apexes: set) -> None:
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
                            nearest_id = self._get_nearest_neighbor(
                                neighbor_id)
                            if nearest_id == vertex:
                                queue.append(neighbor_id)

    def _remove_noise(self) -> None:
        if not self.__adaptive_noise:
            mean_density = np.sum([
                node.density for node in self.__nodes.values()
            ]) / len(self.__nodes)
        # copy of neighbors is used to solve problem with order sensitivity
        neighbors_copy = deepcopy(self.__neighbors)
        for node_id in self.__nodes.copy():
            if self.__adaptive_noise:
                mean_density = self._calc_mean_density_in_subclass(node_id)
            neighbors_count = len(neighbors_copy.get(node_id, set()))
            node_density = self.__nodes[node_id].density
            if not neighbors_count or neighbors_count == 1 \
                    and node_density < self.__C2 * mean_density \
                    or neighbors_count == 2 \
                    and node_density < self.__C1 * mean_density:
                self._logger.debug(f">\tremove {node_id} as NOISE "
                                   f"({node_density} : {mean_density})")
                self._remove_node(node_id)

    # @TODO: add Rc
    def predict(self, input_signal) -> tuple:
        winners, distances = self._find_winners(input_signal)
        chance1 = distances[1] / (distances[0] + distances[1])
        chance2 = distances[0] / (distances[0] + distances[1])
        win1class = self.__nodes[winners[0]].subclass_id
        win2class = self.__nodes[winners[1]].subclass_id
        return win1class, chance1 + chance2 * (win1class == win2class)

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

    def _find_subclass_apex(self, node_id: int) -> int:
        subclass = self.__nodes[node_id].subclass_id
        neighbors = self._find_neighbors(node_id, depth=-1)
        max_id = node_id
        for neighbor_id in neighbors:
            if self.__nodes[neighbor_id].subclass_id == subclass:
                neighbor = self.__nodes[neighbor_id]
                if neighbor.density > self.__nodes[max_id].density:
                    max_id = neighbor_id
        return max_id

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
            'periods': self.__learning_period,
            'configuration': {
                'C1': self.__C1,
                'C2': self.__C2,
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
