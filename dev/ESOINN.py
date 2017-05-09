from copy import deepcopy
import numpy as np
import re
try:
    from dev.commons import enable_logging
except ImportError as error:
    print(error.args)
    from .commons import enable_logging


def euclidean_distance(x, y) -> float:
    return np.sqrt(np.sum(np.square(np.array(x) - np.array(y))))


def learning_rate_generator(k=1, n=1):
    return lambda t: n/(k*t)


class ESOINNNode:
    def __init__(self, feature_vector):
        self.__weights = np.array(feature_vector, dtype=float)
        self.__acc_signals = 0
        self.__total_points = float(0)
        self.__density = float(0)
        self.__subclass_id = -1

    def __repr__(self) -> str:
        return f"{str(self.feature_vector):^30} | {self.subclass_id:10} | " \
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
    def feature_vector(self) -> np.array:
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
    def __init__(self, init_nodes: list, c1=0.001, c2=1,
                 learning_step=200, max_age=50, forget=False, radius_cut_off=1,
                 metrics=euclidean_distance,
                 learning_rate_winner=learning_rate_generator(),
                 learning_rate_winner_neighbor=learning_rate_generator(k=100),
                 logging_level="info", full_logging_info=False,
                 strong_condition=True):
        self.__C1 = c1
        self.__C2 = c2
        self.__lambd = learning_step
        self.__max_age = max_age
        self.__count = 0
        self.__metrics = metrics
        self.__forget = forget
        self.__id = 0
        self.__rc = radius_cut_off
        self.__rate = learning_rate_winner
        self.__rate_neighbor = learning_rate_winner_neighbor
        self.__AND = strong_condition

        self._logger = enable_logging(f"{str(self.__class__)}.ESOINN",
                                      level=logging_level,
                                      full_info=full_logging_info)

        self.__nodes = {}  # key=id; value=ESOINNNode
        for signal in init_nodes:
            self.create_node(signal)
        # @TODO: use {node_id: {neighbor_id: age}} instead of neighbors & edges
        self.__neighbors = {}  # key=id; value=set of neighbors' ids
        self.__edges = {}  # key=tuple(2), where t[0] < t[1]; value=age

    @property
    def metrics(self):
        return self.__metrics

    @property
    def rc(self):
        return self.__rc

    def _set_rc(self, rc):
        if rc > 1:
            self.__rc = rc

    @property
    def unique_id(self):
        return self.__id

    # @FIXME: check condition of adding new node (thresholds usage)
    def partial_fit(self, input_signal: np.array) -> None:
        self.__count += 1
        
        winners_ids, distances = self.find_winners(input_signal)
        self._logger.debug(f">\twinners: {winners_ids} ({distances})")
        if self.__AND:
            condition = distances[0] > self.calc_threshold(winners_ids[0]) \
                        and distances[1] > self.calc_threshold(winners_ids[1])
        else:
            condition = distances[0] > self.calc_threshold(winners_ids[0]) \
                        or distances[1] > self.calc_threshold(winners_ids[1])
        if condition:
            self._logger.debug(">\tNEW NODE CREATION")
            self.create_node(input_signal)
            return
        self._logger.debug(">\tINCLASS INSERTION")
        
        self.update_edges_age(winners_ids[0])
        self.build_connection(winners_ids)
        
        # do it one time only 'cause it takes long time
        winner_neighbors = self.find_neighbors(winners_ids[0], depth=self.__rc)

        # original order was changed: density before update acc signal
        self.__nodes[winners_ids[0]].update_accumulate_signals()
        self.update_node_density(winners_ids[0], winner_neighbors)

        self.update_feature_vectors(node_id=winners_ids[0],
                                    input_signal=input_signal,
                                    neighbors=winner_neighbors)
        self.remove_old_edges()
        
        if not self.__count % self.__lambd:
            self._logger.debug(f">\tseparate classes on {self.__count_signals}"
                               f" step with lambda={self.__learning_step}")
            self.separate_subclasses()
            self.remove_noise()

    def fit(self, signals: list, get_state=False):
        self._logger.debug("Start training")
        for signal in signals:
            self._logger.debug(f"$ proceed {signal}")
            self.partial_fit(signal)
        self._logger.debug("Training complete")
        self._logger.debug("Updating...")
        self.update()
        self._logger.debug("Done")
        if get_state:
            return self.current_state(deep=True)

    # @TODO: remove inf coeff and separate variables for each winner, add Rc
    def find_winners(self, input_signal: np.array) -> tuple:
        winner1 = float('inf')
        winner1_id = -1
        winner2 = float('inf')
        winner2_id = -1
        for node_id in self.__nodes:
            dist = self.__metrics(
                input_signal, self.__nodes[node_id].feature_vector
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
    def find_neighbors(self, start_node_id: int, depth=1) -> set:
        visited = {start_node_id}
        queue = self.__neighbors.get(start_node_id, set()) - visited
        while depth and queue:
            depth -= 1
            for vertex in queue.copy():  # @CHECKME: do not use copy!
                visited.add(vertex)
                queue.remove(vertex)
                for node in self.__neighbors[vertex]:
                    if node not in visited:
                        queue.add(node)
        return visited - {start_node_id}

    def calc_threshold(self, node_id: int) -> float:
        neighbors = self.__neighbors.get(node_id, None)
        node_feature_vector = self.__nodes[node_id].feature_vector
        if neighbors:
            neighbors = self.find_neighbors(node_id, depth=self.__rc)
            threshold = np.max([
                self.__metrics(
                    node_feature_vector,
                    self.__nodes[neighbor_id].feature_vector
                ) for neighbor_id in neighbors
            ])
        else:
            threshold = np.min([
                self.__metrics(
                    node_feature_vector,
                    self.__nodes[node].feature_vector
                ) for node in self.__nodes if node != node_id
            ])
        self._logger.debug(f">\tthreshold for {node_id}: {threshold}")
        return threshold

    def create_node(self, input_signal: np.array) -> None:
        self._logger.debug(f">\tnew node id: {self.__unique_id}")
        self.__nodes[self.__id] = ESOINNNode(input_signal)
        self.__id += 1  # to provide unique ids for each neuron

    def remove_node(self, node_id: int) -> None:
        neighbors = self.find_neighbors(node_id)
        for neighbor_id in neighbors:
            self.remove_edges((node_id, neighbor_id))
        self._logger.debug(f">\tnode {node_id} removal ({neighbors})")
        del self.__nodes[node_id]

    def update_edges_age(self, node_id: int, step=1) -> None:
        neighbors = self.find_neighbors(node_id, depth=self.__rc)
        for neighbor_id in neighbors:
            pair_id = min(node_id, neighbor_id), max(node_id, neighbor_id)
            self.__edges[pair_id] += step

    # algorithm 3.2
    def build_connection(self, nodes_ids: "list of 2 ids") -> None:
        winners_classes = [self.__nodes[nodes_ids[0]].subclass_id,
                           self.__nodes[nodes_ids[1]].subclass_id]
        if winners_classes[0] == -1 \
                or winners_classes[1] == -1 \
                or winners_classes[0] == winners_classes[1] \
                or self.merge_subclass_condition(nodes_ids):
            self._logger.debug(f">\tCREATE CONNECTION {nodes_ids}")
            self.combine_subclasses(nodes_ids)
            self.create_edges(nodes_ids)
        else:
            self._logger.debug(f">\tREMOVE CONNECTION {nodes_ids}")
            self.remove_edges(nodes_ids)

    # @FIXME: do it with indexes
    def create_edges(self, nodes_ids: "list of 2 ids") -> None:
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

    # @FIXME: do it with indexes
    def remove_edges(self, nodes_ids: "list of 2 ids") -> None:
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
            if not self.__neighbors[node_id]:
                del self.__neighbors[node_id]

    # @CHECKME: what if node has no neighbors?
    # @TODO: try to reset points for each learning period
    def update_node_points(self, node_id: int, neighbors: set) -> None:
        points_update = 1
        if neighbors:
            node_feature_vector = self.__nodes[node_id].feature_vector
            mean_dist2neighbors = 1/len(neighbors)*np.sum([
                self.__metrics(
                    node_feature_vector,
                    self.__nodes[neighbor_id].feature_vector
                ) for neighbor_id in neighbors
            ])
            points_update = 1/(1 + mean_dist2neighbors)**2
        self._logger.debug(f">\tadd {points_update} to points of {node_id}")
        self.__nodes[node_id].update_points(points_update)

    def update_node_density(self, node_id: int, neighbors: set) -> None:
        self.update_node_points(node_id, neighbors)
        if self.__forget:
            self.__nodes[node_id].update_density(self.__count)
        else:
            self.__nodes[node_id].update_density()

    def update_feature_vectors(self, node_id: int, input_signal: np.array,
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

    def remove_old_edges(self) -> None:
        for edge in self.__edges.copy():
            if self.__edges[edge] > self.__max_age:
                self._logger.debug(
                    f">\tremove edge {edge} (>{self.__max_age})")
                self.remove_edges(edge)

    # @FIXME: use only nodes from one subclass        <--- ERROR HERE!!!
    def calc_mean_density_in_subclass(self, node_id: int) -> np.ndarray:
        neighbors = self.find_neighbors(node_id, depth=-1)  # WRONG!
        neighbors.add(node_id)
        return 1/len(neighbors)*np.sum([
            self.__nodes[node_id].density for node_id in neighbors
        ])
    
    def calc_alpha(self, node_id: int, apex_density: float) -> float:
        mean_density = self.calc_mean_density_in_subclass(node_id)
        if 2*mean_density >= apex_density:
            return 0
        elif 3*mean_density >= apex_density:
            return 0.5
        return 1
    
    def merge_subclass_condition(self, nodes_ids: "list of 2 ids") -> bool:
        min_winners_density = \
            min([self.__nodes[nodes_ids[i]].density for i in (0, 1)])
        return (
            min_winners_density > self.calc_alpha(
                nodes_ids[0],
                self.__nodes[self.__nodes[nodes_ids[0]].subclass_id].density
            )*self.__nodes[self.__nodes[nodes_ids[0]].subclass_id].density
        ) or (
            min_winners_density > self.calc_alpha(
                nodes_ids[1],
                self.__nodes[self.__nodes[nodes_ids[1]].subclass_id].density
            )*self.__nodes[self.__nodes[nodes_ids[1]].subclass_id].density
        )

    def change_class_id(self, node_id: int, class_id: int) -> None:
        self.__nodes[node_id].subclass_id = class_id
        visited = {node_id}
        queue = self.__neighbors.get(node_id, set()) - visited
        while queue:
            for vertex in queue.copy():
                self.__nodes[vertex].subclass_id = class_id
                visited.add(vertex)
                queue.remove(vertex)
                for node in self.__neighbors[vertex]:
                    if node not in visited:
                        queue.add(node)

    def combine_subclasses(self, nodes_ids: "list of 2 ids") -> None:
        nodes = [self.__nodes[nodes_ids[0]], self.__nodes[nodes_ids[1]]]
        subclass_ids = [nodes[0].subclass_id, nodes[1].subclass_id]
        if subclass_ids[0] == -1 and subclass_ids[1] == -1:
            for node_id in nodes_ids:
                # @TODO: use id of node with max density
                self.change_class_id(node_id, nodes_ids[0])
        elif subclass_ids[0] != subclass_ids[1]:
            if subclass_ids[0] == -1:
                self.change_class_id(nodes_ids[0], subclass_ids[1])
            else:
                self.change_class_id(nodes_ids[1], subclass_ids[0])

    def find_local_maxes(self) -> set:
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

    # @FIXME: ERROR HERE!!!
    # @FIXME: use merge_subclass_condition for overlapped areas
    # @FIXME: delete overlap
    def continue_mark(self, node_ids: set, class_id: int, visited: set):
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

    def check_overlap(self, overlap_ids: set, visited: set) -> set:
        continue_ids = set()  # continue mark this nodes
        for node_id in overlap_ids:
            near_id = self.get_nearest_neighbor(node_id, overlap_ids)
            if near_id in visited:
                continue_ids.add(node_id)
                node_density = self.__nodes[node_id].density
                for neighbor_id in self.__neighbors.get(node_id, {}).copy():
                    if self.__nodes[neighbor_id].density > node_density \
                            and neighbor_id not in visited \
                            and neighbor_id not in overlap_ids:
                        self.remove_edges([neighbor_id, node_id])
            else:
                for neighbor_id in self.__neighbors.get(node_id, {}).copy():
                    if neighbor_id in visited:
                        self.remove_edges((node_id, neighbor_id))
        return continue_ids

    def get_nearest_neighbor(self, node_id: int, overlap_ids: set) -> int:
        min_dist = float('inf')
        near_id = -1
        for neighbor_id in self.__neighbors.get(node_id, {}):
            if self.__nodes[neighbor_id].density > self.__nodes[node_id].density \
                    and neighbor_id not in overlap_ids:
                dist = self.__metrics(self.__nodes[neighbor_id].feature_vector,
                                      self.__nodes[node_id].feature_vector)
                if min_dist > dist:
                    min_dist = dist
                    near_id = neighbor_id
        return near_id

    def separate_subclasses(self):
        apexes = self.find_local_maxes()
        for apex in apexes:
            queue = {apex}
            visited = set()
            while queue:
                predicted_overlap_ids, visit = self.continue_mark(queue, apex,
                                                                  visited)
                visited = visited.union(visit)
                queue = self.check_overlap(predicted_overlap_ids, visited)
    # @FIXME

    def remove_noise(self) -> None:
        mean_density = np.sum([
                node.density for node in self.__nodes.values()
            ])/len(self.__nodes)
        self._logger.debug(f">\tmean density = {mean_density}")
        # copy of neighbors is used to solve problem with order sensitivity
        neighbors_copy = deepcopy(self.__neighbors)
        for node_id in self.__nodes.copy():
            neighbors_count = len(neighbors_copy.get(node_id, ()))
            node_density = self.__nodes[node_id].density
            if not neighbors_count or neighbors_count == 1 \
                    and node_density < self.__C2*mean_density \
                    or neighbors_count == 2 \
                    and node_density < self.__C1*mean_density:
                self._logger.debug(f">\tremove {node_id} as NOISE "
                                   f"({node_density})")
                self.remove_node(node_id)

    # @TODO: add Rc
    def predict(self, input_signal) -> tuple:
        winners, distances = self.find_winners(input_signal)
        chance1 = distances[1]/(distances[0] + distances[1])
        chance2 = distances[0]/(distances[0] + distances[1])
        win1class = self.__nodes[winners[0]].subclass_id
        win2class = self.__nodes[winners[1]].subclass_id
        return win1class, chance1 + chance2*(win1class == win2class)

    # algorithm 3.3
    def update(self) -> dict:
        queue = set(self.__nodes.keys())
        apexes_ids = {}
        while queue:
            node_id = queue.pop()
            apex_id, visited = self.find_class_apex(node_id)
            queue -= visited
            apexes_ids[apex_id] = visited
            for vertex in visited:
                self.__nodes[vertex].subclass_id = apex_id
        return apexes_ids

    def find_class_apex(self, start_node_id: int) -> tuple:
        apex_id = start_node_id
        visited = {start_node_id}
        queue = self.__neighbors.get(start_node_id, set()) - visited
        while queue:
            for vertex in queue.copy():
                if self.__nodes[apex_id].density < self.__nodes[vertex].density:
                    apex_id = vertex
                visited.add(vertex)
                queue.remove(vertex)
                for node in self.__neighbors[vertex] - visited:
                    if node not in visited:
                        queue.add(node)
        return apex_id, visited

    def current_state(self, deep=True) -> dict:
        nodes = self.__nodes
        neighbors = self.__neighbors
        edges = self.__edges
        if deep:
            nodes = deepcopy(self.__nodes)
            neighbors = deepcopy(self.__neighbors)
            edges = deepcopy(self.__edges)
        classes = self.update()
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
        self.__nodes.clear()
        self.__edges.clear()
        self.__neighbors.clear()
        self.__id = 0
        self.__count = 0

    def _init(self, init_nodes: "list of 2 nodes") -> None:
        if len(self.__nodes):
            return
        for signal in init_nodes:
            self.create_node(signal)

    def _configure(self, nodes, neighbors, edges, non_classified: int=-1):
        for node_id in nodes:
            node = nodes[node_id]
            self.__nodes[node_id] = ESOINNNode(node.feature_vector)
            self.__nodes[node_id].update_points(node.density)
            self.__nodes[node_id].update_density(1)
            self.__nodes[node_id].update_accumulate_signals()
            self.__nodes[node_id].subclass_id = node.subclass_id
            if node.subclass_id != -1:
                self.__nodes[node_id].subclass_id = node.subclass_id
            else:
                self.__nodes[node_id].subclass_id = non_classified
        self.__id = len(nodes)  # set correct unique id
        self.__neighbors = deepcopy(neighbors)
        self.__edges = deepcopy(edges)
