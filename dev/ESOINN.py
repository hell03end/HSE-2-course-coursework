from copy import deepcopy
import numpy as np
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

    def __repr__(self):
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

    def update_accumulate_signals(self, n=1) -> None:
        if n < 1 or not isinstance(n, int):
            raise ValueError("Wrong number to update accumulated signals!")
        self.__acc_signals += n
    
    def update_points(self, points: int) -> None:
        if points < 0:
            raise ValueError("Wrong value of points!")
        self.__total_points += points
        
    def update_density(self, coeff=None) -> None:
        if coeff:
            self.__density = self.__total_points/coeff
        else:
            self.__density = self.__total_points/self.accumulate_signals

    def update_feature_vector(self, signal: "feature vector",
                              coeff: float) -> None:
        self.__weights += coeff * (signal - self.__weights)


class EnhancedSelfOrganizingIncrementalNN:
    def __init__(self, init_nodes: "2 feature vectors", c1=0.001, c2=1,
                 learning_step=200, max_age=50, forget=False, radius_cut_off=1,
                 metrics=euclidean_distance,
                 learning_rate_winner=learning_rate_generator(),
                 learning_rate_winner_neighbor=learning_rate_generator(k=100),
                 logging_level="info"):
        self.C1 = c1
        self.C2 = c2
        self.learning_step = learning_step
        self.max_age = max_age
        self.count_signals = 2
        self.metrics = metrics
        self.forget = forget
        self.unique_id = 2
        self.rc = radius_cut_off
        self.rate = learning_rate_winner
        self.rate_neighbor = learning_rate_winner_neighbor
        self._logger = enable_logging(f"{__name__}.ESOINN", logging_level)
        
        self.nodes = {i: ESOINNNode(init_nodes[i]) for i in (0, 1)}
        # @TODO: use { node_id: { neighbor_id: age } } instead of self.neighbors, self.edges
        self.neighbors = {}  # key = id, value = set of neighbors' ids
        self.edges = {}  # key = tuple(2), where t[0] < t[1], value = age/None

    # @FIXME: check condition of adding new node (thresholds usage)
    # @TODO: add partial_fit instead
    # @UNTESTED
    def partial_fit(self, input_signal: "feature vector") -> None:
        self.count_signals += 1
        
        winners_ids, distances = self.find_winners(input_signal)
        if distances[0] > self.calc_threshold(winners_ids[0]) \
                or distances[1] > self.calc_threshold(winners_ids[1]):
            self.create_node(input_signal)
            return
        
        self.update_edges_age(winners_ids[0])
        self.build_connection(winners_ids)
        
        # do it one time only 'cause it takes long time
        winner_neighbors = self.find_neighbors(winners_ids[0], depth=self.rc)
        # @CHECKME: обновление количества побед нейрона - не знаю куда поставить, но точно до подсчета плотности
        self.nodes[winners_ids[0]].update_accumulate_signals()
        
        self.update_node_density(winners_ids[0], winner_neighbors)
        self.update_feature_vectors(
            node_id=winners_ids[0],
            input_signal=input_signal,
            neighbors=winner_neighbors
        )
        self.remove_old_ages()
        
        if not self.count_signals % self.learning_step:
            self.separate_subclasses()  # @TODO: algorithm 3.1
            self.remove_noise()  # @TODO: noize removal

    # @UNTESTED
    def fit(self, signals: "list of feature vectors", get_state=False):
        self._logger.debug("Start training")
        for signal in signals:
            self.partial_fit(signal)
        self._logger.debug("Training complete")
        if get_state:
            return self.current_state(deep=True)

    # @TODO: remove inf coeff and separate variables for each winner
    # @TODO: add Rc
    def find_winners(self, input_signal: "feature vector") -> tuple:
        winner1 = float('inf')
        winner1_id = -1
        winner2 = float('inf')
        winner2_id = -1
        for node_id in self.nodes:
            dist = self.metrics(
                input_signal, self.nodes[node_id].feature_vector
            )
            if dist <= winner1:
                winner1, winner2 = dist, winner1
                winner2_id = winner1_id
                winner1_id = node_id
            elif dist < winner2:
                winner2 = dist
                winner2_id = node_id
        return (winner1_id, winner2_id), (winner1, winner2)

    # @TODO: add filter for all neighbors found
    def find_neighbors(self, start_node_id: int, depth=1) -> set:
        visited = {start_node_id}
        queue = list(self.neighbors.get(start_node_id, set()) - visited)
        while depth and queue:
            depth -= 1
            for vertex in queue.copy():  # @FIXME: do not use copy!
                visited.add(vertex)
                queue.remove(vertex)
                queue.extend([
                    node for node in self.neighbors[vertex] - visited
                    if node not in visited
                ])
        return visited - {start_node_id}    

    def calc_threshold(self, node_id: int) -> float:
        neighbors = self.neighbors.get(node_id, None)
        node_feature_vector = self.nodes[node_id].feature_vector
        if neighbors:
            neighbors = self.find_neighbors(node_id, depth=self.rc)
            # max distance among neighbors
            return np.max([
                self.metrics(
                    node_feature_vector,
                    self.nodes[neighbor_id].feature_vector
                ) for neighbor_id in neighbors
            ])
        else:
            # distance to nearest node
            # @CHECKME: do not use [1][1]
            # 'cause first winner is always current node
            return self.find_winners(node_feature_vector)[1][1]

    def create_node(self, input_signal: "feature vector") -> None:
        self.nodes[self.unique_id] = ESOINNNode(input_signal)
        self.unique_id += 1  # to provide unique ids for each neuron

    def remove_node(self, node_id: int) -> None:
        neighbors = self.find_neighbors(node_id)
        for neighbor_id in neighbors:
            self.remove_edges((node_id, neighbor_id))
        del self.nodes[node_id]

    def update_edges_age(self, node_id: int, step=1) -> None:
        neighbors = self.find_neighbors(node_id, depth=self.rc)
        for neighbor_id in neighbors:
            pair_id = min(node_id, neighbor_id), max(node_id, neighbor_id)
            self.edges[pair_id] += step

    # algorithm 3.2
    # @UNTESTED
    def build_connection(self, nodes_ids: "list of 2 ids") -> None:
        winners_classes = tuple(self.nodes[nodes_ids[i]].subclass_id
                                for i in (0, 1))
        if winners_classes[0] == -1 \
                or winners_classes[1] == -1 \
                or winners_classes[0] == winners_classes[1] \
                or self.merge_subclass_condition(nodes_ids):
            self.combine_subclasses(nodes_ids)
            self.create_edges(nodes_ids)
        else:
            self.remove_edges(nodes_ids)

    # @FIXME: keys repeats in for cycle
    def create_edges(self, nodes_ids) -> None:
        for node_id in nodes_ids:
            if node_id not in self.neighbors:
                self.neighbors[node_id] = set()
            for insert_id in nodes_ids:
                if insert_id != node_id:
                    self.neighbors[node_id] |= {insert_id}
                    nodes_pair = \
                        (min(node_id, insert_id), max(node_id, insert_id))
                    self.edges[nodes_pair] = 0
                    
    # @CHECKME: with index usage
#     def create_edges(self, nodes_ids):
#         for node_index in range(len(nodes_ids)):
#             if nodes_ids[node_index] not in self.neighbors:
#                 self.neighbors[nodes_ids[node_index]] = set()
#             for insert_index in range(node_index+1, len(nodes_ids)):
#                 if insert_index != node_index:
#                     self.neighbors[nodes_ids[node_index]] |= {nodes_ids[insert_index]}
#                     nodes_pair = (nodes_ids[node_index], nodes_ids[insert_index])
#                     self.edges[nodes_pair] = 0
    
    # @FIXME: keys repeats in for cycle
    def remove_edges(self, nodes_ids) -> None:
        for node_id in nodes_ids:
            for remove_id in nodes_ids:
                if remove_id != node_id \
                        and remove_id in self.neighbors.get(node_id, {}):
                    self.neighbors[node_id] -= {remove_id}
                    nodes_pair = \
                        (min(node_id, remove_id), max(node_id, remove_id))
                    if nodes_pair in self.edges:
                        del self.edges[nodes_pair]
            if not self.neighbors[node_id]:
                del self.neighbors[node_id]

    # @CHECKME: with index usage
#     def remove_edges(self, nodes_ids):
#         for node_index in range(len(nodes_ids)):
#             for remove_index in range(node_index+1, len(nodes_ids)):
#                 if remove_index != node_index and nodes_ids[remove_index] in self.neighbors[nodes_ids[node_index]]:
#                     self.neighbors[nodes_ids[node_index]] -= {nodes_ids[remove_index]}
#                     nodes_pair = (nodes_ids[node_index], nodes_ids[remove_index])
#                     if nodes_pair in self.edges:
#                         del self.edges[nodes_pair]
#             if not self.neighbors[nodes_ids[node_index]]:
#                 del self.neighbors[nodes_ids[node_index]]
                         
    def update_node_points(self, node_id: int,
                           neighbors: "set of ids") -> None:
        if neighbors:
            node_feature_vector = self.nodes[node_id].feature_vector
            mean_dist2neighbors = 1/len(neighbors)*np.sum([
                self.metrics(
                    node_feature_vector,
                    self.nodes[neighbor_id].feature_vector
                ) for neighbor_id in neighbors
            ])
            self.nodes[node_id].update_points(1/(1 + mean_dist2neighbors)**2)
        else:
            self.nodes[node_id].update_points(1)

    def update_node_density(self, node_id: int,
                            neighbors: "set of ids") -> None:
        self.update_node_points(node_id, neighbors)
        if self.forget:
            self.nodes[node_id].update_density(self.count_signals)
        else:
            self.nodes[node_id].update_density()

    def update_feature_vectors(self, node_id: int,
                               input_signal: "feature vector",
                               neighbors: "set of ids") -> None:
        acc_signal = self.nodes[node_id].accumulate_signals
        self.nodes[node_id].update_feature_vector(
            signal=input_signal,
            coeff=self.rate(acc_signal)
        )
        neighbors_rate = self.rate_neighbor(acc_signal)
        for neighbor_id in neighbors:
            self.nodes[neighbor_id].update_feature_vector(
                signal=input_signal,
                coeff=neighbors_rate
            )

    def remove_old_ages(self) -> None:
        for edge in self.edges.copy():
            if self.edges[edge] > self.max_age:
                self.remove_edges(edge)
    
    def calc_mean_density_in_subclass(self, node_id: int) -> float:
        neighbors = self.find_neighbors(node_id, depth=-1)
        return float(1/len(neighbors)*np.sum([
            self.nodes[node_id].density for node_id in neighbors
        ]))
    
    def calc_alpha(self, node_id: int, apex_density: float) -> float:
        mean_density = self.calc_mean_density_in_subclass(node_id)
        if 2*mean_density >= apex_density:
            return 0
        elif 3*mean_density >= apex_density:
            return 0.5
        return 1
    
    def merge_subclass_condition(self, nodes_ids: "list of 2 ids") -> bool:
        min_winners_density = \
            min([self.nodes[nodes_ids[i]].density for i in (0, 1)])
        return (
            min_winners_density > self.calc_alpha(
                nodes_ids[0],
                self.nodes[self.nodes[nodes_ids[0]].subclass_id].density
            )*self.nodes[self.nodes[nodes_ids[0]].subclass_id].density
        ) or (
            min_winners_density > self.calc_alpha(
                nodes_ids[1],
                self.nodes[self.nodes[nodes_ids[1]].subclass_id].density
            )*self.nodes[self.nodes[nodes_ids[1]].subclass_id].density
        )

    def change_class_id(self, node_id: int, class_id: int) -> None:
        self.nodes[node_id].subclass_id = class_id
        visited = {node_id}
        queue = list(self.neighbors.get(node_id, set()) - visited)
        # CHECKME: "while" cycle is not needed, but without it - didn't worked
        while queue:
            for vertex in queue:
                self.nodes[vertex].subclass_id = class_id
                visited.add(vertex)
                queue.remove(vertex)
                for node in self.neighbors[vertex] - visited:
                    if node not in visited:
                        queue.append(node)
    
    def combine_subclasses(self, nodes_ids: "list of 2 ids") -> None:
        nodes = [self.nodes[nodes_ids[0]], self.nodes[nodes_ids[1]]]
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

    # @CHECKME: is it necessary?
    def find_neighbors_local_maxes(self, node_id: int) -> set:
        apexes = set()
        visited = set()
        queue = [node_id]
        for vertex in queue:
            is_local_max = True
            vertex_density = self.nodes[vertex].density
            for neighbor_id in self.neighbors.get(vertex, {}):
                if self.nodes[neighbor_id].density > vertex_density:
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

    def find_local_maxes(self) -> set:
        apexes = set()
        for node_id in self.nodes.keys():
            neighbors = self.neighbors.get(node_id, {})
            if not neighbors:
                apexes.add(node_id)
                continue
            is_apex = True
            current_density = self.nodes[node_id].density
            for neighbor_id in neighbors:
                is_apex &= self.nodes[neighbor_id].density < current_density
                if not is_apex:
                    break
            if is_apex:
                apexes.add(node_id)
        return apexes


    def continue_mark(self, node_ids:list, class_id:int, visited:set):
        predicted_overlap_ids = set()
        for node_id in node_ids:
            queue = [node_id]
            while queue:
                for vertex in queue:
                    for_extend = []
                    vertex_density = self.nodes[vertex].density
                    queue.remove(vertex)
                    if vertex not in predicted_overlap_ids:
                        is_overlap = False
                        for neighbor_id in self.neighbors.get(vertex, {}):
                            neighbor_density = self.nodes[neighbor_id].density
                            if  neighbor_density > vertex_density:
                                if neighbor_id not in visited and \
                                neighbor_id not in predicted_overlap_ids:
                                    is_overlap = True
                            else:
                                for_extend.append(neighbor_id)
                        if is_overlap:
                            predicted_overlap_ids.add(vertex)
                        else:
                            visited.add(vertex)
                            self.nodes[vertex].subclass_id = class_id
                            queue.extend(for_extend)
        return predicted_overlap_ids, visited

    def check_overlap(self, overlap_ids: set, visited: set) -> set:
        continue_ids = set()
        for node_id in overlap_ids:
            near_id = self.get_nearest_neighbor(node_id, overlap_ids)
            if near_id in visited:
                continue_ids.add(node_id)
                for neighbor_id in self.neighbors.get(node_id, {}).copy():
                    if self.nodes[neighbor_id].density > self.nodes[node_id].density:
                        if neighbor_id not in visited:
                            if neighbor_id not in overlap_ids:
                                self.remove_edges((neighbor_id, node_id))
            else:
                for neighbor_id in self.neighbors.get(node_id, {}).copy():
                    if neighbor_id in visited:
                        self.remove_edges((node_id, neighbor_id))
        return continue_ids

    def get_nearest_neighbor(self, node_id: int, overlap_ids: set) -> int:
        min_dist = float('inf')
        near_id = -1
        for neighbor_id in self.neighbors.get(node_id, {}):
            if self.nodes[neighbor_id].density > self.nodes[node_id].density:
                if neighbor_id not in overlap_ids:
                    dist = self.metrics(self.nodes[neighbor_id].feature_vector,
                                        self.nodes[node_id].feature_vector)
                    if min_dist > dist:
                        min_dist = dist
                        near_id = neighbor_id
        return near_id

    def separate_subclass(self):
        apexes = self.find_local_maxes()
        for apex in apexes:
            queue = [apex]
            visited = set()
            while queue:
                predicted_overlap_ids, visit = self.continue_mark(queue,
                                                             apex,
                                                             visited)
                visited = visited.union(visit)
                queue = list(self.check_overlap(predicted_overlap_ids,
                                           visited))

    def remove_noise(self) -> None:
        mean_density = np.sum([
                node.density for node in self.nodes.values()
            ])/len(self.nodes)
        # use copy of neighbors to solve problem with order sensitivity
        neighbors_copy = deepcopy(self.neighbors)
        for node_id in self.nodes.copy():
            neighbors_count = len(neighbors_copy.get(node_id, ()))
            node_density = self.nodes[node_id].density
            if neighbors_count == 0:
                self.remove_node(node_id)
            elif neighbors_count == 1:
                if node_density < self.C2*mean_density:
                    self.remove_node(node_id)
            elif neighbors_count == 2:
                if node_density < self.C1*mean_density:
                    self.remove_node(node_id)

    # @TODO: add Rc
    def predict(self, input_signal) -> tuple:
        winners, distances = self.find_winners(input_signal)
        chance1 = distances[1]/(distances[0] + distances[1])
        chance2 = distances[0]/(distances[0] + distances[1])
        win1class = self.nodes[winners[0]].subclass_id
        win2class = self.nodes[winners[1]].subclass_id
        return win1class, chance1 + chance2*(win1class == win2class)

    # algorithm 3.3
    def update(self) -> dict:
        queue = set(self.nodes.keys())
        apexes_ids = {}
        while queue:
            node_id = queue.pop()
            apex_id, visited = self.find_class_apex(node_id)
            queue -= visited
            apexes_ids[apex_id] = visited
            for vertex in visited:
                self.nodes[vertex].subclass_id = apex_id
        return apexes_ids

    def find_class_apex(self, start_node_id: int) -> tuple:
        apex_id = start_node_id
        visited = {start_node_id}
        queue = list(self.neighbors.get(start_node_id, set()) - visited)
        while queue:
            for vertex in queue:
                if self.nodes[apex_id].density < self.nodes[vertex].density:
                    apex_id = vertex
                visited.add(vertex)
                queue.remove(vertex)
                for node in self.neighbors[vertex] - visited:
                    if node not in visited:
                        queue.append(node)
        return apex_id, visited

    def current_state(self, deep=True) -> dict:
        nodes = self.nodes
        neighbors = self.neighbors
        edges = self.edges
        if deep:
            nodes = deepcopy(self.nodes)
            neighbors = deepcopy(self.neighbors)
            edges = deepcopy(self.edges)
        classes = self.update()
        return {
            'count_signals': self.count_signals,
            'count_neurons': len(self.nodes),
            'last_node_id': self.unique_id - 1,
            'nodes': nodes,
            'neighbors': neighbors,
            'edges': edges,
            'classes': classes,
            'configuration': {
                'C1': self.C1,
                'C2': self.C2,
                'Rc': self.rc,
                'lambda': self.learning_step,
                'forget': self.forget,
                'max_age': self.max_age,
                'metrics': self.metrics,
                'learning_rate_winner': self.rate,
                'learning_rate_winner_neighbor': self.rate_neighbor,
            }
        }
