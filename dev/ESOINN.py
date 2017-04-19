import numpy as np
from copy import deepcopy


class ESOINNNode:
    def __init__(self, feature_vector):
        self.__weights = np.array(feature_vector, dtype=float)
        self.__acc_signals = 0
        self.__total_points = float(0)
        self.__density = float(0)
        self.__subclass_id = -1

    def __repr__(self):
        return f"{str(self.feature_vector):^30} | {self.subclass_id:10} | " \
               f"{self.density:13.10} | {self.points:13.10} | " \
               f"{self.accumulate_signals}"

    @property
    def subclass_id(self) -> int:
        return self.__subclass_id

    @subclass_id.setter
    def subclass_id(self, value: int):
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

    def update_accumulate_signals(self, n=1):
        if n < 1 or not isinstance(n, int):
            raise ValueError("Wrong number to update accumulated signals!")
        self.__acc_signals += n
    
    def update_points(self, points: int):
        if points < 0:
            raise ValueError("Wrong value of points!")
        self.__total_points += points
        
    def update_density(self, coeff=None):
        if coeff:
            self.__density = self.__total_points/coeff
        else:
            self.__density = self.__total_points/self.accumulate_signals

    def update_feature_vector(self, signal, coeff):
        self.__weights += coeff * (signal - self.__weights)


class EnhancedSelfOrganizingIncrementalNN:
    def __init__(self, init_nodes, c1=0.001, c2=1, learning_step=200,
                 max_age=50, forget=False, radius_cut_off=1,
                 metrics=lambda x, y: np.sqrt(
                     np.sum(np.square(np.array(x) - np.array(y)))
                 ), learning_rate_winner=lambda t: 1/t,
                 learning_rate_winner_neighbor=lambda t: 1/(100*t)):
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
        
        self.nodes = {i: ESOINNNode(init_nodes[i]) for i in (0, 1)}
        # @TODO: use { node_id: { neighbor_id: age } } instead of self.neighbors, self.edges
        self.neighbors = {}  # key = id, value = set of neighbors' ids
        self.edges = {}  # key = tuple(2), where t[0] < t[1], value = age/None

    # @FIXME: check condition of adding new node (thresholds usage)
    def fit(self, input_signal):
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
        
        self.update_neuron_density(winners_ids[0], winner_neighbors)
        self.update_feature_vectors(
            node_id=winners_ids[0],
            input_signal=input_signal,
            neighbors=winner_neighbors
        )
        self.remove_old_ages()
        
        if not self.count_signals % self.learning_step:
            self.separate_subclasses()  # @TODO: algorithm 3.1
            self.remove_noise()  # @TODO: noize removal

    # @TODO: remove inf coeff and separate variables for each winner
    # @TODO: add Rc
    def find_winners(self, input_signal):
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
        return [winner1_id, winner2_id], [winner1, winner2]

    # @TODO: add filter for all neighbors found
    def find_neighbors(self, start_node_id: int, depth=1) -> set():
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

    def calc_threshold(self, node_id: int):
        neighbors = self.neighbors.get(node_id, None)
        node_feature_vector = self.nodes[node_id].feature_vector
        if neighbors:
            neighbors = self.find_neighbors(node_id, depth=self.rc)
            return np.max([
                self.metrics(
                    node_feature_vector,
                    self.nodes[neighbor_id].feature_vector
                ) for neighbor_id in neighbors
            ])
        else:
            # 'cause first winner is always current node
            # @CHECKME: do not use [1][1]
            return self.find_winners(node_feature_vector)[1][1]

    def create_node(self, input_signal):
        self.nodes[self.unique_id] = ESOINNNode(input_signal)
        self.unique_id += 1  # to provide unique ids for each neuron
    
    def update_edges_age(self, node_id: int, step=1):
        neighbors = self.find_neighbors(node_id, depth=self.rc)
        for neighbor_id in neighbors:
            pair_id = min(node_id, neighbor_id), max(node_id, neighbor_id)
            self.edges[pair_id] += step

    # algorithm 3.2
    def build_connection(self, nodes_ids):
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
                
    def create_edges(self, nodes_ids):
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
    def remove_edges(self, nodes_ids):
        for node_id in nodes_ids:
            for remove_id in nodes_ids:
                if remove_id != node_id \
                        and remove_id in self.neighbors[node_id]:
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
                         
    def update_node_points(self, node_id: int, neighbors):
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

    def update_neuron_density(self, node_id: int, neighbors):
        self.update_node_points(node_id, neighbors)
        if self.forget:
            self.nodes[node_id].update_density(self.count_signals)
        else:
            self.nodes[node_id].update_density()

    def update_feature_vectors(self, node_id: int, input_signal, neighbors):
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

    def remove_old_ages(self):
        for edge in self.edges.copy():
            if self.edges[edge] > self.max_age:
                self.remove_edges(edge)
    
    def calc_mean_density_in_subclass(self, node_id: int):
        neighbors = self.find_neighbors(node_id, depth=-1)
        return 1/len(neighbors)*np.sum([
            self.nodes[node_id].density for node_id in neighbors
        ])
    
    def calc_alpha(self, node_id: int, apex_density) -> float:
        mean_density = self.calc_mean_density_in_subclass(node_id)
        if 2*mean_density >= apex_density:
            return 0
        elif 3*mean_density >= apex_density:
            return 0.5
        return 1
    
    def merge_subclass_condition(self, nodes_ids):
        min_winners_density = \
            min([self.nodes[nodes_ids[i]].density for i in (0, 1)])
        return (
            min_winners_density > self.calc_alpha(
                nodes_ids[0],
                self.nodes[self.nodes[nodes_ids[0]].subclass_id].density
            )
        ) or (
            min_winners_density > self.calc_alpha(
                nodes_ids[1],
                self.nodes[self.nodes[nodes_ids[1]].subclass_id].density
            )
        )
    
    def change_class_id(self, node_id: int, class_id: int):
        self.nodes[node_id].subclass_id = class_id
        visited = {node_id}
        queue = list(self.neighbors.get(node_id, set()) - visited)
        for vertex in queue.copy():
            self.nodes[vertex].subclass_id = class_id
            visited.add(vertex)
            queue.remove(vertex)
            queue.extend([
                node for node in self.neighbors[vertex] - visited
                if node not in visited
            ])
    
    def combine_subclasses(self, nodes_ids):
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

    # @FIXME: is this essential? remove if not.
    def is_extremum(self, node_id: int) -> int:
        neighbors = self.find_neighbors(node_id)
        current_density = self.nodes[node_id].density
        local_min = False
        local_max = False
        for neighbor_id in neighbors:
            if local_min and local_max:
                return 0
            neighbor_density = self.nodes[neighbor_id].density
            if current_density > neighbor_density:
                local_max = True
            elif current_density < neighbor_density:
                local_min = True
            else:
                raise RuntimeError("Equal nodes' density!")
        if local_min and not local_max:
            return -1
        elif local_max and not local_min:
            return 1

    # @FIXME: improve search by removing multi vertex addition in queue
    # @CHECKME: is it necessary?
    def find_neighbors_local_maxes(self, node_id: int):
        apexes = set()
        visited = {node_id}

        queue = []
        node_density = self.nodes[node_id].density
        for neighbor_id in self.neighbors.get(node_id, set()) - visited:
            if self.nodes[neighbor_id].density > node_density:
                queue.append(neighbor_id)
            visited.add(neighbor_id)

        for vertex in queue:
            is_local_max = True
            vertex_density = self.nodes[vertex].density
            for neighbor_id in self.neighbors[vertex]:
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

    def mark_subclasses(self, node_id: int,
                        overlap_nodes: dict,
                        visited: set):

        visited.add(node_id)
        queue = [node_id]
        # Set for keeping node_ids which need to remove edges
        remove_set = set()
        # Mark apex
        self.nodes[node_id].subclass_id = node_id
        apex_id = self.nodes[node_id].subclass_id

        for vertex in queue:
            self.nodes[vertex].subclass_id = apex_id
            vertex_density = self.nodes[vertex].density
            visited.add(vertex)
            for neighbor_id in self.neighbors[vertex].copy():
                if self.nodes[neighbor_id].density < vertex_density:
                    # Если мы нашли нейрон с плотностью меньше, чем у родителя
                    # То нужно понять стоит ли нам его маркировать, либо маркоровать будет кто-то ДРУГОЙ
                    min_dist = self.calc_heavy_neighbor_min_dist(neighbor_id)
                    vertex_weights = self.nodes[vertex].feature_vector
                    neighbor_weights = self.nodes[neighbor_id].feature_vector
                    # Проверка на то, будет ли маркировать кто-то ДРУГОЙ
                    if self.metrics(vertex_weights,
                                    neighbor_weights) <= min_dist:
                        # Если мы ранее не посещяли узел, то значит совсем что-то новое
                        if neighbor_id not in visited:
                            queue.append(neighbor_id)
                    # Если маркировать будет кто-то другой, значит это overlap area
                    else:
                        overlap_nodes.update({neighbor_id: min_dist})
                        remove_set.add(neighbor_id)

        # Удаление ребер
        for remove_id in remove_set:
            # Проверка на то, что к данному узлу из вершины никто НЕ смог пробиться
            if self.nodes[remove_id].subclass_id != apex_id:
                for neighbor_id in self.neighbors[remove_id].copy():
                    if self.nodes[neighbor_id].subclass_id == apex_id:
                        self.remove_edges((remove_id, neighbor_id))

        return overlap_nodes, visited

    def calc_heavy_neighbor_min_dist(self, node_id: int) -> float:
        min_dist = float('inf')
        for neighbor_id in self.neighbors[node_id]:
            if self.nodes[neighbor_id].density > self.nodes[node_id].density:
                dist = self.metrics(self.nodes[neighbor_id].feature_vector,
                                    self.nodes[node_id].feature_vector)
                if min_dist > dist:
                    min_dist = dist

        return min_dist

    # @TODO: subclasses remove connections between subclasses
    # algorithm 3.1
    def separate_subclasses(self):
        visited_in_mark = set()
        # key - overlap node id,
        # value - min_dist to neighbor id, which have more density
        overlap_nodes = dict()
        for node_id in self.nodes:
            if node_id not in visited_in_mark:
                apexes = self.find_neighbors_local_maxes(node_id)
                for apex in apexes:
                    overlap_nodes, visited_in_mark = \
                        self.mark_subclasses(apex,
                                             overlap_nodes,
                                             visited_in_mark)

    def remove_node(self, node_id: int):
        neighbors = self.find_neighbors(node_id)
        for neighbor_id in neighbors:
            self.remove_edges((node_id, neighbor_id))
        del self.nodes[node_id]

    # @FIXME: order sensitive
    def remove_noise(self):
        for node_id in self.nodes.copy():
            mean_density = np.sum([
                self.nodes[node_id].density for node_id in self.nodes
                ])/len(self.nodes)
            neighbors_count = len(self.neighbors.get(node_id, ()))
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
    def predict(self, input_signal):
        winners, distances = self.find_winners(input_signal)
        chance1 = distances[1]/(distances[0] + distances[1])
        chance2 = distances[0]/(distances[0] + distances[1])
        win1class = self.nodes[winners[0]].subclass_id
        win2class = self.nodes[winners[1]].subclass_id
        return win1class, chance1 + chance2*(win1class == win2class)

    # @CHECKME: for Dmitriy, Alexandr
    # algorithm 3.3
    def update(self) -> set:
        visited = set()
        classes_apex_ids = set()
        for node_id in self.nodes:
            if node_id not in visited:
                class_apex_id, class_visited_ids = \
                    self.max_apex_in_class(node_id)
                visited.add(class_visited_ids)
                classes_apex_ids.add(class_apex_id)
        for apex_id in classes_apex_ids:
            self.change_class_id(apex_id, self.nodes[apex_id].subclass_id)
        return classes_apex_ids

    # @CHECKME: for Dmitriy, Alexandr
    def max_apex_in_class(self, start_node_id: int):
        max_apex_id = start_node_id
        visited = {start_node_id}
        queue = list(self.neighbors.get(start_node_id, set()) - visited)
        for vertex in queue.copy():
            if self.nodes[max_apex_id].density < self.nodes[vertex].density:
                max_apex_id = vertex
            visited.add(vertex)
            queue.remove(vertex)
            queue.extend([
                node for node in self.neighbors[vertex] - visited
                if node not in visited
            ])
        return max_apex_id, visited

    def current_state(self, deep=True) -> dict:
        nodes = self.nodes
        neighbors = self.neighbors
        edges = self.edges
        if deep:
            nodes = deepcopy(self.nodes)
            neighbors = deepcopy(self.neighbors)
            edges = deepcopy(self.edges)
        return {
            'count_signals': self.count_signals,
            'count_neurons': len(self.nodes),
            'last_node_id': self.unique_id - 1,
            'nodes': nodes,
            'neighbors': neighbors,
            'edges': edges,
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
