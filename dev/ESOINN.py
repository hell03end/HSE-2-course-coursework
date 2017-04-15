import numpy as np


class ESOINNNode:
    def __init__(self, feature_vector=()):
        self.feature_vector = np.array(feature_vector, dtype=float)  
        self.accumulate_signals = 0
        self.total_points = 0
        self.density = 0
        self.subclass_id = -1
    
    def update_accumulate_signals(self, n=1):
        self.accumulate_signals += n
    
    def update_points(self, points):
        self.total_points += points
        
    def update_density(self, coeff=None):
        if coeff:
            self.density = self.total_points/coeff
        else:
            self.density = self.total_points/self.accumulate_signals

    def update_feature_vector(self, signal, coeff):
        self.feature_vector += coeff*(signal - self.feature_vector)


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
    
    def fit(self, input_signal):
        self.count_signals += 1
        
        winners_ids, distances = self.find_winners(input_signal)
        # @TODO: do not use thesholds list, calc it inplace
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
    
    def find_winners(self, input_signal):
        # @FIXME: inf coef and separate variables for each winner
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
    
    # @CHECKME: use remove_edges() - but no test done
    def remove_old_ages(self):
        for edge in self.edges.copy():
            if self.edges[edge] > self.max_age:
                self.remove_edges(edge)
    
    def calc_mean_density_in_subclass(self, node_id: int):
        neighbors = self.find_neighbors(node_id, depth=-1)
        return 1/len(neighbors)*np.sum([
            self.nodes[node_id].density for node_id in neighbors
        ])
    
    def calc_alpha(self, node_id: int, apex_density):
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
        neighbors = self.find_neighbors(node_id, depth=-1)
        for neighbor_id in neighbors:
            self.nodes[neighbor_id].subclass_id = class_id
    
    def combine_subclasses(self, nodes_ids):
        nodes = [self.nodes[nodes_ids[0]], self.nodes[nodes_ids[1]]]
        subclass_ids = [nodes[0].subclass_id, nodes[1].subclass_id]
        if subclass_ids[0] == -1 and subclass_ids[1] == -1:
            for node in nodes:
                node.subclass_id = nodes_ids[0]
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
        
    # @TODO: paste working algorithm here and adapt it for usage in class
    # @FIXME: improve search by removing multy vertex addition in queue
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

    # @TODO: if local max found, mark all !min as one class
    def mark_subclasses(self):
        pass
    
    # @TODO: separate subclasses
    # algorithm 3.1
    def separate_subclasses(self, visited=set()):
        for node_id in self.nodes():
            node_is_extremum = self.is_extremum(node_id)
            if not node_is_extremum:
                pass
            elif node_is_extremum == 1:
                self.nodes[node_id].subclass_id = node_id
                visited.add(node_id)
                neighbors = self.find_neighbors(node_id)
                # for neighbor_id
            else:
                pass

    def remove_node(self, node_id: int):
        neighbors = self.find_neighbors(node_id)
        for neighbor_id in neighbors:
            self.remove_edges((node_id, neighbor_id))
        del self.nodes[node_id]
    
    def remove_noise(self):

        for node_id in self.nodes.copy():
            neighbors_count = len(self.neighbors[node_id])
            mean_density = np.sum([self.nodes[node_id].density for node_id in self.nodes]) / len(self.nodes)

            if neighbors_count == 0:
                self.delete_node(node_id)
            elif neighbors_count == 1:
                if self.nodes[node_id].density < self.C2*mean_density:
                    self.delete_node(node_id)
            elif neighbors_count == 2:
                if self.nodes[node_id].density < self.C1*mean_density:
                    self.delete_node(node_id)


    def predict(self, input_signal):
        pass  # @TODO: make predictions

    def update(self):
        pass  # @TODO: update topology
    
    def current_state(self):
        return {
            'count_signals': self.count_signals,
            'C1': self.C1,
            'C2': self.C2,
            'lambda': self.learning_step,
            'forget': self.forget,
            'max_age': self.max_age,
            'metrics': self.metrics,
            'learning_rate_winner': self.rate,
            'learning_rate_winner_neighbor': self.rate_neighbor,
            'nodes': self.nodes,  # think about it
            'neighbors': self.neighbors,
            'edges': self.edges
        }
