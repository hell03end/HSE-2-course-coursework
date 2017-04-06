import numpy as np
import random
import matplotlib.pyplot as plt

# Класс для ребра: хранит существование и возраст
class ESOINN_rib:
    def __init__(self):
        self.rib_exist = 0
        self.rib_age = 0

# Класс для нейрона: хранит вектор признаков, накопленные сигналы(при победителе), плотность,
#   метку принадлежности подклассу, степень(как у вершины графа)
class ESOINN_Neuron:
    def __init__(self):
        self.feature_vector = []  # weights
        self.accamulate_signals = 0
        self.s = 0
        self.density = 0  # amount of input signals in current point
        self.mark = -1  # subclass
        self.degree = 0
# Класс для ESOINN объекта
class ESOINN:
    def __init__(self, age_max, C1, C2, lambd):

        # Создание двух начальных нейронов(ниже присвоятся вектора признаков для них)
        neuron1 = ESOINN_Neuron()
        neuron2 = ESOINN_Neuron()

        # Начальные параметры ESOINN
        self.age_max = age_max
        self.C1 = C1
        self.C2 = C2
        self.lambd = lambd

        # ТОЧНО? А как же 2 initial нейрона?
        # Счетчики для удобства (для количества входный данных и удаленных нейронов)
        self.quantity_of_inputVec = 0
        self.quantity_of_delete_neurons = 0

        # Initial 2 neurons
        # Инициализация веторов признаков(из области допустимых значений) для двух начальных нейронов
        # Думаю, надо первые два входных вектора использовать для них
        for i in range(2):
            x = random.random()
            x = round(x,4)
            neuron1.feature_vector.append(x)

        for i in range(2):
            x = random.random()
            x = round(x,4)
            neuron2.feature_vector.append(x)

        # Инициализация матрицы смежности (2 х 2)
        # (можно переделать в словарь)
        self.adjacency_matrix = np.array([[ESOINN_rib(), ESOINN_rib()], [ESOINN_rib(), ESOINN_rib()]], dtype=ESOINN_rib)

        # Инициализация массива нейронов (2 нейрона)
        self.neurons = np.array([neuron1, neuron2], dtype=ESOINN_Neuron)

        # Индексы победителей
        self.winner1 = -1
        self.winner2 = -1

    # Вычисление расстояния между двумя узлами
    def calc_dist(self,vector1,vector2):
        dist_in_vector = [0,0]
        dist_in_vector[0] = abs(vector1[0] - vector2[0])
        dist_in_vector[1] = abs(vector1[1] - vector2[1])
        dist = (dist_in_vector[0]**2+dist_in_vector[1]**2)**(1/2)
        # Округление числа до 4-го числа после запятой
        dist = round(dist, 4)
        return dist

    # Нахождение победителей
    def find_winners(self,inputVec):

        # Изначально победители не найдены
        # Цифры указывают на индекс
        winners = [-1,-1]

        # Вычисление расстояния до первого и второго нейрона из массива
        dist0 = self.calc_dist(inputVec, self.neurons[0].feature_vector)
        dist1 = self.calc_dist(inputVec, self.neurons[1].feature_vector)

        # Инициализация временных победителей исходя из вычисления расстояния до первого и второго нейрона из массива
        if dist0 < dist1:
            min_dist1 = dist0
            min_dist2 = dist1
            winners[0] = 0
            winners[1] = 1
        else:
            min_dist1 = dist1
            min_dist2 = dist0
            winners[0] = 1
            winners[1] = 0

        # Проход по остальным нейронам в массиве (Вдруг кто еще ближе есть?)
        for j in range(2,self.neurons.shape[0]):
            dist = self.calc_dist(inputVec,self.neurons[j].feature_vector)
            if dist < min_dist2-0.00001:
                if dist < min_dist1-0.00001:
                    winners[1] = winners[0]
                    winners[0] = j
                    min_dist2 = min_dist1
                    min_dist1 = dist
                else:
                    winners[1] = j
                    min_dist2 = dist

        return winners[0],winners[1]

    # Функция проверяющая условия для создания нового нейрона
    def point4(self,inputVec):

        # Флаги, для проверки существования соседей для первого и второго победителя
        neighbour_flag_win1 = 0
        neighbour_flag_win2 = 0

        # Проход по матрице смежности для проверки наличия соседей
        # Думаю, этот шаг тоже можно упростить
        for j in range(self.adjacency_matrix.shape[1]):
            if (j != self.winner1 and self.adjacency_matrix[self.winner1][j].rib_exist == 1):
                neighbour_flag_win1 = 1

            if (j != self.winner2 and self.adjacency_matrix[self.winner2][j].rib_exist == 1):
                neighbour_flag_win2 = 1
        print(neighbour_flag_win1)
        print(neighbour_flag_win2)
        # Далее, исходя из наличия соседей, ищутся параметры Та1 (для первого победителя), Та2 (для второго победителя)
        if neighbour_flag_win1 == 1:
            Ta1 = self.if_neighbour_exist(self.winner1)
        else:
            Ta1 = self.if_neighbour_not_exist(self.winner1)

        if neighbour_flag_win2 == 1:
            Ta2 = self.if_neighbour_exist(self.winner2)
        else:
            Ta2 = self.if_neighbour_not_exist(self.winner2)

        # Вычисление расстояния до победителей (обоих)
        dist_inputVec_Win1 = self.calc_dist(inputVec, self.neurons[self.winner1].feature_vector)
        dist_inputVec_Win2 = self.calc_dist(inputVec, self.neurons[self.winner2].feature_vector)

        # Условия создания нового нейрона. После создании нового нейрона нужно про всё забыть и
        #   подать новый входной вектор, поэтому используются return 1 и return 0
        if (dist_inputVec_Win1 > Ta1):
            self.create_new_neuron(inputVec)
            print('Created Ta1')
            return 1

        if (dist_inputVec_Win2 > Ta2):
            self.create_new_neuron(inputVec)
            print('Created Ta2')
            return 1

        return 0

    # Вычисление параметров Та1 и Та2, если смежный узел есть
    def if_neighbour_exist(self,neuron_index):

        param_Tmax = 0
        for j in range(self.adjacency_matrix.shape[1]):
            if self.adjacency_matrix[neuron_index][j].rib_exist == 1:
                dist = self.calc_dist(self.neurons[neuron_index].feature_vector,self.neurons[j].feature_vector)
                if dist > param_Tmax:
                    param_Tmax = dist

        return param_Tmax

    # Вычисление параметров Та1 и Та2, если смежных узлов нет
    def if_neighbour_not_exist(self, neuron_index):
        param_Tmax = 0
        for j in range(self.adjacency_matrix.shape[1]):
            dist = self.calc_dist(self.neurons[neuron_index].feature_vector, self.neurons[j].feature_vector)
            if dist > param_Tmax:
                param_Tmax = dist

        return param_Tmax


    # Функция создания нового нейрона
    def create_new_neuron(self,inputVec):

        # Инициализация нейрона с вектором признаков равным вектору входного
        neuron = ESOINN_Neuron()
        neuron.feature_vector.extend(inputVec)

        # Расширение массива нейронов
        self.neurons = np.hstack([self.neurons, neuron])

        # Создание временного массива(горизонтальный) ребер, чтобы затолкнуть его в матрицу смежности
        ribs = np.array([])
        for j in range(self.adjacency_matrix.shape[1]):
            ribs = np.hstack([ribs, ESOINN_rib()])

        # Добавление временного массива ребер (Добавление снизу - vertical stack) в матрицу смежностей
        self.adjacency_matrix = np.vstack([self.adjacency_matrix, ribs])

        # Трансформация горизонтального массива в вертикальный
        ribs = np.hsplit(ribs, self.adjacency_matrix.shape[0]-1)

        # Добавление еще одного ребра(так как после vstack размерность матрица смежностей поменялась)
        ribs = np.vstack([ribs, ESOINN_rib()])

        # Добавление временного массива ребер (Добавление с правого боку - horizontal stack) в матрицу смежностей
        self.adjacency_matrix = np.hstack([self.adjacency_matrix, ribs])

    # Увеличение возраста ребер смежных с первым победителем
    def update_ribs_age(self):
        for j in range(self.adjacency_matrix.shape[1]):
            if (self.adjacency_matrix[self.winner1][j].rib_exist == 1):
                self.adjacency_matrix[self.winner1][j].rib_age += 1
                self.adjacency_matrix[j][self.winner1].rib_age += 1

    # Проверка условий создания ребра между первым и вторым победителем
    def rib_between_win1_win2(self):

        # Если хотя-бы один из них новый нейрон (нет маркировки принадлежности подклассу)
        if self.neurons[self.winner1].mark == -1 or self.neurons[self.winner2].mark == -1:
            self.set_rib()

        else:
            # Если они принадлежат одному подклассу
            if self.neurons[self.winner1].mark == self.neurons[self.winner2].mark:
                self.set_rib()

            # Условие слияния 2-ух подклассов из Алгоритма 1
            else:
                # Если соединение нужно
                if self.merge_condition_algorithm1(self.winner1,self.winner2) == 1:
                    # self.merge_subclasses(self.winner1, self.winner2)
                    self.set_rib()
                # Если нет, то старое ребро удаляется
                else:
                    self.adjacency_matrix[self.winner1][self.winner2].rib_exist = 0
                    self.adjacency_matrix[self.winner2][self.winner1].rib_exist = 0

                    self.adjacency_matrix[self.winner1][self.winner2].rib_age = 0
                    self.adjacency_matrix[self.winner2][self.winner1].rib_age = 0

        # Увеличение накопленного первым победителем сигнала
        self.neurons[self.winner1].accamulate_signals += 1

    # Функция создания ребра между победителями по условиям из функции rib_between_win1_win2
    def set_rib(self):

        # Создание ребра
        self.adjacency_matrix[self.winner1][self.winner2].rib_exist = 1
        self.adjacency_matrix[self.winner2][self.winner1].rib_exist = 1

        # Обнуление возраста ребра
        self.adjacency_matrix[self.winner1][self.winner2].rib_age = 0
        self.adjacency_matrix[self.winner2][self.winner1].rib_age = 0

    # Функция проверки условия слияния подклассов из Алгоритма 1
    def merge_condition_algorithm1(self,A,B):

        # Если подклассы разные
        if self.neurons[A].mark != self.neurons[B].mark:
            # Вычисление параметров кластера, к которым принадлежат первый и второй победители и
            #   вычисление максимальной плотности в этих кластерах
            param_of_A, max_dens_cluster_of_A = self.param_of_A(A)
            param_of_B, max_dens_cluster_of_B = self.param_of_B(B)

            # Условия соединения двух подклассов в один (Для понимания см. в Алгоритм ESOINN)
            if min(self.neurons[A].density,self.neurons[B].density) > param_of_A * max_dens_cluster_of_A and \
               min(self.neurons[A].density,self.neurons[B].density) > param_of_B * max_dens_cluster_of_B:

                return 1

            # Если условия слияния подклассов не удовлетворились, то вернуть 0 (для функции rib_between_win1_win2)
            else:
                return 0

    # Вычисление параметра (по алгоритму)
    def param_of_A(self, A):
        summ_dens = 0
        count = 0
        max_dens = 0
        for i in range(self.neurons.shape[0]):
            if self.neurons[i].mark == self.neurons[A].mark:
                summ_dens += self.neurons[i].density
                count += 1
                if self.neurons[i].density > max_dens:
                    max_dens = self.neurons[i].density
        average_density = summ_dens / count

        if average_density * 2 >= max_dens:
            param = 0
            return param, max_dens
        elif average_density * 3 >= max_dens and max_dens > 2 * average_density:
            param = 0.5
            return param, max_dens
        elif average_density * 3 < max_dens:
            param = 1
            return param, max_dens

    # Вычисление параметра (по алгоритму)
    def param_of_B(self, B):
        summ_dens = 0
        count = 0
        max_dens = 0
        for i in range(self.neurons.shape[0]):
            if self.neurons[i].mark == self.neurons[B].mark:
                summ_dens += self.neurons[i].density
                count += 1
                if self.neurons[i].density > max_dens:
                    max_dens = self.neurons[i].density
        average_density = summ_dens / count

        if average_density * 2 >= max_dens:
            param = 0
            return param, max_dens
        elif average_density * 3 >= max_dens and max_dens > 2 * average_density:
            param = 0.5
            return param, max_dens
        elif average_density * 3 < max_dens:
            param = 1
            return param, max_dens

    # Вычисление средней дистанции между нейронами смежными победителю
    def calc_aver_dist(self):

        # Пошло само вычисление
        summ_dist = 0
        count_of_neighbours = 0
        for j in range(self.adjacency_matrix.shape[1]):
            if self.adjacency_matrix[self.winner1][j].rib_exist == 1 and j != self.winner1:
                dist = self.calc_dist(self.neurons[self.winner1].feature_vector,self.neurons[j].feature_vector)
                summ_dist += dist
                count_of_neighbours += 1

        # Если у победителя нет соседей
        if count_of_neighbours == 0:
            return 0
        # Если не один
        else:
            aver_dist = summ_dist / count_of_neighbours
            aver_dist = round(aver_dist,4)
            return aver_dist

    def calc_aver_dist_in_subclass(self):

        # Пошло само вычисление
        summ_dist = 0
        count_of_neighbours = 0
        list_neurons_in_sb = []

        for i in range(self.neurons.shape[0]):
            if self.neurons[i].mark == self.neurons[self.winner1].mark:
                list_neurons_in_sb.append(i)

        for i in range(len(list_neurons_in_sb)):
            for j in range(i, len(list_neurons_in_sb)):
                if self.adjacency_matrix[i][j].rib_exist == 1:
                    summ_dist += self.calc_dist(self.neurons[i].feature_vector, self.neurons[j].feature_vector)
                    count_of_neighbours += 1

        # Если у победителя нет соседей
        if count_of_neighbours == 0:
            return 0
        # Если не один
        else:
            aver_dist = summ_dist / count_of_neighbours
            aver_dist = round(aver_dist, 4)
            return aver_dist



    # Вычисление плотности нейрона
    def calc_density(self):

        # self.neurons[self.winner1].density += 1/(((1+self.calc_aver_dist())**2))
        # Вот не понятно, какая модель лучше (ниже учитывается накопленный победителем сигнал)
        # self.neurons[self.winner1].density += 1/(self.neurons[self.winner1].accamulate_signals*(1+self.calc_aver_dist_in_subclass())**2)
        # self.neurons[self.winner1].density += 1/(self.neurons[self.winner1].accamulate_signals*(1+self.calc_aver_dist_in_subclass())**2)
        # self.neurons[self.winner1].density = round(self.neurons[self.winner1].density,3)

        self.neurons[self.winner1].s += 1 / ((1 + self.calc_aver_dist_in_subclass()) * (1 + self.calc_aver_dist_in_subclass()))
        self.neurons[self.winner1].density = self.neurons[self.winner1].s
        self.neurons[self.winner1].density = round(self.neurons[self.winner1].density, 3)

    # Смещение вектора признака нейронов победителей
    def update_feature_vector(self,inputVec):

        # Для первого победителя
        # Вычисление расстояния до входного вектора в векторном виде
        disp = [0, 0]
        disp[0] = inputVec[0] - self.neurons[self.winner1].feature_vector[0]
        disp[1] = inputVec[1] - self.neurons[self.winner1].feature_vector[1]

        # Применения смещения для первого победителя
        self.neurons[self.winner1].feature_vector[0] += \
            1/self.neurons[self.winner1].accamulate_signals * disp[0]
        self.neurons[self.winner1].feature_vector[1] += \
            1/self.neurons[self.winner1].accamulate_signals * disp[1]

        # Округление чисел до 4-ой точки после запятой
        self.neurons[self.winner1].feature_vector[0] = round(self.neurons[self.winner1].feature_vector[0],4)
        self.neurons[self.winner1].feature_vector[1] = round(self.neurons[self.winner1].feature_vector[1],4)

        # Вычисление и применение смещения для других нейронов соединенных с победителем, округление
        for j in range(self.adjacency_matrix.shape[1]):
            # Условие смежности с победителем
            if self.adjacency_matrix[self.winner1][j].rib_exist == 1 and j != self.winner1:

                disp = [0, 0]
                disp[0] = inputVec[0] - self.neurons[j].feature_vector[0]
                disp[1] = inputVec[1] - self.neurons[j].feature_vector[1]

                self.neurons[j].feature_vector[0] += 1/(100*self.neurons[self.winner1].accamulate_signals) * disp[0]
                self.neurons[j].feature_vector[1] += 1/(100*self.neurons[self.winner1].accamulate_signals) * disp[1]

                self.neurons[j].feature_vector[0] = round(self.neurons[j].feature_vector[0], 4)
                self.neurons[j].feature_vector[1] = round(self.neurons[j].feature_vector[1], 4)

    # Функция проверки превышения максимального возраста ребер и кратности "лямбда"
    def threshold_check(self):

        # Проверка превышения максимального возраста ребер, если превысило - удалить ребро
        for i in range(self.adjacency_matrix.shape[0]):
            for j in range(i+1, self.adjacency_matrix.shape[1]):
                if self.adjacency_matrix[i][j].rib_exist == 1:
                    if self.adjacency_matrix[i][j].rib_age > self.age_max:
                        self.adjacency_matrix[i][j].rib_exist = 0
                        self.adjacency_matrix[j][i].rib_exist = 0
                        self.adjacency_matrix[i][j].rib_age = 0
                        self.adjacency_matrix[j][i].rib_age = 0

    # Функция маркировки нейронов
    def mark_neurons(self):

        # Условия маркировки - выделить из всех узлов узлы с максимальными плотностями в локальной области (подклассы)
        # Маркировать эти узлы разными значениями
        # Остальные узлы маркировать значениями подкласса смежных нейронов, чья плотность будет наибольшей

        # Маркируем нейроны (новый алгоритм)

        list_of_apexes = []
        for i in range(self.neurons.shape[0]):
            self.neurons[i].mark = -1

        # Находим вершины подклассов
        for i in range(self.adjacency_matrix.shape[0]):

            max_dens_arround = self.neurons[i].density
            max_dens_index = i

            # Тут глупый быдло-код, по нахождении самих вершин
            for j in range(self.adjacency_matrix.shape[1]):
                if self.adjacency_matrix[i][j].rib_exist == 1:
                    if self.neurons[j].density > max_dens_arround and j!=i:
                        max_dens_index = j
                        break

            # Маркировка вершины
            if max_dens_index == i:
                self.neurons[i].mark = i

                # Добавление в список вершин
                if len(list_of_apexes) == 0:
                    list_of_apexes.append(i)
                else:
                    for j in range(len(list_of_apexes)):
                        if self.neurons[i].density > self.neurons[list_of_apexes[j]].density:
                            list_of_apexes.insert(j,i)
                            break
                    # Если за цикл ни разу не зашел в if
                    list_of_apexes.append(i)

        # if self.quantity_of_inputVec % 100 == 0:
        #     self.build_graph_type1(2)

        print('list of apexes',list_of_apexes)

        # Для каждой вершины маркировать соседние вершины, если их плотность меньше (такой же маркировкой, что и верщина)
        for i in list_of_apexes:
            self.mark_adjacent_neurons(i,self.neurons[i].mark)

        # if self.quantity_of_inputVec % 100 == 0:
        #     self.build_graph_type1(2)

        # Разделение разных подклассов или их объединение
        self.partition_class()

    # рекурсивный цикл по маркировке нейронов
    def mark_adjacent_neurons(self,indexB,mark):
        for j in range(self.adjacency_matrix.shape[1]):
            if self.adjacency_matrix[indexB][j].rib_exist == 1 and \
            self.neurons[j].density <= self.neurons[indexB].density and \
            self.neurons[j].mark == -1:
                self.neurons[j].mark = mark
                self.mark_adjacent_neurons(j,mark)

    def partition_class(self):
        for i in range(self.adjacency_matrix.shape[0]):
            for j in range(i+1, self.adjacency_matrix.shape[1]):
                if self.adjacency_matrix[i][j].rib_exist == 1:
                    if self.neurons[i].mark != self.neurons[j].mark and self.neurons[j].mark != -1 and \
                    self.neurons[i].mark != -1:
                        # Объединение подклассов
                        if self.merge_condition_algorithm1(i,j) == 1:
                            self.merge_subclasses(i,j)
                        # Разъединение узлов
                        else:
                            self.adjacency_matrix[i][j].rib_exist = 0
                            self.adjacency_matrix[j][i].rib_exist = 0

                            self.adjacency_matrix[i][j].rib_age = 0
                            self.adjacency_matrix[j][i].rib_age = 0

        # Разъединение узлов
        # for i in range(self.adjacency_matrix.shape[0]):
        #     for j in range(i+1, self.adjacency_matrix.shape[1]):
        #         if self.adjacency_matrix[i][j].rib_exist == 1:
        #             if self.neurons[i].mark != self.neurons[j].mark:
        #                 self.adjacency_matrix[i][j].rib_exist = 0
        #                 self.adjacency_matrix[j][i].rib_exist = 0
        #
        #                 self.adjacency_matrix[i][j].rib_age = 0
        #                 self.adjacency_matrix[j][i].rib_age = 0

        #  Удаление шумов
        self.remove_noise()

    # Объединение подклассов
    def merge_subclasses(self,A,B):
        # Слияние двух подклассов в один
        merge_mark = self.neurons[A].mark
        temp_mark = self.neurons[B].mark
        for i in range(self.neurons.shape[0]):
            if self.neurons[i].mark == temp_mark:
                self.neurons[i].mark = merge_mark


    # Удаление шумов
    def remove_noise(self):
        # Поиск нейронов - шумов
        flag = 0
        index_of_noise = self.search_noise()
        print(index_of_noise)
        index_of_noise = list(reversed(index_of_noise))
        if index_of_noise != None:
            for i in index_of_noise:
                flag = 1
                self.quantity_of_delete_neurons += 1

                self.adjacency_matrix = np.delete(self.adjacency_matrix, i, 0)
                self.adjacency_matrix = np.delete(self.adjacency_matrix, i, 1)

                self.neurons = np.delete(self.neurons, i, 0)
        # if flag == 1:
        #     self.remove_noise()

    # Поиск нейронов - шумов
    def search_noise(self):

        index_of_noise = []

        for i in range(self.neurons.shape[0]):
            self.neurons[i].degree = 0

        for i in range(self.adjacency_matrix.shape[0]):
            for j in range(i+1, self.adjacency_matrix.shape[1]):
                if self.adjacency_matrix[i][j].rib_exist == 1 and i!=j:
                    self.neurons[i].degree += 1
                    self.neurons[j].degree += 1

        for i in range(self.neurons.shape[0]):
            if self.neurons[i].degree == 2:
                if self.neurons[i].density < self.C1*self.summ_ratio_dens_to_count():
                    index_of_noise.append(i)

            if self.neurons[i].degree == 1:
                if self.neurons[i].density < self.C2*self.summ_ratio_dens_to_count():
                    index_of_noise.append(i)

            if self.neurons[i].degree == 0:
                index_of_noise.append(i)

        return index_of_noise

    # Сумма отношения плотности к количеству (Для поиска шумов)
    def summ_ratio_dens_to_count(self):

        summ = 0
        for i in range(self.neurons.shape[0]):
            summ += self.neurons[i].density
        mean = summ/self.neurons.shape[0]
        return mean


    def inputX(self, inputVec):


        self.winner1, self.winner2 = self.find_winners(inputVec)
        # if self.quantity_of_inputVec % 100 == 0:
        #     self.build_graph_type1(inputVec, 1)
        if self.point4(inputVec) == 1:
            # if self.quantity_of_inputVec % self.lambd == 0 and self.quantity_of_inputVec != 0:
            #     self.mark_neurons()
            return 0
        self.quantity_of_inputVec += 1
        self.update_ribs_age()
        self.rib_between_win1_win2()
        self.calc_density()
        self.update_feature_vector(inputVec)
        self.threshold_check()
        if self.quantity_of_inputVec % self.lambd == 0 and self.quantity_of_inputVec != 0:
            self.mark_neurons()
            print('1')
        if self.quantity_of_inputVec % 100 == 0:
            print('2')
            self.build_graph_type1(2)

    # Рекурсивный обход по компоненту связности
    def recur_matrix(self,i, neurons_in_one_component):
        if len(neurons_in_one_component) == 0:
            neurons_in_one_component.append(i)
        for j in range(self.neurons.shape[0]):
            if self.adjacency_matrix[i][j].rib_exist == 1:
                if j not in neurons_in_one_component:
                    neurons_in_one_component.append(j)
                    self.recur_matrix(j, neurons_in_one_component)
        return neurons_in_one_component

    # Нахождение компонентов связности
    def list_of_connected_components(self):
        neurons_in_components = []
        k = 0
        neurons_to_one_list = []
        temp_list_for_component = []
        flag = 1
        while flag == 1:
            flag = 0
            neurons_to_one_list.extend(ExampleStart.recur_matrix(k, temp_list_for_component))
            neurons_in_components.append(temp_list_for_component)
            temp_list_for_component = []
            for i in range(0, self.neurons.shape[0]):
                if i not in neurons_to_one_list:
                    flag = 1
                    k = i
                    break

        return neurons_in_components


    def build_graph_type1(self, param):
        plt.ion()
        plt.axis([0,1,0,1])
        # Потроение нейронов и вывод их маркировки
        for i in range(self.neurons.shape[0]):
            plt.plot(self.neurons[i].feature_vector[0],self.neurons[i].feature_vector[1], 'ro', color = 'red')
            plt.text(self.neurons[i].feature_vector[0], self.neurons[i].feature_vector[1] + 0.002,
                     self.neurons[i].mark, fontsize=8)
            plt.text(self.neurons[i].feature_vector[0], self.neurons[i].feature_vector[1] - 0.02,
                     self.neurons[i].density, fontsize=8)
            if param == 1:
                plt.plot(self.neurons[self.winner1].feature_vector[0], self.neurons[self.winner1].feature_vector[1],
                         'ro', color = 'green')
                plt.plot(self.neurons[self.winner2].feature_vector[0], self.neurons[self.winner2].feature_vector[1],
                         'ro', color = 'blue')

        for i in range(self.adjacency_matrix.shape[0]):
            for j in range(i+1, self.adjacency_matrix.shape[1]):
                if self.adjacency_matrix[i][j].rib_exist == 1:
                    plt.plot([self.neurons[i].feature_vector[0], self.neurons[j].feature_vector[0]],
                             [self.neurons[i].feature_vector[1], self.neurons[j].feature_vector[1]],
                             color='red', hold='false')

        plt.legend('6')
        plt.pause(0.001)
        plt.clf()





# Создание обхекта ESOINN
ExampleStart = ESOINN(50, 0.001, 1, 100)

# Списки координат для тестов
all_test = []
all_test_x = []
all_test_y = []

# Генерация чисел, надеюсь разберетсь (Гаусс1)
for i in range(10000):
    sample = []

    # noise
    if i % 10 == 0:
        sample.append(random.random())
    # correct data
    else:
        sample.append(random.gauss(0.7,0.15))

    sample[0] = round(sample[0],4)
    all_test_x.append(sample[0])

    if i % 10 == 0:
        sample.append(random.random())
    else:
        sample.append(random.gauss(0.7,0.15))

    all_test_y.append(sample[1])
    sample[1] = round(sample[1],4)
    all_test.append(sample)

    # Тут мы даем это ESOINN
    print('sample = ', i)
    ExampleStart.inputX(sample)

# (Гаусс2)
for i in range(10000):
    sample = []

    # noise
    if i % 10 == 0:
        sample.append(random.random())
    # correct data
    else :
        sample.append(random.gauss(0.3,0.15))

    sample[0] = round(sample[0],4)
    all_test_x.append(sample[0])

    if i % 10 == 0:
        sample.append(random.random())
    else:
        sample.append(random.gauss(0.3,0.15))

    all_test_y.append(sample[1])
    sample[1] = round(sample[1],4)
    all_test.append(sample)

    print('sample = ', i)
    ExampleStart.inputX(sample)


# Вывод конечной матрицы
for i in range(ExampleStart.adjacency_matrix.shape[0]):
    for j in range(ExampleStart.adjacency_matrix.shape[1]):
        print (ExampleStart.adjacency_matrix[i][j].rib_exist, end=' ')
    print()


# Вывод списка нейронов
for i in range(ExampleStart.neurons.shape[0]):
    print('neuron',i,' ',ExampleStart.neurons[i].feature_vector)

print()
print('sample    ',all_test)

# Координаты нейронов (Уже вывод из ESOINN)
x = []
y = []
for i in range(ExampleStart.neurons.shape[0]):
    x.append(ExampleStart.neurons[i].feature_vector[0])
    y.append(ExampleStart.neurons[i].feature_vector[1])

# Вывод количества удаленный нейронов
print('removes neurons = ',ExampleStart.quantity_of_delete_neurons)

# Вывод количества компонентов связности
neurons_in_comp = ExampleStart.list_of_connected_components()
for i in range(len(neurons_in_comp)):
    print('component = ',i,': ')
    print(neurons_in_comp[i])

plt.ioff()

# Построение нейронов
plt.plot([x], [y], 'ro',hold='false')
plt.plot([x], [y], 'ro',hold='false')

# Вывод их маркировки
for i in range(ExampleStart.neurons.shape[0]):
    plt.text(x[i],y[i]+0.002, ExampleStart.neurons[i].mark, fontsize=8)

for i in range(ExampleStart.neurons.shape[0]):
    plt.text(ExampleStart.neurons[i].feature_vector[0], ExampleStart.neurons[i].feature_vector[1] - 0.02,
             ExampleStart.neurons[i].density, fontsize=8)

# Отображение ребер
for i in range(ExampleStart.adjacency_matrix.shape[0]):
    for j in range(i+1, ExampleStart.adjacency_matrix.shape[1]):
        if ExampleStart.adjacency_matrix[i][j].rib_exist == 1:
            plt.plot([ExampleStart.neurons[i].feature_vector[0],ExampleStart.neurons[j].feature_vector[0]],
            [ExampleStart.neurons[i].feature_vector[1], ExampleStart.neurons[j].feature_vector[1]], color = 'red',hold='false')

for i in range(ExampleStart.neurons.shape[0]):
    if ExampleStart.neurons[i].density > 4:
        print(ExampleStart.neurons[i].accamulate_signals)

# Построение тестовой выборки
# plt.plot([all_test_x],[all_test_y],'bo')
plt.axis([0,1,0,1])
plt.legend('6')
plt.show()

