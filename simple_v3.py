# -*- coding: utf-8 -*-
from collections import defaultdict, Counter
import random
from functools import reduce
from random import randint, choice
import datetime
import pickle
from sklearn.metrics import f1_score


def generate_dataset(signal_length: int, inputs_count: int, repeat_blocks_number: int):
    """
    made by Vasyan
    Возвращает массивы входов вида [(x1,..., xn)] * signal_length
    и выходов [y] * signal_length.
    Массивы повторяются в случайном порядке repeat_blocks_number раз.
    Гарантируется, что при нулях на входе будет ноль на выходе;
    кроме того, число нулей в таблице истинности будет равно числу единиц в ответах.
    """
    
    # io_list = []
    # zero_count, ones_count = 0, 0
    # for el in range(1, 2 ** inputs_count):
    #     # получаем бинарное представление элемента
    #     el_binary = bin(el)[2:]
    #     # добавляем столько незначащих нулей слева,
    #     # чтобы длина была равна числу входных сигналов
    #     if len(el_binary) != inputs_count:
    #         el_binary = '0' * (inputs_count - len(el_binary)) + el_binary
    #     # переводим строки в числа и записываем в tuple
    #     el_binary = tuple(map(int, el_binary))
    #     # случайным образом назначаем правильный ответ
    #     io_list.append({'x' : el_binary, 'y' : randint(0, 1)})
    #     # принудительно ставим ответ в ноль, если на входах ноль
    #     io_list[-1]['y'] = 0 if sum(io_list[-1]['x']) == 0 else io_list[-1]['y']
    #     # подсчитываем число нулей и единиц
    #     zero_count = zero_count + 1 if io_list[-1]['y'] == 0 else zero_count
    #     ones_count = ones_count + 1 if io_list[-1]['y'] == 1 else ones_count
    #     # если перевалили за порог, меняем последнее значение на противоположное
    #     if zero_count > 2 ** (inputs_count - 1):
    #         io_list[-1]['y'] = 1
    #     elif ones_count > 2 ** (inputs_count - 1):
    #         io_list[-1]['y'] = 0
    io_list = [{'x': [0, 1], 'y': 1}, {'x': [1, 0], 'y': 1}, {'x': [1, 1], 'y': 0}, {'x': [1, 1], 'y': 0}]

    x_list, y_list = [], []
    for _ in range(repeat_blocks_number):
        rand_index = randint(0, len(io_list)-1)
        for uwu in range(signal_length):
            x_list.append(io_list[rand_index]['x'])
            y_list.append(io_list[rand_index]['y'])
            
    return x_list, y_list


BUFFER_LENGTH = 50


class Synapse:
    global_number = 0

    def __init__(self, input_n, output_n, weight=0.5, genome=None):
        self.number = Synapse.global_number
        Synapse.global_number += 1
        self.input_neuron = input_n
        self.output_neuron = output_n
        if genome:
            self.genome = genome
        else:
            self.genome = {'cd': {'input': {}, 'output': {}}, 'dw': {'input': {}, 'output': {}}}
        self.is_signal = False
        self.weight = weight  # вес
        self.input_vars = {}
        self.output_vars = {}
        self.is_real = False
        self.cd = 0.5
        self.dw = 0
        self.history = [0 for _ in range(BUFFER_LENGTH)]

    # проверка есть ли сигнал в синапсе и сразу зануляем
    def check_signal(self):
        if self.is_signal and self.is_real:
            self.is_signal = False
            return True
        else:
            return False

    # метод активации синапса
    def activate(self):
        if self.is_real:
            self.is_signal = True

    def get_division(self, x, y):
        try:
            return x / y
        except ZeroDivisionError:
            return 0

    def preprocess_vars(self):
        first_history = self.input_vars[1]
        second_history = self.output_vars[1]
        counter = 0
        for i, j in zip(first_history, second_history):
            if (i == 1) and (i == j):
                counter += 1
        self.input_vars[1] = counter / 100
        self.output_vars[1] = counter / 100

        self.input_vars[45] = self.get_division(self.input_vars[0], self.input_vars[1]) / 100
        self.input_vars[46] = self.get_division(self.input_vars[0], self.input_vars[14]) / 100
        self.input_vars[47] = self.get_division(self.input_vars[0], self.input_vars[15]) / 100

        self.output_vars[45] = self.get_division(self.output_vars[0], self.output_vars[1]) / 100
        self.output_vars[46] = self.get_division(self.output_vars[0], self.output_vars[14]) / 100
        self.output_vars[47] = self.get_division(self.output_vars[0], self.output_vars[15]) / 100

        self.input_vars[48] = self.output_vars[48] = self.history.count(1) / 100
        self.input_vars[49] = self.output_vars[49] = self.weight / 100
        for i in range(18):
            self.input_vars[50 + i] = self.get_division(self.input_vars[48], self.input_vars[i + 1]) / 100
            self.output_vars[50 + i] = self.get_division(self.output_vars[48], self.output_vars[i + 1]) / 100
        for i in range(18):
            self.input_vars[68 + i] = self.get_division(self.input_vars[49], self.input_vars[i + 1]) / 100
            self.output_vars[68 + i] = self.get_division(self.output_vars[49], self.output_vars[i + 1]) / 100

    def get_vars(self):
        input_vars = self.input_neuron.get_params()
        self.input_vars = input_vars.copy()
        output_vars = self.output_neuron.get_params()
        self.output_vars = output_vars.copy()
        self.preprocess_vars()

    def calculate_cddw(self):
        self.cd = 0
        self.dw = 0
        '''
        genome:
        cd, input - [0, 1, 2, ..., 85], output - [0, 1, 2, ..., 85]
        dw, input - [0, 1, 2, ..., 85], output - [0, 1, 2, ..., 85]
        '''
        for key, value in self.genome['cd']['input'].items():
            # if key in list(self.input_vars.keys()):
            self.cd += self.input_vars[key] * value
        for key, value in self.genome['cd']['output'].items():
            # if key in list(self.output_vars.keys()):
            self.cd += self.output_vars[key] * value
        for key, value in self.genome['dw']['input'].items():
            # if key in list(self.input_vars.keys()):
            self.dw += self.input_vars[key] * value
        for key, value in self.genome['dw']['output'].items():
            # if key in list(self.input_vars.keys()):
            self.dw += self.output_vars[key] * value

        self.cd = self.cd if self.cd <= 1 else 1
        self.cd = self.cd if self.cd >= 0 else 0
        self.dw = self.dw if self.dw <= 1 else 1
        self.dw = self.dw if self.dw >= -1 else -1

    def check_existing(self):
        if (not self.is_real) and self.cd >= 0.8:
            self.is_real = True
        elif self.is_real and self.cd <= 0.2:
            self.is_real = False
        return self.is_real

    def move_weight(self):
        if not self.is_real:
            return
        self.weight += self.dw * 0.05

    def add_signals(self):
        if self.check_signal():
            self.history.append(1)
            self.output_neuron.add_accumulator(self.weight, self.number)
        else:
            self.history.append(0)
        self.history.pop(0)

    def check_input_signal(self):
        if not self.is_real:
            return
        if self.input_neuron.is_spiked():
            self.is_signal = True


class Neuron:
    """
    Какие параметры нейрона могут быть? Включаем фантазию:
    -0 расстояние до выхода
    -1 показатель как часто одновременно активируется этот нейрон и другой (можно передать историю активаций)
    -2 количество входных синапсов
    -3 количество выходных синапсов
    -4 количество отрицательных входных синапсов
    -5 количество положительных входных синапсов
    -6 количество отрицательных выходных синапсов
    -7 количество положительных выходных синапсов
    -8 сумма весов входных синапсов
    -9 сумма весов выходных синапсов
    -10 сумма весов входных положительных синапсов
    -11 сумма весов входных отрицательных синапсов
    -12 сумма весов выходных положительных синапсов
    -13 сумма весов выходных отрицательных синапсов
    -14 количество спайков за последнее время
    -15 количество входных импульсов за последнее время
    -16 количество положительных входных импульсов за последнее время
    -17 количество отрицательных входных импульсов за последнее время
    -18 количество активаций самого активного входного синапса за последнее время
    -19 отношение 2 и 3
    -20 отношение 4 и 5
    -21 отношение 6 и 7
    -22 отношение 4 и 6
    -23 отношение 5 и 7
    -24 отношение 4 и 7
    -25 отношение 5 и 6
    -26 отношение 8 и 9
    -27 отношение 4 и 8
    -28 отношение 5 и 8
    -29 отношение 6 и 9
    -30 отношение 7 и 9
    -31 отношение 10 и 11
    -32 отношение 12 и 13
    -33 отношение 10 и 13
    -34 отношение 11 и 12
    -35 отношение 14 и 2
    -36 отношение 14 и 3
    -37 отношение 14 и 8
    -38 отношение 14 и 9
    -39 отношение 14 и 15
    -40 отношение 16 и 17
    -41 отношение 16 и 14
    -42 отношение 17 и 14
    -43 отношение 18 и 14
    -44 отношение 18 и 15
    -45 отношение 0 и 1  - в синапсе
    -46 отношение 0 и 14  - в синапсе
    -47 отношение 0 и 15  - в синапсе
    -86 расстояние до входа * на реакцию сети
    -87 чистая реакция сети
    -88..104 отношения 87 и (2-18)
    В самом синапсе:
    -48(а) количество активаций за последнее время (последняя частота)
    -49(б) вес синапса
    -50..67 отношение а и (1-18)
    -68..85 отношение б и (1-18)
    """
    global_number = 0

    def __init__(self, is_input=False, is_output=False):
        self.number = Neuron.global_number
        Neuron.global_number += 1
        self.accumulator = 0
        self.p = defaultdict(int)
        self.coordinate = random.random()
        if is_input and (not is_output):
            self.coordinate = 1
        elif (not is_input) and is_output:
            self.coordinate = 0
        self.spike_history = [0 for _ in range(BUFFER_LENGTH)]
        self.synapse_history = [[] for _ in range(BUFFER_LENGTH)]
        self.input_synapses = []
        self.output_synapses = []

    def get_number_of_synapses(self, array, sign=None):
        counter = 0
        if not sign:
            for synapse in array:
                if synapse.is_real:
                    counter += 1
            return counter
        else:
            if sign == 'pos':
                for synapse in array:
                    if synapse.is_real and synapse.weight >= 0:
                        counter += 1
                return counter
            elif sign == 'neg':
                for synapse in array:
                    if synapse.is_real and synapse.weight < 0:
                        counter += 1
                return counter

    def get_sum_of_weights(self, array, sign=None):
        counter = 0
        if not sign:
            for synapse in array:
                if synapse.is_real:
                    counter += synapse.weight
            return counter
        else:
            if sign == 'pos':
                for synapse in array:
                    if synapse.is_real and synapse.weight >= 0:
                        counter += synapse.weight
                return counter
            elif sign == 'neg':
                for synapse in array:
                    if synapse.is_real and synapse.weight < 0:
                        counter += synapse.weight
                return counter

    def signed_synapses(self, sign='pos'):
        res = []
        if sign == 'pos':
            for s in self.input_synapses:
                if s.is_real and (s.weight >= 0):
                    res.append(s.number)
            return res
        elif sign == 'neg':
            for s in self.input_synapses:
                if s.is_real and (s.weight < 0):
                    res.append(s.number)
            return res

    def get_sign_sum_of_input_impulses(self, sign='pos'):
        counter = 0
        if sign == 'pos':
            positive_synapse_numbers = self.signed_synapses(sign='pos')
            for item in self.synapse_history:
                for synapse in item:
                    if synapse in positive_synapse_numbers:
                        counter += 1
            return counter
        elif sign == 'neg':
            negative_synapse_numbers = self.signed_synapses(sign='neg')
            for item in self.synapse_history:
                for synapse in item:
                    if synapse in negative_synapse_numbers:
                        counter += 1
            return counter

    def get_division(self, x, y):
        try:
            return self.p[x] / self.p[y]
        except ZeroDivisionError:
            return 0

    def get_more_activated(self):
        all_activations = reduce(lambda x, y: x.extend(y) or x, self.synapse_history, [])
        try:
            return Counter(all_activations).most_common(1)[0][1]
        except:
            return 0

    def get_params(self):
        return self.p

    def extract_params(self, is_right=0):
        self.p[0] = self.coordinate
        self.p[1] = self.spike_history
        self.p[2] = self.get_number_of_synapses(self.input_synapses) / 100
        self.p[3] = self.get_number_of_synapses(self.output_synapses) / 100
        self.p[4] = self.get_number_of_synapses(self.input_synapses, sign='neg') / 100
        self.p[5] = self.get_number_of_synapses(self.input_synapses, sign='pos') / 100
        self.p[6] = self.get_number_of_synapses(self.output_synapses, sign='neg') / 100
        self.p[7] = self.get_number_of_synapses(self.output_synapses, sign='pos') / 100
        self.p[8] = self.get_sum_of_weights(self.input_synapses) / 100
        self.p[9] = self.get_sum_of_weights(self.output_synapses) / 100
        self.p[10] = self.get_sum_of_weights(self.input_synapses, sign='pos') / 100
        self.p[11] = self.get_sum_of_weights(self.output_synapses, sign='neg') / 100
        self.p[12] = self.get_sum_of_weights(self.input_synapses, sign='pos') / 100
        self.p[13] = self.get_sum_of_weights(self.output_synapses, sign='neg') / 100
        self.p[14] = self.spike_history.count(1) / 100
        self.p[15] = reduce(lambda x, y: x + len(y), self.synapse_history, 0) / 100
        self.p[16] = self.get_sign_sum_of_input_impulses(sign='pos') / 100
        self.p[17] = self.get_sign_sum_of_input_impulses(sign='neg') / 100
        self.p[18] = self.get_more_activated() / 100
        self.p[19] = self.get_division(2, 3) / 100
        self.p[20] = self.get_division(4, 5) / 100
        self.p[21] = self.get_division(6, 7) / 100
        self.p[22] = self.get_division(4, 6) / 100
        self.p[23] = self.get_division(5, 7) / 100
        self.p[24] = self.get_division(4, 7) / 100
        self.p[25] = self.get_division(5, 6) / 100
        self.p[26] = self.get_division(8, 9) / 100
        self.p[27] = self.get_division(4, 8) / 100
        self.p[28] = self.get_division(5, 8) / 100
        self.p[29] = self.get_division(6, 9) / 100
        self.p[30] = self.get_division(7, 9) / 100
        self.p[31] = self.get_division(10, 11) / 100
        self.p[32] = self.get_division(12, 13) / 100
        self.p[33] = self.get_division(10, 13) / 100
        self.p[34] = self.get_division(11, 12) / 100
        self.p[35] = self.get_division(14, 2) / 100
        self.p[36] = self.get_division(14, 3) / 100
        self.p[37] = self.get_division(14, 8) / 100
        self.p[38] = self.get_division(14, 9) / 100
        self.p[39] = self.get_division(14, 15) / 100
        self.p[40] = self.get_division(16, 17) / 100
        self.p[41] = self.get_division(16, 14) / 100
        self.p[42] = self.get_division(17, 14) / 100
        self.p[43] = self.get_division(18, 14) / 100
        self.p[44] = self.get_division(18, 15) / 100

        self.p[86] = is_right * self.coordinate / 100
        self.p[87] = is_right
        self.p[88] = self.get_division(2, 87) / 100
        self.p[89] = self.get_division(3, 87) / 100
        self.p[90] = self.get_division(4, 87) / 100
        self.p[91] = self.get_division(5, 87) / 100
        self.p[92] = self.get_division(6, 87) / 100
        self.p[93] = self.get_division(7, 87) / 100
        self.p[94] = self.get_division(8, 87) / 100
        self.p[95] = self.get_division(9, 87) / 100
        self.p[96] = self.get_division(10, 87) / 100
        self.p[97] = self.get_division(11, 87) / 100
        self.p[98] = self.get_division(12, 87) / 100
        self.p[99] = self.get_division(13, 87) / 100
        self.p[100] = self.get_division(14, 87) / 100
        self.p[101] = self.get_division(15, 87) / 100
        self.p[102] = self.get_division(16, 87) / 100
        self.p[103] = self.get_division(17, 87) / 100
        self.p[104] = self.get_division(18, 87) / 100

        # self.p[45] = self.get_division(0, 1)
        # self.p[46] = self.get_division(0, 14)
        # self.p[47] = self.get_division(0, 15)
        # return self.p

    def add_accumulator(self, value, synapse_number):
        self.accumulator += value
        self.synapse_history[-1].append(synapse_number)

    def is_spiked(self):
        is_spiked = 0
        if self.accumulator >= 1:
            is_spiked = 1
        self.spike_history[-1] = is_spiked
        return bool(is_spiked)

    def erase(self):
        self.accumulator = 0
        self.spike_history.append(0)
        self.spike_history.pop(0)
        self.synapse_history.append([])
        self.synapse_history.pop(0)


class Net:
    def __init__(self, n_neurons, genome=None, input_numbers=None, output_number=None):
        self.neurons = []
        self.synapses = []
        if input_numbers is None:
            input_numbers = [0, 1]
        if output_number is None:
            output_number = 4
        self.output_number = output_number
        for i in range(n_neurons):
            if i in input_numbers:
                self.neurons.append(Neuron(is_input=True))
            elif i == output_number:
                self.neurons.append(Neuron(is_output=True))
            else:
                self.neurons.append(Neuron())
        for outer_neuron in self.neurons:
            for inner_neuron in self.neurons:
                if outer_neuron.number == inner_neuron.number:
                    continue
                synapse = Synapse(outer_neuron, inner_neuron, genome=genome)
                self.synapses.append(synapse)
                outer_neuron.output_synapses.append(synapse)
                inner_neuron.input_synapses.append(synapse)

    # метод поиска нужного нейрона, возвращает экземпляр найденнного нейрона
    def get_neuron(self, num):
        for n in self.neurons:
            if n.number == num:
                return n
        else:
            raise

    # метод ручного ввода импульса в нейрон
    def probe(self, number):
        n = self.get_neuron(number)
        n.accumulator = 1

    # метод ввода импульсов в несколько нейронов на одном тике
    # индекс числа во входном массиве - номер активируемого нейрона
    def massive_probe(self, array):
        for i in range(len(array)):
            if array[i] != 0:
                self.probe(i)

    def tick(self, reaction):
        out_signal = False
        for s in self.synapses:
            s.check_input_signal()
        for n in self.neurons:
            if n.number == self.output_number and n.is_spiked():
                out_signal = True
                # print('hop')
            n.erase()
        for s in self.synapses:
            s.add_signals()
        for n in self.neurons:
            n.extract_params(is_right=reaction)
        for s in self.synapses:
            s.get_vars()
            s.calculate_cddw()
            s.check_existing()
            s.move_weight()
        return int(out_signal)

    def predict(self, X, y):
        result = []
        for i in range(len(X)):
            cur_x = X[i]
            # cur_y = y[i]
        # for cur_x, cur_y in zip(X, y):  # идем по входным данным
            self.massive_probe(cur_x)  # тут все аналогично как в обучении, только без вызова обучения нейронов
            if i == 0:
                res = self.tick(0)
            else:
                if result[-1] == y[i - 1]:
                    res = self.tick(1)
                else:
                    res = self.tick(0)
            result.append(res)
        return result


class Population:
    def __init__(self, from_file=False, population_size=100, neurons_number=10):
        if from_file:
            # with open('population.pickle', 'rb'):
            #     self.current_population = pickle.load(file)
            self.new_genomes = []
            self.neurons_number = neurons_number
            self.input_numbers = [1, 2]
        else:
            self.current_population = []
            self.new_genomes = []
            self.neurons_number = neurons_number
            self.input_numbers = [1, 2]
            for i in range(population_size):
                test_genome = {
                    'cd': {
                        'input': {x: 2 * random.random() - 1 for x in range(105)},
                        'output': {x: 2 * random.random() - 1 for x in range(105)}
                    },
                    'dw': {
                        'input': {x: 2 * random.random() - 1 for x in range(105)},
                        'output': {x: 2 * random.random() - 1 for x in range(105)}
                    }
                }
                net = Net(neurons_number, genome=test_genome, input_numbers=self.input_numbers)
                self.current_population.append({'net': net, 'genome': test_genome, 'f_value': 0.0, 'metrics': ''})
                Synapse.global_number = 0
                Neuron.global_number = 0

    def create_dataset(self):
        X, y = generate_dataset(16, 2, 80)
        return X, y

    def mutation(self, best_nets):
        for item in best_nets:
            for i in ['cd', 'dw']:
                for j in ['input', 'output']:
                    for k in range(len(item['cd']['input'].keys())):
                        if random.random() < 0.1:
                            item[i][j][k] += (2 * random.random() - 1) * 0.1
        self.new_genomes += best_nets

    def crossingover(self, best_nets):
        new_genomes = []
        for _ in range(40):
            a = random.randint(0, 19)
            b = random.randint(0, 19)
            while a == b:
                b = random.randint(0, 19)
            parent_a = best_nets[a]
            parent_b = best_nets[b]
            child = {'cd': {'input': {}, 'output': {}},
                     'dw': {'input': {}, 'output': {}}}
            for i in ['cd', 'dw']:
                for j in ['input', 'output']:
                    for k in range(len(parent_a['cd']['input'].keys())):
                        if random.random() > 0.5:
                            child[i][j][k] = parent_a[i][j][k]
                        else:
                            child[i][j][k] = parent_b[i][j][k]
            new_genomes.append(child)
        self.new_genomes += new_genomes

    def random_objects(self, all_nets):
        # for _ in range(40):
        #     test_genome = {
        #         'cd': {
        #             'input': {x: 2 * random.random() - 1 for x in range(105)},
        #             'output': {x: 2 * random.random() - 1 for x in range(105)}
        #         },
        #         'dw': {
        #             'input': {x: 2 * random.random() - 1 for x in range(105)},
        #             'output': {x: 2 * random.random() - 1 for x in range(105)}
        #         }
        #     }
        #     self.new_genomes.append(test_genome)
        new_genomes = []
        for _ in range(40):
            a = random.randint(0, 99)
            b = random.randint(0, 99)
            while a == b:
                b = random.randint(0, 99)
            parent_a = all_nets[a]
            parent_b = all_nets[b]
            child = {'cd': {'input': {}, 'output': {}},
                     'dw': {'input': {}, 'output': {}}}
            for i in ['cd', 'dw']:
                for j in ['input', 'output']:
                    for k in range(len(parent_a['cd']['input'].keys())):
                        if random.random() > 0.5:
                            child[i][j][k] = parent_a[i][j][k]
                        else:
                            child[i][j][k] = parent_b[i][j][k]
            new_genomes.append(child)
        self.new_genomes += new_genomes

    def create_nets_from_genomes(self):
        print(len(self.new_genomes))
        for genome in self.new_genomes:
            net = Net(self.neurons_number, genome=genome, input_numbers=self.input_numbers)
            self.current_population.append({'net': net, 'genome': genome, 'f_value': 0.0, 'metrics': ''})
            Synapse.global_number = 0
            Neuron.global_number = 0
        self.new_genomes = []

    def new_population(self):
        self.current_population.sort(key=lambda x: x['f_value'], reverse=True)
        accuracy = self.current_population[0]["f_value"]
        print(f'Metrics on current population: {self.current_population[0]["metrics"]}, {self.current_population[19]["metrics"]}')
        best_nets = [x['genome'] for x in self.current_population[:20]]
        all_nets = [x['genome'] for x in self.current_population]
        self.current_population = []
        self.crossingover(best_nets)
        self.mutation(best_nets)
        self.random_objects(all_nets)
        self.create_nets_from_genomes()
        # сохраняем текущее состояние сети в файл
        with open('population.pickle', 'wb') as file:
            pickle.dump(self.current_population, file)
        return accuracy

    def guessed_number(self, a, b):
        counter_zeros = 0
        counter_ones = 0
        for i, j in zip(a, b):
            if i == j == 0:
                counter_zeros += 1
            elif i == j == 1:
                counter_ones += 1
        return counter_zeros, counter_ones

    def fit(self, accuracy=0.9):
        count = 0
        current_accuracy = 0.0
        train_X, train_y = self.create_dataset()
        # for i, j in zip(train_X, train_y):
        #     print(i, j)
        print(train_y.count(1))
        while current_accuracy < accuracy:
            count += 1
            print(f'Number of generation: {count}')
            t_start = datetime.datetime.now()
            for v, net_dict in enumerate(self.current_population):
                net = net_dict['net']
                predictions = net.predict(train_X, train_y)
                counter_good = 0
                test_number = 16 * 20
                counter_all = len(train_y[test_number:])
                for j, k in zip(predictions[test_number:], train_y[test_number:]):
                    if j == k:
                        counter_good += 1
                fitness_value = counter_good / counter_all * 0.5 + (
                        1 - abs(predictions[test_number:].count(0) - predictions[test_number:].count(1)) / counter_all) * 0.5
                f1_value = f1_score(train_y[test_number:], predictions[test_number:])
                net_dict['f_value'] = f1_value
                counter_zeros, counter_ones = self.guessed_number(predictions[test_number:], train_y[test_number:])
#                 f1_value = f1_score(train_y[test_number:], predictions[test_number:])
                net_dict['metrics'] = str(f'Net number {v}. f: {f1_value}, c: {counter_good}/{counter_all}, guessed zeros {counter_zeros}/{train_y[test_number:].count(0)}, guessed ones {counter_ones}/{train_y[test_number:].count(1)}, f1 {f1_value}')
                if (v % 1) == 0:
                    print(f'Net number {v}. f: {fitness_value}, c: {counter_good}/{counter_all}, guessed zeros {counter_zeros}/{train_y[test_number:].count(0)}, guessed ones {counter_ones}/{train_y[test_number:].count(1)}, f1 {f1_value}')
            t_end = datetime.datetime.now()
            print(t_end - t_start)
            current_accuracy = self.new_population()

BUFFER_LENGTH = 60

# old_train_X, train_y = generate_dataset(2, 2)
# train_X = []
# for item in old_train_X:
#     train_X.append([1] + list(item))
# train_X = train_X * 20
# train_y = train_y * 20
#
#
# all_nets = []
# all_genomes = []
# results = []
# for i in range(10):
#     test_genome = {
#         'cd': {
#             'input': {x: 2 * random.random() - 1 for x in range(105)},
#             'output': {x: 2 * random.random() - 1 for x in range(105)}
#         },
#         'dw': {
#             'input': {x: 2 * random.random() - 1 for x in range(105)},
#             'output': {x: 2 * random.random() - 1 for x in range(105)}
#         }
#     }
#     net = Net(10, genome=test_genome, input_numbers=[0, 1, 2])
#     all_nets.append(net)
#     all_genomes.append(test_genome)
#     Synapse.global_number = 0
#     Neuron.global_number = 0
#
# for net in all_nets:
#     predictions = net.predict(train_X, train_y)
#     counter_good = 0
#     counter_all = len(train_y)
#     # print(predictions)
#     # print(train_y)
#     for j, k in zip(predictions, train_y):
#         if j == k:
#             counter_good += 1
#     fitness_value = counter_good / counter_all * 0.5 + (
#                 1 - abs(predictions.count(0) - predictions.count(1)) / counter_all) * 0.5
#     results.append(fitness_value)
#     print(f'f: {fitness_value}, c: {counter_good}/{counter_all}')

p = Population(population_size=100)
p.fit(0.95)
