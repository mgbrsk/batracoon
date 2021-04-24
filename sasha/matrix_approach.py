# -*- coding: utf-8 -*-
import numpy as np
import datetime
import random
from random import randint
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import balanced_accuracy_score


def generate_dataset(signal_length: int, inputs_count: int, repeat_blocks_number: int, custom_table=None):
    """
    made by Vasyan
    Возвращает массивы входов вида [(x1,..., xn)] * signal_length
    и выходов [y] * signal_length.
    Массивы повторяются в случайном порядке repeat_blocks_number раз.
    Гарантируется, что при нулях на входе будет ноль на выходе;
    кроме того, число нулей в таблице истинности будет равно числу единиц в ответах.
    """
    if not custom_table:
        io_list = []
        zero_count, ones_count = 0, 0
        for el in range(1, 2 ** inputs_count):
            # получаем бинарное представление элемента
            el_binary = bin(el)[2:]
            # добавляем столько незначащих нулей слева,
            # чтобы длина была равна числу входных сигналов
            if len(el_binary) != inputs_count:
                el_binary = '0' * (inputs_count - len(el_binary)) + el_binary
            # переводим строки в числа и записываем в tuple
            el_binary = tuple(map(int, el_binary))
            # случайным образом назначаем правильный ответ
            io_list.append({'x': el_binary, 'y': randint(0, 1)})
            # принудительно ставим ответ в ноль, если на входах ноль
            io_list[-1]['y'] = 0 if sum(io_list[-1]['x']) == 0 else io_list[-1]['y']
            # подсчитываем число нулей и единиц
            zero_count = zero_count + 1 if io_list[-1]['y'] == 0 else zero_count
            ones_count = ones_count + 1 if io_list[-1]['y'] == 1 else ones_count
            # если перевалили за порог, меняем последнее значение на противоположное
            if zero_count > 2 ** (inputs_count - 1):
                io_list[-1]['y'] = 1
            elif ones_count > 2 ** (inputs_count - 1):
                io_list[-1]['y'] = 0
    else:
        io_list = custom_table

    x_list, y_list = [], []
    for _ in range(repeat_blocks_number):
        rand_index = randint(0, len(io_list) - 1)
        for uwu in range(signal_length):
            x_list.append(io_list[rand_index]['x'])
            y_list.append(io_list[rand_index]['y'])

    return x_list, y_list


class Net:
    def __init__(self, neurons_number=5, input_neurons=None, output_neuron=4, genome=None, parameters_number=11):
        self.count_cutoff = 0.0
        self.neurons_number = neurons_number
        self.output_neuron = output_neuron
        if input_neurons is not None:
            self.input_neurons = input_neurons
        else:
            self.input_neurons = [0, 1]
        if genome is not None:
            self.genome = genome
            self.parameters_number = len(genome['cd']['input'])
            self.s_cd_input = np.array(list(genome['cd']['input'].values()))
            self.s_cd_output = np.array(list(genome['cd']['output'].values()))
            self.s_dw_input = np.array(list(genome['dw']['input'].values()))
            self.s_dw_output = np.array(list(genome['dw']['output'].values()))
        else:
            self.parameters_number = parameters_number
            self.s_cd_input = np.random.rand(self.parameters_number) * 2 - 1
            self.s_cd_output = np.random.rand(self.parameters_number) * 2 - 1
            self.s_dw_input = np.random.rand(self.parameters_number) * 2 - 1
            self.s_dw_output = np.random.rand(self.parameters_number) * 2 - 1
        self.n_accumulator = np.zeros((self.neurons_number,))
        self.n_activation_history = np.zeros((BUFFER_SIZE, self.neurons_number))
        self.n_signals = np.zeros((self.neurons_number,))
        self.s_weights = np.zeros((self.neurons_number, self.neurons_number))
        self.s_signals = np.zeros((self.neurons_number, self.neurons_number))
        self.s_real = np.zeros((self.neurons_number, self.neurons_number))
        self.s_cd = np.zeros((self.neurons_number, self.neurons_number))
        self.s_dw = np.zeros((self.neurons_number, self.neurons_number))
        # история активаций (mock),
        # расстояние до входа,
        # правильность*расстояние
        # количество синапсов,
        # количество входных,
        # количество выходных,
        # сумма весов,
        # сумма положительных вх весов,
        # сумма отрицательных вх весов,
        # сумма пол вых весов,
        # сумма отр вых весов
        self.n_parameters = np.zeros((self.neurons_number, self.parameters_number))  # * 2 - 1
        for n in self.input_neurons:
            self.n_parameters[n, 1] = 0.0
        self.n_parameters[self.output_neuron, 1] = 1.0

    def tick(self, reaction=0, learning=True):
        out_signal = 0
        self.n_signals = np.where(self.n_accumulator >= 1.0, 1.0, self.n_signals)
        temp = self.s_weights * self.s_real * self.n_signals
        self.n_accumulator = np.sum(temp, axis=1)  # n_accumulator + temp_sum
        self.random_activation(0.05)
        if self.n_signals[self.output_neuron] > 0.0:
            out_signal = 1.0
        self.update_parameters(reaction)  # reaction
        if learning and self.count_cutoff < 400:
            self.compute_cd()  # reaction
            self.compute_dw()  #
            self.compute_interaction()
            self.check_real()
            self.move_weights()
            self.count_cutoff += 1
        self.n_signals = np.zeros((self.neurons_number,))
        return out_signal

    def random_activation(self, probability):
        r_a_matrix = np.random.rand(self.neurons_number)
        synapse_numbers = np.sum(self.s_real, axis=0) + np.sum(self.s_real, axis=1)
        synapse_numbers = synapse_numbers + 1.0
        r_a_matrix = r_a_matrix / synapse_numbers
        self.n_signals = np.where(r_a_matrix < probability, 1.0, self.n_signals)

    def compute_interaction(self):
        interaction_matrix = np.dot(self.n_activation_history.T, self.n_activation_history)
        np.fill_diagonal(interaction_matrix, 0.0)
        cd_input = interaction_matrix * self.s_cd_input[0]
        cd_output = interaction_matrix * self.s_cd_output[0]
        dw_input = interaction_matrix * self.s_dw_input[0]
        dw_output = interaction_matrix * self.s_dw_output[0]
        cd = cd_input + cd_output
        dw = dw_input + dw_output
        self.s_cd = self.s_cd + cd
        self.s_dw = self.s_dw + dw
        np.fill_diagonal(self.s_cd, 0.0)
        np.fill_diagonal(self.s_dw, 0.0)

    def update_parameters(self, reaction):
        self.n_activation_history = np.roll(self.n_activation_history, shift=1, axis=0)
        self.n_activation_history[0] = self.n_signals
        self.n_parameters[:, 2] = self.n_parameters[:, 1] * reaction
        self.n_parameters[:, 3] = np.sum(self.s_real, axis=0) + np.sum(self.s_real, axis=1)
        self.n_parameters[:, 4] = np.sum(self.s_real, axis=1)
        self.n_parameters[:, 5] = np.sum(self.s_real, axis=0)
        self.n_parameters[:, 6] = np.sum(self.s_weights, axis=0) + np.sum(self.s_weights, axis=1)
        pos_weights = np.where(self.s_weights > 0.0, self.s_weights, 0.0)
        neg_weights = np.where(self.s_weights < 0.0, self.s_weights, 0.0)
        self.n_parameters[:, 7] = np.sum(pos_weights, axis=1)
        self.n_parameters[:, 8] = np.sum(neg_weights, axis=1)
        self.n_parameters[:, 9] = np.sum(pos_weights, axis=0)
        self.n_parameters[:, 10] = np.sum(neg_weights, axis=0)

    def compute_cd(self):
        cd_input = np.sum(self.s_cd_input[1:] * self.n_parameters[:, 1:], axis=1)
        cd_output = np.sum(self.s_cd_output[1:] * self.n_parameters[:, 1:], axis=1)
        self.s_cd = cd_input + cd_output.reshape((self.neurons_number, 1))
        np.fill_diagonal(self.s_cd, 0.0)

    def compute_dw(self):
        dw_input = np.sum(self.s_dw_input[1:] * self.n_parameters[:, 1:], axis=1)
        dw_output = np.sum(self.s_dw_output[1:] * self.n_parameters[:, 1:], axis=1)
        self.s_dw = dw_input + dw_output.reshape((self.neurons_number, 1))
        np.fill_diagonal(self.s_dw, 0.0)

    def check_real(self):
        # получение новых значений реальности s_real
        temp_cd_greater = np.where(self.s_cd > 0.8, 1.0, 0.0)
        temp_cd_less = np.where(self.s_cd < 0.2, 0.0, 1.0)
        s_real_inverse = self.s_real * (-1) + 1.0
        self.s_real = self.s_real + s_real_inverse * temp_cd_greater
        self.s_real = self.s_real * temp_cd_less

    def move_weights(self):
        # получение новых весов s_weights
        self.s_weights = self.s_weights + self.s_dw * 0.2
        # обнуляем веса удаленных синапсов
        self.s_weights = self.s_weights * self.s_real

    def _make_synapse_real(self, n1, n2, weight=0.5):
        self.s_real[n2, n1] = 1.0
        self.s_weights[n2, n1] = weight

    def probe(self, number):
        self.n_accumulator[number] = 1.0

    def massive_probe(self, array):
        for i in range(len(array)):
            if array[i] == 1:
                self.probe(i)

    def predict(self, x, y, learning=True):
        if len(x) != len(y):
            raise
        result = []
        for cur_x, cur_y in zip(x, y):
            self.massive_probe(cur_x)
            out = self.tick(learning=learning, reaction=cur_y)
            result.append(out)
        return result


def test():
    net = Net(neurons_number=5)
    net._make_synapse_real(0, 2, weight=1)
    net._make_synapse_real(1, 3, weight=1)
    net._make_synapse_real(2, 4, weight=0.5)
    net._make_synapse_real(3, 4, weight=0.5)

    x = [[0, 0],
         [1, 1],
         [0, 0],
         [0, 0],
         [0, 0],
         [0, 0],
         [0, 0],
         [0, 0]]

    y = [0, 0, 0, 0, 0, 0, 0, 0]
    res = net.predict(x, y, learning=False)
    assert res == [0, 0, 0, 1, 0, 0, 0, 0]


BUFFER_SIZE = 10
# test()


class Population:
    """
    Класс популяции эволюционного алгоритма.
    В начале создается набор рандомных генов. Затем из них создаем сети, прогоняем через них данные.
    После этого смотрим на качество ответов, берем топ-20 генов.
    Из топа создаем потомков, старое поколение сносим, создаем сети из получившихся потомков, и так пока не достигнем
    необходимого качества.
    """
    def __init__(self, from_file=False, population_size=100, neurons_number=10):
        """
        Метод инициализации популяции
        :param from_file: если True, то создаем первое поколение из сохраненного файла для продолжения обучения
        :param population_size: количество экземпляров сети в первом поколении
        :param neurons_number: количество нейронов в каждой сети
        """
        if from_file:  # если из файла
            with open('population.pickle', 'rb') as file:
                self.current_population = pickle.load(file)  # сразу инициализируем текущую популяцию из файла
            self.new_genomes = []  # переменная для потомков
            self.neurons_number = neurons_number  # сохраняем количество нейронов
            self.input_numbers = [0, 1]  # входные нейроны (можно параметризировать)
            self.output_number = 2
        else:  # если начинаем с нуля
            self.current_population = []  # инициализируем текущую популяцию
            self.new_genomes = []  # потомки
            self.neurons_number = neurons_number  # количество нейронов в каждой сети
            self.input_numbers = [0, 1]  # входные нейроны, надо параметризовать наверное
            self.output_number = 2
            for i in range(population_size):  # повторяем сколько экземпляров у нас должно быть
                # и создаем рандомные геномы
                test_genome = {
                    'cd': {
                        'input': {x: 5 * random.random() - 2.5 for x in range(11)},
                        'output': {x: 5 * random.random() - 2.5 for x in range(11)}
                    },
                    'dw': {
                        'input': {x: 5 * random.random() - 2.5 for x in range(11)},
                        'output': {x: 5 * random.random() - 2.5 for x in range(11)}
                    }
                }
                # из них создаем сети
                net = Net(neurons_number, genome=test_genome, input_neurons=self.input_numbers, output_neuron=self.output_number)
                # и сохраняем в список текущей популяции
                # net - сама сеть, genome - геном с параметрами синапсов, f_value - качество, metrics - заметки о сети
                self.current_population.append({'net': net, 'genome': test_genome, 'f_value': 0.0, 'metrics': ''})
                # обнуляем счетчики синапсов и нейронов для того, чтобы нумерация была идентичной
        self.average_number_of_synapses = 0  # среднее число синапсов в сетях

    # метод создания датасета для сети
    def create_dataset(self):
        X, y = generate_dataset(SIGNAL_LENGTH, 2, IMPULSES_NUMBER, custom_table=[{'x': [0, 1], 'y': 1},
                                                                                 {'x': [1, 0], 'y': 1},
                                                                                 {'x': [1, 1], 'y': 0},
                                                                                 {'x': [1, 1], 'y': 0}])
        return X, y

    # метод мутации лучших геномов
    def mutation(self, best_nets):
        for item in best_nets:  # идем по лучшим геномам
            for i in ['cd', 'dw']:
                for j in ['input', 'output']:
                    for k in range(len(item['cd']['input'].keys())):
                        if random.random() < 0.1:  # с вероятностью 0.1
                            item[i][j][k] += (2 * random.random() - 1) * 0.05  # прибавляем рандомное число к параметру
        self.new_genomes += best_nets  # записываем мутировавший геном в список нового поколения

    # метод размножения лучших геномов
    def crossingover(self, best_nets):
        new_genomes = []  # инициализируем куда будем их складывать
        for _ in range(40):  # нам нужно 40 детей
            a = random.randint(0, 19)  # т. к. лучших сетей придет 20, берем рандомные 2
            b = random.randint(0, 19)
            while a == b:  # проверяем что они не совпадают
                b = random.randint(0, 19)
            # берем эти геномы как родитлей
            parent_a = best_nets[a]
            parent_b = best_nets[b]
            # инициализируем дочерний геном
            child = {'cd': {'input': {}, 'output': {}},
                     'dw': {'input': {}, 'output': {}}}
            for i in ['cd', 'dw']:  # идем по всему геному
                for j in ['input', 'output']:
                    for k in range(len(parent_a['cd']['input'].keys())):
                        if random.random() > 0.5:  # с вероятностью 0.5 берем ген от одного или другого родителя
                            child[i][j][k] = parent_a[i][j][k]
                        else:
                            child[i][j][k] = parent_b[i][j][k]
            # сохраняем ребенка
            new_genomes.append(child)
        self.new_genomes += new_genomes  # сохраняем в общем списке

    # метод размножения, доступный и лучшим, и всяким бомжам
    # идентично методу crossingover, только на вход передаем все сети
    def random_objects(self, all_nets):
        new_genomes = []  # инициализируем куда будем их складывать
        for _ in range(40):  # нам нужно 40 детей
            a = random.randint(0, 99)  # т. к. придет 100 сетей (все), берем рандомные 2
            b = random.randint(0, 99)
            while a == b:  # проверяем что они не совпадают
                b = random.randint(0, 99)
            parent_a = all_nets[a]
            parent_b = all_nets[b]
            # инициализируем дочерний геном
            child = {'cd': {'input': {}, 'output': {}},
                     'dw': {'input': {}, 'output': {}}}
            for i in ['cd', 'dw']:  # идем по всему геному
                for j in ['input', 'output']:
                    for k in range(len(parent_a['cd']['input'].keys())):
                        if random.random() > 0.5:  # с вероятностью 0.5 берем ген от одного или другого родителя
                            child[i][j][k] = parent_a[i][j][k]
                        else:
                            child[i][j][k] = parent_b[i][j][k]
            # сохраняем ребенка
            new_genomes.append(child)
        self.new_genomes += new_genomes  # сохраняем в общем списке

    # метод создания сетей из генома
    def create_nets_from_genomes(self):
        for genome in self.new_genomes:  # идем по всем потомкам
            # и создаем из генома сеть
            net = Net(self.neurons_number, genome=genome, input_neurons=self.input_numbers, output_neuron=self.output_number)
            # добавляем ее к текущей популяции (старая популяция убивается во внешнем методе)
            self.current_population.append({'net': net, 'genome': genome, 'f_value': 0.0, 'metrics': ''})
            # обнуляем счетчики синапсов и нейронов чтобы методы ввода импульсов извне не ломались
        self.new_genomes = []  # очищаем переменную с новым поколением, потому что они уже в текущем

    # метод создания нового поколения
    def new_population(self, generation_number):
        # сортируем все сети по степени их приспособленности, у нас это f_value
        self.current_population.sort(key=lambda x: x['f_value'], reverse=True)
        print(f"Best net notes: {self.current_population[0]['metrics']}")
        accuracy = self.current_population[0]["f_value"]  # берем лучшее качество
        if accuracy > 0.65:
            net_number = self.current_population[0]['metrics'].split('Net number ')[0]
            net_number = net_number.split('.')[0]
            result = {'generation_number': generation_number, 'net_number': net_number, 'synapses': []}
            net = self.current_population[0]['net']
            print(net.s_real)
            print(net.s_weights)
            print(net.s_cd_input)
            print(net.s_cd_output)
            print(net.s_dw_input)
            print(net.s_dw_output)
            # with open('arch_' + str(random.random()).split('.')[1] + '.pickle', 'wb') as file:
            #     pickle.dump(result, file)
        self.get_average_number_of_synapses(self.current_population[:20])  # считаем синапсы
        best_nets = [x['genome'] for x in self.current_population[:20]]  # берем для размножения и мутации лучших
        all_nets = [x['genome'] for x in self.current_population]  # берем для размножения всех всех
        self.current_population = []  # очищаем текущую популяцию
        self.crossingover(best_nets)  # создаем потмков лучших
        self.mutation(best_nets)  # мутируем лучших
        self.random_objects(all_nets)  # создаем детей из рандомных родителей
        self.create_nets_from_genomes()  # создаем сети из геномов потомков
        # сохраняем текущее состояние сети в файл
        with open('population333.pickle', 'wb') as file:
            pickle.dump(self.current_population, file)
        return accuracy  # возвращаем лучший результат

    # вспомогательный метод, возвращает количество угаданных нулей и единиц по отдельности
    def guessed_number(self, a, b):
        counter_zeros = 0
        counter_ones = 0
        for i, j in zip(a, b):
            if i == j == 0:
                counter_zeros += 1
            elif i == j == 1:
                counter_ones += 1
        return counter_zeros, counter_ones

    # подсчет различных метрик качества
    def calculate_quality(self, predictions, train_y, test_length):
        counter_good = 0  # сколько было угадано
        test_number = SIGNAL_LENGTH * test_length  # от скольки тиков смотрим до конца
        counter_all = len(train_y[test_number:])  # сколько это значений всего
        for j, k in zip(predictions[test_number:], train_y[test_number:]):
            if j == k:
                counter_good += 1  # если угадали прибавляем счетчик
        # кастомная мера качества
        fitness_value = counter_good / counter_all * 0.5 + (
                1 - abs(predictions[test_number:].count(0) - predictions[test_number:].count(1)) / counter_all) * 0.5
        # стандартная мера качества
        f1_value = balanced_accuracy_score(train_y[test_number:], predictions[test_number:])
        # сколько нулей и единиц угадали
        counter_zeros, counter_ones = self.guessed_number(predictions[test_number:], train_y[test_number:])
        return f1_value, fitness_value, counter_zeros, counter_ones

    # метод подсчета среднего количества существующих синапсов в лучших сетях
    def get_average_number_of_synapses(self, nets):
        length = len(nets)
        count = 0
        for n in nets:
            count += n['net'].s_real.sum()
        self.average_number_of_synapses = count/length

    # метод обучения и проверки сети, accuracy - заданная требуемая точность
    def fit(self, accuracy=0.9):
        count = 0  # счетчик поколений
        current_accuracy = 0.0  # текущее качество лучшей сети
        train_X, train_y = self.create_dataset()  # создаем датасет
        print(train_y.count(1))  # для отладки, количество единиц в ответах датасета
        while current_accuracy < accuracy:  # пока не достигли требуемого качества
            count += 1  # прибавляем счетчик поколений
            print(f"Now learning {count} generation")
            t_start = datetime.datetime.now()  # сохраняем время начала прогона данных
            pop_accuracy = 0  # качество в текущем поколении
            for v, net_dict in enumerate(self.current_population):  # идем по всем сетям текущего поколения
                net = net_dict['net']  # берем непосредственно сеть
                # net.predict(train_X[:SIGNAL_LENGTH * 60], train_y[:SIGNAL_LENGTH * 60])  # обучаем
                predictions = net.predict(train_X, train_y, learning=True)  # получаем результаты ее работы
                # считаем качество
                test_length = 100 - 20  # от какого с конца сигнала используем датасет для расчета качества
                f1_value, fitness_value, counter_zeros, counter_ones = \
                    self.calculate_quality(predictions, train_y, test_length)
                net_dict['f_value'] = f1_value  # сохраняем качество данной сети
                # сохраняем в заметках параметры
                net_dict['metrics'] = \
                    str(f'Net number {v}. f: {str(f1_value)[:6]}, guessed zeros {counter_zeros}/{train_y[test_length * SIGNAL_LENGTH:].count(0)}, guessed ones {counter_ones}/{train_y[test_length * SIGNAL_LENGTH:].count(1)}, custom quality {str(fitness_value)[:6]}')
                if f1_value > pop_accuracy:  # если у нас лучшая сеть в популяции, говорим о ней
                    pop_accuracy = f1_value
                    print('New best net in current generation: ' + str(net_dict['metrics']))
            t_end = datetime.datetime.now()  # время конца прогона данных
            current_accuracy = self.new_population(count)  # создаем новое поколение
            print("   *****   ")
            print(f"Current population statistic:")
            print(f"generation number: {count}")
            print(f"elapsed time: {t_end - t_start}")
            print(f"best net quality: {str(current_accuracy)[:6]}")
            print(f"average number of synapses: {self.average_number_of_synapses}")
            print("   *****   ")
            print("   *****   ")


BUFFER_LENGTH = 80  # длина буфера памяти истории активаций каждого синапса и каждого нейрона
SIGNAL_LENGTH = 23  # длина сигнала в тиках
IMPULSES_NUMBER = 110


p = Population(population_size=100, neurons_number=5, from_file=False)
p.fit(0.92)
