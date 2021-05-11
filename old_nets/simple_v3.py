# -*- coding: utf-8 -*-
"""
Данный код описывает подход коннективизма в теории нейронных сетей
Основная идея в том, что нейроны обладают простым поведением, одинаковым при одинаковых условиях.
То есть они могут самоорганизовываться, образовывать синапсы, именять их веса в зависимости от условий
окружающей среды. То есть нам необходимо найти такое поведение нейрона, которое обеспечит самоорганизацию
и самообучение на любых типах задач.
В данном подходе под поведением нейрона подразумевается создание/уничтожение синапса, а также изменение
весов существующих синапсов в зависимости от внешних условий (насколько часто активируется данный синапс
и каждый из его соседей, количество синапсов, частота их активаций, правильность ответа сети и т. д.)
Существует несколько ограничений, такие как дискретность времени, на активацию нейрона и переход импульса в
выходные синапсы отводится один тик, отсутствует время рефрактерности, и так далее.
Создание и уничтожение синапса регулируется переменной cd, изменение веса существующих синапсов переменной dw.
Данные переменные считаются для каждого возможного синапса на каждом тике сети, cd считается для существующих и
виртуальных синапсов, dw считается только для существующих.
Если cd превышает определенный порог то синапс создается, если он становится ниже другого порога,
то синапс уничтожается.
Если синапс существует, то его вес изменяется на величину dw.
Эти переменные считаются как:
cd = (a11 * x1) + (b11 * y1) + ... + (a12 * x2) + (b12 * y2) + ...
dw = (a21 * x1) + (b21 * y1) + ... + (a22 * x2) + (b22 * y2) + ...
где:
a11, b11 и т. д. - коэффициенты, отвечающие за создание/уничтожение синапса, учитывающие вклад входного нейрона синапса
a12, b12 и т. д. - кэфы, отвечающие за создание/уничтожение синапса, учитывающие вклад выходного нейрона синапса
a21, b21 и т. д. - кэфы, отвечающие за изменение веса синапса, учитывающие вклад входного нейрона
a22, b22 и т. д. - кэфы, отвечающие за изменение веса синапса, учитывающие вклад выходного нейрона
x1, y1 и т. д. - параметры входного нейрона синапса
x2, y2 и т. д. - параметры выходного нейрона синапса
В данном коде используется эволюционный алгоритм, с целью нахождения наиболее оптимальных коэффициентов синапсов,
которые будут давать адекватное поведение сети в большинстве задач.
made by Sasha
"""

from collections import defaultdict, Counter
import random
from functools import reduce
from random import randint
import datetime
import pickle

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
            io_list.append({'x' : el_binary, 'y' : randint(0, 1)})
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
        rand_index = randint(0, len(io_list)-1)
        for uwu in range(signal_length):
            x_list.append(io_list[rand_index]['x'])
            y_list.append(io_list[rand_index]['y'])
            
    return x_list, y_list


class Synapse:
    """
    Класс синапса
    Проводит сигнал из одного нейрона в другой, содержит в себе параметры для вычисления cd/dw, факт собственного
    существования, вес, экземпляры связанныйх нейронов, есть ли на текущем тике внутри сигнал
    """
    # уникальный номер синапса в сети
    global_number = 0

    def __init__(self, input_n, output_n, weight=0.5, genome=None):
        """
        Инициализация синапса
        :param input_n: экземпляр входного нейрона
        :param output_n: экземпляр выходного нейрона
        :param weight: вес синапса
        :param genome: геном, набор параметров для вычисления cd/dw, имеет вид
        {'
            cd':
                {
                    'input': {0: 0.3, 1: 0.95, 2: -0.735, 3: 0.63, ...},
                    'output': {0: -0.9, 1: 0.01, 2: -0.67, 3: 0.788, ...}
                },
            'dw':
                {
                    'input': {0: 0.99, 1: 0.64, 2: -0.7, 3: -0.87, ...},
                    'output': {0: -0.32, 1: -0.01, 2: -0.245, 3: 0.09, ...}
                }
        }
        """
        self.number = Synapse.global_number  # берем глобальный номер - параметр класса синапса
        Synapse.global_number += 1  # увеличиваем глобальный номер, чтобы следующий синапс имел номер на 1 больше
        self.input_neuron = input_n  # сохраняем внутри себя входной нейрон
        self.output_neuron = output_n  # также сохраняем выходной
        if genome:  # если передан геном то используем его, если нет - то используем заглушку
            self.genome = genome
        else:
            self.genome = {'cd': {'input': {}, 'output': {}}, 'dw': {'input': {}, 'output': {}}}
        self.is_signal = False  # есть ли внутри сигнал, по умолчанию нет
        self.weight = weight  # вес
        self.input_vars = {}  # параметры входного нейрона, обновляем на каждом тике
        self.output_vars = {}  # параметры выходного нейрона, обновляем на каждом тике
        self.is_real = False  # флаг существования синапса, проводит ли он сигнал, по умолчанию все синапсы виртуальные
        self.cd = 0.5  # умолчательное значение переменной, ответственной за создание/уничтожение синапса
        self.dw = 0  # умолчательное значение переменной, ответственной за изменение веса синапса
        self.history = [0 for _ in range(BUFFER_LENGTH)]  # история активаций синапса

    # проверка есть ли сигнал в синапсе и сразу зануляем
    def check_signal(self):
        if self.is_signal and self.is_real:  # если есть сигнал и синапс существует
            self.is_signal = False
            return True
        else:
            return False

    # кастомный метод деления, чтобы заглушить ошибку деления на ноль
    def get_division(self, x, y):
        try:
            return x / y
        except ZeroDivisionError:
            return 0

    # метод получения параметров нейронов, которые связывает данный синапс
    def get_vars(self):
        input_vars = self.input_neuron.get_params()
        self.input_vars = input_vars.copy()  # копируем, чтобы не изменялся словарь параметров внутри синапса
        output_vars = self.output_neuron.get_params()
        self.output_vars = output_vars.copy()
        self.preprocess_vars()  # предобработка параметров нейронов, добавление новых

    # метод предобработки
    def preprocess_vars(self):
        first_history = self.input_vars[1]  # в этом параметре история активаций входного нейрона
        second_history = self.output_vars[1]  # история активаций выходного нейрона
        counter = 0  # счетчик совападений активаций входного и выходного нейрона
        for i, j in zip(first_history, second_history):
            if (i == 1) and (i == j):
                counter += 1
        self.input_vars[1] = counter / 100  # запись количества совпадений активаций
        self.output_vars[1] = counter / 100

        # сохранение отношений различных параметров как новые параметры
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

    # метод подсчета переменных cd и dw
    def calculate_cddw(self):
        self.cd = 0
        self.dw = 0
        '''
        genome:
        cd, input - [0, 1, 2, ..., 85], output - [0, 1, 2, ..., 85]
        dw, input - [0, 1, 2, ..., 85], output - [0, 1, 2, ..., 85]
        '''
        # проходимся по всем коэффициентам синапса и вытащенным параметрам нейронов и считаем переменные cd и dw
        for key, value in self.genome['cd']['input'].items():
            self.cd += self.input_vars[key] * value
        for key, value in self.genome['cd']['output'].items():
            self.cd += self.output_vars[key] * value
        for key, value in self.genome['dw']['input'].items():
            self.dw += self.input_vars[key] * value
        for key, value in self.genome['dw']['output'].items():
            self.dw += self.output_vars[key] * value

        # ограничиваем пределы переменных cd и dw
        self.cd = self.cd if self.cd <= 1 else 1
        self.cd = self.cd if self.cd >= 0 else 0
        self.dw = self.dw if self.dw <= 1 else 1
        self.dw = self.dw if self.dw >= -1 else -1

    # метод проверки создания или удаления данного синапса
    def check_existing(self):
        if (not self.is_real) and self.cd >= 0.8:  # если нейрон не существует и мы преодолели порог
            self.is_real = True  # делаем его реальным
        elif self.is_real and self.cd <= 0.2:  # если нейрон уже существует но переменная cd упала до порога
            self.is_real = False  # говорим что этот синапс уничтожен
        return self.is_real

    # метод изменения веса в соответствии с переменной dw
    def move_weight(self):
        if not self.is_real:  # если нейрон не существует то ничего не меняем
            return
        self.weight += self.dw * 0.05  # иначе изменяем на величину dw * 0.05

    # метод передачи активации в выходной нейрон
    def add_signals(self):
        if self.check_signal():  # если в нас есть сигнал
            self.history.append(1)  # сохраняем его для истории
            self.output_neuron.add_accumulator(self.weight, self.number)  # передаем сигнал в выходной нейрон
        else:
            self.history.append(0)  # иначе сохраняем для истории что мы сидели на попе ровно
        self.history.pop(0)  # убираем самое старое значение из истории

    # метод активации синапса (передачи сигнала из входного нейрона в синапс)
    def check_input_signal(self):
        if not self.is_real:  # если синапс не существует то ничего не делаем
            return
        if self.input_neuron.is_spiked():  # иначе если входной нейрон активирован, то сохраняем сигнал в синапсе
            self.is_signal = True


class Neuron:
    """
    Класс нейрона
    Нейрон хранит в себе расстояние до выхода, историю активаций, историю с каких
    синапсов приходил сигнал, также хранит в себе экземпляры всех входных и выходных нейронов, свои параметры.
    На каждом тике в аккумулятор нейрона прибавляются значения весов тех входных синапсов, которые были активированы.
    Если при проверке аккумулятора он не преодолел порог, то нейрон не считается активированным.
    По сути, нейрон просто проводит сигналы, считает свои параметры и отдает их своим синапсам
    Внутренние параметры нейрона:
    -0 расстояние до выхода
    -1 показатель как часто одновременно активируется этот нейрон и другой (передаем историю активаций)
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
    В самом синапсе будут считаться:
    -48(а) количество активаций за последнее время (последняя частота)
    -49(б) вес синапса
    -50..67 отношение а и (1-18)
    -68..85 отношение б и (1-18)
    """
    global_number = 0  # глобальный номер для нумерации нейронов, аналогично схеме в синапсах

    def __init__(self, is_input=False, is_output=False):
        """
        Метод инициализации нейрона
        :param is_input: входной ли это нейрон
        :param is_output: выходной ли это нейрон
        """
        self.number = Neuron.global_number
        Neuron.global_number += 1
        self.accumulator = 0  # аккумулятор для суммирования входных импульсов
        self.p = defaultdict(int)  # переменная для сохранения параметров данного нейрона
        self.coordinate = random.random()  # по умолчанию расстояние до выходов сети - рандомное
        if is_input and (not is_output):  # но если это входной нейрон то координата 1
            self.coordinate = 1
        elif (not is_input) and is_output:  # или если выходной, то координата 0
            self.coordinate = 0
        self.spike_history = [0 for _ in range(BUFFER_LENGTH)]  # история активаций данного нейрона
        self.synapse_history = [[] for _ in range(BUFFER_LENGTH)]  # история, какие входные синапсы были активированы
        self.input_synapses = []  # входные синапсы (экземпляры)
        self.output_synapses = []  # выходные синапсы (экземпляры)

    def get_number_of_synapses(self, array, sign=None):
        """
        вспомогательный метод подсчета синапсов (всех, либо только с положительными весами,
        либо только с отрицательными) в заданном массиве array
        :param array: массив, в котором будем считать синапсы
        :param sign: знак синапсов, которые надо посчитать, если не указано то считаем все
        :return: возвращаем количество синапсов
        """
        counter = 0  # инициализируем счетчик
        if not sign:  # если не указан знак
            for synapse in array:  # проходимся по всем синапсам из массива
                if synapse.is_real:  # если данный синапс существует то считаем его
                    counter += 1
            return counter
        else:  # если все таки есть знак для подсчета
            if sign == 'pos':  # если нужны положительные синапсы
                for synapse in array:  # проходим по всем синапсам
                    if synapse.is_real and synapse.weight >= 0:  # если синапс существует и его вес больше 0, то считаем
                        counter += 1
                return counter
            elif sign == 'neg':  # если нужны отрицательные синапсы
                for synapse in array:
                    if synapse.is_real and synapse.weight < 0:  # тут считаем только если вес синапса меньше 0
                        counter += 1
                return counter

    def get_sum_of_weights(self, array, sign=None):
        """
        вспомогательный метод подсчета суммы весов синапсов (всех, либо только с положительными весами,
        либо только с отрицательными) в заданном массиве array
        :param array: массив, в котором будем считать
        :param sign: знак синапсов, которые надо будет посчитать, если знака нет то считаем все
        :return: возвращаем сумму весов
        """
        counter = 0  # инициализируем счетчик
        if not sign:  # если знак не передан то считаем все
            for synapse in array:  # проходимся по всем синапсам
                if synapse.is_real:  # если синапс существует
                    counter += synapse.weight  # суммируем все веса
            return counter
        else:  # если передан знак
            if sign == 'pos':  # если нужны положительные веса
                for synapse in array:  # идем по всем синапсам
                    if synapse.is_real and synapse.weight >= 0:  # если синапс существует и его вес больше 0
                        counter += synapse.weight  # суммируем их веса
                return counter
            elif sign == 'neg':  # если нужны отрицательные веса
                for synapse in array:  # идем по всем синапсам
                    if synapse.is_real and synapse.weight < 0:  # если синапс существует и его вес меньше 0
                        counter += synapse.weight  # суммируем их веса
                return counter

    def signed_synapses(self, sign='pos'):
        """
        вспомогательный метод возвращает номера входных существующих синапсов, веса которых имеют либо положительный
        либо отрицательный знак
        :param sign: знак весов синапсов, pos или neg
        :return: возвращает список номеров синапсов
        """
        res = []  # инициализируем список
        if sign == 'pos':  # если нужны положительные входные синапсы
            for s in self.input_synapses:  # идем по входным синапсам
                if s.is_real and (s.weight >= 0):  # если он существует и его вес больше 0
                    res.append(s.number)  # сохраняем в список его номер
            return res
        elif sign == 'neg':  # если нужны отрицательные входные синапсы
            for s in self.input_synapses:  # идем по всем входным синапсам
                if s.is_real and (s.weight < 0):  # если синапс существует и его вес меньше 0
                    res.append(s.number)  # сохраняем его номер в список
            return res

    def get_sign_sum_of_input_impulses(self, sign='pos'):
        """
        вспомогательный метод, возвращает количество положительных либо отрицательных входных синапсов, которые
        активировались за последние BUFFER_LENGTH тиков
        :param sign: знак необходимых синапсов, pos - положительные, neg - отрицательные
        :return: возвращаем количество
        """
        counter = 0  # инициализируем счетчик
        if sign == 'pos':  # если нам нужны положительные синапсы
            # получаем номера всех существующих входных положительных синапсов
            positive_synapse_numbers = self.signed_synapses(sign='pos')
            # идем по истории входных синапсов (каждый элемент -
            # какие входные синапсы имели сигнал на тот тик, когда была запись)
            for item in self.synapse_history:
                for synapse in item:  # идем по всем синапсам, какие были активны на тот тик
                    # если номер этого синапса есть в списке существующих положительных входных синапсов
                    if synapse in positive_synapse_numbers:
                        counter += 1  # то прибавляем 1 к счетчику
            return counter
        elif sign == 'neg':  # если нужны отрицательные синапсы
            # получаем номера всех существующих входных отрицательных синапсов
            negative_synapse_numbers = self.signed_synapses(sign='neg')
            # идем по истории входных синапсов (каждый элемент -
            # какие входные синапсы имели сигнал на тот тик, когда была запись)
            for item in self.synapse_history:
                for synapse in item:  # идем по всем синапсам, какие были активны на тот тик
                    # если номер этого синапса есть в списке существующих положительных входных синапсов
                    if synapse in negative_synapse_numbers:
                        counter += 1  # то прибавляем 1 к счетчику
            return counter

    # вспомогательный метод нахождения отношений параметров нейрона, если в знаменателе 0 то возвращаем 0
    def get_division(self, x, y):
        try:
            return self.p[x] / self.p[y]
        except ZeroDivisionError:
            return 0

    # вспомогательный метод, возвращает количество активаций наиболее часто
    # активирующегося синапса за последние BUFFER_LENGTH тиков
    def get_more_activated(self):
        # склеиваем все внутренние списки общего списка истории активаций синапсов
        all_activations = reduce(lambda x, y: x.extend(y) or x, self.synapse_history, [])
        # try except нужен если у нас совсем пустая история
        try:
            # возвращаем количество вхождений наиболее часто встречающегося номера
            return Counter(all_activations).most_common(1)[0][1]
        except:
            return 0

    # метод возвращает внутренние параметры данного нейрона
    def get_params(self):
        return self.p

    # метод рассчета внутренних параметров нейрона
    # на вход (is_right) подаем правильно ли сеть ответила на предыдущем тике (0 или 1)
    def extract_params(self, is_right=0):
        self.p[0] = self.coordinate  # расстояние до выхода
        # показатель как часто одновременно активируется этот нейрон и другой (передаем историю активаций)
        self.p[1] = self.spike_history
        self.p[2] = self.get_number_of_synapses(self.input_synapses) / 100  # количество входных синапсов
        self.p[3] = self.get_number_of_synapses(self.output_synapses) / 100  # количество выходных синапсов
        # количество отрицательных входных синапсов
        self.p[4] = self.get_number_of_synapses(self.input_synapses, sign='neg') / 100
        # количество положительных входных синапсов
        self.p[5] = self.get_number_of_synapses(self.input_synapses, sign='pos') / 100
        # количество отрицательных выходных синапсов
        self.p[6] = self.get_number_of_synapses(self.output_synapses, sign='neg') / 100
        # количество положительных выходных синапсов
        self.p[7] = self.get_number_of_synapses(self.output_synapses, sign='pos') / 100
        self.p[8] = self.get_sum_of_weights(self.input_synapses) / 100  # сумма весов входных синапсов
        self.p[9] = self.get_sum_of_weights(self.output_synapses) / 100  # сумма весов выходных синапсов
        # сумма весов входных положительных синапсов
        self.p[10] = self.get_sum_of_weights(self.input_synapses, sign='pos') / 100
        # сумма весов входных отрицательных синапсов
        self.p[11] = self.get_sum_of_weights(self.output_synapses, sign='neg') / 100
        # сумма весов выходных положительных синапсов
        self.p[12] = self.get_sum_of_weights(self.input_synapses, sign='pos') / 100
        # сумма весов выходных отрицательных синапсов
        self.p[13] = self.get_sum_of_weights(self.output_synapses, sign='neg') / 100
        self.p[14] = self.spike_history.count(1) / 100  # количество спайков за последнее время
        # количество входных импульсов за последнее время
        self.p[15] = reduce(lambda x, y: x + len(y), self.synapse_history, 0) / 100
        # количество положительных входных импульсов за последнее время
        self.p[16] = self.get_sign_sum_of_input_impulses(sign='pos') / 100
        # количество отрицательных входных импульсов за последнее время
        self.p[17] = self.get_sign_sum_of_input_impulses(sign='neg') / 100
        # количество активаций самого активного входного синапса за последнее время
        self.p[18] = self.get_more_activated() / 100
        # производные параметры - отношения различных параметров
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

    # метод передачи импульса в нейрон (вызывается входным синапсом)
    def add_accumulator(self, value, synapse_number):
        self.accumulator += value  # прибавляем значение в аккумулятор
        self.synapse_history[-1].append(synapse_number)  # сохраняем номер синапса который передал импульс для истории

    # проверка активируется ли данный нейрон
    def is_spiked(self):
        is_spiked = 0  # по умолчанию нейрон не активен
        if self.accumulator >= 1:  # если аккумулятор нейрона преодолел порог
            is_spiked = 1  # нейрон активен
        self.spike_history[-1] = is_spiked  # сохраняем в историю что данный нейрон был активен / не активен
        return bool(is_spiked)

    # метод перезагрузки нейрона (вызывается на каждом тике)
    def erase(self):
        self.accumulator = 0  # обнуляем аккумулятор
        # и сдвигаем историю как в регистрах
        self.spike_history.append(0)
        self.spike_history.pop(0)
        self.synapse_history.append([])
        self.synapse_history.pop(0)


class Net:
    """
    Класс сети. Создает и хранит в себе все нейроны и синапсы, координирует их работу на каждый тик. Вводит
    входные сигналы в сеть, и вытаскивает был ли активен выходной нейрон на данном тике (сохраняет в массив)
    """
    def __init__(self, n_neurons, genome=None, input_numbers=None, output_number=None):
        """
        Метод инициализации сети
        :param n_neurons: количество нейронов в сети
        :param genome: геном, который будет передан в синапсы (их коэффициенты для расчетов cd/dw)
        :param input_numbers: номера входных нейронов, список
        :param output_number: номер выходного нейрона
        """
        self.neurons = []  # список, хранящий все нейроны сети
        self.synapses = []  # список, хранящий все синапсы сети, и реальные и виртуальные
        if input_numbers is None:  # если нет номеров входных нейронов то используем значения по умолчанию
            input_numbers = [0, 1]
        if output_number is None:  # если нет номера выходного нейрона то используем значение по умолчанию
            output_number = 4
        self.output_number = output_number  # сохраняем номер для проверки активируется ли он - для регистрации выхода
        for i in range(n_neurons):  # сколько нейронов надо создать
            if i in input_numbers:  # если этот нейрон входной
                self.neurons.append(Neuron(is_input=True))  # то создаем его и говорим ему о том что он входной
            elif i == output_number:  # если этот нейрон выходной
                self.neurons.append(Neuron(is_output=True))  # создаем его и говорим ему о том что он выходной
            else:  # если это обычный нейрон то просто создаем его
                self.neurons.append(Neuron())
        # идем по всем созданным нейронам
        for outer_neuron in self.neurons:
            # и тут идем по всем созданным нейронам, чтобы создать связи все со всеми
            for inner_neuron in self.neurons:
                # связь нейрона самого с собой не создаем
                if outer_neuron.number == inner_neuron.number:
                    continue
                # если это различные нейроны то создаем синапс между этими нейроннами с заданным геномом
                synapse = Synapse(outer_neuron, inner_neuron, genome=genome)
                self.synapses.append(synapse)  # сохраняем его в сети
                # сохраняем этот синапс в выходном нейроне как выходной синапс
                outer_neuron.output_synapses.append(synapse)
                inner_neuron.input_synapses.append(synapse)  # и во входном нейроне как входной синапс

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

    # все действия, которые происходят на каждый такт времени
    def tick(self, reaction, learning=True):
        out_signal = False  # пока на выходе сети нет сигнала
        for s in self.synapses:
            s.check_input_signal()  # передаем из входных нейрнов сигналы в существующие синапсы
        for n in self.neurons:
            if n.number == self.output_number and n.is_spiked():  # проверка активации выходного нейрона
                out_signal = True
            n.erase()  # очищаем все нейроны
        for s in self.synapses:
            s.add_signals()  # передаем сигналы из синапсов в выходные нейроны
        if learning:
            for n in self.neurons:
                n.extract_params(is_right=reaction)  # рассчитываем внутренние параметры нейронов
            for s in self.synapses:
                s.get_vars()  # получаем параметры нейронов
                s.calculate_cddw()  # рассчитываем cd и dw
                s.check_existing()  # убираем или создаем или оставляем как есть синапс
                s.move_weight()  # изменяем вес синапса
        return int(out_signal)  # возвращаем, активировался ли выходной нейрон

    # метод прогона входной последовательности сигналов и передачи эталонных результатов в сеть
    def predict(self, X, y, learning=True):
        result = []
        # идем по входным данным
        for i in range(len(X)):
            cur_x = X[i]
            self.massive_probe(cur_x)  # запускаем сигналы во входные нейроны
            if i == 0:  # если это самый первый тик
                res = self.tick(0, learning=learning)  # передаем заданное значение ошибки
            else:  # если уже не первый тик
                if result[-1] == y[i - 1]:  # проверяем что выход сети совпадает с эталонным ответом на этом тике
                    res = self.tick(1, learning=learning)  # если совпало говорим сети что ок
                else:
                    res = self.tick(0, learning=learning)  # если не совпало говорим что не ок
            result.append(res)  # записываем что у нас на данном тике на выходе
        return result  # возвращаем результат


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
                        'input': {x: 2 * random.random() - 1 for x in range(105)},
                        'output': {x: 2 * random.random() - 1 for x in range(105)}
                    },
                    'dw': {
                        'input': {x: 2 * random.random() - 1 for x in range(105)},
                        'output': {x: 2 * random.random() - 1 for x in range(105)}
                    }
                }
                # из них создаем сети
                net = Net(neurons_number, genome=test_genome, input_numbers=self.input_numbers, output_number=self.output_number)
                # и сохраняем в список текущей популяции
                # net - сама сеть, genome - геном с параметрами синапсов, f_value - качество, metrics - заметки о сети
                self.current_population.append({'net': net, 'genome': test_genome, 'f_value': 0.0, 'metrics': ''})
                # обнуляем счетчики синапсов и нейронов для того, чтобы нумерация была идентичной
                # иначе ломаются методы введения сигнала в сеть (там по номерам нейронов идет поиск)
                Synapse.global_number = 0
                Neuron.global_number = 0
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
                            item[i][j][k] += (2 * random.random() - 1) * 0.1  # прибавляем рандомное число к параметру
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
            net = Net(self.neurons_number, genome=genome, input_numbers=self.input_numbers, output_number=self.output_number)
            # добавляем ее к текущей популяции (старая популяция убивается во внешнем методе)
            self.current_population.append({'net': net, 'genome': genome, 'f_value': 0.0, 'metrics': ''})
            # обнуляем счетчики синапсов и нейронов чтобы методы ввода импульсов извне не ломались
            Synapse.global_number = 0
            Neuron.global_number = 0
        self.new_genomes = []  # очищаем переменную с новым поколением, потому что они уже в текущем

    # метод создания нового поколения
    def new_population(self, generation_number):
        # сортируем все сети по степени их приспособленности, у нас это f_value
        self.current_population.sort(key=lambda x: x['f_value'], reverse=True)
        print(f"Best net notes: {self.current_population[0]['metrics']}")
        accuracy = self.current_population[0]["f_value"]  # берем лучшее качество
        if accuracy > 0.75:
            net_number = self.current_population[0]['metrics'].split('Net number ')[0]
            net_number = net_number.split('.')[0]
            result = {'generation_number': generation_number, 'net_number': net_number, 'synapses': []}
            net = self.current_population[0]['net']
            for s in net.synapses:
                if s.is_real:
                    result['synapses'].append([s.input_neuron.number, s.output_neuron.number, s.weight])
                    print(s.input_neuron.number, s.output_neuron.number, s.weight)
            with open('arch_' + str(random.random()).split('.')[1] + '.pickle', 'wb') as file:
                pickle.dump(result, file)
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
            for s in n['net'].synapses:
                if s.is_real:
                    count += 1
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


p = Population(population_size=100, neurons_number=5)
p.fit(0.92)
