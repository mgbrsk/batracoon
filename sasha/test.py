# -*- coding: utf-8 -*-

"""
Прямое и обратное распространение сигналов в нейронной сети
У каждого нейрона есть слоты, для входных и выходных нейронов, из которых и в которые они передают информацию.
Все нейроны сети находятся в списке, и последовательно на каждом из нейронов сети выполняются их методы,
в которых выполняются действия и с другими нейронами.
"""

import numpy as np
from copy import deepcopy
from itertools import combinations
import random


BUFFER_SIZE = 30  # длина сохраняемой истории в нейронах


class Neuron:
    def __init__(self, number, is_input=False, is_output=False, weights=None):
        """
        Инициализация нейрона
        :param number: порядковый номер нейрона
        :param is_input: является ли нейрон входныи
        :param is_output: является ли нейрон выходным
        :param weights: входные веса нейрона
        """
        self.number = number  # номер
        self.is_input = is_input  # входной ли
        self.is_output = is_output  # выходной ли
        self.history = [0 for _ in range(BUFFER_SIZE)]  # инициализируем историю активаций нейрона
        if not self.is_input:  # если не входной - создаем список для входных нейронов
            self.input_slots = []  # слоты для других нейронов
        if not self.is_output:  # если не выходной - создаем список для выходных нейронов
            self.output_slots = []  # слоты для других нейронов
        self.accumulator = 0  # аккумулятор для сохранения входных импульсов
        self.temp_signal = 0  # был ли активирован нейрон на предыдущем такте
        if not weights:  # если не заданы входные веса то пустой список, иначе сохраняем
            self.weights = []
        else:
            self.weights = weights
        self.references_from_right = []  # эталоны историй активаций справа, какии истории активаций им нужны
        # # в какие моменты времени нужны были импульсы с положительным весом
        # self.pos_reference_left = [0 for _ in range(BUFFER_SIZE)]
        # # в какие моменты времени нужны были импульсы с отрицательным весом
        # self.neg_reference_left = [0 for _ in range(BUFFER_SIZE)]
        # консолидированная эталонная история для данного нейрона (получается из медианы references_from_right)
        self.wanting = [0 for _ in range(BUFFER_SIZE)]  # для предыдущего слоя
        self.w_m_w = [0 for _ in range(BUFFER_SIZE)]  # для двиганья весов
        # self.w_pos = [0 for _ in range(BUFFER_SIZE)]
        # self.w_neg = [0 for _ in range(BUFFER_SIZE)]

    # проверка, превысило ли число в аккумуляторе порог
    def check_signal(self):
        out = False  # пока нейрон не активен (для ретюрна)
        if self.accumulator >= 1:  # если аккумулятор превысил порог
            self.temp_signal = 1  # говорим что нейрон стал активен
            self.history.append(1)  # записываем в историю что мы были активны на этом такте
            out = True  # нейрон активен
        else:  # если аккумулятор не превысил порог
            self.temp_signal = 0  # нейрон не активен
            self.history.append(0)  # записываем в историю что мы не были активны в этот такт
        self.accumulator = 0  # обнуляем аккумулятор
        self.history.pop(0)  # убираем самое старое значение из истории активаций
        return out  # возвращаем, активировались ли мы

    # передача сигнала из входных нейронов
    def move_forward(self):
        if self.is_input:  # если это входной нейрон то у него нет входных связей
            return
        # идем по всем входным нейронам, если они были активны то прибавляем вес связи к аккумулятору
        for n, w in zip(self.input_slots, self.weights):
            self.accumulator += n.temp_signal * w

    # получение желаемой истории из выходных нейронов, какую историю активаций им хочется
    def get_right_reference(self, y_true=None):
        if self.is_output:
            self.references_from_right.append(y_true)
            return
        for n in self.input_slots:
            self.references_from_right.append(n.wanting)

    # подсчет консолидированной эталонной истории, чтобы угодить как можно большим выходным нейронам
    def calculate_wanting(self):
        pass

    # метод изменения входных весов нейрона
    def move_weights(self):
        # если входной ничего не делаем, нет входных весов
        pass


"""
[0.4, -1]
[-0.6, 0.72]
[0.6399999999999999, 0.6599999999999999]
"""


s_connects = np.array([[0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0],
                       [1, 1, 0, 0, 0],
                       [0, 0, 1, 1, 0]])

s_weights = np.array([[0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [1, -1, 0, 0, 0],
                      [-1, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0]])


def main():
    def update_connectom(net, s_connects, s_weights):
        pass


    def make_connections(net, s_connects, s_weights):
        for n in net:
            n.input_slots = []
            n.output_slots = []
            n.weights = []
        neuron_numbers = len(s_connects)
        for i in range(neuron_numbers):
            for j in range(neuron_numbers):
                if s_connects[i][j] == 1:
                    net[i].input_slots.append(net[j])
                    net[i].weights.append(s_weights[i][j])
                    net[j].output_slots.append(net[i])


    # создаем нейроны, говорим кто они
    n0 = Neuron(0, is_input=True)
    n1 = Neuron(1, is_input=True)
    n2 = Neuron(2)
    n3 = Neuron(3)
    n4 = Neuron(4, is_output=True)
    net = [n0, n1, n2, n3, n4]

    make_connections(net, s_connects, s_weights)

    # обучающий датасет
    x = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
         [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
         [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
         [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
         [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
         [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
         [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
         [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
         [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
         [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
         [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
         [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
         [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
         [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
         [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0],
         [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],]
    y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]
    res = []

    # массив для сохранения правильной истории, фифо для у
    y_true = [0 for _ in range(BUFFER_SIZE)]

    weights_history = []

    counter = 0
    for _ in range(1):
        res = []
        # weights_history = []
        for cur_x, cur_y in zip(x, y):
            counter += 1
            cur_weights = [[round(y, 3) for y in x.weights] for x in net]
            weights_history.append(deepcopy(cur_weights))
            weights_history[-1].append(cur_x)
            weights_history[-1].append([cur_y])
            # обновляем фифо с правильными ответами
            y_true.append(cur_y)
            y_true.pop(0)

            # обновляем коннектом
            update_connectom(net, s_connects, s_weights)
            # обновляем слоты нейронов
            make_connections(net, s_connects, s_weights)

            # ввод сигналов в сеть
            for en, i in enumerate(cur_x):
                if i == 1:
                    net[en].accumulator = 1
            # проверка аккумуляторов и активация нейронов
            for n in net:
                out = n.check_signal()
                if n.is_output and out:
                    weights_history[-1].append([1])
                    res.append(1)
                elif n.is_output and not out:
                    weights_history[-1].append([0])
                    res.append(0)
            # передача сигнала вперед
            for n in net:
                n.move_forward()
            # обратное распространение ошибки
            for n in net:
                n.get_right_reference(y_true=y_true)
            for n in net:
                n.calculate_wanting()
            for n in net:
                n.move_weights()

    # for i, j in zip(y, res):
    #     print(i, j)
    for n in net:
        print(n.weights)
    # # print(weights_history)
    with open('net_log1.csv', 'w') as f:
        for i in weights_history:
            res = []
            for j in i:
                for k in j:
                    res.append(str(k))
            print(','.join(res), file=f)
            # print(','.join(res))


main()
