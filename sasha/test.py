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


BUFFER_SIZE = 20  # длина сохраняемой истории в нейронах


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
        # в какие моменты времени нужны были импульсы с положительным весом
        self.pos_reference_left = [0 for _ in range(BUFFER_SIZE)]
        # в какие моменты времени нужны были импульсы с отрицательным весом
        self.neg_reference_left = [0 for _ in range(BUFFER_SIZE)]
        # консолидированная эталонная история для данного нейрона (получается из медианы references_from_right)
        self.wanting = [0 for _ in range(BUFFER_SIZE)]
        self.w_pos = [0 for _ in range(BUFFER_SIZE)]
        self.w_neg = [0 for _ in range(BUFFER_SIZE)]

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
    def get_right_reference(self):
        # если входной (нет входных весов - не обучается) или выходной (нет выходных нейронов) - ничего не делаем
        if self.is_input or self.is_output:
            return
        # берем эталонную историю выходных нейронов и записываем ее к себе
        for n in self.output_slots:
            self.references_from_right.append([n.w_pos, n.w_neg])

    # подсчет консолидированной эталонной истории, чтобы угодить как можно большим выходным нейронам
    def calculate_wanting(self, y_true=None):
        """
        :param y_true: история правильных ответов - для выходного нейрона, если не выходной то не учитывается
        :return:
        """
        # if self.number == 2:
        #     print(self.wanting)
        # ####
        # если входной то ничего не делаем
        if self.is_input:
            self.references_from_right = []
            return
        # если выходной то эталонная история для него - история правильных ответов датасета
        if self.is_output:
            # self.wanting = y_true
            temp = np.array(y_true) - np.array(self.history)
            self.w_pos = np.where(temp > 0, 1, 0)
            self.w_neg = np.where(temp < 0, 1, 0)
            self.references_from_right = []
            # print(self.w_pos)
            # print(self.w_neg)
            return

        # if self.number == 2:
        #     print(self.w_pos)
        #     # print(self.w_neg)

        if len(self.references_from_right) == 1:
            self.w_pos = self.references_from_right[0][0]
            self.w_neg = self.references_from_right[0][1]
            self.references_from_right = []
            return

        positive = [[en] + x[0] for en, x in enumerate(self.references_from_right)]  # позитивные требования
        negative = [[en] + x[1] for en, x in enumerate(self.references_from_right)]

        # !!! возможно стоит еще сравнить с текущей историей
        temp = list(combinations(positive + negative, r=2))  # все комбинации
        temp = [[x[0][1:], x[1][1:]] for x in temp if x[0][0] != x[1][0]]  # очистка комбинаций от показателей знака
        temp = np.array(temp)   # очищенные комбинации историй
        func = lambda x: x[0] * x[1]
        t = np.apply_along_axis(func, 1, temp)  # произведения комбинаций историй

        temp_sum = np.sum(t, axis=1)  # рейтинг похожестей
        indexes_of_max = np.where(temp_sum == temp_sum.max())
        random_index = random.choice(indexes_of_max)
        reference = list(t[random_index])
        temp_dif = np.array(reference) - np.array(self.history)
        self.w_pos = np.where(temp_dif > 0, 1, 0)
        self.w_neg = np.where(temp_dif < 0, 1, 0)
        self.references_from_right = []

    # # подсчет тех тактов, где нужно было активироваться но мы активны не были, и наоборот,
    # # где не нужно было активироваться но мы были активны
    # def calculate_pos_neg_reference(self):
    #     # если входной то ничего не делаем, они не обучаются, у них нет входных весов
    #     if self.is_input:
    #         return
    #     # temp = np.array(self.history) - np.array(self.wanting)
    #     # берем разницу из настоящий истории активаций нейрона и эталона истории активаций,
    #     # там получится 1 где нужно было быть активным но мы молчали, и -1 где нужно было молчать но мы активировались
    #     temp = np.array(self.wanting) - np.array(self.history)
    #     # разделяем на два массива
    #     self.pos_reference_left = np.where(temp > 0, 1, 0)
    #     self.neg_reference_left = np.where(temp < 0, 1, 0)

    # метод изменения входных весов нейрона
    def move_weights(self):
        # если входной ничего не делаем, нет входных весов
        if self.is_input:
            return
        # идем по всем входным нейронам
        for en, n in enumerate(self.input_slots):
            pos_coef1 = np.array(n.history) * np.array(self.pos_reference_left)
            pos_coef2 = np.array(n.history) == np.array(self.pos_reference_left)
            pos_coef2 = pos_coef2 * 1
            pos_coef = pos_coef1 * 0.2 + pos_coef2 * 0.8
            # pos_coef = np.array(n.history) * np.array(self.pos_reference_left)

            # считаем насколько похожи истории необходимых положительных импульсов
            # и история активации данного входного нейрона
            # pos_coef = np.array(n.history) == np.array(self.w_pos)
            # нормируем от 0 до 1 - это степень похожести
            pos_coef = np.sum(pos_coef * 1) / pos_coef.shape[0]
            # если у нас в истории необходимых положительных импульсов ничего нет -
            # то считаем степень похожести как 0 (чтобы не двигать вес)
            # if (not np.array(self.pos_reference_left).any()):
            #     # (not np.array(n.history).any()):  #  and (not np.array(self.pos_reference_left).any())
            #     pos_coef = 0

            neg_coef1 = np.array(n.history) * np.array(self.neg_reference_left)
            neg_coef2 = np.array(n.history) == np.array(self.neg_reference_left)
            neg_coef2 = neg_coef2 * 1
            neg_coef = neg_coef1 * 0.2 + neg_coef2 * 0.8
            # neg_coef = np.array(n.history) * np.array(self.neg_reference_left)

            # то же самое и для истории отрицательных необходимых импульсов
            # neg_coef = np.array(n.history) == np.array(self.w_neg)
            neg_coef = np.sum(neg_coef * 1) / neg_coef.shape[0]
            # if (not np.array(self.neg_reference_left).any()):
            #     # (not np.array(n.history).any()):  #  and (not np.array(self.neg_reference_left).any())
            #     neg_coef = 0

            # if self.is_output:
            #     print(pos_coef, neg_coef)

            # если отрицательная и положительная похожесть одинакова то ничего не делаем
            # if pos_coef == neg_coef:
            #     continue
            # если положительная похожесть больше чем отрицательная то двигаем вес в сторону его увеличения,
            # если отрицательная больше - то двигаем вес в сторону его уменьшения
            if pos_coef >= neg_coef:
                self.weights[en] += pos_coef * 0.15
            else:
                # if self.is_output:
                #     pass
                # else:
                self.weights[en] -= neg_coef * 0.15
            # жестко ограничиваем веса от -1 до 1
            if self.weights[en] > 1:
                self.weights[en] = 1
            if self.weights[en] < -1:
                self.weights[en] = -1


def make_connection(n1, n2):
    pass

# создаем нейроны, говорим кто они
n0 = Neuron(0, is_input=True)
n1 = Neuron(1, is_input=True)
n2 = Neuron(2, weights=[0, 0])
n3 = Neuron(3, weights=[0, 0])
n4 = Neuron(4, is_output=True, weights=[0, 0])
# и с кем у них есть входные и выходные связи
n0.output_slots = [n2, n3]
n1.output_slots = [n2, n3]
n2.input_slots = [n0, n1]
n3.input_slots = [n0, n1]
n2.output_slots = [n4]
n3.output_slots = [n4]
n4.input_slots = [n2, n3]
# создаем сеть
net = [n0, n1, n2, n3, n4]

# n0 = Neuron(0, is_input=True)
# n1 = Neuron(1, is_input=True)
# n2 = Neuron(2, weights=[0, 0])
# n3 = Neuron(3, weights=[0])
# n4 = Neuron(4, weights=[0])
# n5 = Neuron(5, is_output=True, weights=[0, 0, 0])
#
# n0.output_slots = [n2, n3]
# n1.output_slots = [n2, n4]
#
# n2.input_slots = [n0, n1]
# n2.output_slots = [n5]
#
# n3.input_slots = [n0]
# n3.output_slots = [n5]
#
# n4.input_slots = [n1]
# n4.output_slots = [n5]
#
# n5.input_slots = [n2, n3, n4]
# net = [n0, n1, n2, n3, n4, n5]

# обучающий датасет
x = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
     [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
     [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
     [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
     [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
     [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
     [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
     [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
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
        cur_weights = [[round(y, 3) for y in x.weights] for x in net]  #  + [list(n2.neg_reference_left)]
        # print(cur_weights)
        weights_history.append(deepcopy(cur_weights))
        weights_history[-1].append(cur_x)
        weights_history[-1].append([cur_y])
        # обновляем фифо с правильными ответами
        y_true.append(cur_y)
        y_true.pop(0)
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
        # if counter > 10:
        #     continue
        for n in net:
            n.get_right_reference()
        for n in net:
            n.calculate_wanting(y_true=y_true)
        # for n in net:
        #     n.calculate_pos_neg_reference()
        for n in net:
            n.move_weights()

print(y)
print(res)
for n in net:
    print(n.weights)
# # print(weights_history)
with open('net_log1.csv', 'w') as f:
    for i in weights_history:
        res = []
        for j in i:
            for k in j:
                res.append(str(k))
        # print(','.join(res), file=f)
        print(','.join(res))
