# -*- coding: utf-8 -*-
import datetime

import numpy as np
from itertools import combinations_with_replacement, permutations


a = [0, 1, 0, 1]
b = [0, 0, 1, 1]

c = [0, 1, 1, 0]


# набор весов для конкретного количества нейронов
def make_weights_combinations(n):
    result = [1 / i for i in range(1, n + 1)]
    result += [0]
    result += [-1 / i for i in range(n, 0, -1)]
    return result


# оставляем только те комбинации весов, которые могут дать единицы
def filter_weights(weights):
    new_weights = []
    # TODO: избавиться от цикла
    for w in weights:
        temp = np.where(w <= 0, True, False)
        if temp.all():
            pass
        else:
            new_weights.append(w)
    return np.array(new_weights)


def filter_tuple_weights(weight):
    temp = np.array(weight)
    temp = np.where(temp <= 0, True, False)
    if temp.all():
        return False
    else:
        return True


# все возможные комбинации весов друг с другом
def all_possible_combinations(*args, weights):
    """
    :param args: перечисление входных историй
    :param weights: набор всевозможных весов
    :return: всевозможные комбинации весов
    """
    all_weights = permutations(weights, len(args))
    all_weights = np.array(list(all_weights))
    print(len(all_weights))
    # добавка комбинаций, которые не дает permutations
    for w in weights:
        all_weights = np.concatenate((all_weights, np.array([[w for _ in range(len(args))]])), axis=0)
    all_weights = filter_weights(all_weights)
    return all_weights


def generate_next_layer(*args, combinations):
    result = []
    # TODO: избавиться от цикла
    for c in combinations:
        c = c.reshape((len(c), 1))
        temp = c * np.array(args)
        temp = temp.sum(axis=0)
        temp = np.where(temp >= 1, 1, 0)
        if temp.any():
            result.append(tuple(temp))
    return np.array(list(set(result)))


def go_deeper_forward(*args):
    print(f'tp1, {datetime.datetime.now()}')
    mwc = make_weights_combinations(len(args))
    print(f'tp2, {datetime.datetime.now()}')
    apc = all_possible_combinations(*args, weights=mwc)
    print(f'tp3, {datetime.datetime.now()}')
    layer = generate_next_layer(*args, combinations=apc)
    print(f'tp4, {datetime.datetime.now()}')
    return layer


# p = go_deeper_forward(a, b)
# print(p)
# print('*****')
# p = go_deeper_forward(*p)
# print(p)
# print('*****')

import random
from itertools import product
from pprint import pprint
from random import random as randrand


def truth_table(inputs_count: int):
    """
    made by Vasyan
    Возвращает массивы входов вида [(x1,..., xn)] и выходов [y].
    Гарантируется, что при нулях на входе будет ноль на выходе;
    кроме того, число нулей в таблице истинности будет равно числу единиц в ответах.
    """
    X, y = [], []
    zero_count, ones_count = 0, 0
    for el in range(2 ** inputs_count):
        # получаем бинарное представление элемента
        el_binary = bin(el)[2:]
        # добавляем столько незначащих нулей слева,
        # чтобы длина была равна числу входных сигналов
        if len(el_binary) != inputs_count:
            el_binary = '0' * (inputs_count - len(el_binary)) + el_binary
        # переводим строки в числа и записываем в tuple
        el_binary = tuple(map(int, el_binary))
        # случайным образом назначаем правильный ответ
        X.append(el_binary)
        y.append(int(randrand() > 0.5))
        # принудительно ставим ответ в ноль, если на входах ноль
        y[-1] = 0 if sum(X[-1]) == 0 else y[-1]
        # подсчитываем число нулей и единиц
        zero_count = zero_count + 1 if y[-1] == 0 else zero_count
        ones_count = ones_count + 1 if y[-1] == 1 else ones_count
        # если перевалили за порог, меняем последнее значение на противоположное
        if zero_count > 2 ** (inputs_count - 1):
            y[-1] = 1
        elif ones_count > 2 ** (inputs_count - 1):
            y[-1] = 0
    return np.array(X), np.array(y)


x, y = truth_table(3)
print('X:')
print(x)
print('\n')
print('y:')
print(y)
print('\n')


# net = {'0': {'number': 0, 'input_numbers': [], 'input_weights': [], 'history': a},
#        '1': {'number': 1, 'input_numbers': [], 'input_weights': [], 'history': b}}
net = {}

for i in range(x.shape[1]):
    net[str(i)] = {'number': i, 'input_numbers': [], 'input_weights': [], 'history': x[:, i]}

c = y

print(x.shape)


prev_accuracy = 0.3

while True:

    random_count = random.randint(2, len(net))
    mwc = make_weights_combinations(random_count)
    # print(random_count)
    # print(mwc)

    temp = None
    while True:
        # perm_generator = product(mwc, repeat=random_count)
        max_number = len(mwc) ** random_count
        # rand_number = random.randint(0, max_number - 1)
        weight_combination = None
        rand_neurons = None
        # for en, i in enumerate(perm_generator):
        #     if en == rand_number and filter_tuple_weights(i):
        #         weight_combination = i
        #         break
        count = 0
        while True:
            temp = [random.choice(mwc) for x in range(random_count)]
            if filter_tuple_weights(temp):
                weight_combination = temp
                break
            count += 1
            if count >= max_number:
                break
        if not weight_combination:
            continue
        rand_neurons = random.sample(net.keys(), random_count)

        weight_combination = np.array(weight_combination)
        weight_combination = weight_combination.reshape((len(weight_combination), 1))

        np_neurons = np.array([net[x]['history'] for x in rand_neurons])
        all_np_neurons = np.array([net[x]['history'] for x in list(net.keys())])

        temp = weight_combination * np_neurons
        temp = temp.sum(axis=0)
        temp = np.where(temp >= 1, 1, 0)
        if temp.any() and not (temp == all_np_neurons).all(axis=1).any():
            break
        else:
            continue
    # print(temp)
    current_accuracy = (temp == np.array(c)).sum() / len(c)
    print(current_accuracy)
    if current_accuracy >= prev_accuracy:
        last_number = int(list(net.keys())[-1])
        new_node = {'number': last_number + 1, 'input_numbers': rand_neurons,
                    'input_weights': weight_combination, 'history': temp}
        net[str(last_number + 1)] = new_node

        prev_accuracy += 0.05
        if current_accuracy >= 1.0:
            break
        continue
    else:
        continue


pprint(net)
