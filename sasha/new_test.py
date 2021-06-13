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


print(c)
