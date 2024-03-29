# -*- coding: utf-8 -*-
import random
from pprint import pprint
from random import random as randrand

import numpy as np


# набор весов для конкретного количества нейронов
def make_weights_combinations(n):
    result = [1 / i for i in range(1, n + 1)]
    result += [0]
    result += [-1 / i for i in range(n, 0, -1)]
    return result


# проверка на наличие хотя бы одного положительного веса
def filter_tuple_weights(weight):
    temp = np.array(weight)
    temp = np.where(temp <= 0, True, False)
    if temp.all():
        return False
    else:
        return True


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


# создаем таблицу истинности
x, y = truth_table(3)
print('X:')
print(x)
print('\n')
print('y:')
print(y)
print('\n')


# словарь, в который будем сохранять все узлы сети
net = {}

# из таблицы истинности создаем входные нейроны с заданной историей (колонки Х)
for i in range(x.shape[1]):
    net[str(i)] = {'number': i, 'input_numbers': [], 'input_weights': [], 'history': x[:, i]}

print(x.shape)

# начальное качество, которое будем требовать с новых узлов
prev_accuracy = 0.2

# пока не получили требуемое качество сети
while True:
    # пока не получим ненулевую новую историю активаций узла
    while True:
        # берем рандомное число (два или более), со сколькими входными узлами будем работать
        random_count = random.randint(2, len(net))
        # получаем набор осмысленных весов для данного числа узлов
        mwc = make_weights_combinations(random_count)
        # инициализурем чтобы линтер не ругался
        temp = None
        # максимальное число комбинаций весов
        max_number = len(mwc) ** random_count
        # инициализация чтобы не ругался линтер
        weight_combination = None
        rand_neurons = None
        count = 0
        # пока не получим норм комбинацию весов (которая не будет давать в любом случае нули)
        while True:
            # собираем рандомную комбинацию весов из доступных вариантов
            temp = [random.choice(mwc) for x in range(random_count)]
            # если эта комбинация не вся состоит из нулевых и отрицательных весов
            if filter_tuple_weights(temp):
                # у нас есть новая норм комбинация входных весов!
                weight_combination = temp
                break
            # если не повезло то увеличиваем счетчик
            count += 1
            # и если таки мы сидим в этом цикле больше раз чем всего комбинаций есть, то выходим из цикла с пустым weight_combination
            if count >= max_number:
                break
        # если так ничего не нашли, то еще раз шерстим случайные комбинации весов
        if not weight_combination:
            continue

        # берем random_count узлов из сети
        rand_neurons = random.sample(net.keys(), random_count)

        # преобразуем комбинацию входных весов в массив нампи
        weight_combination = np.array(weight_combination)
        # создаем матрицу-вектор (вертикальную)
        weight_combination = weight_combination.reshape((len(weight_combination), 1))

        # создаем матрицу входных историй активаций для данного узла
        np_neurons = np.array([net[x]['history'] for x in rand_neurons])
        # а также переводим все истории в матричный вид нампая
        all_np_neurons = np.array([net[x]['history'] for x in list(net.keys())])

        # получаем результат для данного узла - умножаем истории на веса, суммируем и применяем пороговую функцию
        temp = weight_combination * np_neurons
        temp = temp.sum(axis=0)
        temp = np.where(temp >= 1, 1, 0)
        # если получилась ненулевая история и ее еще нет в уже имеющихся историях
        if temp.any() and not (temp == all_np_neurons).all(axis=1).any():
            # выходим из цикла и идем дальше
            break
        else:
            # иначе выбираем заново веса и нейроны
            continue

    # считаем текущее качество последнего узла
    current_accuracy = (temp == np.array(y)).sum() / len(y)
    print(current_accuracy, prev_accuracy)
    # если качество больше целевого
    if current_accuracy >= prev_accuracy:
        # записываем новый получившийся узел в сеть
        last_number = int(list(net.keys())[-1])
        new_node = {'number': last_number + 1, 'input_numbers': rand_neurons,
                    'input_weights': weight_combination, 'history': temp}
        net[str(last_number + 1)] = new_node
        # увеличиваем целевое качество
        prev_accuracy += 0.025
        # если уже получили стопроцентное попадание на последнем узле
        if current_accuracy >= 1.0:
            # выходим из главного цикла, закончили
            break
        # если недостаточное качество то идем на еще один круг
        continue
    # если качество хуже целевого
    else:
        # заново выбираем количество входных узлов, веса, и входные нейроны
        continue


pprint(net)
