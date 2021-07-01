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


def split_x(x_train, number_of_splits=2):
    length = x_train.shape[0]
    parts = length // number_of_splits
    last = length % number_of_splits
    print(parts * number_of_splits + last == length)
    result = []
    for i in range(parts):
        first_index = i * number_of_splits
        second_index = (i + 1) * number_of_splits
        temp = x_train[first_index:second_index, :]
        result.append(temp)
    if last:
        result.append(x_train[second_index:, :])
    return result


# создаем таблицу истинности
x, y = truth_table(2)
print('X:')
print(x)
print('\n')
print('y:')
print(y)
print('\n')



x_train = split_x(x, number_of_splits=2)
for i in x_train:
    print(i)



# # словарь, в который будем сохранять все узлы сети
# net = {}
#
# # из таблицы истинности создаем входные нейроны с заданной историей (колонки Х)
# for i in range(x.shape[1]):
#     net[str(i)] = {'number': i, 'input_numbers': [], 'input_weights': [], 'history': x[:, i]}
#
# print(x.shape)
#
# # начальное качество, которое будем требовать с новых узлов
# prev_accuracy = 0.2
#
# # пока не получили требуемое качество сети
# while True:
#     # пока не получим ненулевую новую историю активаций узла
#     while True:
#         # берем рандомное число (два или более), со сколькими входными узлами будем работать
#         random_count = random.randint(2, len(net))
#         # получаем набор осмысленных весов для данного числа узлов
#         mwc = make_weights_combinations(random_count)
#         # инициализурем чтобы линтер не ругался
#         temp = None
#         # максимальное число комбинаций весов
#         max_number = len(mwc) ** random_count
#         # инициализация чтобы не ругался линтер
#         weight_combination = None
#         rand_neurons = None
#         count = 0
#         # пока не получим норм комбинацию весов (которая не будет давать в любом случае нули)
#         while True:
#             # собираем рандомную комбинацию весов из доступных вариантов
#             temp = [random.choice(mwc) for x in range(random_count)]
#             # если эта комбинация не вся состоит из нулевых и отрицательных весов
#             if filter_tuple_weights(temp):
#                 # у нас есть новая норм комбинация входных весов!
#                 weight_combination = temp
#                 break
#             # если не повезло то увеличиваем счетчик
#             count += 1
#             # и если таки мы сидим в этом цикле больше раз чем всего комбинаций есть, то выходим из цикла с пустым weight_combination
#             if count >= max_number:
#                 break
#         # если так ничего не нашли, то еще раз шерстим случайные комбинации весов
#         if not weight_combination:
#             continue
#
#         # берем random_count узлов из сети
#         rand_neurons = random.sample(net.keys(), random_count)
#
#         # преобразуем комбинацию входных весов в массив нампи
#         weight_combination = np.array(weight_combination)
#         # создаем матрицу-вектор (вертикальную)
#         weight_combination = weight_combination.reshape((len(weight_combination), 1))
#
#         # создаем матрицу входных историй активаций для данного узла
#         np_neurons = np.array([net[x]['history'] for x in rand_neurons])
#         # а также переводим все истории в матричный вид нампая
#         all_np_neurons = np.array([net[x]['history'] for x in list(net.keys())])
#
#         # получаем результат для данного узла - умножаем истории на веса, суммируем и применяем пороговую функцию
#         temp = weight_combination * np_neurons
#         temp = temp.sum(axis=0)
#         temp = np.where(temp >= 1, 1, 0)
#         # если получилась ненулевая история и ее еще нет в уже имеющихся историях
#         if temp.any() and not (temp == all_np_neurons).all(axis=1).any():
#             # выходим из цикла и идем дальше
#             break
#         else:
#             # иначе выбираем заново веса и нейроны
#             continue
#
#     # считаем текущее качество последнего узла
#     current_accuracy = (temp == np.array(y)).sum() / len(y)
#     print(current_accuracy, prev_accuracy)
#     # если качество больше целевого
#     if current_accuracy >= prev_accuracy:
#         # записываем новый получившийся узел в сеть
#         last_number = int(list(net.keys())[-1])
#         new_node = {'number': last_number + 1, 'input_numbers': rand_neurons,
#                     'input_weights': weight_combination, 'history': temp}
#         net[str(last_number + 1)] = new_node
#         # увеличиваем целевое качество
#         prev_accuracy += 0.025
#         # если уже получили стопроцентное попадание на последнем узле
#         if current_accuracy >= 1.0:
#             # выходим из главного цикла, закончили
#             break
#         # если недостаточное качество то идем на еще один круг
#         continue
#     # если качество хуже целевого
#     else:
#         # заново выбираем количество входных узлов, веса, и входные нейроны
#         continue
#
#
# pprint(net)


# class Node:
#     def __init__(self, number, input_nodes=None, weights=None, history=None, is_output=False):
#         pass
#
#     def append_link(self):
#         pass
#
#     def append_node(self):
#         pass
#
#     def forward_move(self):
#         pass



# target_accuracy = 0.25
# current_accuracy = 0.0
#
# while True:  # current_accuracy < 0.95
#     while True:  # выбираем вставка или соединение
#         if random.random() > 0.5:
#             # выбрали соединение
#             counter_t1 = 0
#             while True:  # выбираем два рандомных узла
#                 if counter_t1 > len(list(net.keys())):
#                     break
#                 temp_in, temp_out = random.sample(net.keys(), 2)
#                 if temp_in['is_output'] or temp_out['is_input']:
#                     counter_t1 += 1
#                     continue
#         else:
#             # выбрали вставку
#             pass
#
#         break
#
#     if current_accuracy > 0.95:
#         break


def limit_counter(limit):
    def outer_wrapper(f):
        def wrapper(*args, **kwargs):
            # if f.__name__ + '_counter' not in args[0].__dir__():
            #     args[0].getattr(f.__name__ + '_counter') = 0
            # print(f.__name__ in args[0].__dir__())
            res = f(*args, **kwargs)
            return res
        return wrapper
    return outer_wrapper


class System:
    def __init__(self):
        self.state = None
        self.current_accuracy = 0.0
        self.counters = {}

        def get_last(net):
            numbers = list(net.keys())
            numbers = map(lambda x: int(x), numbers)
            return max(numbers)

        self.net = {}

        for i in range(x.shape[1]):
            temp_node = dict(number=i, output_nodes=[], input_nodes=[], weights=[],
                             counter_input=0, history=[], is_output=False, is_input=True)
            self.net[str(i)] = temp_node

        last_number = get_last(self.net)
        self.net[str(last_number + 1)] = dict(number=last_number + 1, output_nodes=[], input_nodes=[], weights=[],
                                              counter_input=0, history=[], is_output=True, is_input=False)

        # pprint(self.net)

    # @limit_counter(5)
    def choice_insertion_type(self):
        if random.random() > 0.5:
            self.state = 'conn_get_two_nodes'
        else:
            self.state = 'ins_get_random_number'

    def check_exist_cnt(self, name):
        if name not in list(self.counters.keys()):
            self.counters[name] = 0

    def get_two_nodes(self):
        self.check_exist_cnt('conn_get_two_nodes_cnt')
        if self.counters['conn_get_two_nodes_cnt'] >= len(list(self.net.keys())):
            self.state = 'choice_insertion_type'
            self.counters['conn_get_two_nodes_cnt'] = 0
            return
        self.temp_in, self.temp_out = random.sample(self.net.keys(), 2)
        self.state = 'conn_check_in_out'
        self.counters['conn_get_two_nodes_cnt'] += 1

    def run(self):
        while True:
            if self.state is None:
                self.state = 'choice_insertion_type'
            m = getattr(self, self.state)
            m()
            print(self.__dir__())
            break


s = System()
s.run()
