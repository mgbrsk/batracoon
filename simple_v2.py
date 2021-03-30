# -*- coding: utf-8 -*-

from sklearn.linear_model import LinearRegression


# линейный регрессор для обучения нейронов
LR = LinearRegression(fit_intercept=False)


# класс синапсов
class Synapse:
    def __init__(self, input_n, output_n, weight=0.5, distance=1):
        self.input_n = input_n  # номер входного нейрона
        self.output_n = output_n  # номер выходного нейрона
        self.weight = weight  # вес
        self.is_signal = 0  # есть ли в нем сигнал

    # TODO: для учета расстояний
    def tick_signal(self):
        pass

    # проверка есть ли сигнал в синапсе и сразу зануляем
    def check_signal(self):
        if self.is_signal == 0:
            return False
        else:
            self.is_signal = 0
            return True

    # метод активации синапса
    def activate(self):
        self.is_signal = 1


# класс нейрона
class Neuron:
    def __init__(self, number):
        self.number = number  # уникальный номер нейрона
        self.lr = LR  # линейный регрессор для обучения
        self.input_synapses = []  # входные синапсы
        self.output_synapses = []  # выходные синапсы
        # буферы для обучения
        self.input_buffer = [[] for _ in range(BUFFER_SIZE)]  # буфер входов
        self.output_buffer = [0 for _ in range(BUFFER_SIZE)]  # буфер выхода
        self.true_output_buffer = [0 for _ in range(BUFFER_SIZE)]  # какой выход должен был быть на самом деле
        self.weight_buffer = [[] for _ in range(BUFFER_SIZE)]  # буфер весов
        ###
        self.accumulator = 0  # аккумулятор для суммирования и проверки преодоления трэшхолда
        self.costil = 0  # костыль для заполнения буферов, просто счетчик

    # метод получения входных весов как массива
    def get_weights(self):
        res = []
        for s in self.input_synapses:
            res.append(s.weight)
        return res

    # добавление входного синапса
    def add_input_synapse(self, synapse):
        self.input_synapses.append(synapse)

    # добавление выходного синапса
    def add_output_synapse(self, synapse):
        self.output_synapses.append(synapse)

    # метод суммирования сигналов из входных синапсов
    def sum_signals(self):
        current_input = []
        current_weights = []
        for i_s in self.input_synapses:  # идем по всем входным синапсам
            current_weights.append(i_s.weight)  # сохраняем вес для массива
            if i_s.check_signal():  # если в синапсе есть сигнал (этот метод и сбрасывает этот сигнал в синапсе)
                self.accumulator += i_s.weight  # в аккумулятор добавляем значение веса этого синапса
                current_input.append(1)  # для истории сохраняем что на входе было, 1 или 0
            else:
                current_input.append(0)
        # записываем что собралось в буферы
        self.input_buffer.append(current_input)  # записываем в конец
        self.input_buffer.pop(0)  # и убираем первую запись
        self.weight_buffer.append(current_weights)
        self.weight_buffer.pop(0)

    # метод проверки активации нейронов и запускания сигнала в выходные синапсы
    def generate_output(self):
        if self.accumulator >= 1:  # если преодолели порог
            for o_s in self.output_synapses:  # идем по всем выходным синапсам
                o_s.activate()  # и активируем их
            self.accumulator = 0  # обнуляем аккумулятор
            self.output_buffer.append(1)  # записываем выход данного нейрона в буфер
            self.output_buffer.pop(0)
            return self.number  # если активировался нейрон, то возвращаем его номер
        self.accumulator = 0
        self.output_buffer.append(0)  # так записываем в буферы если не преодолели порог
        self.output_buffer.pop(0)
        return False

    # записываем в буфер что должен был выдать нейрон
    def set_true_output(self, true_output):
        self.true_output_buffer.append(true_output)
        self.true_output_buffer.pop(0)
        
    # TODO: check this implementation
    def update_buffer(self, buffer, value):
        buffer.append(value)
        buffer.pop(0)

    # метод обучения нейрона, reaction - реакция сети, всем нейронам одинаковая, 1 если все угадали, -1 если лажают
    def learning(self, reaction):
        # -1 или 1 на входе
        if reaction == 1:  # если все ок, нейрон угадал
            last_output = self.output_buffer[-1]  # записываем в буфер действительных ответов то что мы и ответили
        elif reaction == -1:  # если не ок
            # меняем 1 на 0 и 0 на 1
            last_output = self.output_buffer[-1] * (-1) + 1
        self.set_true_output(last_output)  # ну и сама запись в буфер

        if self.costil >= BUFFER_SIZE:  # если наши буферы заполнились
            # print(self.number)
            # print(self.input_buffer, self.true_output_buffer)
            self.lr.fit(self.input_buffer, self.true_output_buffer)  # обучаем регрессор на входных и правильных выходных данных
            need_weights = list(self.lr.coef_)  # вытаскиваем полученные коэффициенты
            for i, s in enumerate(self.input_synapses):  # идем по всем входным синапсам
                s.weight = need_weights[i]  # и записываем в них что наобучали
        else:
            self.costil += 1  # увеличиваем счетчик если буферы еще не заполнились


# класс самой сети
class Net:
    # neurons_number - сколько нейронов должно быть
    def __init__(self, neurons_number):
        self.neurons = []  # нейроны
        self.synapses = []  # синапсы
        self.fitted = False  # обучились уже или нет
        for i in range(neurons_number):  # создаем необходимое количество нейронов
            self.neurons.append(Neuron(i))

    # метод поиска нужного нейрона, возвращает экземпляр найденнного нейрона
    def get_neuron(self, num):
        for n in self.neurons:
            if n.number == num:
                return n
        else:
            raise

    # метод добавления нового синапса
    def add_synapse(self, n_input, n_output, weight=0.5):
        '''
        :param n_input: от какого нейрона должен придти сигнал
        :param n_output: в какой нейрон этот сигнал должен придти
        :param weight: вес данной связи
        :return:
        '''
        s = Synapse(n_input, n_output, weight=weight)  # создаем экземпляр синапса
        self.synapses.append(s)  # добавляем его в общую кучу ко всем синапсам
        output_neuron = self.get_neuron(n_input)  # находим нужные нейроны
        input_neuron = self.get_neuron(n_output)
        input_neuron.add_input_synapse(s)  # и добавляем в них ссылки на их синапсы, входные и выходные
        output_neuron.add_output_synapse(s)

    # метод одного тика сети
    def tick(self):
        out_signal = False  # сигнала с выхода сети пока нет
        for n in self.neurons:  # во всех нейронах суммируем входные сигналы с синапсов, и зануляем что в них было
            n.sum_signals()
        for n in self.neurons:  # в этой части генерируем сигналы из нейронов в синапсы
            p = n.generate_output()  # здесь возвращается номер если данный нейрон активировался
            if p == 4:  # если это был выход (магическое число 4, да да)
                out_signal = True  # значит был сигнал на выходе сети
                # print('hop')
        return out_signal  # возвращаем, был ли сигнал на выходе сети в этом тике
        # set_true_output

    # метод ручного ввода импульса в нейрон
    def probe(self, number):
        n = self.get_neuron(number)
        n.accumulator = 1

    # метод ввода импульсов в несколько нейронов на одном тике, индекс числа во входном массиве - номер активируемого нейрона
    def massive_probe(self, array):
        for i in range(len(array)):
            if array[i] != 0:
                self.probe(i)

    # проверка верности ответа сети и генерация реакциии сети для ее засылки во все нейроны
    def check_right(self, y_true, y_pred):
        if y_true == y_pred:
            return 1
        else:
            return -1

    # метод обучения сети
    def fit(self, X, y):
        # если длины не совпадают то сам баклан
        if len(X) != len(y):
            raise
        for cur_x, cur_y in zip(X, y):  # идем по входным и выходным обучающим данным
            self.massive_probe(cur_x)  # входные данные идут на запись во входные нейроны
            res = self.tick()  # сеть тикает и генерирует ответ - был или не было сигнала на этом тике
            checked_out = self.check_right(cur_y, int(res))  # проверка правильности ответа
            for n in self.neurons:
                if n.number != 0 and n.number != 1:
                    n.learning(checked_out)  # передаем реакцию сети во все нейроны, где они обучаются глядя на нее
        self.fitted = True  # всё, обучили
                
    def predict(self, X):
        # если еще не обучили то иди гуляй
        if not self.fitted:
            raise
        result = []
        for cur_x in X:  # идем по входным данным
            self.massive_probe(cur_x)  # тут все аналогично как в обучении, только без вызова обучения нейронов
            res = self.tick()
            result.append(res)
        return result
            

# net = Net(5)
# net.add_synapse(0, 2)
# net.add_synapse(1, 2)
# net.add_synapse(2, 4, weight=1)
# net.probe(0)
# net.probe(1)
# for i in range(50):
#     net.tick()
# net.probe(0)
# net.probe(1)
# for i in range(50):
#     net.tick()

import random


BUFFER_SIZE = 180  # длина буферов в тиках
EPOCHS = 750  # количество импульсов
SKVAZHNOST = 30  # продолжительность импульса в тиках


train = [{'X': [0, 0], 'y': 0},
         {'X': [0, 1], 'y': 1},
         {'X': [1, 0], 'y': 1},
         {'X': [1, 1], 'y': 0}]

train_X = []
train_y = []
for _ in range(EPOCHS):
    j = random.randint(0, 3)
    for i in range(SKVAZHNOST):
        train_X.append(train[j]['X'])
        train_y.append(train[j]['y'])


def random_weight():
    return random.random() * 2 - 1


def testtest():

    net = Net(6)

    # net.add_synapse(0, 1, weight=random_weight())
    # net.add_synapse(0, 2, weight=random_weight())
    # net.add_synapse(0, 3, weight=random_weight())
    # net.add_synapse(0, 4, weight=random_weight())
    # net.add_synapse(0, 5, weight=random_weight())
    # net.add_synapse(1, 2, weight=random_weight())
    # net.add_synapse(1, 3, weight=random_weight())
    # net.add_synapse(1, 4, weight=random_weight())
    # net.add_synapse(1, 5, weight=random_weight())
    # net.add_synapse(2, 3, weight=random_weight())
    # net.add_synapse(2, 4, weight=random_weight())
    # net.add_synapse(2, 5, weight=random_weight())
    # net.add_synapse(3, 4, weight=random_weight())
    # net.add_synapse(3, 5, weight=random_weight())
    # net.add_synapse(4, 5, weight=random_weight())
    #
    # net.add_synapse(1, 0, weight=random_weight())
    # net.add_synapse(2, 0, weight=random_weight())
    # net.add_synapse(3, 0, weight=random_weight())
    # net.add_synapse(4, 0, weight=random_weight())
    # net.add_synapse(5, 0, weight=random_weight())
    # net.add_synapse(2, 1, weight=random_weight())
    # net.add_synapse(3, 1, weight=random_weight())
    # net.add_synapse(4, 1, weight=random_weight())
    # net.add_synapse(5, 1, weight=random_weight())
    # net.add_synapse(3, 2, weight=random_weight())
    # net.add_synapse(4, 2, weight=random_weight())
    # net.add_synapse(5, 2, weight=random_weight())
    # net.add_synapse(4, 3, weight=random_weight())
    # net.add_synapse(5, 3, weight=random_weight())
    # net.add_synapse(5, 4, weight=random_weight())
    net.add_synapse(0, 2, weight=random_weight())
    net.add_synapse(1, 3, weight=random_weight())
    net.add_synapse(2, 5, weight=random_weight())
    net.add_synapse(3, 5, weight=random_weight())
    net.add_synapse(5, 4, weight=random_weight())
    net.add_synapse(2, 4, weight=random_weight())
    net.add_synapse(3, 4, weight=random_weight())


    net.fit(train_X, train_y)

    test_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    test_X = test_X * SKVAZHNOST
    test_X.sort()

    y_true = list(map(lambda x: sum(x) == 1, test_X))

    y_pred = net.predict(test_X)

    counter_good = 0
    counter_all = len(y_true)
    for i, j, k in zip(test_X, y_pred, y_true):
        # print(i)
        # print(j)
        # print('*********')
        if j == k:
            counter_good += 1

    # print(' ')
    print(f'{counter_good}/{counter_all}')


for _ in range(10):
    testtest()
