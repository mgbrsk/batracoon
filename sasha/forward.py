# -*- coding: utf-8 -*-
import numpy as np


class Net:
    """
    Класс стандартной сети прямого распространения
    """
    def __init__(self, neurons_number: int = 5, input_neurons: list = None, output_neuron: int = 4,
                 buffer_size: int = 10):
        """
        Инициализация сети прямого распространения. В параметрах префикс s_ - отношения к синапсам, n_ - нейронам
        :param neurons_number: количество нейронов сети
        :param input_neurons: список, номера входных нейронов
        :param output_neuron: номер выходного нейрона
        :param buffer_size: длина буфера для запоминания историй активаций нейронов
        """
        self.neurons_number = neurons_number  # количество нейронов
        self.output_neuron = output_neuron  # какой нейрон выходной
        if input_neurons is not None:  # если переданы входные нейроны
            self.input_neurons = input_neurons  # записываем их
        else:
            self.input_neurons = [0, 1]  # иначе по умолчанию 0 и 1
        # аккумулятор, записываются суммы весов, помноженные на входные сигналы
        self.n_accumulator = np.zeros((self.neurons_number,))
        # история активаций, не надо здесь
        self.n_activation_history = np.zeros((buffer_size, self.neurons_number))
        # есть ли сигнал в нейронах
        self.n_signals = np.zeros((self.neurons_number,))
        # матрица весов
        self.s_weights = np.zeros((self.neurons_number, self.neurons_number))
        # матрица наличия сигнала в синапсах
        self.s_signals = np.zeros((self.neurons_number, self.neurons_number))
        # матрица существования синапса
        self.s_real = np.zeros((self.neurons_number, self.neurons_number))

    def tick(self, **kwargs):
        """
        Метод обсчета сети за один момент времени
        :param kwargs: на будущее
        :return: активировался ли нейрон, история состояний
        """
        history = {'weights': None, 'accumulators': None, 'neuron_signals': None, 'synapse_real': None}
        out_signal = 0
        self.n_signals = np.where(self.n_accumulator >= 1.0, 1.0, self.n_signals)
        history['weights'] = self.s_weights
        history['accumulators'] = self.n_accumulator
        history['neuron_signals'] = self.n_signals
        history['synapse_real'] = self.s_real
        temp = self.s_weights * self.s_real * self.n_signals
        self.n_accumulator = np.sum(temp, axis=1)  # n_accumulator + temp_sum
        # self.random_activation(0.05)
        if self.n_signals[self.output_neuron] > 0.0:
            out_signal = 1.0
        self.n_signals = np.zeros((self.neurons_number,))
        return out_signal, history

    def random_activation(self, probability):
        """
        Метод для случайной активации нейронов, пока не надо
        :param probability: вероятность, от 0 до 1
        :return:
        """
        r_a_matrix = np.random.rand(self.neurons_number)
        synapse_numbers = np.sum(self.s_real, axis=0) + np.sum(self.s_real, axis=1)
        synapse_numbers = synapse_numbers + 1.0
        r_a_matrix = r_a_matrix / synapse_numbers
        self.n_signals = np.where(r_a_matrix < probability, 1.0, self.n_signals)

    # метод расчета одновременной активации нейронов, пока не надо
    def compute_interaction(self):
        interaction_matrix = np.dot(self.n_activation_history.T, self.n_activation_history)
        np.fill_diagonal(interaction_matrix, 0.0)

    def _make_synapse_real(self, n1, n2, weight=0.5):
        """
        Служебный метод создания синапсов
        :param n1: выходной нейрон, номер
        :param n2: входной нейрон, номер
        :param weight: вес связи
        :return:
        """
        self.s_real[n2, n1] = 1.0
        self.s_weights[n2, n1] = weight

    # метод засылки сигнала в произвольный нейрон
    def probe(self, number):
        self.n_accumulator[number] = 1.0

    # метод одновременной активации нескольких нейронов
    def massive_probe(self, array):
        for i in range(len(array)):
            if array[i] == 1:
                self.probe(i)

    # метод расчета выхода по входам
    def predict(self, x, y, learning=True, return_history=False):
        all_history = []
        if len(x) != len(y):
            raise
        result = []
        for cur_x, cur_y in zip(x, y):
            self.massive_probe(cur_x)
            out, history = self.tick(learning=learning, reaction=cur_y)
            all_history.append(history)
            result.append(out)
        if return_history:
            return result, all_history
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
    res = net.predict(x, y, learning=False, return_history=False)
    assert res == [0, 0, 0, 1, 0, 0, 0, 0]
    print('Forward test is good.')


# test()
