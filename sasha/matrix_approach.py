# -*- coding: utf-8 -*-
import numpy as np

# n_accumulator = np.zeros((5,))
# n_signals = np.zeros((5,))
# s_weights = np.array([[0, 0, 0, 0, 0],
#                       [0, 0, 0, 4, 0],
#                       [2, 3, 0, 0, 0],
#                       [0, 0, -8, 0, 0],
#                       [0, 0, 10, -2, 0]])
# s_signals = np.zeros((5, 5))
# s_real = np.array([[0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0],
#                    [1, 0, 0, 0, 0],
#                    [0, 1, 0, 0, 0],
#                    [0, 0, 1, 1, 0]])
#
# n_accumulator[0] = 30
# n_accumulator[1] = 15
#
# for _ in range(50):
#     n_signals = np.where(n_accumulator > 1, 1, n_signals)
#     temp = s_weights * s_real * n_signals
#     n_accumulator = np.sum(temp, axis=1)  # n_accumulator + temp_sum
#     if n_signals[4] > 0:
#         print('hop')
#     n_signals = np.zeros((5,))
#
# n_accumulator[0] = 30
# n_accumulator[1] = 15
#
# for _ in range(50):
#     n_signals = np.where(n_accumulator > 1, 1, n_signals)
#     temp = s_weights * s_real * n_signals
#     n_accumulator = np.sum(temp, axis=1)  # n_accumulator + temp_sum
#     if n_signals[4] > 0:
#         print('hop')
#     n_signals = np.zeros((5,))
#
# s_cd = np.zeros((5, 5))
# s_dw = np.zeros((5, 5))
# s_cd_input = np.random.rand(3, 5, 5) * 2 - 1
# s_cd_output = np.random.rand(3, 5, 5) * 2 - 1
#
# s_dw_input = np.random.rand(3, 5, 5) * 2 - 1
# s_dw_output = np.random.rand(3, 5, 5) * 2 - 1
#
# # расстояние до выхода, верно-неверно, еще байда
# n_parameters = np.random.rand(3, 5) * 2 - 1
#
# # считаем cd на все синапсы
# cd_input = np.transpose(np.transpose(s_cd_input, (0, 2, 1)) *
#                         n_parameters.reshape((3, 5, 1)), (0, 2, 1))
# cd_output = s_cd_output * n_parameters.reshape((3, 5, 1))
# s_cd = np.sum(cd_output, axis=0) + np.sum(cd_input, axis=0)
#
# # считаем dw на все синапсы
# dw_input = np.transpose(np.transpose(s_dw_input, (0, 2, 1)) *
#                         n_parameters.reshape((3, 5, 1)), (0, 2, 1))
# dw_output = s_dw_output * n_parameters.reshape((3, 5, 1))
# s_dw = np.sum(dw_output, axis=0) + np.sum(dw_input, axis=0)
# s_dw = s_dw * s_real
#
# # получение новых значений реальности s_real
# temp_cd_greater = np.where(s_cd > 0.8, 1, 0)
# temp_cd_less = np.where(s_cd < 0.2, 0, 1)
# s_real_inverse = s_real * (-1) + 1
# s_real = s_real + s_real_inverse * temp_cd_greater
# s_real = s_real * temp_cd_less
# # обнуляем веса удаленных синапсов
# s_weights = s_weights * s_real
#
# # получение новых весов s_weights
# s_weights = s_weights + s_dw * 0.5


class Net:
    def __init__(self, neurons_number=5, input_neurons=None, output_neuron=4, genome=None, parameters_number=3):
        self.neurons_number = neurons_number
        self.output_neuron = output_neuron
        if input_neurons is not None:
            self.input_neurons = input_neurons
        else:
            self.input_neurons = [0, 1]
        if genome is not None:
            self.genome = genome
            self.parameters_number = len(genome['cd']['input'])
            self.s_cd_input = None
            self.s_cd_output = None
            self.s_dw_input = None
            self.s_dw_output = None
            raise
        else:
            self.parameters_number = parameters_number
            self.s_cd_input = np.random.rand(self.parameters_number, self.neurons_number, self.neurons_number) * 2 - 1
            self.s_cd_output = np.random.rand(self.parameters_number, self.neurons_number, self.neurons_number) * 2 - 1
            self.s_dw_input = np.random.rand(self.parameters_number, self.neurons_number, self.neurons_number) * 2 - 1
            self.s_dw_output = np.random.rand(self.parameters_number, self.neurons_number, self.neurons_number) * 2 - 1
        self.n_accumulator = np.zeros((self.neurons_number,))
        self.n_signals = np.zeros((self.neurons_number,))
        self.s_weights = np.zeros((self.neurons_number, self.neurons_number))
        self.s_signals = np.zeros((self.neurons_number, self.neurons_number))
        self.s_real = np.zeros((self.neurons_number, self.neurons_number))
        self.s_cd = np.zeros((self.neurons_number, self.neurons_number))
        self.s_dw = np.zeros((self.neurons_number, self.neurons_number))
        # расстояние до входа, правильность*расстояние, история активаций
        self.n_parameters = np.zeros((self.parameters_number, self.neurons_number))  # * 2 - 1
        for n in self.input_neurons:
            self.n_parameters[0, n] = 0
        self.n_parameters[0, self.output_neuron] = 1

    def tick(self, reaction=0, learning=True):
        out_signal = 0
        self.n_signals = np.where(self.n_accumulator >= 1, 1, self.n_signals)
        temp = self.s_weights * self.s_real * self.n_signals
        self.n_accumulator = np.sum(temp, axis=1)  # n_accumulator + temp_sum
        if self.n_signals[self.output_neuron] > 0:
            out_signal = 1
        self.n_signals = np.zeros((self.neurons_number,))
        if learning:
            self.compute_cd()
            self.compute_dw()
            self.check_real()
            self.move_weights()
        return out_signal

    def compute_cd(self):
        cd_input = np.transpose(np.transpose(self.s_cd_input, (0, 2, 1)) *
                                self.n_parameters.reshape((self.parameters_number, self.neurons_number, 1)), (0, 2, 1))
        cd_output = self.s_cd_output * self.n_parameters.reshape((self.parameters_number, self.neurons_number, 1))
        self.s_cd = np.sum(cd_output, axis=0) + np.sum(cd_input, axis=0)

    def compute_dw(self):
        dw_input = np.transpose(np.transpose(self.s_dw_input, (0, 2, 1)) *
                                self.n_parameters.reshape((self.parameters_number, self.neurons_number, 1)), (0, 2, 1))
        dw_output = self.s_dw_output * self.n_parameters.reshape((self.parameters_number, self.neurons_number, 1))
        self.s_dw = np.sum(dw_output, axis=0) + np.sum(dw_input, axis=0)
        self.s_dw = self.s_dw * self.s_real

    def check_real(self):
        # получение новых значений реальности s_real
        temp_cd_greater = np.where(self.s_cd > 0.8, 1, 0)
        temp_cd_less = np.where(self.s_cd < 0.2, 0, 1)
        s_real_inverse = self.s_real * (-1) + 1
        self.s_real = self.s_real + s_real_inverse * temp_cd_greater
        self.s_real = self.s_real * temp_cd_less

    def move_weights(self):
        # получение новых весов s_weights
        self.s_weights = self.s_weights + self.s_dw * 0.05
        # обнуляем веса удаленных синапсов
        self.s_weights = self.s_weights * self.s_real

    def _make_synapse_real(self, n1, n2, weight=0.5):
        self.s_real[n2, n1] = 1
        self.s_weights[n2, n1] = weight

    def probe(self, number):
        self.n_accumulator[number] = 1

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
            out = self.tick(learning=learning, reaction=y)
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


test()
