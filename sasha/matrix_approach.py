# -*- coding: utf-8 -*-
import numpy as np


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
            self.s_cd_input = np.array(genome['cd']['input'])
            self.s_cd_output = np.array(genome['cd']['output'])
            self.s_dw_input = np.array(genome['dw']['input'])
            self.s_dw_output = np.array(genome['dw']['output'])
            raise
        else:
            self.parameters_number = parameters_number
            self.s_cd_input = np.random.rand(self.parameters_number) * 2 - 1
            self.s_cd_output = np.random.rand(self.parameters_number) * 2 - 1
            self.s_dw_input = np.random.rand(self.parameters_number) * 2 - 1
            self.s_dw_output = np.random.rand(self.parameters_number) * 2 - 1
        self.n_accumulator = np.zeros((self.neurons_number,))
        self.n_activation_history = np.zeros((self.neurons_number, BUFFER_SIZE))
        self.n_signals = np.zeros((self.neurons_number,))
        self.s_weights = np.zeros((self.neurons_number, self.neurons_number))
        self.s_signals = np.zeros((self.neurons_number, self.neurons_number))
        self.s_real = np.zeros((self.neurons_number, self.neurons_number))
        self.s_cd = np.zeros((self.neurons_number, self.neurons_number))
        self.s_dw = np.zeros((self.neurons_number, self.neurons_number))
        # расстояние до входа, правильность*расстояние, история активаций (mock)
        self.n_parameters = np.zeros((self.neurons_number, self.parameters_number))  # * 2 - 1
        for n in self.input_neurons:
            self.n_parameters[n, 0] = 0
        self.n_parameters[self.output_neuron, 0] = 1

    def tick(self, reaction=0, learning=True):
        out_signal = 0
        self.n_signals = np.where(self.n_accumulator >= 1, 1, self.n_signals)
        temp = self.s_weights * self.s_real * self.n_signals
        self.n_accumulator = np.sum(temp, axis=1)  # n_accumulator + temp_sum
        if self.n_signals[self.output_neuron] > 0:
            out_signal = 1
        if learning:
            self.update_parameters()  # reaction
            self.compute_cd()  # reaction
            self.compute_dw()  #
            self.check_real()
            self.move_weights()
        self.n_signals = np.zeros((self.neurons_number,))
        return out_signal

    def update_parameters(self):
        self.n_activation_history = np.roll(self.n_activation_history, shift=1, axis=0)
        self.n_activation_history[0] = self.n_signals
        pass

    def compute_cd(self):
        cd_input = np.sum(self.s_cd_input * self.n_parameters, axis=1)
        cd_output = np.sum(self.s_cd_output * self.n_parameters, axis=1)
        self.s_cd = cd_input + cd_output.reshape((self.neurons_number, 1))
        np.fill_diagonal(self.s_cd, 0)

    def compute_dw(self):
        dw_input = np.sum(self.s_dw_input * self.n_parameters, axis=1)
        dw_output = np.sum(self.s_dw_output * self.n_parameters, axis=1)
        self.s_dw = dw_input + dw_output.reshape((self.neurons_number, 1))
        np.fill_diagonal(self.s_dw, 0)

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


BUFFER_SIZE = 10
test()

a = np.zeros((BUFFER_SIZE, 3))
b = np.array([1, 0, 1])
a = np.array([[0, 0, 0],
              [1, 0, 0],
              [0, 1, 1],
              [1, 0, 0],
              [0, 1, 1],
              [1, 1, 1],
              [1, 0, 1]])
print(a)
a = np.roll(a, shift=1, axis=0)
print(a)
# c = np.dot(a.T, a)
# print(c)
# a = np.zeros((3, 2))
# print(a)
# a = np.array([0, 1, 2, 3, 4])
# b = np.array([7, 8, 9, 0, 1])
# print(a + b.reshape((5, 1)))


