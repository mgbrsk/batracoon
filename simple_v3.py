# -*- coding: utf-8 -*-

class Synapse:
    global_number = 0

    def __init__(self, input_n, output_n, weight=0.5, genome=None):
        self.number = Synapse.global_number
        Synapse.global_number += 1
        self.input_neuron = input_n
        self.output_neuron = output_n
        if genome:
            self.genome = genome
        else:
            self.genome = {'cd': {}, 'dw': {}}
        self.is_signal = False
        self.weight = weight  # вес
        self.input_vars = {}
        self.output_vars = {}
        self.is_real = False
        self.cd = 0
        self.dw = 0

    # проверка есть ли сигнал в синапсе и сразу зануляем
    def check_signal(self):
        if self.is_signal and self.is_real:
            self.is_signal = False
            return True
        else:
            return False

    # метод активации синапса
    def activate(self):
        if self.is_real:
            self.is_signal = True

    def get_vars(self):
        self.input_vars = self.input_neuron.get_params()
        self.output_vars = self.output_neuron.get_params()

    def calculate_cddw(self):
        self.cd = 0
        self.dw = 0
        for key, value in self.genome['cd'].items():
            if key in list(self.input_vars.keys()):
                self.cd += self.input_vars[key] * value
            elif key in list(self.output_vars.keys()):
                self.cd += self.output_vars[key] * value
        for key, value in self.genome['dw'].items():
            if key in list(self.input_vars.keys()):
                self.dw += self.input_vars[key] * value
            elif key in list(self.output_vars.keys()):
                self.dw += self.output_vars[key] * value

        self.cd = self.cd if self.cd <= 1 else 1
        self.cd = self.cd if self.cd >= 0 else 0
        self.dw = self.cd if self.cd <= 1 else 1
        self.dw = self.cd if self.cd >= -1 else -1

    def check_existing(self):
        if (not self.is_real) and self.cd >= 0.8:
            self.is_real = True
        elif self.is_real and self.cd <= 0.2:
            self.is_real = False
        return self.is_real

    def move_weight(self):
        if not self.is_real:
            return
        self.weight += self.dw * 0.05

    def add_signals(self):
        if self.check_signal():
            self.output_neuron.add_accumulator(self.weight)

    def check_input_signal(self):
        if not self.is_real:
            return
        if self.input_neuron.is_spiked():
            self.is_signal = True


class Neuron:
    global_number = 0

    def __init__(self):
        self.number = Neuron.global_number
        Neuron.global_number += 1
        self.accumulator = 0

    def get_params(self):
        pass

    def add_accumulator(self, value):
        self.accumulator += value

    def is_spiked(self):
        if self.accumulator >= 1:
            return True
        else:
            return False

    def erase(self):
        self.accumulator = 0


class Net:
    def __init__(self, n_neurons):
        self.neurons = []
        self.synapses = []
        for _ in range(n_neurons):
            self.neurons.append(Neuron())
        for outer_neuron in self.neurons:
            for inner_neuron in self.neurons:
                if outer_neuron.number == inner_neuron.number:
                    continue
                synapse = Synapse(outer_neuron, inner_neuron)
                self.synapses.append(synapse)

    # метод поиска нужного нейрона, возвращает экземпляр найденнного нейрона
    def get_neuron(self, num):
        for n in self.neurons:
            if n.number == num:
                return n
        else:
            raise

    # метод ручного ввода импульса в нейрон
    def probe(self, number):
        n = self.get_neuron(number)
        n.accumulator = 1

    # метод ввода импульсов в несколько нейронов на одном тике
    # индекс числа во входном массиве - номер активируемого нейрона
    def massive_probe(self, array):
        for i in range(len(array)):
            if array[i] != 0:
                self.probe(i)

    def tick(self):
        for s in self.synapses:
            s.check_input_signal()
        for n in self.neurons:
            if n.number == 4 and n.is_spiked():
                print('hop')
            n.erase()
        for s in self.synapses:
            s.add_signals()


net = Net(5)


def make_synapse_real(first, second, weight=0.5):
    for s in net.synapses:
        if s.input_neuron.number == first and s.output_neuron.number == second:
            s.is_real = True
            s.weight = weight


make_synapse_real(0, 2, weight=0.5)
make_synapse_real(1, 2, weight=0.5)
make_synapse_real(2, 4, weight=1)

net.probe(0)
net.probe(1)

for _ in range(50):
    net.tick()

net.probe(0)
net.probe(1)

for _ in range(50):
    net.tick()
