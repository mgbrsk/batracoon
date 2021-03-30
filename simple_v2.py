# -*- coding: utf-8 -*-

from sklearn.linear_model import LinearRegression


LR = LinearRegression(fit_intercept=False)


class Synapse:
    def __init__(self, input_n, output_n, weight=0.5, distance=1):
        self.input_n = input_n
        self.output_n = output_n
        self.weight = weight
        self.is_signal = 0

    def tick_signal(self):
        pass

    def check_signal(self):
        if self.is_signal == 0:
            return False
        else:
            self.is_signal = 0
            return True

    def activate(self):
        self.is_signal = 1


class Neuron:
    def __init__(self, number):
        self.number = number
        self.lr = LR
        self.input_synapses = []
        self.output_synapses = []
        # буферы для обучения
        self.input_buffer = [[] for _ in range(10)]
        self.output_buffer = [0 for _ in range(10)]
        self.true_output_buffer = [0 for _ in range(10)]
        self.weight_buffer = [[] for _ in range(10)]
        ###
        self.accumulator = 0

    def get_weights(self):
        res = []
        for s in self.input_synapses:
            res.append(s.weight)
        return res

    def add_input_synapse(self, synapse):
        self.input_synapses.append(synapse)

    def add_output_synapse(self, synapse):
        self.output_synapses.append(synapse)

    def sum_signals(self):
        current_input = []
        current_weights = []
        for i_s in self.input_synapses:
            current_weights.append(i_s.weight)
            if i_s.check_signal():
                self.accumulator += i_s.weight
                current_input.append(1)
            else:
                current_input.append(0)
        self.input_buffer.append(current_input)
        self.input_buffer.pop(0)
        self.weight_buffer.append(current_weights)
        self.weight_buffer.pop(0)

    def generate_output(self):
        if self.accumulator >= 1:
            for o_s in self.output_synapses:
                o_s.activate()
            self.accumulator = 0
            self.output_buffer.append(1)
            self.output_buffer.pop(0)
            return self.number
        self.output_buffer.append(0)
        self.output_buffer.pop(0)
        return False

    def set_true_output(self, true_output):
        self.true_output_buffer.append(true_output)
        self.true_output_buffer.pop(0)
        
    # TODO: check this implementation
    def update_buffer(self, buffer, value):
        buffer.append(value)
        buffer.pop(0)
        
    def learning(self, reaction):
        # -1 or 1 input
        if reaction == 1:
            last_output = self.output_buffer[-1]
        elif reaction == -1:
            # меняем 1 на 0 и 0 на 1
            last_output = self.output_buffer[-1] * (-1) + 1
        self.set_true_output(self, last_output)
        
        lr.fit(self.input_buffer, self.true_output_buffer)
        need_weights = list(lr.coef_)
        for i, s in enumerate(self.input_synapses):
            s.weight = need_weights[i]


class Net:
    def __init__(self, neurons_number):
        self.neurons = []
        self.synapses = []
        for i in range(neurons_number):
            self.neurons.append(Neuron(i))

    def get_neuron(self, num):
        for n in self.neurons:
            if n.number == num:
                return n
        else:
            raise

    def add_synapse(self, n_input, n_output, weight=0.5):
        s = Synapse(n_input, n_output, weight=weight)
        self.synapses.append(s)
        output_neuron = self.get_neuron(n_input)
        input_neuron = self.get_neuron(n_output)
        input_neuron.add_input_synapse(s)
        output_neuron.add_output_synapse(s)

    def tick(self):
        out_signal = False
        for n in self.neurons:
            n.sum_signals()
        for n in self.neurons:
            p = n.generate_output()
            if p == 2:
                out_signal = True
                print('hop')
        return out_sinal
        # set_true_output

    def probe(self, number):
        n = self.get_neuron(number)
        n.accumulator = 1

    def massive_probe(self, array):
        for i in range(array):
            if array[i] != 0:
                self.probe(i)
        
    def check_right(self, y_true, y_pred):
        if y_true == y_pred:
            return 1
        else:
            return -1
        
    def fit(self, X, y):
        if len(X) != len(y):
            raise
        for cur_x, cur_y in zip(X, y):
            self.massive_probe(cur_x)
            res = self.tick()
            checked_out = self.check_right(cur_y, int(res))
            for n in self.neurons:
                n.learning(checked_out)
            


net = Net(5)
net.add_synapse(0, 2)
net.add_synapse(1, 2)
net.add_synapse(2, 4, weight=1)
net.probe(0)
net.probe(1)
for i in range(50):
    net.tick()
net.probe(0)
net.probe(1)
for i in range(50):
    net.tick()
