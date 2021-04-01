# -*- coding: utf-8 -*-


class Synapse:
    def __init__(self, input_n, output_n, weight=0.5):
        self.input_n = input_n
        self.output_n = output_n
        self.weight = weight
        self.is_signal = 0

    def check_signal(self):
        if self.is_signal == 0:
            return False
        else:
            self.is_signal = 0
            return True

    def activate(self):
        self.is_signal = 1


class Neuron:
    def __init__(self, number, accumulator=0):
        self.accumulator = accumulator
        self.number = number

    def adding(self, value):
        self.accumulator += value

    def is_activated(self):
        if self.accumulator >= 1:
            self.accumulator = 0
            return True
        else:
            self.accumulator -= 0.1
            if self.accumulator <= 0:
                self.accumulator = 0
            return False


class Net:
    def __init__(self, n_neurons, synapses=None, output_n=None):
        self.n_neurons = n_neurons
        self.output_n = output_n
        self.neurons = []
        if not synapses:
            self.synapses = []
        else:
            self.synapses = synapses

        for i in range(self.n_neurons):
            self.neurons.append(Neuron(i))

    def add_synapse(self, input_n, output_n, weight=0.5):
        s = Synapse(input_n, output_n, weight=weight)
        self.synapses.append(s)

    def get_neuron(self, number):
        for n in self.neurons:
            if n.number == number:
                return n
        else:
            raise

    def get_target_synapses(self, out):
        result = []
        for s in self.synapses:
            if s.input_n == out:
                result.append(s)
        return result

    def probe(self, number):
        n = self.get_neuron(number)
        n.accumulator += 1

    def tik(self):
        # проверяем сигнал в синапсах
        # суммируем сигналы в целевых нейронах
        for s in self.synapses:
            if s.check_signal():
                current_neuron = self.get_neuron(s.output_n)
                current_neuron.adding(s.weight)

        # >1 - активируем
        # выдаем сигнал в синапс
        for n in self.neurons:
            if n.is_activated():
                synapses_to_activate = self.get_target_synapses(n.number)
                for item in synapses_to_activate:
                    item.activate()
                if n.number == self.output_n:
                    print('hop')


synapses = [Synapse(0, 2, weight=0.5), Synapse(1, 2, weight=0.5), Synapse(2, 3, weight=1.5)]
net = Net(5, synapses=synapses, output_n=3)
net.probe(0)
net.probe(1)
for i in range(50):
    net.tik()
net.probe(0)
net.probe(1)
for i in range(50):
    net.tik()

# какие техники помогут в обучении?
'''
1. самопроизвольная активация нейронов
2. создание связей (каких?) между часто одновременно активирующимися нейронами
(если эта техника для упрощения связей, что скорее всео так и есть, то что тогда надо для роста новых синапсов?)
3. учитывание обратной связи - поощрение/наказание
4. тормозящие нейроны
5. усиление/ослабление весов со временем
6. надо подумать, что может заставить синапсы расти в верном направлении - помним об аксональном наведении

Для начала сделаем линейный регрессор внутри нейрона для определения весов связей
для сохранения состояний входов необходимо допилить буферы, а так же отказаться от аккумулятора
Нужно разработать эвристики для создания новых синапсов и удаления старых
Пока на предопределенном наборе нейронов будем тестировать

В дальнейшем планируется использование генетических алгоритмов для выяснения алгоритмов поведения нейронов
'''

from sklearn.linear_model import LinearRegression

    
class Neuron_v2(Neuron):
    def __init__(self, number, accumulator=0):
        super().__init__(number, accumulator=accumulator)
        self.LR = LinearRegression()
        # TODO: в инициализации нейрона в сети сделать запись синапса в нейрон
        self.inputs = []  # для имеющихся входных синапсов
        self.output = None
        self.input_buffer = []
        self.output_buffer = []
        self.weight_buffer = []

    def extract_signals(self):
        result_input = []
        for s in self.inputs:
            if s.check_signal():
                result_input.append(1)
            else:
                result_input.append(0)
        return result_input
                
    def extract_weights(self):
        result_weights = []
        for s in self.inputs:
            result_weights.append(s.weight)
        return result_weights
        
    def is_activated(self):
        if self.accumulator >= 1:
            self.accumulator = 0
            return True
        else:
            self.accumulator -= 0.1
            if self.accumulator <= 0:
                self.accumulator = 0
            return False
    
    
class Net_v2(Net):
    pass
#     def __init__(self, n_neurons, synapses=None, output_n=None):
#         self.n_neurons = n_neurons
#         self.output_n = output_n
#         self.neurons = []
#         if not synapses:
#             self.synapses = []
#         else:
#             self.synapses = synapses

#         for i in range(self.n_neurons):
#             self.neurons.append(Neuron(i))

#     def add_synapse(self, input_n, output_n, weight=0.5):
#         s = Synapse(input_n, output_n, weight=weight)
#         self.synapses.append(s)

#     def get_neuron(self, number):
#         for n in self.neurons:
#             if n.number == number:
#                 return n
#         else:
#             raise

#     def get_target_synapses(self, out):
#         result = []
#         for s in self.synapses:
#             if s.input_n == out:
#                 result.append(s)
#         return result

#     def probe(self, number):
#         n = self.get_neuron(number)
#         n.accumulator += 1

#     def tik(self):
#         # проверяем сигнал в синапсах
#         # суммируем сигналы в целевых нейронах
#         for s in self.synapses:
#             if s.check_signal():
#                 current_neuron = self.get_neuron(s.output_n)
#                 current_neuron.adding(s.weight)

#         # >1 - активируем
#         # выдаем сигнал в синапс
#         for n in self.neurons:
#             if n.is_activated():
#                 synapses_to_activate = self.get_target_synapses(n.number)
#                 for item in synapses_to_activate:
#                     item.activate()
#                 if n.number == self.output_n:
#                     print('hop')
