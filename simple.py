# -*- coding: utf-8 -*-
'''
В этом файлк буду реализовывать подход из методички Шумилова
'''

print('Hello world!')


# class Neuron:
#     def __init__(self, p_excited=0.01, p_join=0.9, koef_old=0.5):
#         self.p_excited = p_excited
#         self.p_join = p_join
#         self.koef_old = koef_old
#         self.input_weights = []


# class Net:
#     def __init__(self, n_neurons=10, p_excited=0.01, p_join=0.9, koef_old=0.5):
#         self.n_neurons = n_neurons
#         self.p_excited = p_excited
#         self.p_join = p_join
#         self.koef_old = koef_old

#         self.synapse = []
#         self.neurons = []
#         for n in range(self.n_neurons):
#             self.neurons.append(Neuron(p_excited=self.p_excited,
#                                        p_join=self.p_join,
#                                        koef_old=self.koef_old))

#     def tik(self):
#         pass

import random


neurons = [[0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0],
           [2, 0, 0, 0, 0],
           [3, 0, 0, 0, 0],
           [4, 0, 0, 0, 0],
           # [5, 0, 0, 0],
           # [6, 0, 0, 0],
           # [7, 0, 0, 0],
           # [8, 0, 0, 0],
           [9, 0, 0, 0, 0]]  # айдишники нейронов, активирован ли, x, y, аккумулятор

synapses = [[0, 1, 0.5, 0],  # откуда, куда, вес, активировался ли в этом тике
            [2, 1, 0.5, 0],
            [1, 4, 1, 0]]

probes = [[0, 2], [0, 0], [0, 0], [0, 0]]  # [[0, 1], [0, 2], [1, 2], ...] на какие нейроны над подать сигнал


for n in neurons:
    n[2], n[3] = random.random(), random.random()


# TODO: нужна проверка нейронов на активацию в этом тике?
# TODO: проверка на сумму весов на входе
for item in probes:
    for s in synapses:
        s[3] = 0
    for s in synapses:
        first_neuron = s[0]
        second_neuron = s[1]
        # активируем синапсы
        if (neurons[first_neuron][1] == 1) and (s[3] == 0):
            neurons[second_neuron][4] += s[2]
            neurons[first_neuron][1] = 0
            s[3] = 1
            if neurons[second_neuron][4] >= 1:
                neurons[first_neuron][1] = 1
            else:
                neurons[first_neuron][1] = 0

    # for s in synapses:
    #     first_neuron = s[0]
    #     second_neuron = s[1]
    #     if s[3] == 1:


    # возбуждаем нейроны входными импульсами
    for p in item:
        neurons[p] = [p, 1, neurons[p][2], neurons[p][3], neurons[p][4]]

    if neurons[4][1] == 1:
        print('yeah!')

    for n in neurons:
        if n[1] == 0:
            n[4] = 0
        if n[1] == 1:
            zroid = True
            for s in synapses:
                if s[0] == n[0]:
                    zroid = False
        if zroid:
            n[1] = 0



# def probe_signal(neuron_ids):
#     activated = []
#     for n in neuron_ids:
#         a[n] = [n, 1]
#         activated.append(n)

#     for t in a:
#         if t[0] not in activated:
#             t = [t[0], 0]
