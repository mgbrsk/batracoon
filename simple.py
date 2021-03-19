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

neurons = [[0, 0],
           [1, 0],
           [2, 0],
           [3, 0],
           [4, 0],
           [5, 0],
           [6, 0],
           [7, 0],
           [8, 0],
           [9, 0]]  # айдишники нейронов, активирован ли

synapses = [[0, 1, 0.5, 0],  # откуда, куда, вес, есть ли импульс в синапсе
            [2, 1, 0.5, 0],
            [2, 9, 1, 0]]

probes = []  # [[0, 1], [0, 2], [1, 2], ...] на какие нейроны над подать сигнал

for item in probes:
    # возбуждаем нейроны входными импульсами
    for p in item:
        neurons[p] = [p, 1]

    for s in synapses:
        pass



# def probe_signal(neuron_ids):
#     activated = []
#     for n in neuron_ids:
#         a[n] = [n, 1]
#         activated.append(n)

#     for t in a:
#         if t[0] not in activated:
#             t = [t[0], 0]
