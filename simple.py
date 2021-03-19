'''
В этом файлк буду реализовывать подход из методички Шумилова
'''

print('Hello world!')


class Neuron:
    def __init__(self):
        pass


class Net:
    def __init__(self, n_neurons=10, p_excited=0.01, p_join=0.9, koef_old=0.5):
        self.n_neurons = n_neurons
        self.p_excited = p_excited
        self.p_join = p_join
        self.koef_old = koef_old

        self.neurons = []
        for n in range(self.n_neurons):
            self.neurons.append(Neuron(p_excited=self.p_excited,
                                       p_join=self.p_join,
                                       koef_old=self.koef_old))
