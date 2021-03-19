import numpy as np


class SimpleNet:
    def __init__(self, n_neurons=5, n_synapse=4, inputs=None,
                 output=4, neurons=None, synapse=None):
        # сеть состоит из нейронов (n_neurons штук), связей между ними (n_synapse штук)

        if inputs is None:
            inputs = [0, 1]
        if synapse is None:
            # 1 нейрон, 2 нейрон, дист, вес, счетчик пути
            self.synapse = np.zeros((n_synapse, 5))
        else:
            self.synapse = synapse
        if neurons is None:
            # номер, аккумулятор, счетчик реполяризации, флаг пост.
            self.neurons = np.zeros((n_neurons, 4))
        else:
            self.neurons = neurons
        self.result = .0
        self.output = output
        self.inputs = inputs

    def from_adjacent(self, ad, w, const=None):
        # формирует сеть из матрицы смежности ad и массива весов связей w

        if not const:
            const = np.zeros(len(self.neurons))

        counter = 0
        for i in range(len(ad)):
            for j in range(len(ad)):
                if ad[i][j] != 0:
                    self.synapse[counter][0] = i
                    self.synapse[counter][1] = j
                    self.synapse[counter][2] = ad[i][j]
                    self.synapse[counter][3] = w[i][j]
                    counter += 1
            self.neurons[i][3] = const[i]    
        self.neurons[:, 0] = range(len(self.neurons))

    def to_adjacent(self):
        # переводит сеть в матрицу смежности и массив весов связей

        ad = np.zeros((len(self.neurons), len(self.neurons)))
        w = np.zeros((len(self.neurons), len(self.neurons)))
        for item in self.synapse:
            ad[int(item[0]), int(item[1])] = int(item[2])
            w[int(item[0]), int(item[1])] = item[3]
        return ad, w

    def reset(self):
        # сбрасывает сеть в исходное состояние
        self.neurons[:, 1:3] = .0
        self.synapse[:, -1] = .0

    def activate_input_neuron(self, n_input=0):
        self.neurons[n_input, 1] = 1.1

    def tik_signal(self):
        # берем сигналы которые на последнем шаге
        last_step_signals = self.synapse[self.synapse[:, -1] == 1]
        # увеличиваем аккумуляторы всех целевых нейронов
        # self.neurons[last_step_signals[:, 1].astype(int), 1] += \
        #                             last_step_signals[:, 3]
        # TODO: сделать быстрее через высокие технологии numpy
        for i in range(len(last_step_signals)):
            self.neurons[int(last_step_signals[i, 1]), 1] +=\
                    last_step_signals[i, 3]
        # обнуляем аккумуляторы реполяризованных нейронов
        self.neurons[:, 1] =\
                np.where(self.neurons[:, 2] > 0, 0, self.neurons[:, 1])
        # убавляем счетчик пути
        self.synapse[:, -1] = np.where(self.synapse[:, -1] > 0, 
                                       self.synapse[:, -1] - 1, 0)

    def tik_neurons(self):
        # обнуляем константные и создаем их сигналы
        self.neurons[self.neurons[:, -1] == 1, 2] = 0
        self.neurons[self.neurons[:, -1] == 1, 1] = 1
        # убавляем счетчик реполяризации
        self.neurons[:, 2] = np.where(self.neurons[:, 2] > 0,
                                      self.neurons[:, 2] - 1, 0)
        # 1. создаем сигналы
        # нужные нейроны
        temp_neurons = self.neurons[self.neurons[:, 1] >= 1.0]
        # print(self.synapse[np.isin(self.synapse[:, 0], 
        #                      temp_neurons[:, 0])][:, 2])
        # print('tp0')
        # print(self.synapse)
        self.synapse[np.isin(self.synapse[:, 0], 
                             temp_neurons[:, 0]), -1] =\
                self.synapse[np.isin(self.synapse[:, 0], 
                             temp_neurons[:, 0]), 2]
        # print('tp1')
        # print(self.synapse)
        # 2. реполяризация
        self.neurons[self.neurons[:, 1] >= 1.0, 2] = 7
        # 2.1 сохраняем в результат если нейрон выходной
        self.result += temp_neurons[np.isin(temp_neurons[:, 0], 
                                            [self.output]), 1].sum()
        # 3. обнуление аккумуляторов
        self.neurons[self.neurons[:, 1] >= 1.0, 1] = .0
        # релаксация аккумулятора
        self.neurons[:, 1] = np.where(self.neurons[:, 1] < -0.1,
                                      self.neurons[:, 1] + 0.1,
                                      self.neurons[:, 1])
        self.neurons[:, 1] = np.where(self.neurons[:, 1] > 0.1,
                                      self.neurons[:, 1] - 0.1, 0)

    def predict(self, X=None, limit=20):
        # пока только бинарный ввод (1 или 0 на входе)
        # input_counter = 0
        if X is None:
            X = [0, 1]

        counter = 0
        for n, i in enumerate(self.inputs):
            if X[n] == 1:
                # if X[input_counter] == 1:
                self.activate_input_neuron(i)
                # input_counter += 1
        while (self.neurons[:, 1].sum() > 0 or\
              self.synapse[:, -1].sum() > 0) and\
              counter < limit:
            print('new stage:')
            print('before all')
            print(self.neurons)
            print(self.synapse)
            self.tik_signal()
            print('after signal')
            print(self.neurons)
            print(self.synapse)
            self.tik_neurons()
            counter += 1
            print('after neurons')
            print(self.neurons)
            print(self.synapse)
            print('****')
        return self.result


net = SimpleNet(inputs=[0, 1], output=4)

ad = [[0, 0, 1, 0, 0],
      [0, 0, 1, 0, 0],
      [0, 0, 0, 0, 1],
      [0, 0, 0, 0, 1],
      [0, 0, 0, 0, 0]]

w = [[0, 0, 0.5, 0, 0],
     [0, 0, 0.6, 0, 0],
     [0, 0, 0, 0, -0.3],
     [0, 0, 0, 0, 0.7],
     [0, 0, 0, 0, 0]]

const = [0, 0, 0, 1, 0]

net.from_adjacent(ad, w, const)

res = net.predict(X=[1, 1])
print(res)


# temp_neurons = np.array([[0, 1.1, 0, 0],
#                          [1, 1.1, 0, 0]])
# synapse = np.array([[0, 2, 1, 1, 0],
#                     [1, 3, 1, 1, 0],
#                     [2, 4, 1, 0.7, 0],
#                     [3, 4, 1, 0.6, 0]])

# print(synapse)

# synapse[np.isin(synapse[:, 0], 
#                      temp_neurons[:, 0])][:, 4] =\
#         synapse[np.isin(synapse[:, 0], 
#                      temp_neurons[:, 0])][:, 2]

# synapse = np.where(np.isin(synapse[:, 0], temp_neurons[:, 0]),
#                    synapse[:, -1] = synapse[:, 2], synapse)

# synapse[[True, True, False, False], [-1]] = 99

# print(synapse)
# print('***')
# print(synapse[np.isin(synapse[:, 0], 
#                      temp_neurons[:, 0])])

