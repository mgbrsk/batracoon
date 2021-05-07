import numpy as np
from copy import deepcopy


BUFFER_SIZE = 10


class Neuron:
    def __init__(self, number, is_input=False, is_output=False, weights=None):
        self.number = number
        self.is_input = is_input
        self.is_output = is_output
        self.history = [0 for _ in range(BUFFER_SIZE)]
        if not self.is_input:
            self.input_slots = []  # слоты для других нейронов
        if not self.is_output:
            self.output_slots = []  # слоты для других нейронов
        self.accumulator = 0
        self.temp_signal = 0
        if not weights:
            self.weights = []
        else:
            self.weights = weights
        self.references_from_right = []
        self.pos_reference_left = [0 for _ in range(BUFFER_SIZE)]
        self.neg_reference_left = [0 for _ in range(BUFFER_SIZE)]
        self.wanting = [0 for _ in range(BUFFER_SIZE)]
        # self.wanting_for_left = [0 for _ in range(BUFFER_SIZE)]

    def check_signal(self):
        out = False
        if self.accumulator >= 1:
            self.temp_signal = 1
            self.history.append(1)
            out = True
        else:
            self.temp_signal = 0
            self.history.append(0)
        self.accumulator = 0
        self.history.pop(0)
        return out

    def move_forward(self):
        if self.is_input:
            return
        for n, w in zip(self.input_slots, self.weights):
            self.accumulator += n.temp_signal * w

    def get_right_reference(self):
        if self.is_input or self.is_output:
            return
        for n in self.output_slots:
            self.references_from_right.append(n.wanting)

    def calculate_wanting(self, y_true=None):
        # if self.number == 2:
        #     print(self.wanting)
        # ####
        if self.is_input:
            return
        if self.is_output:
            self.wanting = y_true
            return
        if len(self.references_from_right) % 2 == 0:
            self.references_from_right.append([0 for _ in range(BUFFER_SIZE)])
        # if self.number == 2:
        #     print(self.references_from_right)
        # ####
        self.wanting = np.median(np.array(self.references_from_right), axis=0)
        self.references_from_right = []
        # if self.number == 2:
        #     print(self.wanting)
        self.wanting = list(self.wanting)

    def calculate_pos_neg_reference(self):
        if self.is_input:
            return
        # temp = np.array(self.history) - np.array(self.wanting)
        temp = np.array(self.wanting) - np.array(self.history)
        self.pos_reference_left = np.where(temp > 0, 1, 0)
        self.neg_reference_left = np.where(temp < 0, 1, 0)

    def move_weights(self):
        if self.is_input:
            return
        for en, n in enumerate(self.input_slots):
            # pos_coef1 = np.array(n.history) * np.array(self.pos_reference_left)
            # pos_coef2 = np.array(n.history) == np.array(self.pos_reference_left)
            # pos_coef2 = pos_coef2 * 1
            # pos_coef = pos_coef1 * 0.2 + pos_coef2 * 0.8
            pos_coef = np.array(n.history) == np.array(self.pos_reference_left)
            pos_coef = np.sum(pos_coef * 1) / pos_coef.shape[0]
            if (not np.array(self.pos_reference_left).any()):  # (not np.array(n.history).any()):  #  and (not np.array(self.pos_reference_left).any())
                pos_coef = 0

            # neg_coef1 = np.array(n.history) * np.array(self.neg_reference_left)
            # neg_coef2 = np.array(n.history) == np.array(self.neg_reference_left)
            # neg_coef2 = neg_coef2 * 1
            # neg_coef = neg_coef1 * 0.8 + neg_coef2 * 0.2
            neg_coef = np.array(n.history) == np.array(self.neg_reference_left)
            neg_coef = np.sum(neg_coef * 1) / neg_coef.shape[0]
            if (not np.array(self.neg_reference_left).any()):  # (not np.array(n.history).any()):  #  and (not np.array(self.neg_reference_left).any())
                neg_coef = 0
            if pos_coef == neg_coef:
                continue
            if pos_coef > neg_coef:
                self.weights[en] += pos_coef * 0.2
            else:
                # if self.is_output:
                #     pass
                # else:
                self.weights[en] -= neg_coef * 0.1
            if self.weights[en] > 1:
                self.weights[en] = 1
            if self.weights[en] < -1:
                self.weights[en] = -1


def make_connection(n1, n2):
    pass


n0 = Neuron(0, is_input=True)
n1 = Neuron(1, is_input=True)
n2 = Neuron(2, weights=[0, 0])
n3 = Neuron(3, weights=[0, 0])
n4 = Neuron(4, is_output=True, weights=[0, 0])
n0.output_slots = [n2, n3]
n1.output_slots = [n2, n3]
n2.input_slots = [n0, n1]
n3.input_slots = [n0, n1]
n2.output_slots = [n4]
n3.output_slots = [n4]
n4.input_slots = [n2, n3]
net = [n0, n1, n2, n3, n4]
# n0 = Neuron(0, is_input=True)
# n1 = Neuron(1, is_input=True)
# n2 = Neuron(2, weights=[0, 0])
# n3 = Neuron(3, weights=[0])
# n4 = Neuron(4, weights=[0])
# n5 = Neuron(5, is_output=True, weights=[0, 0, 0])
#
# n0.output_slots = [n2, n3]
# n1.output_slots = [n2, n4]
#
# n2.input_slots = [n0, n1]
# n2.output_slots = [n5]
#
# n3.input_slots = [n0]
# n3.output_slots = [n5]
#
# n4.input_slots = [n1]
# n4.output_slots = [n5]
#
# n5.input_slots = [n2, n3, n4]
# net = [n0, n1, n2, n3, n4, n5]
x = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
     [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
     [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
     [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],]
y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,]
res = []

y_true = [0 for _ in range(BUFFER_SIZE)]

weights_history = []

counter = 0
for _ in range(1):
    res = []
    weights_history = []
    for cur_x, cur_y in zip(x, y):
        counter += 1
        cur_weights = [x.weights for x in net]  #  + [list(n2.neg_reference_left)]
        # print(cur_weights)
        weights_history.append(deepcopy(cur_weights))
        y_true.append(cur_y)
        y_true.pop(0)
        # ввод сигналов в сеть
        for en, i in enumerate(cur_x):
            if i == 1:
                net[en].accumulator = 1
        # проверка аккумуляторов и активация нейронов
        for n in net:
            out = n.check_signal()
            if n.is_output and out:
                res.append(1)
            elif n.is_output and not out:
                res.append(0)
        # передача сигнала вперед
        for n in net:
            n.move_forward()
        # обратное распространение ошибки
        # if counter > 10:
        #     continue
        for n in net:
            n.get_right_reference()
        for n in net:
            n.calculate_wanting(y_true=y_true)
        for n in net:
            n.calculate_pos_neg_reference()
        for n in net:
            n.move_weights()

print(y)
print(res)
for n in net:
    print(n.weights)
# print(weights_history)
for i in weights_history:
    res = []
    for j in i:
        for k in j:
            res.append(str(k))
    print(','.join(res))
