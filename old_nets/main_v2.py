
import numpy as np
import copy
import random


# const_neurons = [0, 0, 0, 0, 0]
# const_neurons = np.array(const_neurons)

# # input - 0, output - 4
# dist = [[0, 0, 1, 0, 0],
#         [0, 0, 0, 0, 1],
#         [0, 0, 0, 1, 0],
#         [0, 0, 0, 0, 0],
#         [0, 0, 0, 1, 0]]
# dist = np.array(dist)

# weights = [[0, 0, 1, 0, 0],
#            [0, 0, 0, 0, 1],
#            [0, 0, 0, 0.6, 0],
#            [0, 0, 0, 0, 0],
#            [0, 0, 0, 0.8, 0]]
# weights = np.array(weights, dtype=float)

# accum = [[0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0],
#          [0, 0, 0, 0, 0]]
# accum = np.array(accum, dtype=float)

# repolar_counter = [[0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0],
#                    [0, 0, 0, 0, 0]]
# repolar_counter = np.array(repolar_counter)

# #signals
# way_counter = [[0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0]]
# way_counter = np.array(way_counter)

# sig_value = [[0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0],
#              [0, 0, 0, 0, 0]]
# sig_value = np.array(sig_value, dtype=float)


# def create_input_signal(n_input=0):
#     res_sig = np.array([1, 1, 1, 1, 1]) * weights[n_input]
#     sig_value[n_input] = res_sig
#     res_way = np.array([1, 1, 1, 1, 1]) * dist[n_input]
#     way_counter[n_input] = res_way
#     return sig_value, way_counter


# def tik_signal():
#     global sig_value
#     global way_counter
#     global accum
#     # пропускаем сигналы которые на последнем шаге
#     vfunc = np.vectorize(lambda x: 1 if x == 1 else 0)
#     temp_way_counter = vfunc(way_counter)
#     # и если целевой нейрон не реполяризован
#     vfunc2 = np.vectorize(lambda x: 1 if x == 0 else 0)
#     temp_is_repolar = vfunc2(repolar_counter)
#     result_signals = temp_way_counter * temp_is_repolar * sig_value
#     # добавляем в аккумулятор целевого нейрона (столбцом)
#     #print('old', accum)
#     accum = accum + result_signals
#     #print('tppp', accum)
#     # columns_sum = accum.sum(axis=0)
#     # accum[:] = columns_sum
#     # убавляем счетчик пути на один
#     vfunc3 = np.vectorize(lambda x: x - 1 if x > 0 else 0)
#     way_counter = vfunc3(way_counter)
#     # удаляем дошедшие сигналы
#     vfunc4 = np.vectorize(lambda x: 0 if x == 0 else 1)
#     temp_wc = vfunc4(way_counter)
#     sig_value = sig_value * way_counter

#     return accum, way_counter, sig_value

# accum, way_counter, sig_value = tik_signal()
# print(accum)


# def get_simple_acc(t_accum):
#     temp = copy.deepcopy(t_accum)
#     columns_sum = temp.sum(axis=0)
#     temp[:] = columns_sum
#     return temp


# def tik_neurons():
#     global repolar_counter
#     global accum
#     global way_counter
#     global sig_value
#     # обнуляем аккумуляторы нейронов, которые реполяризованы
#     # (счетчик реполяризации > 0)
#     vfunc = np.vectorize(lambda x: 1 if x == 0 else 0)
#     temp_is_repolar = vfunc(repolar_counter)
#     accum = accum * temp_is_repolar
#     # 1) обнуляем реполяризацию и аккумулятор константных нейронов
#     temp_const = vfunc(const_neurons)
#     accum = accum * temp_const
#     repolar_counter = repolar_counter * temp_const
#     # активация нейронов, создание сигналов и реполяризация
#     vfunc2 = np.vectorize(lambda x: 1 if x >= 1 else 0)
#     #print('test1', accum)
#     temp_accum = get_simple_acc(accum)
#     #print('test2', accum)
#     more_th = vfunc2(temp_accum)
#     columns_repolar_source = more_th.sum(axis=0)
#     # columns_repolar_t - строка, 1 - реполяризован целевой, 0 - нет
#     # применяем функцию "трешхолда"
#     columns_repolar_t = vfunc2(columns_repolar_source)
#     columns_repolar = columns_repolar_t * 7  # на сколько тиков реполя
#     repolar_counter_t = np.zeros((5, 5))
#     repolar_counter_t[:] = columns_repolar
#     repolar_counter = repolar_counter + repolar_counter_t

#     # создаем сигналы
#     # columns_repolar_t - сработал или нет нейрон по индексу
#     way_counter = (dist.T * (columns_repolar_t + 
#                    const_neurons)).T + way_counter
#     sig_value = (weights.T * (columns_repolar_t + 
#                  const_neurons)).T + sig_value

#     # убавляем счетчик реляризации на один
#     vfunc3 = np.vectorize(lambda x: x - 1 if x > 0 else 0)
#     repolar_counter = vfunc3(repolar_counter)
#     # обнуляем столбцы аккумуляторов активированных нейронов
#     columns_repolar_t = vfunc(columns_repolar_t)  # инвертирование
#     accum = accum * columns_repolar_t
#     vfunc4 = np.vectorize(lambda x: x - 0.1 if x > 0.1 else .0)
#     #print('bef', accum)
#     accum = vfunc4(accum)
#     # if is_accum_debug:
#     #     print('tp4', accum)

#     # 2) обнуляем реполяризацию и аккумулятор константных нейронов
#     temp_const = vfunc(const_neurons)
#     accum = accum * temp_const
#     repolar_counter = repolar_counter * temp_const
#     # if is_accum_debug:
#     #     print('tp5', accum)
#     return accum, repolar_counter, way_counter, sig_value
    
# accum, repolar_counter, way_counter, sig_value = tik_neurons()
# print(repolar_counter)
# print(accum)




# sig_value, way_counter = create_input_signal(0)
# #sig_value, way_counter = create_input_signal(1)
# accum, way_counter, sig_value = tik_signal()
# accum, repolar_counter, way_counter, sig_value = tik_neurons()
# print('sig_value')
# print(sig_value)
# print('way_counter')
# print(way_counter)
# print('repolar_counter')
# print(repolar_counter)
# print('accum')
# print(accum)

# sig_value, way_counter = create_input_signal(1)

# for _ in range(3):
#     # sig_value, way_counter = create_input_signal()
#     # if _ == 1:
#     #     is_accum_debug = True
#     accum, way_counter, sig_value = tik_signal()
#     accum, repolar_counter, way_counter, sig_value = tik_neurons()

#     print('**********')
#     print('sig_value')
#     print(sig_value)
#     print('way_counter')
#     print(way_counter)
#     print('repolar_counter')
#     print(repolar_counter)
#     print('accum')
#     print(accum)


class SimpleNet:
    def __init__(self, n_neurons=5, dist=None, weights=None,
                 outputs=None, inputs=None, const_neurons=None):

        if (dist is None) or (weights is None) or (outputs is None)\
                or (inputs is None):
            raise Exception()

        self.n_neurons = n_neurons

        if const_neurons is not None:
            self.const_neurons = np.array(const_neurons)
        else:
            self.const_neurons = np.zeros(self.n_neurons)

        self.outputs = np.array(outputs)

        self.result = 0

        self.inputs = np.array(inputs)

        self.dist = np.array(dist)

        self.weights = np.array(weights, dtype=float)

        self.accum = np.zeros((self.n_neurons, self.n_neurons))

        self.repolar_counter = np.zeros((self.n_neurons,
                                         self.n_neurons))

        self.way_counter = np.zeros((self.n_neurons, self.n_neurons))

        self.sig_value = np.zeros((self.n_neurons, self.n_neurons))

    def add_neuron(self):
        raise NotImplementedError()

    def reset(self):
        self.accum = np.zeros((self.n_neurons, self.n_neurons))
        self.repolar_counter = np.zeros((self.n_neurons,
                                         self.n_neurons))
        self.way_counter = np.zeros((self.n_neurons, self.n_neurons))
        self.sig_value = np.zeros((self.n_neurons, self.n_neurons))

    def create_input_signal(self, n_input=0):
        res_sig = np.ones(self.n_neurons) * self.weights[n_input]
        self.sig_value[n_input] = res_sig
        res_way = np.ones(self.n_neurons) * self.dist[n_input]
        self.way_counter[n_input] = res_way

    def tik_signal(self):
        # пропускаем сигналы которые на последнем шаге
        temp_way_counter = np.where(self.way_counter == 1, 1, 0)
        # и если целевой нейрон не реполяризован
        temp_is_repolar = np.where(self.repolar_counter == 0, 1, 0)
        result_signals = temp_way_counter * \
                         temp_is_repolar * \
                         self.sig_value
        # добавляем в аккумулятор целевого нейрона (столбцом)
        self.accum = self.accum + result_signals
        # убавляем счетчик пути на один
        self.way_counter = np.where(self.way_counter > 0, self.way_counter - 1, 0)
        # удаляем дошедшие сигналы
        temp_wc = np.where(self.way_counter == 0, 0, 1)
        self.sig_value = self.sig_value * self.way_counter

    def get_simple_acc(self, t_accum):
        # вспомогательный метод
        temp = copy.deepcopy(t_accum)
        columns_sum = temp.sum(axis=0)
        temp[:] = columns_sum
        return temp

    def tik_neurons(self):
        # обнуляем аккумуляторы нейронов, которые реполяризованы
        # (счетчик реполяризации > 0)
        temp_is_repolar = np.where(self.repolar_counter == 0, 1, 0)
        self.accum = self.accum * temp_is_repolar
        # 1) обнуляем реполяризацию и аккумулятор константных нейронов
        temp_const = np.where(self.const_neurons == 0, 1, 0)
        self.accum = self.accum * temp_const
        self.repolar_counter = self.repolar_counter * temp_const
        # активация нейронов, создание сигналов и реполяризация
        temp_accum = self.get_simple_acc(self.accum)
        more_th = np.where(temp_accum >= 1, 1, 0)
        columns_repolar_source = more_th.sum(axis=0)
        # columns_repolar_t - строка, 1 - реполяризован целевой, 0 - нет
        # применяем функцию "трешхолда"
        columns_repolar_t = np.where(columns_repolar_source >= 1, 1, 0)
        # на сколько тиков реполяр (7)
        columns_repolar = columns_repolar_t * 7  
        repolar_counter_t = np.zeros((self.n_neurons, self.n_neurons))
        repolar_counter_t[:] = columns_repolar
        self.repolar_counter = self.repolar_counter + repolar_counter_t

        # записываем в результат если сработал выходной
        # TODO: !!!!
        self.result += (columns_repolar_t * self.outputs).sum()

        # создаем сигналы
        # columns_repolar_t - сработал или нет нейрон по индексу
        self.way_counter = (self.dist.T * (columns_repolar_t + 
                            self.const_neurons)).T + self.way_counter
        self.sig_value = (self.weights.T * (columns_repolar_t + 
                          self.const_neurons)).T + self.sig_value

        # убавляем счетчик реляризации на один
        self.repolar_counter = np.where(self.repolar_counter > 0, self.repolar_counter - 1, 0)
        # обнуляем столбцы аккумуляторов активированных нейронов
        # инвертирование
        columns_repolar_t = np.where(columns_repolar_t == 0, 1, 0)
        self.accum = self.accum * columns_repolar_t
        self.accum = np.where(self.accum > 0.1, self.accum - 0.1, .0)

        # 2) обнуляем реполяризацию и аккумулятор константных нейронов
        temp_const = np.where(self.const_neurons == 0, 1, 0)
        self.accum = self.accum * temp_const
        self.repolar_counter = self.repolar_counter * temp_const

    def predict(self, X=[0, 1], limit=50):
        # пока только бинарный ввод (1 или 0 на входе)
        input_counter = 0
        counter = 0
        for n, i in enumerate(self.inputs):
            if i == 1:
                if X[input_counter] == 1:
                    self.create_input_signal(n)
                input_counter += 1
        while self.way_counter.sum() > 0 and counter < limit:
            self.tik_signal()
            self.tik_neurons()
            counter += 1
            # print('sig_value')
            # print(self.sig_value)
            # print('way_counter')
            # print(self.way_counter)
            # print('repolar_counter')
            # print(self.repolar_counter)
            # print('accum')
            # print(self.accum)
            # print('****')
        return self.result



# dist = [[0, 0, 1, 0, 0],
#         [0, 0, 0, 0, 1],
#         [0, 0, 0, 1, 0],
#         [0, 0, 0, 0, 0],
#         [0, 0, 0, 1, 0]]
# dist = np.array(dist)
# weights = [[0, 0, 1, 0, 0],
#            [0, 0, 0, 0, 1],
#            [0, 0, 0, 0.6, 0],
#            [0, 0, 0, 0, 0],
#            [0, 0, 0, 0.8, 0]]
# weights = np.array(weights, dtype=float)
# outputs = [0, 0, 0, 1, 0]
# inputs = [1, 1, 0, 0, 0]
# const_neurons = [0, 0, 0, 0, 0]

# net = SimpleNet(n_neurons=5, dist=dist, weights=weights,
#                 outputs=outputs, inputs=inputs,
#                 const_neurons=const_neurons)
# res = net.predict(X=[1, 1])
# print(res)


def rand_dist_weights(n_neurons):
    dist = np.random.randint(2, size=(n_neurons, n_neurons))
    rand_weights = (np.random.rand(n_neurons, n_neurons) - 0.5) * 3
    weights = dist * rand_weights
    return dist, weights

nnn = 10
adjicence = [rand_dist_weights(nnn) for _ in range(100)]
const_neurons = [np.random.randint(2, size=nnn) for _ in range(100)]

population = [SimpleNet(n_neurons=nnn, dist=adjicence[i][0],
                        weights=adjicence[i][1],
                        outputs=np.array([0 for x in range(nnn - 1)] + [1]),
                        inputs=np.array([1, 1] + [0 for x in range(nnn - 2)]),
                        const_neurons=const_neurons[i]) 
              for i in range(100)]

print(len(population))



test_x = [[0, 0], [0, 1], [1, 0], [1, 1]]
test_y = [0, 1, 1, 0]
# self.result_net = None
quality = 0

def create_dateset(num=399):
    result_x = []
    result_y = []
    for _ in range(num):
        index = random.randint(0, 3)
        result_x.append(test_x[index])
        result_y.append(test_y[index])
    return result_x, result_y

def val_score(x_in: list, y_in: list):
    truly_num = 0
    for x, y in zip(x_in, y_in):
        temp_y = 1 if y > 0.5 else 0
        if x == temp_y:
            truly_num += 1
    result = truly_num/len(x_in)
    return result



# def fit(quality=0.9, limit_steps=500):
quality=0.9
limit_steps=500
current_quality = 0
step = 0
while ((not (current_quality > quality)) and 
        (not (step > limit_steps))):
    all_scores = []
    X, Y = create_dateset()
    for i, p in enumerate(population):
        predicts = []
        for j in range(len(X)):
            p.reset()
            out = p.predict(X=X[j])
            predicts.append(out)
            print(j)
        score = val_score(Y, predicts)
        print(f'score: {score}')
        all_scores.append(score)
    paired_sorted = sorted(zip(all_scores, population),
                            key = lambda x: x[0])
    c1, c2 = zip(*paired_sorted)
    current_quality = c1[-1]
    print(current_quality)
    break
#     population = []
#     new_age = list(c2[int(0.8*self.n_childs):])
#     new_mutants = mutation(new_age)  # 50
#     new_childs = sex(new_age)  # 30
#     population = new_age + new_mutants + new_childs
#     step += 1
#     print(f'step: {step}, current_quality: {current_quality}')
# result_net = new_age[-1]
# print(current_quality)
# print(result_net.dist)
# print(result_net.weights)
