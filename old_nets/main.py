
import random
#from heapq import nsmallest
import copy


class Neuron:
    def __init__(self, n_id, out_neurons: list = [],
                 out_weights: list = [], th=1,
                 rep_counter_start=2, out_dist: list = [],
                 relax_val=0.2):
        self.n_id = n_id
        self.out_neurons = out_neurons
        self.out_weights = out_weights
        self.out_dist = out_dist
        self.accumulator = 0.0
        self.repolarization_counter = 0
        self.rep_counter_start = rep_counter_start
        self.state_repolar = False
        self.relax_val = relax_val
        self.th = th

    def get_impulse(self, impulse):
        self.accumulator += impulse

    def tik(self, is_out, tik_num):
        if self.state_repolar:
            if self.repolarization_counter > 0:
                self.repolarization_counter -= 1
            if self.repolarization_counter == 0:
                self.state_repolar = False
            self.accumulator = 0.0
            return None
        else:
            if self.accumulator >= self.th:
                temp_out = self.accumulator
                self.accumulator = 0
                self.state_repolar = True
                self.repolarization_counter = self.rep_counter_start
                out_signals = []
                for i in range(len(self.out_neurons)):
                    signal = Signal(
                            self.out_neurons[i],
                            self.out_weights[i],
                            self.out_dist[i],
                            tik_num=tik_num)
                    out_signals.append(signal)
                if is_out:
                    return temp_out
                else:
                    return out_signals
            else:
                self.accumulator -= self.relax_val
                return None


class LockedNeuron(Neuron):
    def tik(self, is_out, tik_num):
        self.accumulator = 0
        out_signals = []
        if not is_out:
            for i in range(len(self.out_neurons)):
                signal = Signal(
                        self.out_neurons[i],
                        self.out_weights[i],
                        self.out_dist[i],
                        tik_num=tik_num)
                out_signals.append(signal)
            return out_signals
        else:
            return 0


class Signal:
    def __init__(self, n_dest, value, way_counter, tik_num=0):
        self.n_dest = n_dest
        self.value = value
        self.way_counter = way_counter
        self.is_dead = False
        self.tik_num = tik_num

    def tik(self, dest_neuron):
        self.way_counter -= 1
        if self.way_counter <= 0:
            # try:
            dest_neuron.get_impulse(self.value)
            # except:
            #     print(isinstance(self.value, list))
            #     print(self.value)
            #     raise
            self.is_dead = True


class Net:
    def __init__(self, n_inputs, n_outputs, n_neurons):
        self.signals = []
        self.neurons = []
        self.inputs = []
        self.output = None
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_neurons = n_neurons
        self.finish = False
        self.out_result = 0

    def reset(self):
        self.signals = []
        for n in self.neurons:
            n.accumulator = 0.0
            n.repolarization_counter = 0
            n.state_repolar = False
        self.out_result = 0

    def create_neurons(self):
        for i in range(self.n_neurons):
            self.neurons.append(Neuron(i))

    def make_inputs(self, percents=0.1):
        for i in range(self.n_inputs):
            self.inputs.append(random.randint(0, self.n_neurons - 1))
        # for n in self.neurons:
        #     if ((n.n_id not in self.inputs) and 
        #             (random.random() < percents)):
        #         self.inputs.appends(n.n_id)
        ###DEBUG###
        #print(f'make_inputs, self.inputs: {self.inputs}')
        ###END DEBUG###

    def make_neurons_connections(self, percents=0.1):
        for n in self.neurons:
            if n.n_id != self.output:
                outs = [random.randint(0, self.n_neurons - 1) for x 
                        in range(int(self.n_neurons*percents))]
                outs = [x for x in outs if x != n.n_id]
                n.out_neurons = outs[:]
                n.out_weights = [(random.random() - 0.5) * 2 for _
                        in range(int(self.n_neurons*percents))]
                n.out_dist = [1 for _
                        in range(int(self.n_neurons*percents))]
        ###DEBUG###
        #print(f'make_conns, output: {self.output}')
        # print(f'make_conns, output n - outputs: {self.neurons[self.output].out_neurons}')
        # for n in self.neurons:
        #     print(f'make_conns, neuron: {n.n_id}, conns: {n.out_neurons}')
        ###END DEBUG###

    def make_outputs(self):
        done = False
        while not done:
            n_check = random.randint(0, self.n_neurons - 1)
            if n_check not in self.inputs:
                done = True
                self.output = n_check

    def start_calc(self, input_vals: list):
        # запускаем стартовые сигналы
        for i in range(self.n_inputs):
            s = Signal(self.inputs[i], input_vals[i], 1)
            self.signals.append(s)
        # for i in range(self.n_inputs):
        #     s = Signal(self.inputs[i], input_vals[i], 1)
        #     self.signals.append(s)
        # for i in range(self.n_inputs):
        #     s = Signal(self.inputs[i], input_vals[i], 1)
        #     self.signals.append(s)
        # for i in range(self.n_inputs):
        #     s = Signal(self.inputs[i], input_vals[i], 1)
        #     self.signals.append(s)
        # pass

    def tik(self, tik_num):
        #print('net tik')
        for s in self.signals:
            s.tik(self.neurons[s.n_dest])
        temp = [x for x in self.signals if not x.is_dead]
        self.signals = temp[:]

        for n in self.neurons:
            if n.n_id != self.output:
                new_signals = n.tik(False, tik_num)
                if new_signals:
                    self.signals += new_signals
            else:
                res = n.tik(True, tik_num)
                if res:
                    self.out_result += res

    def calc_net(self, limit_steps=20):
        for _ in range(limit_steps):
            self.tik(_)
            if not self.signals:
                self.finish = True
                break
        return self.out_result


class Evolution:
    def __init__(self, n_childs=100, n_neurs=30, perc=0.3):
        self.parents = []
        self.n_childs = n_childs
        self.n_neurs = n_neurs
        for _ in range(self.n_childs):
            net = Net(2, 1, self.n_neurs)
            net.create_neurons()
            net.make_inputs()
            net.make_outputs()
            net.make_neurons_connections(percents=perc)
            self.parents.append(net)
        self.childs = []
        self.test_x = [[0, 0], [0, 1], [1, 0], [1, 1]]
        self.test_y = [0, 1, 1, 0]
        self.result_net = None
        self.quality = 0

    def create_dateset(self, num=399):
        result_x = []
        result_y = []
        for _ in range(num):
            index = random.randint(0, 3)
            result_x.append(self.test_x[index])
            result_y.append(self.test_y[index])
        return result_x, result_y

    def val_score(self, x_in: list, y_in: list):
        truly_num = 0
        for x, y in zip(x_in, y_in):
            temp_y = 1 if y > 0.5 else 0
            if x == temp_y:
                truly_num += 1
        # print(truly_num)
        # print(len(x_in))
        # raise Exception()
        result = truly_num/len(x_in)
        return result

    def mutation(self, population, n=50, rate=0.1):
        result_nets = []
        for i in range(n):
            cur = copy.deepcopy(population[
                            random.randint(0, len(population) - 1)])
            for neuron in cur.neurons:
                rand_choice = random.random()
                if neuron.n_id != cur.output and rand_choice < rate:
                    outs = [random.randint(0, cur.n_neurons - 1) for x 
                            in range(int(cur.n_neurons*0.6))]
                    outs = [x for x in outs if x != neuron.n_id]
                    neuron.out_neurons = outs[:]
                    neuron.out_weights = [(random.random() - 0.5) * 2 for _
                            in range(int(cur.n_neurons*0.6))]
                    neuron.out_dist = [1 for _
                        in range(int(cur.n_neurons*0.6))]
            for neuron_index in range(len(cur.neurons)):
                rand_choice = random.random()
                if cur.neurons[neuron_index].n_id != cur.output and rand_choice < rate:
                    new_neur = LockedNeuron(cur.neurons[neuron_index].n_id)
                    outs = [random.randint(0, cur.n_neurons - 1) for x 
                            in range(int(cur.n_neurons*0.6))]
                    outs = [x for x in outs if x != neuron.n_id]
                    new_neur.out_neurons = outs[:]
                    new_neur.out_weights = [(random.random() - 0.5) * 2 for _
                            in range(int(cur.n_neurons*0.6))]
                    new_neur.out_dist = [1 for _
                        in range(int(cur.n_neurons*0.6))]
                    cur.neurons[neuron_index] = new_neur
            result_nets.append(cur)
        return result_nets

    def sex(self, population, n=30):
        # popul [net1, net2, ...]
        half = int(n/2)
        result = []

        for _ in range(half):
            new_net = Net(2, 1, self.n_neurs)
            new_net.inputs = population[-1].inputs[:]
            new_net.output = population[-1].output
            # сохраняем и входные нейроны
            for i in range(new_net.n_neurons):
                if i != new_net.output and i not in new_net.inputs:
                    rand_net = random.randint(0, len(population) - 1)
                    new_net.neurons.append(
                            copy.deepcopy(population[rand_net].neurons[i]))
                else:
                    new_net.neurons.append(
                            copy.deepcopy(population[-1].neurons[i]))
            result.append(new_net)

        for _ in range(half, n):
            new_net = Net(2, 1, self.n_neurs)
            for i in range(new_net.n_neurons):
                done = False
                counter = 0
                while not done:
                    rand_net = random.randint(0, len(population) - 1)
                    counter += 1
                    if (population[rand_net].neurons[i].n_id not in
                            population[rand_net].inputs) and \
                            (population[rand_net].neurons[i].n_id != 
                                population[rand_net].output):
                        new_net.neurons.append(
                                copy.deepcopy(population[rand_net].neurons[i]))
                        done = True
                    if counter > 50:
                        new_net.neurons.append(
                                copy.deepcopy(population[rand_net].neurons[i]))
                        done = True
            new_net.output = random.randint(0, new_net.n_neurons - 1)
            for _ in range(new_net.n_inputs):
                done = False
                while not done:
                    new_input = random.randint(0, new_net.n_neurons - 1)
                    if (new_input not in new_net.inputs) and \
                            (new_input != new_net.output):
                        new_net.inputs.append(new_input)
                        done = True
            result.append(new_net)
            
        return result

    def predict(self, input_a=1.0, input_b=1.0):
        self.result_net.reset()
        self.result_net.start_calc([input_a, input_b])
        out = self.result_net.calc_net()
        return out

        # net = Net(2, 1, 30)
        # net.create_neurons()
        # net.make_inputs()
        # net.make_outputs()
        # net.make_neurons_connections(percents=0.3)
        # net.start_calc([input_a, input_b])
        # out = net.calc_net()
        # print(out)
        # return out

    def fit(self, quality=0.9, limit_steps=500):
        # X, Y = self.create_dateset()
        current_quality = 0
        step = 0
        while ((not (current_quality > quality)) and 
                (not (step > limit_steps))):
            all_scores = []
            X, Y = self.create_dateset()
            for i, p in enumerate(self.parents):
                predicts = []
                for j in range(len(X)):
                    # print(p, i)
                    p.reset()
                    p.start_calc(X[j])
                    out = p.calc_net()
                    predicts.append(out)
                    # print(Y)
                #print(predicts)
                score = self.val_score(Y, predicts)
                all_scores.append(score)
                # print(score)
            paired_sorted = sorted(zip(all_scores, self.parents),
                                    key = lambda x: x[0])
            c1, c2 = zip(*paired_sorted)
            current_quality = c1[-1]
            print(c1[0])
            #print(predicts)
            # print(len(all_scores))
            # print(len(self.parents))
            # print(len(c1))
            # print(len(c2))
            self.parents = []
            new_age = list(c2[int(0.8*self.n_childs):])
            # print(len(new_age))
            new_mutants = self.mutation(new_age)  # 50
            # print(type(new_mutants))
            new_childs = self.sex(new_age)  # 30
            # print(type(new_childs))
            # print(type(new_age))
            self.parents = new_age + new_mutants + new_childs
            step += 1
            print(f'step: {step}, current_quality: {current_quality}')
        self.result_net = new_age[-1]
        self.quality = current_quality


evolution = Evolution(n_neurs=7, perc=0.5)
evolution.fit()
nnn = evolution.result_net
for n in nnn.neurons:
    print(n.out_neurons)
print(' ')
print(nnn.inputs)
print(' ')
for n in nnn.neurons:
    print(n.out_weights)

# net = Net(2, 1, 30)
# net.create_neurons()
# net.make_inputs()
# net.make_outputs()
# net.make_neurons_connections(percents=0.4)
# net.start_calc([1.0, 0])
# out = net.calc_net()
# # for n in net.neurons:
# #     print(n.out_neurons, n.out_weights)
# print(out)
# # for s in net.signals:
# #     print(s.tik_num)
# # print(net.signals)
