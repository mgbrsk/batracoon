import time
from logging import info


class Entity():
    """Класс организма.

    """
    def __init__(self,
                 sensor_neuron_amount,
                 connect_neurons_amount,
                 motor_neuron_amount):
        self.life_status: bool = True
        self.sensor_neuron_amount = sensor_neuron_amount
        self.connect_neurons_amount = connect_neurons_amount
        self.motor_neuron_amount = motor_neuron_amount
        self.net = []

        def generate_cluster_network(neuron_amount,
                                     neuron_class,
                                     neuron_counter):
            """Генерирует набор нейронов нужного класса."""
            for _ in range(neuron_amount):
                neuron = Neuron(neuron_counter, neuron_class)
                self.net.append(neuron)
                neuron_counter += 1
            return neuron_counter

        # Для каждого класса генерируем группу нейронов.
        neuron_counter = 0
        neuron_counter = generate_cluster_network(sensor_neuron_amount,
                                                  0,
                                                  neuron_counter)
        neuron_counter = generate_cluster_network(connect_neurons_amount,
                                                  1,
                                                  neuron_counter)
        generate_cluster_network(motor_neuron_amount, 2, neuron_counter)


    def kill(self):
        """Можно избавиться от организма.

        """
        self.life_status = False

    def structure_info(self):
        if self.life_status:
            print(f'q_q{self.sensor_neuron_amount}'
                  f'(__{self.connect_neurons_amount}__)'
                  f'{self.motor_neuron_amount}~~')
        else:
            print(f'x_x{self.sensor_neuron_amount}'
                  f'(__{self.connect_neurons_amount}__)'
                  f'{self.motor_neuron_amount}--')

    def affect(self, signal_input):
        for neuron in range(self.sensor_neuron_amount):
            print(neuron)


class Neuron():
    """Модель нейрона.
        TODO: Сделать его универсальным.

    """
    def __init__(self, number, neuron_type):
        self.number = number
        self.neuron_type = neuron_type
        self.signal_input = 0
        self.signal_output = 0
        self.associated_with = []
    def activate_output(self):
        if self.signal_input > 0.5:
            print('Active!')
            self.signal_output = 1
            self.signal_input = 0
        else:
            print('not active')
            self.signal_output = 0
    # def interaction(self):
    #     if sel

def simulation(organism):
    """Тики симуляции.

    """
    while True:
        print(f'life_status = {organism.life_status}')
        time.sleep(1)