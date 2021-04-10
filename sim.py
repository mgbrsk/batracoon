import time
from logging import info


class Entity():
    def __init__(self,
                 sensor_neuron_amount,
                 connect_neurons_amount,
                 motor_neuron_amount):
        self.life_status: bool = True
        # self.doors = doors
        # self.tires = tires
        self.sensor_neuron_amount = sensor_neuron_amount
        self.connect_neurons_amount = connect_neurons_amount
        self.motor_neuron_amount = motor_neuron_amount

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


class Neuron():
    """А попробуем сделать его универсальным.

    """
    def __init__(self, number):
        self.number = number
        self.signal_input = 0
        self.signal_output = 0

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
    while True:
        print(f'life_status = {organism.life_status}')
        time.sleep(1)
