import time
from logging import info


class Entity():
    def __init__(self):
        self.life_status: bool = True
        # self.doors = doors
        # self.tires = tires

    def kill(self):
        """Можно избавиться от организма.

        """
        self.life_status = False


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
