# -*- coding: utf-8 -*-
import numpy as np


n_accumulator = np.zeros((5,))
n_signals = np.zeros((5,))
s_weights = np.array([[0, 0, 0, 0, 0],
                      [0, 0, 0, 4, 0],
                      [2, 3, 0, 0, 0],
                      [0, 0, -8, 0, 0],
                      [0, 0, 10, -2, 0]])
s_signals = np.zeros((5, 5))
s_real = np.array([[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0],
                   [0, 0, 1, 1, 0]])

n_accumulator[0] = 30
n_accumulator[1] = 15

for _ in range(50):
    n_signals = np.where(n_accumulator > 1, 1, n_signals)
    temp = s_weights * s_real * n_signals
    n_accumulator = np.sum(temp, axis=1)  # n_accum + temp_sum
    if n_signals[4] > 0:
        print('hop')
    n_signals = np.zeros((5,))

n_accumulator[0] = 30
n_accumulator[1] = 15

for _ in range(50):
    n_signals = np.where(n_accumulator > 1, 1, n_signals)
    temp = s_weights * s_real * n_signals
    n_accumulator = np.sum(temp, axis=1)  # n_accum + temp_sum
    if n_signals[4] > 0:
        print('hop')
    n_signals = np.zeros((5,))

s_cd = np.zeros((5, 5))
s_dw = np.zeros((5, 5))
s_cd_input = np.random.rand(3, 5, 5) * 2 - 1
s_cd_output = np.random.rand(3, 5, 5) * 2 - 1

s_dw_input = np.random.rand(3, 5, 5) * 2 - 1
s_dw_output = np.random.rand(3, 5, 5) * 2 - 1

# расстояние до выхода, верно-неверно, еще байда
n_parameters = np.random.rand(3, 5) * 2 - 1

# s_cd_input = np.array([[[0, 0, 0, 0, 0],
#                         [0, 0, 0, 0, 0],
#                         [1, 0, 0, 17, 0],
#                         [0, 0, 0, 0, 0],
#                         [0, 0, 0, 11, 0]],
#
#                        [[0, 0, 0, 0, 0],
#                         [0, 0, 0, 0, 0],
#                         [2, 0, 0, 0, 0],
#                         [0, 0, 0, 0, 0],
#                         [0, 0, 0, 22, 0]],
#
#                        [[0, 0, 0, 0, 0],
#                         [0, 0, 0, 0, 0],
#                         [3, 0, 0, 0, 0],
#                         [0, 0, 0, 0, 0],
#                         [0, 0, 0, 33, 0]]])
#
# s_cd_output = np.array([[[0, 0, 0, 0, 0],
#                          [0, 0, 0, 0, 0],
#                          [1, 0, 0, 17, 0],
#                          [0, 0, 0, 0, 0],
#                          [0, 0, 0, 11, 0]],
#
#                         [[0, 0, 0, 0, 0],
#                          [0, 0, 0, 0, 0],
#                          [2, 0, 0, 0, 0],
#                          [0, 0, 0, 0, 0],
#                          [0, 0, 0, 22, 0]],
#
#                         [[0, 0, 0, 0, 0],
#                          [0, 0, 0, 0, 0],
#                          [3, 0, 0, 0, 0],
#                          [0, 0, 0, 0, 0],
#                          [0, 0, 0, 33, 0]]])
#
# n_parameters = np.array([[1, 2, 3, 4, 5],
#                          [5, 4, 3, 2, 1],
#                          [3, 0, 0, 3, 0]])

# считаем cd на все синапсы
cd_input = np.transpose(np.transpose(s_cd_input, (0, 2, 1)) *
                        n_parameters.reshape((3, 5, 1)), (0, 2, 1))
cd_output = s_cd_output * n_parameters.reshape((3, 5, 1))
s_cd = np.sum(cd_output, axis=0) + np.sum(cd_input, axis=0)

# считаем dw на все синапсы
dw_input = np.transpose(np.transpose(s_dw_input, (0, 2, 1)) *
                        n_parameters.reshape((3, 5, 1)), (0, 2, 1))
dw_output = s_dw_output * n_parameters.reshape((3, 5, 1))
s_dw = np.sum(dw_output, axis=0) + np.sum(dw_input, axis=0)
s_dw = s_dw * s_real

# получение новых значений реальности s_real
temp_cd_greater = np.where(s_cd > 0.8, 1, 0)
temp_cd_less = np.where(s_cd < 0.2, 0, 1)
s_real_inverse = s_real * (-1) + 1
s_real = s_real + s_real_inverse * temp_cd_greater
s_real = s_real * temp_cd_less

# получение новых весов s_weights
s_weights = s_weights + s_dw * 0.5
