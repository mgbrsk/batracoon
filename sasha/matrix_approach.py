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
