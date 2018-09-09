# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import numpy as np
import math
import cmath


I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1], [1, 0]]) * 1j
Z = np.array([[1, 0], [0, -1]])
H = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
S = np.array([[1, 0], [0, 1j]])
T = np.array([[1, 0], [0, cmath.exp(1j * np.pi / 4)]])
Sdag = np.conj(S.T)
Tdag = np.conj(T.T)

rx = lambda phi: math.cos(phi/2) * I - 1j * math.sin(phi/2) * X
ry = lambda phi: math.cos(phi/2) * I - 1j * math.sin(phi/2) * Y
rz = lambda phi: math.cos(phi/2) * I - 1j * math.sin(phi/2) * Z

sqrt_not = np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2
phase_shift = lambda phi: np.array([[1, 0], [0, cmath.exp(1j * phi)]])


def swap():
    operator = np.zeros((2, 2, 2, 2))
    operator[0, 0, 0, 0] = 1
    operator[0, 1, 1, 0] = 1
    operator[1, 0, 0, 1] = 1
    operator[1, 1, 1, 1] = 1
    return operator


def sqrt_swap():
    operator = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    operator[0, 0, 0, 0] = 1
    operator[0, 1, 0, 1] = (1 + 1j) / 2
    operator[1, 0, 1, 0] = (1 + 1j) / 2
    operator[0, 1, 1, 0] = (1 - 1j) / 2
    operator[1, 0, 0, 1] = (1 - 1j) / 2
    operator[1, 1, 1, 1] = 1
    return operator


swap = swap()
sqrt_swap = sqrt_swap()

# alias
sqrt_X = sqrt_not
sqrt_Z = S
sqrt_Zdag = Sdag


if __name__ == '__main__':
    print(swap.reshape((4,4)))
    print(sqrt_swap.reshape((4,4)))
