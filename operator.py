import numpy as np
import math
import cmath
import sys
try:
    import cupy
except:
    pass


def hadamard():
    return np.array([[1, 1], [1, -1]]) / cmath.sqrt(2)


def pauli_x():
    return np.array([[0, 1], [1, 0]])


def pauli_y():
    return np.array([[0, -1j], [1j, 0]])


def pauli_z():
    return np.array([[1, 0], [0, -1]])


def rx():
    return np.array([[0, 1], [1, 0]])  # todo


def ry(phi):
    return np.array([[0, -1j], [1j, 0]])  # todo


def rz(phi):
    return np.array([[1, 0], [0, cmath.exp(-1j * phi)]])


def sqrt_not():
    return np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2


def phase_shift(phi):
    return np.array([[1, 0], [0, cmath.exp(-1j * phi)]]) / 2


def _swap():
    operator = np.zeros((2, 2, 2, 2))
    operator[0, 0, 0, 0] = 1
    operator[0, 1, 1, 0] = 1
    operator[1, 0, 0, 1] = 1
    operator[1, 1, 1, 1] = 1
    return operator


def swap():
    operator = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
         ]
    )
    return operator


def sqrt_swap():
    operator = np.zeros((2, 2, 2, 2))
    operator[0, 0, 0, 0] = 1
    operator[0, 0, 1, 1] = (1 + 1j) / 2
    operator[1, 1, 0, 0] = (1 + 1j) / 2
    operator[0, 1, 1, 0] = (1 - 1j) / 2
    operator[1, 0, 0, 1] = (1 - 1j) / 2
    operator[1, 1, 1, 1] = 1
    return operator


def cnot():
    return pauli_x()


def toffoli():
    return pauli_x()


def fredkin():
    return swap()


def ising(phi):
    operator = np.zeros((2, 2, 2, 2))
    operator[0, 0, 0, 0] = 1
    operator[0, 0, 1, 1] = 1
    operator[1, 1, 0, 0] = 1
    operator[1, 1, 1, 1] = 1
    operator[0, 1, 0, 1] = -1j * cmath.exp(1j * phi)
    operator[0, 1, 1, 0] = -1j
    operator[1, 0, 0, 1] = -1j
    operator[1, 0, 1, 0] = -1j * cmath.exp(-1j * phi)
    operator /= cmath.sqrt(2)
    return operator
