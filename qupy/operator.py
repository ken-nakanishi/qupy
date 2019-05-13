from __future__ import division
import numpy as np
import os
import math
import cmath

dtype = getattr(np, os.environ.get('QUPY_DTYPE', 'complex128'))
device = os.environ.get('QUPY_GPU', -1)

if device >= 0:
    import cupy
    cupy.cuda.Device(device).use()
    xp = cupy
else:
    xp = np


I = xp.array([[1, 0], [0, 1]], dtype=dtype)
X = xp.array([[0, 1], [1, 0]], dtype=dtype)
Y = xp.array([[0, -1j], [1j, 0]], dtype=dtype)
Z = xp.array([[1, 0], [0, -1]], dtype=dtype)
H = xp.array([[1, 1], [1, -1]], dtype=dtype) / math.sqrt(2)
S = xp.array([[1, 0], [0, 1j]], dtype=dtype)
T = xp.array([[1, 0], [0, (1 + 1j) / math.sqrt(2)]], dtype=dtype)
Sdag = xp.array([[1, 0], [0, -1j]], dtype=dtype)
Tdag = xp.array([[1, 0], [0, (1 - 1j) / math.sqrt(2)]], dtype=dtype)
sqrt_not = xp.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=dtype) / 2

# alias
sqrt_X = sqrt_not
sqrt_Z = S
sqrt_Zdag = Sdag


def rx(phi):
    return math.cos(phi / 2) * I - 1j * math.sin(phi / 2) * X


def ry(phi):
    return math.cos(phi / 2) * I - 1j * math.sin(phi / 2) * Y


def rz(phi):
    return math.cos(phi / 2) * I - 1j * math.sin(phi / 2) * Z


def phase_shift(phi):
    return xp.array([[1, 0], [0, cmath.exp(1j * phi)]], dtype=dtype)


def _swap():
    operator = xp.zeros((2, 2, 2, 2), dtype=dtype)
    operator[0, 0, 0, 0] = 1
    operator[0, 1, 1, 0] = 1
    operator[1, 0, 0, 1] = 1
    operator[1, 1, 1, 1] = 1
    return operator


def _sqrt_swap():
    operator = xp.zeros((2, 2, 2, 2), dtype=dtype)
    operator[0, 0, 0, 0] = 1
    operator[0, 1, 0, 1] = (1 + 1j) / 2
    operator[1, 0, 1, 0] = (1 + 1j) / 2
    operator[0, 1, 1, 0] = (1 - 1j) / 2
    operator[1, 0, 0, 1] = (1 - 1j) / 2
    operator[1, 1, 1, 1] = 1
    return operator


def qft(n):
    dim = 2 ** n
    v = xp.arange(dim)
    return np.exp(2j * np.pi * xp.einsum('i,j->ij', v, v) / dim) / math.sqrt(dim)


def iqft(n):
    dim = 2 ** n
    v = xp.arange(dim)
    return np.exp(-2j * np.pi * xp.einsum('i,j->ij', v, v) / dim) / math.sqrt(dim)


swap = _swap()
sqrt_swap = _sqrt_swap()
