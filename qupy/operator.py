from __future__ import division
import numpy as np
import math
import cmath


class Operator:

    def __init__(self, xp=np, dtype=np.complex128):
        self.xp = xp
        self.dtype = dtype

        self.I = self.xp.array([[1, 0], [0, 1]], dtype=dtype)
        self.X = self.xp.array([[0, 1], [1, 0]], dtype=dtype)
        self.Y = self.xp.array([[0, -1j], [1j, 0]], dtype=dtype)
        self.Z = self.xp.array([[1, 0], [0, -1]], dtype=dtype)
        self.H = self.xp.array([[1, 1], [1, -1]], dtype=dtype) / math.sqrt(2)
        self.S = self.xp.array([[1, 0], [0, 1j]], dtype=dtype)
        self.T = self.xp.array([[1, 0], [0, (1 + 1j) / math.sqrt(2)]], dtype=dtype)
        self.Sdag = self.xp.array([[1, 0], [0, -1j]], dtype=dtype)
        self.Tdag = self.xp.array([[1, 0], [0, (1 - 1j) / math.sqrt(2)]], dtype=dtype)
        self.sqrt_not = self.xp.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=dtype) / 2
        self.swap = self._swap()
        self.sqrt_swap = self._sqrt_swap()

        # alias
        self.sqrt_X = self.sqrt_not
        self.sqrt_Z = self.S
        self.sqrt_Zdag = self.Sdag

    def rx(self, phi):
        return math.cos(phi / 2) * self.I - 1j * math.sin(phi / 2) * self.X

    def ry(self, phi):
        return math.cos(phi / 2) * self.I - 1j * math.sin(phi / 2) * self.Y

    def rz(self, phi):
        return math.cos(phi / 2) * self.I - 1j * math.sin(phi / 2) * self.Z

    def phase_shift(self, phi):
        return self.xp.array([[1, 0], [0, cmath.exp(1j * phi)]], dtype=self.dtype)

    def _swap(self):
        operator = self.xp.zeros((2, 2, 2, 2), dtype=self.dtype)
        operator[0, 0, 0, 0] = 1
        operator[0, 1, 1, 0] = 1
        operator[1, 0, 0, 1] = 1
        operator[1, 1, 1, 1] = 1
        return operator

    def _sqrt_swap(self):
        operator = self.xp.zeros((2, 2, 2, 2), dtype=self.dtype)
        operator[0, 0, 0, 0] = 1
        operator[0, 1, 0, 1] = (1 + 1j) / 2
        operator[1, 0, 1, 0] = (1 + 1j) / 2
        operator[0, 1, 1, 0] = (1 - 1j) / 2
        operator[1, 0, 0, 1] = (1 - 1j) / 2
        operator[1, 1, 1, 1] = 1
        return operator


op = Operator()

I = op.I
X = op.X
Y = op.Y
Z = op.Z
H = op.H
S = op.S
T = op.T
Sdag = op.Sdag
Tdag = op.Tdag
sqrt_not = op.sqrt_not
swap = op.swap
sqrt_swap = op.sqrt_swap
sqrt_X = op.sqrt_X
sqrt_Z = op.sqrt_Z
sqrt_Zdag = op.sqrt_Zdag
rx = op.rx
ry = op.ry
rz = op.rz
phase_shift = op.phase_shift
