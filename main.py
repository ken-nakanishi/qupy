# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import numpy as np
import math
import cmath
import sys
try:
    import cupy
except:
    pass


class Qubits:
    def __init__(self, size, dtype=np.complex128, gpu=-1):
        if gpu >= 0:
            try:
                self.xp = cupy
                self.xp.cuda.Device(gpu).use()
            except:
                print('Cupy does not exist.')
                sys.exit(-1)
        else:
            self.xp = np

        xp = self.xp
        self.size = size
        self.dtype = dtype

        self.data = xp.zeros([2] * self.size, dtype=dtype)
        self.data[tuple([0] * self.size)] = 1

    def gate(self, operator, target, control=None):
        '''
        :param (numpy.array or cupy.ndarray) operator: unitary operator
        :param (None or int or tuple of int) target: operated qubits
        :param (None or int or tuple of int) control: operate target qubits where all control qubits are 1
        '''
        xp = self.xp

        if isinstance(target, int):
            target = (target,)
        if isinstance(control, int):
            control = (control,)

        self.data = xp.array(self.data).astype(self.dtype)
        operator = xp.array(operator).astype(self.dtype)

        # for i in range(int(math.log2(operator.shape[0])) - 1, 0, -1):
        #     operator = np.asarray([np.split(_o, operator.shape[0] // 2, axis=1) for _o in
        #                            np.split(operator, operator.shape[0] // 2, axis=0)])

        self.data = self.data.reshape([2] * int(math.log2(self.data.size)))
        operator = operator.reshape([2] * int(math.log2(operator.size)))

        # print(operator.real.astype(int))

        # if (2 ** int(math.log2(operator.ndim)) != operator.ndim) or operator.ndim < 2:
        #     print('Operator dimention must be 2^n (n=1,2,3,...)')
        #     print('{} is not 2^n (n=1,2,3,...)'.format(operator.ndim))
        #     sys.exit(-1)

        if operator.ndim != len(target) * 2:
            print('You must set operator.size==exp(len(target)*2)')
            sys.exit(-1)

        c_index = [slice(None)] * self.size
        if control is not None:
            for _c in control:
                c_index[_c] = slice(1, 2)

        t_index = list(range(self.size))
        for i, _t in enumerate(target):
            t_index[_t] = self.size + i
        o_index = [*target, *list(range(self.size, self.size + len(target)))]
        # o_index = o_index[0::3] + o_index[1::3] + o_index[2::3]

        # print(o_index)
        # print(c_index)
        # print(t_index)

        self.data[c_index] = xp.einsum(operator, o_index,
                                       self.data[c_index], list(range(self.size)),
                                       t_index)

    def projection(self, target):
        xp = self.xp

        data = xp.split(self.data, [1], axis=target)
        p = [xp.sum(xp.square(data[i])).real for i in (0,1)]
        obs = xp.random.choice([0, 1], p=p)
        q_index = [1] * self.size
        q_index[target] = 2
        self.data = xp.tile(data[obs] / p[obs], q_index)
        return obs

    def info(self, data=True, shape=False):
        if data:
            print('data: {}'.format(self.data.flatten()))
        if shape:
            print('shape: {}'.format(self.data.shape))


I = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
H = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
S = np.array([[1, 0], [0, 1j]])

rx = lambda phi: math.cos(phi/2) * I - 1j * math.sin(phi/2) * X
ry = lambda phi: math.cos(phi/2) * I - 1j * math.sin(phi/2) * Y
rz = lambda phi: math.cos(phi/2) * I - 1j * math.sin(phi/2) * Z

sqrt_not = np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2
phase = lambda phi: np.array([[1, 0], [0, cmath.exp(1j * phi)]])
g_phase = lambda phi: cmath.exp(1j * phi)

swap = np.array([[1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]])

iswap = np.array([[1,  0,  0, 0],
                  [0,  0, 1j, 0],
                  [0, 1j,  0, 0],
                  [0,  0,  0, 1]])


if __name__ == '__main__':
    q = Qubits(3)
    q.gate(H, target=0)
    q.gate(H, target=1)
    q.info()
    q.gate(X, target=2, control=(0, 1))
    q.info()
    q.gate(swap, target=(1, 2))
    q.info()
    res = q.projection(target=1)
    print(res)
