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

    def gate(self, operator, target, control=None, control_sub=None):
        '''
        :param (numpy.array or cupy.ndarray) operator: unitary operator
        :param (None or int or tuple of int) target: operated qubits
        :param (None or int or tuple of int) control: operate target qubits where all control qubits are 1
        :param (None or int or tuple of int) control_sub: operate target qubits where all control qubits are 0
        '''
        xp = self.xp

        if isinstance(target, int):
            target = (target,)
        if isinstance(control, int):
            control = (control,)

        self.data = xp.array(self.data).astype(self.dtype).reshape([2] * int(math.log2(self.data.size)))
        operator = xp.array(operator).astype(self.dtype).reshape([2] * int(math.log2(operator.size)))

        if operator.ndim != len(target) * 2:
            print('You must set operator.size==exp(len(target)*2)')
            sys.exit(-1)

        c_slice = [slice(None)] * self.size
        if control is not None:
            for _c in control:
                c_slice[_c] = slice(1, 2)
        if control_sub is not None:
            for _c in control_sub:
                c_slice[_c] = slice(0, 1)

        c_index = list(range(self.size))
        t_index = list(range(self.size))
        for i, _t in enumerate(target):
            t_index[_t] = self.size + i
        o_index = [*target, *list(range(self.size, self.size + len(target)))]

        # use following code when numpy bug is removed
        # self.data[c_slice] = xp.einsum(operator, o_index, self.data[c_slice], c_index, t_index)
        '''
        >>> np.einsum(
            np.array([[[[[[[[[[[[[[[[1]]]]]]]]]]]]]]]]),
            [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
            np.array([[[[[[[[[[[[[[[[1]]]]]]]]]]]]]]]]),
            [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
            [0,1,2,3,4,5,6,7,8,9,10,14,15,16,17,18,19,20,21,22,23,24,25]
        )
        array([[[[[[[[[[[[[[[[[[[[[[[1]]]]]]]]]]]]]]]]]]]]]]])
        
        >>> np.einsum(
            np.array([[[[[[[[[[[[[[[[[1]]]]]]]]]]]]]]]]]),
            [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
            np.array([[[[[[[[[[[[[[[[[1]]]]]]]]]]]]]]]]]),
            [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26],
            [0,1,2,3,4,5,6,7,8,9,10,14,15,16,17,18,19,20,21,22,23,24,25,26]
        )
        ValueError: invalid subscript '{' in einstein sum subscripts string, subscripts must be letters
        '''

        # alternative code
        if np.max(t_index) <= 25:
            self.data[c_slice] = xp.einsum(operator, o_index, self.data[c_slice], c_index, t_index)
        else:
            character = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
            o_index = ''.join([character[i] for i in o_index])
            c_index = ''.join([character[i] for i in c_index])
            t_index = ''.join([character[i] for i in t_index])
            subscripts = '{},{}->{}'.format(o_index, c_index, t_index)
            self.data[c_slice] = xp.einsum(subscripts, operator, self.data[c_slice])

    def projection(self, target):
        xp = self.xp

        data = xp.split(self.data, [1], axis=target)
        if xp == np:
            p = [xp.sum(data[i] ** 2).real for i in (0, 1)]  # xp.square has bug when it get complex128's array
            obs = xp.random.choice([0, 1], p=p)
        else:
            p = [np.asscalar(cupy.asnumpy(xp.sum(data[i] ** 2).real)) for i in (0, 1)]  # difference between numpy and cupy
            obs = np.asscalar(cupy.asnumpy(xp.random.choice([0, 1], size=1, p=p)))

        q_index = [1] * self.size
        q_index[target] = 2
        self.data = xp.tile(data[obs] / p[obs], q_index)
        return obs

    def info(self, data=True, shape=False):
        if data:
            print('data: {}'.format(self.data.flatten()))
        if shape:
            print('shape: {}'.format(self.data.shape))


if __name__ == '__main__':
    I = np.array([[1, 0], [0, 1]])
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    H = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
    S = np.array([[1, 0], [0, 1j]])

    rx = lambda phi: math.cos(phi / 2) * I - 1j * math.sin(phi / 2) * X
    ry = lambda phi: math.cos(phi / 2) * I - 1j * math.sin(phi / 2) * Y
    rz = lambda phi: math.cos(phi / 2) * I - 1j * math.sin(phi / 2) * Z

    sqrt_not = np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2
    phase = lambda phi: np.array([[1, 0], [0, cmath.exp(1j * phi)]])
    g_phase = lambda phi: cmath.exp(1j * phi)

    swap = np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]])

    iswap = np.array([[1, 0, 0, 0],
                      [0, 0, 1j, 0],
                      [0, 1j, 0, 0],
                      [0, 0, 0, 1]])

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
