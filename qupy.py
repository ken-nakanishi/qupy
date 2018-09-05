# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import numpy as np
import math
try:
    import cupy
except:
    pass


class Qubits:
    def __init__(self, size, dtype=np.complex128, gpu=-1):
        if gpu >= 0:
            self.xp = cupy
            self.xp.cuda.Device(gpu).use()
        else:
            self.xp = np

        self.size = size
        self.dtype = dtype

        self.data = self.xp.zeros([2] * self.size, dtype=dtype)
        self.data[tuple([0] * self.size)] = 1

    def gate(self, operator, target, control=None, control_0=None):
        '''
        :param (numpy.array or cupy.ndarray) operator: unitary operator
        :param (None or int or tuple of int) target: operated qubits
        :param (None or int or tuple of int) control: operate target qubits where all control qubits are 1
        :param (None or int or tuple of int) control_0: operate target qubits where all control qubits are 0
        '''
        xp = self.xp

        if isinstance(target, int):
            target = (target,)
        if isinstance(control, int):
            control = (control,)
        if isinstance(control_0, int):
            control_0 = (control_0,)

        self.data = xp.asarray(self.data, dtype=self.dtype).reshape([2] * int(math.log2(self.data.size)))
        operator = xp.asarray(operator, dtype=self.dtype).reshape([2] * int(math.log2(operator.size)))

        assert operator.ndim == len(target) * 2, 'You must set operator.size==exp(len(target)*2)'

        c_slice = [slice(None)] * self.size
        if control is not None:
            for _c in control:
                c_slice[_c] = slice(1, 2)
        if control_0 is not None:
            for _c in control_0:
                c_slice[_c] = slice(0, 1)

        c_index = list(range(self.size))
        t_index = list(range(self.size))
        for i, _t in enumerate(target):
            t_index[_t] = self.size + i
        o_index = [*target, *list(range(self.size, self.size + len(target)))]

        # Use following code when numpy bug is removed and cupy can use this einsum format.
        # self.data[c_slice] = xp.einsum(operator, o_index, self.data[c_slice], c_index, t_index)

        # Alternative code
        character = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        o_index = ''.join([character[i] for i in o_index])
        c_index = ''.join([character[i] for i in c_index])
        t_index = ''.join([character[i] for i in t_index])
        subscripts = '{},{}->{}'.format(o_index, c_index, t_index)
        self.data[c_slice] = xp.einsum(subscripts, operator, self.data[c_slice])

    def projection(self, target):
        xp = self.xp

        data = xp.split(self.data, [1], axis=target)
        p = [self.to_scalar(xp.sum(data[i] * xp.conj(data[i])).real) for i in (0, 1)]
        obs = self.to_scalar(xp.random.choice([0, 1], p=p))

        q_index = [1] * self.size
        q_index[target] = 2
        self.data = xp.tile(data[obs] / p[obs], q_index)
        return obs

    def info(self, data=True, shape=False):
        if data:
            print('data: {}'.format(self.data.flatten()))
        if shape:
            print('shape: {}'.format(self.data.shape))

    def to_scalar(self, x):
        if self.xp != np:
            if isinstance(x, cupy.ndarray):
                x = cupy.asnumpy(x)
        if isinstance(x, np.ndarray):
            x = np.asscalar(x)
        return x


if __name__ == '__main__':
    from operators import H, X, rz, swap

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
    q.gate(X, target=0, control=1, control_0=2)
    q.info()
    q.gate(swap, target=(1, 2))
    q.info()
    q.gate(rz(np.pi / 4), target=1)
    q.info()
    q.gate(iswap, target=(1, 2))
    q.info()
    res = q.projection(target=1)
    print(res)
