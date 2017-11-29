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
        xp = self.xp

        opt = {}
        if xp == np:
            opt.update({'optimize': True})

        if isinstance(target, int):
            target = (target,)
        if isinstance(control, int):
            control = (control,)

        self.data = xp.array(self.data)
        operator = xp.array(operator).astype(self.dtype)

        for i in range(int(math.log2(operator.shape[0])) - 1, 0, -1):
            operator = np.asarray([np.split(_o, operator.shape[0] // 2, axis=1) for _o in
                                   np.split(operator, operator.shape[0] // 2, axis=0)])

        if (2 ** int(math.log2(operator.ndim)) != operator.ndim) or operator.ndim < 2:
            print('Operator dimention must be 2^n (n=1,2,3,...)')
            print('{} is not 2^n (n=1,2,3,...)'.format(operator.ndim))
            sys.exit(-1)

        if operator.ndim != len(target) * 2:
            print('You must set operator.ndim==len(target)*2')
            sys.exit(-1)

        c_index = [slice(None)] * self.size
        if control is not None:
            for _c in control:
                c_index[_c] = slice(1, 2)

        t_index = list(range(self.size))
        for i, _t in enumerate(target):
            t_index[_t] = self.size + i
        o_index = [*target, *list(range(self.size, self.size + len(target)))]

        self.data[c_index] = xp.einsum(operator, o_index,
                                       self.data[c_index], list(range(self.size)),
                                       t_index, **opt)

    def hadamard(self, target, control=None):
        operator = self.xp.array([[1, 1], [1, -1]]) / cmath.sqrt(2)
        self.gate(operator, target, control)

    def pauli_x(self, target, control=None):
        operator = self.xp.array([[0, 1], [1, 0]])
        self.gate(operator, target, control)

    def pauli_y(self, target, control=None):
        operator = self.xp.array([[0, -1j], [1j, 0]])
        self.gate(operator, target, control)

    def pauli_z(self, target, control=None):
        operator = self.xp.array([[1, 0], [0, -1]])
        self.gate(operator, target, control)

    def rx(self, phi, target, control=None):
        operator = self.xp.array([[0, 1], [1, 0]])  # todo
        self.gate(operator, target, control)

    def ry(self, phi, target, control=None):
        operator = self.xp.array([[0, -1j], [1j, 0]]) # todo
        self.gate(operator, target, control)

    def rz(self, phi, target, control=None):
        operator = self.xp.array([[1, 0], [0, cmath.exp(-1j * phi)]])
        self.gate(operator, target, control)

    def sqrt_not(self, target, control=None):
        operator = self.xp.array([[1+1j, 1-1j], [1-1j, 1+1j]]) / 2
        self.gate(operator, target, control)

    def phase_shift(self, phi, target, control=None):
        operator = self.xp.array([[1, 0], [0, cmath.exp(-1j * phi)]]) / 2
        self.gate(operator, target, control)

    def swap(self, target, control=None):
        if len(target) != 2:
            print('len(target) must be 2.')
            sys.exit(-1)
        operator = xp.zeros((2, 2, 2, 2))
        operator[0, 0, 0, 0] = 1
        operator[0, 1, 1, 0] = 1
        operator[1, 0, 0, 1] = 1
        operator[1, 1, 1, 1] = 1
        self.gate(operator, target, control)

    def sqrt_swap(self, target, control=None):
        if len(target) != 2:
            print('len(target) must be 2.')
            sys.exit(-1)
        operator = xp.zeros((2, 2, 2, 2))
        operator[0, 0, 0, 0] = 1
        operator[0, 0, 1, 1] = (1 + 1j) / 2
        operator[1, 1, 0, 0] = (1 + 1j) / 2
        operator[0, 1, 1, 0] = (1 - 1j) / 2
        operator[1, 0, 0, 1] = (1 - 1j) / 2
        operator[1, 1, 1, 1] = 1
        self.gate(operator, target)

    def cnot(self, target, control):
        self.pauli_x(target, control)

    def toffoli(self, target, control):
        self.pauli_x(target, control)

    def fredkin(self, target, control):
        self.swap(target, control)

    def ising(self, phi, target):
        if len(target) != 2:
            print('len(target) must be 2.')
            sys.exit(-1)
        operator = xp.zeros((2, 2, 2, 2))
        operator[0, 0, 0, 0] = 1
        operator[0, 0, 1, 1] = 1
        operator[1, 1, 0, 0] = 1
        operator[1, 1, 1, 1] = 1
        operator[0, 1, 0, 1] = -1j * cmath.exp(1j * phi)
        operator[0, 1, 1, 0] = -1j
        operator[1, 0, 0, 1] = -1j
        operator[1, 0, 1, 0] = -1j * cmath.exp(-1j * phi)
        operator /= cmath.sqrt(2)
        self.gate(operator, target)

    def measure(self, target, control):
        xp = self.xp

        opt = {}
        if xp == np:
            opt.update({'optimize': True})

        c_index = [slice(None)] * self.size
        if control is not None:
            if isinstance(control, int):
                c_index[control] = slice(1, 2)
            else:
                for _c in control:
                    c_index[_c] = slice(1, 2)

        data = xp.split(self.data[c_index], [1], axis=target)
        p = [xp.sum(xp.square(data[i])).real for i in (0,1)]
        p = [_p / sum(p) for _p in p]
        # print(p[0], p[1])
        obs = xp.random.choice([0, 1], p=[p[0], p[1]])
        q_index = [1] * self.size
        q_index[target] = 2
        self.data[c_index] = xp.tile(data[obs] / p[obs], q_index)
        return obs


if __name__ == '__main__':
    q = Qubits(3)
    xp = q.xp
    h = xp.array([[1, 1], [1, -1]]) / cmath.sqrt(2)
    q.gate(h, target=0)
    q.gate(h, target=1)
    print(q.data)
    print(q.data.shape)
    print('#####')
    n = xp.array([[0, 1], [1, 0]])
    q.gate(n, target=2, control=(0,1))
    print(q.data)
    print(q.data.shape)
    q.swap(target=(1,2))
    print(q.data)
    print(q.data.shape)
    print(q.data.flatten())
    res = q.measure(target=1, control=2)
    print(res)
