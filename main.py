import numpy as np
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
        # else:
        #     if xp.get_array_module(self.data) == np:
        #         self.data = xp.array(self.data)
        #     if xp.get_array_module(operator) == np:
        #         self.data = xp.array(operator)
        # operator = operator.astype(self.dtype)

        c_index = [slice(None)] * self.size
        if control is not None:
            if isinstance(control, int):
                c_index[control] = slice(1, 2)
            else:
                for _c in control:
                    c_index[_c] = slice(1, 2)

        t_index = list(range(self.size))

        if isinstance(target, int):
            t_index[target] = self.size
            o_index = [target, self.size]
        else:
            for i, _t in enumerate(target):
                t_index[_t] = self.size + i
            o_index = [*target, *list(range(self.size, self.size + len(target)))]

        self.data[c_index] = xp.einsum(operator, o_index,
                                       self.data[c_index], list(range(self.size)),
                                       t_index, **opt)

    def hadamard(self, target):
        operator = self.xp.array([[1, 1], [1, -1]]) / cmath.sqrt(2)
        self.gate(operator, target)

    def cnot(self, target, control):
        operator = self.xp.array([[0, 1], [1, 0]])
        self.gate(operator, target, control)

    def pauli_x(self, target):
        operator = self.xp.array([[0, 1], [1, 0]])
        self.gate(operator, target)

    def pauli_y(self, target):
        operator = self.xp.array([[0, -1j], [1j, 0]])
        self.gate(operator, target)

    def pauli_z(self, target):
        operator = self.xp.array([[1, 0], [0, -1]])
        self.gate(operator, target)

    def sqrt_not(self, target):
        operator = self.xp.array([[1+1j, 1-1j], [1-1j, 1+1j]]) / 2
        self.gate(operator, target)

    def phase_shift(self, psi, target):
        operator = self.xp.array([[1, 0], [0, cmath.exp(-1j * psi)]]) / 2
        self.gate(operator, target)

    def swap(self, target):
        if len(target) != 2:
            print('len(target) must be 2.')
            sys.exit(-1)
        operator = xp.zeros((2, 2, 2, 2))
        operator[0, 0, 0, 0] = 1
        operator[0, 1, 1, 0] = 1
        operator[1, 0, 0, 1] = 1
        operator[1, 1, 1, 1] = 1
        self.gate(operator, target)

    def sqrt_swap(self, target):
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

    def toffoli(self, target, control):
        if len(control) != 2:
            print('toffoli must have 2 controls.')
            sys.exit(-1)
        operator = self.xp.array([[0, 1], [1, 0]])
        self.gate(operator, target, control)

    def fredkin(self, target, control):
        if len(target) != 2:
            print('len(target) must be 2.')
            sys.exit(-1)
        operator = xp.zeros((2, 2, 2, 2))
        operator[0, 0, 0, 0] = 1
        operator[0, 1, 1, 0] = 1
        operator[1, 0, 0, 1] = 1
        operator[1, 1, 1, 1] = 1
        self.gate(operator, target, control)

    def ising(self, psi, target):
        if len(target) != 2:
            print('len(target) must be 2.')
            sys.exit(-1)
        operator = xp.zeros((2, 2, 2, 2))
        operator[0, 0, 0, 0] = 1
        operator[0, 0, 1, 1] = 1
        operator[1, 1, 0, 0] = 1
        operator[1, 1, 1, 1] = 1
        operator[0, 1, 0, 1] = -1j * cmath.exp(-1j * psi)
        operator[0, 1, 1, 0] = -1j
        operator[1, 0, 0, 1] = -1j
        operator[1, 0, 1, 0] = -1j * cmath.exp(-1j * psi)
        operator /= cmath.sqrt(2)
        self.gate(operator, target)


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
    swap = xp.zeros((2,2,2,2))
    swap[0,0,0,0] = 1
    swap[0,1,1,0] = 1
    swap[1,0,0,1] = 1
    swap[1,1,1,1] = 1
    q.gate(swap, target=(1,2))
    print(q.data)
    print(q.data.shape)
