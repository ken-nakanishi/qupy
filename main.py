import numpy as np
import math
import sys
try:
    import cupy
except:
    pass


class Qubits():
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
        self.data = xp.zeros([2] * self.size, dtype=dtype)
        self.data[tuple([0] * self.size)] = 1

    # def gate(self, operator, k):
    #     xp = self.xp
    #     opt = {}
    #     if xp == numpy:
    #         opt.update({'optimize': True})
    #
    #     indices = list(range(self.size))
    #     indices[k] = self.size
    #     self.data = xp.einsum(operator, [k, self.size], self.data, list(range(self.size)), indices, **opt)

    # def gate(self, operator, target, control=None):
    #     xp = self.xp
    #     opt = {}
    #     # if xp == np:
    #     #     opt.update({'optimize': True})
    #
    #     c_index = [None] * self.size
    #     if control is not None:
    #         c_index[control] = 1
    #     c_index = tuple(c_index)
    #
    #     t_index = list(range(self.size))
    #
    #     if isinstance(target, int):
    #         t_index[target] = self.size
    #     else:
    #         for i, t in enumerate(target):
    #             t_index[t] = self.size + i
    #
    #     print(operator)
    #     print('#####')
    #     print([target, self.size])
    #     print('#####')
    #     print(self.data[c_index])
    #     print('#####')
    #     print(list(range(self.size)))
    #     print('#####')
    #     print(t_index)
    #     print('#####')
    #
    #     self.data[c_index] = xp.einsum(operator, [target, self.size],
    #                                    self.data[c_index], list(range(self.size)),
    #                                    t_index, **opt)

    def gate(self, operator, target, control=None):
        xp = self.xp
        opt = {}
        # if xp == np:
        #     opt.update({'optimize': True})

        c_index = [slice(None)] * self.size
        if control is not None:
            c_index[control] = slice(1,2)

        t_index = list(range(self.size))
        t_index[target] = self.size

        # print(operator)
        # print('#####')
        # print([target, self.size])
        # print('#####')
        # print(c_index)
        # print('#####')
        # print(self.data[c_index])
        # print(type(self.data[c_index]))
        # print(self.data[c_index].shape)
        # print(self.data)
        # print(type(self.data))
        # print(self.data.shape)
        # print('#####')
        # print(list(range(self.size)))
        # print('#####')
        # print(t_index)
        # print('#####')
        #
        # hoge = xp.einsum(operator, [target, self.size],
        #                                self.data[c_index], list(range(self.size)),
        #                                t_index, **opt)
        # print(hoge)
        # print('#####')

        self.data[c_index] = xp.einsum(operator, [target, self.size],
                                       self.data[c_index], list(range(self.size)),
                                       t_index, **opt)

        # self.data = xp.einsum(operator, [target, self.size],
        #                                self.data, list(range(self.size)),
        #                                t_index, **opt)


if __name__ == '__main__':
    q = Qubits(3)
    h = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
    q.gate(h, target=0)
    q.gate(h, target=1)
    # q.gate(h, target=2)
    print(q.data)
    print(q.data.shape)
    print('#####')
    n = np.array([[0, 1], [1, 0]])
    q.gate(n, target=2, control=0)
    print(q.data)
    print(q.data.shape)
