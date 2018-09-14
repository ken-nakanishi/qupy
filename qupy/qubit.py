# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import numpy as np
import math
import qupy.operator as operator
try:
    import cupy
except:
    pass


class Qubits:
    """
    Creating qubits.

    Args:
        size (:class:`int`):
            Number of qubits.
        dtype:
            Data type of the data array.
        gpu (:class:`int`):
            GPU machine number.

    Attributes:
        data (:class:`numpy.ndarray` or :class:`cupy.ndarray`):
            The state of qubits.
        size:
            Number of qubits.
        dtype:
            Data type of the data array.
    """

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

    def set_state(self, state):
        """set_state(self, state)

        Set state.

        Args:
            state (:class:`str` or :class:`list` or :class:`numpy.ndarray` or :class:`cupy.ndarray`):
                If you set state as :class:`str`, you can set state \state>
                (e.g. state='0110' -> \0110>.)
                otherwise, qubit state is set that you entered as state.
        """
        if isinstance(state, str):
            assert len(state) == self.data.ndim, 'There were {} qubits prepared, but you specified {} qubits'.format(
                self.data.ndim, len(state))
            self.data = self.xp.zeros_like(self.data)
            self.data[tuple([int(i) for i in state])] = 1
        else:
            self.data = self.xp.asarray(state, dtype=self.dtype)
            if self.data.ndim == 1:
                self.data = self.data.reshape([2] * self.size)

    def get_state(self, flatten=True):
        """get_state(self, flatten=True)

        Get state.

        Args:
            flatten (:class:`bool`):
                If you set flatten=False, you can get data format used in QuPy.
                otherwise, you get state reformated to 1D-array.
        """
        if flatten:
            return self.data.flatten()
        else:
            return self.data

    def gate(self, operator, target, control=None, control_0=None):
        """gate(self, operator, target, control=None, control_0=None)

        Gate method.

        Args:
            operator (:class:`numpy.ndarray` or :class:`cupy.ndarray`):
                Unitary operator
            target (None or :class:`int` or :class:`tuple` of :class:`int`):
                Operated qubits
            control (None or :class:`int` or :class:`tuple` of :class:`int`):
                Operate target qubits where all control qubits are 1
            control_0 (None or :class:`int` or :class:`tuple` of :class:`int`):
                Operate target qubits where all control qubits are 0
        """
        xp = self.xp

        if np.issubdtype(type(target), np.integer):
            target = (target,)
        if np.issubdtype(type(control), np.integer):
            control = (control,)
        if np.issubdtype(type(control_0), np.integer):
            control_0 = (control_0,)

        self.data = xp.asarray(self.data, dtype=self.dtype)
        operator = xp.asarray(operator, dtype=self.dtype)

        if self.data.ndim == 1:
            self.data = self.data.reshape([2] * self.size)
        if operator.shape[0] != 2:
            operator = operator.reshape([2] * int(math.log2(operator.size)))

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
        # o_index = [*list(range(self.size, self.size + len(target))), *target]
        o_index = list(range(self.size, self.size + len(target))) + list(target)

        # Use following code when numpy bug is removed and cupy can use this einsum format.
        # self.data[c_slice] = xp.einsum(operator, o_index, self.data[c_slice], c_index, t_index)

        # Alternative code
        character = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        o_index = ''.join([character[i] for i in o_index])
        c_index = ''.join([character[i] for i in c_index])
        t_index = ''.join([character[i] for i in t_index])
        subscripts = '{},{}->{}'.format(o_index, c_index, t_index)
        self.data[tuple(c_slice)] = xp.einsum(subscripts, operator, self.data[tuple(c_slice)])

    def projection(self, target):
        """projection(self, target)

        Projection method.

        Args:
            target (None or :class:`int` or :class:`tuple` of :class:`int`):
                projected qubits

        Returns:
            :class:`int`: O or 1.
        """
        xp = self.xp

        self.data = xp.asarray(self.data, dtype=self.dtype)
        if self.data.ndim == 1:
            self.data = self.data.reshape([2] * self.size)

        data = xp.split(self.data, [1], axis=target)
        p = [self._to_scalar(xp.sum(data[i] * xp.conj(data[i])).real) for i in (0, 1)]
        obs = self._to_scalar(xp.random.choice([0, 1], p=p))

        if obs == 0:
            self.data = xp.concatenate((data[obs] / math.sqrt(p[obs]), xp.zeros_like(data[obs])), target)
        else:
            self.data = xp.concatenate((xp.zeros_like(data[obs]), data[obs] / math.sqrt(p[obs])), target)
        return obs

    def _to_scalar(self, x):
        if self.xp != np:
            if isinstance(x, cupy.ndarray):
                x = cupy.asnumpy(x)
        if isinstance(x, np.ndarray):
            x = np.asscalar(x)
        return x

    def expect(self, H):
        """expect(self, H)
        returns expectation value of a hamiltonian H.

        Args:
            H (:class:`qupy.hamiltonian.Hamiltonian`):
                Hamiltonian object that contains the hamiltonian.
        
        Returns:
            :class:`float`: expectation value of H
        """
        assert self.size == H.n_qubit
        xp = self.xp
        ret = 0
        org_data = np.copy(self.data)
        
        character = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        c_index = character[:self.size]
        subscripts = '{},{}'.format(c_index,c_index)
        for coef, op in zip(H.coefs, H.ops):
            for i in range(H.n_qubit):
                if op[i] == "I":
                    pass
                elif op[i] == "X":
                    q.gate(operator.X, target = i)
                elif op[i] == "Y":
                    q.gate(operator.Y, target = i)
                elif op[i] == "Z":
                    q.gate(operator.Z, target = i)
            ret += coef*xp.real(xp.einsum(subscripts, np.conj(org_data), q.data))
            self.set_state(np.copy(org_data))
        return ret

if __name__ == '__main__':
    from qupy.operator import H, X, rz, S, swap
    np.set_printoptions(precision=3, suppress=True, linewidth=1000)

    iswap = np.array([[1, 0, 0, 0],
                      [0, 0, 1j, 0],
                      [0, 1j, 0, 0],
                      [0, 0, 0, 1]])

    q = Qubits(3)
    print(q.data.flatten())

    q.gate(H, target=0)
    q.gate(H, target=1)
    print(q.data.flatten())

    q.data = [0, 1, 0, 0, 0, 0, 0, 0]
    q.gate(X, target=2)
    print(q.data.flatten())

    q.gate(H, target=0)
    q.gate(H, target=1)
    q.gate(X, target=2, control=(0, 1))
    q.gate(X, target=0, control=1, control_0=2)
    q.gate(swap, target=(0, 2))
    q.gate(rz(np.pi / 8), target=2, control_0=1)
    print(q.data.flatten())

    q.gate(iswap, target=(2, 1))
    print(q.data.flatten())

    res = q.projection(target=1)
    print(res)

    from hamiltonian import Hamiltonian
    ham = Hamiltonian(3, coefs=[2,1,1], ops = ["XII", "IYI", "IIZ"])
    q.set_state("000")
    q.gate(H, target = 0)
    q.gate(H, target = 1)
    q.gate(S, target = 1)    
    print(q.get_state())
    print(q.expect(ham))

