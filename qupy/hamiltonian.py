# -*- coding: utf-8 -*-
import numpy as np
from qupy.operator import X, Y, Z
from scipy.sparse import csr_matrix, kron


class Hamiltonian:
    """
    Creats Hamiltonian as a sum of pauli terms.
    $ H = \sum_j coefs[j]*ops[j] $
    Args:
        n_qubit (:class:`int`):
            Number of qubits.
        coefs (:class:`list`):
            list of floats. coefficient of each pauli terms.
        ops (:class:`list`):
            pauli terms
            each element of this argument is a str specifying the pauli term.
            e.g. ops[0] = "IIZZI"

    Attributes:
        n_qubit (:class:`int`):
            Number of qubits.
        coefs (:class:`list`):
            list of floats. coefficient of each pauli terms.
        ops (:class:`list`):
            pauli terms
            each element of this argument is a str specifying the pauli term.
            e.g. ops[0] = "IIZZI"
    """

    def __init__(self, n_qubit, coefs=None, ops=None):
        self.n_qubit = n_qubit
        self.coefs = coefs if coefs is not None else np.array([])
        self.ops = ops if ops is not None else []

    def append(self, coef, op):
        """append(self, coef, op)
        appends an pauli term to the hamiltonian

        Args:
            coef (:class:`float`):
                coefficient of the term you want to add
            op (:class:`str`):
                pauli term represented by a string, e.g. "IIZZI"

        """
        self.coefs.append(coef)
        self.ops.append(op)

    def get_matrix(self, ifdense=False):
        """get_matrix(self)
        get a matrix representation of the Hamiltonian

        Args:
            ifdense (:class:`Bool`): 
                select sparse or dense.
                Default is False.

        Returns:
            :class:`scipy.sparse.csr_matrix` or `numpy.ndarray`:
                matrix representation of the Hamiltonian
        """
        op = self.ops[0]
        coef = self.coefs[0]
        H = self._kron_N(*[self._get_sparse_pauli_matrix(c)
                           for c in op]).multiply(coef)
        for coef, op in zip(self.coefs[1:], self.ops[1:]):
            H += self._kron_N(*[self._get_sparse_pauli_matrix(c)
                                for c in op]).multiply(coef)
        if ifdense:
            H = H.toarray()
        return H

    def _get_sparse_pauli_matrix(self, pauli_char):
        if pauli_char == "I":
            return csr_matrix([[1, 0], [0, 1]])
        elif pauli_char == "X":
            return csr_matrix(X)
        elif pauli_char == "Y":
            return csr_matrix(Y)
        elif pauli_char == "Z":
            return csr_matrix(Z)

    def _kron_N(self, *args):
        if len(args) > 2:
            return kron(args[0], self._kron_N(*args[1:]))
        elif len(args) == 2:
            return kron(args[0], args[1])
        else:
            raise ValueError("kron_N needs at least 2 arguments.")


def expect(q, H):
    """expect(q, H)
    calculates expectation value of H with respect to q

    Args:
        q (:class:`qupy.qubit.Qubits`):
            the state you want to take the expectation
        H (:class:`qupy.hamiltonian.Hamiltonian`)
            the Hamiltonian

    Return:
        :class:`float`: expectation value of H with respect to q
    """
    assert q.size == H.n_qubit
    xp = q.xp
    ret = 0
    org_data = np.copy(q.data)

    character = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    c_index = character[:q.size]
    subscripts = '{},{}'.format(c_index, c_index)
    for coef, op in zip(H.coefs, H.ops):
        for i in range(H.n_qubit):
            if op[i] == "I":
                pass
            elif op[i] == "X":
                q.gate(operator.X, target=i)
            elif op[i] == "Y":
                q.gate(operator.Y, target=i)
            elif op[i] == "Z":
                q.gate(operator.Z, target=i)
        ret += coef * xp.real(xp.einsum(subscripts, np.conj(org_data), q.data))
        q.set_state(np.copy(org_data))
    return ret

if __name__ == '__main__':
    from qubit import Qubits
    import qupy.operator as operator

    ham = Hamiltonian(3, coefs=[2, 1, 3], ops=["XII", "IYI", "IIZ"])
    q = Qubits(3)
    q.set_state("101")
    q.gate(operator.H, target=0)
    q.gate(operator.H, target=1)
    q.gate(operator.S, target=1)
    print(q.get_state())
    print(q.expect(ham))
    print(expect(q, ham))

    from scipy.sparse.linalg import eigsh
    from numpy.linalg import eigh
    ham_matrix = ham.get_matrix(ifdense=True)
    print(ham_matrix)
    eigvals, eigvecs = eigh(ham_matrix)
    # for sparse matrix
    #eigvals, eigvecs = eigsh(ham_matrix)
    print(eigvals)
