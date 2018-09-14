# -*- coding: utf-8 -*-
import numpy as np
from qupy.operator import X, Y, Z

class Hamiltonian:
    """
    Creats Hamiltonian as a sum of pauli terms.
    $ H = \sum_j coefs[j]*ops[j]$
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

    def __init__(self, n_qubit, coefs = None, ops = None):
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
        q.set_state(np.copy(org_data))
    return ret
    
if __name__ == '__main__':
    from qubit import Qubits
    import qupy.operator as operator
    ham = Hamiltonian(3, coefs=[2,1,1], ops = ["XII", "IYI", "IIZ"])
    q = Qubits(3)
    q.set_state("101")
    q.gate(operator.H, target = 0)
    q.gate(operator.H, target = 1)
    q.gate(operator.S, target = 1)    
    print(q.get_state())
    print(q.expect(ham))
    print(expect(q,ham))
