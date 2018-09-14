# -*- coding: utf-8 -*-
import numpy as np

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

    


