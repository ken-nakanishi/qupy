# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import numpy as np
import math
import cmath
import pytest
from qupy.hamiltonian import *
from qupy.operator import X, Y, Z


def test_init():
    n_qubit = 2
    coefs = [1, 2, 3, 4]
    ops = ["IZ", "YI", "XX", "II"]
    H = Hamiltonian(coefs, ops)
    assert all([coefs[i] == H.coefs[i] for i in range(len(coefs))])
    assert all([ops[i] == H.ops[i] for i in range(len(ops))])


def test_append()
    n_qubit = 2
    coefs = [1, 2, 3, 4]
    ops = ["IZ", "YI", "XX", "II"]
    H = Hamiltonian(coefs[:-1], ops[:-1])
    H.append(coefs[-1], ops[-1])
    assert all([coefs[i] == H.coefs[i] for i in range(len(coefs))])
    assert all([ops[i] == H.ops[i] for i in range(len(ops))])


def test_get_matrix():
    n_qubit = 2
    coefs = [1, 2, 3, 4]
    ops = ["IZ", "YI", "XX", "II"]
    I = np.array([[1, 0], [0, 1]])
    test_matrix = np.kron(I, Z) + 2 * np.kron(Y, I) + 3 * \
        np.kron(X, X) + 4 * np.kron(I, I)
    H = Hamiltonian(coefs[:-1], ops[:-1])
    H_matrix = H.get_matrix(ifdense = True)
    assert np.allclose(test_matrix, H_matrix)