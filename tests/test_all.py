# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import numpy as np
import math
import cmath
import pytest
from qupy.qubit import Qubits
from qupy.operator import *


# famous formula
def test_swap_is_3cnot():
    q = Qubits(2)
    q.gate(rx(0.1), target=0)
    q.gate(rx(0.1), target=1, control=0)
    q.gate(swap, target=(0, 1))
    psi1 = q.data

    q = Qubits(2)
    q.gate(rx(0.1), target=0)
    q.gate(rx(0.1), target=1, control=0)
    q.gate(X, target=0, control=1)
    q.gate(X, target=1, control=0)
    q.gate(X, target=0, control=1)
    psi2 = q.data

    assert np.allclose(psi1, psi2)
