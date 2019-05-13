from __future__ import division
import pytest
import numpy as np
from qupy import Qubits, Operator

op = Operator()
X = op.X
rx = op.rx
swap = op.swap


# famous formula
def test_swap_is_3cnot():
    q = Qubits(2)
    q.gate(rx(0.1), target=0)
    q.gate(rx(0.1), target=1, control=0)
    q.gate(swap, target=(0, 1))
    psi1 = q.state

    q = Qubits(2)
    q.gate(rx(0.1), target=0)
    q.gate(rx(0.1), target=1, control=0)
    q.gate(X, target=0, control=1)
    q.gate(X, target=1, control=0)
    q.gate(X, target=0, control=1)
    psi2 = q.state

    assert np.allclose(psi1, psi2)
