from __future__ import division
import pytest
import os
import numpy as np
from qupy import Qubits
from qupy.operator import X, rx, swap

dtype = getattr(np, os.environ.get('QUPY_DTYPE', 'complex128'))
device = int(os.environ.get('QUPY_GPU', -1))

if device >= 0:
    import cupy
    cupy.cuda.Device(device).use()
    xp = cupy
else:
    xp = np


def _allclose(x0, x1):
    if device >= 0:
        return np.allclose(xp.asnumpy(x0), xp.asnumpy(x1))
    return np.allclose(x0, x1)


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

    assert _allclose(psi1, psi2)
