from __future__ import division
import pytest
import os
import numpy as np
from qupy import Qubits
from qupy.operator import I, X, Y, ry, rz, swap

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


def test_init():
    q = Qubits(1)
    assert q.state[0] == 1
    assert q.state[1] == 0
    assert _allclose(
        q.state.flatten(),
        xp.array([1, 0])
    )

    q = Qubits(2)
    assert q.state[0, 0] == 1
    assert q.state[0, 1] == 0
    assert q.state[1, 0] == 0
    assert q.state[1, 1] == 0
    assert _allclose(
        q.state.flatten(),
        xp.array([1, 0, 0, 0])
    )


def test_set_state():
    q = Qubits(2)
    q.set_state('10')
    assert q.state[0, 0] == 0
    assert q.state[0, 1] == 0
    assert q.state[1, 0] == 1
    assert q.state[1, 1] == 0
    assert _allclose(
        q.state.flatten(),
        xp.array([0, 0, 1, 0])
    )
    q.set_state([0, 1, 0, 0])
    assert q.state[0, 0] == 0
    assert q.state[0, 1] == 1
    assert q.state[1, 0] == 0
    assert q.state[1, 1] == 0
    assert _allclose(
        q.state.flatten(),
        xp.array([0, 1, 0, 0])
    )
    q.set_state([[0, 0], [0, 1]])
    assert q.state[0, 0] == 0
    assert q.state[0, 1] == 0
    assert q.state[1, 0] == 0
    assert q.state[1, 1] == 1
    assert _allclose(
        q.state,
        xp.array([[0, 0], [0, 1]])
    )


def test_get_state():
    q = Qubits(2)
    q.set_state('11')
    assert _allclose(
        q.get_state(),
        xp.array([0, 0, 0, 1])
    )
    q.set_state('01')
    assert _allclose(
        q.get_state(flatten=False),
        xp.array([[0, 1], [0, 0]])
    )


def test_gate_single_qubit():
    q = Qubits(1)
    q.gate(Y, target=0)

    assert _allclose(
        q.state,
        xp.array([0, 1j])
    )

    q = Qubits(1)
    q.gate(ry(0.1), target=0)
    q.gate(rz(0.1), target=0)
    psi1 = q.state

    q = Qubits(1)
    q.gate(xp.dot(rz(0.1), ry(0.1)), target=0)
    psi2 = q.state

    q = Qubits(1)
    q.gate(xp.dot(ry(0.1), rz(0.1)), target=0)
    psi3 = q.state

    assert _allclose(psi1, psi2)
    assert not _allclose(psi1, psi3)


def test_gate_single_target():
    q = Qubits(2)
    q.gate(X, target=0)
    assert q.state[0, 0] == 0
    assert q.state[0, 1] == 0
    assert q.state[1, 0] == 1
    assert q.state[1, 1] == 0
    assert _allclose(
        q.state.flatten(),
        xp.array([0, 0, 1, 0])
    )

    q = Qubits(2)
    q.gate(X, target=1)
    assert q.state[0, 0] == 0
    assert q.state[0, 1] == 1
    assert q.state[1, 0] == 0
    assert q.state[1, 1] == 0
    assert _allclose(
        q.state.flatten(),
        xp.array([0, 1, 0, 0])
    )


def test_gate_multi_targets():
    q = Qubits(2)
    q.gate(X, target=0)
    assert q.state[0, 0] == 0
    assert q.state[0, 1] == 0
    assert q.state[1, 0] == 1
    assert q.state[1, 1] == 0
    q.gate(swap, target=(0, 1))
    assert q.state[0, 0] == 0
    assert q.state[0, 1] == 1
    assert q.state[1, 0] == 0
    assert q.state[1, 1] == 0

    q = Qubits(2)
    q.gate(X, target=0)
    q.gate(swap, target=(1, 0))
    assert q.state[0, 0] == 0
    assert q.state[0, 1] == 1
    assert q.state[1, 0] == 0
    assert q.state[1, 1] == 0


def test_gate_control():
    q = Qubits(2)
    q.gate(X, target=0, control=1)
    assert q.state[0, 0] == 1
    assert q.state[0, 1] == 0
    assert q.state[1, 0] == 0
    assert q.state[1, 1] == 0

    q = Qubits(2)
    q.gate(X, target=1)
    q.gate(X, target=0, control=1)
    assert q.state[0, 0] == 0
    assert q.state[0, 1] == 0
    assert q.state[1, 0] == 0
    assert q.state[1, 1] == 1


def test_gate_control_0():
    q = Qubits(2)
    q.gate(X, target=0, control_0=1)
    assert q.state[0, 0] == 0
    assert q.state[0, 1] == 0
    assert q.state[1, 0] == 1
    assert q.state[1, 1] == 0

    q = Qubits(2)
    q.gate(X, target=1)
    q.gate(X, target=0, control_0=1)
    assert q.state[0, 0] == 0
    assert q.state[0, 1] == 1
    assert q.state[1, 0] == 0
    assert q.state[1, 1] == 0


def test_project():
    q = Qubits(2)
    res0 = q.project(target=0)
    res1 = q.project(target=1)
    assert res0 == 0
    assert res1 == 0

    q = Qubits(2)
    q.gate(Y, target=0)
    res0 = q.project(target=0)
    res1 = q.project(target=1)
    assert res0 == 1
    assert res1 == 0

    q = Qubits(2)
    q.gate(Y, target=1)
    res0 = q.project(target=0)
    res1 = q.project(target=1)
    assert res0 == 0
    assert res1 == 1

    q = Qubits(2)
    q.gate(Y, target=0)
    q.gate(Y, target=1)
    res0 = q.project(target=0)
    res1 = q.project(target=1)
    assert res0 == 1
    assert res1 == 1


def test_expect():
    q = Qubits(2)
    res = q.expect(xp.kron(I, I))
    assert res == 1

    q = Qubits(2)
    res = q.expect(xp.kron(I, X))
    assert res == 0

    q = Qubits(2)
    res = q.expect({'YI': 2.5, 'II': 2, 'IX': 1.5, 'IZ': 1})
    assert res == 3

    q = Qubits(2)
    q.set_state('01')
    res = q.expect({'IZ': 2})
    assert res == -2

    q = Qubits(2)
    q.set_state('10')
    res = q.expect({'IZ': 2})
    assert res == 2

    q = Qubits(2)
    q.set_state('01')
    res = q.expect({'ZI': 2})
    assert res == 2

    q = Qubits(2)
    q.set_state('11')
    res = q.expect({'ZZ': 2})
    assert res == 2
