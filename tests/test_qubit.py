# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import numpy as np
import math
import cmath
import pytest
from qupy.qubit import Qubits
from qupy.operator import *


def test_init():
    q = Qubits(1)
    assert q.data[0] == 1
    assert q.data[1] == 0
    assert np.allclose(
        q.data.flatten(),
        np.array([1, 0])
    )

    q = Qubits(2)
    assert q.data[0, 0] == 1
    assert q.data[0, 1] == 0
    assert q.data[1, 0] == 0
    assert q.data[1, 1] == 0
    assert np.allclose(
        q.data.flatten(),
        np.array([1, 0, 0, 0])
    )


def test_gate_single_qubit():
    q = Qubits(1)
    q.gate(Y, target=0)

    assert np.allclose(
        q.data,
        np.array([0, 1j])
    )

    q = Qubits(1)
    q.gate(ry(0.1), target=0)
    q.gate(rz(0.1), target=0)
    psi1 = q.data

    q = Qubits(1)
    q.gate(np.dot(rz(0.1), ry(0.1)), target=0)
    psi2 = q.data

    q = Qubits(1)
    q.gate(np.dot(ry(0.1), rz(0.1)), target=0)
    psi3 = q.data

    assert np.allclose(psi1, psi2)
    assert not np.allclose(psi1, psi3)


def test_gate_single_target():
    q = Qubits(2)
    q.gate(X, target=0)
    assert q.data[0, 0] == 0
    assert q.data[0, 1] == 0
    assert q.data[1, 0] == 1
    assert q.data[1, 1] == 0
    assert np.allclose(
        q.data.flatten(),
        np.array([0, 0, 1, 0])
    )

    q = Qubits(2)
    q.gate(X, target=1)
    assert q.data[0, 0] == 0
    assert q.data[0, 1] == 1
    assert q.data[1, 0] == 0
    assert q.data[1, 1] == 0
    assert np.allclose(
        q.data.flatten(),
        np.array([0, 1, 0, 0])
    )


def test_gate_multi_targets():
    q = Qubits(2)
    q.gate(X, target=0)
    assert q.data[0, 0] == 0
    assert q.data[0, 1] == 0
    assert q.data[1, 0] == 1
    assert q.data[1, 1] == 0
    q.gate(swap, target=(0, 1))
    assert q.data[0, 0] == 0
    assert q.data[0, 1] == 1
    assert q.data[1, 0] == 0
    assert q.data[1, 1] == 0

    q = Qubits(2)
    q.gate(X, target=0)
    q.gate(swap, target=(1, 0))
    assert q.data[0, 0] == 0
    assert q.data[0, 1] == 1
    assert q.data[1, 0] == 0
    assert q.data[1, 1] == 0


def test_gate_control():
    q = Qubits(2)
    q.gate(X, target=0, control=1)
    assert q.data[0, 0] == 1
    assert q.data[0, 1] == 0
    assert q.data[1, 0] == 0
    assert q.data[1, 1] == 0

    q = Qubits(2)
    q.gate(X, target=1)
    q.gate(X, target=0, control=1)
    assert q.data[0, 0] == 0
    assert q.data[0, 1] == 0
    assert q.data[1, 0] == 0
    assert q.data[1, 1] == 1


def test_gate_control_0():
    q = Qubits(2)
    q.gate(X, target=0, control_0=1)
    assert q.data[0, 0] == 0
    assert q.data[0, 1] == 0
    assert q.data[1, 0] == 1
    assert q.data[1, 1] == 0

    q = Qubits(2)
    q.gate(X, target=1)
    q.gate(X, target=0, control_0=1)
    assert q.data[0, 0] == 0
    assert q.data[0, 1] == 1
    assert q.data[1, 0] == 0
    assert q.data[1, 1] == 0


def test_projection():
    q = Qubits(2)
    res0 = q.projection(target=0)
    res1 = q.projection(target=1)
    assert res0 == 0
    assert res1 == 0

    q = Qubits(2)
    q.gate(Y, target=0)
    res0 = q.projection(target=0)
    res1 = q.projection(target=1)
    assert res0 == 1
    assert res1 == 0

    q = Qubits(2)
    q.gate(Y, target=1)
    res0 = q.projection(target=0)
    res1 = q.projection(target=1)
    assert res0 == 0
    assert res1 == 1

    q = Qubits(2)
    q.gate(Y, target=0)
    q.gate(Y, target=1)
    res0 = q.projection(target=0)
    res1 = q.projection(target=1)
    assert res0 == 1
    assert res1 == 1

