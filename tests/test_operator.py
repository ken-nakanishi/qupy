from __future__ import division
import pytest
import os
import numpy as np
import math
import cmath
from qupy.operator import I, X, Y, Z, H, S, T, Sdag, Tdag,\
    sqrt_not, swap, sqrt_swap, rx, ry, rz, phase_shift, qft, iqft

dtype = getattr(np, os.environ.get('QUPY_DTYPE', 'complex128'))
device = int(os.environ.get('QUPY_GPU', -1))

if device >= 0:
    import cupy
    cupy.cuda.Device(device).use()
    xp = cupy
else:
    xp = np


ket0 = xp.array([1, 0]).reshape((2, 1))
ket1 = xp.array([0, 1]).reshape((2, 1))


def _allclose(x0, x1):
    if device >= 0:
        return np.allclose(xp.asnumpy(x0), xp.asnumpy(x1))
    return np.allclose(x0, x1)


def dot(*args):
    args = list(args)
    while len(args) != 1:
        x = args.pop()
        args[-1] = xp.dot(args[-1], x)
    return args[0]


def test_I():
    assert _allclose(
        xp.dot(I, xp.conj(I.T)),
        np.eye(2)
    )

    assert _allclose(
        xp.dot(I, ket0),
        ket0
    )

    assert _allclose(
        xp.dot(I, ket1),
        ket1
    )


def test_X():
    assert _allclose(
        xp.dot(X, xp.conj(X.T)),
        xp.eye(2)
    )

    assert _allclose(
        xp.dot(X, ket0),
        ket1
    )

    assert _allclose(
        xp.dot(X, ket1),
        ket0
    )


def test_Y():
    assert _allclose(
        xp.dot(Y, xp.conj(Y.T)),
        xp.eye(2)
    )

    assert _allclose(
        xp.dot(Y, ket0),
        1j * ket1
    )

    assert _allclose(
        xp.dot(Y, ket1),
        -1j * ket0
    )


def test_Z():
    assert _allclose(
        xp.dot(Z, xp.conj(Z.T)),
        xp.eye(2)
    )

    assert _allclose(
        xp.dot(Z, ket0),
        ket0
    )

    assert _allclose(
        xp.dot(Z, ket1),
        - ket1
    )


def test_H():
    assert _allclose(
        xp.dot(H, xp.conj(H.T)),
        xp.eye(2)
    )

    assert _allclose(
        xp.dot(H, ket0),
        1 / math.sqrt(2) * ket0 + 1 / math.sqrt(2) * ket1
    )

    assert _allclose(
        xp.dot(H, ket1),
        1 / math.sqrt(2) * ket0 - 1 / math.sqrt(2) * ket1
    )


def test_S():
    assert _allclose(
        xp.dot(S, xp.conj(S.T)),
        xp.eye(2)
    )

    assert _allclose(
        xp.dot(S, ket0),
        ket0
    )

    assert _allclose(
        xp.dot(S, ket1),
        1j * ket1
    )

    assert _allclose(
        xp.dot(S, S),
        Z
    )


def test_T():
    assert _allclose(
        xp.dot(T, xp.conj(T.T)),
        xp.eye(2)
    )

    assert _allclose(
        xp.dot(T, T),
        S
    )

    assert _allclose(
        xp.dot(T, ket0),
        ket0
    )

    assert _allclose(
        xp.dot(T, ket1),
        (1 + 1j) / math.sqrt(2) * ket1
    )


def test_Sdag():
    assert _allclose(
        xp.dot(S, Sdag),
        xp.eye(2)
    )


def test_Tdag():
    assert _allclose(
        xp.dot(T, Tdag),
        xp.eye(2)
    )


def test_rz():
    assert _allclose(
        xp.dot(rz(0.1), xp.conj(rz(0.1).T)),
        xp.eye(2)
    )

    _rz = lambda phi: cmath.exp(0.5j * phi) * rz(phi)

    assert _allclose(_rz(np.pi * 2), I)
    assert _allclose(_rz(np.pi), Z)
    assert _allclose(_rz(np.pi / 2), S)
    assert _allclose(_rz(np.pi / 4), T)
    assert _allclose(_rz(-np.pi / 2), Sdag)
    assert _allclose(_rz(-np.pi / 4), Tdag)


def test_rx():
    assert _allclose(
        xp.dot(rx(0.1), xp.conj(rx(0.1).T)),
        xp.eye(2)
    )

    _rx = lambda phi: cmath.exp(0.5j * phi) * rx(phi)

    assert _allclose(_rx(np.pi * 2), I)
    assert _allclose(_rx(np.pi), X)
    assert _allclose(rx(0.1), dot(H, rz(0.1), H))


def test_ry():
    assert _allclose(
        xp.dot(ry(0.1), xp.conj(ry(0.1).T)),
        xp.eye(2)
    )

    _ry = lambda phi: cmath.exp(0.5j * phi) * ry(phi)

    assert _allclose(_ry(np.pi * 2), I)
    assert _allclose(_ry(np.pi), Y)
    assert _allclose(ry(0.1), dot(S, H, rz(0.1), H, Sdag))


def test_sqrt_not():
    assert _allclose(
        xp.dot(sqrt_not, xp.conj(sqrt_not.T)),
        xp.eye(2)
    )

    assert _allclose(xp.dot(sqrt_not, sqrt_not), X)
    assert _allclose(sqrt_not, dot(H, S, H))


def test_phase_shift():
    assert _allclose(
        xp.dot(phase_shift(0.1), xp.conj(phase_shift(0.1).T)),
        xp.eye(2)
    )

    _rz = lambda phi: cmath.exp(0.5j * phi) * rz(phi)

    assert _allclose(phase_shift(0.1), _rz(0.1))


def test_swap():
    assert _allclose(
        xp.dot(swap.reshape(4, 4), xp.conj(swap.reshape(4, 4).T)),
        xp.eye(4)
    )
    assert _allclose(xp.dot(swap.reshape(4, 4), swap.reshape(4, 4)), xp.eye(4))


def test_sqrt_swap():
    assert _allclose(
        xp.dot(sqrt_swap.reshape(4, 4), xp.conj(sqrt_swap.reshape(4, 4).T)),
        xp.eye(4)
    )
    assert _allclose(xp.dot(sqrt_swap.reshape(4, 4), sqrt_swap.reshape(4, 4)), swap.reshape(4, 4))


def test_qft():
    assert _allclose(qft(2), xp.array([
        [1, 1, 1, 1],
        [1, 1j, -1, -1j],
        [1, -1, 1, -1],
        [1, -1j, -1, 1j]
    ]) / 2)


def test_iqft():
    assert _allclose(iqft(2), xp.conj(qft(2)))
    assert _allclose(iqft(3), xp.conj(qft(3)))
