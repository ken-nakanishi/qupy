from __future__ import division
import numpy as np
import math
import cmath
import pytest
from qupy.operator import I, X, Y, Z, H, S, T, Sdag, Tdag, sqrt_not, \
    rx, ry, rz, phase_shift, swap, sqrt_swap, qft, iqft


ket0 = np.array([1, 0]).reshape((2, 1))
ket1 = np.array([0, 1]).reshape((2, 1))

def dot(*args):
    args = list(args)
    while len(args) != 1:
        x = args.pop()
        args[-1] = np.dot(args[-1], x)
    return args[0]


def test_I():
    assert np.allclose(
        np.dot(I, np.conj(I.T)),
        np.eye(2)
    )

    assert np.allclose(
        np.dot(I, ket0),
        ket0
    )

    assert np.allclose(
        np.dot(I, ket1),
        ket1
    )


def test_X():
    assert np.allclose(
        np.dot(X, np.conj(X.T)),
        np.eye(2)
    )

    assert np.allclose(
        np.dot(X, ket0),
        ket1
    )

    assert np.allclose(
        np.dot(X, ket1),
        ket0
    )


def test_Y():
    assert np.allclose(
        np.dot(Y, np.conj(Y.T)),
        np.eye(2)
    )

    assert np.allclose(
        np.dot(Y, ket0),
        1j * ket1
    )

    assert np.allclose(
        np.dot(Y, ket1),
        -1j * ket0
    )


def test_Z():
    assert np.allclose(
        np.dot(Z, np.conj(Z.T)),
        np.eye(2)
    )

    assert np.allclose(
        np.dot(Z, ket0),
        ket0
    )

    assert np.allclose(
        np.dot(Z, ket1),
        - ket1
    )


def test_H():
    assert np.allclose(
        np.dot(H, np.conj(H.T)),
        np.eye(2)
    )

    assert np.allclose(
        np.dot(H, ket0),
        1 / math.sqrt(2) * ket0 + 1 / math.sqrt(2) * ket1
    )

    assert np.allclose(
        np.dot(H, ket1),
        1 / math.sqrt(2) * ket0 - 1 / math.sqrt(2) * ket1
    )


def test_S():
    assert np.allclose(
        np.dot(S, np.conj(S.T)),
        np.eye(2)
    )

    assert np.allclose(
        np.dot(S, ket0),
        ket0
    )

    assert np.allclose(
        np.dot(S, ket1),
        1j * ket1
    )

    assert np.allclose(
        np.dot(S, S),
        Z
    )


def test_T():
    assert np.allclose(
        np.dot(T, np.conj(T.T)),
        np.eye(2)
    )

    assert np.allclose(
        np.dot(T, T),
        S
    )

    assert np.allclose(
        np.dot(T, ket0),
        ket0
    )

    assert np.allclose(
        np.dot(T, ket1),
        (1 + 1j) / math.sqrt(2) * ket1
    )


def test_Sdag():
    assert np.allclose(
        np.dot(S, Sdag),
        np.eye(2)
    )


def test_Tdag():
    assert np.allclose(
        np.dot(T, Tdag),
        np.eye(2)
    )


def test_rz():
    assert np.allclose(
        np.dot(rz(0.1), np.conj(rz(0.1).T)),
        np.eye(2)
    )

    _rz = lambda phi: cmath.exp(0.5j * phi) * rz(phi)

    assert np.allclose(_rz(np.pi * 2), I)
    assert np.allclose(_rz(np.pi), Z)
    assert np.allclose(_rz(np.pi / 2), S)
    assert np.allclose(_rz(np.pi / 4), T)
    assert np.allclose(_rz(-np.pi / 2), Sdag)
    assert np.allclose(_rz(-np.pi / 4), Tdag)


def test_rx():
    assert np.allclose(
        np.dot(rx(0.1), np.conj(rx(0.1).T)),
        np.eye(2)
    )

    _rx = lambda phi: cmath.exp(0.5j * phi) * rx(phi)

    assert np.allclose(_rx(np.pi * 2), I)
    assert np.allclose(_rx(np.pi), X)
    assert np.allclose(rx(0.1), dot(H, rz(0.1), H))


def test_ry():
    assert np.allclose(
        np.dot(ry(0.1), np.conj(ry(0.1).T)),
        np.eye(2)
    )

    _ry = lambda phi: cmath.exp(0.5j * phi) * ry(phi)

    assert np.allclose(_ry(np.pi * 2), I)
    assert np.allclose(_ry(np.pi), Y)
    assert np.allclose(ry(0.1), dot(S, H, rz(0.1), H, Sdag))


def test_sqrt_not():
    assert np.allclose(
        np.dot(sqrt_not, np.conj(sqrt_not.T)),
        np.eye(2)
    )

    assert np.allclose(np.dot(sqrt_not, sqrt_not), X)
    assert np.allclose(sqrt_not, dot(H, S, H))


def test_phase_shift():
    assert np.allclose(
        np.dot(phase_shift(0.1), np.conj(phase_shift(0.1).T)),
        np.eye(2)
    )

    _rz = lambda phi: cmath.exp(0.5j * phi) * rz(phi)

    assert np.allclose(phase_shift(0.1), _rz(0.1))


def test_swap():
    assert np.allclose(
        np.dot(swap.reshape(4, 4), np.conj(swap.reshape(4, 4).T)),
        np.eye(4)
    )
    assert np.allclose(np.dot(swap.reshape(4, 4), swap.reshape(4, 4)), np.eye(4))


def test_sqrt_swap():
    assert np.allclose(
        np.dot(sqrt_swap.reshape(4, 4), np.conj(sqrt_swap.reshape(4, 4).T)),
        np.eye(4)
    )
    assert np.allclose(np.dot(sqrt_swap.reshape(4, 4), sqrt_swap.reshape(4, 4)), swap.reshape(4, 4))


def test_qft():
    assert np.allclose(qft(2), np.array([
        [1, 1, 1, 1],
        [1, 1j, -1, -1j],
        [1, -1, 1, -1],
        [1, -1j, -1, 1j]
    ]) / 2)


def test_iqft():
    assert np.allclose(iqft(2), np.conj(qft(2)))
    assert np.allclose(iqft(3), np.conj(qft(3)))
