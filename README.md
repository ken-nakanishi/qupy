# QuPy

[![Build Status](https://travis-ci.org/ken-nakanishi/qupy.svg?branch=master)](https://travis-ci.org/ken-nakanishi/qupy)

QuPy is a quantum circuit simulator for both CPU and GPU.

The features of QuPy are as follows.
- **fast**: You can experiment with your idea quickly!
- **simple**: You can experiment with your idea easily!

QuPy supports both Python 3 and Python 2.

QuPy uses [CuPy](https://cupy.chainer.org/) to support GPU.

## Install

```bash
pip install qupy
```
or
```bash
pip3 install qupy
```
or
```bash
conda install conda-forge::qupy
```

## GPU
If you use a GPU, set the following environment variables.

- QUPY_DTYPE
    - Data type of the data array.
    - `complex64` or `complex128` or `complex256`
    - If you does not set QUPY_DTYPE, Data type is set `complex128`.
- QUPY_GPU (integer)
    - GPU machine number.
    - Set QUPY_GPU only if you use GPU.

Read the [CuPy documentation](https://docs-cupy.chainer.org/en/stable/) for using CuPy.

## Documents
https://qupy.readthedocs.io/en/latest/ (In preparation. Your contribution is welcome!)

## Development
Your contribution is welcome!

## Example

```python
>>> import numpy as np
>>> from qupy import Qubits
>>> from qupy.operator import H, X, rz, swap

>>> q = Qubits(3)
>>> print(q.get_state())
[1.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]

>>> q.gate(H, target=0)
>>> q.gate(H, target=1)
>>> print(q.get_state())
[0.5+0.j 0. +0.j 0.5+0.j 0. +0.j 0.5+0.j 0. +0.j 0.5+0.j 0. +0.j]

>>> q.set_state('011')
>>> print(q.get_state())
[0.+0.j 0.+0.j 0.+0.j 1.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]

>>> q.gate(X, target=2)
>>> print(q.get_state())
[0.+0.j 0.+0.j 1.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]

>>> q.set_state([0, 1, 0, 0, 0, 0, 0, 0])
>>> print(q.get_state())
[0.+0.j 1.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]

>>> q.gate(X, target=2)
>>> print(q.get_state())
[1.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]

>>> q.gate(H, target=0)
>>> q.gate(H, target=1)
>>> q.gate(X, target=2, control=(0, 1))
>>> q.gate(X, target=0, control=1, control_0=2)
>>> q.gate(swap, target=(0, 2))
>>> q.gate(rz(np.pi / 8), target=2, control_0=1)
>>> print(q.get_state())
[0.49039264-0.09754516j 0.49039264+0.09754516j 0.        +0.j
 0.5       +0.j         0.        +0.j         0.        +0.j
 0.        +0.j         0.5       +0.j        ]

>>> iswap = np.array([[1, 0, 0, 0],
...                   [0, 0, 1j, 0],
...                   [0, 1j, 0, 0],
...                   [0, 0, 0, 1]])

>>> q.gate(iswap, target=(2, 1))
>>> print(q.get_state())
[ 0.49039264-0.09754516j  0.        +0.j         -0.09754516+0.49039264j
  0.5       +0.j          0.        +0.j          0.        +0.j
  0.        +0.j          0.5       +0.j        ]

>>> res = q.project(target=0)
>>> print(res)
0
>>> print(q.get_state())
[ 0.56625665-0.11263545j  0.        +0.j         -0.11263545+0.56625665j
  0.57735027+0.j          0.        +0.j          0.        +0.j
  0.        +0.j          0.        +0.j        ]

>>> q.gate(H, target=1)
>>> q.gate(swap, target=(2, 0), control=1)
>>> print(q.get_state())
[ 0.32075862+0.32075862j  0.40824829+0.j          0.4800492 -0.4800492j
  0.        +0.j          0.        +0.j          0.        +0.j
 -0.40824829+0.j          0.        +0.j        ]

>>> res = q.project(target=1)
>>> print(res)
1
>>> print(q.get_state())
[ 0.        +0.j          0.        +0.j          0.60597922-0.60597922j
  0.        +0.j          0.        +0.j          0.        +0.j
 -0.51534296+0.j          0.        +0.j        ]

>>> hamiltonian = {'XXI': 1, 'IIZ': -0.5}
>>> print(q.expect(hamiltonian))
-0.4999999999999999
>>> hamiltonian = np.kron(np.kron(X, X), X)
>>> print(q.expect(hamiltonian))
0.0
```

## Citation
```
@Misc{QuPy,
  author = {Nakanishi, Ken M},
  title = {{QuPy}: A quantum circuit simulator for both CPU and GPU},
  year = {2018--},
  url = "https://github.com/ken-nakanishi/qupy",
  note = {[Online; accessed <today>]}
}
```

## License
MIT License (see LICENSE file).
