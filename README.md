# QuPy

QuPy is a quantum circuit simulator for both CPU and GPU.

QuPy uses [CuPy](https://cupy.chainer.org/) to support GPU.

## example

```python
>>> import numpy as np
>>> from qupy.qubit import Qubits
>>> from qupy.operator import H, X, rz, swap

>>> iswap = np.array([[1, 0, 0, 0],
...                   [0, 0, 1j, 0],
...                   [0, 1j, 0, 0],
...                   [0, 0, 0, 1]])

>>> q = Qubits(3)
>>> print(q.data.flatten())
[1.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]

>>> q.gate(H, target=0)
>>> q.gate(H, target=1)
>>> print(q.data.flatten())
[0.5+0.j 0. +0.j 0.5+0.j 0. +0.j 0.5+0.j 0. +0.j 0.5+0.j 0. +0.j]

>>> q.data = [0, 1, 0, 0, 0, 0, 0, 0]
>>> q.gate(X, target=2)
>>> print(q.data.flatten())
[1.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]

>>> q.gate(H, target=0)
>>> q.gate(H, target=1)
>>> q.gate(X, target=2, control=(0, 1))
>>> q.gate(X, target=0, control=1, control_0=2)
>>> q.gate(swap, target=(0, 2))
>>> q.gate(rz(np.pi / 8), target=2, control_0=1)
>>> print(q.data.flatten())
[0.49039264-0.09754516j 0.49039264+0.09754516j 0.        +0.j
 0.5       +0.j         0.        +0.j         0.        +0.j
 0.        +0.j         0.5       +0.j        ]

>>> q.gate(iswap, target=(2, 1))
>>> print(q.data.flatten())
[ 0.49039264-0.09754516j  0.        +0.j         -0.09754516+0.49039264j
  0.5       +0.j          0.        +0.j          0.        +0.j
  0.        +0.j          0.5       +0.j        ]

>>> res = q.projection(target=1)
>>> print(res)
1
```
