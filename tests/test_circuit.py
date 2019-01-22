import numpy as np 
from qupy.qubit import Qubits
from qupy.operator import *
from qupy.circuit import Gate
from qupy.model.QFT import *

size = 4
q = Qubits(size)
qft_circuit = QFT_circuit(size)
q.apply_circuit(qft_circuit)
print(q.get_state())
q.apply_inverse_circuit(qft_circuit)
print(q.get_state())
q = QFT(q)
print(q.get_state())

