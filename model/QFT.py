import numpy as np 
from qupy.qubit import Qubits
from qupy.operator import *

def swap_all(q):
	for x in range(q.size // 2):
		q.gate(swap, target = (x, q.size - x - 1))
	return q

def QFT(q):
	for x in range(q.size):
		q.gate(H, target = x)
		for y in range(x + 1, q.size):
			theta = (0.5 ** (y - x + 1)) * 2 * np.pi
			q.gate(phase_shift(theta), target = x, control = y)
	return swap_all(q)

