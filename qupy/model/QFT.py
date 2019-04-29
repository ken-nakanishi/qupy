import numpy as np 
from qupy.qubit import Qubits
from qupy.operator import *
from qupy.circuit import Gate

def swap_all(q):
	for x in range(q.size // 2):
		q.gate(swap, target = (x, q.size - x - 1))
	return q

def swap_all_circuit(size):
	circuit = []
	for x in range(size // 2):
		circuit.append(Gate(swap, target = (x, size - x - 1)))
	return circuit

def QFT(q):
	for x in range(q.size):
		q.gate(H, target = x)
		for y in range(x + 1, q.size):
			theta = (0.5 ** (y - x + 1)) * 2 * np.pi
			q.gate(phase_shift(theta), target = x, control = y)
	return swap_all(q)


def QFT_circuit(size):
	circuit = []
	for x in range(size):
		circuit.append(Gate(operator = H, target = x))
		for y in range(x + 1, size):
			theta = (0.5 ** (y - x + 1)) * 2 * np.pi
			circuit.append(Gate(operator = phase_shift(theta), target = x, control = y))
	return circuit + swap_all_circuit(size)


