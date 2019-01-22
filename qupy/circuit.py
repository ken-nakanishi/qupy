# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import numpy as np
import math
import sys
import qupy.operator

class Gate:

	def __init__(self, gate_type, target, control = None, control_0 = None):
		self.operator = gate_type
		self.target = target
		self.control = control
		self.control_0 = control_0