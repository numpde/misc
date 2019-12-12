# RA, 2019-12-10

import qsharp
import numpy as np
from math import sin, cos

from Quantum.Uncer import MeasureX, MeasureZ

for theta in np.linspace(0, np.pi, 10):
	xx = np.var([MeasureX.simulate(theta=theta) for __ in range(1000)])
	zz = np.var([MeasureZ.simulate(theta=theta) for __ in range(1000)])
	print(F"theta = {theta:.3}, xx + zz = {(xx + zz):.3} = {xx:.3} + {zz:.3}, expected: {(1 - sin(theta) ** 2) / 4 :.3} + {(sin(theta) ** 2) / 4 :.3}")
