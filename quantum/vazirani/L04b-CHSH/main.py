# RA, 2019-12-11

import qsharp
import numpy as np
from math import cos, pi

from Quantum.CHSH import Test

print("Running simulation...")

r = np.mean([Test.simulate() for __ in range(10000)])
print(F"Average result: {r}; expected: {cos(pi/8) ** 2}")
