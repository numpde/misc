# RA, 2019-12-10

import qsharp
import numpy as np

from Quantum.RotBell import Corr
c = 100 * np.mean([Corr.simulate() for __ in range(100)])
print(F"Correlation: {c}%")
