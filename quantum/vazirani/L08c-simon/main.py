# RA, 2019-12-14

import qsharp
import numpy as np
from scipy.linalg import null_space as null

from Quantum.Simon import SampleY
Y = np.vstack([SampleY.simulate() for __ in range(10)])
print(null(Y).T.round())
