# RA, 2019-12-10

import qsharp
import numpy as np

from Quantum.Bell import Corr, Corr_SignBasis, Corr_SignBasis_Simul

r = 100 * np.mean([Corr.simulate() for __ in range(1000)])
print(F"Agreement (0/1 basis): {r}%")

r = 100 * np.mean([Corr_SignBasis.simulate() for __ in range(1000)])
print(F"Agreement (+/- basis): {r}%")

r = np.mean([((-1) ** Corr_SignBasis_Simul.simulate()) for __ in range(1000)])
print(F"Average eigenvalue (+/- basis, simul): {r}")
