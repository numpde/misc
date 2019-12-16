# RA, 2019-12-16

import qsharp
import numpy as np

from Quantum.PeriodFinding import Test
(pp, n) = zip(*sorted(set([Test.simulate() for __ in range(10)])))
n = max(n)  # Number of bits
M = 2 ** n  # Max encoded integer
gcd = np.gcd.reduce(pp)
print(F"Results: {pp} with gcd {gcd} on {n} bits => period {M / gcd}")
