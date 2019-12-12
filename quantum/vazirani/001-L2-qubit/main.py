# RA, 2019-12-08

# https://www.youtube.com/watch?v=xGAyI5fufmk&list=PL74Rel4IAsETUwZS_Se_P-fSEyEVQwni7&index=5

import qsharp
import numpy as np

from MyFirstQubit import Hello
Hello.simulate()

from MyFirstQubit import GetOne
r = GetOne.simulate()
print(F"GetOne returned '{r}'")

from MyFirstQubit import MeasureQubit_H0
rr = np.mean([MeasureQubit_H0.simulate() for __ in range(10000)])
print(F"Measured values average (H): {rr}")

from MyFirstQubit import MeasureQubit_RX
rr = np.mean([MeasureQubit_RX.simulate() for __ in range(10000)])
print(F"Measured values average (R): {rr}")
