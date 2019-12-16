// RA, 2019-12-16

// This shows the limitations of `IncrementByModularInteger`
// https://docs.microsoft.com/en-us/qsharp/api/qsharp/microsoft.quantum.arithmetic.incrementbymodularinteger
namespace Quantum.IncrementByModularInteger_Test {
	open Microsoft.Quantum.Intrinsic;
	open Microsoft.Quantum.Arithmetic;

	operation Test() : Unit {
		let n = 3;
		using (x = Qubit[n]) {
			X(x[1]);
			X(x[0]);

			let aa = [4, 3, 2];
			for (a in aa) {
				IncrementByModularInteger(0, a, LittleEndian(x));
				Message($"Mod {a} OK");
			}

			ResetAll(x);
		}
	}
}


namespace Quantum.PeriodFinding {
	open Microsoft.Quantum.Intrinsic;
	open Microsoft.Quantum.Canon;
	open Microsoft.Quantum.Diagnostics;
	open Microsoft.Quantum.Arithmetic;
	open Microsoft.Quantum.Arrays;

	operation U(x: Qubit[], b: Qubit[]) : Unit {
		// Number of qubits
		let n = Length(x);

		// Period of the function is 2^m
		// because the output will only depend
		// on the first m qubits (LittleEndian)
		let m = 3;

		Fact(2 * m < n, "Not enough qubits to construct the function");

		for (k in 0..(m - 1)) {
			CNOT(x[k], b[2 * k]);
		}
	}

	// Return a pair (p, n) where
	// n is the number of qubits and
	// p is the observed multiple of M/r
	operation Test() : (Int, Int) {

		// Result placeholder
		mutable p = 0;

		// Number of qubits
		let n = 8;

		using ((q, b) = (Qubit[n], Qubit[n])) {
			// Prepare uniform superposition
			// Can use H here but QFT leads
			// to a "more self-adjoint" operation overall
			QFTLE(LittleEndian(q));

			// Apply the black box
			U(q, b);

			// Measuring the output is optional
			// by the "principle of deferred measurement"
			//let mb = ForEach(M, b);

			// Now, q is a periodic pulse with period r
			//DumpRegister((), q);

			// Fourier sampling
			QFTLE(LittleEndian(q));

			// Now, q is a periodic pulse with period 2ⁿ / r
			// and, importantly, starting at 0
			//DumpRegister((), q);

			// Observe a multiple of the period
			set p = MeasureInteger(LittleEndian(q));

			ResetAll(q);
			ResetAll(b);
		}

		return (p, n);
	}
}
