// RA, 2019-12-11

namespace Quantum.Gates {
	open Microsoft.Quantum.Intrinsic;
	open Microsoft.Quantum.Convert;
	open Microsoft.Quantum.Diagnostics;

	// Compare the transformation X with the composition HZH
	// Return True iff the measurments agree
	operation X_vs_HZH() : Bool {
		mutable agree = false;

		using ((q1, q2) = (Qubit(), Qubit())) {
			// Prepare two qubits in the same state
			H(q1);
			CNOT(q1, q2);

			// Transform the first qubit
			X(q1);

			// Transform the second qubit
			H(q2);
			Z(q2);
			H(q2);

			// Measure the two qubits
			let r1 = M(q1);
			let r2 = M(q2);

			set agree = (r1 == r2);

			// Check that the measurements agree
			// https://docs.microsoft.com/en-us/qsharp/api/qsharp/microsoft.quantum.diagnostics.fact?view=qsharp-preview
			Fact(agree, "Expect the same state for both qubits");

			ResetAll([q1, q2]);
		}

		return agree;
	}

	// Compute the average output of f over n runs
	operation Average(f : (() => Bool), n : Int) : Double {
		mutable s = 0;
		for (i in 1..n) {
			let r = f();
			set s = s + (r ? 1 | 0);
		}
		return (IntAsDouble(s) / IntAsDouble(n));
	}

	operation Test() : Unit {
		let repetitions = 1000;
		let a = Average(X_vs_HZH, repetitions);
		Message($"Average: {a}");
	}
}
