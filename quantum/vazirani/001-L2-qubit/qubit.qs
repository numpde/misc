namespace MyFirstQubit {
	// https://docs.microsoft.com/en-us/qsharp/api/qsharp/microsoft.quantum.intrinsic
	open Microsoft.Quantum.Intrinsic;
	open Microsoft.Quantum.Canon;
	open Microsoft.Quantum.Math;

	operation Hello() : Result {
		Message("Hello!");
		return Zero;
	}

	// 'Result' can be Zero or One
	operation GetOne() : Result {
		return One;
	}

	// Set up and measure the qubit "(|0> + |1>) / √2"
	operation MeasureQubit_H0() : Result {
		// This will be the return value
		mutable r = Zero;

		// https://docs.microsoft.com/en-us/quantum/techniques/working-with-qubits?view=qsharp-preview
		using (q = Qubit()) {
			// q is |0>

			// Transform to "(|0> + |1>) / √2"
			H(q);

			// Measure
			set r = M(q);

			// Illegal here:
			// return r;

			// Set to Zero before leaving
			Reset(q);
		}

		return r;
	}

	// Apply a PauliX rotation to |0⟩
	operation MeasureQubit_RX() : Result {
		// Return value
		mutable r = Zero;

		// Expected fraction of "One" measurements
		let frac = 0.75;

		using (q = Qubit()) {
			// Angle of rotation
			let theta = ArcSin(Sqrt(frac)) * 2.0;

			// https://docs.microsoft.com/en-us/qsharp/api/qsharp/microsoft.quantum.intrinsic.r?view=qsharp-preview
			// The new state is:  cos(θ/2) |0⟩ + sin(θ/2) |1⟩
			R(PauliX, theta, q);

			// Measure; expect E[r] = sin²(θ/2) = frac
			set r = M(q);

			Reset(q);
		}

		return r;
	}
}
