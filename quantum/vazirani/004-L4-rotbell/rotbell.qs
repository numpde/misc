// RA, 2019-12-10

namespace Quantum.RotBell {
	open Microsoft.Quantum.Intrinsic;
	open Microsoft.Quantum.Math as Math;

	// Consider the Bell state |ϕ⁺⟩ = 1/√2 (|00⟩ + |11⟩)
	// Then, for any orthogonal basis u/v, |ϕ⁺⟩ = 1/√2 (|uu⟩ + |vv⟩)
	// Thus, measuring |u⟩ of the first qubit and
	// then measuring |u⟩ of the second qubit will give the same result
	operation Corr() : Bool {
		mutable r = false;

		using ((qc, qt) = (Qubit(), Qubit())) {
			// Prepare Bell state
			H(qc);
			CNOT(qc, qt);

			// Change the measurement direction/basis by rotating the qubits
			Exp([PauliX, PauliX], Math.PI() * 0.1, [qc, qt]);
			Exp([PauliY, PauliY], Math.PI() * 0.2, [qc, qt]);
			Exp([PauliZ, PauliZ], Math.PI() * 0.3, [qc, qt]);

			// Do the measurements agree?
			set r = (M(qc) == M(qt));

			ResetAll([qc, qt]);
		}

		return r;
	}
}
