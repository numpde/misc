// RA, 2019-12-10

namespace Quantum.Bell {
	open Microsoft.Quantum.Intrinsic;

	// Do the measurements of the two qubits
	// in a Bell state agree?
	operation Corr() : Bool {
		mutable r = false;

		using ((qc, qt) = (Qubit(), Qubit())) {
			// qc is the control qubit
			// qt is the target qubit

			// New state: qc = 1/√2 (|0⟩ + |1⟩)
			H(qc);

			// Entangle the qubits
			// New state: |qc qt⟩ = 1/√2 (|00⟩ + |11⟩)
			CNOT(qc, qt);

			set r = (M(qc) == M(qt));

			ResetAll([qc, qt]);
		}

		return r;
	}

	// Measure the Bell state in the "sign" basis |+⟩/|-⟩, sequentially
	// EPR paradox
	operation Corr_SignBasis() : Bool {
		mutable r = false;

		using ((qc, qt) = (Qubit(), Qubit())) {
			// New state: |qc qt⟩ = 1/√2 (|00⟩ + |11⟩)
			H(qc);
			CNOT(qc, qt);

			// The entangled state can equivalently be written as
			// |qt qc⟩ = 1/√2 (|++⟩ + |--⟩)

			// Measure in the "sign" basis
			// https://docs.microsoft.com/en-us/qsharp/api/qsharp/microsoft.quantum.intrinsic.measure
			let mc = Measure([PauliX], [qc]);
			let mt = Measure([PauliX], [qt]);

			// DEBUG:
			// Message($"mc = {mc}, mt = {mt}");

			set r = (mc == mt);

			ResetAll([qc, qt]);
		}

		return r;
	}

	// Measure the Bell state in the "sign" basis |+⟩/|-⟩, simultaneously
	operation Corr_SignBasis_Simul() : Result {
		mutable r = Zero;

		using ((qc, qt) = (Qubit(), Qubit())) {
			// New state: |qc qt⟩ = 1/√2 (|00⟩ + |11⟩)
			H(qc);
			CNOT(qc, qt);

			// The entangled state can equivalently be written as
			// |qt qc⟩ = 1/√2 (|++⟩ + |--⟩)

			// Measure in the "sign" basis
			// https://docs.microsoft.com/en-us/qsharp/api/qsharp/microsoft.quantum.intrinsic.measure
			set r = Measure([PauliX, PauliX], [qc, qt]);

			ResetAll([qc, qt]);
		}

		return r;
	}
}
