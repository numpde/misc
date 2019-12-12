// RA, 2019-12-12

namespace Quantum.Hada {
	open Microsoft.Quantum.Intrinsic;
	open Microsoft.Quantum.Diagnostics;

	operation Test(n : Int) : Unit {
		// n is the number of qubits
		using (qq = Qubit[n]) {
			// Flip some qubits
			X(qq[0]);
			X(qq[1]);

			// New state:
			// qq = |a⟩ := |10..01⟩

			// Apply the Hadamard gate to each qubit
			for (q in qq) {
				H(q);
			}

			// Expect the following state:
			// qq = 1/√2ⁿ Σₓ (-)^(a⋅x) |x⟩
			DumpRegister((), qq);

			// More generally, the state
			// Σₐ αₐ |a⟩
			// is transformed into
			// 1/√2ⁿ Σₓ (Σₐ αₐ (-)^(a⋅x)) |x⟩
			// Thus, measure |x⟩ with probability
			// 1/√2ⁿ |Σₐ αₐ (-)^(a⋅x)|²

			ResetAll(qq);
		}
	}
}
