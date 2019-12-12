// RA, 2019-12-12

namespace Quantum.TestReversible {
	open Microsoft.Quantum.Intrinsic;
	open Microsoft.Quantum.Diagnostics;

	// We construct a reversible AND gate
	// using a controlled SWAP
	operation Test() : Unit {
		Message("==== Enter TestReversible ====");

		using (q = Qubit[4]) {
			// q[0] | answer qubit
			// q[1] | input qubits
			// q[2] | input qubits
			// q[3] | auxiliary qubit

			Message("Answer qubit before:");
			DumpRegister((), [q[0]]);

			X(q[1]); // Now, q[1] = |1⟩
			X(q[2]); // Now, q[2] = |1⟩

			// Next, compute (q[1] AND q[2]) into q[0]

			within {
				// For the "Controlled" keyword, see the subsection in
				// https://docs.microsoft.com/en-us/quantum/language/type-model?view=qsharp-preview#operation-and-function-types
				Controlled SWAP([q[1]], (q[2], q[3]));
			} apply {
				// "Copy" result to the answer qubit
				CNOT(q[3], q[0]);
			}

			Message("Answer qubit after:");
			DumpRegister((), [q[0]]);

			Fact(M(q[3]) == Zero, "Auxiliary qubit should be Zero");
			Fact(M(q[0]) == One, "Answer qubit should be in state One now");

			ResetAll(q);
		}

		Message("[TestReversible success]");
	}
}
