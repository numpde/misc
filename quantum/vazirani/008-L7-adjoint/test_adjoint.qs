// RA, 2019-12-12

namespace Quantum.TestAdjoint {
	open Microsoft.Quantum.Intrinsic;
	open Microsoft.Quantum.Diagnostics;

	operation U(q0: Qubit, q1: Qubit) : Unit is Adj {
		H(q0);
		Z(q1);
		CNOT(q0, q1);
	}

	operation Ut(q0: Qubit, q1: Qubit) : Unit {
		// "Adjoint" is redundant below because all
		// these operations are self-adjoint already
		Adjoint CNOT(q0, q1);
		Adjoint Z(q1);
		Adjoint H(q0);
	}

	operation Test() : Unit {
		Message("==== Enter TestAdjoint ====");

		using ((q0, q1) = (Qubit(), Qubit())) {
			U(q0, q1);
			Ut(q0, q1);
			AssertAllZero([q0, q1]);

			U(q0, q1);
			Adjoint U(q0, q1);
			AssertAllZero([q0, q1]);
		}

		Message("[TestAdjoint success]");
	}
}
