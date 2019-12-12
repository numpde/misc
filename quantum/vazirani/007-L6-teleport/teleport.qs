// RA, 2019-12-12

namespace Quantum.Teleport {
	open Microsoft.Quantum.Intrinsic;
	open Microsoft.Quantum.Math;
	open Microsoft.Quantum.Canon;
	open Microsoft.Quantum.Diagnostics;

	operation Test_TeleportationUsingCNOT() : Unit {
		using ((qa, qb) = (Qubit(), Qubit())) {
			// Prepare Alice's qubit
			// New state qa = α |0⟩ + β |1⟩
			Rx(PI() * 0.1, qa);
			Ry(PI() * 0.2, qa);
			Rz(PI() * 0.3, qa);

			Message("Alice's original qubit:");
			DumpRegister((), [qa]);

			// Bob's qubit remains |0⟩

			// New state: |qa qb⟩ = α |00⟩ + β |11⟩
			CNOT(qa, qb);

			// This state also equals
			// |qa qb⟩  =  1/√2 |+⟩ (α |0⟩ + β |1⟩)  +  1/√2 |-⟩ (α |0⟩ - β |1⟩)

			// Alice measures in the +/- basis
			// https://docs.microsoft.com/en-us/qsharp/api/qsharp/microsoft.quantum.intrinsic.measure
			let ma = ((Measure([PauliX], [qa]) == Zero) ? +1 | -1);

			// If Alice observes |+⟩ then Bob has (α |0⟩ + β |1⟩)
			// If Alice observes |-⟩ then Bob has (α |0⟩ - β |1⟩)

			if (ma == -1) {
				// If Alice observed |-⟩ then
				// apply sign flip to Bob's qubit
				Z(qb);
			}

			// Now Bob has
			// qb = α |0⟩ + β |1⟩

			Message("Bob's new qubit:");
			DumpRegister((), [qb]);

			ResetAll([qa, qb]);
		}
	}

	operation Test_Teleportation() : Unit {
		using ((q0, qa, qb) = (Qubit(), Qubit(), Qubit())) {
			// Prepare Alice's qubit
			// New state qa = α |0⟩ + β |1⟩
			Rx(PI() * 0.1, qa);
			Ry(PI() * 0.2, qa);
			Rz(PI() * 0.3, qa);

			Message("Alice's original qubit:");
			DumpRegister((), [qa]);

			// We wish to teleport the state of Alice's qa to Bob's qb
			// using a shared Bell state |q0 qb⟩
			H(q0);
			CNOT(q0, qb);

			// New state
			// |qa q0 qb⟩ = 1/√2 α |0⟩ (|00⟩ + |11⟩)  +  1/√2 β |1⟩ (|10⟩ + |01⟩)
			CNOT(qa, q0);

			// If Alice measures her Bell qubit q0
			// and sees |0⟩ then |qa qb⟩ = 1/√2 (α |00⟩ + β |11⟩)
			// However, if
			// she sees |1⟩ then |qa qb⟩ = 1/√2 (α |01⟩ + β |10⟩)
			// and Bob should apply a bit flip to his Bell qubit qb
			// to get the same state:
			// if (M(q0) == One) { X(qb); }
			// Equivalently, this is a controlled bit flip:
			CX(q0, qb);

			// Now, |qa qb⟩ = 1/√2 (α |00⟩ + β |11⟩)

			// Alice now measures her private qubit qa in the +/- basis
			// https://docs.microsoft.com/en-us/qsharp/api/qsharp/microsoft.quantum.intrinsic.measure
			// let ma = ((Measure([PauliX], [qa]) == Zero) ? +1 | -1);
			// Equivalently, she applies H and measures in the standard basis
			H(qa);

			// If Alice observes |0⟩ then Bob has (α |0⟩ + β |1⟩)
			// If Alice observes |1⟩ then Bob has (α |0⟩ - β |1⟩)

			// If Alice observed |1⟩ then Bob should apply sign flip
			// if (M(qa) == One) { Z(qb); }
			// Equivalently, this is a controlled sign flip:
			CZ(qa, qb);

			// Now Bob has
			// qb = α |0⟩ + β |1⟩

			Message("Bob's new qubit:");
			DumpRegister((), [qb]);

			ResetAll([q0, qa, qb]);
		}
	}

	operation Test() : Unit {
		Message("===============================");
		Message("Test_TeleportationUsingCNOT");
		Message("===============================");
		//
		Test_TeleportationUsingCNOT();

		Message("===============================");
		Message("Test_Teleportation");
		Message("===============================");
		//
		Test_Teleportation();
	}
}