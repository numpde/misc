// RA, 2019-12-11

namespace Quantum.CHSH {
	open Microsoft.Quantum.Intrinsic;
	open Microsoft.Quantum.Math;

	operation Alice(x : Bool, qa : Qubit) : Bool {
		// Alice's strategy:
		//     If x is false, measure in the 0/1 basis
		//     Else measure in the basis rotated by +π/4
		let π = PI();
		let θ = (x ? -π/4. | 0.);
		mutable a = false;
		within {
			// Factor of 2 on the Bloch sphere
			// https://docs.microsoft.com/en-us/qsharp/api/qsharp/microsoft.quantum.intrinsic.ry?view=qsharp-preview
			Ry(2. * θ, qa);
		} apply {
			// Alice's response
			set a = (M(qa) == One);
		}
		return a;
	}

	operation Bob(y : Bool, qb : Qubit) : Bool {
		// Bob's strategy:
		//     If x is false, measure in the basis rotated by +π/8
		//     Else measure in the basis rotated by -π/8
		let π = PI();
		let θ = (y ? π/8. | -π/8.);
		mutable b = false;
		within {
			// Factor of 2 on the Bloch sphere
			// https://docs.microsoft.com/en-us/qsharp/api/qsharp/microsoft.quantum.intrinsic.ry?view=qsharp-preview
			Ry(2. * θ, qb);
		} apply {
			// Bob's response
			set b = (M(qb) == One);
		}
		return b;
	}

	operation Play2(x : Bool, y : Bool) : (Bool, Bool) {
		// Alice's and Bob's response placeholder
		mutable a = false;
		mutable b = false;

		// Alice and Bob share a Bell pair
		using ((qa, qb) = (Qubit(), Qubit())) {
			// Construct the entangled Bell pair
			H(qa);
			CNOT(qa, qb);

			// Alice's and Bob's response
			set a = Alice(x, qa);
			set b = Bob(y, qb);

			ResetAll([qa, qb]);
		}

		return (a, b);
	}

	operation Test() : Bool {
		let r = false;

		// Alice's input
		let x = [false, true][Random([1., 1.])];
		// Bob's input
		let y = [false, true][Random([1., 1.])];

		//Message($"x = {x}, y = {y}");

		// Alice and Bob's output
		let (a, b) = Play2(x, y);

		// Alice and Bob's outputs should match UNLESS both inputs are True
		let success = ( (x and y) ? (a != b) | (a == b) );

		return success;
	}
}
