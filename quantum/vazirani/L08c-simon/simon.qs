// RA, 2019-12-14

namespace Quantum.Simon {
	open Microsoft.Quantum.Intrinsic;
	open Microsoft.Quantum.Diagnostics;
	open Microsoft.Quantum.Canon;
	open Microsoft.Quantum.Arrays;

	// https://docs.microsoft.com/en-us/qsharp/api/qsharp/microsoft.quantum.arrays.fold
	function Plus(a : Int, b : Int) : Int {
		return a + b;
	}

	// Secret bit-string s
	operation secret() : Int[] {
		let s = [0, 1, 0, 0];
		return s;
	}

	// We have a function
	// f : {0, 1}ⁿ --> {0, 1}ⁿ
	// that is 2-to-1 with
	// f(x) = f(x ⊕ s) for some "secret" s
	//
	// It works on qubits as
	// U : |x⟩ |b⟩ --> |x⟩ |b ⊕ f(x)⟩
	// where '⊕' is applied bitwise

	operation U(x: Qubit[], b: Qubit[]) : Unit {
		let s = secret();

		let n = Length(s);
		Fact(n == Length(x), $"Register x has {Length(x)} qubits, expected {n}");
		Fact(n == Length(b), $"Register b has {Length(b)} qubits, expected {n}");

		// The output will only be based on the qubits xₖ where sₖ == 0
		// This way, f(x) = f(x ⊕ s)

		// To make this f 2-to-1 we require that |s| = 1
		Fact(Fold(Plus, 0, s) == 1, "The secret bit-string should have exactly one bit on");

		for (k in 0..(n-1)) {
			if (s[k] == 0) {
				CNOT(x[k], b[k]);
			}
		}
	}

	operation SampleY() : Result[] {
		// Return value (set below)
		mutable mx = new Result[0];

		// Input/output dimension
		let n = Length(secret());

		using ((x, b) = (Qubit[n], Qubit[n])) {
			// Prepare uniform superposition
			ApplyToEach(H, x);

			// Apply black box
			U(x, b);

			// Measure the answer qubits
			let mb = ForEach(M, b);

			// The x register now contains
			// 1/√2 |r⟩ + 1/√2 |r ⊕ s⟩
			// for some bit-string r

			// Fourier sampling the x register
			ApplyToEach(H, x);
			set mx = ForEach(M, x);

			ResetAll(x);
			ResetAll(b);
		}

		return mx;
	}
}
