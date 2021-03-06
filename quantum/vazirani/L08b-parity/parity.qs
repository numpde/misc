// RA, 2019-12-12

// We have f(x) = u⋅x (mod 2) for an unknown bitstring u
// Problem: recover u
// https://en.wikipedia.org/wiki/Bernstein%E2%80%93Vazirani_algorithm
namespace Quantum.ParityProblem {
	open Microsoft.Quantum.Intrinsic;
	open Microsoft.Quantum.Diagnostics;
	open Microsoft.Quantum.Arrays;

	operation secret() : Int[] {
		let u = [1, 0, 1, 1, 1];
		return u;
	}

	operation U(x: Qubit[], b: Qubit) : Unit {
		// Compute (b ⊕ (u⋅x)) into b
		// That is, flip the qubit b
		// for every k where uₖxₖ = 1
		let u = secret();

		Fact(Length(x) <= Length(u), "u should be at least as long as x.");

		for (k in 0..(Length(x) - 1)) {
			if (u[k] == 1) {
				CNOT(x[k], b);
			}
		}
	}

	operation Test() : Unit {
		using ((x, b) = (Qubit[Length(secret())], Qubit())) {
			// The Bernstein–Vazirani algorithm uses the fact that
			// for the answer bit |b⟩ := |-⟩,
			//     f(x) = 0  ==>  |b ⊕ f(x)⟩ = +1 |-⟩
			//     f(x) = 1  ==>  |b ⊕ f(x)⟩ = -1 |-⟩
			//
			// Thus the uniform mixture input
			//     1/√2ⁿ Σₓ |x⟩ |-⟩
			// is mapped by U to
			//     1/√2ⁿ Σₓ (-)^f(x) |x⟩ |-⟩
			//
			// We know that f(x) = u⋅x for some u
			// Moreover, the Hadamard transform maps u to
			//     1/√2ⁿ Σₓ (-)^(u⋅x) |x⟩
			// Thus, need to apply the (inverse) H transform
			// to the x register to obtain u (Fourier sampling)

			// Prepare uniform state
			for (q in x) {
				H(q);
			}

			// Prepare the answer qubit into the |-⟩ state
			H(b);
			Z(b);

			// Apply the black box
			U(x, b);

			// Fourier sampling
			for (q in x) {
				H(q);
			}

			// Measure the answer qubit
			let mb = Measure([PauliX], [b]);
			Fact(mb == One, "The answer qubit should be in state |-⟩");

			// Measure the other qubits
			let mx = ForEach(M, x);
			Message($"Guessed secret: {mx}");

			ResetAll(x);
			Reset(b);
		}
	}
}
