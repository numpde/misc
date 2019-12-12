// RA, 2019-12-10

namespace Quantum.Uncer {
	open Microsoft.Quantum.Intrinsic;
	open Microsoft.Quantum.Math;
	open Microsoft.Quantum.Convert;

	operation Prepare(q : Qubit, theta: Double) : Unit {

		// Message(DoubleAsString(theta));

		// New state is:
		// q = cos(θ/2) |0⟩ + sin(θ/2) |1⟩
		R(PauliY, theta, q);

		// Coefficients in the Z basis:
		let aZ = Cos(theta / 2.0);
		let bZ = Sin(theta / 2.0);

		// Note:
		// The uncertainty Var(Measure([PauliZ], [Q]))
		// is the product of the two probabilities,
		// which equals 1/4 sin²(θ)

		// In the +/- basis, we have
		//     q = αX |+⟩ + βX |-⟩
		// with:
		let aX = (Cos(theta / 2.0) + Sin(theta / 2.0)) / Sqrt(2.0);
		let bX = (Cos(theta / 2.0) - Sin(theta / 2.0)) / Sqrt(2.0);

		// Thus, expect to measure:
		// q = |+⟩ with probability |αX|² = (1 + sin(θ)) / 2
		// q = |-⟩ with probability |βX|² = (1 - sin(θ)) / 2

		// Note:
		// The uncertainty Var(Measure([PauliX], [Q]))
		// is the product of the two probabilities,
		// which equals 1/4 (1 - sin²(θ))

		// Note:
		// The uncertainties add up to 1/4
	}

	operation MeasureX(theta : Double) : Result {
		mutable r = Zero;

		using (q = Qubit()) {
			Prepare(q, theta);
			set r = Measure([PauliX], [q]);
			Reset(q);
		}

		return r;
	}

	operation MeasureZ(theta : Double) : Result {
		mutable r = Zero;

		using (q = Qubit()) {
			Prepare(q, theta);
			set r = Measure([PauliZ], [q]);
			Reset(q);
		}

		return r;
	}
}
