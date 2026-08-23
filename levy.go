// Lévy flight helpers.
//
// Implements Lévy-stable steps via Mantegna's algorithm, used by the Dragonfly
// Algorithm for the random walk a dragonfly performs when it has no neighbours:
//
//	X_i += Levy(d) ⊙ X_i
//
// Reference:
// Mantegna, R.N. (1994). Fast, Accurate Algorithm for Numerical Simulation of
// Lévy Stable Stochastic Processes. Physical Review E, 49(5), 4677-4683.
// DOI: 10.1103/PhysRevE.49.4677
//
// Mirsalehi/Mirjalili's DA reference implementation uses β = 1.5 and a step
// scale of 0.01, which yields σ ≈ 0.6966.

package dragonfly

import (
	"math"
	"math/rand"
)

const (
	// DefaultLevyBeta is the stability index used by the DA reference
	// implementation. β must lie in (0, 2]; at β = 2 the numerator's
	// sin(πβ/2) vanishes and σ collapses to zero, so β = 1.5 is the
	// practical heavy-tailed default.
	DefaultLevyBeta = 1.5

	// DefaultLevyScale is the multiplicative step scale from the DA paper.
	DefaultLevyScale = 0.01

	// levyDenomFloor bounds |r₂| away from zero. See levyFlight.
	levyDenomFloor = 1e-10
)

// levySigma returns Mantegna's scale parameter for the given stability index:
//
//	σ = ( Γ(1+β)·sin(πβ/2) / ( Γ((1+β)/2)·β·2^((β-1)/2) ) )^(1/β)
//
// For β = 1.5 this evaluates to ≈ 0.6965745, the value quoted in the DA
// literature.
func levySigma(beta float64) float64 {
	numerator := math.Gamma(1+beta) * math.Sin(math.Pi*beta/2)
	denominator := math.Gamma((1+beta)/2) * beta * math.Pow(2, (beta-1)/2)

	return math.Pow(numerator/denominator, 1/beta)
}

// levyFlight draws a single Lévy-distributed step:
//
//	Levy = scale · r₁·σ / |r₂|^(1/β)      r₁, r₂ ~ N(0,1)
//
// rng must not be nil (ensured by the caller).
//
// Near-zero |r₂| handling: |r₂| is clamped to levyDenomFloor rather than
// redrawn. Clamping keeps the number of RNG draws per call fixed at exactly
// two, which is what makes a seeded run bit-for-bit reproducible; a redraw
// loop would consume a seed-dependent number of values and desynchronise every
// subsequent draw in the population. The clamp is also unbiased in sign — the
// sign of r₂ is irrelevant because only |r₂| enters the formula — and the
// truncation it introduces is confined to a region of probability ≈ 8e-11,
// far below any effect on the search. The clamp still admits a very large but
// finite step (~1e15 · scale · σ), preserving the heavy tail.
//
// A final non-finite guard covers degenerate β (e.g. β ≤ 0 or β = 2, where σ
// is zero or the Gamma terms overflow); in that case a plain Gaussian step of
// the requested scale is returned so callers never see NaN or ±Inf.
func levyFlight(beta, scale float64, rng *rand.Rand) float64 {
	sigma := levySigma(beta)

	r1 := randn(rng)
	r2 := randn(rng)

	denom := math.Abs(r2)
	if !(denom > levyDenomFloor) { // also catches NaN
		denom = levyDenomFloor
	}

	step := scale * r1 * sigma / math.Pow(denom, 1/beta)

	if math.IsNaN(step) || math.IsInf(step, 0) {
		return scale * r1
	}

	return step
}

// levyVector draws size independent Lévy steps.
// rng must not be nil (ensured by the caller).
func levyVector(size int, beta, scale float64, rng *rand.Rand) []float64 {
	if size <= 0 {
		return nil
	}

	vec := make([]float64, size)
	for i := range vec {
		vec[i] = levyFlight(beta, scale, rng)
	}

	return vec
}
