package dragonfly

import (
	"math"
	"math/rand"
	"sort"
	"testing"
)

// sigmaTolerance is loose enough to accept the literature's rounded 0.6966 but
// tight enough to reject any algebraically different variant of the formula.
const sigmaTolerance = 1e-4

func TestLevySigmaKnownValues(t *testing.T) {
	tests := []struct {
		name string
		beta float64
		want float64
	}{
		// The value quoted throughout the Lévy-flight metaheuristic
		// literature (Yang & Deb, Mirjalili) for the DA default.
		{name: "beta=1.5 (DA default)", beta: DefaultLevyBeta, want: 0.6966},
		// σ(1) = (Γ(2)·sin(π/2) / (Γ(1)·1·2^0))^1 = 1.
		{name: "beta=1.0", beta: 1.0, want: 1.0},
		{name: "beta=0.5", beta: 0.5, want: 1.4793376},
		{name: "beta=1.2", beta: 1.2, want: 0.8788288},
		{name: "beta=1.8", beta: 1.8, want: 0.4586381},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := levySigma(tt.beta)
			if math.Abs(got-tt.want) > sigmaTolerance {
				t.Errorf("levySigma(%v) = %.7f, want %.7f (tol %g)",
					tt.beta, got, tt.want, sigmaTolerance)
			}
		})
	}
}

// TestLevySigmaClosedForm re-derives σ from the formula independently of the
// implementation, guarding against a transcription slip inside levySigma.
func TestLevySigmaClosedForm(t *testing.T) {
	for _, beta := range []float64{0.3, 0.7, 1.0, 1.3, 1.5, 1.7, 1.9} {
		num := math.Gamma(1+beta) * math.Sin(math.Pi*beta/2)
		den := math.Gamma((1+beta)/2) * beta * math.Pow(2, (beta-1)/2)
		want := math.Pow(num/den, 1/beta)

		if got := levySigma(beta); math.Abs(got-want) > 1e-12 {
			t.Errorf("levySigma(%v) = %v, want %v", beta, got, want)
		}
	}
}

func TestLevySigmaIsPositiveAndFinite(t *testing.T) {
	for beta := 0.1; beta < 2.0; beta += 0.1 {
		got := levySigma(beta)
		if math.IsNaN(got) || math.IsInf(got, 0) || got <= 0 {
			t.Errorf("levySigma(%.1f) = %v, want a finite positive value", beta, got)
		}
	}
}

func TestLevyFlightDeterministicForSameSeed(t *testing.T) {
	const n = 128

	a := rand.New(rand.NewSource(42))
	b := rand.New(rand.NewSource(42))

	for i := range n {
		x := levyFlight(DefaultLevyBeta, DefaultLevyScale, a)
		y := levyFlight(DefaultLevyBeta, DefaultLevyScale, b)

		if x != y {
			t.Fatalf("draw %d: same seed produced %v and %v", i, x, y)
		}
	}
}

func TestLevyFlightDiffersAcrossSeeds(t *testing.T) {
	a := levyVector(64, DefaultLevyBeta, DefaultLevyScale, rand.New(rand.NewSource(1)))
	b := levyVector(64, DefaultLevyBeta, DefaultLevyScale, rand.New(rand.NewSource(2)))

	identical := true

	for i := range a {
		if a[i] != b[i] {
			identical = false

			break
		}
	}

	if identical {
		t.Error("different seeds produced identical sequences")
	}
}

func TestLevyFlightAllFinite(t *testing.T) {
	rng := rand.New(rand.NewSource(7))

	for i := range 100000 {
		step := levyFlight(DefaultLevyBeta, DefaultLevyScale, rng)
		if math.IsNaN(step) || math.IsInf(step, 0) {
			t.Fatalf("draw %d is not finite: %v", i, step)
		}
	}
}

// TestLevyFlightDegenerateBeta exercises the non-finite guard: at β = 2 the
// sin term vanishes so σ = 0, and β ≤ 0 makes the Gamma terms degenerate.
func TestLevyFlightDegenerateBeta(t *testing.T) {
	for _, beta := range []float64{2.0, 0.0, -1.0} {
		rng := rand.New(rand.NewSource(11))

		for range 1000 {
			step := levyFlight(beta, DefaultLevyScale, rng)
			if math.IsNaN(step) || math.IsInf(step, 0) {
				t.Fatalf("beta=%v produced a non-finite step: %v", beta, step)
			}
		}
	}
}

func TestLevyFlightScalesLinearly(t *testing.T) {
	a := rand.New(rand.NewSource(99))
	b := rand.New(rand.NewSource(99))

	for i := range 64 {
		one := levyFlight(DefaultLevyBeta, 1.0, a)
		hundredth := levyFlight(DefaultLevyBeta, 0.01, b)

		if math.Abs(one*0.01-hundredth) > 1e-12*math.Max(1, math.Abs(one)) {
			t.Fatalf("draw %d: scale is not linear: %v vs %v", i, one*0.01, hundredth)
		}
	}
}

func TestLevyVectorLengthAndFiniteness(t *testing.T) {
	rng := rand.New(rand.NewSource(3))

	for _, size := range []int{1, 2, 5, 30, 100} {
		vec := levyVector(size, DefaultLevyBeta, DefaultLevyScale, rng)
		if len(vec) != size {
			t.Fatalf("levyVector(%d) returned length %d", size, len(vec))
		}

		for i, v := range vec {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Errorf("levyVector(%d)[%d] = %v, want finite", size, i, v)
			}
		}
	}
}

func TestLevyVectorNonPositiveSize(t *testing.T) {
	rng := rand.New(rand.NewSource(3))

	for _, size := range []int{0, -1} {
		if vec := levyVector(size, DefaultLevyBeta, DefaultLevyScale, rng); len(vec) != 0 {
			t.Errorf("levyVector(%d) returned %d elements, want 0", size, len(vec))
		}
	}
}

// TestLevyFlightHeavyTailed asserts the defining shape of the distribution:
// the typical step is tiny while the extremes are orders of magnitude larger.
// A Gaussian of the same median would fail this by a wide margin. Fixed seed
// plus generous thresholds keep it non-flaky.
func TestLevyFlightHeavyTailed(t *testing.T) {
	const n = 200000

	rng := rand.New(rand.NewSource(2024))

	mags := make([]float64, n)
	for i := range mags {
		mags[i] = math.Abs(levyFlight(DefaultLevyBeta, DefaultLevyScale, rng))
	}

	sort.Float64s(mags)

	median := mags[n/2]
	maximum := mags[n-1]
	p999 := mags[(n*999)/1000]

	// With scale = 0.01 and σ ≈ 0.6966 the median magnitude sits near 3e-3.
	if median <= 0 || median > 0.05 {
		t.Errorf("median |step| = %g, want a small positive value (<= 0.05)", median)
	}

	// The tail must dwarf the bulk. Empirically the ratio is >1e3; require
	// only 100x so the test cannot flake on a different Go rand stream.
	if ratio := maximum / median; ratio < 100 {
		t.Errorf("max/median = %g, want >= 100 (distribution is not heavy-tailed)", ratio)
	}

	if p999 < 10*median {
		t.Errorf("p99.9/median = %g, want >= 10", p999/median)
	}

	// Sanity: the bulk really is the bulk — most draws stay small.
	if mags[(n*90)/100] > 20*median {
		t.Errorf("p90 = %g is not concentrated around the median %g", mags[(n*90)/100], median)
	}
}

// TestLevyFlightSymmetric checks the step is sign-symmetric, as r₁ ~ N(0,1)
// implies. Roughly half of a large sample must be negative.
func TestLevyFlightSymmetric(t *testing.T) {
	const n = 100000

	rng := rand.New(rand.NewSource(5150))

	negative := 0

	for range n {
		if levyFlight(DefaultLevyBeta, DefaultLevyScale, rng) < 0 {
			negative++
		}
	}

	frac := float64(negative) / float64(n)
	if frac < 0.45 || frac > 0.55 {
		t.Errorf("negative fraction = %.4f, want ~0.5", frac)
	}
}
