package dragonfly

import (
	"math"
	"math/rand"
	"testing"
)

// weightTolerance is the slack allowed on a scheduled coefficient. The
// schedules are a handful of multiplications, so anything looser would hide a
// real formula error.
const weightTolerance = 1e-12

// weightTestConfig returns a fully populated default config over a known box, so
// every schedule assertion can be written against concrete numbers.
func weightTestConfig() *Config {
	config := NewDefaultConfig()
	config.ProblemSize = 5
	config.LowerBound = -10
	config.UpperBound = 30 // span 40, so span/4 = 10 and span/10 = 4

	return config
}

func weightTestRNG(seed int64) *rand.Rand {
	return rand.New(rand.NewSource(seed))
}

func weightsClose(got, want float64) bool {
	return math.Abs(got-want) <= weightTolerance
}

func TestInertiaDecreasesMonotonically(t *testing.T) {
	config := weightTestConfig()
	config.MaxIterations = 200

	rng := weightTestRNG(1)
	previous := math.Inf(1)

	for iteration := 0; iteration <= config.MaxIterations; iteration++ {
		weights := computeWeights(config, iteration, config.MaxIterations, rng)

		if weights.Inertia >= previous {
			t.Fatalf("iteration %d: inertia %v did not decrease below %v",
				iteration, weights.Inertia, previous)
		}

		previous = weights.Inertia
	}

	first := computeWeights(config, 0, config.MaxIterations, rng)
	if !weightsClose(first.Inertia, config.InertiaWeightStart) {
		t.Errorf("inertia at t=0 = %v, want InertiaWeightStart %v",
			first.Inertia, config.InertiaWeightStart)
	}

	last := computeWeights(config, config.MaxIterations, config.MaxIterations, rng)
	if !weightsClose(last.Inertia, config.InertiaWeightEnd) {
		t.Errorf("inertia at t=T = %v, want InertiaWeightEnd %v",
			last.Inertia, config.InertiaWeightEnd)
	}
}

func TestInertiaHonorsCustomBracket(t *testing.T) {
	config := weightTestConfig()
	config.MaxIterations = 100
	config.InertiaWeightStart = 1.2
	config.InertiaWeightEnd = 0.2

	rng := weightTestRNG(2)

	// Halfway through, a linear schedule is exactly the midpoint.
	weights := computeWeights(config, 50, config.MaxIterations, rng)
	if !weightsClose(weights.Inertia, 0.7) {
		t.Errorf("inertia at t=T/2 = %v, want 0.7", weights.Inertia)
	}
}

func TestConvergenceFactorSchedule(t *testing.T) {
	const maxIterations = 100

	tests := []struct {
		name      string
		iteration int
		want      float64
	}{
		{"start", 0, 0.1},
		{"quarter", 25, 0.05},
		{"halfway", 50, 0.0},
		{"three quarters", 75, 0.0},
		{"end", 100, 0.0},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := convergenceFactor(test.iteration, maxIterations)
			if !weightsClose(got, test.want) {
				t.Errorf("convergenceFactor(%d, %d) = %v, want %v",
					test.iteration, maxIterations, got, test.want)
			}
		})
	}
}

func TestConvergenceFactorStaysZeroPastHalfway(t *testing.T) {
	const maxIterations = 60

	for iteration := maxIterations / 2; iteration <= maxIterations; iteration++ {
		if got := convergenceFactor(iteration, maxIterations); got != 0 {
			t.Fatalf("convergenceFactor(%d, %d) = %v, want exactly 0",
				iteration, maxIterations, got)
		}
	}
}

func TestConvergenceFactorNeverNegative(t *testing.T) {
	const maxIterations = 40

	// Deliberately walk past the end of the run: the max(0, ...) must hold even
	// for an iteration index the schedule was never meant to see.
	for iteration := 0; iteration <= 2*maxIterations; iteration++ {
		if got := convergenceFactor(iteration, maxIterations); got < 0 {
			t.Fatalf("convergenceFactor(%d, %d) = %v, want non-negative",
				iteration, maxIterations, got)
		}
	}
}

func TestEnemyWeightZeroPastCutoff(t *testing.T) {
	config := weightTestConfig()
	config.MaxIterations = 100

	rng := weightTestRNG(3)

	// The default cutoff is three quarters of the run.
	for iteration := 76; iteration <= config.MaxIterations; iteration++ {
		weights := computeWeights(config, iteration, config.MaxIterations, rng)
		if weights.Enemy != 0 {
			t.Fatalf("iteration %d: enemy = %v, want exactly 0", iteration, weights.Enemy)
		}
	}
}

// TestEnemyWeightCutoffBeatsNonZeroFactor pins the cutoff early enough that mc
// is still positive when it bites, so the assertion is about the cutoff rather
// than about mc having already decayed to zero.
func TestEnemyWeightCutoffBeatsNonZeroFactor(t *testing.T) {
	config := weightTestConfig()
	config.MaxIterations = 100
	config.EnemyCutoffFraction = 0.25

	rng := weightTestRNG(4)

	before := computeWeights(config, 25, config.MaxIterations, rng)
	if before.Enemy <= 0 {
		t.Errorf("enemy at the cutoff = %v, want the positive convergence factor", before.Enemy)
	}

	if !weightsClose(before.Enemy, convergenceFactor(25, config.MaxIterations)) {
		t.Errorf("enemy at the cutoff = %v, want mc %v",
			before.Enemy, convergenceFactor(25, config.MaxIterations))
	}

	after := computeWeights(config, 26, config.MaxIterations, rng)
	if after.Enemy != 0 {
		t.Errorf("enemy past the cutoff = %v, want exactly 0", after.Enemy)
	}

	if mc := convergenceFactor(26, config.MaxIterations); mc <= 0 {
		t.Fatalf("test is vacuous: mc past the cutoff is already %v", mc)
	}
}

func TestEnemyWeightNonZeroBeforeCutoff(t *testing.T) {
	config := weightTestConfig()
	config.MaxIterations = 100

	rng := weightTestRNG(5)

	// mc is positive over the first half of the run, and the default cutoff is
	// well past it, so the enemy weight must track mc exactly.
	for iteration := range 50 {
		weights := computeWeights(config, iteration, config.MaxIterations, rng)

		want := convergenceFactor(iteration, config.MaxIterations)
		if weights.Enemy <= 0 || !weightsClose(weights.Enemy, want) {
			t.Fatalf("iteration %d: enemy = %v, want positive mc %v",
				iteration, weights.Enemy, want)
		}
	}
}

// TestEnemyCutoffFractionIsInertAtDefault documents that the cutoff cannot
// change the enemy weight anywhere at or above 0.5: mc has already decayed to
// zero by T/2, so every fraction from there up produces the same schedule. Each
// variant gets its own seeded RNG because computeWeights consumes exactly four
// draws per call, and the streams must line up iteration for iteration.
func TestEnemyCutoffFractionIsInertAtDefault(t *testing.T) {
	baseline := weightTestConfig()
	baseline.MaxIterations = 100

	if baseline.EnemyCutoffFraction != 0.75 {
		t.Fatalf("default EnemyCutoffFraction = %v, want 0.75", baseline.EnemyCutoffFraction)
	}

	const seed = 8

	reference := make([]float64, baseline.MaxIterations+1)
	referenceRNG := weightTestRNG(seed)

	for iteration := range reference {
		reference[iteration] = computeWeights(baseline, iteration, baseline.MaxIterations, referenceRNG).Enemy
	}

	for _, fraction := range []float64{0.5, 0.75, 1.0} {
		config := weightTestConfig()
		config.MaxIterations = baseline.MaxIterations
		config.EnemyCutoffFraction = fraction

		rng := weightTestRNG(seed)

		for iteration := range reference {
			got := computeWeights(config, iteration, config.MaxIterations, rng).Enemy
			if got != reference[iteration] {
				t.Fatalf("cutoff %v, iteration %d: enemy = %v, want %v",
					fraction, iteration, got, reference[iteration])
			}
		}
	}
}

func TestSwarmingWeightsFollowConvergenceFactor(t *testing.T) {
	config := weightTestConfig()
	config.MaxIterations = 100

	rng := weightTestRNG(6)

	// Past the halfway point mc is zero, so s, a and c collapse with it while
	// the food weight -- which takes no mc factor -- stays positive.
	weights := computeWeights(config, 60, config.MaxIterations, rng)

	if weights.Separation != 0 || weights.Alignment != 0 || weights.Cohesion != 0 {
		t.Errorf("s/a/c past halfway = %v/%v/%v, want all exactly 0",
			weights.Separation, weights.Alignment, weights.Cohesion)
	}

	if weights.Food <= 0 || weights.Food > 2 {
		t.Errorf("food = %v, want a value in (0, 2]", weights.Food)
	}
}

func TestScheduledWeightsStayInRange(t *testing.T) {
	config := weightTestConfig()
	config.MaxIterations = 250

	rng := weightTestRNG(7)

	for iteration := range config.MaxIterations + 1 {
		weights := computeWeights(config, iteration, config.MaxIterations, rng)
		mc := convergenceFactor(iteration, config.MaxIterations)

		for name, value := range map[string]float64{
			"separation": weights.Separation,
			"alignment":  weights.Alignment,
			"cohesion":   weights.Cohesion,
		} {
			if value < 0 || value > 2*mc+weightTolerance {
				t.Fatalf("iteration %d: %s = %v, want within [0, 2*mc] = [0, %v]",
					iteration, name, value, 2*mc)
			}
		}

		if weights.Food < 0 || weights.Food > 2 {
			t.Fatalf("iteration %d: food = %v, want within [0, 2]", iteration, weights.Food)
		}
	}
}

func TestRadiusGrowsMonotonically(t *testing.T) {
	config := weightTestConfig()
	config.MaxIterations = 150

	span := config.UpperBound - config.LowerBound

	initial := neighborhoodRadius(config, 0, config.MaxIterations)
	if !weightsClose(initial, span/config.RadiusInitialDivisor) {
		t.Errorf("radius at t=0 = %v, want (ub-lb)/divisor = %v",
			initial, span/config.RadiusInitialDivisor)
	}

	previous := math.Inf(-1)

	for iteration := 0; iteration <= config.MaxIterations; iteration++ {
		radius := neighborhoodRadius(config, iteration, config.MaxIterations)
		if radius <= previous {
			t.Fatalf("iteration %d: radius %v did not grow beyond %v", iteration, radius, previous)
		}

		previous = radius
	}

	final := neighborhoodRadius(config, config.MaxIterations, config.MaxIterations)
	want := span/config.RadiusInitialDivisor + span*config.RadiusGrowth

	if !weightsClose(final, want) {
		t.Errorf("radius at t=T = %v, want %v", final, want)
	}
}

func TestRadiusMatchesScheduleField(t *testing.T) {
	config := weightTestConfig()
	config.MaxIterations = 80

	rng := weightTestRNG(8)

	for _, iteration := range []int{0, 1, 40, 79, 80} {
		weights := computeWeights(config, iteration, config.MaxIterations, rng)

		want := neighborhoodRadius(config, iteration, config.MaxIterations)
		if weights.Radius != want {
			t.Errorf("iteration %d: schedule radius = %v, want %v", iteration, weights.Radius, want)
		}
	}
}

func TestRadiusZeroGrowthIsConstant(t *testing.T) {
	config := weightTestConfig()
	config.MaxIterations = 30
	config.RadiusGrowth = 0

	span := config.UpperBound - config.LowerBound
	want := span / config.RadiusInitialDivisor

	for iteration := 0; iteration <= config.MaxIterations; iteration++ {
		if got := neighborhoodRadius(config, iteration, config.MaxIterations); !weightsClose(got, want) {
			t.Fatalf("iteration %d: radius = %v, want the constant %v", iteration, got, want)
		}
	}
}

func TestMaxStepIsSpanTimesRatio(t *testing.T) {
	config := weightTestConfig()
	config.MaxIterations = 100

	span := config.UpperBound - config.LowerBound
	want := span * config.MaxStepRatio

	rng := weightTestRNG(9)

	for _, iteration := range []int{0, 50, 100} {
		weights := computeWeights(config, iteration, config.MaxIterations, rng)
		if !weightsClose(weights.MaxStep, want) {
			t.Errorf("iteration %d: max step = %v, want (ub-lb)*ratio = %v",
				iteration, weights.MaxStep, want)
		}
	}
}

func TestFixedWeightsArePassedThrough(t *testing.T) {
	config := weightTestConfig()
	config.MaxIterations = 100
	config.SeparationWeight = 0.11
	config.AlignmentWeight = 0.22
	config.CohesionWeight = 0.33
	config.FoodWeight = 0.44
	config.EnemyWeight = 0.55

	rng := weightTestRNG(10)

	// Every iteration, including ones where the schedule would return zero.
	for _, iteration := range []int{0, 1, 50, 99, 100} {
		weights := computeWeights(config, iteration, config.MaxIterations, rng)

		fixed := map[string][2]float64{
			"separation": {weights.Separation, config.SeparationWeight},
			"alignment":  {weights.Alignment, config.AlignmentWeight},
			"cohesion":   {weights.Cohesion, config.CohesionWeight},
			"food":       {weights.Food, config.FoodWeight},
			"enemy":      {weights.Enemy, config.EnemyWeight},
		}
		for name, pair := range fixed {
			if pair[0] != pair[1] {
				t.Errorf("iteration %d: %s = %v, want the configured %v",
					iteration, name, pair[0], pair[1])
			}
		}
	}
}

// TestZeroWeightIsNotTreatedAsAuto guards the reason WeightAuto is -1 rather
// than the zero value: a caller who switches a behavior off must keep it off.
func TestZeroWeightIsNotTreatedAsAuto(t *testing.T) {
	config := weightTestConfig()
	config.MaxIterations = 100
	config.SeparationWeight = 0
	config.FoodWeight = 0

	rng := weightTestRNG(11)

	weights := computeWeights(config, 0, config.MaxIterations, rng)
	if weights.Separation != 0 || weights.Food != 0 {
		t.Errorf("explicit zeros became %v and %v, want both 0", weights.Separation, weights.Food)
	}
}

func TestOnlyOverriddenWeightChanges(t *testing.T) {
	auto := weightTestConfig()
	auto.MaxIterations = 100

	pinned := weightTestConfig()
	pinned.MaxIterations = 100
	pinned.FoodWeight = 1.5

	autoWeights := computeWeights(auto, 10, auto.MaxIterations, weightTestRNG(12))
	pinnedWeights := computeWeights(pinned, 10, pinned.MaxIterations, weightTestRNG(12))

	if pinnedWeights.Food != 1.5 {
		t.Errorf("food = %v, want the pinned 1.5", pinnedWeights.Food)
	}

	// The documented rule: a pinned weight discards its draw rather than
	// skipping it, so the remaining scheduled weights are untouched.
	if autoWeights.Separation != pinnedWeights.Separation ||
		autoWeights.Alignment != pinnedWeights.Alignment ||
		autoWeights.Cohesion != pinnedWeights.Cohesion {
		t.Errorf("pinning food perturbed s/a/c: %v vs %v", autoWeights, pinnedWeights)
	}
}

// TestRNGConsumptionIsIndependentOfOverrides is the other half of the same
// rule: the number of draws must not depend on how many weights were pinned, so
// that the rest of the run stays aligned with a default-config run.
func TestRNGConsumptionIsIndependentOfOverrides(t *testing.T) {
	auto := weightTestConfig()
	auto.MaxIterations = 100

	pinned := weightTestConfig()
	pinned.MaxIterations = 100
	pinned.SeparationWeight = 0.1
	pinned.AlignmentWeight = 0.2
	pinned.CohesionWeight = 0.3
	pinned.FoodWeight = 0.4
	pinned.EnemyWeight = 0.5

	autoRNG := weightTestRNG(13)
	pinnedRNG := weightTestRNG(13)

	computeWeights(auto, 7, auto.MaxIterations, autoRNG)
	computeWeights(pinned, 7, pinned.MaxIterations, pinnedRNG)

	for draw := range 4 {
		got, want := pinnedRNG.Float64(), autoRNG.Float64()
		if got != want {
			t.Fatalf("draw %d after the call differs: %v vs %v -- the streams desynchronized",
				draw, got, want)
		}
	}
}

func TestComputeWeightsIsDeterministicForSameSeed(t *testing.T) {
	config := weightTestConfig()
	config.MaxIterations = 100

	for _, iteration := range []int{0, 1, 33, 99, 100} {
		first := computeWeights(config, iteration, config.MaxIterations, weightTestRNG(14))
		second := computeWeights(config, iteration, config.MaxIterations, weightTestRNG(14))

		if first != second {
			t.Errorf("iteration %d: %+v != %+v for the same seed", iteration, first, second)
		}
	}
}

func TestComputeWeightsDiffersAcrossSeeds(t *testing.T) {
	config := weightTestConfig()
	config.MaxIterations = 100

	first := computeWeights(config, 0, config.MaxIterations, weightTestRNG(15))
	second := computeWeights(config, 0, config.MaxIterations, weightTestRNG(16))

	if first == second {
		t.Errorf("two seeds produced the identical schedule %+v", first)
	}
}

// TestComputeWeightsNilRNG ensures missing explicit RNG threading fails at the
// call site instead of silently using package-global state.
func TestComputeWeightsNilRNG(t *testing.T) {
	config := weightTestConfig()
	config.MaxIterations = 100

	defer func() {
		if recover() == nil {
			t.Fatal("computeWeights(nil rng) did not panic")
		}
	}()

	computeWeights(config, 0, config.MaxIterations, nil)
}

func TestSingleIterationRun(t *testing.T) {
	config := weightTestConfig()
	config.MaxIterations = 1

	span := config.UpperBound - config.LowerBound
	rng := weightTestRNG(17)

	start := computeWeights(config, 0, 1, rng)
	if !weightsClose(start.Inertia, config.InertiaWeightStart) {
		t.Errorf("inertia at t=0 of a one-iteration run = %v, want %v",
			start.Inertia, config.InertiaWeightStart)
	}

	if !weightsClose(start.Radius, span/config.RadiusInitialDivisor) {
		t.Errorf("radius at t=0 of a one-iteration run = %v, want %v",
			start.Radius, span/config.RadiusInitialDivisor)
	}

	if mc := convergenceFactor(0, 1); !weightsClose(mc, 0.1) {
		t.Errorf("mc at t=0 of a one-iteration run = %v, want 0.1", mc)
	}

	end := computeWeights(config, 1, 1, rng)
	if !weightsClose(end.Inertia, config.InertiaWeightEnd) {
		t.Errorf("inertia at t=T of a one-iteration run = %v, want %v",
			end.Inertia, config.InertiaWeightEnd)
	}

	if end.Enemy != 0 {
		t.Errorf("enemy at t=T of a one-iteration run = %v, want 0 (past the cutoff)", end.Enemy)
	}
}

// TestDegenerateMaxIterations covers indices and totals the optimizer should
// never produce: validateConfig rejects a non-positive MaxIterations, but the
// schedules must still return finite numbers rather than NaN if one arrives.
func TestDegenerateMaxIterations(t *testing.T) {
	config := weightTestConfig()

	tests := []struct {
		name          string
		iteration     int
		maxIterations int
	}{
		{"zero total", 0, 0},
		{"negative total", 3, -5},
		{"iteration past the end", 500, 100},
		{"negative iteration", -3, 100},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			weights := computeWeights(config, test.iteration, test.maxIterations, weightTestRNG(18))

			values := map[string]float64{
				"inertia":    weights.Inertia,
				"separation": weights.Separation,
				"alignment":  weights.Alignment,
				"cohesion":   weights.Cohesion,
				"food":       weights.Food,
				"enemy":      weights.Enemy,
				"radius":     weights.Radius,
				"max step":   weights.MaxStep,
			}
			for name, value := range values {
				if math.IsNaN(value) || math.IsInf(value, 0) {
					t.Errorf("%s = %v, want a finite number", name, value)
				}
			}

			if mc := convergenceFactor(test.iteration, test.maxIterations); math.IsNaN(mc) || mc < 0 {
				t.Errorf("mc = %v, want a non-negative number", mc)
			}
		})
	}
}

func TestScheduleProgressIsClamped(t *testing.T) {
	tests := []struct {
		name          string
		iteration     int
		maxIterations int
		want          float64
	}{
		{"start", 0, 100, 0},
		{"midpoint", 50, 100, 0.5},
		{"end", 100, 100, 1},
		{"past the end", 300, 100, 1},
		{"negative iteration", -1, 100, 0},
		{"zero total", 0, 0, 0},
		{"single iteration end", 1, 1, 1},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := scheduleProgress(test.iteration, test.maxIterations)
			if !weightsClose(got, test.want) {
				t.Errorf("scheduleProgress(%d, %d) = %v, want %v",
					test.iteration, test.maxIterations, got, test.want)
			}
		})
	}
}

func TestNeighborhoodRadiusDegenerateDivisor(t *testing.T) {
	config := weightTestConfig()
	config.MaxIterations = 100
	config.RadiusInitialDivisor = 0

	radius := neighborhoodRadius(config, 0, config.MaxIterations)
	if math.IsInf(radius, 0) || math.IsNaN(radius) || radius <= 0 {
		t.Errorf("radius with a zero divisor = %v, want a positive finite fallback", radius)
	}
}
