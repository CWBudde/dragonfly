package dragonfly

import (
	"context"
	"errors"
	"math"
	"math/rand"
	"testing"
)

// newTestConfig builds a runnable configuration on the given objective and box,
// seeded so that every assertion in this file is reproducible.
func newTestConfig(objective ObjectiveFunction, size int, lower, upper float64, seed int64) *Config {
	config := NewDefaultConfig()
	config.ObjectiveFunc = objective
	config.ProblemSize = size
	config.LowerBound = lower
	config.UpperBound = upper
	config.MaxIterations = 120
	config.NPop = 20
	config.Rand = rand.New(rand.NewSource(seed))

	return config
}

// TestOptimizeConvergence is the Phase 2 gate: the assembled loop has to make
// real progress on the four standard benchmarks at ProblemSize = 10.
//
// The tolerances below are EMPIRICAL, not theoretical. They were measured by
// running seeds 1..20 of each benchmark with this exact configuration and
// rounding the worst observed cost up; the test then exercises three of those
// seeds. They are deliberately loose. The Dragonfly Algorithm's convergence
// factor mc reaches zero at the halfway point, after which only the food term
// and inertia remain and the swarm collapses onto the incumbent, so DA is
// well known to stall on higher-dimensional problems -- the numbers here are
// what a faithful implementation of the paper actually achieves, not what a
// well-tuned optimizer could. Treat a regression below them as a bug and a
// large improvement as a change in the algorithm, not as luck.
func TestOptimizeConvergence(t *testing.T) {
	if testing.Short() {
		t.Skip("convergence runs are slow; skipped under -short")
	}

	tests := []struct {
		name      string
		objective ObjectiveFunction
		lower     float64
		upper     float64
		tolerance float64
		// maxRatio caps the final cost as a fraction of the initial swarm's
		// best. It is the scale-free half of the gate: a benchmark whose values
		// span four orders of magnitude over its box (Sphere, Rosenbrock) has
		// far more headroom than one that spans one (Rastrigin, Ackley), so the
		// ratio is per benchmark and, like the tolerance, empirical.
		maxRatio float64
	}{
		{"Sphere", Sphere, -100, 100, 150, 0.1},
		{"Rastrigin", Rastrigin, -5.12, 5.12, 60, 0.5},
		{"Ackley", Ackley, -32.768, 32.768, 8, 0.5},
		{"Rosenbrock", Rosenbrock, -5, 10, 400, 0.1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for seed := int64(1); seed <= 3; seed++ {
				config := newTestConfig(tt.objective, 10, tt.lower, tt.upper, seed)
				config.MaxIterations = 1500
				config.NPop = 60

				result, err := Optimize(config)
				if err != nil {
					t.Fatalf("Optimize() error = %v", err)
				}

				if result.GlobalBest.Cost > tt.tolerance {
					t.Errorf("seed %d: cost = %g, want <= %g", seed, result.GlobalBest.Cost, tt.tolerance)
				}

				// Independently of the absolute tolerance, the run must be a
				// large improvement on where the initial swarm started.
				start := result.ConvergenceCurve[0]
				if result.GlobalBest.Cost > start*tt.maxRatio {
					t.Errorf("seed %d: cost = %g, only a weak improvement on the initial best %g",
						seed, result.GlobalBest.Cost, start)
				}
			}
		})
	}
}

// TestOptimizeDeterminismSameSeed asserts that a seeded run is reproducible in
// every observable: the curve, the incumbent, its position and the evaluation
// count.
func TestOptimizeDeterminismSameSeed(t *testing.T) {
	first, err := Optimize(newTestConfig(Rastrigin, 6, -5.12, 5.12, 20240823))
	if err != nil {
		t.Fatalf("first Optimize() error = %v", err)
	}

	second, err := Optimize(newTestConfig(Rastrigin, 6, -5.12, 5.12, 20240823))
	if err != nil {
		t.Fatalf("second Optimize() error = %v", err)
	}

	if first.GlobalBest.Cost != second.GlobalBest.Cost {
		t.Errorf("cost = %g and %g, want identical", first.GlobalBest.Cost, second.GlobalBest.Cost)
	}

	if first.FuncEvalCount != second.FuncEvalCount {
		t.Errorf("FuncEvalCount = %d and %d, want identical", first.FuncEvalCount, second.FuncEvalCount)
	}

	if len(first.ConvergenceCurve) != len(second.ConvergenceCurve) {
		t.Fatalf("curve lengths = %d and %d, want identical",
			len(first.ConvergenceCurve), len(second.ConvergenceCurve))
	}

	for i := range first.ConvergenceCurve {
		if first.ConvergenceCurve[i] != second.ConvergenceCurve[i] {
			t.Fatalf("ConvergenceCurve[%d] = %g and %g, want identical",
				i, first.ConvergenceCurve[i], second.ConvergenceCurve[i])
		}
	}

	if len(first.GlobalBest.Position) != len(second.GlobalBest.Position) {
		t.Fatalf("position lengths = %d and %d, want identical",
			len(first.GlobalBest.Position), len(second.GlobalBest.Position))
	}

	for i := range first.GlobalBest.Position {
		if first.GlobalBest.Position[i] != second.GlobalBest.Position[i] {
			t.Fatalf("GlobalBest.Position[%d] = %g and %g, want identical",
				i, first.GlobalBest.Position[i], second.GlobalBest.Position[i])
		}
	}
}

// TestOptimizeDeterminismDifferentSeeds asserts that the seed actually drives
// the run: a different stream has to produce a different trajectory.
func TestOptimizeDeterminismDifferentSeeds(t *testing.T) {
	first, err := Optimize(newTestConfig(Rastrigin, 6, -5.12, 5.12, 1))
	if err != nil {
		t.Fatalf("first Optimize() error = %v", err)
	}

	second, err := Optimize(newTestConfig(Rastrigin, 6, -5.12, 5.12, 2))
	if err != nil {
		t.Fatalf("second Optimize() error = %v", err)
	}

	identical := first.GlobalBest.Cost == second.GlobalBest.Cost
	for i := range first.ConvergenceCurve {
		if first.ConvergenceCurve[i] != second.ConvergenceCurve[i] {
			identical = false

			break
		}
	}

	if identical {
		t.Error("different seeds produced identical results")
	}
}

// TestOptimizeSeedWriteback covers the RNG contract: a nil Config.Rand is
// filled in with a seeded generator and the seed is reported.
func TestOptimizeSeedWriteback(t *testing.T) {
	config := newTestConfig(Sphere, 4, -10, 10, 1)
	config.Rand = nil

	result, err := Optimize(config)
	if err != nil {
		t.Fatalf("Optimize() error = %v", err)
	}

	if config.Rand == nil {
		t.Error("Optimize() did not write its generator back into Config.Rand")
	}

	if result.Seed == 0 {
		t.Error("Result.Seed = 0, want the generated seed")
	}

	if !result.SeedKnown {
		t.Error("Result.SeedKnown = false for a library-generated seed")
	}
}

func TestOptimizeExplicitSeedIsReportedAndReproducible(t *testing.T) {
	seed := int64(42)
	config := newTestConfig(Sphere, 4, -10, 10, 1)
	config.Rand = nil
	config.Seed = &seed

	first, err := Optimize(config)
	if err != nil {
		t.Fatalf("first Optimize() error = %v", err)
	}

	second, err := Optimize(config)
	if err != nil {
		t.Fatalf("second Optimize() error = %v", err)
	}

	if !first.SeedKnown || first.Seed != seed {
		t.Fatalf("seed metadata = (%d, %v), want (%d, true)", first.Seed, first.SeedKnown, seed)
	}

	if first.GlobalBest.Cost != second.GlobalBest.Cost {
		t.Fatalf("explicit seed did not reproduce: %v != %v", first.GlobalBest.Cost, second.GlobalBest.Cost)
	}
}

func TestOptimizeCallerRandHasUnknownSeed(t *testing.T) {
	result, err := Optimize(newTestConfig(Sphere, 4, -10, 10, 1))
	if err != nil {
		t.Fatalf("Optimize() error = %v", err)
	}

	if result.SeedKnown || result.Seed != 0 {
		t.Fatalf("seed metadata = (%d, %v), want (0, false)", result.Seed, result.SeedKnown)
	}
}

func TestOptimizeRejectsAllNonFiniteObjectives(t *testing.T) {
	config := newTestConfig(func([]float64) float64 { return math.NaN() }, 3, -1, 1, 1)

	result, err := Optimize(config)
	if !errors.Is(err, ErrNoFiniteObjective) {
		t.Fatalf("Optimize() error = %v, want ErrNoFiniteObjective", err)
	}

	if result != nil {
		t.Fatalf("Optimize() result = %+v alongside error", result)
	}
}

func TestPrepareSwarmStepGatesEnemyIndependently(t *testing.T) {
	config := NewDefaultConfig()
	config.LowerBound = -100
	config.UpperBound = 100
	config.UseLevyWalk = false
	state := &runState{
		swarm: []Dragonfly{
			{Position: []float64{0, 0}, Step: []float64{0, 0}},
			{Position: []float64{1, 1}, Step: []float64{0, 0}},
		},
		food:  Best{Position: []float64{0, 0}},
		enemy: Best{Position: []float64{10, 10}},
	}
	weights := weightSchedule{Enemy: 1, Radius: 2, MaxStep: 100}

	prepareSwarmStep(state, 0, config, weights, rand.New(rand.NewSource(1)))

	assertVecEqual(t, state.swarm[0].Step, []float64{0, 0})
}

func TestFidelityModesSeparatePaperAndMATLABBranches(t *testing.T) {
	newState := func() *runState {
		return &runState{
			swarm: []Dragonfly{
				{Position: []float64{0, 0}, Step: []float64{0, 0}},
				{Position: []float64{1, 1}, Step: []float64{0, 0}},
			},
			food:  Best{Position: []float64{10, 10}},
			enemy: Best{Position: []float64{10, 10}},
		}
	}
	weights := weightSchedule{Separation: 1, Radius: 2, MaxStep: 100}

	paper := NewDefaultConfig()
	paper.LowerBound, paper.UpperBound = -100, 100
	paper.UseLevyWalk = false
	paperState := newState()
	prepareSwarmStep(paperState, 0, paper, weights, rand.New(rand.NewSource(1)))
	assertVecEqual(t, paperState.swarm[0].Step, []float64{1, 1})

	matlab := NewDefaultConfig()
	matlab.FidelityMode = FidelityMATLAB
	matlab.LowerBound, matlab.UpperBound = -100, 100
	matlab.UseLevyWalk = false
	matlabState := newState()
	prepareSwarmStep(matlabState, 0, matlab, weights, rand.New(rand.NewSource(1)))
	assertVecEqual(t, matlabState.swarm[0].Step, []float64{0, 0})
}

// TestOptimizeRejectsInvalidConfig asserts that validateConfig's rejections
// reach the caller through Optimize rather than being swallowed or panicking.
func TestOptimizeRejectsInvalidConfig(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*Config)
	}{
		{"no objective", func(c *Config) { c.ObjectiveFunc = nil }},
		{"zero problem size", func(c *Config) { c.ProblemSize = 0 }},
		{"inverted bounds", func(c *Config) { c.LowerBound, c.UpperBound = 10, -10 }},
		{"zero population", func(c *Config) { c.NPop = 0 }},
		{"zero iterations", func(c *Config) { c.MaxIterations = 0 }},
		{"negative workers", func(c *Config) { c.MaxWorkers = -1 }},
		{"non-finite weight", func(c *Config) { c.FoodWeight = math.NaN() }},
		{"zero radius divisor", func(c *Config) { c.RadiusInitialDivisor = 0 }},
		{"zero step ratio", func(c *Config) { c.MaxStepRatio = 0 }},
		{"enemy cutoff above one", func(c *Config) { c.EnemyCutoffFraction = 1.5 }},
		{"levy beta out of range", func(c *Config) { c.LevyBeta = 2 }},
		{"unknown boundary method", func(c *Config) { c.BoundaryMethod = "bounce" }},
		{"unknown fidelity mode", func(c *Config) { c.FidelityMode = "hybrid" }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := newTestConfig(Sphere, 4, -10, 10, 1)
			tt.mutate(config)

			result, err := Optimize(config)
			if err == nil {
				t.Fatal("Optimize() error = nil, want a validation error")
			}

			if result != nil {
				t.Error("Optimize() returned a result alongside an error")
			}
		})
	}

	nilResult, nilErr := Optimize(nil)
	if nilErr == nil || nilResult != nil {
		t.Errorf("Optimize(nil) = %v, %v, want nil result and an error", nilResult, nilErr)
	}
}

// TestOptimizeContextCancelled asserts the cancellation contract: ctx.Err() is
// returned verbatim and no partial result comes with it.
func TestOptimizeContextCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	result, err := OptimizeContext(ctx, newTestConfig(Sphere, 4, -10, 10, 1))
	if err == nil {
		t.Fatal("OptimizeContext() error = nil, want context.Canceled")
	}

	if !errors.Is(err, ctx.Err()) {
		t.Errorf("OptimizeContext() error = %v, want %v", err, ctx.Err())
	}

	if result != nil {
		t.Error("OptimizeContext() returned a result for a canceled run")
	}
}

// TestOptimizeContextCancelledMidRun cancels from inside the objective, so the
// cancellation is observed at an iteration boundary rather than before the run
// starts.
func TestOptimizeContextCancelledMidRun(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	calls := 0
	config := newTestConfig(Sphere, 4, -10, 10, 1)
	config.MaxIterations = 10000
	config.ObjectiveFunc = func(x []float64) float64 {
		calls++
		if calls > 200 {
			cancel()
		}

		return Sphere(x)
	}

	result, err := OptimizeContext(ctx, config)
	if err == nil {
		t.Fatal("OptimizeContext() error = nil, want context.Canceled")
	}

	if result != nil {
		t.Error("OptimizeContext() returned a result for a canceled run")
	}
}

// TestOptimizeRespectsBounds asserts that every position the run reports is
// inside the search box, for each boundary rule.
func TestOptimizeRespectsBounds(t *testing.T) {
	methods := []BoundaryMethod{BoundaryWrap, BoundaryClamp, BoundaryReflect}

	for _, method := range methods {
		t.Run(string(method), func(t *testing.T) {
			const (
				lower = -5.12
				upper = 5.12
			)

			config := newTestConfig(Rastrigin, 8, lower, upper, 99)
			config.BoundaryMethod = method

			result, err := Optimize(config)
			if err != nil {
				t.Fatalf("Optimize() error = %v", err)
			}

			assertWithinBounds(t, "GlobalBest", result.GlobalBest.Position, lower, upper)
			assertWithinBounds(t, "Worst", result.Worst.Position, lower, upper)
		})
	}
}

func assertWithinBounds(t *testing.T, label string, position []float64, lower, upper float64) {
	t.Helper()

	if len(position) == 0 {
		t.Fatalf("%s.Position is empty", label)
	}

	for i, value := range position {
		if value < lower || value > upper {
			t.Errorf("%s.Position[%d] = %g, outside [%g, %g]", label, i, value, lower, upper)
		}
	}
}

// TestOptimizeConvergenceCurve asserts the curve's two structural invariants:
// one entry per completed iteration, and a best-so-far that never worsens.
func TestOptimizeConvergenceCurve(t *testing.T) {
	config := newTestConfig(Sphere, 6, -100, 100, 7)

	result, err := Optimize(config)
	if err != nil {
		t.Fatalf("Optimize() error = %v", err)
	}

	if result.IterationCount != config.MaxIterations {
		t.Errorf("IterationCount = %d, want %d", result.IterationCount, config.MaxIterations)
	}

	if len(result.ConvergenceCurve) != result.IterationCount {
		t.Fatalf("len(ConvergenceCurve) = %d, want IterationCount = %d",
			len(result.ConvergenceCurve), result.IterationCount)
	}

	for i := 1; i < len(result.ConvergenceCurve); i++ {
		if result.ConvergenceCurve[i] > result.ConvergenceCurve[i-1] {
			t.Fatalf("ConvergenceCurve[%d] = %g worsened on [%d] = %g",
				i, result.ConvergenceCurve[i], i-1, result.ConvergenceCurve[i-1])
		}
	}

	last := result.ConvergenceCurve[len(result.ConvergenceCurve)-1]
	if last != result.GlobalBest.Cost {
		t.Errorf("last curve entry = %g, want GlobalBest.Cost = %g", last, result.GlobalBest.Cost)
	}

	if result.TerminationReason != TerminationMaxIterations {
		t.Errorf("TerminationReason = %q, want %q", result.TerminationReason, TerminationMaxIterations)
	}
}

// TestOptimizeReportsWorst asserts that the enemy is reported and is never
// better than the food.
func TestOptimizeReportsWorst(t *testing.T) {
	config := newTestConfig(Sphere, 6, -100, 100, 11)

	result, err := Optimize(config)
	if err != nil {
		t.Fatalf("Optimize() error = %v", err)
	}

	if len(result.Worst.Position) != config.ProblemSize {
		t.Fatalf("len(Worst.Position) = %d, want %d", len(result.Worst.Position), config.ProblemSize)
	}

	if result.Worst.Cost < result.GlobalBest.Cost {
		t.Errorf("Worst.Cost = %g is better than GlobalBest.Cost = %g",
			result.Worst.Cost, result.GlobalBest.Cost)
	}
}

// TestOptimizeFuncEvalCount pins the evaluation budget: the initial swarm plus
// one full re-evaluation per iteration.
func TestOptimizeFuncEvalCount(t *testing.T) {
	config := newTestConfig(Sphere, 3, -10, 10, 5)

	result, err := Optimize(config)
	if err != nil {
		t.Fatalf("Optimize() error = %v", err)
	}

	want := config.NPop * (config.MaxIterations + 1)
	if result.FuncEvalCount != want {
		t.Errorf("FuncEvalCount = %d, want %d", result.FuncEvalCount, want)
	}
}

func TestFidelityModeControlsEvaluationLifecycle(t *testing.T) {
	tests := []struct {
		name FidelityMode
		want int
	}{
		{FidelityPaper, 4 * (3 + 1)},
		{FidelityMATLAB, 4 * 3},
	}

	for _, test := range tests {
		config := newTestConfig(Sphere, 2, -1, 1, 41)
		config.FidelityMode = test.name
		config.NPop = 4
		config.MaxIterations = 3

		result, err := Optimize(config)
		if err != nil {
			t.Fatalf("%s: Optimize() error = %v", test.name, err)
		}

		if result.FuncEvalCount != test.want {
			t.Errorf("%s: FuncEvalCount = %d, want %d", test.name, result.FuncEvalCount, test.want)
		}
	}
}

func TestFidelityModeControlsEarlyStopEvaluationLifecycle(t *testing.T) {
	target := 1e300
	tests := []struct {
		name FidelityMode
		want int
	}{
		{FidelityPaper, 8},
		{FidelityMATLAB, 4},
	}

	for _, test := range tests {
		config := newTestConfig(Sphere, 2, -1, 1, 42)
		config.FidelityMode = test.name
		config.NPop = 4
		config.MaxIterations = 5
		config.Convergence = &ConvergenceConfig{TargetCost: &target, MinIterations: 1}

		result, err := Optimize(config)
		if err != nil {
			t.Fatalf("%s: Optimize() error = %v", test.name, err)
		}

		if result.IterationCount != 1 || result.TerminationReason != TerminationTargetCost {
			t.Errorf("%s: termination = (%d, %q), want (1, %q)",
				test.name, result.IterationCount, result.TerminationReason, TerminationTargetCost)
		}

		if result.FuncEvalCount != test.want {
			t.Errorf("%s: FuncEvalCount = %d, want %d", test.name, result.FuncEvalCount, test.want)
		}
	}
}

func TestMATLABPopulationSnapshotContainsEvaluatedPositions(t *testing.T) {
	config := newTestConfig(Sphere, 1, 0, 1, 43)
	config.FidelityMode = FidelityMATLAB
	config.NPop = 2
	config.MaxIterations = 1

	var snapshot PopulationSnapshot

	result, err := OptimizeContext(context.Background(), config,
		WithInitialPopulation([][]float64{{0.25}, {0.75}}),
		WithPopulationObserver(func(observed PopulationSnapshot) { snapshot = observed }),
	)
	if err != nil {
		t.Fatalf("OptimizeContext() error = %v", err)
	}

	if snapshot.EvaluationCount != config.NPop {
		t.Fatalf("snapshot EvaluationCount = %d, want %d", snapshot.EvaluationCount, config.NPop)
	}

	for i := range snapshot.Swarm {
		fly := snapshot.Swarm[i]
		if fly.Cost != Sphere(fly.Position) {
			t.Errorf("swarm[%d] cost %v does not describe evaluated position %v", i, fly.Cost, fly.Position)
		}
	}

	if result.FuncEvalCount != config.NPop {
		t.Errorf("FuncEvalCount = %d, want final moved swarm to remain unevaluated at %d calls",
			result.FuncEvalCount, config.NPop)
	}
}

func TestMATLABMovementEnemyUsesOnlyStrictInteriorPositions(t *testing.T) {
	config := newTestConfig(func(x []float64) float64 {
		if x[0] == 0 {
			return 100
		}

		return 10
	}, 1, 0, 1, 44)
	config.FidelityMode = FidelityMATLAB
	config.NPop = 2
	config.MaxIterations = 1

	var snapshot PopulationSnapshot

	result, err := OptimizeContext(context.Background(), config,
		WithInitialPopulation([][]float64{{0}, {0.5}}),
		WithPopulationObserver(func(observed PopulationSnapshot) { snapshot = observed }),
	)
	if err != nil {
		t.Fatalf("OptimizeContext() error = %v", err)
	}

	if result.Worst.Cost != 100 || result.Worst.Position[0] != 0 {
		t.Errorf("Result.Worst = %+v, want actual evaluated worst {100, [0]}", result.Worst)
	}

	if snapshot.Worst.Cost != 10 || snapshot.Worst.Position[0] != 0.5 {
		t.Errorf("snapshot movement enemy = %+v, want strict-interior candidate {10, [0.5]}", snapshot.Worst)
	}
}

func TestMATLABBoundaryOrderIgnoresConfiguredMethod(t *testing.T) {
	config := NewDefaultConfig()
	config.FidelityMode = FidelityMATLAB
	config.BoundaryMethod = BoundaryReflect
	config.LowerBound = 0
	config.UpperBound = 1
	config.UseLevyWalk = false

	state := &runState{
		swarm:         []Dragonfly{{Position: []float64{2}, Step: []float64{0.25}}},
		food:          Best{Position: []float64{2}},
		enemy:         Best{Position: []float64{0}},
		movementEnemy: Best{Position: []float64{0}},
	}
	weights := weightSchedule{Inertia: 1, Radius: 1, MaxStep: 10}
	rng := rand.New(rand.NewSource(45))
	wantRNG := rand.New(rand.NewSource(45))
	want := wantRNG.Float64()

	prepareSwarmStep(state, 0, config, weights, rng)

	if state.swarm[0].Position[0] != want || state.swarm[0].Step[0] != want {
		t.Errorf("MATLAB pre-wrap then move = (position %v, step %v), want (%v, %v)",
			state.swarm[0].Position[0], state.swarm[0].Step[0], want, want)
	}
}

// TestFoodInRadiusIncludesZeroDistance is the regression test for the trap that
// makes foodInRadius a separate helper from withinRadius: a dragonfly sitting
// exactly on the food source must see the food as in range.
func TestFoodInRadiusIncludesZeroDistance(t *testing.T) {
	position := []float64{1, 2, 3}

	if !foodInRadius(position, []float64{1, 2, 3}, 0.5) {
		t.Error("foodInRadius() = false for a dragonfly on the food source, want true")
	}

	if withinRadius(position, []float64{1, 2, 3}, 0.5) {
		t.Error("withinRadius() = true for an all-zero distance; the two helpers must differ here")
	}

	if foodInRadius(position, []float64{1, 2, 9}, 0.5) {
		t.Error("foodInRadius() = true for a food source outside the radius")
	}

	if foodInRadius(position, []float64{1, 2}, 0.5) {
		t.Error("foodInRadius() = true for a mismatched food vector")
	}

	if foodInRadius(position, []float64{1, 2, math.NaN()}, math.Inf(1)) {
		t.Error("foodInRadius() = true for a NaN component")
	}
}

// TestOptimizeSurvivesHostileObjective asserts that NaN and infinite costs do
// not pin the incumbent: the run still ends with a usable finite answer.
func TestOptimizeSurvivesHostileObjective(t *testing.T) {
	config := newTestConfig(Sphere, 4, -10, 10, 13)
	config.ObjectiveFunc = func(x []float64) float64 {
		if x[0] < 0 {
			return math.NaN()
		}

		return Sphere(x)
	}

	result, err := Optimize(config)
	if err != nil {
		t.Fatalf("Optimize() error = %v", err)
	}

	if math.IsNaN(result.GlobalBest.Cost) || math.IsInf(result.GlobalBest.Cost, 0) {
		t.Errorf("GlobalBest.Cost = %g, want a finite cost", result.GlobalBest.Cost)
	}

	for i, value := range result.GlobalBest.Position {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			t.Errorf("GlobalBest.Position[%d] = %g, want a finite value", i, value)
		}
	}
}

func TestOptimizeContextRejectsArchiveObserver(t *testing.T) {
	config := NewDefaultConfig()
	config.ObjectiveFunc = Sphere
	config.ProblemSize = 2
	config.LowerBound = -1
	config.UpperBound = 1
	config.MaxIterations = 2

	result, err := OptimizeContext(context.Background(), config,
		WithArchiveObserver(func(ArchiveSnapshot) {}))
	if err == nil {
		t.Fatal("a single-objective run accepted WithArchiveObserver")
	}

	if result != nil {
		t.Error("a rejected run must not return a partial result")
	}
}
