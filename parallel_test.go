package dragonfly

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
)

// parallelCase is one configuration axis the determinism test sweeps. Each case
// changes exactly one thing about an otherwise identical run, so a failure
// names the feature that broke rather than "some parallel run".
type parallelCase struct {
	name  string
	apply func(*Config)
}

// parallelTestCases covers every code path in the step update whose behavior a
// worker goroutine could plausibly disturb: the weight schedules (automatic and
// pinned), all three boundary rules, the Lévy branch in both states, and both
// constraint-handling policies.
func parallelTestCases() []parallelCase {
	return []parallelCase{
		{name: "default", apply: func(*Config) {}},
		{name: "matlab_fidelity", apply: func(config *Config) { config.FidelityMode = FidelityMATLAB }},
		{name: "pinned_weights", apply: func(config *Config) {
			config.SeparationWeight = 0.1
			config.AlignmentWeight = 0.2
			config.CohesionWeight = 0.3
			config.FoodWeight = 1.5
			// Zero is a pinned value, not the WeightAuto sentinel; a run that
			// switches the enemy term off must stay deterministic too.
			config.EnemyWeight = 0
		}},
		{name: "boundary_wrap", apply: func(config *Config) { config.BoundaryMethod = BoundaryWrap }},
		{name: "boundary_clamp", apply: func(config *Config) { config.BoundaryMethod = BoundaryClamp }},
		{name: "boundary_reflect", apply: func(config *Config) { config.BoundaryMethod = BoundaryReflect }},
		{name: "levy_off", apply: func(config *Config) { config.UseLevyWalk = false }},
		{name: "levy_on_tight_radius", apply: func(config *Config) {
			// A small, slowly growing radius isolates dragonflies, which is what
			// drives them down the Lévy branch.
			config.UseLevyWalk = true
			config.RadiusInitialDivisor = 200
			config.RadiusGrowth = 0
		}},
		{name: "constrained_feasibility", apply: func(config *Config) {
			config.Constraints = parallelTestConstraints(ConstraintHandlingFeasibility)
		}},
		{name: "constrained_penalty", apply: func(config *Config) {
			config.Constraints = parallelTestConstraints(ConstraintHandlingPenalty)
		}},
		{name: "early_stopping", apply: func(config *Config) {
			config.Convergence = &ConvergenceConfig{
				MinImprovement:       1,
				StagnationIterations: 3,
				MinIterations:        2,
			}
		}},
	}
}

// parallelTestConstraints makes roughly half the search box infeasible, so that
// both branches of the ranking rules are exercised and the food source and the
// enemy actually change hands during the run.
func parallelTestConstraints(handling ConstraintHandlingMethod) *ConstraintConfig {
	return &ConstraintConfig{
		Handling:      handling,
		PenaltyMethod: PenaltyQuadratic,
		PenaltyFactor: 10,
		Inequalities: []ConstraintFunction{
			func(x []float64) float64 { return x[0] - 1 },
			func(x []float64) float64 { return -x[1] - 2 },
		},
	}
}

// parallelTestConfig builds the run every determinism case starts from. The
// swarm is small and the run short, because the property under test is exact
// equality rather than solution quality.
func parallelTestConfig(seed int64) *Config {
	config := NewDefaultConfig()
	config.ObjectiveFunc = Rastrigin
	config.ProblemSize = 5
	config.LowerBound = -5.12
	config.UpperBound = 5.12
	config.NPop = 20
	config.MaxIterations = 40
	config.Rand = rand.New(rand.NewSource(seed))

	return config
}

// TestParallelIsDeterministicForSeedAcrossSchedules is the guard on the rule
// that the whole parallel/sequential split exists to keep: every random number
// a run consumes is drawn on the calling goroutine, so a seeded run must be
// BIT-IDENTICAL whether or not the objective calls fan out.
//
// It compares whole Results rather than only the final cost. A single
// rng.Float64() moved into a worker, or a reduction that resolves a tie by
// arrival order instead of by swarm index, changes the trajectory long before
// it changes the answer -- and on an easy function it may never change the
// answer at all. The convergence curve and the best position are what catch it.
func TestParallelIsDeterministicForSeedAcrossSchedules(t *testing.T) {
	seeds := []int64{1, 7, 42}
	// One worker, a couple of workers, and more workers than there are
	// dragonflies -- the last one leaves workers idle and is where an
	// off-by-one in the index handout would show.
	workerCounts := []int{1, 2, 4, 64}

	for _, testCase := range parallelTestCases() {
		for _, seed := range seeds {
			sequentialConfig := parallelTestConfig(seed)
			testCase.apply(sequentialConfig)

			sequential, err := Optimize(sequentialConfig)
			if err != nil {
				t.Fatalf("%s seed %d: sequential Optimize() error = %v", testCase.name, seed, err)
			}

			for _, workers := range workerCounts {
				parallelConfig := parallelTestConfig(seed)
				testCase.apply(parallelConfig)
				parallelConfig.EnableParallel = true
				parallelConfig.MaxWorkers = workers

				parallel, err := Optimize(parallelConfig)
				if err != nil {
					t.Fatalf("%s seed %d workers %d: parallel Optimize() error = %v",
						testCase.name, seed, workers, err)
				}

				name := fmt.Sprintf("%s/seed=%d/workers=%d", testCase.name, seed, workers)
				assertResultsIdentical(t, name, sequential, parallel)
			}
		}
	}
}

// assertResultsIdentical compares two Results field by field with exact float
// equality. Seed is deliberately not compared: both runs were handed their own
// *rand.Rand, so the recorded fallback seed is a wall-clock value neither run
// used.
func assertResultsIdentical(t *testing.T, name string, want, got *Result) {
	t.Helper()

	if got.TerminationReason != want.TerminationReason {
		t.Errorf("%s: TerminationReason = %q, want %q", name, got.TerminationReason, want.TerminationReason)
	}

	if got.FuncEvalCount != want.FuncEvalCount {
		t.Errorf("%s: FuncEvalCount = %d, want %d", name, got.FuncEvalCount, want.FuncEvalCount)
	}

	if got.IterationCount != want.IterationCount {
		t.Errorf("%s: IterationCount = %d, want %d", name, got.IterationCount, want.IterationCount)
	}

	assertBestIdentical(t, name+" GlobalBest", want.GlobalBest, got.GlobalBest)
	assertBestIdentical(t, name+" Worst", want.Worst, got.Worst)

	if len(got.ConvergenceCurve) != len(want.ConvergenceCurve) {
		t.Fatalf("%s: len(ConvergenceCurve) = %d, want %d",
			name, len(got.ConvergenceCurve), len(want.ConvergenceCurve))
	}

	for i := range want.ConvergenceCurve {
		if got.ConvergenceCurve[i] != want.ConvergenceCurve[i] {
			t.Fatalf("%s: ConvergenceCurve[%d] = %v, want %v",
				name, i, got.ConvergenceCurve[i], want.ConvergenceCurve[i])
		}
	}
}

func assertBestIdentical(t *testing.T, name string, want, got Best) {
	t.Helper()

	if got.Cost != want.Cost {
		t.Errorf("%s: Cost = %v, want %v", name, got.Cost, want.Cost)
	}

	if got.ConstraintViolation != want.ConstraintViolation {
		t.Errorf("%s: ConstraintViolation = %v, want %v", name, got.ConstraintViolation, want.ConstraintViolation)
	}

	if len(got.Position) != len(want.Position) {
		t.Fatalf("%s: len(Position) = %d, want %d", name, len(got.Position), len(want.Position))
	}

	for i := range want.Position {
		if got.Position[i] != want.Position[i] {
			t.Errorf("%s: Position[%d] = %v, want %v", name, i, got.Position[i], want.Position[i])
		}
	}
}

// TestParallelRunWithDefaultWorkerCount checks that leaving MaxWorkers at its
// default -- one worker per CPU -- is the same run as any other worker count,
// since that is the configuration a caller who only sets EnableParallel gets.
func TestParallelRunWithDefaultWorkerCount(t *testing.T) {
	sequential, err := Optimize(parallelTestConfig(3))
	if err != nil {
		t.Fatalf("sequential Optimize() error = %v", err)
	}

	config := parallelTestConfig(3)
	config.EnableParallel = true
	config.MaxWorkers = 0

	parallel, err := Optimize(config)
	if err != nil {
		t.Fatalf("parallel Optimize() error = %v", err)
	}

	assertResultsIdentical(t, "default worker count", sequential, parallel)
}

// newCancellationTestState builds a parallel runState by hand, with every
// dragonfly carrying a recognizable sentinel cost, so a test can tell an
// untouched swarm from a partially committed one.
func newCancellationTestState(objective ObjectiveFunction, npop, workers int) *runState {
	const sentinelCost = 12345.0

	swarm := make([]Dragonfly, npop)
	for i := range swarm {
		swarm[i] = Dragonfly{
			Position:            []float64{float64(i), float64(i)},
			Step:                []float64{0, 0},
			Cost:                sentinelCost,
			ConstraintViolation: sentinelCost,
		}
	}

	evaluator := newConstraintEvaluator(objective, nil)

	return &runState{
		evaluator: evaluator,
		pool:      newEvaluationPool(evaluator, workers, npop),
		swarm:     swarm,
		food: Best{
			Position:            make([]float64, 2),
			Cost:                math.Inf(1),
			ConstraintViolation: math.Inf(1),
		},
		enemy: Best{
			Position: make([]float64, 2),
			Cost:     math.Inf(-1),
		},
	}
}

// assertSwarmUncommitted reports any dragonfly whose cost left the sentinel,
// which would mean a canceled batch committed part of its work.
func assertSwarmUncommitted(t *testing.T, state *runState) {
	t.Helper()

	const sentinelCost = 12345.0

	for i := range state.swarm {
		if state.swarm[i].Cost != sentinelCost {
			t.Errorf("swarm[%d].Cost = %v after a canceled batch, want the untouched sentinel %v",
				i, state.swarm[i].Cost, sentinelCost)
		}
	}

	if !math.IsInf(state.food.Cost, 1) {
		t.Errorf("food.Cost = %v after a canceled batch, want the untouched +Inf", state.food.Cost)
	}

	if !math.IsInf(state.enemy.Cost, -1) {
		t.Errorf("enemy.Cost = %v after a canceled batch, want the untouched -Inf", state.enemy.Cost)
	}
}

// TestParallelCancellationDoesNotCommitPartialBatch pins the rule that a
// canceled iteration leaves no trace: the swarm carries either every cost from
// this iteration or every cost from the previous one, never a mixture.
//
// It matters because the alternative is silent. A half-scored population still
// optimizes, still converges, and still returns a plausible answer -- it just
// ranks some dragonflies by a cost belonging to a position they no longer
// occupy.
func TestParallelCancellationDoesNotCommitPartialBatch(t *testing.T) {
	t.Run("canceled before the batch starts", func(t *testing.T) {
		state := newCancellationTestState(Sphere, 32, 4)

		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		err := state.evaluate(ctx)
		if err == nil {
			t.Fatal("evaluate() error = nil for a canceled context, want context.Canceled")
		}

		assertSwarmUncommitted(t, state)

		if state.funcEvals != 0 {
			t.Errorf("funcEvals = %d after a canceled batch, want 0", state.funcEvals)
		}
	})

	t.Run("canceled midway through the batch", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		var calls atomic.Int64

		objective := func(x []float64) float64 {
			// Cancel from inside the very first objective call, so the batch is
			// interrupted with most of its indices still unissued.
			if calls.Add(1) == 1 {
				cancel()
			}

			return Sphere(x)
		}

		state := newCancellationTestState(objective, 256, 4)

		err := state.evaluate(ctx)
		if err == nil {
			t.Fatal("evaluate() error = nil for a batch canceled midway, want context.Canceled")
		}

		if calls.Load() >= 256 {
			t.Errorf("objective called %d times, want the batch to stop short of all 256", calls.Load())
		}

		assertSwarmUncommitted(t, state)
	})

	t.Run("a canceled run returns no result", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		config := parallelTestConfig(11)
		config.EnableParallel = true

		result, err := OptimizeContext(ctx, config)
		if err == nil {
			t.Fatal("OptimizeContext() error = nil for a canceled context, want context.Canceled")
		}

		if result != nil {
			t.Errorf("OptimizeContext() result = %+v for a canceled context, want nil", result)
		}
	})
}

// TestBatchBestBreaksTiesByStableIndex is the tie-breaking guard. Two equally
// good dragonflies must resolve to the same one whichever order the workers
// happen to offer them in, and that one has to be the one the sequential loop
// would have kept: the earlier index.
func TestBatchBestBreaksTiesByStableIndex(t *testing.T) {
	evaluator := newConstraintEvaluator(Sphere, nil)

	// Indices 1 and 3 tie for best, 0 and 4 tie for worst.
	costs := []float64{9, 2, 5, 2, 9}

	orders := [][]int{
		{0, 1, 2, 3, 4},
		{4, 3, 2, 1, 0},
		{3, 1, 4, 0, 2},
	}

	for _, order := range orders {
		best := newBatchBest(evaluator)
		worst := newBatchWorst(evaluator)

		for _, index := range order {
			evaluation := CandidateEvaluation{Cost: costs[index]}
			best.consider(index, []float64{costs[index]}, evaluation)
			worst.consider(index, []float64{costs[index]}, evaluation)
		}

		if best.index != 1 {
			t.Errorf("offered in order %v: best.index = %d, want the earlier of the tied minima (1)",
				order, best.index)
		}

		if worst.index != 0 {
			t.Errorf("offered in order %v: worst.index = %d, want the earlier of the tied maxima (0)",
				order, worst.index)
		}
	}
}

// TestBatchBestMatchesSequentialScan pins the reduction against the loop it has
// to imitate, on a swarm built so that the best and the worst cost each occur
// twice.
func TestBatchBestMatchesSequentialScan(t *testing.T) {
	evaluator := newConstraintEvaluator(Sphere, nil)
	costs := []float64{4, 1, 7, 1, 7, 4}

	// The sequential loop's rule: scan in index order, replace only on a strict
	// improvement.
	wantBest, wantWorst := 0, 0

	for i := 1; i < len(costs); i++ {
		if costs[i] < costs[wantBest] {
			wantBest = i
		}

		if costs[i] > costs[wantWorst] {
			wantWorst = i
		}
	}

	best := newBatchBest(evaluator)
	worst := newBatchWorst(evaluator)

	// Offer them backwards, the order a scheduler is least likely to produce by
	// accident.
	for i, cost := range slices.Backward(costs) {
		evaluation := CandidateEvaluation{Cost: cost}
		best.consider(i, []float64{cost}, evaluation)
		worst.consider(i, []float64{cost}, evaluation)
	}

	if best.index != wantBest {
		t.Errorf("best.index = %d, want %d (the sequential scan's choice)", best.index, wantBest)
	}

	if worst.index != wantWorst {
		t.Errorf("worst.index = %d, want %d (the sequential scan's choice)", worst.index, wantWorst)
	}
}

// TestBatchBestConcurrentConsiderIsOrderIndependent offers the same batch from
// many goroutines at once, which is how the pool actually uses it.
func TestBatchBestConcurrentConsiderIsOrderIndependent(t *testing.T) {
	evaluator := newConstraintEvaluator(Sphere, nil)
	costs := []float64{3, 1, 8, 1, 8, 3}

	for range 50 {
		best := newBatchBest(evaluator)
		worst := newBatchWorst(evaluator)

		var offered sync.WaitGroup

		offered.Add(len(costs))

		for i := range costs {
			go func() {
				defer offered.Done()

				evaluation := CandidateEvaluation{Cost: costs[i]}
				best.consider(i, []float64{costs[i]}, evaluation)
				worst.consider(i, []float64{costs[i]}, evaluation)
			}()
		}

		offered.Wait()

		if best.index != 1 || worst.index != 2 {
			t.Fatalf("concurrent consider: best.index = %d, worst.index = %d; want 1 and 2",
				best.index, worst.index)
		}
	}
}

// TestMergeBestKeepsIncumbentOnATie checks the half of the two-stage reduction
// that makes it agree with the sequential loop: an incumbent food source or
// enemy that merely ties the batch's extreme keeps its place, because the
// sequential loop replaces only on a strict improvement.
func TestMergeBestKeepsIncumbentOnATie(t *testing.T) {
	evaluator := newConstraintEvaluator(Sphere, nil)

	incumbent := Best{Position: []float64{1, 1}, Cost: 5}

	tied := newBatchBest(evaluator)
	tied.consider(0, []float64{2, 2}, CandidateEvaluation{Cost: 5})
	mergeBest(&incumbent, tied)

	if incumbent.Position[0] != 1 {
		t.Errorf("mergeBest() replaced the incumbent on a tie: Position = %v, want [1 1]", incumbent.Position)
	}

	better := newBatchBest(evaluator)
	better.consider(0, []float64{3, 3}, CandidateEvaluation{Cost: 4})
	mergeBest(&incumbent, better)

	if incumbent.Cost != 4 || incumbent.Position[0] != 3 {
		t.Errorf("mergeBest() = {Cost: %v, Position: %v}, want a strict improvement to {4, [3 3]}",
			incumbent.Cost, incumbent.Position)
	}

	empty := newBatchBest(evaluator)
	mergeBest(&incumbent, empty)

	if incumbent.Cost != 4 {
		t.Errorf("mergeBest() with an empty batch changed the incumbent to %v, want 4 untouched", incumbent.Cost)
	}
}

// TestParallelForRunsEveryIndexOnce guards the index handout itself: every
// index runs, and none runs twice, at any worker count.
func TestParallelForRunsEveryIndexOnce(t *testing.T) {
	for _, workers := range []int{0, 1, 3, 8, 64} {
		const count = 37

		visits := make([]atomic.Int64, count)

		err := parallelFor(context.Background(), count, workers, func(i int) {
			visits[i].Add(1)
		})
		if err != nil {
			t.Fatalf("workers %d: parallelFor() error = %v", workers, err)
		}

		for i := range visits {
			if got := visits[i].Load(); got != 1 {
				t.Errorf("workers %d: index %d ran %d times, want exactly once", workers, i, got)
			}
		}
	}

	emptyErr := parallelFor(context.Background(), 0, 4, func(int) {
		t.Error("parallelFor() ran work for an empty range")
	})
	if emptyErr != nil {
		t.Errorf("parallelFor() with no work: error = %v, want nil", emptyErr)
	}
}

// BenchmarkNeighborScan measures the per-iteration cost of the neighborhood
// scan, the O(n^2 * d) part of a DA iteration and the only candidate for a
// second parallel phase.
//
// It is here so that decision can be made on evidence rather than on the shape
// of the complexity: the scan draws no random numbers, so parallelizing it
// would be sound, but it is only worth the concurrency if it actually dominates
// an iteration at a swarm size anyone runs.
func BenchmarkNeighborScan(b *testing.B) {
	const span = 20.0

	// The two ends of the radius schedule. Early in a run the radius is a
	// quarter of the box and the per-dimension test short-circuits on the first
	// component that misses; late in a run it exceeds the box width, every
	// dragonfly is everyone's neighbor, and every scan runs to completion and
	// fills its result slice. The truth of a whole run lies between them.
	radii := []struct {
		name  string
		value float64
	}{
		{name: "early", value: span / 4},
		{name: "late", value: span/4 + span*2},
	}

	for _, npop := range []int{50, 100, 200, 500} {
		swarm := benchmarkSwarm(npop, 30, rand.New(rand.NewSource(1)))

		for _, radius := range radii {
			b.Run(fmt.Sprintf("npop=%d/radius=%s", npop, radius.name), func(b *testing.B) {
				found := 0

				for range b.N {
					for i := range swarm {
						found += len(findNeighbors(swarm, i, radius.value))
					}
				}

				// Consume the result so the scan cannot be optimized away.
				if found < 0 {
					b.Fatal("negative neighbor count")
				}
			})
		}
	}
}

// BenchmarkSwarmStep measures a whole prepare pass -- the scan plus the five
// primitives, the step update and the boundary repair -- so BenchmarkNeighborScan
// can be read as a fraction of something rather than in isolation.
func BenchmarkSwarmStep(b *testing.B) {
	for _, npop := range []int{50, 100, 200, 500} {
		config := NewDefaultConfig()
		config.ObjectiveFunc = Sphere
		config.ProblemSize = 30
		config.NPop = npop

		rng := rand.New(rand.NewSource(1))
		evaluator := newConstraintEvaluator(config.ObjectiveFunc, nil)
		state := &runState{
			evaluator: evaluator,
			swarm:     benchmarkSwarm(npop, config.ProblemSize, rng),
			food:      Best{Position: make([]float64, config.ProblemSize), Cost: math.Inf(1)},
			enemy:     Best{Position: make([]float64, config.ProblemSize), Cost: math.Inf(-1)},
		}
		weights := computeWeights(config, config.MaxIterations/2, config.MaxIterations, rng)

		b.Run(fmt.Sprintf("npop=%d", npop), func(b *testing.B) {
			for range b.N {
				prepareSwarm(state, config, weights, rng)
			}
		})
	}
}

func benchmarkSwarm(npop, problemSize int, rng *rand.Rand) []Dragonfly {
	swarm := make([]Dragonfly, npop)
	for i := range swarm {
		swarm[i] = Dragonfly{
			Position: unifrndVec(-10, 10, problemSize, rng),
			Step:     unifrndVec(-1, 1, problemSize, rng),
			Cost:     math.Inf(1),
		}
	}

	return swarm
}

// binaryParallelTestConfig builds the binary run every BDA determinism case
// starts from. The swarm is small and the run short, because the property under
// test is exact equality rather than solution quality.
func binaryParallelTestConfig(seed int64, transfer TransferFunction) *Config {
	config := NewBinaryConfig()
	config.ObjectiveFunc = binaryTestObjective
	config.ProblemSize = 24
	config.NPop = 20
	config.MaxIterations = 40
	config.TransferFunc = transfer
	config.Rand = rand.New(rand.NewSource(seed))

	return config
}

// binaryTestObjective is the binary determinism sweep's objective, a OneMax
// written as a minimization: it counts the unset bits, so driving it to zero
// sets every bit. It lives here rather than in functions.go because it exists
// only to give the binary runs a cheap landscape with a known optimum to move
// through.
func binaryTestObjective(x []float64) float64 {
	unset := 0.0
	for _, bit := range x {
		unset += 1 - bit
	}

	return unset
}

// TestBinaryParallelIsDeterministicForSeed is the BDA half of the guard
// TestParallelIsDeterministicForSeedAcrossSchedules pins for the continuous
// variant: a seeded binary run must be BIT-IDENTICAL whether or not the
// objective calls fan out.
//
// The binary loop draws one uniform per position component in flipBits, on top
// of everything the continuous step already draws, so it has strictly more RNG
// to lose track of. Every transfer function is swept because each one turns the
// same step into a different flip probability, and a stream that shifted by a
// single draw would show up as a different trajectory rather than as a
// different answer.
func TestBinaryParallelIsDeterministicForSeed(t *testing.T) {
	seeds := []int64{1, 7, 42}
	// One worker, a couple of workers, and more workers than there are
	// dragonflies -- the last one leaves workers idle and is where an
	// off-by-one in the index handout would show.
	workerCounts := []int{1, 2, 4, 64}

	for _, transfer := range TransferFunctionNames() {
		for _, seed := range seeds {
			sequential, err := OptimizeBinary(binaryParallelTestConfig(seed, transfer))
			if err != nil {
				t.Fatalf("%s seed %d: sequential OptimizeBinary() error = %v", transfer, seed, err)
			}

			assertBinaryPositions(t, fmt.Sprintf("%s/seed=%d sequential", transfer, seed), sequential)

			for _, workers := range workerCounts {
				config := binaryParallelTestConfig(seed, transfer)
				config.EnableParallel = true
				config.MaxWorkers = workers

				parallel, err := OptimizeBinary(config)
				if err != nil {
					t.Fatalf("%s seed %d workers %d: parallel OptimizeBinary() error = %v",
						transfer, seed, workers, err)
				}

				name := fmt.Sprintf("%s/seed=%d/workers=%d", transfer, seed, workers)
				assertResultsIdentical(t, name, sequential, parallel)
				assertBinaryPositions(t, name, parallel)
			}
		}
	}
}

// assertBinaryPositions checks the invariant the parallel path must not be
// allowed to break: the positions a binary run reports are still bit vectors.
// A worker that mutated the swarm instead of only reading it would show up here
// as a component that is neither zero nor one.
func assertBinaryPositions(t *testing.T, name string, result *Result) {
	t.Helper()

	if !BinaryPositionsValid(result.GlobalBest.Position) {
		t.Errorf("%s: GlobalBest.Position = %v, want a 0/1 vector", name, result.GlobalBest.Position)
	}

	if !BinaryPositionsValid(result.Worst.Position) {
		t.Errorf("%s: Worst.Position = %v, want a 0/1 vector", name, result.Worst.Position)
	}
}

// TestBinaryParallelWithConstraintsAndEarlyStopping sweeps the two features
// that change what the reduction and the loop have to agree on -- the
// constraint ranking the food and enemy are chosen by, and the stagnation
// tracker that decides when the run ends -- over a parallel binary run.
func TestBinaryParallelWithConstraintsAndEarlyStopping(t *testing.T) {
	cases := parallelBinaryCases()

	for _, testCase := range cases {
		for _, seed := range []int64{2, 13} {
			sequentialConfig := binaryParallelTestConfig(seed, DefaultTransferFunction)
			testCase.apply(sequentialConfig)

			sequential, err := OptimizeBinary(sequentialConfig)
			if err != nil {
				t.Fatalf("%s seed %d: sequential OptimizeBinary() error = %v", testCase.name, seed, err)
			}

			for _, workers := range []int{1, 3, 64} {
				config := binaryParallelTestConfig(seed, DefaultTransferFunction)
				testCase.apply(config)
				config.EnableParallel = true
				config.MaxWorkers = workers

				parallel, err := OptimizeBinary(config)
				if err != nil {
					t.Fatalf("%s seed %d workers %d: parallel OptimizeBinary() error = %v",
						testCase.name, seed, workers, err)
				}

				name := fmt.Sprintf("%s/seed=%d/workers=%d", testCase.name, seed, workers)
				assertResultsIdentical(t, name, sequential, parallel)
				assertBinaryPositions(t, name, parallel)
			}
		}
	}
}

// parallelBinaryCases are the binary-run variations worth sweeping: pinned
// weights, both constraint policies, and early stopping.
func parallelBinaryCases() []parallelCase {
	// The constraint is on the bit string itself: the first two bits must be
	// set, which roughly half the swarm violates at any time, so both branches
	// of the ranking rules are exercised and the food and enemy change hands.
	binaryConstraints := func(handling ConstraintHandlingMethod) *ConstraintConfig {
		return &ConstraintConfig{
			Handling:      handling,
			PenaltyMethod: PenaltyQuadratic,
			PenaltyFactor: 10,
			Inequalities: []ConstraintFunction{
				func(x []float64) float64 { return 1 - x[0] },
				func(x []float64) float64 { return 1 - x[1] },
			},
		}
	}

	return []parallelCase{
		{name: "binary_default", apply: func(*Config) {}},
		{name: "binary_matlab_fidelity", apply: func(config *Config) {
			config.FidelityMode = FidelityMATLAB
		}},
		{name: "binary_pinned_weights", apply: func(config *Config) {
			config.SeparationWeight = 0.1
			config.AlignmentWeight = 0.2
			config.CohesionWeight = 0.3
			config.FoodWeight = 1.5
			config.EnemyWeight = 0
		}},
		{name: "binary_constrained_feasibility", apply: func(config *Config) {
			config.Constraints = binaryConstraints(ConstraintHandlingFeasibility)
		}},
		{name: "binary_constrained_penalty", apply: func(config *Config) {
			config.Constraints = binaryConstraints(ConstraintHandlingPenalty)
		}},
		{name: "binary_early_stopping", apply: func(config *Config) {
			config.Convergence = &ConvergenceConfig{
				MinImprovement:       1,
				StagnationIterations: 3,
				MinIterations:        2,
			}
		}},
	}
}

// TestBinaryParallelCancellationReturnsNoResult pins the cancellation half of
// the contract for the binary loop: a canceled parallel run reports the error
// and hands back nothing, so a caller cannot mistake a half-scored population
// for a finished run.
func TestBinaryParallelCancellationReturnsNoResult(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	config := binaryParallelTestConfig(11, DefaultTransferFunction)
	config.EnableParallel = true

	result, err := OptimizeBinaryContext(ctx, config)
	if err == nil {
		t.Fatal("OptimizeBinaryContext() error = nil for a canceled context, want context.Canceled")
	}

	if result != nil {
		t.Errorf("OptimizeBinaryContext() result = %+v for a canceled context, want nil", result)
	}
}

// TestNewEvaluationPoolNormalizesCapacity covers the degenerate capacities the
// optimization loop never passes: a negative capacity must be clamped to zero
// rather than reaching make, and the buffer must still grow on first use.
func TestNewEvaluationPoolNormalizesCapacity(t *testing.T) {
	evaluator := newConstraintEvaluator(Sphere, nil)

	tests := []struct {
		name           string
		capacity       int
		wantCapAtLeast int
	}{
		{"negative capacity", -8, 0},
		{"zero capacity", 0, 0},
		{"positive capacity", 16, 16},
	}

	for _, test := range tests {
		pool := newEvaluationPool(evaluator, 2, test.capacity)
		if pool == nil {
			t.Fatalf("%s: newEvaluationPool returned nil", test.name)
		}

		if len(pool.scores) != 0 {
			t.Errorf("%s: pool starts with %d scores, want 0", test.name, len(pool.scores))
		}

		if cap(pool.scores) < test.wantCapAtLeast {
			t.Errorf("%s: pool scratch capacity = %d, want at least %d",
				test.name, cap(pool.scores), test.wantCapAtLeast)
		}

		if pool.maxWorkers != 2 {
			t.Errorf("%s: pool.maxWorkers = %d, want 2", test.name, pool.maxWorkers)
		}
	}

	// A pool built with a non-positive capacity still evaluates a full swarm:
	// the scratch buffer grows on the first batch.
	pool := newEvaluationPool(evaluator, 2, -1)
	swarm := []Dragonfly{
		{Position: []float64{1, 0}, Step: []float64{0, 0}},
		{Position: []float64{0, 0}, Step: []float64{0, 0}},
		{Position: []float64{3, 4}, Step: []float64{0, 0}},
	}

	batch, err := pool.evaluate(context.Background(), swarm)
	if err != nil {
		t.Fatalf("evaluate on a zero-capacity pool: %v", err)
	}

	if len(batch.scores) != len(swarm) {
		t.Fatalf("evaluate scored %d of %d dragonflies", len(batch.scores), len(swarm))
	}

	if batch.best.best.Cost != 0 {
		t.Errorf("batch best cost = %v, want 0", batch.best.best.Cost)
	}

	if batch.worst.best.Cost != 25 {
		t.Errorf("batch worst cost = %v, want 25", batch.worst.best.Cost)
	}
}
