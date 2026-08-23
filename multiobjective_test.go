package dragonfly

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"log/slog"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
)

// newTestArchive builds an archive whose grid bounds are exactly [0,10] in both
// objectives, so every cell assignment below can be worked out by hand: with
// NGrid = 10 each bin is one unit wide, and bin m of objective k is [k, k+1).
//
// The four corner points are mutually non-dominated (they lie on the descending
// anti-diagonal), which is what makes them all survive insertion.
func newTestArchive(t *testing.T) *ParetoArchive {
	t.Helper()

	archive := NewParetoArchiveWithGrid(100, 10, 4, 2, 2)

	for _, values := range [][]float64{
		{0, 10},
		{10, 0},
		{2.5, 7.5},
		{7.5, 2.5},
	} {
		accepted := archive.Add(&ParetoSolution{
			Position:        []float64{values[0]},
			ObjectiveValues: values,
		}, nil)
		if !accepted {
			t.Fatalf("Add(%v) rejected a non-dominated solution", values)
		}
	}

	return archive
}

// TestDominates checks the domination predicate, including the two cases that
// are easy to get backwards: equal vectors do not dominate each other, and
// vectors of different lengths are incomparable.
func TestDominates(t *testing.T) {
	tests := []struct {
		name     string
		a        []float64
		b        []float64
		expected bool
	}{
		{name: "strictly_better_in_all", a: []float64{1, 1}, b: []float64{2, 2}, expected: true},
		{name: "better_in_one_equal_in_other", a: []float64{1, 2}, b: []float64{2, 2}, expected: true},
		{name: "equal", a: []float64{1, 2}, b: []float64{1, 2}, expected: false},
		{name: "worse_in_one", a: []float64{1, 3}, b: []float64{2, 2}, expected: false},
		{name: "strictly_worse", a: []float64{3, 3}, b: []float64{2, 2}, expected: false},
		{name: "length_mismatch", a: []float64{1}, b: []float64{2, 2}, expected: false},
		{name: "empty", a: []float64{}, b: []float64{}, expected: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := dominates(tt.a, tt.b); got != tt.expected {
				t.Errorf("dominates(%v, %v) = %v, want %v", tt.a, tt.b, got, tt.expected)
			}
		})
	}
}

// TestParetoArchiveRejectsDominatedAndDuplicate checks the two rejection rules:
// a candidate an archived solution already dominates never enters, and neither
// does a second copy of an objective vector the archive already holds.
func TestParetoArchiveRejectsDominatedAndDuplicate(t *testing.T) {
	archive := NewParetoArchive(10)

	if !archive.Add(&ParetoSolution{Position: []float64{0}, ObjectiveValues: []float64{1, 1}}, nil) {
		t.Fatal("first Add rejected")
	}

	if archive.Add(&ParetoSolution{Position: []float64{1}, ObjectiveValues: []float64{2, 2}}, nil) {
		t.Error("Add accepted a dominated candidate")
	}

	if archive.Add(&ParetoSolution{Position: []float64{2}, ObjectiveValues: []float64{1, 1}}, nil) {
		t.Error("Add accepted a duplicate objective vector")
	}

	if archive.Len() != 1 {
		t.Errorf("archive length = %d, want 1", archive.Len())
	}
}

// TestParetoArchiveRemovesNewlyDominated checks that a candidate that dominates
// archived solutions evicts them, rather than sitting alongside them.
func TestParetoArchiveRemovesNewlyDominated(t *testing.T) {
	archive := NewParetoArchive(10)

	for _, values := range [][]float64{{5, 5}, {6, 4}, {4, 6}} {
		archive.Add(&ParetoSolution{Position: []float64{0}, ObjectiveValues: values}, nil)
	}

	if archive.Len() != 3 {
		t.Fatalf("setup archive length = %d, want 3", archive.Len())
	}

	if !archive.Add(&ParetoSolution{Position: []float64{0}, ObjectiveValues: []float64{1, 1}}, nil) {
		t.Fatal("Add rejected a dominating candidate")
	}

	if archive.Len() != 1 {
		t.Errorf("archive length = %d, want 1 after a dominating insert", archive.Len())
	}

	if !archive.IsNonDominated() {
		t.Error("archive is not mutually non-dominated")
	}
}

// TestParetoArchiveInvariantHoldsAfterEveryMutation is the invariant test the
// plan calls for. Non-domination and the capacity bound are asserted after every
// single Add, not once at the end: archive maintenance is exactly where the
// invariant breaks silently, and a run that ends non-dominated may still have
// passed through states that were not.
func TestParetoArchiveInvariantHoldsAfterEveryMutation(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	archive := NewParetoArchiveWithGrid(15, 10, 4, 2, 2)

	for i := range 400 {
		candidate := &ParetoSolution{
			Position:        []float64{rng.Float64(), rng.Float64()},
			ObjectiveValues: []float64{rng.Float64() * 10, rng.Float64() * 10},
		}

		archive.Add(candidate, rng)

		if !archive.IsNonDominated() {
			t.Fatalf("archive is not mutually non-dominated after insert %d", i)
		}

		if archive.Len() > archive.MaxSize {
			t.Fatalf("archive length %d exceeds MaxSize %d after insert %d",
				archive.Len(), archive.MaxSize, i)
		}

		for j, solution := range archive.Solutions {
			if len(solution.GridIndex) != len(solution.ObjectiveValues) {
				t.Fatalf("solution %d has grid index %v for objectives %v after insert %d",
					j, solution.GridIndex, solution.ObjectiveValues, i)
			}
		}
	}
}

// TestParetoArchiveNeverExceedsMaxSize drives the archive with points that are
// all mutually non-dominated, so nothing is ever evicted by domination and the
// only thing keeping the archive at capacity is the hypercube eviction.
func TestParetoArchiveNeverExceedsMaxSize(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	archive := NewParetoArchiveWithGrid(20, 10, 4, 2, 2)

	// Points on the curve f2 = 1/f1 are mutually non-dominated by construction.
	for i := 1; i <= 500; i++ {
		f1 := float64(i)

		archive.Add(&ParetoSolution{
			Position:        []float64{f1},
			ObjectiveValues: []float64{f1, 1 / f1},
		}, rng)

		if archive.Len() > 20 {
			t.Fatalf("archive length %d exceeds MaxSize 20 at insert %d", archive.Len(), i)
		}
	}

	if archive.Len() != 20 {
		t.Errorf("archive length = %d, want a full archive of 20", archive.Len())
	}
}

// TestHypercubeGridAssignment checks the cell coordinates of a hand-constructed
// archive against values worked out by hand. With bounds [0,10] in both
// objectives and NGrid = 10, bin k of an objective is the interval [k, k+1),
// and a value sitting exactly on the upper bound is clamped into the last bin
// rather than falling one past the grid.
func TestHypercubeGridAssignment(t *testing.T) {
	archive := newTestArchive(t)

	expected := map[float64][]int{
		0.0: {0, 9}, // (0, 10): first bin of f1, last bin of f2 after clamping
		10.: {9, 0}, // (10, 0): clamped into the last bin of f1
		2.5: {2, 7}, // (2.5, 7.5)
		7.5: {7, 2}, // (7.5, 2.5)
	}

	for _, solution := range archive.Solutions {
		want, ok := expected[solution.ObjectiveValues[0]]
		if !ok {
			t.Fatalf("unexpected archived objective vector %v", solution.ObjectiveValues)
		}

		if len(solution.GridIndex) != 2 || solution.GridIndex[0] != want[0] || solution.GridIndex[1] != want[1] {
			t.Errorf("grid index of %v = %v, want %v",
				solution.ObjectiveValues, solution.GridIndex, want)
		}

		// The key flattens the coordinate in base NGrid: i0 + 10*i1.
		wantKey := want[0] + 10*want[1]
		if solution.GridKey != wantKey {
			t.Errorf("grid key of %v = %d, want %d",
				solution.ObjectiveValues, solution.GridKey, wantKey)
		}
	}
}

// TestOccupiedCellsIsSortedAndComplete checks that every archived solution is
// accounted for exactly once and that the cells come back in ascending key
// order -- the ordering a seeded roulette draw depends on.
func TestOccupiedCellsIsSortedAndComplete(t *testing.T) {
	archive := newTestArchive(t)
	cells := archive.occupiedCells()

	if len(cells) != 4 {
		t.Fatalf("occupied cells = %d, want 4 distinct cells", len(cells))
	}

	seen := 0

	for i, cell := range cells {
		seen += len(cell.Members)

		if i > 0 && cells[i-1].Key >= cell.Key {
			t.Errorf("cells are not in ascending key order: %d then %d", cells[i-1].Key, cell.Key)
		}
	}

	if seen != archive.Len() {
		t.Errorf("cells hold %d members, want %d", seen, archive.Len())
	}
}

// newSelectionArchive builds an archive with two lone solutions and one cluster
// of five, all in distinct hypercubes, so a selection bias is visible as a
// simple frequency count. Bounds are again [0,10] in both objectives.
//
// The cluster sits inside the box [7,7.1) x [2.9,3.0), so all five share the
// cell (7, 2); the two lone points are the corners (0,10) and (10,0).
func newSelectionArchive(t *testing.T) *ParetoArchive {
	t.Helper()

	archive := NewParetoArchiveWithGrid(100, 10, 4, 2, 2)

	values := [][]float64{
		{0, 10},
		{10, 0},
		{7.01, 2.99},
		{7.02, 2.98},
		{7.03, 2.97},
		{7.04, 2.96},
		{7.05, 2.95},
	}

	for _, objective := range values {
		if !archive.Add(&ParetoSolution{
			Position:        []float64{objective[0]},
			ObjectiveValues: objective,
		}, nil) {
			t.Fatalf("Add(%v) rejected a non-dominated solution", objective)
		}
	}

	cells := archive.occupiedCells()
	if len(cells) != 3 {
		t.Fatalf("selection archive has %d cells, want 3", len(cells))
	}

	return archive
}

// isClustered reports whether a selected solution came from the crowded cell of
// newSelectionArchive.
func isClustered(solution *ParetoSolution) bool {
	return solution != nil && solution.ObjectiveValues[0] > 7 && solution.ObjectiveValues[0] < 7.1
}

// TestSelectSparsePrefersTheSparseHypercube asserts the food-selection bias
// statistically. With Beta = 4 the crowded cell of five carries weight 1/5^4 =
// 1/625 against 1 for each lone cell, so it should almost never be drawn.
func TestSelectSparsePrefersTheSparseHypercube(t *testing.T) {
	archive := newSelectionArchive(t)
	rng := rand.New(rand.NewSource(3))

	const draws = 2000

	clustered := 0

	for range draws {
		if isClustered(archive.selectSparse(rng)) {
			clustered++
		}
	}

	// The analytic probability is 1/627 ≈ 0.16%; 5% leaves ample room for
	// sampling noise while still failing an unbiased or inverted selection.
	if float64(clustered)/draws > 0.05 {
		t.Errorf("food selection drew the crowded cell %d/%d times, want well under 5%%", clustered, draws)
	}
}

// TestSelectCrowdedPrefersTheCrowdedHypercube asserts the enemy-selection bias.
// With Gamma = 2 the crowded cell carries weight 5² = 25 against 1 each, so it
// should be drawn about 93% of the time.
func TestSelectCrowdedPrefersTheCrowdedHypercube(t *testing.T) {
	archive := newSelectionArchive(t)
	rng := rand.New(rand.NewSource(5))

	const draws = 2000

	clustered := 0

	for range draws {
		if isClustered(archive.selectCrowded(rng)) {
			clustered++
		}
	}

	if float64(clustered)/draws < 0.8 {
		t.Errorf("enemy selection drew the crowded cell %d/%d times, want well over 80%%", clustered, draws)
	}
}

// TestEvictionPrefersTheCrowdedHypercube checks the overflow rule: filling an
// archive one past capacity should almost always cost the crowded cell a
// member, not one of the two lone corner solutions.
func TestEvictionPrefersTheCrowdedHypercube(t *testing.T) {
	rng := rand.New(rand.NewSource(13))

	const trials = 200

	cornersLost := 0

	for range trials {
		archive := newSelectionArchive(t)
		archive.MaxSize = archive.Len()

		// A seventh non-dominated point in a cell of its own, forcing exactly
		// one eviction.
		archive.Add(&ParetoSolution{
			Position:        []float64{4},
			ObjectiveValues: []float64{4.5, 5.5},
		}, rng)

		corners := 0

		for _, solution := range archive.Solutions {
			if solution.ObjectiveValues[0] == 0 || solution.ObjectiveValues[0] == 10 {
				corners++
			}
		}

		if corners < 2 {
			cornersLost++
		}
	}

	// Weight N^2 gives the crowded cell 25 against 1 for each of the four
	// single-member cells, so a corner is lost about 7% of the time.
	if float64(cornersLost)/trials > 0.25 {
		t.Errorf("eviction removed a lone corner in %d/%d trials, want well under 25%%", cornersLost, trials)
	}
}

// TestRouletteIndexHandlesDegenerateWeights checks the fallbacks: a nil RNG is
// deterministic, and weights carrying no information yield a uniform draw
// rather than an out-of-range index.
func TestRouletteIndexHandlesDegenerateWeights(t *testing.T) {
	if got := rouletteIndex([]float64{1, 2, 3}, nil); got != 0 {
		t.Errorf("rouletteIndex with a nil rng = %d, want 0", got)
	}

	rng := rand.New(rand.NewSource(1))

	for range 100 {
		got := rouletteIndex([]float64{0, 0, 0}, rng)
		if got < 0 || got > 2 {
			t.Fatalf("rouletteIndex with zero weights = %d, want an index in [0,2]", got)
		}
	}

	if got := rouletteIndex(nil, rng); got != 0 {
		t.Errorf("rouletteIndex(nil) = %d, want 0", got)
	}
}

// newMultiObjectiveTestConfig builds a small, fast MODA configuration on the
// unit box, seeded for reproducibility.
func newMultiObjectiveTestConfig(objective MultiObjectiveFunction, dims, iterations int, seed int64) *MultiObjectiveConfig {
	config := NewMultiObjectiveConfig()
	config.ObjectiveFunc = objective
	config.ArchiveSize = 50
	config.Swarm.ProblemSize = dims
	config.Swarm.LowerBound = 0
	config.Swarm.UpperBound = 1
	config.Swarm.NPop = 30
	config.Swarm.MaxIterations = iterations
	config.Swarm.Seed = &seed

	return config
}

// TestOptimizeMultiObjectiveValidation checks that the entry point rejects the
// configurations it cannot run, rather than panicking partway through a loop.
func TestOptimizeMultiObjectiveValidation(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*MultiObjectiveConfig)
	}{
		{name: "nil_objective", mutate: func(c *MultiObjectiveConfig) { c.ObjectiveFunc = nil }},
		{name: "nil_swarm", mutate: func(c *MultiObjectiveConfig) { c.Swarm = nil }},
		{name: "zero_archive_size", mutate: func(c *MultiObjectiveConfig) { c.ArchiveSize = 0 }},
		{name: "zero_ngrid", mutate: func(c *MultiObjectiveConfig) { c.NGrid = 0 }},
		{name: "negative_beta", mutate: func(c *MultiObjectiveConfig) { c.Beta = -1 }},
		{name: "bad_bounds", mutate: func(c *MultiObjectiveConfig) { c.Swarm.LowerBound = 2 }},
		{name: "penalty_handling", mutate: func(c *MultiObjectiveConfig) {
			c.Swarm.Constraints = &ConstraintConfig{
				Handling:     ConstraintHandlingPenalty,
				Inequalities: []ConstraintFunction{func(x []float64) float64 { return x[0] - 0.5 }},
			}
		}},
		{name: "target_cost", mutate: func(c *MultiObjectiveConfig) {
			target := 0.0
			c.Swarm.Convergence = &ConvergenceConfig{TargetCost: &target}
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := newMultiObjectiveTestConfig(ZDT1, 3, 5, 1)
			tt.mutate(config)

			_, err := OptimizeMultiObjective(context.Background(), config)
			if err == nil {
				t.Error("OptimizeMultiObjective accepted an invalid configuration")
			}
		})
	}

	_, err := OptimizeMultiObjective(context.Background(), nil)
	if err == nil {
		t.Error("OptimizeMultiObjective accepted a nil config")
	}
}

// TestOptimizeMultiObjectiveCancellation checks that an already-canceled context
// aborts the run without reporting a partial result.
func TestOptimizeMultiObjectiveCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	result, err := OptimizeMultiObjective(ctx, newMultiObjectiveTestConfig(ZDT1, 3, 50, 1))
	if err == nil {
		t.Fatal("a canceled run returned no error")
	}

	if result != nil {
		t.Error("a canceled run returned a result")
	}
}

// TestOptimizeMultiObjectiveIsDeterministicForSeed checks that two runs of the
// same seed produce identical archives -- the same solutions, in the same order,
// with the same objective values.
func TestOptimizeMultiObjectiveIsDeterministicForSeed(t *testing.T) {
	first, err := OptimizeMultiObjective(context.Background(), newMultiObjectiveTestConfig(ZDT1, 4, 40, 99))
	if err != nil {
		t.Fatalf("first run failed: %v", err)
	}

	second, err := OptimizeMultiObjective(context.Background(), newMultiObjectiveTestConfig(ZDT1, 4, 40, 99))
	if err != nil {
		t.Fatalf("second run failed: %v", err)
	}

	if first.Archive.Len() != second.Archive.Len() {
		t.Fatalf("archive sizes differ: %d and %d", first.Archive.Len(), second.Archive.Len())
	}

	if first.FuncEvalCount != second.FuncEvalCount {
		t.Errorf("evaluation counts differ: %d and %d", first.FuncEvalCount, second.FuncEvalCount)
	}

	for i := range first.Archive.Solutions {
		a := first.Archive.Solutions[i]
		b := second.Archive.Solutions[i]

		for m := range a.ObjectiveValues {
			if a.ObjectiveValues[m] != b.ObjectiveValues[m] {
				t.Fatalf("solution %d objective %d differs: %v and %v",
					i, m, a.ObjectiveValues[m], b.ObjectiveValues[m])
			}
		}

		for j := range a.Position {
			if a.Position[j] != b.Position[j] {
				t.Fatalf("solution %d component %d differs: %v and %v", i, j, a.Position[j], b.Position[j])
			}
		}
	}
}

func TestOptimizeMultiObjectiveFidelityEvaluationLifecycle(t *testing.T) {
	tests := []struct {
		name       string
		fidelity   FidelityMode
		iterations int
		wantEvals  int
	}{
		{name: "paper_one_iteration", fidelity: FidelityPaper, iterations: 1, wantEvals: 8},
		{name: "matlab_one_iteration", fidelity: FidelityMATLAB, iterations: 1, wantEvals: 4},
		{name: "paper_three_iterations", fidelity: FidelityPaper, iterations: 3, wantEvals: 16},
		{name: "matlab_three_iterations", fidelity: FidelityMATLAB, iterations: 3, wantEvals: 12},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			config := newMultiObjectiveTestConfig(ZDT1, 2, testCase.iterations, 73)
			config.Swarm.NPop = 4
			config.Swarm.FidelityMode = testCase.fidelity

			result, err := OptimizeMultiObjective(context.Background(), config)
			if err != nil {
				t.Fatalf("run failed: %v", err)
			}

			if result.FuncEvalCount != testCase.wantEvals {
				t.Errorf("FuncEvalCount = %d, want %d", result.FuncEvalCount, testCase.wantEvals)
			}

			if result.IterationCount != testCase.iterations {
				t.Errorf("IterationCount = %d, want %d", result.IterationCount, testCase.iterations)
			}
		})
	}
}

func TestOptimizeMultiObjectiveMATLABOneIterationArchivesInitialPopulation(t *testing.T) {
	initial := [][]float64{{0, 0}, {0.25, 0}, {0.75, 0}}
	config := newMultiObjectiveTestConfig(ZDT1, 2, 1, 81)
	config.Swarm.NPop = len(initial)
	config.Swarm.FidelityMode = FidelityMATLAB

	result, err := OptimizeMultiObjective(
		context.Background(), config, WithInitialPopulation(initial))
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}

	if result.Archive.Len() != len(initial) {
		t.Fatalf("archive contains %d points, want the %d evaluated initial points",
			result.Archive.Len(), len(initial))
	}

	for _, solution := range result.Archive.Solutions {
		found := false
		for _, position := range initial {
			found = found || reflect.DeepEqual(solution.Position, position)
		}

		if !found {
			t.Errorf("archive contains moved position %v; the final MATLAB movement must remain unevaluated",
				solution.Position)
		}

		if want := ZDT1(solution.Position); !reflect.DeepEqual(solution.ObjectiveValues, want) {
			t.Errorf("objectives %v do not match archived position %v (want %v)",
				solution.ObjectiveValues, solution.Position, want)
		}
	}
}

func TestComputeMATLABMODAWeightsScheduleAndRNGConsumption(t *testing.T) {
	config := NewDefaultConfig()
	config.LowerBound = 0
	config.UpperBound = 1

	const seed = int64(91)

	wantRNG := rand.New(rand.NewSource(seed))
	wantFood := 2 * wantRNG.Float64()
	wantNext := wantRNG.Float64()

	rng := rand.New(rand.NewSource(seed))
	first := computeMATLABMODAWeights(config, 1, 10, rng)

	if math.Abs(first.Inertia-0.83) > 1e-12 {
		t.Errorf("first one-based inertia = %g, want 0.83", first.Inertia)
	}

	for name, value := range map[string]float64{
		"separation": first.Separation,
		"alignment":  first.Alignment,
		"cohesion":   first.Cohesion,
		"enemy":      first.Enemy,
	} {
		if math.Abs(value-0.08) > 1e-12 {
			t.Errorf("first %s weight = %g, want 0.08", name, value)
		}
	}

	if first.Food != wantFood {
		t.Errorf("first food weight = %.17g, want %.17g", first.Food, wantFood)
	}

	if next := rng.Float64(); next != wantNext {
		t.Errorf("MATLAB MODA schedule consumed the wrong number of draws: next %.17g, want %.17g",
			next, wantNext)
	}

	last := computeMATLABMODAWeights(config, 10, 10, rand.New(rand.NewSource(seed)))
	if math.Abs(last.Inertia-0.2) > 1e-12 {
		t.Errorf("last inertia = %g, want 0.2", last.Inertia)
	}

	if last.Separation != 0 || last.Alignment != 0 || last.Cohesion != 0 || last.Enemy != 0 {
		t.Errorf("last automatic swarm weights = S:%g A:%g C:%g E:%g, want all zero",
			last.Separation, last.Alignment, last.Cohesion, last.Enemy)
	}
}

// TestOptimizeMultiObjectiveKeepsArchiveInvariants checks the two archive
// guarantees on a real run rather than on synthetic inserts.
func TestOptimizeMultiObjectiveKeepsArchiveInvariants(t *testing.T) {
	config := newMultiObjectiveTestConfig(ZDT1, 4, 60, 4)
	config.ArchiveSize = 12

	result, err := OptimizeMultiObjective(context.Background(), config)
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}

	if result.Archive.Len() > config.ArchiveSize {
		t.Errorf("archive length %d exceeds ArchiveSize %d", result.Archive.Len(), config.ArchiveSize)
	}

	if !result.Archive.IsNonDominated() {
		t.Error("the final archive is not mutually non-dominated")
	}

	if len(result.ArchiveSizeCurve) != result.IterationCount {
		t.Errorf("archive size curve has %d entries, want %d",
			len(result.ArchiveSizeCurve), result.IterationCount)
	}

	for i, size := range result.ArchiveSizeCurve {
		if size > config.ArchiveSize {
			t.Fatalf("archive size curve entry %d is %d, above ArchiveSize %d", i, size, config.ArchiveSize)
		}
	}

	for _, position := range result.Archive.Solutions {
		for j, value := range position.Position {
			if value < config.Swarm.LowerBound || value > config.Swarm.UpperBound {
				t.Fatalf("archived position component %d is %v, outside the search box", j, value)
			}
		}
	}
}

// TestMODAPaperQualityGate replaces the v0.1 single-seed legacy trajectory
// checks with a 15-seed gate on the paper-default algorithm. The thresholds are
// deliberately degradation bars rather than golden observations: at least
// twelve seeds must retain a populated, non-dominated front with useful
// normalized convergence, coverage, and hypervolume.
func TestMODAPaperQualityGate(t *testing.T) {
	if testing.Short() {
		t.Skip("the paper MODA quality gate runs 30 optimizations; skipped under -short")
	}

	tests := []struct {
		name            string
		maxGD           float64
		maxIGD          float64
		minHV           float64
		minimumSegments int
	}{
		{name: "ZDT1", maxGD: 0.40, maxIGD: 0.20, minHV: 0.70, minimumSegments: 1},
		{name: "ZDT3", maxGD: 0.20, maxIGD: 0.30, minHV: 0.60, minimumSegments: 3},
	}

	for _, testCase := range tests {
		t.Run(testCase.name, func(t *testing.T) {
			problem, ok := qualityProblemNamed(testCase.name)
			if !ok {
				t.Fatalf("missing quality definition for %s", testCase.name)
			}

			successes := 0

			for seed := int64(3000); seed <= 3014; seed++ {
				config := newMultiObjectiveTestConfig(problem.objective, 5, 400, seed)
				config.Swarm.NPop = 60

				result, err := OptimizeMultiObjective(context.Background(), config)
				if err != nil {
					t.Fatalf("seed %d failed: %v", seed, err)
				}

				metrics, metricErr := evaluateFrontQuality(result.Archive, problem)
				if metricErr != nil {
					t.Fatalf("seed %d metrics failed: %v", seed, metricErr)
				}

				passed := result.Archive.IsNonDominated() && result.Archive.Len() >= 10 &&
					metrics.gd <= testCase.maxGD && metrics.igd <= testCase.maxIGD &&
					metrics.hypervolumeRatio >= testCase.minHV &&
					metrics.segmentsCovered >= testCase.minimumSegments
				if passed {
					successes++
				}

				t.Logf("seed %d: archive=%d GD=%.4f IGD=%.4f HV=%.4f segments=%d pass=%t",
					seed, result.Archive.Len(), metrics.gd, metrics.igd,
					metrics.hypervolumeRatio, metrics.segmentsCovered, passed)
			}

			if successes < 12 {
				t.Errorf("%d/15 seeds passed the paper quality bar, want at least 12/15", successes)
			}
		})
	}
}

func TestZDT3QualityReferenceUsesOnlyTrueFrontIntervals(t *testing.T) {
	problem, ok := qualityProblemNamed("ZDT3")
	if !ok {
		t.Fatal("missing ZDT3 quality definition")
	}

	reference := sampleQualityFront(problem)
	if len(reference) != 10001 {
		t.Fatalf("ZDT3 reference contains %d samples, want 10001", len(reference))
	}

	for _, point := range reference {
		inside := false
		for _, interval := range zdt3TrueFrontIntervals {
			inside = inside || point.x >= interval.lo && point.x <= interval.hi
		}

		if !inside {
			t.Fatalf("reference includes f1=%g from a dominated gap", point.x)
		}
	}

	gapPoint := objectivePoint{x: 0.1, y: problem.front(0.1)}
	ideal, span := objectiveNormalization(reference)
	distance := math.Sqrt(nearestSquaredDistance(
		normalizeObjectivePoint(gapPoint, ideal, span),
		normalizeObjectivePoints(reference, ideal, span),
	))

	if distance < 0.01 {
		t.Errorf("a point on the dominated analytic gap is only %.6g from the true front", distance)
	}
}

func TestQualityMetricsRecognizeSampledZDT3Front(t *testing.T) {
	problem, ok := qualityProblemNamed("ZDT3")
	if !ok {
		t.Fatal("missing ZDT3 quality definition")
	}

	archive := NewParetoArchive(1000)

	reference := sampleQualityFront(problem)
	for i, point := range reference {
		if i%20 != 0 && i != len(reference)-1 {
			continue
		}

		archive.Solutions = append(archive.Solutions, &ParetoSolution{
			Position:        []float64{point.x, 0},
			ObjectiveValues: []float64{point.x, point.y},
		})
	}

	metrics, err := evaluateFrontQuality(archive, problem)
	if err != nil {
		t.Fatalf("evaluateFrontQuality failed: %v", err)
	}

	if metrics.gd != 0 || metrics.igd > 0.002 {
		t.Errorf("sampled analytic front has GD=%g IGD=%g, want GD zero and IGD <= 0.002",
			metrics.gd, metrics.igd)
	}

	if metrics.hypervolumeRatio < 0.995 || metrics.hypervolumeRatio > 1 {
		t.Errorf("sampled analytic front has hypervolume ratio %g, want [0.995, 1]", metrics.hypervolumeRatio)
	}

	if metrics.segmentsCovered != len(zdt3TrueFrontIntervals) {
		t.Errorf("sampled analytic front covers %d segments, want %d",
			metrics.segmentsCovered, len(zdt3TrueFrontIntervals))
	}
}

// TestOptimizeMultiObjectiveSchaffer checks the one-variable Schaffer N.1
// problem, whose front is the image of x in [0, 2].
func TestOptimizeMultiObjectiveSchaffer(t *testing.T) {
	config := newMultiObjectiveTestConfig(SchafferN1, 1, 200, 8)
	config.Swarm.LowerBound = -10
	config.Swarm.UpperBound = 10

	result, err := OptimizeMultiObjective(context.Background(), config)
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}

	if result.Archive.Len() < 5 {
		t.Fatalf("archive holds %d solutions, want at least 5", result.Archive.Len())
	}

	for _, solution := range result.Archive.Solutions {
		x := solution.Position[0]
		if x < -0.5 || x > 2.5 {
			t.Errorf("archived x = %v, want it near the Pareto set [0, 2]", x)
		}
	}
}

// TestExportParetoRoundTrip writes both exports into the test's temporary
// directory and reads them back. Nothing is written into the repository.
func TestExportParetoRoundTrip(t *testing.T) {
	result, err := OptimizeMultiObjective(context.Background(), newMultiObjectiveTestConfig(ZDT1, 3, 30, 21))
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}

	dir := t.TempDir()
	csvPath := filepath.Join(dir, "front.csv")
	jsonPath := filepath.Join(dir, "front.json")

	csvErr := result.ExportParetoCSV(csvPath)
	if csvErr != nil {
		t.Fatalf("ExportParetoCSV failed: %v", csvErr)
	}

	jsonErr := result.ExportParetoJSON(jsonPath)
	if jsonErr != nil {
		t.Fatalf("ExportParetoJSON failed: %v", jsonErr)
	}

	file, err := os.Open(csvPath)
	if err != nil {
		t.Fatalf("open exported CSV: %v", err)
	}

	defer file.Close()

	records, err := csv.NewReader(file).ReadAll()
	if err != nil {
		t.Fatalf("read exported CSV: %v", err)
	}

	if len(records) != result.Archive.Len()+1 {
		t.Fatalf("CSV holds %d rows, want %d plus a header", len(records)-1, result.Archive.Len())
	}

	// index + two objectives + three decision variables + constraint violation.
	if len(records[0]) != 7 {
		t.Errorf("CSV header has %d columns, want 7: %v", len(records[0]), records[0])
	}

	raw, err := os.ReadFile(jsonPath)
	if err != nil {
		t.Fatalf("read exported JSON: %v", err)
	}

	var document ParetoExport

	decodeErr := json.Unmarshal(raw, &document)
	if decodeErr != nil {
		t.Fatalf("decode exported JSON: %v", decodeErr)
	}

	if document.ArchiveSize != result.Archive.Len() || len(document.Front) != result.Archive.Len() {
		t.Errorf("JSON reports %d archived solutions and %d front points, want %d",
			document.ArchiveSize, len(document.Front), result.Archive.Len())
	}

	if document.Seed != result.Seed || document.SeedKnown != result.SeedKnown ||
		document.IterationCount != result.IterationCount {
		t.Error("JSON summary does not match the result it was exported from")
	}

	for i, point := range document.Front {
		want := result.Archive.Solutions[i].ObjectiveValues
		for m := range want {
			if point.Objectives[m] != want[m] {
				t.Fatalf("JSON point %d objective %d = %v, want %v", i, m, point.Objectives[m], want[m])
			}
		}
	}
}

// TestExportParetoNilResult checks that the exporters refuse a nil result
// instead of panicking.
func TestExportParetoNilResult(t *testing.T) {
	var result *MultiObjectiveResult

	dir := t.TempDir()

	csvErr := result.ExportParetoCSV(filepath.Join(dir, "front.csv"))
	if csvErr == nil {
		t.Error("ExportParetoCSV accepted a nil result")
	}

	jsonErr := result.ExportParetoJSON(filepath.Join(dir, "front.json"))
	if jsonErr == nil {
		t.Error("ExportParetoJSON accepted a nil result")
	}
}

// TestMultiObjectiveRejectsVaryingObjectiveCount checks that an objective
// function returning a different number of values between calls is reported as
// an error rather than silently producing an incomparable archive.
func TestMultiObjectiveRejectsVaryingObjectiveCount(t *testing.T) {
	calls := 0

	config := newMultiObjectiveTestConfig(func(x []float64) []float64 {
		calls++
		if calls > 3 {
			return []float64{x[0]}
		}

		return []float64{x[0], 1 - x[0]}
	}, 2, 5, 1)

	_, err := OptimizeMultiObjective(context.Background(), config)
	if err == nil {
		t.Error("a varying objective count was accepted")
	}
}

// TestParetoSolutionCloneIsADeepCopy covers the nil guard and pins the copy as
// deep: mutating the clone must not reach back into the original.
func TestParetoSolutionCloneIsADeepCopy(t *testing.T) {
	var absent *ParetoSolution
	if got := absent.clone(); got != nil {
		t.Errorf("(*ParetoSolution)(nil).clone() = %v, want nil", got)
	}

	original := &ParetoSolution{
		Position:         []float64{1, 2, 3},
		ObjectiveValues:  []float64{4, 5},
		GridIndex:        []int{6, 7},
		GridKey:          67,
		CrowdingDistance: 1.5,
		Rank:             2,
		DominationCount:  3,
	}

	copied := original.clone()
	if copied == original {
		t.Fatal("clone returned the receiver itself")
	}

	copied.Position[0] = -1
	copied.ObjectiveValues[1] = -1
	copied.GridIndex[0] = -1

	if original.Position[0] != 1 || original.ObjectiveValues[1] != 5 || original.GridIndex[0] != 6 {
		t.Errorf("clone aliased the original: %v / %v / %v",
			original.Position, original.ObjectiveValues, original.GridIndex)
	}

	if copied.GridKey != 67 {
		t.Errorf("clone GridKey = %d, want 67", copied.GridKey)
	}

	// The sorting bookkeeping is scratch space and is deliberately not copied.
	if copied.CrowdingDistance != 0 || copied.Rank != 0 || copied.DominationCount != 0 {
		t.Errorf("clone carried sorting scratch: distance %v rank %d dominated-by %d",
			copied.CrowdingDistance, copied.Rank, copied.DominationCount)
	}
}

// TestNonNegativeClampsInvalidExponents covers the guard that keeps a roulette
// exponent from inverting the preference it exists to express.
func TestNonNegativeClampsInvalidExponents(t *testing.T) {
	tests := []struct {
		name  string
		value float64
		want  float64
	}{
		{"zero", 0, 0},
		{"positive", 4, 4},
		{"large finite", 1e300, 1e300},
		{"negative", -2, 0},
		{"negative zero is zero", math.Copysign(0, -1), 0},
		{"NaN", math.NaN(), 0},
		{"negative infinity", math.Inf(-1), 0},
		{"positive infinity", math.Inf(1), 0},
	}

	for _, test := range tests {
		if got := nonNegative(test.value); got != test.want {
			t.Errorf("%s: nonNegative(%v) = %v, want %v", test.name, test.value, got, test.want)
		}
	}
}

// TestNewParetoArchiveWithGridFallsBackOnBadSizes covers the two defaulting
// branches a well-formed caller never takes.
func TestNewParetoArchiveWithGridFallsBackOnBadSizes(t *testing.T) {
	tests := []struct {
		name      string
		maxSize   int
		nGrid     int
		wantSize  int
		wantNGrid int
	}{
		{"honest values", 12, 5, 12, 5},
		{"zero max size", 0, 5, DefaultArchiveSize, 5},
		{"negative max size", -3, 5, DefaultArchiveSize, 5},
		{"zero grid", 12, 0, 12, DefaultArchiveNGrid},
		{"negative grid", 12, -1, 12, DefaultArchiveNGrid},
		{"both degenerate", 0, 0, DefaultArchiveSize, DefaultArchiveNGrid},
	}

	for _, test := range tests {
		archive := NewParetoArchiveWithGrid(test.maxSize, test.nGrid, -1, 2, math.NaN())
		if archive.MaxSize != test.wantSize {
			t.Errorf("%s: MaxSize = %d, want %d", test.name, archive.MaxSize, test.wantSize)
		}

		if archive.NGrid != test.wantNGrid {
			t.Errorf("%s: NGrid = %d, want %d", test.name, archive.NGrid, test.wantNGrid)
		}

		if archive.Beta != 0 || archive.Gamma != 2 || archive.Delta != 0 {
			t.Errorf("%s: exponents = (%v, %v, %v), want (0, 2, 0)",
				test.name, archive.Beta, archive.Gamma, archive.Delta)
		}
	}
}

// TestParetoArchiveLenAndNonDominationGuards covers the nil-receiver branches of
// Len and IsNonDominated, and the negative answer IsNonDominated must give when
// the invariant is deliberately broken.
func TestParetoArchiveLenAndNonDominationGuards(t *testing.T) {
	var absent *ParetoArchive

	if got := absent.Len(); got != 0 {
		t.Errorf("(*ParetoArchive)(nil).Len() = %d, want 0", got)
	}

	if !absent.IsNonDominated() {
		t.Error("(*ParetoArchive)(nil).IsNonDominated() = false, want true")
	}

	empty := NewParetoArchive(4)
	if got := empty.Len(); got != 0 {
		t.Errorf("empty archive Len = %d, want 0", got)
	}

	if !empty.IsNonDominated() {
		t.Error("an empty archive is not reported non-dominated")
	}

	// Two mutually non-dominated points on the anti-diagonal.
	front := NewParetoArchive(4)
	front.Solutions = []*ParetoSolution{
		{ObjectiveValues: []float64{1, 4}},
		{ObjectiveValues: []float64{4, 1}},
	}

	if front.Len() != 2 {
		t.Errorf("front Len = %d, want 2", front.Len())
	}

	if !front.IsNonDominated() {
		t.Error("a genuine front is not reported non-dominated")
	}

	// Injecting a dominated member behind Add's back must be detected: this is
	// the negative answer the invariant tests rely on.
	front.Solutions = append(front.Solutions, &ParetoSolution{ObjectiveValues: []float64{5, 5}})
	if front.IsNonDominated() {
		t.Error("IsNonDominated missed a dominated member")
	}
}

// TestArchivePositionHandlesAMissingSolution covers the nil guard that makes an
// empty archive yield the zero food and enemy vectors.
func TestArchivePositionHandlesAMissingSolution(t *testing.T) {
	if got := archivePosition(nil); got != nil {
		t.Errorf("archivePosition(nil) = %v, want nil", got)
	}

	position := []float64{1, 2, 3}

	got := archivePosition(&ParetoSolution{Position: position})
	if len(got) != len(position) {
		t.Fatalf("archivePosition returned %d components, want %d", len(got), len(position))
	}

	for i := range position {
		if got[i] != position[i] {
			t.Errorf("archivePosition()[%d] = %v, want %v", i, got[i], position[i])
		}
	}
}

// TestOptimizeMultiObjectiveStopsOnStagnation checks that a run whose archive
// stops accepting candidates ends before the iteration cap, and says so.
//
// The objective is constant, so the archive accepts exactly one solution -- the
// first candidate of the first evaluation -- and every iteration after that is
// stagnant by construction.
func TestOptimizeMultiObjectiveStopsOnStagnation(t *testing.T) {
	config := newMultiObjectiveTestConfig(func([]float64) []float64 { return []float64{1, 1} }, 3, 200, 7)
	config.Swarm.Convergence = &ConvergenceConfig{StagnationIterations: 5, MinIterations: 2}

	result, err := OptimizeMultiObjective(context.Background(), config)
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}

	if result.TerminationReason != TerminationStagnation {
		t.Errorf("termination reason is %q, want %q", result.TerminationReason, TerminationStagnation)
	}

	if result.IterationCount >= config.Swarm.MaxIterations {
		t.Errorf("run completed %d iterations, expected it to stop before %d",
			result.IterationCount, config.Swarm.MaxIterations)
	}

	if len(result.ArchiveSizeCurve) != result.IterationCount {
		t.Errorf("archive size curve has %d entries, want %d",
			len(result.ArchiveSizeCurve), result.IterationCount)
	}
}

// TestOptimizeMultiObjectiveRunsFullBudgetWithoutConvergence checks that the
// early-stopping machinery is inert when no Convergence block is configured: the
// run uses its whole budget and still reports the iteration cap.
func TestOptimizeMultiObjectiveRunsFullBudgetWithoutConvergence(t *testing.T) {
	config := newMultiObjectiveTestConfig(func([]float64) []float64 { return []float64{1, 1} }, 3, 25, 7)

	result, err := OptimizeMultiObjective(context.Background(), config)
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}

	if result.TerminationReason != TerminationMaxIterations {
		t.Errorf("termination reason is %q, want %q", result.TerminationReason, TerminationMaxIterations)
	}

	if result.IterationCount != config.Swarm.MaxIterations {
		t.Errorf("run completed %d iterations, want %d", result.IterationCount, config.Swarm.MaxIterations)
	}
}

// TestMOStagnationTrackerRespectsMinIterations checks the gate directly: the
// counter runs from the first observation, but the run may not stop before
// MinIterations completed iterations.
func TestMOStagnationTrackerRespectsMinIterations(t *testing.T) {
	tracker := &moStagnationTracker{
		config: &ConvergenceConfig{StagnationIterations: 2, MinIterations: 5},
	}

	for iteration := 1; iteration <= 4; iteration++ {
		if _, stop := tracker.observe(iteration, 0); stop {
			t.Fatalf("tracker stopped at iteration %d, before MinIterations", iteration)
		}
	}

	reason, stop := tracker.observe(5, 0)
	if !stop || reason != TerminationStagnation {
		t.Errorf("tracker returned (%q, %v) at the gate, want (%q, true)",
			reason, stop, TerminationStagnation)
	}
}

// TestMOStagnationTrackerResetsOnAcceptance checks that an accepted candidate
// clears the counter, so a run that keeps improving never stops early.
func TestMOStagnationTrackerResetsOnAcceptance(t *testing.T) {
	tracker := &moStagnationTracker{config: &ConvergenceConfig{StagnationIterations: 2}}

	for iteration := 1; iteration <= 10; iteration++ {
		accepted := iteration % 2
		if _, stop := tracker.observe(iteration, accepted); stop {
			t.Fatalf("tracker stopped at iteration %d despite regular acceptances", iteration)
		}
	}
}

// TestMOStagnationTrackerIsInertWithoutConfig checks that a nil Convergence
// block never stops a run.
func TestMOStagnationTrackerIsInertWithoutConfig(t *testing.T) {
	tracker := &moStagnationTracker{}

	if _, stop := tracker.observe(100, 0); stop {
		t.Error("a tracker with no config stopped the run")
	}
}

// TestConstrainedDominatesFollowsDebRules checks the partial order directly: it
// is the one place a feasibility mistake would silently reshape every archive.
func TestConstrainedDominatesFollowsDebRules(t *testing.T) {
	solution := func(violation float64, objectives ...float64) *ParetoSolution {
		return &ParetoSolution{ObjectiveValues: objectives, ConstraintViolation: violation}
	}

	tests := []struct {
		a        *ParetoSolution
		b        *ParetoSolution
		name     string
		expected bool
	}{
		{name: "feasible_beats_infeasible", a: solution(0, 9, 9), b: solution(1, 1, 1), expected: true},
		{name: "infeasible_loses_to_feasible", a: solution(1, 1, 1), b: solution(0, 9, 9), expected: false},
		{name: "smaller_violation_wins", a: solution(1, 9, 9), b: solution(2, 1, 1), expected: true},
		{name: "equal_violation_incomparable", a: solution(2, 1, 1), b: solution(2, 9, 9), expected: false},
		{name: "feasible_pareto", a: solution(0, 1, 1), b: solution(0, 2, 2), expected: true},
		{name: "feasible_non_dominating", a: solution(0, 1, 3), b: solution(0, 3, 1), expected: false},
		{name: "nil_never_dominates", a: nil, b: solution(0, 1, 1), expected: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := constrainedDominates(tt.a, tt.b); got != tt.expected {
				t.Errorf("constrainedDominates() = %v, want %v", got, tt.expected)
			}
		})
	}
}

// TestParetoArchiveKeepsInvariantWithMixedFeasibility checks that the archive
// still holds its non-domination invariant when it is fed a population that
// mixes feasible and infeasible candidates, and that feasibility wins.
func TestParetoArchiveKeepsInvariantWithMixedFeasibility(t *testing.T) {
	archive := NewParetoArchive(20)
	rng := rand.New(rand.NewSource(3))

	population := []*ParetoSolution{
		{Position: []float64{0}, ObjectiveValues: []float64{5, 5}, ConstraintViolation: 3},
		{Position: []float64{1}, ObjectiveValues: []float64{4, 6}, ConstraintViolation: 1},
		{Position: []float64{2}, ObjectiveValues: []float64{9, 9}, ConstraintViolation: 0},
		{Position: []float64{3}, ObjectiveValues: []float64{8, 10}, ConstraintViolation: 0},
		{Position: []float64{4}, ObjectiveValues: []float64{1, 1}, ConstraintViolation: 7},
	}

	archive.UpdateFromPopulation(population, rng)

	if !archive.IsNonDominated() {
		t.Fatal("the archive is not mutually non-dominated after a mixed insert")
	}

	for _, solution := range archive.Solutions {
		if !IsFeasible(solution.ConstraintViolation) {
			t.Errorf("archive kept an infeasible solution %v alongside feasible ones",
				solution.ObjectiveValues)
		}
	}
}

// TestOptimizeMultiObjectiveHonoursConstraints checks that a constrained run
// ends with a feasible-only archive.
//
// The inequality x0 >= 0.5 makes the unconstrained ZDT1 optimum -- the whole
// x0 < 0.5 stretch of the front -- infeasible, so an archive that ignored the
// constraint would be full of violating points.
func TestOptimizeMultiObjectiveHonoursConstraints(t *testing.T) {
	config := newMultiObjectiveTestConfig(ZDT1, 3, 60, 12)
	config.Swarm.Constraints = &ConstraintConfig{
		Inequalities: []ConstraintFunction{func(x []float64) float64 { return 0.5 - x[0] }},
	}

	result, err := OptimizeMultiObjective(context.Background(), config)
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}

	if result.Archive.Len() == 0 {
		t.Fatal("the constrained run produced an empty archive")
	}

	feasible := 0

	for _, solution := range result.Archive.Solutions {
		if IsFeasible(solution.ConstraintViolation) {
			feasible++
		}
	}

	if feasible == 0 {
		t.Fatal("the run never found a feasible point; the test problem is too hard to be a check")
	}

	if feasible != result.Archive.Len() {
		t.Errorf("archive holds %d feasible of %d solutions; once a feasible point exists "+
			"every infeasible one is dominated", feasible, result.Archive.Len())
	}

	if !result.Archive.IsNonDominated() {
		t.Error("the constrained archive is not mutually non-dominated")
	}
}

// TestOptimizeMultiObjectiveRecordsViolationOnDragonflies checks that the
// violation is recorded on the swarm as well as on the archive, so an
// unconstrained run reports zero rather than leaving the field stale.
func TestOptimizeMultiObjectiveUnconstrainedReportsZeroViolation(t *testing.T) {
	result, err := OptimizeMultiObjective(context.Background(), newMultiObjectiveTestConfig(ZDT1, 3, 20, 5))
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}

	for i, solution := range result.Archive.Solutions {
		if solution.ConstraintViolation != 0 {
			t.Fatalf("unconstrained solution %d reports violation %v", i, solution.ConstraintViolation)
		}
	}
}

// TestOptimizeMultiObjectiveParallelMatchesSequential is the acceptance test for
// the parallel path: a seeded run must be bit-identical with EnableParallel on
// and off, element for element.
//
// It is the multi-objective counterpart of
// TestParallelIsDeterministicForSeedAcrossSchedules. The guarantee holds for the
// same reason: no worker draws a random number, so the only thing EnableParallel
// changes is which goroutine calls the objective.
func TestOptimizeMultiObjectiveParallelMatchesSequential(t *testing.T) {
	const seed = 4242

	sequential, err := OptimizeMultiObjective(context.Background(),
		newMultiObjectiveTestConfig(ZDT1, 5, 40, seed))
	if err != nil {
		t.Fatalf("sequential run failed: %v", err)
	}

	parallelConfig := newMultiObjectiveTestConfig(ZDT1, 5, 40, seed)
	parallelConfig.Swarm.EnableParallel = true
	parallelConfig.Swarm.MaxWorkers = 4

	parallel, err := OptimizeMultiObjective(context.Background(), parallelConfig)
	if err != nil {
		t.Fatalf("parallel run failed: %v", err)
	}

	if parallel.FuncEvalCount != sequential.FuncEvalCount {
		t.Errorf("evaluation counts differ: sequential %d, parallel %d",
			sequential.FuncEvalCount, parallel.FuncEvalCount)
	}

	if parallel.IterationCount != sequential.IterationCount {
		t.Errorf("iteration counts differ: sequential %d, parallel %d",
			sequential.IterationCount, parallel.IterationCount)
	}

	if parallel.Archive.Len() != sequential.Archive.Len() {
		t.Fatalf("archive sizes differ: sequential %d, parallel %d",
			sequential.Archive.Len(), parallel.Archive.Len())
	}

	for i := range sequential.Archive.Solutions {
		want := sequential.Archive.Solutions[i]
		got := parallel.Archive.Solutions[i]

		for m := range want.ObjectiveValues {
			if got.ObjectiveValues[m] != want.ObjectiveValues[m] {
				t.Fatalf("solution %d objective %d differs: sequential %v, parallel %v",
					i, m, want.ObjectiveValues[m], got.ObjectiveValues[m])
			}
		}

		for j := range want.Position {
			if got.Position[j] != want.Position[j] {
				t.Fatalf("solution %d component %d differs: sequential %v, parallel %v",
					i, j, want.Position[j], got.Position[j])
			}
		}
	}
}

func TestOptimizeMultiObjectiveMATLABParallelMatchesSequential(t *testing.T) {
	const seed = 4243

	sequentialConfig := newMultiObjectiveTestConfig(ZDT1, 5, 40, seed)
	sequentialConfig.Swarm.FidelityMode = FidelityMATLAB

	sequential, err := OptimizeMultiObjective(context.Background(), sequentialConfig)
	if err != nil {
		t.Fatalf("sequential run failed: %v", err)
	}

	parallelConfig := newMultiObjectiveTestConfig(ZDT1, 5, 40, seed)
	parallelConfig.Swarm.FidelityMode = FidelityMATLAB
	parallelConfig.Swarm.EnableParallel = true
	parallelConfig.Swarm.MaxWorkers = 4

	parallel, err := OptimizeMultiObjective(context.Background(), parallelConfig)
	if err != nil {
		t.Fatalf("parallel run failed: %v", err)
	}

	if parallel.FuncEvalCount != sequential.FuncEvalCount ||
		parallel.IterationCount != sequential.IterationCount ||
		parallel.TerminationReason != sequential.TerminationReason {
		t.Fatalf("run metadata differs: sequential (%d, %d, %q), parallel (%d, %d, %q)",
			sequential.FuncEvalCount, sequential.IterationCount, sequential.TerminationReason,
			parallel.FuncEvalCount, parallel.IterationCount, parallel.TerminationReason)
	}

	if !reflect.DeepEqual(parallel.ArchiveSizeCurve, sequential.ArchiveSizeCurve) {
		t.Fatalf("archive curves differ: sequential %v, parallel %v",
			sequential.ArchiveSizeCurve, parallel.ArchiveSizeCurve)
	}

	if !reflect.DeepEqual(parallel.Archive.Solutions, sequential.Archive.Solutions) {
		t.Fatal("MATLAB archive differs between sequential and parallel evaluation")
	}
}

func TestOptimizeMultiObjectiveMATLABStagnationStopsAfterMovement(t *testing.T) {
	config := newMultiObjectiveTestConfig(func([]float64) []float64 { return []float64{0, 0} }, 2, 10, 4244)
	config.Swarm.FidelityMode = FidelityMATLAB
	config.Swarm.NPop = 4
	config.Swarm.Convergence = &ConvergenceConfig{
		StagnationIterations: 1,
		MinIterations:        1,
	}

	var snapshots []ArchiveSnapshot

	result, err := OptimizeMultiObjective(context.Background(), config,
		WithArchiveObserver(func(snapshot ArchiveSnapshot) { snapshots = append(snapshots, snapshot) }))
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}

	// Generation one evaluates the initial population and seeds the archive.
	// Generation two accepts no duplicate objective vector, but still completes
	// its final movement before stagnation terminates the run.
	if result.IterationCount != 2 || result.FuncEvalCount != 8 ||
		result.TerminationReason != TerminationStagnation {
		t.Fatalf("result = (iterations %d, evals %d, reason %q), want (2, 8, %q)",
			result.IterationCount, result.FuncEvalCount, result.TerminationReason, TerminationStagnation)
	}

	if len(snapshots) != 2 || snapshots[1].EvaluationCount != 8 {
		t.Fatalf("observer snapshots = %+v, want two completed evaluated generations", snapshots)
	}
}

// TestOptimizeMultiObjectiveParallelHonoursCancellation checks that a context
// canceled while the run is under way aborts it rather than being noticed only
// at the next iteration boundary.
func TestOptimizeMultiObjectiveParallelHonoursCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	config := newMultiObjectiveTestConfig(ZDT1, 3, 500, 8)
	config.Swarm.EnableParallel = true
	config.Swarm.MaxWorkers = 2

	var calls atomic.Int64

	front := ZDT1
	config.ObjectiveFunc = func(x []float64) []float64 {
		if calls.Add(1) == 100 {
			cancel()
		}

		return front(x)
	}

	result, err := OptimizeMultiObjective(ctx, config)
	if err == nil {
		t.Fatal("a canceled parallel run returned no error")
	}

	if result != nil {
		t.Error("a canceled parallel run returned a result")
	}
}

// TestMODAArchivePoliciesRemainDeterministic replaces the obsolete v0.1.0
// trajectory golden. Fidelity changes are expected to change trajectories;
// what must remain exact is reproducibility within each named policy.
func TestMODAArchivePoliciesRemainDeterministic(t *testing.T) {
	policies := []ArchivePolicy{
		ArchivePolicyPaperSegments,
		ArchivePolicyMATLABDensity,
		ArchivePolicyMOPSOGrid,
	}

	for _, policy := range policies {
		t.Run(string(policy), func(t *testing.T) {
			first, err := runPolicyMODA(ZDT1, 10, 11, 12, policy)
			if err != nil {
				t.Fatalf("first run: %v", err)
			}

			second, err := runPolicyMODA(ZDT1, 10, 11, 12, policy)
			if err != nil {
				t.Fatalf("second run: %v", err)
			}

			if !reflect.DeepEqual(first.Archive.Solutions, second.Archive.Solutions) {
				t.Fatal("same seed and archive policy produced different fronts")
			}

			if !first.Archive.IsNonDominated() || first.Archive.Len() > 12 {
				t.Fatalf("invalid final archive: length=%d, non-dominated=%v",
					first.Archive.Len(), first.Archive.IsNonDominated())
			}
		})
	}
}

func runPolicyMODA(
	objective MultiObjectiveFunction,
	dimension int,
	seed int64,
	archiveSize int,
	policy ArchivePolicy,
) (*MultiObjectiveResult, error) {
	config := NewMultiObjectiveConfig()
	config.ObjectiveFunc = objective
	config.ArchivePolicy = policy
	config.ArchiveSize = archiveSize
	config.Swarm.ProblemSize = dimension
	config.Swarm.LowerBound = 0
	config.Swarm.UpperBound = 1
	config.Swarm.MaxIterations = 25
	config.Swarm.NPop = 20
	config.Swarm.Seed = &seed

	return OptimizeMultiObjective(context.Background(), config)
}

// TestOptimizeMultiObjectiveNotifiesArchiveObserver pins where the observer
// fires as much as that it fires. The assertion that ties snapshot i's size to
// ArchiveSizeCurve[i] is the one that fails if the notification moves before
// advance or after the stagnation break.
func TestOptimizeMultiObjectiveNotifiesArchiveObserver(t *testing.T) {
	config := newMultiObjectiveTestConfig(ZDT1, 3, 10, 7)

	var snapshots []ArchiveSnapshot

	result, err := OptimizeMultiObjective(context.Background(), config,
		WithArchiveObserver(func(snapshot ArchiveSnapshot) {
			snapshots = append(snapshots, snapshot)
		}),
	)
	if err != nil {
		t.Fatalf("OptimizeMultiObjective: %v", err)
	}

	if len(snapshots) != result.IterationCount {
		t.Fatalf("observer called %d times, want one per completed iteration (%d)",
			len(snapshots), result.IterationCount)
	}

	for i, snapshot := range snapshots {
		if snapshot.Iteration != i+1 {
			t.Fatalf("snapshot %d reports iteration %d, want %d (one-based, no gaps)",
				i, snapshot.Iteration, i+1)
		}

		if len(snapshot.Solutions) != result.ArchiveSizeCurve[i] {
			t.Fatalf("snapshot %d holds %d solutions, want ArchiveSizeCurve[%d] = %d",
				i, len(snapshot.Solutions), i, result.ArchiveSizeCurve[i])
		}

		if snapshot.NGrid != config.NGrid {
			t.Errorf("snapshot %d reports NGrid %d, want %d", i, snapshot.NGrid, config.NGrid)
		}

		if len(snapshot.Solutions) > 0 {
			if len(snapshot.GridLower) != 2 || len(snapshot.GridUpper) != 2 {
				t.Fatalf("snapshot %d has grid bounds %v / %v, want one pair per objective",
					i, snapshot.GridLower, snapshot.GridUpper)
			}
		}

		if i > 0 && snapshot.EvaluationCount < snapshots[i-1].EvaluationCount {
			t.Errorf("evaluation count fell from %d to %d at snapshot %d",
				snapshots[i-1].EvaluationCount, snapshot.EvaluationCount, i)
		}
	}

	last := snapshots[len(snapshots)-1]
	if len(last.Solutions) != result.Archive.Len() {
		t.Errorf("final snapshot holds %d solutions, archive holds %d",
			len(last.Solutions), result.Archive.Len())
	}

	if last.EvaluationCount != result.FuncEvalCount {
		t.Errorf("final snapshot reports %d evaluations, result reports %d",
			last.EvaluationCount, result.FuncEvalCount)
	}
}

// TestOptimizeMultiObjectiveObserverDoesNotChangeASeededRun is the invariant
// the whole observer design rests on: watching a run must not be able to alter
// it, whether through the RNG or through evaluation ordering.
func TestOptimizeMultiObjectiveObserverDoesNotChangeASeededRun(t *testing.T) {
	plain, err := OptimizeMultiObjective(context.Background(),
		newMultiObjectiveTestConfig(ZDT1, 4, 30, 99))
	if err != nil {
		t.Fatalf("unobserved run failed: %v", err)
	}

	observed, err := OptimizeMultiObjective(context.Background(),
		newMultiObjectiveTestConfig(ZDT1, 4, 30, 99),
		WithArchiveObserver(func(ArchiveSnapshot) {}),
	)
	if err != nil {
		t.Fatalf("observed run failed: %v", err)
	}

	if plain.FuncEvalCount != observed.FuncEvalCount {
		t.Errorf("evaluation counts differ: %d and %d", plain.FuncEvalCount, observed.FuncEvalCount)
	}

	if plain.IterationCount != observed.IterationCount {
		t.Errorf("iteration counts differ: %d and %d", plain.IterationCount, observed.IterationCount)
	}

	if plain.TerminationReason != observed.TerminationReason {
		t.Errorf("termination reasons differ: %q and %q",
			plain.TerminationReason, observed.TerminationReason)
	}

	if !reflect.DeepEqual(plain.ArchiveSizeCurve, observed.ArchiveSizeCurve) {
		t.Errorf("archive size curves differ:\n%v\n%v", plain.ArchiveSizeCurve, observed.ArchiveSizeCurve)
	}

	if plain.Archive.Len() != observed.Archive.Len() {
		t.Fatalf("archive sizes differ: %d and %d", plain.Archive.Len(), observed.Archive.Len())
	}

	for i := range plain.Archive.Solutions {
		a := plain.Archive.Solutions[i]
		b := observed.Archive.Solutions[i]

		if !reflect.DeepEqual(a.ObjectiveValues, b.ObjectiveValues) {
			t.Fatalf("solution %d objectives differ: %v and %v", i, a.ObjectiveValues, b.ObjectiveValues)
		}

		if !reflect.DeepEqual(a.Position, b.Position) {
			t.Fatalf("solution %d position differs: %v and %v", i, a.Position, b.Position)
		}
	}
}

// TestOptimizeMultiObjectiveObserverCannotMutateTheArchive writes a dominating
// sentinel into every snapshot it is handed. A leak would dominate the whole
// front, so the invariant check afterwards fails loudly rather than subtly.
func TestOptimizeMultiObjectiveObserverCannotMutateTheArchive(t *testing.T) {
	const sentinel = -1e9

	result, err := OptimizeMultiObjective(context.Background(),
		newMultiObjectiveTestConfig(ZDT1, 3, 20, 11),
		WithArchiveObserver(func(snapshot ArchiveSnapshot) {
			for _, solution := range snapshot.Solutions {
				for m := range solution.ObjectiveValues {
					solution.ObjectiveValues[m] = sentinel
				}

				for j := range solution.Position {
					solution.Position[j] = sentinel
				}
			}

			for m := range snapshot.GridLower {
				snapshot.GridLower[m] = sentinel
				snapshot.GridUpper[m] = sentinel
			}
		}),
	)
	if err != nil {
		t.Fatalf("OptimizeMultiObjective: %v", err)
	}

	if !result.Archive.IsNonDominated() {
		t.Error("the archive lost its non-domination invariant to an observer")
	}

	for i, solution := range result.Archive.Solutions {
		for m, value := range solution.ObjectiveValues {
			if value == sentinel {
				t.Fatalf("solution %d objective %d was written by the observer", i, m)
			}
		}

		for j, value := range solution.Position {
			if value == sentinel {
				t.Fatalf("solution %d component %d was written by the observer", i, j)
			}
		}
	}
}

// TestOptimizeMultiObjectiveRejectsSingleObjectiveRunOptions covers the rule
// that an option with no multi-objective reading is refused rather than
// silently ignored -- the same rule validateMultiObjectiveConvergence applies
// to a target cost.
func TestOptimizeMultiObjectiveRejectsSingleObjectiveRunOptions(t *testing.T) {
	tests := []struct {
		option RunOption
		name   string
		names  string
	}{
		{
			name:   "progress_observer",
			option: WithProgressObserver(func(Progress) {}),
			names:  "WithProgressObserver",
		},
		{
			name:   "population_observer",
			option: WithPopulationObserver(func(PopulationSnapshot) {}),
			names:  "WithPopulationObserver",
		},
		{
			name:   "logger",
			option: WithLogger(slog.Default()),
			names:  "WithLogger",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result, err := OptimizeMultiObjective(context.Background(),
				newMultiObjectiveTestConfig(ZDT1, 3, 5, 3), test.option)
			if err == nil {
				t.Fatal("the option was accepted")
			}

			if result != nil {
				t.Error("a rejected run must not return a partial result")
			}

			if !strings.Contains(err.Error(), test.names) {
				t.Errorf("error does not name the option: %v", err)
			}
		})
	}

	// A nil observer registers nothing, so it has nothing to reject.
	t.Run("nil_observer_is_accepted", func(t *testing.T) {
		_, err := OptimizeMultiObjective(context.Background(),
			newMultiObjectiveTestConfig(ZDT1, 3, 5, 3), WithProgressObserver(nil))
		if err != nil {
			t.Errorf("WithProgressObserver(nil) was rejected: %v", err)
		}
	})
}

func TestOptimizeMultiObjectiveSeedsInitialPopulation(t *testing.T) {
	config := newMultiObjectiveTestConfig(ZDT1, 3, 5, 5)

	result, err := OptimizeMultiObjective(context.Background(), config,
		WithInitialPopulation([][]float64{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}}))
	if err != nil {
		t.Fatalf("a valid seeded population was rejected: %v", err)
	}

	if result.Archive.Len() == 0 {
		t.Error("a seeded run produced an empty archive")
	}

	t.Run("wrong_dimension", func(t *testing.T) {
		_, err := OptimizeMultiObjective(context.Background(),
			newMultiObjectiveTestConfig(ZDT1, 3, 5, 5),
			WithInitialPopulation([][]float64{{0.1, 0.2}}))
		if err == nil {
			t.Error("a position of the wrong dimension was accepted")
		}
	})

	t.Run("out_of_bounds", func(t *testing.T) {
		_, err := OptimizeMultiObjective(context.Background(),
			newMultiObjectiveTestConfig(ZDT1, 3, 5, 5),
			WithInitialPopulation([][]float64{{0.1, 0.2, 7}}))
		if err == nil {
			t.Error("a position outside the bounds was accepted")
		}
	})
}

// TestOptimizeMultiObjectiveObserverStopsWithTheRun covers both ways a run can
// end early. A canceled iteration reports nothing, and a stagnating one still
// reports the iteration that triggered the stop.
func TestOptimizeMultiObjectiveObserverStopsWithTheRun(t *testing.T) {
	t.Run("cancellation", func(t *testing.T) {
		config := newMultiObjectiveTestConfig(ZDT1, 3, 50, 13)
		ctx, cancel := context.WithCancel(context.Background())

		highest := 0

		_, err := OptimizeMultiObjective(ctx, config,
			WithArchiveObserver(func(snapshot ArchiveSnapshot) {
				highest = snapshot.Iteration

				if snapshot.Iteration == 3 {
					cancel()
				}
			}),
		)
		if err == nil {
			t.Fatal("a canceled run must return an error")
		}

		if highest >= config.Swarm.MaxIterations {
			t.Errorf("observer saw iteration %d of %d; the run was not cut short",
				highest, config.Swarm.MaxIterations)
		}
	})

	t.Run("stagnation", func(t *testing.T) {
		config := newMultiObjectiveTestConfig(ZDT1, 3, 60, 21)
		config.Swarm.Convergence = &ConvergenceConfig{
			StagnationIterations: 3,
			MinIterations:        1,
		}

		count := 0

		result, err := OptimizeMultiObjective(context.Background(), config,
			WithArchiveObserver(func(ArchiveSnapshot) { count++ }))
		if err != nil {
			t.Fatalf("OptimizeMultiObjective: %v", err)
		}

		if count != result.IterationCount {
			t.Errorf("observer called %d times, want %d -- the final, stagnant iteration counts too",
				count, result.IterationCount)
		}
	})
}

// TestParetoArchiveGridBoundsReturnsCopies covers the reason GridBounds copies:
// the archive rewrites its bounds through the same backing arrays on every
// mutation, so a shared slice would be a write path back into a running run.
func TestParetoArchiveGridBoundsReturnsCopies(t *testing.T) {
	var nilArchive *ParetoArchive

	lower, upper := nilArchive.GridBounds()
	if lower != nil || upper != nil {
		t.Errorf("a nil archive reported bounds %v / %v, want nil / nil", lower, upper)
	}

	archive := NewParetoArchiveWithGrid(8, 4, 4, 2, 2)

	lower, upper = archive.GridBounds()
	if lower != nil || upper != nil {
		t.Errorf("an empty archive reported bounds %v / %v, want nil / nil", lower, upper)
	}

	rng := rand.New(rand.NewSource(2))
	archive.Add(&ParetoSolution{Position: []float64{0}, ObjectiveValues: []float64{0.25, 0.75}}, rng)
	archive.Add(&ParetoSolution{Position: []float64{1}, ObjectiveValues: []float64{0.75, 0.25}}, rng)

	lower, upper = archive.GridBounds()

	if !reflect.DeepEqual(lower, []float64{0.25, 0.25}) {
		t.Errorf("lower bounds = %v, want [0.25 0.25]", lower)
	}

	if !reflect.DeepEqual(upper, []float64{0.75, 0.75}) {
		t.Errorf("upper bounds = %v, want [0.75 0.75]", upper)
	}

	lower[0] = 999
	upper[0] = 999

	if archive.lowerBounds[0] == 999 || archive.upperBounds[0] == 999 {
		t.Errorf("GridBounds aliased the archive: %v / %v", archive.lowerBounds, archive.upperBounds)
	}

	// The grid must still bin a new solution against the untouched bounds.
	archive.Add(&ParetoSolution{Position: []float64{2}, ObjectiveValues: []float64{0.5, 0.5}}, rng)

	for _, solution := range archive.Solutions {
		for m, index := range solution.GridIndex {
			if index < 0 || index >= archive.NGrid {
				t.Fatalf("grid index %d on objective %d is outside [0, %d)", index, m, archive.NGrid)
			}
		}
	}
}

func TestParetoArchiveRejectsMalformedPublicCandidates(t *testing.T) {
	archive := NewParetoArchive(10)
	if !archive.Add(&ParetoSolution{ObjectiveValues: []float64{1, 2}}, nil) {
		t.Fatal("valid initial solution was rejected")
	}

	tests := []struct {
		name      string
		candidate *ParetoSolution
	}{
		{name: "mixed arity", candidate: &ParetoSolution{ObjectiveValues: []float64{1}}},
		{name: "NaN objective", candidate: &ParetoSolution{ObjectiveValues: []float64{math.NaN(), 1}}},
		{name: "infinite objective", candidate: &ParetoSolution{ObjectiveValues: []float64{math.Inf(1), 1}}},
		{name: "negative violation", candidate: &ParetoSolution{ObjectiveValues: []float64{0, 3}, ConstraintViolation: -1}},
		{name: "NaN violation", candidate: &ParetoSolution{ObjectiveValues: []float64{0, 3}, ConstraintViolation: math.NaN()}},
		{name: "infinite violation", candidate: &ParetoSolution{ObjectiveValues: []float64{0, 3}, ConstraintViolation: math.Inf(1)}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if archive.Add(test.candidate, nil) {
				t.Fatal("malformed candidate was accepted")
			}

			if archive.Len() != 1 {
				t.Fatalf("rejected candidate mutated archive to length %d", archive.Len())
			}
		})
	}
}

func TestParetoArchiveAcceptanceReflectsDurableChange(t *testing.T) {
	archive := NewParetoArchive(2)
	first := &ParetoSolution{ObjectiveValues: []float64{0, 2}}

	second := &ParetoSolution{ObjectiveValues: []float64{2, 0}}
	if !archive.Add(first, nil) || !archive.Add(second, nil) {
		t.Fatal("setup insertion failed")
	}

	before := append([]*ParetoSolution(nil), archive.Solutions...)
	if archive.Add(&ParetoSolution{ObjectiveValues: []float64{3, -1}}, nil) {
		t.Fatal("Add reported acceptance after overflow immediately evicted the candidate")
	}

	if !sameSolutionSequence(before, archive.Solutions) {
		t.Fatal("candidate eviction did not restore the prior archive membership")
	}

	if !archive.Add(&ParetoSolution{ObjectiveValues: []float64{1, -1}}, nil) {
		t.Fatal("Add missed a durable change that removed a dominated member")
	}
}

func TestMATLABDensityRanksUseStrictSpanTwentieth(t *testing.T) {
	archive := newParetoArchive(10, 10, 1, 1, 1, ArchivePolicyMATLABDensity)
	for _, values := range [][]float64{{0, 20}, {0.5, 19.5}, {1, 19}, {20, 0}} {
		if !archive.Add(&ParetoSolution{ObjectiveValues: values}, nil) {
			t.Fatalf("failed to add %v", values)
		}
	}

	// Radius is exactly one in both objectives. The first two points are
	// neighbors; the point at distance exactly one is excluded by MATLAB's <.
	want := []int{2, 3, 2, 1}
	if got := archive.densityRanks(); !reflect.DeepEqual(got, want) {
		t.Errorf("density ranks = %v, want %v", got, want)
	}
}

func TestArchivePolicyDefaultsFollowFidelityMode(t *testing.T) {
	config := NewMultiObjectiveConfig()
	if got := effectiveArchivePolicy(config); got != ArchivePolicyPaperSegments {
		t.Errorf("default policy = %q, want %q", got, ArchivePolicyPaperSegments)
	}

	config.Swarm.FidelityMode = FidelityMATLAB
	if got := effectiveArchivePolicy(config); got != ArchivePolicyMATLABDensity {
		t.Errorf("MATLAB policy = %q, want %q", got, ArchivePolicyMATLABDensity)
	}

	config.ArchivePolicy = ArchivePolicyMOPSOGrid
	if got := effectiveArchivePolicy(config); got != ArchivePolicyMOPSOGrid {
		t.Errorf("explicit policy = %q, want %q", got, ArchivePolicyMOPSOGrid)
	}
}

func TestGridKeyDoesNotOverflowForManyObjectives(t *testing.T) {
	archive := NewParetoArchive(10)
	first := make([]float64, 128)

	second := make([]float64, 128)
	for i := range second {
		if i%2 == 0 {
			first[i] = 0
			second[i] = 1
		} else {
			first[i] = 1
			second[i] = 0
		}
	}

	if !archive.Add(&ParetoSolution{ObjectiveValues: first}, nil) ||
		!archive.Add(&ParetoSolution{ObjectiveValues: second}, nil) {
		t.Fatal("failed to build many-objective archive")
	}

	for _, solution := range archive.Solutions {
		if solution.GridKey >= 0 {
			t.Errorf("128-objective key %d did not take checked hash fallback", solution.GridKey)
		}
	}

	if cells := archive.occupiedCells(); len(cells) != 2 {
		t.Errorf("overflow-safe grid produced %d occupied cells, want 2", len(cells))
	}
}

func TestRouletteSelectionRemainsBiasedAtExtremeExponent(t *testing.T) {
	archive := newSelectionArchive(t)
	archive.Policy = ArchivePolicyMOPSOGrid
	archive.Gamma = 1e300

	rng := rand.New(rand.NewSource(91))
	for range 100 {
		if !isClustered(archive.selectCrowded(rng)) {
			t.Fatal("overflow-scale exponent fell back to uniform selection")
		}
	}
}

func TestScorePositionsCopiesReusableObjectiveBuffer(t *testing.T) {
	shared := make([]float64, 2)
	config := NewMultiObjectiveConfig()
	config.ObjectiveFunc = func(x []float64) []float64 {
		shared[0], shared[1] = x[0], 1-x[0]

		return shared
	}

	state := &moState{
		swarm:  []Dragonfly{{Position: []float64{0.25}}, {Position: []float64{0.75}}},
		scores: make([]moEvaluation, 2),
	}

	err := state.scorePositions(context.Background(), config)
	if err != nil {
		t.Fatalf("scorePositions: %v", err)
	}

	if !reflect.DeepEqual(state.scores[0].values, []float64{0.25, 0.75}) ||
		!reflect.DeepEqual(state.scores[1].values, []float64{0.75, 0.25}) {
		t.Fatalf("callback result slices aliased: %v / %v",
			state.scores[0].values, state.scores[1].values)
	}
}

func TestEvaluateSwarmValidatesFirstBatchBeforeArchiveMutation(t *testing.T) {
	calls := 0
	config := NewMultiObjectiveConfig()
	config.ObjectiveFunc = func([]float64) []float64 {
		calls++
		if calls == 1 {
			return []float64{1, 2}
		}

		return []float64{1}
	}

	state := &moState{
		archive: NewParetoArchive(10),
		swarm:   []Dragonfly{{Position: []float64{0}}, {Position: []float64{1}}},
		scores:  make([]moEvaluation, 2),
	}

	_, err := state.evaluateSwarm(context.Background(), config, nil)
	if err == nil {
		t.Fatal("mixed first-batch objective arity was accepted")
	}

	if state.archive.Len() != 0 {
		t.Fatalf("arity error partially mutated archive to length %d", state.archive.Len())
	}
}

func TestParetoExportRejectsNilArchiveMembers(t *testing.T) {
	dir := t.TempDir()

	result := &MultiObjectiveResult{}

	err := result.ExportParetoJSON(filepath.Join(dir, "nil-archive.json"))
	if !errors.Is(err, errNilParetoArchive) {
		t.Fatalf("nil archive error = %v, want %v", err, errNilParetoArchive)
	}

	result.Archive = NewParetoArchive(10)

	result.Archive.Solutions = append(result.Archive.Solutions, nil)

	err = result.ExportParetoCSV(filepath.Join(dir, "nil-member.csv"))
	if !errors.Is(err, errNilParetoSolution) {
		t.Fatalf("nil member error = %v, want %v", err, errNilParetoSolution)
	}
}

func TestOptimizeMultiObjectiveRejectsUnsupportedSharedSettings(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*MultiObjectiveConfig)
	}{
		{name: "binary swarm", mutate: func(config *MultiObjectiveConfig) { config.Swarm.UseBinary = true }},
		{name: "minimum improvement", mutate: func(config *MultiObjectiveConfig) {
			config.Swarm.Convergence = &ConvergenceConfig{MinImprovement: 0.1}
		}},
		{name: "unknown policy", mutate: func(config *MultiObjectiveConfig) {
			config.ArchivePolicy = "mystery"
		}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			config := newMultiObjectiveTestConfig(ZDT1, 3, 1, 1)
			test.mutate(config)

			_, err := OptimizeMultiObjective(context.Background(), config)
			if err == nil {
				t.Fatal("unsupported setting was accepted")
			}
		})
	}
}

func TestOptimizeMultiObjectiveRejectsNonFiniteObjectives(t *testing.T) {
	config := newMultiObjectiveTestConfig(func([]float64) []float64 {
		return []float64{math.NaN(), math.Inf(1)}
	}, 2, 1, 1)

	_, err := OptimizeMultiObjective(context.Background(), config)
	if !errors.Is(err, ErrNoFiniteObjective) {
		t.Fatalf("error = %v, want ErrNoFiniteObjective", err)
	}
}

func TestOptimizeMultiObjectiveReportsSeedKnowledge(t *testing.T) {
	seeded := newMultiObjectiveTestConfig(ZDT1, 2, 1, 44)

	result, err := OptimizeMultiObjective(context.Background(), seeded)
	if err != nil {
		t.Fatalf("seeded run: %v", err)
	}

	if !result.SeedKnown || result.Seed != 44 {
		t.Errorf("seed metadata = (%d, %v), want (44, true)", result.Seed, result.SeedKnown)
	}

	direct := newMultiObjectiveTestConfig(ZDT1, 2, 1, 44)
	direct.Swarm.Seed = nil
	direct.Swarm.Rand = rand.New(rand.NewSource(44))

	result, err = OptimizeMultiObjective(context.Background(), direct)
	if err != nil {
		t.Fatalf("direct-rand run: %v", err)
	}

	if result.SeedKnown || result.Seed != 0 {
		t.Errorf("direct-rand metadata = (%d, %v), want (0, false)", result.Seed, result.SeedKnown)
	}
}
