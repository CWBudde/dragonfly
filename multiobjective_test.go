package dragonfly

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
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
	config.Swarm.Rand = rand.New(rand.NewSource(seed))

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

	if len(result.ArchiveSizeCurve) != config.Swarm.MaxIterations {
		t.Errorf("archive size curve has %d entries, want %d",
			len(result.ArchiveSizeCurve), config.Swarm.MaxIterations)
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

// frontSpreadAndErrors summarizes an archive against an analytic front: the
// extent of the archive along the first objective, and the smallest and median
// vertical distance from an archived point to the front.
//
// The median rather than the maximum is the honest measure of convergence here.
// A Pareto archive keeps every non-dominated point it has seen, and a point with
// a very small f1 stays non-dominated however far it is from the front, so the
// worst archive member is a statement about what the archive remembers rather
// than about where the swarm converged.
func frontSpreadAndErrors(archive *ParetoArchive, front func(f1 float64) float64) (float64, float64, float64) {
	if archive.Len() == 0 {
		return 0, math.Inf(1), math.Inf(1)
	}

	minF1 := math.Inf(1)
	maxF1 := math.Inf(-1)
	errors := make([]float64, 0, archive.Len())

	for _, solution := range archive.Solutions {
		f1 := solution.ObjectiveValues[0]
		minF1 = math.Min(minF1, f1)
		maxF1 = math.Max(maxF1, f1)
		errors = append(errors, math.Abs(solution.ObjectiveValues[1]-front(f1)))
	}

	sort.Float64s(errors)

	return maxF1 - minF1, errors[0], errors[len(errors)/2]
}

// assertNearFront is the shared body of the two ZDT convergence checks.
func assertNearFront(
	t *testing.T, archive *ParetoArchive, front func(float64) float64,
	minSpread, maxClosest, maxMedian float64,
) {
	t.Helper()

	if archive.Len() < 10 {
		t.Fatalf("archive holds %d solutions, want at least 10", archive.Len())
	}

	spread, closest, median := frontSpreadAndErrors(archive, front)

	if spread < minSpread {
		t.Errorf("archive spans only %.3f of the first objective, want at least %.3f", spread, minSpread)
	}

	if closest > maxClosest {
		t.Errorf("closest archived point is %.3f from the front, want at most %.3f", closest, maxClosest)
	}

	if median > maxMedian {
		t.Errorf("median distance to the front is %.3f, want at most %.3f", median, maxMedian)
	}
}

// TestOptimizeMultiObjectiveZDT1 checks that a seeded run recovers a spread of
// solutions near ZDT1's analytic front f2 = 1 - sqrt(f1).
//
// It is gated by testing.Short: a run long enough to approach the front does not
// belong in `just test-quick`.
func TestOptimizeMultiObjectiveZDT1(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping the ZDT1 convergence run in short mode")
	}

	config := newMultiObjectiveTestConfig(ZDT1, 5, 400, 4242)
	config.Swarm.NPop = 60

	result, err := OptimizeMultiObjective(context.Background(), config)
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}

	assertNearFront(t, result.Archive, func(f1 float64) float64 {
		return 1 - math.Sqrt(f1)
	}, 0.5, 0.05, 0.3)
}

// TestOptimizeMultiObjectiveZDT3 is the same check against ZDT3, whose front is
// disconnected: f2 = 1 - sqrt(f1) - f1*sin(10*pi*f1). The tolerances are looser
// because the five disconnected pieces are harder to sit on than one continuous
// curve, not because the run is allowed to do less well.
func TestOptimizeMultiObjectiveZDT3(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping the ZDT3 convergence run in short mode")
	}

	config := newMultiObjectiveTestConfig(ZDT3, 5, 400, 4242)
	config.Swarm.NPop = 60

	result, err := OptimizeMultiObjective(context.Background(), config)
	if err != nil {
		t.Fatalf("run failed: %v", err)
	}

	assertNearFront(t, result.Archive, func(f1 float64) float64 {
		return 1 - math.Sqrt(f1) - f1*math.Sin(10*math.Pi*f1)
	}, 0.5, 0.2, 0.6)
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

	// index + two objectives + three decision variables.
	if len(records[0]) != 6 {
		t.Errorf("CSV header has %d columns, want 6: %v", len(records[0]), records[0])
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

	if document.Seed != result.Seed || document.IterationCount != result.IterationCount {
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
