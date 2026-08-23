package dragonfly

import (
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// runsFromCosts builds the RunResult slice the statistics operate on.
func runsFromCosts(costs ...float64) []RunResult {
	runs := make([]RunResult, len(costs))
	for i, cost := range costs {
		runs[i] = RunResult{BestCost: cost, Seed: int64(i)}
	}

	return runs
}

// TestWilcoxonSignedRankHandWorked checks the test against a sample worked out
// by hand rather than against whatever the implementation returns.
//
// The paired costs are chosen so the eight differences are exactly
// ±{1,2,3,4,5,6,7,8}, with only the 4 negative:
//
//	A: 11 12 13  6 15 16 17 18
//	B: 10 10 10 10 10 10 10 10
//	d: +1 +2 +3 -4 +5 +6 +7 +8
//
// The absolute differences are distinct, so their ranks are 1..8 with no ties.
//
//	W- = rank of the one negative difference = 4
//	W+ = 1+2+3+5+6+7+8                       = 32   (and W+ + W- = 36 = 8·9/2 ✓)
//	W  = min(W+, W-)                         = 4
//
// Enumerating the 2^8 possible sign assignments gives 14 assignments at
// least as far from E[W+]=18 as the observed W+=32, hence p=14/256=0.0546875.
// The exact test therefore correctly does not call this marginal sample
// significant; the old uncorrected normal approximation incorrectly did.
func TestWilcoxonSignedRankHandWorked(t *testing.T) {
	runsA := runsFromCosts(11, 12, 13, 6, 15, 16, 17, 18)
	runsB := runsFromCosts(10, 10, 10, 10, 10, 10, 10, 10)

	result := wilcoxonSignedRankTest("A", "B", runsA, runsB)

	if result.WStatistic != 4 {
		t.Errorf("W = %v, want 4", result.WStatistic)
	}

	wantP := 14.0 / 256.0
	if math.Abs(result.PValue-wantP) > 1e-12 {
		t.Errorf("p = %v, want exact %v", result.PValue, wantP)
	}

	if result.Significant {
		t.Errorf("Significant = true, want false at exact p = %v", result.PValue)
	}

	if result.Winner != wilcoxonTie {
		t.Errorf("Winner = %q, want %q", result.Winner, wilcoxonTie)
	}
}

// TestWilcoxonSignedRankIsAntisymmetric: swapping the arguments must flip every
// difference's sign, which swaps W+ and W- and therefore leaves W and p alone.
func TestWilcoxonSignedRankIsAntisymmetric(t *testing.T) {
	runsA := runsFromCosts(11, 12, 13, 6, 15, 16, 17, 18)
	runsB := runsFromCosts(10, 10, 10, 10, 10, 10, 10, 10)

	forward := wilcoxonSignedRankTest("A", "B", runsA, runsB)
	reverse := wilcoxonSignedRankTest("B", "A", runsB, runsA)

	if forward.WStatistic != reverse.WStatistic || forward.PValue != reverse.PValue {
		t.Errorf("swapping arguments changed the statistic: %+v vs %+v", forward, reverse)
	}

	if reverse.Winner != forward.Winner {
		t.Errorf("reversed Winner = %q, want %q", reverse.Winner, forward.Winner)
	}
}

// TestWilcoxonSignedRankTiesAndDegenerateInput covers the two shapes the test
// cannot compute a statistic for.
func TestWilcoxonSignedRankTiesAndDegenerateInput(t *testing.T) {
	identical := runsFromCosts(1, 2, 3, 4)

	tie := wilcoxonSignedRankTest("A", "B", identical, identical)
	if tie.Winner != wilcoxonTie {
		t.Errorf("identical samples: Winner = %q, want %q", tie.Winner, wilcoxonTie)
	}

	if tie.Significant || tie.PValue != 1 || !tie.Available {
		t.Errorf("identical samples reported %+v; every pair was a tie", tie)
	}

	unequal := wilcoxonSignedRankTest("A", "B", runsFromCosts(1, 2), runsFromCosts(1))
	if !strings.HasPrefix(unequal.Winner, "Error") {
		t.Errorf("unequal sample sizes: Winner = %q, want an error marker", unequal.Winner)
	}
}

// TestWilcoxonSignedRankFindsNoDifference: a sample whose differences alternate
// in sign in rank order splits the signed ranks nearly evenly, so W lands near
// its expected value and the test finds nothing.
func TestWilcoxonSignedRankFindsNoDifference(t *testing.T) {
	// d = -1, +2, -3, +4, -5, +6, -7, +8. Ranks 1..8 again.
	// W- = 1+3+5+7 = 16, W+ = 2+4+6+8 = 20, W = 16, E[W] = 18,
	// z = 2/sqrt(51) ≈ 0.28, p ≈ 0.78.
	runsA := runsFromCosts(9, 12, 7, 14, 5, 16, 3, 18)
	runsB := runsFromCosts(10, 10, 10, 10, 10, 10, 10, 10)

	result := wilcoxonSignedRankTest("A", "B", runsA, runsB)

	if result.WStatistic != 16 {
		t.Errorf("W = %v, want 16", result.WStatistic)
	}

	if result.Significant {
		t.Errorf("Significant = true at p = %v, want false", result.PValue)
	}

	if result.Winner != wilcoxonTie {
		t.Errorf("Winner = %q, want %q", result.Winner, wilcoxonTie)
	}
}

// TestFriedmanHandWorked checks the Friedman statistic against a case computed
// by hand.
//
// Three variants over four runs, with A always cheapest and C always dearest,
// so every run ranks them 1, 2, 3:
//
//	run 1: A=1 B=2 C=3   ranks 1 2 3
//	run 2: A=1 B=2 C=3   ranks 1 2 3
//	run 3: A=1 B=2 C=3   ranks 1 2 3
//	run 4: A=1 B=2 C=3   ranks 1 2 3
//
//	rank sums R = (4, 8, 12);  Σ R² = 16 + 64 + 144 = 224
//
// With k = 3 variants and n = 4 runs:
//
//	chi² = 12/(n·k·(k+1))·ΣR² − 3·n·(k+1)
//	     = 12/(4·3·4)·224 − 3·4·4
//	     = 0.25·224 − 48 = 56 − 48 = 8
//
// df = k − 1 = 2. For two degrees of freedom the chi-square survival function
// is exactly exp(−x/2), so
//
//	p = exp(−4) = 0.018315638888734...
//
// which is below alpha = 0.05.
func TestFriedmanHandWorked(t *testing.T) {
	runResults := [][]RunResult{
		runsFromCosts(1, 1, 1, 1),
		runsFromCosts(2, 2, 2, 2),
		runsFromCosts(3, 3, 3, 3),
	}

	result := friedmanTest(runResults)
	if result == nil {
		t.Fatal("friedmanTest returned nil for three variants")
	}

	if math.Abs(result.ChiSquare-8) > 1e-12 {
		t.Errorf("chi-square = %v, want 8", result.ChiSquare)
	}

	if result.DegreesOfFreedom != 2 {
		t.Errorf("df = %d, want 2", result.DegreesOfFreedom)
	}

	wantP := math.Exp(-4)
	if math.Abs(result.PValue-wantP) > 1e-9 {
		t.Errorf("p = %v, want exp(-4) = %v", result.PValue, wantP)
	}

	if !result.Significant {
		t.Errorf("Significant = false at p = %v", result.PValue)
	}
}

// TestFriedmanFindsNoDifference: when every variant scores identically on every
// run, all three share the average rank 2, the rank sums are (8, 8, 8), and
//
//	chi² = 0.25·(64·3) − 48 = 48 − 48 = 0,  p = 1.
func TestFriedmanFindsNoDifference(t *testing.T) {
	runResults := [][]RunResult{
		runsFromCosts(5, 5, 5, 5),
		runsFromCosts(5, 5, 5, 5),
		runsFromCosts(5, 5, 5, 5),
	}

	result := friedmanTest(runResults)
	if result == nil {
		t.Fatal("friedmanTest returned nil")
	}

	if math.Abs(result.ChiSquare) > 1e-12 {
		t.Errorf("chi-square = %v, want 0", result.ChiSquare)
	}

	if result.PValue != 1 {
		t.Errorf("p = %v, want 1", result.PValue)
	}

	if result.Significant {
		t.Error("Significant = true for identical samples")
	}
}

func TestFriedmanNeedsTwoVariants(t *testing.T) {
	if friedmanTest([][]RunResult{runsFromCosts(1, 2, 3)}) != nil {
		t.Error("friedmanTest returned a result for a single variant")
	}

	if friedmanTest(nil) != nil {
		t.Error("friedmanTest returned a result for no variants")
	}

	if friedmanTest([][]RunResult{{}, {}}) != nil {
		t.Error("friedmanTest returned a result for zero runs")
	}
}

// TestChiSquareSurvivalKnownValues checks the tail against values that can be
// written down in closed form or looked up in any table.
func TestChiSquareSurvivalKnownValues(t *testing.T) {
	tests := []struct {
		x         float64
		df        int
		want      float64
		tolerance float64
	}{
		// df = 2 has the closed form exp(-x/2).
		{0.5, 2, math.Exp(-0.25), 1e-12},
		{4, 2, math.Exp(-2), 1e-12},
		{9.21034037, 2, 0.01, 1e-8},
		// df = 1: P(X > x) = erfc(sqrt(x/2)).
		{1, 1, math.Erfc(math.Sqrt(0.5)), 1e-12},
		{3.8414588, 1, 0.05, 1e-7},
		// Standard 0.05 critical values from a chi-square table.
		{7.8147279, 3, 0.05, 1e-7},
		{18.3070381, 10, 0.05, 1e-7},
		// The tail is 1 at and below zero.
		{0, 4, 1, 0},
		{-3, 4, 1, 0},
	}

	for _, test := range tests {
		got := chiSquareSurvival(test.x, test.df)
		if math.Abs(got-test.want) > test.tolerance {
			t.Errorf("chiSquareSurvival(%v, %d) = %v, want %v", test.x, test.df, got, test.want)
		}
	}

	if !math.IsNaN(chiSquareSurvival(1, 0)) {
		t.Error("chiSquareSurvival with zero degrees of freedom must be NaN")
	}
}

func TestNormalCDFKnownValues(t *testing.T) {
	tests := []struct {
		x    float64
		want float64
	}{
		{0, 0.5},
		{1, 0.8413447460685429},
		{-1, 0.15865525393145707},
		{1.959963985, 0.975},
	}

	for _, test := range tests {
		got := normalCDF(test.x)
		if math.Abs(got-test.want) > 1e-9 {
			t.Errorf("normalCDF(%v) = %v, want %v", test.x, got, test.want)
		}
	}
}

// TestRankValuesAveragesTies is the textbook mid-rank convention: the two tied
// smallest values share ranks 1 and 2 as 1.5, and the three tied at the top
// share 4, 5 and 6 as 5.
func TestRankValuesAveragesTies(t *testing.T) {
	ranks := rankValues([]float64{7, 2, 7, 3, 2, 7})
	want := []float64{5, 1.5, 5, 3, 1.5, 5}

	for i, rank := range ranks {
		if rank != want[i] {
			t.Errorf("rankValues()[%d] = %v, want %v", i, rank, want[i])
		}
	}

	// The ranks of n values always sum to n(n+1)/2 whatever the ties.
	total := 0.0
	for _, rank := range ranks {
		total += rank
	}

	if total != 21 {
		t.Errorf("ranks sum to %v, want 21", total)
	}

	if len(rankValues(nil)) != 0 {
		t.Error("rankValues(nil) is not empty")
	}
}

func TestCalculateAlgorithmStatisticsHandWorked(t *testing.T) {
	// Costs 2, 4, 4, 4, 5, 5, 7, 9: mean 5, population sigma 2, best 2,
	// worst 9, and an even count so the median is (4+5)/2 = 4.5.
	runs := runsFromCosts(2, 4, 4, 4, 5, 5, 7, 9)
	for i := range runs {
		runs[i].FuncEvals = 100
		runs[i].ExecutionTime = 0.5
	}

	stats := calculateAlgorithmStatistics(runs, 4)

	checks := map[string][2]float64{
		"mean":         {stats.Mean, 5},
		"stddev":       {stats.StdDev, 2},
		"median":       {stats.Median, 4.5},
		"best":         {stats.Best, 2},
		"worst":        {stats.Worst, 9},
		"avgFuncEvals": {stats.AvgFuncEvals, 100},
		"avgTime":      {stats.AvgTime, 0.5},
		// Four of eight runs came in at or below the target of 4.
		"successRate": {stats.SuccessRate, 50},
	}

	for name, pair := range checks {
		if math.Abs(pair[0]-pair[1]) > 1e-12 {
			t.Errorf("%s = %v, want %v", name, pair[0], pair[1])
		}
	}

	if stats.SuccessfulRuns != len(runs) || stats.FailedRuns != 0 {
		t.Errorf("run counts = %d successful/%d failed, want %d/0",
			stats.SuccessfulRuns, stats.FailedRuns, len(runs))
	}

	mixed := append(runsFromCosts(1, 3), RunResult{BestCost: math.Inf(1), Error: "boom"})

	mixedStats := calculateAlgorithmStatistics(mixed, 0, true)
	if mixedStats.Mean != 2 || mixedStats.SuccessfulRuns != 2 || mixedStats.FailedRuns != 1 {
		t.Errorf("mixed statistics = %+v, want mean 2 over two successes and one failure", mixedStats)
	}

	// An odd count takes the middle element rather than an average.
	if got := calculateAlgorithmStatistics(runsFromCosts(1, 5, 9), 0).Median; got != 5 {
		t.Errorf("odd-count median = %v, want 5", got)
	}

	if (calculateAlgorithmStatistics(nil, 0) != AlgorithmStatistics{}) {
		t.Error("statistics over no runs must be the zero value")
	}
}

func TestRankAlgorithmsByMean(t *testing.T) {
	statistics := []AlgorithmStatistics{{Mean: 3}, {Mean: 1}, {Mean: 2}}

	rankings := rankAlgorithms(statistics)
	want := []int{3, 1, 2}

	for i, rank := range rankings {
		if rank != want[i] {
			t.Errorf("rankAlgorithms()[%d] = %d, want %d", i, rank, want[i])
		}
	}

	// Ties keep the input order, so the ranking is reproducible.
	tied := rankAlgorithms([]AlgorithmStatistics{{Mean: 1}, {Mean: 1}, {Mean: 0}})
	if tied[0] != 2 || tied[1] != 3 || tied[2] != 1 {
		t.Errorf("tied rankings = %v, want [2 3 1]", tied)
	}
}

// comparisonRunnerForTest is a small, fast runner: the point of these tests is
// the seeding and the plumbing, not convergence quality.
func comparisonRunnerForTest() *ComparisonRunner {
	return NewComparisonRunner().
		WithVariantNames("da", "bda").
		WithRuns(4).
		WithIterations(12).
		WithSeed(4242)
}

// TestComparisonSeedsArePaired is the gate: run k of every variant must get the
// same seed, and the seeds must not depend on how the runs were scheduled.
func TestComparisonSeedsArePaired(t *testing.T) {
	runner := comparisonRunnerForTest()

	result := runner.Compare("Sphere", Sphere, 5, -5, 5)
	if result == nil {
		t.Fatal("Compare returned nil")
	}

	for run := range 4 {
		want := int64(4242 + run)
		for variant, name := range result.AlgorithmNames {
			got := result.RunResults[variant][run].Seed
			if got != want {
				t.Errorf("%s run %d used seed %d, want the paired seed %d", name, run, got, want)
			}
		}
	}

	if result.BaseSeed != 4242 {
		t.Errorf("BaseSeed = %d, want 4242", result.BaseSeed)
	}
}

// TestComparisonParallelMatchesSequential enforces that concurrency changes only
// the order runs finish in, never what they compute.
func TestComparisonParallelMatchesSequential(t *testing.T) {
	sequential := comparisonRunnerForTest().WithParallel(false).Compare("Sphere", Sphere, 5, -5, 5)
	parallel := comparisonRunnerForTest().WithParallel(true).WithMaxWorkers(4).
		Compare("Sphere", Sphere, 5, -5, 5)

	if len(sequential.AlgorithmNames) != len(parallel.AlgorithmNames) {
		t.Fatalf("variant counts differ: %v vs %v", sequential.AlgorithmNames, parallel.AlgorithmNames)
	}

	for variant, name := range sequential.AlgorithmNames {
		if parallel.AlgorithmNames[variant] != name {
			t.Fatalf("variant %d is %q sequentially and %q in parallel",
				variant, name, parallel.AlgorithmNames[variant])
		}

		for run := range sequential.RunResults[variant] {
			left := sequential.RunResults[variant][run]
			right := parallel.RunResults[variant][run]

			if left.Seed != right.Seed {
				t.Errorf("%s run %d: seed %d sequentially, %d in parallel", name, run, left.Seed, right.Seed)
			}

			if left.BestCost != right.BestCost {
				t.Errorf("%s run %d: cost %v sequentially, %v in parallel",
					name, run, left.BestCost, right.BestCost)
			}

			if left.FuncEvals != right.FuncEvals || left.Iterations != right.Iterations {
				t.Errorf("%s run %d: %+v sequentially, %+v in parallel", name, run, left, right)
			}
		}
	}

	if sequential.FriedmanResult.ChiSquare != parallel.FriedmanResult.ChiSquare {
		t.Errorf("Friedman chi-square differs: %v vs %v",
			sequential.FriedmanResult.ChiSquare, parallel.FriedmanResult.ChiSquare)
	}
}

// TestComparisonRepeatsForTheSameSeed: the same base seed must reproduce the
// whole comparison.
func TestComparisonRepeatsForTheSameSeed(t *testing.T) {
	first := comparisonRunnerForTest().Compare("Sphere", Sphere, 5, -5, 5)
	second := comparisonRunnerForTest().Compare("Sphere", Sphere, 5, -5, 5)

	for variant := range first.RunResults {
		for run := range first.RunResults[variant] {
			if first.RunResults[variant][run].BestCost != second.RunResults[variant][run].BestCost {
				t.Fatalf("variant %d run %d is not reproducible: %v vs %v", variant, run,
					first.RunResults[variant][run].BestCost, second.RunResults[variant][run].BestCost)
			}
		}
	}

	// A different base seed must actually change something, or the pairing is
	// not doing anything.
	other := comparisonRunnerForTest().WithSeed(999).Compare("Sphere", Sphere, 5, -5, 5)
	if other.RunResults[0][0].BestCost == first.RunResults[0][0].BestCost {
		t.Error("changing the base seed did not change the first run")
	}
}

// TestComparisonKeepsBinaryBounds: BDA's unit box must survive the runner's
// problem setup, or its step clamp and transfer function stop matching.
func TestComparisonKeepsBinaryBounds(t *testing.T) {
	runner := comparisonRunnerForTest()

	_, _, jobs, err := runner.prepareJobs(Sphere, 5, -100, 100)
	if err != nil {
		t.Fatalf("prepareJobs: %v", err)
	}

	for _, job := range jobs {
		if job.config.UseBinary {
			if job.config.LowerBound != 0 || job.config.UpperBound != 1 {
				t.Errorf("BDA job bounds = [%v, %v], want [0, 1]",
					job.config.LowerBound, job.config.UpperBound)
			}
		} else if job.config.LowerBound != -100 || job.config.UpperBound != 100 {
			t.Errorf("DA job bounds = [%v, %v], want [-100, 100]",
				job.config.LowerBound, job.config.UpperBound)
		}
	}
}

func TestComparisonRejectsMultiObjectiveVariant(t *testing.T) {
	runner := NewComparisonRunner().WithVariantNames("da", "moda").WithRuns(2).WithIterations(5)

	_, err := runner.CompareContext(context.Background(), "Sphere", Sphere, 3, -1, 1)
	if !errors.Is(err, ErrMultiObjectiveVariant) {
		t.Fatalf("CompareContext with MODA = %v, want ErrMultiObjectiveVariant", err)
	}
}

func TestComparisonValidation(t *testing.T) {
	tests := []struct {
		name   string
		runner *ComparisonRunner
		fn     ObjectiveFunction
		size   int
		lower  float64
		upper  float64
	}{
		{"no variants", NewComparisonRunner().WithVariants(), Sphere, 3, -1, 1},
		{"unknown variant name", NewComparisonRunner().WithVariantNames("nope"), Sphere, 3, -1, 1},
		{"zero runs", comparisonRunnerForTest().WithRuns(0), Sphere, 3, -1, 1},
		{"zero iterations", comparisonRunnerForTest().WithIterations(0), Sphere, 3, -1, 1},
		{"negative workers", comparisonRunnerForTest().WithMaxWorkers(-1), Sphere, 3, -1, 1},
		{"nil objective", comparisonRunnerForTest(), nil, 3, -1, 1},
		{"zero problem size", comparisonRunnerForTest(), Sphere, 0, -1, 1},
		{"inverted bounds", comparisonRunnerForTest(), Sphere, 3, 1, -1},
		{"infinite bounds", comparisonRunnerForTest(), Sphere, 3, math.Inf(-1), 1},
	}

	for _, test := range tests {
		_, err := test.runner.CompareContext(context.Background(), "Sphere", test.fn, test.size, test.lower, test.upper)
		if err == nil {
			t.Errorf("%s: CompareContext returned no error", test.name)
		}
	}

	legacy := comparisonRunnerForTest().WithRuns(0).Compare("Sphere", Sphere, 3, -1, 1)
	if legacy == nil || legacy.Error == "" {
		t.Error("Compare hid its validation error")
	}

	//nolint:staticcheck // passing a nil context is exactly what is under test.
	_, err := comparisonRunnerForTest().CompareContext(nil, "Sphere", Sphere, 3, -1, 1)
	if err == nil {
		t.Error("a nil context returned no error")
	}
}

func TestComparisonCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := comparisonRunnerForTest().CompareContext(ctx, "Sphere", Sphere, 5, -5, 5)
	if !errors.Is(err, context.Canceled) {
		t.Errorf("CompareContext on a canceled context = %v, want context.Canceled", err)
	}
}

func TestComparisonExportsRoundTrip(t *testing.T) {
	result := comparisonRunnerForTest().Compare("Sphere", Sphere, 5, -5, 5)
	dir := t.TempDir()

	csvPath := filepath.Join(dir, "comparison.csv")

	err := result.ExportToCSV(csvPath)
	if err != nil {
		t.Fatalf("ExportToCSV: %v", err)
	}

	file, err := os.Open(csvPath)
	if err != nil {
		t.Fatalf("open exported CSV: %v", err)
	}

	records, err := csv.NewReader(file).ReadAll()

	closeErr := file.Close()
	if err != nil || closeErr != nil {
		t.Fatalf("read exported CSV: %v / %v", err, closeErr)
	}

	wantRows := 1 + len(result.AlgorithmNames)*len(result.RunResults[0])
	if len(records) != wantRows {
		t.Fatalf("CSV has %d rows, want %d", len(records), wantRows)
	}

	if len(records[0]) != len(comparisonCSVHeader()) {
		t.Errorf("CSV header has %d columns, want %d", len(records[0]), len(comparisonCSVHeader()))
	}

	if records[1][0] != "Sphere" || records[1][1] != result.AlgorithmNames[0] {
		t.Errorf("first CSV row = %v, want it to start with the benchmark and the first variant", records[1])
	}

	jsonPath := filepath.Join(dir, "comparison.json")

	err = result.ExportToJSON(jsonPath)
	if err != nil {
		t.Fatalf("ExportToJSON: %v", err)
	}

	data, err := os.ReadFile(jsonPath)
	if err != nil {
		t.Fatalf("read exported JSON: %v", err)
	}

	var decoded ComparisonResult

	err = json.Unmarshal(data, &decoded)
	if err != nil {
		t.Fatalf("decode exported JSON: %v", err)
	}

	if decoded.BenchmarkName != result.BenchmarkName || decoded.BaseSeed != result.BaseSeed {
		t.Errorf("JSON round trip lost the header: %+v", decoded)
	}

	if len(decoded.RunResults) != len(result.RunResults) ||
		decoded.RunResults[0][0].BestCost != result.RunResults[0][0].BestCost {
		t.Error("JSON round trip lost the run results")
	}

	if decoded.FriedmanResult == nil ||
		decoded.FriedmanResult.ChiSquare != result.FriedmanResult.ChiSquare {
		t.Error("JSON round trip lost the Friedman result")
	}
}

func TestComparisonExportsRejectMalformedResults(t *testing.T) {
	dir := t.TempDir()

	var nilResult *ComparisonResult

	if nilResult.ExportToCSV(filepath.Join(dir, "a.csv")) == nil {
		t.Error("ExportToCSV on a nil result returned no error")
	}

	if nilResult.ExportToJSON(filepath.Join(dir, "a.json")) == nil {
		t.Error("ExportToJSON on a nil result returned no error")
	}

	if nilResult.WriteComparisonResults(&bytes.Buffer{}) == nil {
		t.Error("WriteComparisonResults on a nil result returned no error")
	}

	inconsistent := &ComparisonResult{AlgorithmNames: []string{"DA", "BDA"}}
	if inconsistent.ExportToCSV(filepath.Join(dir, "b.csv")) == nil {
		t.Error("ExportToCSV on an inconsistent result returned no error")
	}

	if (&ComparisonResult{}).ExportToJSON(filepath.Join(dir, "c.json")) == nil {
		t.Error("ExportToJSON on an empty result returned no error")
	}
}

func TestWriteComparisonResults(t *testing.T) {
	result := comparisonRunnerForTest().Compare("Sphere", Sphere, 5, -5, 5)

	var report bytes.Buffer

	err := result.WriteComparisonResults(&report)
	if err != nil {
		t.Fatalf("WriteComparisonResults: %v", err)
	}

	text := report.String()
	for _, want := range []string{"Sphere", "DA", "BDA", "Friedman", "Wilcoxon", "base seed 4242"} {
		if !strings.Contains(text, want) {
			t.Errorf("the report does not mention %q:\n%s", want, text)
		}
	}

	if result.WriteComparisonResults(nil) == nil {
		t.Error("WriteComparisonResults(nil) returned no error")
	}
}

func TestWorkerCount(t *testing.T) {
	runner := NewComparisonRunner()

	runner.Parallel = false
	if got := runner.workerCount(100); got != 1 {
		t.Errorf("sequential workerCount = %d, want 1", got)
	}

	runner.Parallel = true
	runner.MaxWorkers = 8

	if got := runner.workerCount(3); got != 3 {
		t.Errorf("workerCount capped by job count = %d, want 3", got)
	}

	if got := runner.workerCount(100); got != 8 {
		t.Errorf("workerCount = %d, want 8", got)
	}

	if got := runner.workerCount(0); got != 1 {
		t.Errorf("workerCount with no jobs = %d, want at least 1", got)
	}
}

func TestNewComparisonRunnerDefaults(t *testing.T) {
	runner := NewComparisonRunner()

	if len(runner.Variants) != 2 {
		t.Errorf("default variants = %d, want the two single-objective ones", len(runner.Variants))
	}

	for _, variant := range runner.Variants {
		if variant.IsMultiObjective() {
			t.Errorf("default variants include the multi-objective %s", variant.Name())
		}
	}

	if runner.Runs != 30 || runner.MaxIterations != 500 {
		t.Errorf("default budget = %d runs of %d iterations, want 30 of 500",
			runner.Runs, runner.MaxIterations)
	}
}

func TestQualityBarHandlesNonFiniteMeans(t *testing.T) {
	result := &ComparisonResult{Statistics: []AlgorithmStatistics{{Mean: 1}, {Mean: 3}}}

	bar, label := result.qualityBar(math.Inf(1), 10)
	if strings.Contains(bar, "#") || label != "failed/unavailable" {
		t.Errorf("an infinite mean rendered as %q / %q", bar, label)
	}

	// The best mean fills the bar and the worst empties it.
	bar, _ = result.qualityBar(1, 10)
	if bar != strings.Repeat("#", 10) {
		t.Errorf("the best mean rendered as %q", bar)
	}

	bar, _ = result.qualityBar(3, 10)
	if strings.Contains(bar, "#") {
		t.Errorf("the worst mean rendered as %q", bar)
	}
}

// captureStdout redirects os.Stdout for the duration of fn and returns
// everything written to it. The Print* helpers have no io.Writer form, so this
// is the only way to assert what they produce.
func captureStdout(t *testing.T, fn func()) string {
	t.Helper()

	reader, writer, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe: %v", err)
	}

	original := os.Stdout
	os.Stdout = writer

	defer func() { os.Stdout = original }()

	captured := make(chan string, 1)

	go func() {
		var buffer bytes.Buffer

		_, _ = buffer.ReadFrom(reader)
		captured <- buffer.String()
	}()

	fn()

	_ = writer.Close()
	text := <-captured
	_ = reader.Close()

	return text
}

// TestComparisonRunnerOptionSettersAssignTheirField pins each fluent setter to
// the field it claims, and to returning the same runner so the chain works.
func TestComparisonRunnerOptionSettersAssignTheirField(t *testing.T) {
	runner := NewComparisonRunner()

	if got := runner.WithTarget(1e-8); got != runner {
		t.Errorf("WithTarget returned %p, want the receiver %p", got, runner)
	}

	if runner.TargetCost != 1e-8 {
		t.Errorf("WithTarget(1e-8) left TargetCost = %v, want 1e-8", runner.TargetCost)
	}

	// Zero is a legitimate threshold and differs from never calling WithTarget.
	runner.WithTarget(0)

	if runner.TargetCost != 0 {
		t.Errorf("WithTarget(0) left TargetCost = %v, want 0", runner.TargetCost)
	}

	if !runner.targetEnabled() {
		t.Error("WithTarget(0) did not enable the zero threshold")
	}

	runner.WithTarget(-1)

	if !runner.targetEnabled() || runner.TargetCost != -1 {
		t.Error("WithTarget(-1) did not enable a negative threshold")
	}

	if got := runner.WithVerbose(true); got != runner {
		t.Errorf("WithVerbose returned %p, want the receiver %p", got, runner)
	}

	if !runner.Verbose {
		t.Error("WithVerbose(true) left Verbose false")
	}

	runner.WithVerbose(false)

	if runner.Verbose {
		t.Error("WithVerbose(false) left Verbose true")
	}
}

// TestPrintComparisonResults is a smoke test for the stdout wrapper around
// WriteComparisonResults: it must delegate, not swallow the report.
func TestPrintComparisonResults(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping a full comparison run in short mode")
	}

	result := comparisonRunnerForTest().Compare("Sphere", Sphere, 5, -5, 5)

	text := captureStdout(t, result.PrintComparisonResults)
	if text == "" {
		t.Fatal("PrintComparisonResults wrote nothing to stdout")
	}

	for _, want := range []string{"Sphere", "DA", "BDA"} {
		if !strings.Contains(text, want) {
			t.Errorf("the printed report does not mention %q:\n%s", want, text)
		}
	}
}

// TestRegularizedGammaQEdgesAndKnownValues covers the two guards
// chiSquareSurvival can never reach, and checks the function against the closed
// form Q(1, x) = e^-x and against chi-square table values.
func TestRegularizedGammaQEdgesAndKnownValues(t *testing.T) {
	// Guards: a non-positive shape or a negative argument is undefined.
	for _, guard := range []struct{ a, x float64 }{{0, 1}, {-1, 1}, {1, -1}, {-1, -1}} {
		if got := regularizedGammaQ(guard.a, guard.x); !math.IsNaN(got) {
			t.Errorf("regularizedGammaQ(%v, %v) = %v, want NaN", guard.a, guard.x, got)
		}
	}

	if got := regularizedGammaQ(2.5, 0); got != 1 {
		t.Errorf("regularizedGammaQ(2.5, 0) = %v, want 1", got)
	}

	tests := []struct {
		name      string
		a         float64
		x         float64
		want      float64
		tolerance float64
	}{
		// Q(1, x) = exp(-x) exactly. x < a+1 = 2 takes the series branch,
		// x >= 2 the continued fraction, so both are exercised.
		{"series branch", 1, 0.5, math.Exp(-0.5), 1e-12},
		{"boundary", 1, 2, math.Exp(-2), 1e-12},
		{"continued fraction branch", 1, 7, math.Exp(-7), 1e-12},
		// Q(a, x) with a = df/2 and x = c/2 is the chi-square upper tail;
		// 11.0704977 is the 0.05 critical value for 5 degrees of freedom.
		{"chi-square df=5 at 0.05", 2.5, 11.0704977 / 2, 0.05, 1e-7},
		// Q(1/2, x) = erfc(sqrt(x)).
		{"half shape", 0.5, 0.25, math.Erfc(0.5), 1e-12},
	}

	for _, test := range tests {
		got := regularizedGammaQ(test.a, test.x)
		if math.Abs(got-test.want) > test.tolerance {
			t.Errorf("%s: regularizedGammaQ(%v, %v) = %v, want %v",
				test.name, test.a, test.x, got, test.want)
		}
	}
}

// TestCollectReportsFirstErrorAndCancels drives collect directly: it must keep
// the first failure, cancel the remaining work unless told to continue, and
// still store every result it was handed.
func TestCollectReportsFirstErrorAndCancels(t *testing.T) {
	newResults := func() chan comparisonJobResult {
		results := make(chan comparisonJobResult, 3)
		results <- comparisonJobResult{run: RunResult{BestCost: 1.5}, variantIndex: 0, runIndex: 0}

		results <- comparisonJobResult{
			err:          errors.New("first failure"),
			run:          RunResult{BestCost: math.Inf(1)},
			variantIndex: 1,
			runIndex:     0,
		}

		results <- comparisonJobResult{err: errors.New("later failure"), variantIndex: 1, runIndex: 1}

		close(results)

		return results
	}

	names := []string{"DA", "BDA"}

	t.Run("stops on the first error", func(t *testing.T) {
		runner := NewComparisonRunner()
		runner.Verbose = true

		runResults := [][]RunResult{make([]RunResult, 2), make([]RunResult, 2)}
		canceled := 0

		var err error

		text := captureStdout(t, func() {
			err = runner.collect(newResults(), names, runResults, 3, false, func() { canceled++ })
		})

		if err == nil {
			t.Fatal("collect returned no error")
		}

		if !strings.Contains(err.Error(), "compare BDA run 1") ||
			!strings.Contains(err.Error(), "first failure") {
			t.Errorf("collect error = %q, want it to name BDA run 1 and the first failure", err)
		}

		if strings.Contains(err.Error(), "later failure") {
			t.Errorf("collect error = %q, want only the first failure kept", err)
		}

		if canceled != 1 {
			t.Errorf("cancel called %d times, want 1", canceled)
		}

		if runResults[0][0].BestCost != 1.5 {
			t.Errorf("runResults[0][0].BestCost = %v, want 1.5", runResults[0][0].BestCost)
		}

		if !strings.Contains(text, "Completed 3/3 comparison runs") {
			t.Errorf("verbose output = %q, want it to report 3/3 completed", text)
		}
	})

	t.Run("continues on error", func(t *testing.T) {
		runner := NewComparisonRunner()

		runResults := [][]RunResult{make([]RunResult, 2), make([]RunResult, 2)}
		canceled := 0

		err := runner.collect(newResults(), names, runResults, 3, true, func() { canceled++ })
		if err == nil {
			t.Fatal("collect returned no error")
		}

		if canceled != 0 {
			t.Errorf("cancel called %d times with continueOnError, want 0", canceled)
		}
	})
}

// TestExecuteJobFailureAndTargetPaths covers the two branches a successful
// default comparison never takes: a variant whose Run fails, and a target cost
// that the convergence curve actually reaches.
func TestExecuteJobFailureAndTargetPaths(t *testing.T) {
	jobConfig := func() *Config {
		config := NewDefaultConfig()
		config.ObjectiveFunc = Sphere
		config.ProblemSize = 3
		config.LowerBound = -5
		config.UpperBound = 5
		config.NPop = 8
		config.MaxIterations = 6
		config.Rand = rand.New(rand.NewSource(42))

		return config
	}

	t.Run("variant failure", func(t *testing.T) {
		moda, err := NewVariant("moda")
		if err != nil {
			t.Fatalf("NewVariant(moda): %v", err)
		}

		runner := NewComparisonRunner()
		job := comparisonJob{
			config: jobConfig(), variant: moda, variantIndex: 1, runIndex: 2, seed: 77,
		}

		got := runner.executeJob(context.Background(), job)
		if !errors.Is(got.err, ErrMultiObjectiveVariant) {
			t.Errorf("executeJob err = %v, want ErrMultiObjectiveVariant", got.err)
		}

		if !math.IsInf(got.run.BestCost, 1) {
			t.Errorf("failed run BestCost = %v, want +Inf", got.run.BestCost)
		}

		if got.run.Error == "" {
			t.Error("failed run carries no Error string")
		}

		if got.variantIndex != 1 || got.runIndex != 2 || got.run.Seed != 77 {
			t.Errorf("executeJob returned indices (%d, %d) seed %d, want (1, 2) seed 77",
				got.variantIndex, got.runIndex, got.run.Seed)
		}
	})

	t.Run("target reached on the first iteration", func(t *testing.T) {
		da, err := NewVariant("da")
		if err != nil {
			t.Fatalf("NewVariant(da): %v", err)
		}

		runner := NewComparisonRunner().WithTarget(math.MaxFloat64)
		job := comparisonJob{config: jobConfig(), variant: da, variantIndex: 0, runIndex: 0, seed: 3}

		got := runner.executeJob(context.Background(), job)
		if got.err != nil {
			t.Fatalf("executeJob: %v", got.err)
		}

		if got.run.ConvergenceAt != 1 {
			t.Errorf("ConvergenceAt = %d, want 1 for an unreachably loose target", got.run.ConvergenceAt)
		}

		if got.run.Iterations == 0 || got.run.FuncEvals == 0 {
			t.Errorf("executeJob recorded Iterations = %d, FuncEvals = %d, want both positive",
				got.run.Iterations, got.run.FuncEvals)
		}
	})

	t.Run("target never reached", func(t *testing.T) {
		da, err := NewVariant("da")
		if err != nil {
			t.Fatalf("NewVariant(da): %v", err)
		}

		runner := NewComparisonRunner().WithTarget(math.SmallestNonzeroFloat64)
		job := comparisonJob{config: jobConfig(), variant: da, variantIndex: 0, runIndex: 0, seed: 3}

		got := runner.executeJob(context.Background(), job)
		if got.err != nil {
			t.Fatalf("executeJob: %v", got.err)
		}

		if got.run.ConvergenceAt != 0 {
			t.Errorf("ConvergenceAt = %d, want 0 when the target is never reached", got.run.ConvergenceAt)
		}
	})
}

func TestFailedRunSerializesUnavailableCostAsNull(t *testing.T) {
	data, err := json.Marshal(RunResult{BestCost: math.Inf(1), Error: "objective failed"})
	if err != nil {
		t.Fatalf("json.Marshal failed RunResult: %v", err)
	}

	if !bytes.Contains(data, []byte(`"best_cost":null`)) {
		t.Errorf("failed RunResult JSON = %s, want best_cost:null", data)
	}
}
