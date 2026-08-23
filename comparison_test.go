package dragonfly

import (
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"math"
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
// Under the null hypothesis, for n = 8 non-tied pairs:
//
//	E[W]  = n(n+1)/4        = 8·9/4        = 18
//	SD[W] = sqrt(n(n+1)(2n+1)/24) = sqrt(8·9·17/24) = sqrt(51) ≈ 7.14143
//	z     = |W − E[W]|/SD[W] = 14/sqrt(51) ≈ 1.96039
//	p     = 2(1 − Φ(z)) = erfc(z/√2)       ≈ 0.0500
//
// which is just below alpha = 0.05, so the difference is significant, and B --
// the variant with the lower costs -- wins.
func TestWilcoxonSignedRankHandWorked(t *testing.T) {
	runsA := runsFromCosts(11, 12, 13, 6, 15, 16, 17, 18)
	runsB := runsFromCosts(10, 10, 10, 10, 10, 10, 10, 10)

	result := wilcoxonSignedRankTest("A", "B", runsA, runsB)

	if result.WStatistic != 4 {
		t.Errorf("W = %v, want 4", result.WStatistic)
	}

	// The p-value is rebuilt here from the hand-computed W and n, so the
	// assertion does not read the implementation's arithmetic back to itself.
	wantP := math.Erfc((14 / math.Sqrt(51)) / math.Sqrt2)
	if math.Abs(result.PValue-wantP) > 1e-12 {
		t.Errorf("p = %v, want %v (2(1-Phi(14/sqrt(51))))", result.PValue, wantP)
	}

	if wantP >= 0.05 || wantP <= 0.049 {
		t.Fatalf("the hand-worked p-value drifted to %v; the example is no longer the one documented", wantP)
	}

	if !result.Significant {
		t.Errorf("Significant = false, want true at p = %v", result.PValue)
	}

	if result.Winner != "B" {
		t.Errorf("Winner = %q, want B (the variant with the lower costs)", result.Winner)
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

	if reverse.Winner != "B" {
		t.Errorf("reversed Winner = %q, want B", reverse.Winner)
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

	if tie.Significant || tie.PValue != 0 {
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
