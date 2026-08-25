package dragonfly

import (
	"math"
	"sort"
	"testing"
)

// A regression baseline in this file is a TOLERATED DEGRADATION FACTOR, not a
// golden value.
//
// A stochastic optimizer has no golden output. Rerunning it with a different
// seed gives a different answer, and a change that shifts the random stream --
// drawing one extra number, reordering two draws -- changes every number in
// this file without changing the algorithm at all. Pinning observed values
// would therefore produce a suite that fails on refactors and passes on real
// regressions, which is exactly backwards.
//
// So each baseline pairs a REFERENCE MEAN, a round number of the right order of
// magnitude for what a faithful DA achieves on that benchmark, with a
// TOLERANCE, the factor by which the measured mean is allowed to exceed it.
// Only the product is asserted. The question the suite answers is "did this
// change make the algorithm meaningfully worse", and that is a statistical
// question with a tolerance attached.
//
// NEVER paste a measured number into a baseline to make a test pass. A run that
// exceeds ReferenceMean*Tolerance is a finding: either the change degraded the
// algorithm, or the reference was wrong to begin with and moving it needs its
// own justification in this comment block. The reference means below were
// chosen by rounding the observed means over each baseline's documented
// 15-seed block up to a clean figure, in the same spirit as the empirical
// tolerances in dragonfly_test.go. The tolerance is uniform at 3x because
// stochastic run-to-run spread can approach a factor of two.
//
// The absolute numbers are deliberately unflattering. DA's convergence factor
// mc reaches zero at the halfway point, after which only the food term and
// inertia survive and the swarm collapses onto the incumbent, so the paper's
// algorithm stalls well short of the optimum on 10-dimensional multimodal
// problems. These are the numbers a faithful port actually produces, not the
// numbers a well-tuned optimizer could.

// RegressionBaseline describes one benchmark the regression suite watches.
type RegressionBaseline struct {
	Function ObjectiveFunction
	Name     string
	Variant  string

	// ReferenceMean is the mean final cost a healthy implementation reaches on
	// this benchmark with this configuration. It is a reference point for the
	// tolerance below, never an assertion in its own right.
	ReferenceMean float64

	// Tolerance is the factor by which the measured mean may exceed
	// ReferenceMean before the run counts as a regression.
	Tolerance float64

	// SuccessThreshold is the fraction of individual runs that must land at or
	// below ReferenceMean*Tolerance. It catches a change that leaves the mean
	// intact by trading a few very good runs for a few very bad ones.
	SuccessThreshold float64

	LowerBound float64
	UpperBound float64

	Dimensions int
	Iterations int
	Population int

	// Seed is the first seed of the run block; run i uses Seed+i, so the whole
	// block is reproducible and independent of wall-clock time.
	Seed int64

	// Binary selects the BDA entry point and NewBinaryConfig instead of the
	// continuous pair.
	Binary bool
}

// regressionRuns is how many seeds each baseline is measured over. Fifteen is
// the measured block recorded above: DA uses seeds 1000..1014 and BDA uses
// seeds 2000..2014.
const regressionRuns = 15

// oneMaxBits is the OneMax objective the BDA baseline minimizes: the count of
// zero bits, so an all-ones string costs zero. It is deliberately trivial --
// the baseline watches the bit-flip machinery, not the landscape.
func oneMaxBits(x []float64) float64 {
	zeros := 0.0
	for _, bit := range x {
		zeros += 1 - bit
	}

	return zeros
}

// regressionBaselines is the watched set: the four benchmarks
// dragonfly_test.go already gates on, plus Griewank for a wide box and a BDA
// entry so the binary variant is not left unwatched.
var regressionBaselines = []RegressionBaseline{
	{
		Name:             "DA_Sphere_10D",
		Function:         Sphere,
		Dimensions:       10,
		LowerBound:       -100,
		UpperBound:       100,
		Iterations:       500,
		Population:       40,
		Seed:             1000,
		ReferenceMean:    100,
		Tolerance:        3,
		SuccessThreshold: 0.8,
	},
	{
		Name:             "DA_Rastrigin_10D",
		Function:         Rastrigin,
		Dimensions:       10,
		LowerBound:       -5.12,
		UpperBound:       5.12,
		Iterations:       500,
		Population:       40,
		Seed:             1000,
		ReferenceMean:    30,
		Tolerance:        3,
		SuccessThreshold: 0.8,
	},
	{
		Name:             "DA_Ackley_10D",
		Function:         Ackley,
		Dimensions:       10,
		LowerBound:       -32.768,
		UpperBound:       32.768,
		Iterations:       500,
		Population:       40,
		Seed:             1000,
		ReferenceMean:    6,
		Tolerance:        3,
		SuccessThreshold: 0.8,
	},
	{
		Name:             "DA_Rosenbrock_10D",
		Function:         Rosenbrock,
		Dimensions:       10,
		LowerBound:       -5,
		UpperBound:       10,
		Iterations:       500,
		Population:       40,
		Seed:             1000,
		ReferenceMean:    200,
		Tolerance:        3,
		SuccessThreshold: 0.8,
	},
	{
		Name:             "DA_Griewank_10D",
		Function:         Griewank,
		Dimensions:       10,
		LowerBound:       -600,
		UpperBound:       600,
		Iterations:       500,
		Population:       40,
		Seed:             1000,
		ReferenceMean:    2,
		Tolerance:        3,
		SuccessThreshold: 0.8,
	},
	{
		// The reference is one wrong bit out of thirty, not zero: a factor
		// tolerance is meaningless against a reference of zero, and "three
		// wrong bits is a regression" is the statement worth making.
		Name:             "BDA_OneMax_30bit",
		Function:         oneMaxBits,
		Dimensions:       30,
		LowerBound:       0,
		UpperBound:       1,
		Iterations:       300,
		Population:       30,
		Seed:             2000,
		ReferenceMean:    1,
		Tolerance:        3,
		SuccessThreshold: 0.8,
		Binary:           true,
		Variant:          nameBDA,
	},
	{
		Name:             "MHDA_Sphere_10D",
		Variant:          nameMHDA,
		Function:         Sphere,
		Dimensions:       10,
		LowerBound:       -100,
		UpperBound:       100,
		Iterations:       200,
		Population:       30,
		Seed:             3000,
		ReferenceMean:    1e-8,
		Tolerance:        3,
		SuccessThreshold: 0.8,
	},
	{
		Name:             "CDA_Rastrigin_10D",
		Variant:          nameCDA,
		Function:         Rastrigin,
		Dimensions:       10,
		LowerBound:       -5.12,
		UpperBound:       5.12,
		Iterations:       200,
		Population:       30,
		Seed:             4000,
		ReferenceMean:    60,
		Tolerance:        3,
		SuccessThreshold: 0.8,
	},
	{
		Name:             "QGDA_Rastrigin_10D",
		Variant:          nameQGDA,
		Function:         Rastrigin,
		Dimensions:       10,
		LowerBound:       -5.12,
		UpperBound:       5.12,
		Iterations:       200,
		Population:       30,
		Seed:             5000,
		ReferenceMean:    1e-9,
		Tolerance:        3,
		SuccessThreshold: 0.8,
	},
}

// regressionSummary is the shape of one measured run block.
type regressionSummary struct {
	Min    float64
	Mean   float64
	Median float64
	Max    float64
	StdDev float64
}

// summarize reduces a block of final costs to the statistics the assertions
// need. It is a local helper on purpose: the regression suite must not depend
// on the comparison framework it is meant to guard.
func summarize(costs []float64) regressionSummary {
	sorted := make([]float64, len(costs))
	copy(sorted, costs)
	sort.Float64s(sorted)

	sum := 0.0
	for _, cost := range sorted {
		sum += cost
	}

	mean := sum / float64(len(sorted))

	variance := 0.0
	for _, cost := range sorted {
		variance += (cost - mean) * (cost - mean)
	}

	return regressionSummary{
		Min:    sorted[0],
		Mean:   mean,
		Median: sorted[len(sorted)/2],
		Max:    sorted[len(sorted)-1],
		StdDev: math.Sqrt(variance / float64(len(sorted))),
	}
}

// configFor builds the configuration a baseline is measured with. Run index i
// selects seed Seed+i.
func (baseline RegressionBaseline) configFor(run int) *Config {
	var config *Config

	switch baseline.Variant {
	case nameBDA:
		config = NewBinaryConfig()
	case nameCDA:
		config = NewChaoticConfig()
	case nameMHDA:
		config = NewMemoryHybridConfig()
	case nameQGDA:
		config = NewQuantumConfig()
	default:
		config = NewDefaultConfig()
	}

	config.LowerBound = baseline.LowerBound
	config.UpperBound = baseline.UpperBound

	config.ObjectiveFunc = baseline.Function
	config.ProblemSize = baseline.Dimensions
	config.MaxIterations = baseline.Iterations
	config.NPop = baseline.Population
	seed := baseline.Seed + int64(run)
	config.Seed = &seed

	return config
}

// optimize dispatches to the entry point the baseline's variant belongs to.
func (baseline RegressionBaseline) optimize(config *Config) (*Result, error) {
	switch baseline.Variant {
	case nameBDA:
		return OptimizeBinary(config)
	case nameCDA:
		return OptimizeChaotic(config)
	case nameMHDA:
		return OptimizeMemoryHybrid(config)
	case nameQGDA:
		return OptimizeQuantum(config)
	default:
		return Optimize(config)
	}
}

// threshold is the only number the assertions compare against.
func (baseline RegressionBaseline) threshold() float64 {
	return baseline.ReferenceMean * baseline.Tolerance
}

// measure runs one baseline over regressionRuns seeds and returns the costs.
func measureBaseline(t *testing.T, baseline RegressionBaseline, runs int) []float64 {
	t.Helper()

	costs := make([]float64, runs)

	for run := range runs {
		result, err := baseline.optimize(baseline.configFor(run))
		if err != nil {
			t.Fatalf("%s: run %d failed: %v", baseline.Name, run, err)
		}

		wantSeed := baseline.Seed + int64(run)
		if !result.SeedKnown || result.Seed != wantSeed {
			t.Fatalf("%s: run %d reports seed (%d, known=%t), want %d",
				baseline.Name, run, result.Seed, result.SeedKnown, wantSeed)
		}

		costs[run] = result.GlobalBest.Cost
	}

	return costs
}

// TestRegressionSuite is the statistical gate: for every watched baseline the
// mean final cost, and the fraction of runs that clear the same bar, must stay
// inside the tolerated degradation.
func TestRegressionSuite(t *testing.T) {
	if testing.Short() {
		t.Skip("the multi-seed regression suite is skipped under -short")
	}

	for _, baseline := range regressionBaselines {
		t.Run(baseline.Name, func(t *testing.T) {
			costs := measureBaseline(t, baseline, regressionRuns)
			stats := summarize(costs)
			threshold := baseline.threshold()

			successes := 0

			for _, cost := range costs {
				if cost <= threshold {
					successes++
				}
			}

			successRate := float64(successes) / float64(len(costs))

			t.Logf("min %.6g | mean %.6g | median %.6g | max %.6g | stddev %.6g | success %.0f%% | threshold %.6g",
				stats.Min, stats.Mean, stats.Median, stats.Max, stats.StdDev, successRate*100, threshold)

			if stats.Mean > threshold {
				t.Errorf("mean cost %.6g exceeds the tolerated %.6g (reference %.6g x tolerance %.1f); "+
					"this is a regression, not a baseline to update",
					stats.Mean, threshold, baseline.ReferenceMean, baseline.Tolerance)
			}

			if successRate < baseline.SuccessThreshold {
				t.Errorf("only %.0f%% of runs reached %.6g, want at least %.0f%%",
					successRate*100, threshold, baseline.SuccessThreshold*100)
			}

			// Informational: a large improvement is a change in the algorithm,
			// not luck, and is worth noticing even though it fails nothing.
			// A mean of exactly zero is the BDA baseline finding the discrete
			// optimum, which is the expected outcome rather than news.
			if stats.Mean > 0 && stats.Mean < baseline.ReferenceMean/3 {
				t.Logf("INFO: mean %.6g is far below the reference %.6g -- if this is intentional, "+
					"revisit the reference and say why here", stats.Mean, baseline.ReferenceMean)
			}
		})
	}
}

// TestRegressionQuick is the fast gate that survives -short: one seeded run per
// continuous baseline, compared against the same tolerated threshold. It cannot
// see a distribution shift, only an outright break.
func TestRegressionQuick(t *testing.T) {
	for _, baseline := range regressionBaselines {
		if baseline.Binary {
			continue
		}

		t.Run(baseline.Name, func(t *testing.T) {
			quick := baseline
			quick.Iterations = 200
			quick.Population = 30

			// A shorter, smaller run cannot reach the full baseline, so the bar
			// is relaxed by a further factor of three rather than pretending
			// the two configurations are comparable.
			threshold := quick.threshold() * 3

			result, err := quick.optimize(quick.configFor(0))
			if err != nil {
				t.Fatalf("optimization failed: %v", err)
			}

			t.Logf("cost %.6g (threshold %.6g)", result.GlobalBest.Cost, threshold)

			if result.GlobalBest.Cost > threshold {
				t.Errorf("cost %.6g exceeds the tolerated %.6g", result.GlobalBest.Cost, threshold)
			}
		})
	}
}

// TestRegressionReproducibility guards the one property that IS exact: a run
// with a fixed seed is bit-identical to itself. Unlike the cost baselines this
// is not statistical, so it is asserted without tolerance.
func TestRegressionReproducibility(t *testing.T) {
	baseline := regressionBaselines[0]
	baseline.Iterations = 100

	costs := make([]float64, 3)
	for i := range costs {
		result, err := baseline.optimize(baseline.configFor(0))
		if err != nil {
			t.Fatalf("optimization failed: %v", err)
		}

		costs[i] = result.GlobalBest.Cost
	}

	for i := 1; i < len(costs); i++ {
		if costs[i] != costs[0] {
			t.Errorf("repeat %d returned %.17g, want %.17g (identical seed)", i, costs[i], costs[0])
		}
	}
}

// TestRegressionParallelMatchesSequential is a regression gate on the
// determinism contract rather than on solution quality: enabling parallel
// evaluation must not move a single bit of a seeded run.
func TestRegressionParallelMatchesSequential(t *testing.T) {
	baseline := regressionBaselines[0]
	baseline.Iterations = 150

	sequential, err := baseline.optimize(baseline.configFor(0))
	if err != nil {
		t.Fatalf("sequential run failed: %v", err)
	}

	parallelConfig := baseline.configFor(0)
	parallelConfig.EnableParallel = true
	parallelConfig.MaxWorkers = 4

	parallel, err := baseline.optimize(parallelConfig)
	if err != nil {
		t.Fatalf("parallel run failed: %v", err)
	}

	if parallel.GlobalBest.Cost != sequential.GlobalBest.Cost {
		t.Errorf("parallel cost %.17g differs from sequential %.17g",
			parallel.GlobalBest.Cost, sequential.GlobalBest.Cost)
	}

	if parallel.FuncEvalCount != sequential.FuncEvalCount {
		t.Errorf("parallel evaluation count %d differs from sequential %d",
			parallel.FuncEvalCount, sequential.FuncEvalCount)
	}
}
