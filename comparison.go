// Head-to-head variant comparison with paired seeds, non-parametric
// significance tests and CSV/JSON export.

package dragonfly

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// wilcoxonTie is the Winner of a Wilcoxon test that found no significant
// difference.
const wilcoxonTie = "Tie"

// significanceAlpha is the significance level both tests report against.
const significanceAlpha = 0.05

// RunResult is the outcome of one optimization run inside a comparison.
type RunResult struct {
	// Error is the run's error message, empty when the run succeeded. It is a
	// string rather than an error so the result document serializes.
	Error string `json:"error,omitempty"`

	BestCost      float64 `json:"best_cost"`
	ExecutionTime float64 `json:"execution_seconds"`
	Seed          int64   `json:"seed"`
	FuncEvals     int     `json:"function_evaluations"`
	Iterations    int     `json:"iterations"`

	// ConvergenceAt is the one-based iteration at which the run first reached
	// ComparisonRunner.TargetCost, or zero if it never did or no target was set.
	ConvergenceAt int `json:"convergence_at"`
}

// MarshalJSON represents an unavailable best cost as null. Failed comparison
// jobs deliberately carry +Inf internally so they always lose a numeric rank,
// but encoding/json cannot encode non-finite floats and an invented large
// finite value would be data corruption.
func (run RunResult) MarshalJSON() ([]byte, error) {
	var bestCost *float64

	if isFinite(run.BestCost) {
		value := run.BestCost
		bestCost = &value
	}

	return json.Marshal(runResultJSON{
		BestCost:      bestCost,
		Error:         run.Error,
		ExecutionTime: run.ExecutionTime,
		Seed:          run.Seed,
		FuncEvals:     run.FuncEvals,
		Iterations:    run.Iterations,
		ConvergenceAt: run.ConvergenceAt,
	})
}

type runResultJSON struct {
	BestCost      *float64 `json:"best_cost"`
	Error         string   `json:"error,omitempty"`
	ExecutionTime float64  `json:"execution_seconds"`
	Seed          int64    `json:"seed"`
	FuncEvals     int      `json:"function_evaluations"`
	Iterations    int      `json:"iterations"`
	ConvergenceAt int      `json:"convergence_at"`
}

// AlgorithmStatistics aggregates one variant's runs.
type AlgorithmStatistics struct {
	Mean   float64 `json:"mean"`
	Median float64 `json:"median"`
	StdDev float64 `json:"stddev"`
	Best   float64 `json:"best"`
	Worst  float64 `json:"worst"`

	// SuccessRate is the percentage of runs that reached the target cost. It is
	// zero when no target was configured.
	SuccessRate    float64 `json:"success_rate"`
	AvgFuncEvals   float64 `json:"avg_function_evaluations"`
	AvgTime        float64 `json:"avg_execution_seconds"`
	SuccessfulRuns int     `json:"successful_runs"`
	FailedRuns     int     `json:"failed_runs"`
}

// WilcoxonResult is a Wilcoxon signed-rank test between two variants.
//
// The test is paired: run k of both variants used the same seed, so the two
// samples are matched observations of the same starting conditions rather than
// independent draws. That is what makes the signed-rank test the right test
// and what makes it far more sensitive than an unpaired alternative on the
// small run counts a comparison can afford.
type WilcoxonResult struct {
	Algorithm1 string `json:"algorithm_1"`
	Algorithm2 string `json:"algorithm_2"`

	// Winner is the variant with the lower costs when the difference is
	// significant, and wilcoxonTie otherwise.
	Winner string `json:"winner"`

	// WStatistic is min(W+, W-), the smaller of the positive and negative
	// signed-rank sums.
	WStatistic float64 `json:"w_statistic"`

	// PValue is two-tailed. Samples of at most 20 non-zero pairs use the exact
	// conditional sign distribution; larger samples use a continuity- and
	// tie-corrected normal approximation.
	PValue         float64 `json:"p_value"`
	AdjustedPValue float64 `json:"adjusted_p_value"`
	Pairs          int     `json:"pairs"`
	Significant    bool    `json:"significant"`
	Available      bool    `json:"available"`
}

// FriedmanTestResult is a Friedman test across every variant at once: the
// non-parametric analog of a repeated-measures ANOVA over the per-run ranks.
type FriedmanTestResult struct {
	ChiSquare        float64 `json:"chi_square"`
	PValue           float64 `json:"p_value"`
	DegreesOfFreedom int     `json:"degrees_of_freedom"`
	Blocks           int     `json:"blocks"`
	Significant      bool    `json:"significant"`
	Available        bool    `json:"available"`
}

// ComparisonResult is the complete outcome of a comparison.
type ComparisonResult struct {
	FriedmanResult *FriedmanTestResult   `json:"friedman,omitempty"`
	BenchmarkName  string                `json:"benchmark"`
	Error          string                `json:"error,omitempty"`
	AlgorithmNames []string              `json:"algorithms"`
	RunResults     [][]RunResult         `json:"runs"`
	Statistics     []AlgorithmStatistics `json:"statistics"`
	Rankings       []int                 `json:"rankings"`
	WilcoxonTests  [][]WilcoxonResult    `json:"wilcoxon"`
	BestAlgorithm  int                   `json:"best_algorithm"`
	BaseSeed       int64                 `json:"base_seed"`
}

// ComparisonRunner runs several variants over the same problem and the same
// seeds, and reports whether their differences are significant.
//
// Seeds are paired: run k of every variant is given the seed BaseSeed + k, so
// the variants face identical starting swarms and identical random streams and
// the differences that remain are the algorithms'. That pairing is also what
// the Wilcoxon and Friedman tests assume.
//
// When Parallel is true the objective function must be safe for concurrent use.
// MaxWorkers bounds concurrent runs; Config.EnableParallel is an independent
// inner limit. Parallelism changes only the order runs complete in, never which
// seed a run gets, so a parallel comparison is bit-identical to a sequential
// one.
type ComparisonRunner struct {
	// Variants defaults to SingleObjectiveVariants(). A multi-objective
	// variant is rejected: every statistic here is defined over a scalar cost.
	Variants []AlgorithmVariant

	// TargetCost is the success threshold used for SuccessRate and
	// ConvergenceAt. Call WithTarget to enable zero or negative thresholds.
	TargetCost float64

	// Seed is the base seed run 0 uses.
	Seed int64

	Runs          int
	MaxIterations int

	// MaxWorkers bounds concurrent runs when Parallel is set. Zero means
	// runtime.NumCPU().
	MaxWorkers int

	Verbose   bool
	Parallel  bool
	targetSet bool
}

// NewComparisonRunner creates a runner over every single-objective variant,
// with the 30 runs conventional for statistical significance in the
// metaheuristics literature.
func NewComparisonRunner() *ComparisonRunner {
	return &ComparisonRunner{
		Variants:      SingleObjectiveVariants(),
		Runs:          30,
		MaxIterations: 500,
		MaxWorkers:    runtime.NumCPU(),
		Seed:          time.Now().UnixNano(),
	}
}

// WithVariants sets the variants to compare.
func (cr *ComparisonRunner) WithVariants(variants ...AlgorithmVariant) *ComparisonRunner {
	cr.Variants = variants

	return cr
}

// WithVariantNames sets the variants to compare by name. An unrecognized name
// is an error, reported when Compare runs.
func (cr *ComparisonRunner) WithVariantNames(names ...string) *ComparisonRunner {
	variants := make([]AlgorithmVariant, 0, len(names))

	for _, name := range names {
		variant, err := NewVariant(name)
		if err != nil {
			// A nil entry is rejected by validate with the index, which is a
			// better message than dropping the name silently.
			variants = append(variants, nil)

			continue
		}

		variants = append(variants, variant)
	}

	cr.Variants = variants

	return cr
}

// WithRuns sets the number of runs per variant.
func (cr *ComparisonRunner) WithRuns(runs int) *ComparisonRunner {
	cr.Runs = runs

	return cr
}

// WithTarget sets the success threshold.
func (cr *ComparisonRunner) WithTarget(target float64) *ComparisonRunner {
	cr.TargetCost = target
	cr.targetSet = true

	return cr
}

// WithIterations sets the maximum iterations per run.
func (cr *ComparisonRunner) WithIterations(iterations int) *ComparisonRunner {
	cr.MaxIterations = iterations

	return cr
}

// WithVerbose enables per-run progress output.
func (cr *ComparisonRunner) WithVerbose(verbose bool) *ComparisonRunner {
	cr.Verbose = verbose

	return cr
}

// WithParallel enables or disables concurrent runs.
func (cr *ComparisonRunner) WithParallel(parallel bool) *ComparisonRunner {
	cr.Parallel = parallel

	return cr
}

// WithMaxWorkers bounds concurrent runs. Zero means runtime.NumCPU().
func (cr *ComparisonRunner) WithMaxWorkers(workers int) *ComparisonRunner {
	cr.MaxWorkers = workers

	return cr
}

// WithSeed sets the base seed the paired per-run seeds are derived from.
func (cr *ComparisonRunner) WithSeed(seed int64) *ComparisonRunner {
	cr.Seed = seed

	return cr
}

// Compare runs every variant on the problem and returns the comparison. A
// failing run is recorded in its RunResult and does not stop the comparison;
// use CompareContext when a failure should abort.
func (cr *ComparisonRunner) Compare(
	benchmarkName string,
	fn ObjectiveFunction,
	problemSize int,
	lower, upper float64,
) *ComparisonResult {
	result, err := cr.compare(context.Background(), benchmarkName, fn, problemSize, lower, upper, true)
	if result == nil {
		result = &ComparisonResult{BenchmarkName: benchmarkName, BestAlgorithm: -1}
		if cr != nil {
			result.BaseSeed = cr.Seed
		}
	}

	if err != nil {
		result.Error = err.Error()
	}

	return result
}

// CompareContext runs every variant with cancellation and explicit error
// reporting. It returns no partial aggregate when any run fails, so a caller
// cannot mistake a broken comparison for a completed one.
func (cr *ComparisonRunner) CompareContext(
	ctx context.Context,
	benchmarkName string,
	fn ObjectiveFunction,
	problemSize int,
	lower, upper float64,
) (*ComparisonResult, error) {
	return cr.compare(ctx, benchmarkName, fn, problemSize, lower, upper, false)
}

func (cr *ComparisonRunner) targetEnabled() bool {
	if cr == nil {
		return false
	}

	// A positive directly assigned TargetCost retains the pre-v0.2 behavior.
	// Zero and negative targets are enabled through WithTarget, which records
	// the otherwise unrepresentable distinction between zero and "unset".
	return cr.targetSet || cr.TargetCost > 0
}

type comparisonJob struct {
	config       *Config
	variant      AlgorithmVariant
	variantIndex int
	runIndex     int
	seed         int64
}

type comparisonJobResult struct {
	err          error
	run          RunResult
	variantIndex int
	runIndex     int
}

func (cr *ComparisonRunner) compare(
	ctx context.Context,
	benchmarkName string,
	fn ObjectiveFunction,
	problemSize int,
	lower, upper float64,
	continueOnError bool,
) (*ComparisonResult, error) {
	err := cr.validate(ctx, fn, problemSize, lower, upper)
	if err != nil {
		return &ComparisonResult{
			BenchmarkName: benchmarkName,
			BestAlgorithm: -1,
			BaseSeed:      cr.Seed,
			Error:         err.Error(),
		}, err
	}

	names, runResults, jobs, err := cr.prepareJobs(fn, problemSize, lower, upper)
	if err != nil {
		return nil, err
	}

	err = cr.runJobs(ctx, jobs, names, runResults, continueOnError)
	if err != nil {
		return nil, err
	}

	return cr.aggregate(benchmarkName, names, runResults), nil
}

// prepareJobs builds one job per (run, variant) pair. The run index is the
// outer loop so that the paired seed is drawn once and shared, which is what
// makes the comparison paired.
func (cr *ComparisonRunner) prepareJobs(
	fn ObjectiveFunction,
	problemSize int,
	lower, upper float64,
) ([]string, [][]RunResult, []comparisonJob, error) {
	names := make([]string, len(cr.Variants))
	runResults := make([][]RunResult, len(cr.Variants))

	for i, variant := range cr.Variants {
		names[i] = variant.Name()
		runResults[i] = make([]RunResult, cr.Runs)
	}

	jobs := make([]comparisonJob, 0, len(cr.Variants)*cr.Runs)

	for run := range cr.Runs {
		seed := cr.Seed + int64(run)

		for variantIndex, variant := range cr.Variants {
			config := variant.GetConfig()
			if config == nil {
				return nil, nil, nil, fmt.Errorf("variant %s returned a nil config", variant.Name())
			}

			cr.applyProblem(config, fn, problemSize, lower, upper, seed)
			jobs = append(jobs, comparisonJob{
				config:       config,
				variant:      variant,
				variantIndex: variantIndex,
				runIndex:     run,
				seed:         seed,
			})
		}
	}

	return names, runResults, jobs, nil
}

// applyProblem writes the problem and the paired seed into a variant's default
// configuration.
//
// A binary configuration keeps its own bounds. NewBinaryConfig fixes the search
// box at the unit interval because a position component is a bit and every
// schedule that scales with (ub-lb) is written for that box; overwriting it
// with the caller's continuous bounds would break BDA's step clamp rather than
// make it comparable.
func (cr *ComparisonRunner) applyProblem(
	config *Config,
	fn ObjectiveFunction,
	problemSize int,
	lower, upper float64,
	seed int64,
) {
	config.ObjectiveFunc = fn
	config.ProblemSize = problemSize
	config.MaxIterations = cr.MaxIterations
	config.Rand = nil
	config.Seed = &seed

	if !config.UseBinary {
		config.LowerBound = lower
		config.UpperBound = upper
	}
}

func (cr *ComparisonRunner) runJobs(
	ctx context.Context,
	jobs []comparisonJob,
	names []string,
	runResults [][]RunResult,
	continueOnError bool,
) error {
	workerCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	jobCh := make(chan comparisonJob, len(jobs))
	resultCh := make(chan comparisonJobResult, len(jobs))

	for _, job := range jobs {
		jobCh <- job
	}

	close(jobCh)

	workerCount := cr.workerCount(len(jobs))

	var workers sync.WaitGroup

	workers.Add(workerCount)

	for range workerCount {
		go cr.worker(workerCtx, jobCh, resultCh, &workers)
	}

	go func() {
		workers.Wait()
		close(resultCh)
	}()

	firstErr := cr.collect(resultCh, names, runResults, len(jobs), continueOnError, cancel)
	if firstErr != nil && !continueOnError {
		return firstErr
	}

	contextErr := ctx.Err()
	if contextErr != nil && !continueOnError {
		return contextErr
	}

	return nil
}

func (cr *ComparisonRunner) workerCount(jobCount int) int {
	if !cr.Parallel {
		return 1
	}

	workers := cr.MaxWorkers
	if workers <= 0 {
		workers = runtime.NumCPU()
	}

	return max(1, min(workers, jobCount))
}

func (cr *ComparisonRunner) worker(
	ctx context.Context,
	jobs <-chan comparisonJob,
	results chan<- comparisonJobResult,
	workers *sync.WaitGroup,
) {
	defer workers.Done()

	for {
		select {
		case <-ctx.Done():
			return
		case job, ok := <-jobs:
			if !ok {
				return
			}

			results <- cr.executeJob(ctx, job)
		}
	}
}

func (cr *ComparisonRunner) collect(
	results <-chan comparisonJobResult,
	names []string,
	runResults [][]RunResult,
	jobCount int,
	continueOnError bool,
	cancel context.CancelFunc,
) error {
	var firstErr error

	completed := 0

	for jobResult := range results {
		completed++

		if jobResult.err != nil && firstErr == nil {
			firstErr = fmt.Errorf("compare %s run %d: %w",
				names[jobResult.variantIndex], jobResult.runIndex+1, jobResult.err)

			if !continueOnError {
				cancel()
			}
		}

		runResults[jobResult.variantIndex][jobResult.runIndex] = jobResult.run

		if cr.Verbose {
			fmt.Printf("Completed %d/%d comparison runs\n", completed, jobCount)
		}
	}

	return firstErr
}

func (cr *ComparisonRunner) validate(
	ctx context.Context,
	fn ObjectiveFunction,
	problemSize int,
	lower, upper float64,
) error {
	err := requireContext(ctx)
	if err != nil {
		return err
	}

	err = cr.validateVariants()
	if err != nil {
		return err
	}

	err = cr.validateBudget()
	if err != nil {
		return err
	}

	if fn == nil {
		return errors.New("comparison objective function is required")
	}

	if problemSize <= 0 {
		return fmt.Errorf("comparison problem size must be positive, got %d", problemSize)
	}

	if !isFinite(lower) || !isFinite(upper) {
		return errors.New("comparison bounds must be finite")
	}

	if lower >= upper {
		return fmt.Errorf("comparison lower bound %v must be less than upper bound %v", lower, upper)
	}

	return ctx.Err()
}

func (cr *ComparisonRunner) validateVariants() error {
	if len(cr.Variants) == 0 {
		return errors.New("at least one comparison variant is required")
	}

	for i, variant := range cr.Variants {
		if variant == nil {
			return fmt.Errorf("comparison variant %d is nil or was not recognized", i)
		}

		if variant.IsMultiObjective() {
			return fmt.Errorf(
				"comparison variant %d (%s) is multi-objective: %w",
				i, variant.Name(), ErrMultiObjectiveVariant)
		}
	}

	return nil
}

func (cr *ComparisonRunner) validateBudget() error {
	if cr.Runs <= 0 {
		return fmt.Errorf("comparison runs must be positive, got %d", cr.Runs)
	}

	if cr.MaxIterations <= 0 {
		return fmt.Errorf("comparison iterations must be positive, got %d", cr.MaxIterations)
	}

	if cr.MaxWorkers < 0 {
		return fmt.Errorf("comparison MaxWorkers must be non-negative, got %d", cr.MaxWorkers)
	}

	return nil
}

func (cr *ComparisonRunner) executeJob(ctx context.Context, job comparisonJob) comparisonJobResult {
	start := time.Now()
	result, err := job.variant.Run(ctx, job.config)

	run := RunResult{BestCost: math.Inf(1), ExecutionTime: time.Since(start).Seconds(), Seed: job.seed}
	if err != nil {
		run.Error = err.Error()

		return comparisonJobResult{
			run: run, variantIndex: job.variantIndex, runIndex: job.runIndex, err: err,
		}
	}

	if cr.targetEnabled() {
		for iteration, cost := range result.ConvergenceCurve {
			if cost <= cr.TargetCost {
				run.ConvergenceAt = iteration + 1

				break
			}
		}
	}

	run.BestCost = result.GlobalBest.Cost
	run.FuncEvals = result.FuncEvalCount
	run.Iterations = result.IterationCount

	return comparisonJobResult{run: run, variantIndex: job.variantIndex, runIndex: job.runIndex}
}

func (cr *ComparisonRunner) aggregate(
	benchmarkName string,
	names []string,
	runResults [][]RunResult,
) *ComparisonResult {
	count := len(cr.Variants)

	statistics := make([]AlgorithmStatistics, count)
	for i := range count {
		statistics[i] = calculateAlgorithmStatistics(runResults[i], cr.TargetCost, cr.targetEnabled())
	}

	rankings := rankAlgorithms(statistics)
	bestAlgorithm := -1

	for i, rank := range rankings {
		if rank == 1 && statistics[i].SuccessfulRuns > 0 {
			bestAlgorithm = i

			break
		}
	}

	wilcoxonTests := make([][]WilcoxonResult, count)
	for i := range count {
		wilcoxonTests[i] = make([]WilcoxonResult, count)
	}

	for i := range count {
		for j := i + 1; j < count; j++ {
			result := wilcoxonSignedRankTest(names[i], names[j], runResults[i], runResults[j])
			wilcoxonTests[i][j] = result

			reverse := result
			reverse.Algorithm1, reverse.Algorithm2 = result.Algorithm2, result.Algorithm1
			wilcoxonTests[j][i] = reverse
		}
	}

	adjustWilcoxonHolm(wilcoxonTests)

	return &ComparisonResult{
		FriedmanResult: friedmanTest(runResults),
		BenchmarkName:  benchmarkName,
		AlgorithmNames: names,
		RunResults:     runResults,
		Statistics:     statistics,
		Rankings:       rankings,
		WilcoxonTests:  wilcoxonTests,
		BestAlgorithm:  bestAlgorithm,
		BaseSeed:       cr.Seed,
	}
}

func adjustWilcoxonHolm(results [][]WilcoxonResult) {
	type pair struct {
		p    float64
		i, j int
	}

	pairs := make([]pair, 0)

	for i := range results {
		for j := i + 1; j < len(results[i]); j++ {
			result := results[i][j]
			if result.Available {
				pairs = append(pairs, pair{p: result.PValue, i: i, j: j})
			}
		}
	}

	sort.SliceStable(pairs, func(i, j int) bool { return pairs[i].p < pairs[j].p })

	previous := 0.0

	for rank, item := range pairs {
		adjusted := min(1, float64(len(pairs)-rank)*item.p)
		adjusted = max(previous, adjusted)
		previous = adjusted

		forward := &results[item.i][item.j]
		reverse := &results[item.j][item.i]
		forward.AdjustedPValue = adjusted
		reverse.AdjustedPValue = adjusted
		forward.Significant = adjusted < significanceAlpha

		reverse.Significant = forward.Significant
		if !forward.Significant {
			forward.Winner = wilcoxonTie
			reverse.Winner = wilcoxonTie
		}
	}
}

// calculateAlgorithmStatistics aggregates one variant's runs.
func calculateAlgorithmStatistics(
	runs []RunResult,
	targetCost float64,
	targetEnabled ...bool,
) AlgorithmStatistics {
	if len(runs) == 0 {
		return AlgorithmStatistics{}
	}

	enabled := targetCost > 0
	if len(targetEnabled) > 0 {
		enabled = targetEnabled[0]
	}

	costs := make([]float64, 0, len(runs))
	funcEvals := 0.0
	execTime := 0.0
	successes := 0
	failures := 0

	for _, run := range runs {
		if run.Error != "" || !isFinite(run.BestCost) {
			failures++

			continue
		}

		costs = append(costs, run.BestCost)
		funcEvals += float64(run.FuncEvals)
		execTime += run.ExecutionTime

		if enabled && run.BestCost <= targetCost {
			successes++
		}
	}

	statistics := AlgorithmStatistics{
		SuccessRate:    float64(successes) / float64(len(runs)) * 100.0,
		SuccessfulRuns: len(costs),
		FailedRuns:     failures,
	}
	if len(costs) == 0 {
		return statistics
	}

	mean, stdDev := meanAndStdDev(costs)

	sorted := make([]float64, len(costs))
	copy(sorted, costs)
	sort.Float64s(sorted)

	statistics.Mean = mean
	statistics.Median = medianOfSorted(sorted)
	statistics.StdDev = stdDev
	statistics.Best = sorted[0]
	statistics.Worst = sorted[len(sorted)-1]
	statistics.AvgFuncEvals = funcEvals / float64(len(costs))
	statistics.AvgTime = execTime / float64(len(costs))

	return statistics
}

// medianOfSorted returns the median of an already sorted, non-empty slice.
func medianOfSorted(sorted []float64) float64 {
	middle := len(sorted) / 2
	if len(sorted)%2 == 0 {
		return (sorted[middle-1] + sorted[middle]) / 2.0
	}

	return sorted[middle]
}

// rankAlgorithms ranks variants by mean cost, 1 being best. Ties are broken by
// index, so the ranking is stable.
func rankAlgorithms(statistics []AlgorithmStatistics) []int {
	indices := make([]int, len(statistics))
	for i := range indices {
		indices[i] = i
	}

	sort.SliceStable(indices, func(i, j int) bool {
		left := statistics[indices[i]]

		right := statistics[indices[j]]
		if (left.SuccessfulRuns == 0) != (right.SuccessfulRuns == 0) {
			return left.SuccessfulRuns > 0
		}

		return left.Mean < right.Mean
	})

	rankings := make([]int, len(statistics))
	for rank, index := range indices {
		rankings[index] = rank + 1
	}

	return rankings
}

// wilcoxonSignedRankTest performs a two-tailed Wilcoxon signed-rank test on the
// paired best costs of two variants.
//
// Zero differences are dropped and the ranks recomputed over what remains,
// which is Wilcoxon's original handling. Small samples use the exact
// conditional sign distribution; larger samples use a corrected normal
// approximation.
func wilcoxonSignedRankTest(name1, name2 string, runs1, runs2 []RunResult) WilcoxonResult {
	if len(runs1) != len(runs2) {
		return WilcoxonResult{Algorithm1: name1, Algorithm2: name2, Winner: "Error: unequal sample sizes"}
	}

	differences := make([]float64, 0, len(runs1))
	absolute := make([]float64, 0, len(runs1))

	for i := range runs1 {
		if !successfulRun(runs1[i]) || !successfulRun(runs2[i]) {
			continue
		}

		diff := runs1[i].BestCost - runs2[i].BestCost
		if math.Abs(diff) > 1e-10 {
			differences = append(differences, diff)
			absolute = append(absolute, math.Abs(diff))
		}
	}

	if len(differences) == 0 {
		return WilcoxonResult{
			Algorithm1: name1, Algorithm2: name2, Winner: wilcoxonTie,
			PValue: 1, AdjustedPValue: 1, Available: true,
		}
	}

	ranks := rankValues(absolute)
	wPlus, wMinus := 0.0, 0.0

	for i, diff := range differences {
		if diff > 0 {
			wPlus += ranks[i]
		} else {
			wMinus += ranks[i]
		}
	}

	w := math.Min(wPlus, wMinus)

	var pValue float64

	if len(differences) <= 20 {
		pValue = exactSignedRankPValue(ranks, wPlus)
	} else {
		pValue = approximateSignedRankPValue(absolute, wPlus)
	}

	significant := pValue < significanceAlpha

	// A positive difference means variant 1 cost more on that run, so the
	// smaller signed-rank sum belongs to the better variant.
	winner := wilcoxonTie
	if significant {
		winner = name2
		if wPlus < wMinus {
			winner = name1
		}
	}

	return WilcoxonResult{
		Algorithm1:     name1,
		Algorithm2:     name2,
		Winner:         winner,
		WStatistic:     w,
		PValue:         pValue,
		AdjustedPValue: pValue,
		Pairs:          len(differences),
		Significant:    significant,
		Available:      true,
	}
}

func successfulRun(run RunResult) bool {
	return run.Error == "" && isFinite(run.BestCost)
}

// exactSignedRankPValue enumerates the conditional sign distribution for the
// observed midranks. It remains exact in the presence of tied absolute
// differences and is cheap for the small samples where the normal
// approximation is least trustworthy.
func exactSignedRankPValue(ranks []float64, observedWPlus float64) float64 {
	totalRank := 0.0
	for _, rank := range ranks {
		totalRank += rank
	}

	mean := totalRank / 2
	observedDistance := math.Abs(observedWPlus - mean)
	combinations := uint64(1) << len(ranks)
	extreme := uint64(0)

	for mask := range combinations {
		wPlus := 0.0

		for i, rank := range ranks {
			if mask&(uint64(1)<<i) != 0 {
				wPlus += rank
			}
		}

		if math.Abs(wPlus-mean)+1e-12 >= observedDistance {
			extreme++
		}
	}

	return float64(extreme) / float64(combinations)
}

func approximateSignedRankPValue(absolute []float64, observedWPlus float64) float64 {
	n := float64(len(absolute))
	mean := n * (n + 1) / 4
	variance := n * (n + 1) * (2*n + 1) / 24

	sorted := append([]float64(nil), absolute...)
	sort.Float64s(sorted)

	for i := 0; i < len(sorted); {
		j := i + 1
		for j < len(sorted) && math.Abs(sorted[j]-sorted[i]) < 1e-10 {
			j++
		}

		tie := float64(j - i)
		if tie > 1 {
			variance -= (tie*tie*tie - tie) / 48
		}

		i = j
	}

	if !(variance > 0) {
		return 1
	}

	z := max(0, math.Abs(observedWPlus-mean)-0.5) / math.Sqrt(variance)

	return min(1, 2*(1-normalCDF(z)))
}

// friedmanTest performs a Friedman test over the per-run ranks of every
// variant. It returns nil for fewer than two variants or no runs.
//
//	chi² = 12 / (n·k·(k+1)) · Σ R_j²  −  3·n·(k+1)
//
// with k variants, n runs and R_j the sum of variant j's ranks. Under the null
// hypothesis chi² follows a chi-square distribution with k−1 degrees of
// freedom, and the p-value is its upper tail.
func friedmanTest(runResults [][]RunResult) *FriedmanTestResult {
	k := len(runResults)
	if k < 2 || len(runResults[0]) == 0 {
		return nil
	}

	n := len(runResults[0])
	for _, runs := range runResults[1:] {
		n = min(n, len(runs))
	}

	if n == 0 {
		return nil
	}

	rankSums := make([]float64, k)
	tieTerm := 0.0
	blocks := 0

	for run := range n {
		costs := make([]float64, k)
		complete := true

		for algorithm := range k {
			observation := runResults[algorithm][run]
			if !successfulRun(observation) {
				complete = false

				break
			}

			costs[algorithm] = observation.BestCost
		}

		if !complete {
			continue
		}

		for algorithm, rank := range rankValues(costs) {
			rankSums[algorithm] += rank
		}

		tieTerm += friedmanTieTerm(costs)
		blocks++
	}

	if blocks == 0 {
		return nil
	}

	sumSquaredRanks := 0.0
	for _, rankSum := range rankSums {
		sumSquaredRanks += rankSum * rankSum
	}

	nf, kf := float64(blocks), float64(k)
	chiSquare := 12.0/(nf*kf*(kf+1))*sumSquaredRanks - 3.0*nf*(kf+1)

	correction := 1 - tieTerm/(nf*kf*(kf*kf-1))
	if correction > 0 {
		chiSquare /= correction
	} else {
		chiSquare = 0
	}

	chiSquare = max(0, chiSquare)
	df := k - 1
	pValue := chiSquareSurvival(chiSquare, df)

	return &FriedmanTestResult{
		ChiSquare:        chiSquare,
		PValue:           pValue,
		DegreesOfFreedom: df,
		Blocks:           blocks,
		Significant:      pValue < significanceAlpha,
		Available:        true,
	}
}

func friedmanTieTerm(values []float64) float64 {
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)

	term := 0.0

	for i := 0; i < len(sorted); {
		j := i + 1
		for j < len(sorted) && math.Abs(sorted[j]-sorted[i]) < 1e-10 {
			j++
		}

		tie := float64(j - i)
		if tie > 1 {
			term += tie*tie*tie - tie
		}

		i = j
	}

	return term
}

// rankValues ranks values from 1 (smallest) upward, giving tied values their
// average rank.
func rankValues(values []float64) []float64 {
	indices := make([]int, len(values))
	for i := range indices {
		indices[i] = i
	}

	sort.SliceStable(indices, func(i, j int) bool {
		return values[indices[i]] < values[indices[j]]
	})

	ranks := make([]float64, len(values))

	for i := 0; i < len(indices); {
		j := i
		for j < len(indices) && math.Abs(values[indices[j]]-values[indices[i]]) < 1e-10 {
			j++
		}

		averageRank := 0.0
		for position := i; position < j; position++ {
			averageRank += float64(position + 1)
		}

		averageRank /= float64(j - i)

		for position := i; position < j; position++ {
			ranks[indices[position]] = averageRank
		}

		i = j
	}

	return ranks
}

// normalCDF is the standard normal cumulative distribution function.
func normalCDF(x float64) float64 {
	return 0.5 * (1.0 + math.Erf(x/math.Sqrt2))
}

// chiSquareSurvival returns P(X > x) for a chi-square variable with df degrees
// of freedom -- the upper tail, which is the p-value the Friedman test needs.
func chiSquareSurvival(x float64, df int) float64 {
	if df <= 0 {
		return math.NaN()
	}

	if x <= 0 {
		return 1
	}

	return regularizedGammaQ(float64(df)/2.0, x/2.0)
}

// regularizedGammaQ is Q(a, x), the regularized upper incomplete gamma
// function. The series converges quickly below a+1 and the continued fraction
// above it, which is the standard split.
func regularizedGammaQ(a, x float64) float64 {
	if x < 0 || a <= 0 {
		return math.NaN()
	}

	if x == 0 {
		return 1
	}

	if x < a+1 {
		return 1 - lowerGammaSeries(a, x)
	}

	return upperGammaContinuedFraction(a, x)
}

const (
	gammaMaxIterations = 300
	gammaEpsilon       = 3e-14
	gammaTiny          = 1e-300
)

// lowerGammaSeries evaluates P(a, x) by its series expansion.
func lowerGammaSeries(a, x float64) float64 {
	logPrefactor := -x + a*math.Log(x) - logGamma(a)
	term := 1.0 / a
	sum := term
	next := a

	for range gammaMaxIterations {
		next++
		term *= x / next
		sum += term

		if math.Abs(term) < math.Abs(sum)*gammaEpsilon {
			break
		}
	}

	return sum * math.Exp(logPrefactor)
}

// upperGammaContinuedFraction evaluates Q(a, x) by its continued fraction,
// using the modified Lentz algorithm.
func upperGammaContinuedFraction(a, x float64) float64 {
	logPrefactor := -x + a*math.Log(x) - logGamma(a)

	b := x + 1 - a
	c := 1 / gammaTiny
	d := 1 / b
	h := d

	for i := 1; i <= gammaMaxIterations; i++ {
		an := -float64(i) * (float64(i) - a)
		b += 2

		d = an*d + b
		if math.Abs(d) < gammaTiny {
			d = gammaTiny
		}

		c = b + an/c
		if math.Abs(c) < gammaTiny {
			c = gammaTiny
		}

		d = 1 / d
		delta := d * c
		h *= delta

		if math.Abs(delta-1) < gammaEpsilon {
			break
		}
	}

	return h * math.Exp(logPrefactor)
}

func logGamma(x float64) float64 {
	value, _ := math.Lgamma(x)

	return value
}

// PrintComparisonResults writes the formatted report to standard output.
func (cr *ComparisonResult) PrintComparisonResults() {
	_ = cr.WriteComparisonResults(os.Stdout)
}

// WriteComparisonResults writes a formatted statistical report and a relative
// quality chart. A longer bar is a better (lower) finite mean cost.
func (cr *ComparisonResult) WriteComparisonResults(w io.Writer) error {
	if w == nil {
		return errors.New("comparison report writer cannot be nil")
	}

	err := cr.validateShape()
	if err != nil {
		return err
	}

	var writeErr error

	writef := func(format string, args ...any) {
		if writeErr != nil {
			return
		}

		_, writeErr = fmt.Fprintf(w, format, args...)
	}

	cr.writeSummary(writef)
	cr.writeQualityChart(writef)
	cr.writeSignificance(writef)

	writef("%s\n", strings.Repeat("=", 80))

	return writeErr
}

// comparisonCSVHeader is the column order of ExportToCSV.
func comparisonCSVHeader() []string {
	return []string{
		"benchmark", "variant", "rank", "run", "seed", "best_cost", "function_evaluations",
		"iterations", "convergence_at", "execution_seconds", "error", "mean", "median", "stddev",
		"best", "worst", "success_rate", "avg_function_evaluations", "avg_execution_seconds",
		"successful_runs", "failed_runs",
	}
}

// comparisonCSVRow renders one run, with its variant's aggregate statistics
// repeated on every row so the file is usable without a join.
func comparisonCSVRow(
	benchmark, name string,
	rank, runIndex int,
	run RunResult,
	stats AlgorithmStatistics,
) []string {
	number := func(value float64) string {
		if !isFinite(value) {
			return ""
		}

		return strconv.FormatFloat(value, 'g', -1, 64)
	}

	return []string{
		benchmark, name, strconv.Itoa(rank), strconv.Itoa(runIndex + 1),
		strconv.FormatInt(run.Seed, 10), number(run.BestCost),
		strconv.Itoa(run.FuncEvals), strconv.Itoa(run.Iterations), strconv.Itoa(run.ConvergenceAt),
		number(run.ExecutionTime), run.Error,
		number(stats.Mean), number(stats.Median), number(stats.StdDev),
		number(stats.Best), number(stats.Worst), number(stats.SuccessRate),
		number(stats.AvgFuncEvals), number(stats.AvgTime),
		strconv.Itoa(stats.SuccessfulRuns), strconv.Itoa(stats.FailedRuns),
	}
}

// ExportToCSV writes one row per run, in variant then run order.
func (cr *ComparisonResult) ExportToCSV(path string) error {
	err := cr.validateShape()
	if err != nil {
		return err
	}

	return writeExportFile(path, "comparison CSV", func(sink io.Writer) error {
		writer := csv.NewWriter(sink)

		err := writer.Write(comparisonCSVHeader())
		if err != nil {
			return fmt.Errorf("write comparison CSV header: %w", err)
		}

		for algorithm, name := range cr.AlgorithmNames {
			for runIndex, run := range cr.RunResults[algorithm] {
				err = writer.Write(comparisonCSVRow(
					cr.BenchmarkName, name, cr.Rankings[algorithm], runIndex,
					run, cr.Statistics[algorithm]))
				if err != nil {
					return fmt.Errorf("write comparison CSV row: %w", err)
				}
			}
		}

		writer.Flush()

		err = writer.Error()
		if err != nil {
			return fmt.Errorf("flush comparison CSV: %w", err)
		}

		return nil
	})
}

// ExportToJSON writes the complete comparison result as an indented document.
func (cr *ComparisonResult) ExportToJSON(path string) error {
	err := cr.validateShape()
	if err != nil {
		return err
	}

	return writeExportFile(path, "comparison JSON", func(sink io.Writer) error {
		encoder := json.NewEncoder(sink)
		encoder.SetIndent("", "  ")

		encodeErr := encoder.Encode(cr)
		if encodeErr != nil {
			return fmt.Errorf("encode comparison JSON: %w", encodeErr)
		}

		return nil
	})
}

func (cr *ComparisonResult) writeSummary(writef func(string, ...any)) {
	line := strings.Repeat("=", 80)
	writef("\n%s\nBenchmark Comparison: %s\n%s\n", line, cr.BenchmarkName, line)
	writef("\nStatistical Summary (base seed %d):\n%s\n", cr.BaseSeed, strings.Repeat("-", 80))
	writef("%-10s | %8s | %8s | %8s | %8s | %8s | %5s | %7s\n",
		"Variant", "Mean", "Median", "StdDev", "Best", "Worst", "Rank", "OK/Fail")
	writef("%s\n", strings.Repeat("-", 80))

	for i, name := range cr.AlgorithmNames {
		stats := cr.Statistics[i]
		writef("%-10s | %8.2e | %8.2e | %8.2e | %8.2e | %8.2e | %5d | %3d/%-3d\n",
			name, stats.Mean, stats.Median, stats.StdDev, stats.Best, stats.Worst, cr.Rankings[i],
			stats.SuccessfulRuns, stats.FailedRuns)
	}

	writef("%s\n", strings.Repeat("-", 80))

	if cr.BestAlgorithm >= 0 {
		writef("\nBest variant: %s (rank 1)\n", cr.AlgorithmNames[cr.BestAlgorithm])
	} else {
		writef("\nBest variant: unavailable (all runs failed)\n")
	}
}

func (cr *ComparisonResult) writeQualityChart(writef func(string, ...any)) {
	writef("\nRelative quality (lower mean cost is better):\n")

	for _, index := range cr.rankedIndices() {
		stats := cr.Statistics[index]
		if stats.SuccessfulRuns == 0 && stats.FailedRuns > 0 {
			writef("%2d. %-10s |%-24s| failed/unavailable\n", cr.Rankings[index],
				cr.AlgorithmNames[index], "")

			continue
		}

		bar, label := cr.qualityBar(stats.Mean, 24)
		writef("%2d. %-10s |%-24s| %s\n", cr.Rankings[index], cr.AlgorithmNames[index], bar, label)
	}
}

func (cr *ComparisonResult) writeSignificance(writef func(string, ...any)) {
	writef("\nSignificant pairwise differences (Wilcoxon signed-rank, alpha=%.2f):\n", significanceAlpha)
	writef("%s\n", strings.Repeat("-", 80))

	found := false

	for i := range cr.AlgorithmNames {
		for j := i + 1; j < len(cr.AlgorithmNames); j++ {
			test := cr.WilcoxonTests[i][j]
			if test.Significant {
				found = true

				writef("%s vs %s: W=%.1f, p=%.4f, Holm p=%.4f, winner: %s\n",
					test.Algorithm1, test.Algorithm2, test.WStatistic, test.PValue,
					test.AdjustedPValue, test.Winner)
			}
		}
	}

	if !found {
		writef("No significant differences found.\n")
	}

	if cr.FriedmanResult == nil {
		return
	}

	significance := "not significant"
	if cr.FriedmanResult.Significant {
		significance = fmt.Sprintf("significant at alpha=%.2f", significanceAlpha)
	}

	writef("\nFriedman test (overall difference):\n")
	writef("  chi-square = %.4f, df = %d, p = %.4f (%s)\n",
		cr.FriedmanResult.ChiSquare, cr.FriedmanResult.DegreesOfFreedom,
		cr.FriedmanResult.PValue, significance)
}

// rankedIndices returns the variant indices in rank order.
func (cr *ComparisonResult) rankedIndices() []int {
	indices := make([]int, len(cr.AlgorithmNames))
	for i := range indices {
		indices[i] = i
	}

	sort.SliceStable(indices, func(i, j int) bool {
		return cr.Rankings[indices[i]] < cr.Rankings[indices[j]]
	})

	return indices
}

// qualityBar renders one variant's mean cost as a bar scaled between the best
// and worst finite means in the comparison.
func (cr *ComparisonResult) qualityBar(mean float64, width int) (string, string) {
	if !isFinite(mean) {
		return strings.Repeat(" ", width), "failed/unavailable"
	}

	best, worst := math.Inf(1), math.Inf(-1)

	for _, stats := range cr.Statistics {
		if !isFinite(stats.Mean) {
			continue
		}

		best = min(best, stats.Mean)
		worst = max(worst, stats.Mean)
	}

	quality := 1.0
	if worst > best {
		quality = (worst - mean) / (worst - best)
	}

	quality = max(0, min(1, quality))
	filled := int(math.Round(quality * float64(width)))

	return strings.Repeat("#", filled) + strings.Repeat(" ", width-filled), fmt.Sprintf("mean=%g", mean)
}

// validateShape rejects a result whose parallel slices disagree, which would
// otherwise surface as an index panic inside an exporter.
func (cr *ComparisonResult) validateShape() error {
	if cr == nil {
		return errors.New("comparison result cannot be nil")
	}

	count := len(cr.AlgorithmNames)
	if count == 0 {
		return errors.New("comparison result has no variants")
	}

	if len(cr.RunResults) != count || len(cr.Statistics) != count ||
		len(cr.Rankings) != count || len(cr.WilcoxonTests) != count {
		return errors.New("comparison result fields have inconsistent variant counts")
	}

	if cr.BestAlgorithm < -1 || cr.BestAlgorithm >= count {
		return fmt.Errorf("comparison best variant index %d is out of range", cr.BestAlgorithm)
	}

	for i := range count {
		if len(cr.WilcoxonTests[i]) != count {
			return fmt.Errorf("comparison Wilcoxon row %d has inconsistent length", i)
		}
	}

	return nil
}
