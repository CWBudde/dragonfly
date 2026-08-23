# Comparison Framework

`ComparisonRunner` runs several variants over the same problem with the same seeds and reports
whether the differences between them are significant. It is the layer that makes a claim like
"BDA beats DA on this problem" checkable rather than anecdotal.

## Quick start

```go
// oneMax counts the zero bits: minimized at the all-ones vector, and defined
// for continuous positions in [0,1] too, so DA and BDA can be compared on it.
func oneMax(x []float64) float64 {
	zeros := 0.0
	for _, v := range x {
		zeros += 1 - v
	}

	return zeros
}

runner := dragonfly.NewComparisonRunner().
	WithVariantNames("da", "bda").
	WithRuns(20).
	WithIterations(200).
	WithTarget(1e-6).
	WithParallel(true).
	WithSeed(4242)

result := runner.Compare("OneMax_30bit", oneMax, 30, 0, 1)
result.PrintComparisonResults()
```

```
================================================================================
Benchmark Comparison: OneMax_30bit
================================================================================

Statistical Summary (base seed 4242):
--------------------------------------------------------------------------------
Variant    |     Mean |   Median |   StdDev |     Best |    Worst |  Rank
--------------------------------------------------------------------------------
DA         | 4.85e+00 | 4.80e+00 | 1.17e+00 | 3.11e+00 | 8.16e+00 |     2
BDA        | 1.50e-01 | 0.00e+00 | 3.57e-01 | 0.00e+00 | 1.00e+00 |     1
--------------------------------------------------------------------------------

Best variant: BDA (rank 1)

Relative quality (lower mean cost is better):
 1. BDA        |########################| mean=0.15
 2. DA         |                        | mean=4.852649718526511

Significant pairwise differences (Wilcoxon signed-rank, alpha=0.05):
--------------------------------------------------------------------------------
DA vs BDA: W=0.0, p=0.0001, winner: BDA

Friedman test (overall difference):
  chi-square = 20.0000, df = 1, p = 0.0000 (significant at alpha=0.05)
================================================================================
```

Success rates on that run: DA 0%, BDA 85%.

## Paired seeds

This is the design decision the whole framework rests on. Run `k` of **every** variant is given
the seed `BaseSeed + k`, so the variants face identical starting swarms and identical random
streams, and the differences that remain are the algorithms'.

Pairing is also what makes the Wilcoxon signed-rank test the right test rather than a
convenience: the two samples are matched observations of the same starting conditions, not
independent draws. On the small run counts a comparison can afford, that is a large gain in
sensitivity.

`ComparisonResult.BaseSeed` records the base, so the entire comparison is reproducible from one
number.

## Configuring a comparison

Every setter returns the runner, so they chain. All are also plain exported fields if you
prefer a struct literal.

| Setter                   | Field           | Default                     | Meaning                                               |
| ------------------------ | --------------- | --------------------------- | ----------------------------------------------------- |
| `WithVariants(v...)`     | `Variants`      | `SingleObjectiveVariants()` | the variants to compare                               |
| `WithVariantNames(n...)` | `Variants`      | —                           | the same, by name or alias                            |
| `WithRuns(n)`            | `Runs`          | `30`                        | runs per variant; must be positive                    |
| `WithIterations(n)`      | `MaxIterations` | `500`                       | iterations per run; must be positive                  |
| `WithTarget(c)`          | `TargetCost`    | unset                       | success threshold; zero and negative values are valid |
| `WithParallel(b)`        | `Parallel`      | `false`                     | run jobs concurrently                                 |
| `WithMaxWorkers(n)`      | `MaxWorkers`    | `runtime.NumCPU()`          | bound on concurrent runs; must be non-negative        |
| `WithSeed(s)`            | `Seed`          | `time.Now().UnixNano()`     | the base seed                                         |
| `WithVerbose(b)`         | `Verbose`       | `false`                     | per-run progress output                               |

Thirty runs is the default because it is the count conventional for statistical significance in
the metaheuristics literature, not because thirty is magic.

Each job starts from `variant.GetConfig()` — the variant's own default configuration — and has
the problem, the iteration count and the paired seed written into it. A binary configuration
keeps its own `[0, 1]` bounds: overwriting them with your continuous bounds would break BDA's
step clamp rather than make it comparable.

### What is rejected

`validate` refuses a nil context, an empty variant list, a nil or unrecognized variant, a
**multi-objective** variant, a non-positive `Runs` or `MaxIterations`, a negative `MaxWorkers`,
a nil objective, a non-positive problem size, and non-finite or inverted bounds.

A multi-objective variant is rejected with `ErrMultiObjectiveVariant` because every statistic
here is defined over a scalar cost. There is no MODA column in a comparison table, and there is
no honest way to invent one.

### Parallelism and objective safety

When `Parallel` is true the objective function **must be safe for concurrent use**: several
runs call it at once. `MaxWorkers` bounds concurrent runs, and `Config.EnableParallel` — which
a variant's default config may or may not set — is an independent inner limit.

Parallelism changes only the order runs complete in, never which seed a run gets, so a parallel
comparison is bit-identical to a sequential one.

## Running

```go
// Records a failing run in its RunResult and keeps going.
result := runner.Compare(name, fn, problemSize, lower, upper)

// Aborts on the first failure and returns no partial aggregate.
result, err := runner.CompareContext(ctx, name, fn, problemSize, lower, upper)
```

`Compare` is the forgiving form: a run that errors is recorded with `BestCost = +Inf` and its
error message in `RunResult.Error`, and the comparison completes. `CompareContext` is the strict
one: it returns no aggregate when any run fails, so a caller cannot mistake a broken comparison
for a completed one, and it honours cancellation.

## Reading the result

```go
type ComparisonResult struct {
	FriedmanResult *FriedmanTestResult   // nil when fewer than two variants
	BenchmarkName  string
	AlgorithmNames []string
	RunResults     [][]RunResult         // [variant][run]
	Statistics     []AlgorithmStatistics // per variant
	Rankings       []int                 // rank by mean cost, 1 is best
	WilcoxonTests  [][]WilcoxonResult    // pairwise; the diagonal is zero
	BestAlgorithm  int                   // index of the rank-1 variant, or -1
	BaseSeed       int64
}
```

### Per-run

```go
type RunResult struct {
	Error         string  // empty when the run succeeded
	BestCost      float64
	ExecutionTime float64 // seconds
	Seed          int64
	FuncEvals     int
	Iterations    int
	ConvergenceAt int // one-based iteration where TargetCost was first reached, or 0
}
```

### Per-variant

```go
type AlgorithmStatistics struct {
	Mean, Median, StdDev, Best, Worst float64
	SuccessRate                       float64 // percent of runs at or below TargetCost
	AvgFuncEvals, AvgTime             float64
	SuccessfulRuns, FailedRuns        int
}
```

`StdDev` is the population standard deviation. `SuccessRate` is zero when no target was
configured. Failed or non-finite runs are counted but excluded from numerical statistics and
paired tests. Rankings are by **mean cost** over successful runs; variants with none rank last.

## The statistics

### Wilcoxon signed-rank, pairwise

```go
type WilcoxonResult struct {
	Algorithm1, Algorithm2 string
	Winner                 string  // the lower-cost variant, or "Tie"
	WStatistic             float64 // min(W+, W-)
	PValue                 float64 // two-tailed
	AdjustedPValue         float64 // Holm correction across available pairs
	Pairs                  int
	Significant, Available bool
}
```

A two-tailed paired test on successful matched runs. Zero differences are dropped and ranks
recomputed over what remains. Up to 20 non-zero pairs use exact sign enumeration; larger
samples use a continuity- and tie-corrected normal approximation. Pairwise significance uses
the Holm-adjusted p-value.

`Winner` is `"Tie"` whenever the difference is not significant, so a non-significant result
never reads as a win.

### Friedman, across all variants

```go
type FriedmanTestResult struct {
	ChiSquare        float64
	PValue           float64
	DegreesOfFreedom int  // k - 1
	Blocks           int  // complete successful paired runs
	Significant      bool // PValue < 0.05
	Available        bool
}
```

The tie-corrected non-parametric analogue of a repeated-measures ANOVA, over complete per-run
ranks:

```
chi² = 12 / (n·k·(k+1)) · Σ R_j²  −  3·n·(k+1)
```

with `k` variants, `n` runs and `R_j` the sum of variant `j`'s ranks. Tied values get their
average rank. Under the null hypothesis `chi²` follows a chi-square distribution with `k−1`
degrees of freedom, and the p-value is its upper tail.

`FriedmanResult` is nil when fewer than two variants were compared. With exactly two it is
reported, but the pairwise Wilcoxon test is the one to read.

Both tests use `alpha = 0.05`.

## Reporting and export

```go
result.PrintComparisonResults()                 // to stdout
err := result.WriteComparisonResults(os.Stderr) // to any io.Writer

err = result.ExportToCSV("comparison.csv")
err = result.ExportToJSON("comparison.json")
```

The report has three sections: the statistical summary table, a relative-quality bar chart
scaled between the best and worst finite means, and the significant pairwise differences
followed by the Friedman result. A variant whose mean is not finite — every run failed — gets a
blank bar labelled `failed/unavailable` rather than being dropped.

`ExportToCSV` writes one row per run, in variant then run order, with the variant's aggregate
statistics repeated on every row so the file is usable without a join:

```
benchmark, variant, rank, run, seed, best_cost, function_evaluations, iterations,
convergence_at, execution_seconds, error, mean, median, stddev, best, worst,
success_rate, avg_function_evaluations, avg_execution_seconds, successful_runs, failed_runs
```

Unavailable CSV costs are empty and unavailable JSON costs are `null`. Every export is written
to a same-directory temporary file and atomically renamed only after encoding succeeds.

`ExportToJSON` writes the whole `ComparisonResult` as an indented document, including every
`RunResult` and both test results.

Both exporters validate the result's shape first, so a result whose parallel slices disagree
produces an error rather than an index panic inside the writer.

## Comparing something that is not a bundled variant

`ComparisonRunner` works against the `AlgorithmVariant` interface, so anything implementing it
can be compared — including two configurations of the same algorithm, which is the usual way to
justify a tuning change:

```go
type tunedDA struct{ dragonfly.DAVariant }

func (v *tunedDA) Name() string { return "DA-tuned" }

func (v *tunedDA) GetConfig() *dragonfly.Config {
	config := dragonfly.NewDefaultConfig()
	config.MaxStepRatio = 0.05
	config.RadiusGrowth = 1.0

	return config
}

result := dragonfly.NewComparisonRunner().
	WithVariants(&dragonfly.DAVariant{}, &tunedDA{}).
	WithRuns(30).
	Compare("Rosenbrock", dragonfly.Rosenbrock, 10, -5, 10)
```

Note that `ComparisonRunner` overwrites `ObjectiveFunc`, `ProblemSize`, `MaxIterations`, `Rand`
and (for non-binary configs) the bounds on whatever `GetConfig` returns. Everything else your
config sets survives, which is exactly what makes this work.

## Related documentation

- [API Quick Reference](quick-reference.md) — the task-oriented map
- [Configuration Guide](configuration.md) — the fields a variant's `GetConfig` returns
- [Standard DA](../algorithms/standard-da.md), [BDA](../algorithms/bda.md),
  [MODA](../algorithms/moda.md) — what each variant is for
- [Benchmark Functions](../benchmarks.md) — problems worth comparing on
