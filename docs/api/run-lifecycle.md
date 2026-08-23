# Run Lifecycle API

How to start a run, watch it, stop it and keep what it produced. Everything here is a
`RunOption` passed variadically to `OptimizeContext` or `OptimizeBinaryContext`, plus the
exporters on `Result` and `MultiObjectiveResult`.

```go
result, err := dragonfly.OptimizeContext(ctx, config, options...)
```

`Optimize` and `OptimizeBinary` are the same calls with `context.Background()` and no options.
`OptimizeMultiObjective` takes a context but **no** run options: observers, logging and initial
populations are not wired into the multi-objective path. It does honour the shared `Config`
block — `Constraints`, `Convergence` and `EnableParallel` all apply, with the two exceptions
[MODA documents](../algorithms/moda.md): `Convergence.TargetCost` and
`ConstraintHandlingPenalty` are rejected rather than ignored.

## Cancellation

Cancellation is checked at the top of every iteration.

```go
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

result, err := dragonfly.OptimizeContext(ctx, config)
if errors.Is(err, context.DeadlineExceeded) {
	// the run did not finish
}
```

A cancelled run returns a **nil result** and `ctx.Err()`. Partial results are deliberately not
reported, so a caller cannot mistake an aborted run for a completed one. If you want the best
answer found before the deadline, record it from a progress observer as the run goes.

A nil context is an error (`context cannot be nil`), not a panic from the first cancellation
check.

Under `EnableParallel`, a batch can also be cancelled between objective calls. Nothing is
committed in that case: the swarm carries either every cost from this iteration or every cost
from the previous one, never a mixture.

## Progress observers

```go
result, err := dragonfly.OptimizeContext(ctx, config,
	dragonfly.WithProgressObserver(func(p dragonfly.Progress) {
		fmt.Printf("iteration %3d: best %.6g after %d evaluations\n",
			p.Iteration, p.Best.Cost, p.EvaluationCount)
	}),
)
```

```go
type Progress struct {
	Best            Best // deep copy — retain or modify it freely
	Iteration       int  // one-based
	EvaluationCount int
}
```

Observers run **synchronously on the calling goroutine**, once per completed iteration, and
receive deep copies. They must not draw random numbers or reach back into the optimizer: a
seeded run is required to be reproducible, and an observer that mutated shared state would be a
back door around that.

Passing a nil observer disables progress reporting.

## Population observers

```go
dragonfly.WithPopulationObserver(func(s dragonfly.PopulationSnapshot) {
	spread := diversity(s.Swarm)          // safe: every dragonfly is a deep copy
	fmt.Printf("%d: spread %.3f, enemy %.3g\n", s.Iteration, spread, s.Worst.Cost)
})
```

```go
type PopulationSnapshot struct {
	Swarm           []Dragonfly // deep copies, Position and Step both
	Best            Best        // the food source
	Worst           Best        // the enemy the swarm is repelled from
	Iteration       int         // one-based
	EvaluationCount int
}
```

This is a separate option from `WithProgressObserver`, not an extension of it, because copying
`NPop` position and step vectors once per iteration is not free. The measured cost on a 30-D
Sphere over 100 iterations with `NPop` 40: 24,927 allocations without any observer, 25,027 with
a progress observer, 33,227 with a population observer — about 83 extra allocations and 22 KB
per iteration. No copying happens unless an observer is registered.

`Worst` is carried because every step of the algorithm is computed against the enemy, so a
snapshot without it cannot explain a move. It has no Mayfly counterpart.

Population observers run after progress observers within the same iteration.

## Archive observers — multi-objective runs

`OptimizeMultiObjective` accepts run options too, and one of its own:

```go
result, err := dragonfly.OptimizeMultiObjective(ctx, config,
	dragonfly.WithArchiveObserver(func(s dragonfly.ArchiveSnapshot) {
		fmt.Printf("%d: %d members in %d objectives\n",
			s.Iteration, len(s.Solutions), len(s.GridLower))
	}),
)
```

```go
type ArchiveSnapshot struct {
	Solutions       []*ParetoSolution // deep copies
	GridLower       []float64         // copies of the archive's per-objective extent
	GridUpper       []float64
	Iteration       int               // one-based
	EvaluationCount int
	NGrid           int               // bins per objective
}
```

`GridLower`, `GridUpper` and `NGrid` are the frame each solution's `GridIndex` is expressed in:
bin _b_ of objective _m_ spans one `NGrid`-th of `[GridLower[m], GridUpper[m]]`. They move as
the archive does, and all three are zero for an empty archive. The same extent is available
after the fact from `ParetoArchive.GridBounds()`, which likewise returns copies — the archive
rewrites its bounds through the existing backing arrays on every mutation.

Copying the archive is not free either, and as with `PopulationSnapshot` nothing is copied
unless an observer is registered.

### Which options a multi-objective run accepts

`WithProgressObserver`, `WithPopulationObserver` and `WithLogger` are **rejected**, not ignored:

```go
_, err := dragonfly.OptimizeMultiObjective(ctx, config,
	dragonfly.WithProgressObserver(func(dragonfly.Progress) {}))
// err: WithProgressObserver has no meaning for a multi-objective run: a Pareto front has
//      no single best cost; use WithArchiveObserver
```

A Pareto front has no incumbent, so `Progress.Best` and `PopulationSnapshot.Best`/`Worst` have
nothing honest to report — MODA's food source and enemy are per-iteration roulette draws from
the archive, not running bests. The logger's iteration and completion events report a single
best cost and a `*Result`, neither of which a multi-objective run has. This is the rule
`validateMultiObjectiveConfig` already applies to `Convergence.TargetCost` and
`ConstraintHandlingPenalty`, applied one layer out: a caller who registers an observer is
waiting for something, and a run that quietly never calls it is worse than one that refuses to
start.

Symmetrically, `OptimizeContext` and `OptimizeBinaryContext` reject `WithArchiveObserver`.

`WithInitialPopulation` works on all three entry points. A nil observer registers nothing and
is never rejected.

## Structured logging

```go
logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))

result, err := dragonfly.OptimizeContext(ctx, config, dragonfly.WithLogger(logger))
```

`*slog.Logger` satisfies the `Logger` interface directly:

```go
type Logger interface {
	Log(ctx context.Context, level slog.Level, message string, args ...any)
}
```

Three events are emitted, all at `slog.LevelInfo`:

| `event`                  | When                       | Attributes                                                                                           |
| ------------------------ | -------------------------- | ---------------------------------------------------------------------------------------------------- |
| `optimization_started`   | before the first iteration | `problem_size`, `max_iterations`, `population`, `parallel`                                           |
| `iteration_completed`    | once per iteration         | `iteration`, `evaluations`, `best_cost`, `constraint_violation`                                      |
| `optimization_completed` | after the last iteration   | `iterations`, `evaluations`, `best_cost`, `constraint_violation`, `worst_cost`, `termination_reason` |

`iteration_completed` fires every iteration, so on a 1000-iteration run at info level you get
1000 lines. Filter with a handler level or a `slog` group rather than by not passing a logger,
if you want the start and end events but not the middle. Passing `nil` disables logging
entirely.

Loggers, like observers, run synchronously on the calling goroutine.

## Early termination

```go
target := 1e-4
config.Convergence = &dragonfly.ConvergenceConfig{
	TargetCost:           &target,
	MinImprovement:       1e-9,
	StagnationIterations: 20,
	MinIterations:        5,
}
```

`Result.TerminationReason` reports which criterion ended the run: `maximum_iterations`,
`target_cost` or `stagnation`. The full semantics are in the
[Configuration Guide](configuration.md#convergence-detection). Two things worth repeating here:
the target-cost stop refuses to fire on an infeasible incumbent, and `MaxIterations` remains the
hard upper bound whatever the convergence block says.

MODA reports `maximum_iterations` or `stagnation`. It has no target-cost stop: a Pareto front
has no single best cost, so `Convergence.TargetCost` is rejected by validation rather than
ignored.

## Seeding the initial population

```go
seeds := [][]float64{
	{1, 1, 1, 1, 1},
	{-1, 2, -3, 4, -5},
}

result, err := dragonfly.OptimizeContext(ctx, config,
	dragonfly.WithInitialPopulation(seeds))
```

The argument may contain fewer positions than `NPop`; unfilled slots are initialized randomly.
Positions are copied when the option is constructed and again when it is applied, so the caller
keeps ownership of the slices.

Only the **positions** of the leading slots are replaced. The step draw and every slot beyond
the supplied positions are left alone, so seeding the first few dragonflies does not shift the
random stream the rest of the swarm is built from.

Rejected with an error: more positions than `NPop`; a position whose length is not
`ProblemSize`; a non-finite component; a component outside `[LowerBound, UpperBound]`. In binary
mode, a component that is not exactly `0` or `1` — rounding it silently would hand you a
different starting swarm than the one you wrote.

## Reading the result

```go
type Result struct {
	ConvergenceCurve  []float64
	TerminationReason TerminationReason
	GlobalBest        Best
	Worst             Best
	FuncEvalCount     int
	IterationCount    int
	Seed              int64
}
```

`ConvergenceCurve` holds the best cost known at the end of each completed iteration, so it has
`IterationCount` entries. It is a history of costs, not a point in the search space — the
solution is `GlobalBest.Position`. It is non-increasing for unconstrained optimization; under
constraints a raw cost may rise when feasibility or a lower violation takes priority.

`FuncEvalCount` counts every call to `ObjectiveFunc`, including the ones made while initializing
the swarm, so a complete run of `T` iterations reports `NPop × (T + 1)`.

`Worst` is the enemy: the worst position seen during the run, which the enemy term of every step
was computed against. It is reported for inspection.

## Convergence export

```go
if err := result.ExportConvergenceCSV("convergence.csv"); err != nil {
	log.Fatal(err)
}
if err := result.ExportConvergenceJSON("convergence.json"); err != nil {
	log.Fatal(err)
}
```

The CSV has two columns, `iteration` (one-based) and `best_cost`, one row per completed
iteration. An empty curve yields a header-only file rather than an error.

The JSON document is a `ConvergenceExport`: the curve as `{iteration, best_cost}` objects, plus
`best_position`, `worst_position`, `best_cost`, `best_constraint_violation`, `worst_cost`,
`worst_constraint_violation`, `termination_reason`, `seed`, `func_eval_count` and
`iteration_count`. The enemy is a run-level value with no per-iteration column, which is why it
appears in the JSON and not in the CSV.

Both exporters report a close failure that would otherwise hide a short write.

## Pareto front export

```go
result, err := dragonfly.OptimizeMultiObjective(ctx, moConfig)

_ = result.ExportParetoCSV("front.csv")
_ = result.ExportParetoJSON("front.json")
```

The CSV has an `index` column, one `objective_k` column per objective, one `x_j` column per
decision variable and a trailing `constraint_violation` column, one row per archived solution in
archive order. The objective and variable counts follow the archive's contents, so an empty
archive yields a header-only file with just the fixed columns.

The JSON document adds the run summary: `termination_reason`, `seed`, `archive_size`,
`func_eval_count`, `iteration_count` and `archive_size_curve`.

## A complete example

```go
config := dragonfly.NewDefaultConfig()
config.ObjectiveFunc = dragonfly.Sphere
config.ProblemSize = 5
config.LowerBound = -10
config.UpperBound = 10
config.MaxIterations = 200
config.NPop = 20
config.Rand = rand.New(rand.NewSource(3))

target := 1e-4
config.Convergence = &dragonfly.ConvergenceConfig{
	TargetCost:           &target,
	MinImprovement:       1e-9,
	StagnationIterations: 20,
	MinIterations:        5,
}

ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
	Level: slog.LevelWarn, // start and end events only would need a filtering handler
}))

result, err := dragonfly.OptimizeContext(ctx, config,
	dragonfly.WithInitialPopulation([][]float64{{1, 1, 1, 1, 1}, {-1, 2, -3, 4, -5}}),
	dragonfly.WithProgressObserver(func(p dragonfly.Progress) {
		if p.Iteration%10 == 0 {
			fmt.Printf("  iteration %3d: best %.6g after %d evaluations\n",
				p.Iteration, p.Best.Cost, p.EvaluationCount)
		}
	}),
	dragonfly.WithPopulationObserver(func(s dragonfly.PopulationSnapshot) {
		_ = s.Swarm
		_ = s.Worst
	}),
	dragonfly.WithLogger(logger),
)
if err != nil {
	log.Fatal(err)
}

fmt.Printf("stopped after %d iterations: %s\n", result.IterationCount, result.TerminationReason)
fmt.Printf("best %.6g, enemy %.6g\n", result.GlobalBest.Cost, result.Worst.Cost)

_ = result.ExportConvergenceCSV("convergence.csv")
_ = result.ExportConvergenceJSON("convergence.json")
```

Output on this seed:

```
  iteration  10: best 2.62699 after 220 evaluations
  iteration  20: best 2.62699 after 420 evaluations
  iteration  30: best 1.87873 after 620 evaluations
  iteration  40: best 1.87873 after 820 evaluations
stopped after 43 iterations: stagnation
best 1.87873, enemy 285.424
```

## Related documentation

- [API Quick Reference](quick-reference.md) — the task-oriented map
- [Configuration Guide](configuration.md) — convergence, constraints, parallelism, RNG
- [Comparison Framework](comparison-framework.md) — running many seeds at once
- [MODA](../algorithms/moda.md) — what the multi-objective path does and does not support
