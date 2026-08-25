# API Quick Reference

A task-oriented map of `package dragonfly`. Everything lives in one flat package at the
repository root; there is no `internal/`, no `pkg/` and no sub-package to import.

```go
import "github.com/CWBudde/dragonfly"
```

For the field-by-field reference see the [Configuration Guide](configuration.md); for
observers, cancellation and export see the [Run Lifecycle](run-lifecycle.md).

## Minimal optimization

```go
config := dragonfly.NewDefaultConfig()
config.ObjectiveFunc = dragonfly.Sphere
config.ProblemSize = 10
config.LowerBound = -10
config.UpperBound = 10
config.MaxIterations = 1000

result, err := dragonfly.Optimize(config)
if err != nil {
	log.Fatal(err)
}

fmt.Printf("best cost: %.6g\n", result.GlobalBest.Cost)
```

`ObjectiveFunc`, `ProblemSize`, `LowerBound` and `UpperBound` have no defaults. Everything else
comes from the factory.

## Entry points

| Function                                                               | Runs | Returns          |
| ---------------------------------------------------------------------- | ---- | ---------------- |
| `Optimize(config) (*Result, error)`                                    | DA   | one incumbent    |
| `OptimizeContext(ctx, config, options...) (*Result, error)`            | DA   | one incumbent    |
| `OptimizeBinary(config) (*Result, error)`                              | BDA  | one bit string   |
| `OptimizeBinaryContext(ctx, config, options...) (*Result, error)`      | BDA  | one bit string   |
| `OptimizeMultiObjective(ctx, moConfig) (*MultiObjectiveResult, error)` | MODA | a Pareto archive |

`Optimize` is `OptimizeContext` with `context.Background()`. A cancelled run returns a nil
result and `ctx.Err()`: partial results are deliberately not reported, so a caller cannot
mistake an aborted run for a completed one. A nil context is an error, not a panic.

`Optimize` ignores `Config.UseBinary` — run BDA through the binary entry points, or through the
variant layer, which rejects the mismatch.

## Configuration factories

| Function                     | What it layers on                                                 |
| ---------------------------- | ----------------------------------------------------------------- |
| `NewDefaultConfig()`         | the paper's continuous DA, every weight on its schedule           |
| `NewHighDimensionalConfig()` | `NPop` 100, 3000 iterations, `RadiusGrowth` 1.0                   |
| `NewFastConvergenceConfig()` | `NPop` 30, 300 iterations, `RadiusGrowth` 4.0, `MaxStepRatio` 0.2 |
| `NewBinaryConfig()`          | bounds `[0,1]`, `v3` transfer, `MaxStepRatio` 6.0, `UseBinary`    |
| `NewMultiObjectiveConfig()`  | a `MultiObjectiveConfig` wrapping `NewDefaultConfig()`            |
| `NewPresetConfig(preset)`    | any of the above by name                                          |

Preset names: `PresetDefault`, `PresetHighDimensional`, `PresetFastConvergence`,
`PresetBinary`. `ListPresets()` returns them with one-line descriptions, `PresetNames()` in
alphabetical order, `PrintPresets()` writes the table to stdout.

## Run options

Passed variadically to `OptimizeContext` and `OptimizeBinaryContext`. Observers receive deep
copies and run synchronously on the caller's goroutine.

```go
result, err := dragonfly.OptimizeContext(ctx, config,
	dragonfly.WithInitialPopulation([][]float64{{1, 1, 1, 1, 1}}),
	dragonfly.WithProgressObserver(func(p dragonfly.Progress) { /* ... */ }),
	dragonfly.WithPopulationObserver(func(s dragonfly.PopulationSnapshot) { /* ... */ }),
	dragonfly.WithLogger(slog.Default()),
)
```

| Option                             | Effect                                                                  |
| ---------------------------------- | ----------------------------------------------------------------------- |
| `WithInitialPopulation(positions)` | seeds the leading swarm slots; the rest stay random                     |
| `WithProgressObserver(fn)`         | one `Progress` per completed iteration                                  |
| `WithPopulationObserver(fn)`       | one `PopulationSnapshot` per iteration — the whole swarm, deep-copied   |
| `WithArchiveObserver(fn)`          | one `ArchiveSnapshot` per iteration — **multi-objective runs only**     |
| `WithLogger(logger)`               | `optimization_started`, `iteration_completed`, `optimization_completed` |

Passing `nil` to any observer option disables it.

The last two rows are mutually exclusive by problem class, and an option used
on the wrong one is **rejected rather than silently ignored**:
`OptimizeContext` and `OptimizeBinaryContext` refuse `WithArchiveObserver`,
and `OptimizeMultiObjective` refuses `WithProgressObserver`,
`WithPopulationObserver` and `WithLogger` — a Pareto run has no single
incumbent for any of them to report. `WithInitialPopulation` works on all
three.

## Result fields and exports

```go
type Result struct {
	ConvergenceCurve  []float64         // best cost after each completed iteration
	TerminationReason TerminationReason // maximum_iterations | target_cost | stagnation
	GlobalBest        Best              // the food source
	Worst             Best              // the enemy — no Mayfly counterpart
	FuncEvalCount     int
	IterationCount    int
	Seed              int64
}

type Best struct {
	Position            []float64
	Cost                float64
	ConstraintViolation float64
}
```

`ConvergenceCurve` is a history of costs, not a point in the search space; the solution is
`GlobalBest.Position`. It has `IterationCount` entries and is non-increasing for unconstrained
optimization — under constraints a raw cost may rise when feasibility takes priority.

```go
_ = result.ExportConvergenceCSV("convergence.csv")   // iteration, best_cost
_ = result.ExportConvergenceJSON("convergence.json") // curve + run summary
```

`MultiObjectiveResult` carries `Archive`, `ArchiveSizeCurve`, `TerminationReason`,
`FuncEvalCount`, `IterationCount` and `Seed`, with `ExportParetoCSV` and `ExportParetoJSON`.

## Builder

```go
result, err := dragonfly.NewBuilder("da").
	ForProblem(dragonfly.Rastrigin, 30, -5.12, 5.12).
	WithIterations(500).
	WithPopulation(40).
	WithConfig(func(c *dragonfly.Config) { c.MaxStepRatio = 0.05 }).
	Optimize()
```

| Method                                         | Effect                                                          |
| ---------------------------------------------- | --------------------------------------------------------------- |
| `NewBuilder(name)`                             | by variant name or alias; an unknown name surfaces from `Build` |
| `NewBuilderFromVariant(variant)`               | from an `AlgorithmVariant` instance                             |
| `ForProblem(fn, size, lower, upper)`           | objective, dimensionality, bounds (bounds ignored for BDA)      |
| `WithIterations(n)` / `WithPopulation(n)`      | run length and swarm size                                       |
| `WithConfig(func(*Config))`                    | any other edit                                                  |
| `Build() (*Config, error)`                     | validated config, or the first recorded error                   |
| `Optimize()` / `OptimizeContext(ctx, opts...)` | build and run                                                   |
| `GetVariant()`                                 | the variant, or nil if the name was not recognized              |

The builder carries only the single-objective `Config`; use
`MODAVariant.GetMultiObjectiveConfig()` for MODA.

## Variant registry

```go
variant, err := dragonfly.NewVariant("bda")   // "da"/"standard", "bda"/"binary", "moda"
all := dragonfly.GetAllVariants()             // canonical order: DA, BDA, MODA
single := dragonfly.SingleObjectiveVariants() // DA, BDA
names := dragonfly.ListVariants()             // ["DA" "BDA" "MODA"]
aliases := dragonfly.VariantAliases()          // alphabetical, including aliases
```

`AlgorithmVariant` exposes `Name`, `FullName`, `Description`, `GetConfig`, `IsMultiObjective`,
`Run`, `ApplicableTo`, `EstimatedOverhead` and `RecommendedFor`. Two sentinel errors matter:

- `ErrMultiObjectiveVariant` — `MODAVariant.Run` cannot honour the single-objective contract
- `ErrBinaryConfigOnContinuousVariant` — `DAVariant.Run` refuses a config with `UseBinary` set

## Variant selection

```go
characteristics := dragonfly.ClassifyProblem(fn, 30, -5.12, 5.12, rng)

selector := dragonfly.NewAlgorithmSelector()          // or NewAlgorithmSelectorFor(...)
best := selector.RecommendBest(characteristics)        // one AlgorithmRecommendation
ranked := selector.RecommendAlgorithms(characteristics) // all, best first, stable ties

dragonfly.PrintRecommendations(ranked)
```

`ProblemCharacteristics` carries `Dimensionality`, `Modality` (`Unimodal`, `Multimodal`,
`HighlyMultimodal`), `Landscape` (`Smooth`, `Rugged`, `Deceptive`, `NarrowValley`), and the
booleans `Discrete`, `ExpensiveEvaluations`, `RequiresFastConvergence`,
`RequiresStableConvergence`, `MultiObjective`.

`ClassifyProblem` fills in only the first two enums, `Dimensionality` and
`RequiresStableConvergence`. It never reports `Deceptive` or `NarrowValley` — those are claims
about where the optimum sits relative to the terrain, which a few dozen samples cannot
establish. Set them yourself if you know.

`AlgorithmRecommendation` carries `Variant`, `Reason` (never empty), `Preset`, `Score` and
`Confidence`. `RecommendPreset(characteristics)` gives just the preset;
`RecommendForBenchmark(name)` reads a hand-classified table instead of sampling.
`BenchmarkCharacteristics(name)` and `BenchmarkNames()` expose that table.

## Comparison framework

```go
runner := dragonfly.NewComparisonRunner().
	WithVariantNames("da", "bda").
	WithRuns(30).
	WithIterations(500).
	WithTarget(1e-3).
	WithParallel(true).
	WithSeed(4242)

result := runner.Compare("Sphere", dragonfly.Sphere, 10, -10, 10)
// or: result, err := runner.CompareContext(ctx, "Sphere", dragonfly.Sphere, 10, -10, 10)

result.PrintComparisonResults()
_ = result.ExportToCSV("comparison.csv")
_ = result.ExportToJSON("comparison.json")
```

See [comparison-framework.md](comparison-framework.md) for the statistics and the result shape.

## Constraints and convergence

```go
config.Constraints = &dragonfly.ConstraintConfig{
	Handling:          dragonfly.ConstraintHandlingFeasibility, // or ...Penalty
	Inequalities:      []dragonfly.ConstraintFunction{ /* g(x) <= 0 */ },
	Equalities:        []dragonfly.ConstraintFunction{ /* h(x) = 0 */ },
	EqualityTolerance: 1e-6,
	PenaltyMethod:     dragonfly.PenaltyQuadratic, // or PenaltyLinear
	PenaltyFactor:     1e3,                        // required under penalty handling
}

target := 1e-6
config.Convergence = &dragonfly.ConvergenceConfig{
	TargetCost:           &target, // pointer: distinguishes "disabled" from "zero"
	MinImprovement:       1e-9,
	StagnationIterations: 50,
	MinIterations:        10,
}
```

Helpers: `EvaluateConstraints(position, config)`, `IsFeasible(violation)`,
`PenalizedCost(cost, violation, factor, method)`,
`BetterConstrainedCandidate(candidate, incumbent, config)`.

## Configuration files and validation

```go
_ = dragonfly.SaveConfig(config, "config.json")
loaded, err := dragonfly.LoadConfig("config.json")
loaded.ObjectiveFunc = myObjective // ObjectiveFunc, Rand and constraint funcs are json:"-"

err = dragonfly.ValidateConfig(loaded) // the same checks Optimize runs

dragonfly.AutoTuneConfig(config) // coarse NPop / MaxIterations / RadiusGrowth heuristics
```

Write configuration files with `SaveConfig` rather than hand-authoring a partial one: absent
JSON fields decode as Go zero values, and `0` is a legitimate pinned weight rather than a
request for the adaptive schedule.

## Bundled objective functions

**Single-objective** — `func([]float64) float64`, all minimization:
`Sphere`, `Rastrigin`, `Rosenbrock`, `Ackley`, `Griewank`, `Schwefel`, `Levy`, `Zakharov`,
`Michalewicz`, `DixonPrice`, `BentCigar`, `Discus`, `Weierstrass`, `HappyCat`,
`ExpandedSchafferF6`, `Himmelblau`

**Multi-objective** — `func([]float64) []float64`, all objectives minimized:
`ZDT1`, `ZDT2`, `ZDT3`, `SchafferN1`

Bounds, optima and measured results are in [../benchmarks.md](../benchmarks.md).

### CEC2017 and CEC2020 competition cases

```go
data := os.DirFS("/path/to/official/input_data")
problem, err := dragonfly.NewCEC2020Problem(data, 8, 10)
config, err := problem.NewConfig(nil) // normalized [0,1]^10 search
result, err := dragonfly.Optimize(config)
physicalBest, err := problem.Decode(result.GlobalBest.Position)
```

`CEC2017Suite(data, dimension)` loads all 29 usable functions; `CEC2020Suite` loads all ten.
Each `BenchmarkCase` exposes its physical bounds and optimum, biased minimum, function number,
competition evaluation budget, validated evaluator and normalized configuration adapter.
Organizer transformation files remain external to the module. See
[../benchmarks.md](../benchmarks.md#official-cec2017-and-cec2020-suites) for supported
dimensions, data layout and evaluator-compatibility notes.

## Binary helpers

```go
transfer, err := dragonfly.LookupTransferFunction(dragonfly.TransferV3)
names := dragonfly.TransferFunctionNames() // v1..v4, s1..s4, stable order
ok := dragonfly.BinaryPositionsValid(result.GlobalBest.Position)
```

Constants: `TransferV1`…`TransferV4`, `TransferS1`…`TransferS4`, `DefaultTransferFunction`
(= `TransferV3`).

## Pareto archive

```go
archive := dragonfly.NewParetoArchive(100)
// or with explicit grid parameters:
archive = dragonfly.NewParetoArchiveWithGrid(100, 10, 4, 2, 2)

accepted := archive.Add(&dragonfly.ParetoSolution{
	Position:        position,
	ObjectiveValues: objectives,
}, rng)

count := archive.UpdateFromPopulation(candidates, rng)
n := archive.Len()
ok := archive.IsNonDominated()
```

`rng` is the last parameter by package convention and is used only for the overflow eviction; a
nil `rng` makes that eviction deterministic.

## Sentinels and named constants

| Name                                                                           | Value                                                      |
| ------------------------------------------------------------------------------ | ---------------------------------------------------------- |
| `WeightAuto`                                                                   | `-1.0` — "use the paper's schedule"                        |
| `FidelityPaper` / `FidelityMATLAB`                                             | `"paper"` / `"matlab"`                                     |
| `BoundaryWrap` / `BoundaryClamp` / `BoundaryReflect`                           | `"wrap"` / `"clamp"` / `"reflect"`                         |
| `TerminationMaxIterations` / `TerminationTargetCost` / `TerminationStagnation` | `"maximum_iterations"` / `"target_cost"` / `"stagnation"`  |
| `ConstraintHandlingFeasibility` / `ConstraintHandlingPenalty`                  | `"feasibility"` / `"penalty"`                              |
| `PenaltyLinear` / `PenaltyQuadratic`                                           | `"linear"` / `"quadratic"`                                 |
| `DefaultLevyBeta` / `DefaultLevyScale`                                         | `1.5` / `0.01`                                             |
| `ArchivePolicyPaperSegments` / `MATLABDensity` / `MOPSOGrid`                   | paper default / reference compatibility / legacy extension |
| `DefaultArchiveBeta` / `Gamma` / `Delta` / `NGrid` / `Size`                    | `4` / `2` / `2` / `10` / `100`; exponents are MOPSO-only   |

## Related documentation

- [Configuration Guide](configuration.md) — every field and how it is resolved
- [Run Lifecycle](run-lifecycle.md) — cancellation, observers, logging, export
- [Comparison Framework](comparison-framework.md) — statistics and reporting
- [Algorithms](../algorithms/) — what each variant actually does
