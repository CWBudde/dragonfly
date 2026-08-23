# Dragonfly Algorithm (Go)

A dependency-free Go implementation of the **Dragonfly Algorithm (DA)**, the swarm
metaheuristic Seyedali Mirjalili introduced in 2016. It models the static and dynamic
swarming behaviour of dragonflies — separation, alignment, cohesion, attraction to food and
distraction from enemies — and covers all three variants from the original paper.

[![Go Reference](https://pkg.go.dev/badge/github.com/CWBudde/Dragonfly.svg)](https://pkg.go.dev/github.com/CWBudde/Dragonfly)

Sibling project: [Mayfly](https://github.com/cwbudde/mayfly), which shares this library's API
style, tooling and conventions.

## Overview

**Key features:**

- **Three variants** — continuous `DA`, binary `BDA`, multi-objective `MODA`
- **Standard library only** — the sole direct dependency is `godog`, and it is test-only
- **Deterministic** — a seeded run reproduces bit-for-bit, with parallel evaluation on or off
- **Paper-faithful** — the two-branch step update, the per-dimension neighbourhood test and the
  enemy sum are implemented as the reference `DA.m` computes them, with the deviations
  [documented](#deviations-from-the-reference-matlab) rather than quietly smoothed over
- **Constraint handling** — Deb's feasibility rules, or linear/quadratic penalties
- **Observable** — progress and population observers, `log/slog` integration, CSV/JSON export
- **Benchmark suite** — 15 single-objective and 4 multi-objective test functions
- **Statistical comparison** — paired-seed runs with Wilcoxon signed-rank and Friedman tests

**What DA is good at, and what it is not.** DA explores well: the growing neighbourhood radius
and the Lévy walk for isolated dragonflies keep the swarm spread out far into the run. It
exploits poorly. The shared convergence factor `mc` reaches zero at the halfway point, after
which only the food term and inertia still move a dragonfly, so the paper's algorithm stalls
well short of the optimum on hard landscapes. The numbers in
[docs/benchmarks.md](docs/benchmarks.md) are what a faithful port actually produces, not what a
well-tuned optimizer could.

## Quick Start

### Installation

```sh
go get github.com/CWBudde/Dragonfly
```

### Basic usage

```go
package main

import (
	"fmt"
	"log"

	dragonfly "github.com/CWBudde/Dragonfly"
)

func main() {
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

	fmt.Printf("best cost:   %.6g\n", result.GlobalBest.Cost)
	fmt.Printf("iterations:  %d\n", result.IterationCount)
	fmt.Printf("evaluations: %d\n", result.FuncEvalCount)
	fmt.Printf("terminated:  %s\n", result.TerminationReason)
	fmt.Printf("seed:        %d\n", result.Seed)
}
```

`ObjectiveFunc`, `ProblemSize`, `LowerBound` and `UpperBound` have no usable defaults; always
start from a factory function and set those four. Typical output on this problem is a best cost
around `1` — see the note on exploitation above.

### Custom objective function

The library minimizes. For a maximization problem, negate the objective:

```go
func profit(x []float64) float64 {
	return 10 - (x[0]-2)*(x[0]-2) - (x[1]+1)*(x[1]+1)
}

func maximizeProfit(x []float64) float64 { return -profit(x) }

config := dragonfly.NewDefaultConfig()
config.ObjectiveFunc = maximizeProfit
config.ProblemSize = 2
config.LowerBound = -5
config.UpperBound = 5

result, _ := dragonfly.Optimize(config)
fmt.Printf("best profit: %.6f\n", -result.GlobalBest.Cost)
```

### Constrained optimization

Inequalities are satisfied when `g(x) <= 0`; equalities when `|h(x)| <= EqualityTolerance`.
Deb's feasibility rules are the default, so a feasible candidate always outranks an infeasible
one and no penalty factor has to be tuned.

```go
config.Constraints = &dragonfly.ConstraintConfig{
	Inequalities: []dragonfly.ConstraintFunction{
		func(x []float64) float64 { return 1 - (x[0] + x[1]) },
	},
	Equalities: []dragonfly.ConstraintFunction{
		func(x []float64) float64 { return x[2] - 0.5 },
	},
	EqualityTolerance: 1e-6,
}

result, err := dragonfly.Optimize(config)
if err != nil {
	log.Fatal(err)
}

fmt.Printf("cost %.6f, violation %g, feasible %v\n",
	result.GlobalBest.Cost,
	result.GlobalBest.ConstraintViolation,
	dragonfly.IsFeasible(result.GlobalBest.ConstraintViolation))
```

### Reproducing a run

When `Config.Rand` is nil, `OptimizeContext` draws a seed, records it in `Result.Seed` and
writes the generator back into the config. Feeding that seed back reproduces the trajectory
exactly:

```go
first, _ := dragonfly.Optimize(newConfig())

replay := newConfig()
replay.Rand = rand.New(rand.NewSource(first.Seed))

second, _ := dragonfly.Optimize(replay)
// second.GlobalBest.Cost == first.GlobalBest.Cost
```

## Algorithm Variants

| Variant                                  | Problem class                       | Entry point                                | Overhead |
| ---------------------------------------- | ----------------------------------- | ------------------------------------------ | -------- |
| **[DA](docs/algorithms/standard-da.md)** | Single-objective, continuous        | `Optimize` / `OptimizeContext`             | baseline |
| **[BDA](docs/algorithms/bda.md)**        | Single-objective, binary / discrete | `OptimizeBinary` / `OptimizeBinaryContext` | 1.0x     |
| **[MODA](docs/algorithms/moda.md)**      | Multi-objective, continuous         | `OptimizeMultiObjective`                   | 1.2x     |

### Using the variants

DA — the paper's continuous algorithm:

```go
config := dragonfly.NewDefaultConfig()
config.ObjectiveFunc = dragonfly.Sphere
config.ProblemSize = 10
config.LowerBound = -10
config.UpperBound = 10

result, err := dragonfly.Optimize(config)
```

BDA — binary positions, bounds already fixed at `[0, 1]`:

```go
config := dragonfly.NewBinaryConfig()
config.ObjectiveFunc = oneMax
config.ProblemSize = 30

result, err := dragonfly.OptimizeBinary(config)
```

MODA — a Pareto archive instead of a single incumbent:

```go
config := dragonfly.NewMultiObjectiveConfig()
config.ObjectiveFunc = dragonfly.ZDT1
config.Swarm.ProblemSize = 5
config.Swarm.LowerBound = 0
config.Swarm.UpperBound = 1

result, err := dragonfly.OptimizeMultiObjective(context.Background(), config)

fmt.Printf("archive: %d solutions, non-dominated: %v\n",
	result.Archive.Len(), result.Archive.IsNonDominated())
```

Two other continuous presets layer on top of `NewDefaultConfig`:
`NewHighDimensionalConfig` (larger swarm, longer run, slower radius growth) and
`NewFastConvergenceConfig` (short run, faster radius growth, wider step clamp).

## Intelligent Selection

Describe the problem — or let the library sample it — and ask which variant to run:

```go
characteristics := dragonfly.ClassifyProblem(
	dragonfly.Rastrigin, 30, -5.12, 5.12, rand.New(rand.NewSource(1)))
// classified: 30-D, highly multimodal, rugged

selector := dragonfly.NewAlgorithmSelector()
best := selector.RecommendBest(characteristics)
// DA, score 0.70, confidence 0.85, preset "default"

result, err := dragonfly.NewBuilderFromVariant(best.Variant).
	ForProblem(dragonfly.Rastrigin, 30, -5.12, 5.12).
	WithIterations(300).
	WithPopulation(40).
	Optimize()
```

`ClassifyProblem` fills in `Dimensionality`, `Modality`, `Landscape` and
`RequiresStableConvergence` from a handful of straight-line scans across the box. `Discrete`,
`MultiObjective`, `ExpensiveEvaluations` and `RequiresFastConvergence` are facts about your
problem and your budget that no amount of sampling can recover — set them yourself. For a
benchmark function from `functions.go`, `RecommendForBenchmark("Schwefel")` skips the sampling
and reads a hand-classified table instead.

The fluent builder also works from a name:

```go
result, err := dragonfly.NewBuilder("bda").
	ForProblem(oneMax, 30, 0, 1).
	WithIterations(500).
	Optimize()
```

## Comparison Framework

`ComparisonRunner` runs several variants over the same problem with **paired seeds** — run `k`
of every variant is given `BaseSeed + k`, so the variants face identical starting swarms and
the differences that remain are the algorithms':

```go
runner := dragonfly.NewComparisonRunner().
	WithVariantNames("da", "bda").
	WithRuns(30).
	WithIterations(500).
	WithTarget(1e-3).
	WithParallel(true).
	WithSeed(4242)

result := runner.Compare("Sphere", dragonfly.Sphere, 10, -10, 10)

result.PrintComparisonResults()
_ = result.ExportToCSV("comparison.csv")
_ = result.ExportToJSON("comparison.json")
```

The report carries per-variant mean, median, standard deviation, best, worst, success rate and
rank; pairwise Wilcoxon signed-rank tests; and one Friedman test across every variant at once.
See [docs/api/comparison-framework.md](docs/api/comparison-framework.md).

## Benchmark Functions

**Single-objective (15):** Sphere, Rastrigin, Rosenbrock, Ackley, Griewank, Schwefel, Levy,
Zakharov, Michalewicz, DixonPrice, BentCigar, Discus, Weierstrass, HappyCat,
ExpandedSchafferF6

**Multi-objective (4):** ZDT1, ZDT2, ZDT3, SchafferN1

Every single-objective function has the signature `func([]float64) float64` and is a
minimization problem; the multi-objective ones return `[]float64`, one value per objective, all
minimized. See [docs/benchmarks.md](docs/benchmarks.md) for bounds, optima and measured
results.

## Documentation

### Guides

- **[Documentation hub](docs/README.md)** — the navigation guide for everything below
- **[API Quick Reference](docs/api/quick-reference.md)** — entry points, options, result fields
- **[Configuration Guide](docs/api/configuration.md)** — every `Config` field explained
- **[Run Lifecycle](docs/api/run-lifecycle.md)** — cancellation, observers, logging, export
- **[Comparison Framework](docs/api/comparison-framework.md)** — statistical testing

### Algorithms

- **[Standard DA](docs/algorithms/standard-da.md)** — the continuous algorithm
- **[BDA](docs/algorithms/bda.md)** — the binary variant and its transfer functions
- **[MODA](docs/algorithms/moda.md)** — the multi-objective variant and its hypercube archive

### Reference

- **[Benchmark Functions](docs/benchmarks.md)** — test functions and measured results
- **[Performance and Profiling](docs/performance.md)** — measured timings, scaling, profiling
- **[Research References](docs/research.md)** — citations and BibTeX
- **[Releasing](docs/releasing.md)** — version policy and release checklist

## Examples

Each example directory is its own Go module with a `replace` directive pointing at the
repository root, so they build against the working tree rather than a published version:

```sh
(cd examples/basic && go run .)             # a first continuous run
(cd examples/constrained && go run .)       # inequality and equality constraints
(cd examples/feature_selection && go run .) # BDA on a wrapper-style feature-selection problem
(cd examples/multiobjective && go run .)    # MODA on ZDT1, exporting the front
(cd examples/parallel && go run .)          # deterministic parallel evaluation
(cd examples/comparison && go run .)        # the statistical comparison framework
```

## Build Commands

Using the [Just](https://github.com/casey/just) task runner:

```sh
just                 # list every recipe
just build           # go build -v ./...
just test            # tests with coverage -> coverage.out + coverage.html
just test-quick      # -short, quickest
just test-race       # -race -short, 5m timeout
just test-full       # everything, including the long suites
just test-integration # the godog feature files only
just bench           # go test -bench=. -benchmem ./...
just fmt             # treefmt (gofumpt -> gci, prettier, taplo, shfmt)
just lint            # golangci-lint run --config ./.golangci.toml
just check           # check-formatted + check-tidy + lint + test
just ci              # verify + check
just profile-cpu     # CPU profile of BenchmarkOptimizeBaseline
just profile-mem     # memory profile of the same benchmark
```

Or with plain Go:

```sh
go build -v ./...
go test -v -race -short ./...
go test -bench=. -benchmem -run='^$' ./...
go test -coverprofile=coverage.out ./... && go tool cover -html=coverage.out
```

## Research & Citations

**Primary source.** Mirjalili, S. (2016). Dragonfly algorithm: a new meta-heuristic
optimization technique for solving single-objective, discrete, and multi-objective problems.
_Neural Computing and Applications_, 27(4), 1053–1073.
doi:[10.1007/s00521-015-1920-1](https://doi.org/10.1007/s00521-015-1920-1)

Supporting work: Reynolds (1987) for separation, alignment and cohesion; Mantegna (1994) for
the Lévy flight; Deb (2000) for the constraint feasibility rules; Coello Coello, Pulido &
Lechuga (2004) for the hypercube archive MODA borrows. Full citations and BibTeX entries are in
[docs/research.md](docs/research.md).

### Algorithm Implementation Map

| File                | Algorithm / operator                                     | Reference                           |
| ------------------- | -------------------------------------------------------- | ----------------------------------- |
| `dragonfly.go`      | Main loop, two-branch step update, food-in-radius test   | Mirjalili (2016), §3                |
| `swarm.go`          | Separation, alignment, cohesion, food, enemy; neighbours | Mirjalili (2016); Reynolds (1987)   |
| `weights.go`        | Adaptive `w, s, a, c, f, e`, radius, step clamp          | Mirjalili (2016), §3.2              |
| `levy.go`           | Lévy flight (Mantegna's algorithm, β = 1.5)              | Mantegna (1994), Phys. Rev. E       |
| `binary.go`         | BDA: V/S transfer functions, bit-flip update             | Mirjalili (2016), §4                |
| `multiobjective.go` | MODA: Pareto archive, hypercube grid, food/enemy draws   | Mirjalili (2016), §5; Coello (2004) |
| `constraints.go`    | Deb's feasibility rules, linear/quadratic penalties      | Deb (2000)                          |
| `comparison.go`     | Wilcoxon signed-rank, Friedman test                      | Standard non-parametric statistics  |
| `functions.go`      | 15 single-objective + 4 multi-objective benchmarks       | CEC / standard test suites, ZDT     |

### Verified, and not yet verified

The reference `DA.m`, `BDA.m` and `MODA.m` are the authority wherever the paper and a "cleaner"
formulation disagree. Three constants deserve a plain statement of their provenance, because
they are easy to mistake for settled paper values:

| Constant                                                    | Status                                                                                                                                                                                                                                                         |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Lévy σ and the `0.01` scale factor                          | **Verified.** `levySigma(1.5)` evaluates to `0.6965745026`, the accepted Mantegna value for β = 1.5, and `0.01` is the scale the DA reference implementation uses.                                                                                             |
| MODA's `β = 4, γ = 2, δ = 2, NGrid = 10, ArchiveSize = 100` | **Unverified.** These are the MOPSO defaults from Coello Coello et al. (2004), the lineage MODA borrows its archive from. `MODA.m` is not available to this repository, so they have not been read off the author's code. Do not cite them as DA paper values. |
| `NewBinaryConfig`'s `MaxStepRatio = 6.0`                    | **This implementation's choice.** The transfer functions saturate by \|Δx\| ≈ 6, so clamping there is what makes the whole range of flip probabilities reachable. It has not been checked against `BDA.m`.                                                     |

### Deviations from the reference MATLAB

Each is deliberate, and each is commented at the point in the source where it happens:

1. **Boundary repair runs after the position update, not before.** `DA.m` repairs a dragonfly
   at the top of its per-dragonfly block, so the swarm it computes S, A and C against is partly
   repaired and partly not, and the positions left in the population when the loop ends are the
   unrepaired ones. This implementation repairs immediately after the position update instead.
   The repair still happens exactly once per dragonfly per iteration, so the dynamics are
   otherwise the same — but every position handed to the objective function, and every position
   in the returned `Result`, is inside `[LowerBound, UpperBound]`.
   (`parallel_phases.go`, `prepareSwarmStep`)
2. **Binary mode ignores `BoundaryMethod` and the Lévy branch.** A 0/1 vector cannot leave
   `[0, 1]`, so there is nothing for a wrap, clamp or reflect rule to repair — and applying the
   wrap rule anyway would reset the very step the next bit-flip decision is made from. The Lévy
   walk is a multiplicative displacement of a real-valued position and has no binary
   counterpart, so the food-out-of-range branch is the local-swarming step for every dragonfly,
   isolated or not. `Config.UseLevyWalk` is ignored. (`binary.go`, `OptimizeBinaryContext`)
3. **`selector.go` does not port Mayfly's gradient-magnitude landscape heuristic.** That
   heuristic is scale-dependent: it called Sphere over `[-5, 5]` rugged and Sphere over
   `[-1, 1]` smooth, which says more about the bounds than about the function. It is replaced
   by two scale-free statistics — direction changes per line scan, and total variation in units
   of that line's own value range. (`selector.go`, `lineScanStatistics`)

## Performance

Measured on Linux/amd64, Go 1.26.0, AMD Ryzen 5 4600H (12 threads). Absolute timings are
machine-specific; the shapes are not. Full tables in [docs/performance.md](docs/performance.md).

| Workload                                                           |       Time |   Allocations |
| ------------------------------------------------------------------ | ---------: | ------------: |
| `BenchmarkOptimizeBaseline` (30-D Sphere, 100 iterations, NPop 40) | 49.8 ms/op | 24,927 allocs |
| The same with `EnableParallel` on a cheap objective                | 57.3 ms/op | 27,050 allocs |
| 50 iterations on a 200x-more-expensive objective, sequential       | 37.3 ms/op |  9,274 allocs |
| The same, parallel                                                 | 22.3 ms/op | 10,347 allocs |

Two findings worth planning around:

- **The evaluation pool only pays for itself on an expensive objective.** On Sphere it is about
  15% _slower_ than the sequential path. On an objective 200 times more expensive, the same
  workload is 1.67x faster. Below roughly a microsecond per evaluation, leave `EnableParallel`
  off.
- **Population scaling is super-linear.** The neighbour scan is `O(n²·d)` and dominates for
  large swarms: `NPop` 10 → 40 → 100 → 250 costs 1.27 → 8.76 → 39.5 → 207 ms per run, roughly
  `n^1.8`. Dimension scaling, by contrast, is linear.

Parallel evaluation never changes the answer: every RNG draw happens on the calling goroutine
during the prepare phase, and workers only call the objective. A seeded run is bit-identical
with `EnableParallel` on or off, and `TestParallelIsDeterministicForSeedAcrossSchedules`
enforces it.

## Status

`v0.1.0`. Phases 1–10 of [PLAN.md](PLAN.md) have landed: the three variants, the framework
layer, deterministic parallelism, constraints, lifecycle and monitoring, the BDD feature
files, the regression baselines, the benchmark suite, the documentation and the release
preparation.

**PLAN.md is the source of truth for progress** — read its checkboxes rather than this
paragraph, and read the `## [0.1.0]` entry in [CHANGELOG.md](CHANGELOG.md) for the release's
own list of known limitations. The two that most often surprise a reader: `Levy(nil)` panics
and empty-input handling is inconsistent across the benchmark suite (neither affects a real
optimization, where every position has `ProblemSize >= 1` components), and MODA's hypercube
parameters are this implementation's choices rather than values read off the author's MATLAB.

## Contributing

1. Fork the repository and create a feature branch
2. Follow the conventions in [AGENTS.md](AGENTS.md) and [CLAUDE.md](CLAUDE.md)
3. Add tests for new behaviour — a new operator wants a hand-computed unit test, not only an
   end-to-end convergence check
4. Run `just check` and make sure it is green
5. Open a pull request describing the goal, the key changes and the impact

## License

MIT — see [LICENSE](LICENSE).

---

**Quick links:**
[Documentation](docs/README.md) |
[API](docs/api/quick-reference.md) |
[Algorithms](docs/algorithms/) |
[Benchmarks](docs/benchmarks.md) |
[Research](docs/research.md)
