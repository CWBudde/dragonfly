# Dragonfly Algorithm (Go)

A dependency-free Go implementation of the **Dragonfly Algorithm (DA)**, the swarm
metaheuristic Seyedali Mirjalili introduced in 2016. It models the static and dynamic
swarming behaviour of dragonflies — separation, alignment, cohesion, attraction to food and
distraction from enemies — and covers all three variants from the original paper plus three
published improved variants.

[![Go Reference](https://pkg.go.dev/badge/github.com/CWBudde/dragonfly.svg)](https://pkg.go.dev/github.com/CWBudde/dragonfly)

Sibling project: [Mayfly](https://github.com/cwbudde/mayfly), which shares this library's API
style, tooling and conventions.

## Overview

**Key features:**

- **Six variants** — `DA`, `BDA`, `MODA`, memory-based `MHDA`, chaotic `CDA`, and
  quantum/Gaussian `QGDA`
- **Standard library only** — the sole direct dependency is `godog`, and it is test-only
- **Deterministic** — a seeded run reproduces bit-for-bit, with parallel evaluation on or off
- **Explicit fidelity** — paper behavior is the default; `FidelityMATLAB` names the reference
  operator choices where the sources disagree instead of hiding a paper/MATLAB hybrid
- **Constraint handling** — Deb's feasibility rules, or linear/quadratic penalties
- **Observable** — progress and population observers, `log/slog` integration, CSV/JSON export
- **Benchmark suite** — 16 standalone single-objective and 4 multi-objective functions, plus
  all 29 usable CEC2017 and all 10 CEC2020 competition problems
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
go get github.com/CWBudde/dragonfly
```

Use the lowercase repository component exactly as shown. The obsolete
`github.com/CWBudde/Dragonfly@v0.1.0` path was cached by the Go module proxy before the
repository rename; it is a distinct, unsupported module path and will receive no updates.

### Basic usage

```go
package main

import (
	"fmt"
	"log"

	"github.com/CWBudde/dragonfly"
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

Set `Config.Seed` when the seed must be reportable and replayable. Library-created seeds are
also reported with `Result.SeedKnown = true`; a directly injected `Config.Rand` has unknown
seed metadata.

```go
first, _ := dragonfly.Optimize(newConfig())

replay := newConfig()
replay.Seed = &first.Seed

second, _ := dragonfly.Optimize(replay)
// second.GlobalBest.Cost == first.GlobalBest.Cost
```

## Algorithm Variants

| Variant                                  | Problem class                       | Entry point                                | Overhead |
| ---------------------------------------- | ----------------------------------- | ------------------------------------------ | -------- |
| **[DA](docs/algorithms/standard-da.md)** | Single-objective, continuous        | `Optimize` / `OptimizeContext`             | baseline |
| **[BDA](docs/algorithms/bda.md)**        | Single-objective, binary / discrete | `OptimizeBinary` / `OptimizeBinaryContext` | 1.0x     |
| **[MODA](docs/algorithms/moda.md)**      | Multi-objective, continuous         | `OptimizeMultiObjective`                   | 1.2x     |
| **[MHDA](docs/algorithms/mhda.md)**      | Single-objective, continuous        | `OptimizeMemoryHybrid`                     | 2.0x     |
| **[CDA](docs/algorithms/cda.md)**        | Single-objective, continuous        | `OptimizeChaotic`                          | 1.0x     |
| **[QGDA](docs/algorithms/qgda.md)**      | Single-objective, continuous        | `OptimizeQuantum`                          | 3.0x     |

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
// QGDA for this rugged multimodal landscape

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

**Single-objective (16):** Sphere, Rastrigin, Rosenbrock, Ackley, Griewank, Schwefel, Levy,
Zakharov, Michalewicz, DixonPrice, BentCigar, Discus, Weierstrass, HappyCat,
ExpandedSchafferF6, Himmelblau

**Multi-objective (4):** ZDT1, ZDT2, ZDT3, SchafferN1

**Competition suites:** all 29 usable CEC2017 functions (F2 was withdrawn) and all 10 CEC2020
functions, with the organizers' shifts, rotations, permutations, hybrids, compositions, biases
and evaluation budgets loaded from an external `fs.FS`.

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
- **[MHDA](docs/algorithms/mhda.md)** — personal memory plus PSO exploitation
- **[CDA](docs/algorithms/cda.md)** — one of ten chaotic maps drives all DA coefficients
- **[QGDA](docs/algorithms/qgda.md)** — Gaussian mutation and quantum rotation

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
(cd examples/mhda && go run .)              # MHDA with personal/global memory and a PSO phase
(cd examples/cda && go run .)               # continuous chaotic DA with Gauss-map weights
(cd examples/qgda && go run .)              # QGDA with Gaussian and quantum operators
(cd examples/parallel && go run .)          # deterministic parallel evaluation
(cd examples/comparison && go run .)        # the statistical comparison framework
```

## Web Demo

A browser demo of the library lives in [`examples/wasm-demo`](examples/wasm-demo)
and is published to <https://cwbudde.github.io/Dragonfly/>. It has four pages:

- a **Swarm Lab** that animates the swarm over a benchmark landscape, colouring
  each dragonfly by its update regime and drawing the neighbourhood as the
  axis-aligned box it actually is;
- a **Pareto** page that animates MODA's archive filling in, over the hypercube
  grid its food and enemy draws turn on;
- a **Binary** page showing BDA's swarm as a bit matrix beside the transfer
  function's own probability curve;
- a **Shootout** that runs the comparison framework over DA's configurable
  choices — boundary rule, Lévy walk, enemy term — with paired seeds, Wilcoxon
  signed-rank tests and a Friedman test.

Everything they show is computed by this library compiled to `js/wasm`; there is
no JavaScript reimplementation of the algorithm.

```bash
just run-wasm-demo   # build into ./dist and serve at http://localhost:8090
```

See [`examples/wasm-demo/README.md`](examples/wasm-demo/README.md) for what it
exercises and how to read its numbers.

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
just ci              # quality + coverage + examples/WASM + security
just ci-race         # the same plus the short race suite
just security        # pinned Nancy and govulncheck scans
just profile-cpu     # CPU profile of BenchmarkOptimizeBaseline
just profile-mem     # memory profile of the same benchmark
```

CI tests Go 1.23 and 1.26. Formatter/linter/scanner versions are pinned and verified by the
Just recipes; release validation includes race, 80% coverage, examples/WASM and security, and a
separate Go 1.26 security workflow runs weekly and on demand.

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
| `functions.go`      | 16 single-objective + 4 multi-objective benchmarks       | CEC / standard test suites, ZDT     |
| `cec*.go`           | CEC2017 and CEC2020 transformed competition suites       | CEC technical reports and software  |

### Fidelity and verified reference values

The paper is the default authority. `Config.FidelityMode = FidelityMATLAB` selects the author's
reference operators where the sources disagree, and MODA exposes paper, MATLAB-density and
legacy MOPSO archive policies by name. Three constants deserve a plain provenance statement:

| Constant                                      | Status                                                                                                                                                                                                   |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Lévy σ and the `0.01` scale factor            | **Verified.** `levySigma(1.5)` is `0.6965745026`; the official DA and BDA `Levy.m` files use β `1.5` and scale `0.01`.                                                                                   |
| MODA's `β = 4, γ = 2, δ = 2, NGrid = 10`      | **Legacy extension only.** These are not `MODA.m` values. They remain available through `ArchivePolicyMOPSOGrid`; paper mode uses `1/N` and `N`, while MATLAB mode uses objective-space density ranking. |
| `ArchiveSize = 100` and BDA's `±6` step clamp | **Verified.** Both occur in the official `MODA.m` / `BDA.m` packages.                                                                                                                                    |

MATLAB-compatible MODA also uses its own reference schedule: inertia decreases from `0.9` to
`0.2`, automatic separation/alignment/cohesion/enemy weights follow `mc` directly, and only
the automatic food weight draws the `2·rand` factor. Explicitly pinned weights remain library
extensions and still win over their automatic values.

### Paper and MATLAB lifecycle contracts

Paper mode evaluates the initialized swarm and every moved swarm, so a complete `NPop = N`,
`MaxIterations = T` run makes `N·(T+1)` objective calls. MATLAB mode instead reproduces the
reference evaluate-before-move loop: each generation evaluates its current population, updates
the incumbents or archive, moves once, and leaves the final moved population unevaluated. It
therefore makes `N·T` calls. Returned best/worst values and Pareto solutions always come from
evaluated candidates.

For continuous DA and MODA, `FidelityMATLAB` overrides `BoundaryMethod`. Its exact per-dragonfly
order is: compute primitives from the current swarm, apply the reference pre-move wrap and step
reset, move using the already-computed primitives, sanitize non-finite safety cases, then clamp
the moved position to the box. The final positions are repaired, not left out of bounds.

MATLAB-compatible DA keeps two worst references. Movement follows `DA.m` and only updates its
enemy from strictly interior evaluated positions; `Result.Worst` independently reports the
actual worst evaluated candidate. A population snapshot exposes the movement enemy, while its
swarm copy is the evaluated pre-move population so every cost still describes its position.

Other deliberate differences and extensions are:

1. **Binary mode ignores `BoundaryMethod` and the Lévy branch.** A bit cannot leave `[0,1]`,
   and BDA uses every other dragonfly as a neighbor with one unconditional five-factor step.
   V-shaped transfers complement the current bit; S-shaped transfers assign the sampled bit.
   (`binary.go`, `OptimizeBinaryContext`)
2. **`selector.go` does not port Mayfly's gradient-magnitude landscape heuristic.** That
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

`v0.1.0` is the latest published release. The v0.2.0 correctness/fidelity remediation and
local release preparation are underway; they are not released yet.

**PLAN.md is the source of truth for progress** — read its checkboxes rather than this
paragraph, and read the `## [0.1.0]` entry in [CHANGELOG.md](CHANGELOG.md) for that release's
own known limitations. Unreleased behavior described here belongs to the v0.2.0 preparation.

## Contributing

1. Fork the repository and create a feature branch
2. Follow the conventions in [AGENTS.md](AGENTS.md)
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
