# Dragonfly Algorithm - AI Coding Instructions

## Project Overview

Go implementation of Seyedali Mirjalili's Dragonfly Algorithm (DA), a swarm-intelligence metaheuristic derived from the static and dynamic swarming behaviour of dragonflies. It covers all three variants of the original paper: **DA** (single-objective continuous), **BDA** (binary/discrete, via transfer functions), and **MODA** (multi-objective, via a hypercube-partitioned Pareto archive).

Module `github.com/MeKo-Christian/dragonfly`, package `dragonfly`, flat at the repository root, Go 1.23.3. Standard library only; the sole planned direct dependency is `github.com/cucumber/godog`, test-only.

**There is no Go source in the repository yet** — only the scaffold (tooling, workflows, docs skeleton). `PLAN.md` is the specification and the progress tracker; read it before writing code, and take all algorithm facts from its §1 rather than from recall.

## Core Architecture

### Main Components

Planned file layout (`PLAN.md` §2), one concern per file:

- **`dragonfly.go`**: package doc, `Optimize`, `OptimizeContext` — the main entry points.
- **`types.go`**: `Config`, `Dragonfly`, `Best`, `Result`, `ConvergenceConfig`, `TerminationReason`, `BoundaryMethod`, the `WeightAuto` sentinel.
- **`swarm.go`**: neighbourhood scan and the five swarming vectors S/A/C/F/E.
- **`weights.go`**: the adaptive `w, s, a, c, f, e, r, ΔX_max` schedules.
- **`levy.go`**: Mantegna Lévy flight (β = 1.5).
- **`binary.go`** / **`multiobjective.go`**: BDA transfer functions and bit flipping; MODA archive and hypercube grid.
- **`functions.go`**: benchmark functions, ported verbatim from Mayfly.
- **`parallel*.go`**: `evaluationPool`, `parallelFor`, the `prepare*` phases, the per-variant evaluators.
- **`variants.go` / `selector.go` / `comparison.go`**: the framework layer over the three variants.

### Key Design Patterns

**Configuration-Driven Design**: one flat `Config` struct with snake_case JSON tags, fields ordered for `fieldalignment` (pointers/interfaces → strings → float64 → int → bool). Start from a factory and override:

```go
config := dragonfly.NewDefaultConfig()
config.ObjectiveFunc = myFunction
config.ProblemSize = 30
```

`NewBinaryConfig()` and friends layer presets on top. Schedule fields default to the sentinel `WeightAuto = -1`, which selects the paper schedule.

**Functional Interface**: problems are `func([]float64) float64`, always minimization — including BDA, whose input is 0/1-valued so the benchmark, comparison, and constraint machinery is reused unchanged.

**Explicit RNG threading**: every stochastic helper takes `rng *rand.Rand` as its last parameter.

**Paper fidelity first, ergonomics second**: where the reference MATLAB (`DA.m`, `BDA.m`, `MODA.m`) and a cleaner formulation disagree, implement the paper and expose the alternative behind a `Config` field.

## Development Workflows

```bash
just build            # go build ./...
just test             # coverage.out + coverage.html
just test-integration # godog features
just check            # format + tidy + lint + test — the PR gate
just bench            # benchmarks with -benchmem
```

Formatting is treefmt (gofumpt → gci for Go, prettier for md/json/yaml, taplo for toml, shfmt for shell). Linting is golangci-lint v2 against **`.golangci.toml`** — the config is TOML, not `.golangci.yml`.

Examples live in `examples/`, each subdirectory its own module with a local `replace` directive.

## Critical Implementation Details

### The five swarming primitives

For dragonfly `i` with `N` neighbours inside radius `r`:

```
Separation  S_i = -Σ_j (X_i - X_j)
Alignment   A_i = (Σ_j V_j) / N
Cohesion    C_i = (Σ_j X_j) / N - X_i
Food        F_i = X⁺ - X_i          X⁺ = best position so far
Enemy       E_i = X⁻ + X_i          X⁻ = worst position so far
```

Two details that are easy to get wrong and must both carry unit tests:

- **The enemy term is a sum, not a difference.** `X⁻ + X_i` is what the paper and the reference code use. A sign "fix" here is a bug.
- **The neighbourhood test is per-dimension, not Euclidean.** The reference computes a component-wise distance vector and requires `all(dist <= r)`; self-neighbouring is excluded via `all(dist != 0)`. A Euclidean shortcut changes swarm dynamics and silently degrades convergence.

### Two-branch step update

The single most important fidelity detail — the reference branches on whether the food source is inside the radius:

```
if any(dist2Food > r):                 # food out of range → local swarming only
    if neighbours > 1:
        ΔX_i = w·ΔX_i + rand·A_i + rand·C_i + rand·S_i   # per-dimension rand; no f, no e
        clamp ΔX to ±ΔX_max
        X_i += ΔX_i
    else:
        X_i += Levy(d) ⊙ X_i                              # Lévy random walk
        ΔX_i = 0
else:                                  # food in range → full five-factor step
    ΔX_i = (s·S_i + a·A_i + c·C_i + f·F_i + e·E_i) + w·ΔX_i
    clamp ΔX to ±ΔX_max
    X_i += ΔX_i
```

### Weight schedules

```
w  = 0.9 - t·(0.9 - 0.4)/T                 inertia, linearly decreasing
mc = max(0, 0.1 - t·0.1/(T/2))             shared convergence factor
s = a = c = 2·rand·mc                      separation / alignment / cohesion
f  = 2·rand                                food attraction
e  = mc, forced to 0 once t > 3T/4         enemy distraction
r  = (ub - lb)/4 + (ub - lb)·(t/T)·2       neighbourhood radius, growing
ΔX_max = (ub - lb)/10                      step clamp
```

Assert these as properties in `weights_test.go` (w monotonically decreasing, mc zero at the halfway point, e exactly zero past 3T/4) rather than inferring them from optimization outcomes.

### Boundary handling

DA uses **wrap-with-step-reset**, not Mayfly's clamp:

```
if x_j > ub_j { x_j = lb_j ; Δx_j = rand() }
if x_j < lb_j { x_j = ub_j ; Δx_j = rand() }
```

Selectable via `Config.BoundaryMethod`: `"wrap"` (default, paper behaviour), `"clamp"` (Mayfly's `maxVec`/`minVec` idiom), `"reflect"`.

### Randomization and determinism

- `Config.Rand` is the injection point. When nil, `OptimizeContext` creates a generator, writes it back to the config, and records the seed in `Result.Seed`.
- **All RNG draws happen on the calling goroutine during the `prepare*` phase**; worker goroutines only evaluate the objective function. A seeded run must be bit-identical with `EnableParallel` on or off.
- DA's prepare phase is two sequential passes: first `r`, the weights, food and enemy; then, per dragonfly, the neighbour scan and ΔX. The neighbour scan is O(n²·d) and is the real hot spot — it may be parallelised precisely because it draws no random numbers.

## Common Extension Points

### Custom objective functions

Always a minimization problem. For maximization, negate:

```go
func maximizeProfit(x []float64) float64 {
    return -calculateProfit(x) // Negate for maximization
}
```

### Transfer functions (BDA)

`TransferFunction` is a named-string type backed by a registry: V-shaped `v1`–`v4` (`v3 = |Δx/√(Δx²+1)|` is the paper default) and S-shaped `s1`–`s4`. Add a family by registering it, not by branching in the update loop.

### Algorithm parameters

Override any schedule field to pin it; leave it at `WeightAuto` for the paper schedule. Widen `r` or raise the population for rugged landscapes; `BoundaryMethod` is the first knob to try on constrained problems.

## Testing and Validation

Tests are `*_test.go` siblings at the root in `package dragonfly` (white-box); `example_test.go` is the only black-box file. No `t.Parallel()` — runs are deterministic and seed-driven.

`swarm_test.go` should build a 3-dragonfly, 2-dimensional swarm and compare S, A, C, F, E and one full ΔX step against values worked out by hand from `DA.m`. That single test is what catches a Euclidean-vs-per-dimension neighbour mistake or a sign error in the enemy term; neither shows up as an obvious failure in an end-to-end convergence test.

### Benchmark functions

Ported verbatim from `Mayfly/functions.go`. Known minima and typical bounds:

- `Sphere`: f(0,…,0) = 0, bounds [-10, 10] — smooth convex bowl
- `Rastrigin`: f(0,…,0) = 0, bounds [-5.12, 5.12] — highly multimodal lattice
- `Rosenbrock`: f(1,…,1) = 0, bounds [-5, 10] — narrow curved valley
- `Ackley`: f(0,…,0) = 0, bounds [-32.768, 32.768] — flat outer region, deep central basin
- `Griewank`: f(0,…,0) = 0, bounds [-600, 600]
- `Schwefel`: f(420.9687,…) = 0, bounds [-500, 500] — deceptive, optimum far from the next best minima
- `Levy`: f(1,…,1) = 0, bounds [-10, 10]
- `Zakharov`: f(0,…,0) = 0, bounds [-5, 10] or [-10, 10] — unimodal
- `Michalewicz`: minimum is dimension-dependent (steepness m = 10), bounds [0, π]
- `DixonPrice`: f(x_i = 2^-((2^i - 2)/2^i)) = 0, bounds [-10, 10]
- `BentCigar`: f(0,…,0) = 0, bounds [-100, 100] — severely ill-conditioned
- `Discus`: f(0,…,0) = 0, bounds [-100, 100] — ill-conditioned along one direction
- `Weierstrass`: f(0,…,0) = 0, bounds [-0.5, 0.5] — continuous everywhere, differentiable nowhere
- `HappyCat`: f(-1,…,-1) = 0, bounds [-2, 2] — curved thin optimal region
- `ExpandedSchafferF6`: f(0,…,0) = 0, bounds [-100, 100] — concentric ripples

MODA adds ZDT1–ZDT3 and Schaffer.

### Gates

- Phase 2: converges on Sphere, Rastrigin, Ackley, Rosenbrock at d = 10 within the tolerances recorded in `regression_test.go`; two runs with the same seed produce identical `Result` values.
- Phase 4: a seeded parallel run is bit-identical to the sequential one; cancellation commits no partial batch.
- Phase 6: recovers the known ZDT1/ZDT3 fronts; the archive never exceeds `ArchiveSize`; every archived solution is mutually non-dominated.

Regression baselines encode **tolerated degradation factors**, not exact expected values — a stochastic optimizer's output is not a golden file.

## Research Context

Mirjalili, S. (2016). *Dragonfly algorithm: a new meta-heuristic optimization technique for solving single-objective, discrete, and multi-objective problems.* Neural Computing and Applications 27(4), 1053–1073. doi:10.1007/s00521-015-1920-1.

Supporting sources: Reynolds (1987) for separation/alignment/cohesion; Mantegna (1994) for the Lévy generator; Deb (2000) for the constraint feasibility rules; Coello Coello et al. (2004) for the hypercube archive MODA borrows.

Maintain fidelity to the reference MATLAB implementations. This library is a deliberate sibling of [Mayfly](https://github.com/cwbudde/mayfly): the conventions, the `Config` shape, and the comparison framework are shared so the two can be run head-to-head. Do not diverge from those conventions without recording the reason in `PLAN.md`.
