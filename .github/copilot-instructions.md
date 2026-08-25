# Dragonfly Algorithm - AI Coding Instructions

## Project Overview

Go implementation of Seyedali Mirjalili's Dragonfly Algorithm (DA), a swarm-intelligence metaheuristic derived from the static and dynamic swarming behaviour of dragonflies. It covers all three variants of the original paper: **DA** (single-objective continuous), **BDA** (binary/discrete, via transfer functions), and **MODA** (multi-objective, via a hypercube-partitioned Pareto archive).

Module `github.com/CWBudde/dragonfly`, package `dragonfly`, flat at the repository root, Go 1.23.3. Standard library only in production; the sole direct dependency is `github.com/cucumber/godog`, test-only.

`v0.1.0` is published; the Phase 11 correctness/fidelity remediation and local `v0.2.0`
release preparation are underway. `PLAN.md` is the source of truth for progress. Use only the
lowercase repository component: proxy-cached `github.com/CWBudde/Dragonfly@v0.1.0` is an
obsolete, distinct module path that receives no updates.

## Core Architecture

### Main Components

Implemented file layout (`PLAN.md` §2), one concern per file:

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

**Named fidelity, never a hybrid**: paper behavior is the default. `FidelityMATLAB` reproduces
the reference MATLAB (`DA.m`, `BDA.m`, `MODA.m`) lifecycle and operators when they differ.

## Development Workflows

```bash
just build            # go build ./...
just test             # coverage.out + coverage.html
just test-integration # godog features
just check            # format + tidy + lint + test — the PR gate
just bench            # benchmarks with -benchmem
```

Formatting is treefmt (gofumpt → gci for Go, prettier for md/json/yaml, taplo for toml, shfmt for shell). Linting is golangci-lint v2 against **`.golangci.toml`** — the config is TOML, not `.golangci.yml`.

Exact tool versions live in `justfile` and are validated. CI tests Go 1.23 and 1.26; `just ci`
and release validation include coverage, examples/WASM and pinned Nancy/govulncheck security
scans, with a separate weekly/manual Go 1.26 security workflow.

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

### Paper and MATLAB step updates

Paper mode uses the full five-factor step whenever a neighbor exists, independently zeroing
food and enemy when out of range; isolation takes the Lévy walk. MATLAB mode uses the
reference food-distance branch:

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

MATLAB-compatible MODA is the schedule exception: inertia decreases `0.9 → 0.2`, automatic
S/A/C/E equal `mc` directly (including the reference late-run adjustment), and only automatic
food uses `2·rand`. Explicit pinned weights still override automatic values.

### Boundary handling

DA uses **wrap-with-step-reset**, not Mayfly's clamp:

```
if x_j > ub_j { x_j = lb_j ; Δx_j = rand() }
if x_j < lb_j { x_j = ub_j ; Δx_j = rand() }
```

Selectable via `Config.BoundaryMethod`: `"wrap"` (default, paper behaviour), `"clamp"` (Mayfly's `maxVec`/`minVec` idiom), `"reflect"`.

Those are paper-mode policies. Continuous `FidelityMATLAB` ignores `BoundaryMethod` and uses
the exact sequence primitives → pre-wrap/reset → move → sanitize → post-clamp. Its final moved
swarm is repaired but deliberately unevaluated.

Paper mode evaluates initialization plus every moved population (`NPop·(T+1)`). MATLAB mode
evaluates before each move (`NPop·T`) and leaves the final move unevaluated. MATLAB DA updates
its movement enemy only from strictly interior candidates; `Result.Worst` separately remains
the actual worst evaluated candidate, and population snapshots expose the movement enemy with
the evaluated pre-move swarm.

### Randomization and determinism

- Prefer `Config.Seed` when a reported reproducible seed is required. A directly supplied
  `Config.Rand` is honored but cannot expose its original seed; supplying both is rejected.
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
- `Himmelblau`: f(3, 2, 3, 2, …) = 0, bounds [-5, 5] — four equal minima per coordinate pair, with an odd trailing coordinate scored as x²

MODA adds ZDT1–ZDT3 and Schaffer.

### Gates

- Phase 2: converges on Sphere, Rastrigin, Ackley, Rosenbrock at d = 10 within the tolerances recorded in `regression_test.go`; two runs with the same seed produce identical `Result` values.
- Phase 4: a seeded parallel run is bit-identical to the sequential one; cancellation commits no partial batch.
- Phase 6: recovers the known ZDT1/ZDT3 fronts; the archive never exceeds `ArchiveSize`; every archived solution is mutually non-dominated.

Regression baselines encode **tolerated degradation factors**, not exact expected values — a stochastic optimizer's output is not a golden file.

## Research Context

Mirjalili, S. (2016). _Dragonfly algorithm: a new meta-heuristic optimization technique for solving single-objective, discrete, and multi-objective problems._ Neural Computing and Applications 27(4), 1053–1073. doi:10.1007/s00521-015-1920-1.

Supporting sources: Reynolds (1987) for separation/alignment/cohesion; Mantegna (1994) for the Lévy generator; Deb (2000) for the constraint feasibility rules; Coello Coello et al. (2004) for the hypercube archive MODA borrows.

Maintain fidelity to the reference MATLAB implementations. This library is a deliberate sibling of [Mayfly](https://github.com/cwbudde/mayfly): the conventions, the `Config` shape, and the comparison framework are shared so the two can be run head-to-head. Do not diverge from those conventions without recording the reason in `PLAN.md`.
