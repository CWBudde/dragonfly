# Dragonfly Algorithm (Go) — Implementation Plan

Module: `github.com/MeKo-Christian/dragonfly`
Package: `dragonfly` (flat, at the repository root)
Status: **Phase 1 in progress** — scaffold laid, no algorithm code yet.

This document is the roadmap. It is organised the same way as the sibling
[Mayfly](https://github.com/cwbudde/mayfly) project's `PLAN.md`: numbered phases,
`- [ ]` / `- [x]` task checkboxes, and a bolded `**Rationale**:` paragraph after each
subsection explaining *why* the task exists. Update the checkboxes as work lands.

---

## 0. Project Overview

A dependency-free Go implementation of Seyedali Mirjalili's **Dragonfly Algorithm (DA)**,
covering all three variants from the original paper:

| Variant  | Problem class                      | Entry point                |
| -------- | ---------------------------------- | -------------------------- |
| **DA**   | Single-objective, continuous       | `Optimize` / `OptimizeContext` |
| **BDA**  | Single-objective, binary / discrete | `OptimizeContext` with a transfer function |
| **MODA** | Multi-objective, continuous        | `OptimizeMultiObjective`   |

### Design principles (inherited from Mayfly — do not diverge without a note here)

1. **Standard library only.** The single direct dependency is `github.com/cucumber/godog`,
   and it is test-only. No numeric or utility third-party packages.
2. **Flat root package.** Source and tests are siblings at the repo root. No `internal/`,
   no `pkg/`, no `cmd/`.
3. **Configuration-driven.** One flat `Config` struct with snake_case JSON tags, field
   order chosen for `fieldalignment` (pointers/interfaces → strings → float64 → int →
   bool). Factory functions (`NewDefaultConfig`, `NewBinaryConfig`, …) layer presets on top.
4. **Explicit RNG threading.** Every stochastic helper takes `rng *rand.Rand` as its last
   parameter. `Config.Rand` is the injection point; when nil, `OptimizeContext` creates one,
   writes it back to the config, and records the seed in `Result.Seed`.
5. **Deterministic parallelism.** All RNG draws happen on the calling goroutine during a
   `prepare*` phase; worker goroutines only evaluate the objective function. A seeded run
   must produce bit-identical results with `EnableParallel` on or off.
6. **White-box tests.** Test files live in `package dragonfly` and may exercise unexported
   helpers. `example_test.go` is the sole black-box file.
7. **Paper fidelity first, ergonomics second.** Where the reference MATLAB code and a
   "cleaner" formulation disagree, implement the paper and expose the alternative behind a
   config field.

**Rationale**: The two libraries are meant to read as a family. A user who knows
`mayfly.Optimize(config)` should be able to guess `dragonfly.Optimize(config)` without
opening the docs, and a contributor moving between the repos should not have to relearn
the conventions.

---

## 1. The algorithm — specification to implement against

Primary source: Mirjalili, S. (2016). *"Dragonfly algorithm: a new meta-heuristic
optimization technique for solving single-objective, discrete, and multi-objective
problems."* **Neural Computing and Applications** 27(4), 1053–1073.
DOI: [10.1007/s00521-015-1920-1](https://doi.org/10.1007/s00521-015-1920-1).
Reference implementations: the author's `DA.m`, `BDA.m`, `MODA.m`.

### 1.1 The five swarming primitives (`swarm.go`)

For dragonfly `i` with `N` neighbours inside radius `r`:

```
Separation  S_i = -Σ_j (X_i - X_j)          repel from crowding
Alignment   A_i = (Σ_j V_j) / N             match neighbour velocity
Cohesion    C_i = (Σ_j X_j) / N - X_i       move toward the local centroid
Food        F_i = X⁺ - X_i                  X⁺ = best position found so far
Enemy       E_i = X⁻ + X_i                  X⁻ = worst position found so far
```

Two details that are easy to get wrong and must both be covered by unit tests:

- The **enemy term is a sum, not a difference**. `X⁻ + X_i` is what the paper and the
  reference code use.
- The **neighbourhood test is per-dimension, not Euclidean**. The reference code computes
  a component-wise distance vector and requires `all(dist <= r)`; self-neighbouring is
  excluded via `all(dist != 0)`. A Euclidean shortcut changes the swarm dynamics and will
  silently degrade convergence.

### 1.2 Step and position update (`dragonfly.go`)

The reference implementation branches on whether the food source is inside the radius.
This two-branch structure is the single most important fidelity detail in the whole
algorithm:

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

### 1.3 Boundary handling

DA uses **wrap-with-step-reset**, not the clamp Mayfly uses:

```
if x_j > ub_j { x_j = lb_j ; Δx_j = rand() }
if x_j < lb_j { x_j = ub_j ; Δx_j = rand() }
```

Expose the choice as `Config.BoundaryMethod` with values `"wrap"` (default, paper
behaviour), `"clamp"` (Mayfly's `maxVec`/`minVec` idiom), and `"reflect"`.

**Rationale**: Wrapping is genuinely part of DA's exploration behaviour, but it surprises
users coming from other metaheuristics and it interacts badly with some constrained
problems. Making it a named, documented choice is cheaper than fielding the bug reports.

### 1.4 Adaptive weight schedules (`weights.go`)

```
w  = 0.9 - t·(0.9 - 0.4)/T                 inertia, linearly decreasing
mc = max(0, 0.1 - t·0.1/(T/2))             the shared "convergence factor"
s  = 2·rand·mc                             separation
a  = 2·rand·mc                             alignment
c  = 2·rand·mc                             cohesion
f  = 2·rand                                food attraction
e  = mc,  forced to 0 once t > 3T/4        enemy distraction
r  = (ub - lb)/4 + (ub - lb)·(t/T)·2       neighbourhood radius, growing
ΔX_max = (ub - lb)/10                      step clamp
```

Each of these becomes an overridable `Config` field defaulting to the sentinel
`WeightAuto = -1`, which selects the schedule above — the same convention as Mayfly's
`NCAuto` and `AquilaWeightAuto`.

**Rationale**: Keeping every schedule in one file lets `weights_test.go` assert the
properties directly (w decreases monotonically, mc hits zero at the halfway point, e is
exactly zero past 3T/4) instead of inferring them from optimization outcomes.

### 1.5 Lévy flight (`levy.go`)

Mantegna's algorithm with β = 1.5:

```
σ = ( Γ(1+β)·sin(πβ/2) / ( Γ((1+β)/2)·β·2^((β-1)/2) ) )^(1/β)
Levy(x) = 0.01 · r₁·σ / |r₂|^(1/β)          r₁, r₂ ~ N(0,1)
```

Port from `Mayfly/levy.go`, but verify σ and the `0.01` scale factor against the DA paper
rather than assuming Mayfly's variant uses the same β.

### 1.6 BDA — binary variant (`binary.go`)

ΔX is computed exactly as in the continuous case; only the position update changes:

```
V-shaped (paper default):  T(Δx) = | Δx / sqrt(Δx² + 1) |
x_j ← ¬x_j  if rand < T(Δx_j)   else   x_j
```

Ship a `TransferFunction` named-string type with the standard families and a registry:

| Name | Form                                     |
| ---- | ---------------------------------------- |
| `v1` | \|erf(√π/2 · Δx)\|                       |
| `v2` | \|tanh(Δx)\|                             |
| `v3` | \|Δx / √(Δx²+1)\|  *(paper default)*     |
| `v4` | \|(2/π)·arctan((π/2)·Δx)\|               |
| `s1` | 1/(1+e^(-2Δx))                           |
| `s2` | 1/(1+e^(-Δx))                            |
| `s3` | 1/(1+e^(-Δx/2))                          |
| `s4` | 1/(1+e^(-Δx/3))                          |

The objective signature stays `func([]float64) float64` with 0/1-valued input so the
benchmark, comparison, and constraint machinery is reused unchanged.

### 1.7 MODA — multi-objective variant (`multiobjective.go`)

- Start from `Mayfly/multiobjective.go`, which already provides `MultiObjectiveFunction`,
  `ParetoSolution`, `ParetoArchive`, `NewParetoArchive`, `Add`, and
  `UpdateFromPopulation`. Port and extend; do not rewrite.
- Add the **hypercube grid** MODA needs on top of the archive: partition objective space
  into `NGrid` hypercubes per objective, then
  - **food** = roulette draw from the *least* populated occupied hypercube, weight `1/N^β`
  - **enemy** = roulette draw from the *most* populated hypercube, weight `N^γ`
  - **archive overflow** = delete from the most crowded hypercube, weight `N^δ`
- Proposed defaults `β = 4, γ = 2, δ = 2, NGrid = 10, ArchiveSize = 100`.
  **Verify every one of these against `MODA.m` before locking them in** — they are
  recalled from the MOPSO lineage the paper borrows from, not read off the source.
- New types: `MultiObjectiveConfig`, `MultiObjectiveResult` (carrying the final archive),
  plus `ExportParetoCSV` / `ExportParetoJSON` mirroring `monitoring.go`.
- Keep `OptimizeMultiObjective` a separate entry point rather than overloading `Result`.

---

## 2. Target repository layout

```
Dragonfly/
├── go.mod  LICENSE  .gitignore
├── README.md  CHANGELOG.md  PLAN.md  CLAUDE.md  AGENTS.md
├── justfile  .golangci.toml  treefmt.toml
├── .github/workflows/{test.yml,release.yml}
│
│  ── core ──
├── dragonfly.go        package doc, Optimize, OptimizeContext
├── types.go            Config, Dragonfly, Best, Result, ConvergenceConfig, sentinels
├── config.go           NewDefaultConfig and the preset factories
├── config_loader.go    JSON load/save, ValidateConfig, presets, AutoTuneConfig
├── swarm.go            neighbourhood scan, S/A/C/F/E vectors, radius schedule
├── weights.go          adaptive w, s, a, c, f, e schedules
├── levy.go             Lévy flight
├── functions.go        benchmark functions (ported from Mayfly)
├── helpers.go          unifrnd, randn, maxVec/minVec, sanitize*, effective*, validate*
├── constraints.go      constraintEvaluator, Deb feasibility rules, penalty methods
├── convergence.go      convergenceTracker
├── lifecycle.go        RunOption, observers, Logger
├── monitoring.go       slog events, convergence CSV/JSON export
│
│  ── variants ──
├── binary.go           BDA: transfer functions, bit flipping
├── multiobjective.go   MODA: Pareto archive, hypercube grid, food/enemy selection
├── variants.go         AlgorithmVariant interface, DA/BDA/MODA impls, VariantBuilder
├── selector.go         AlgorithmSelector, ClassifyProblem, RecommendForBenchmark
├── comparison.go       ComparisonRunner, Wilcoxon, Friedman, CSV/JSON export
│
│  ── parallelism ──
├── parallel.go          evaluationPool, batchBest, mergeBest, effectiveMaxWorkers
├── parallel_compute.go  parallelFor
├── parallel_phases.go   prepareSwarmStep, prepareLevyStep  (own the RNG, evaluate nothing)
├── parallel_variants.go evaluateParallelStep, evaluateParallelBinary
│
│  ── tests, siblings at root, package dragonfly ──
├── swarm_test.go weights_test.go levy_test.go binary_test.go multiobjective_test.go
├── helpers_test.go constraints_test.go convergence_test.go lifecycle_test.go
├── monitoring_test.go variants_test.go selector_test.go comparison_test.go
├── config_loader_test.go types_test.go functions_test.go parallel_test.go
├── benchmark_test.go regression_test.go integration_test.go
├── performance_benchmark_test.go
├── example_test.go      (package dragonfly_test — the only black-box file)
│
├── features/    *.feature, run by godog
├── docs/        README.md, algorithms/, api/, benchmarks.md, research.md, …
├── examples/    each subdirectory its own module, with a replace directive
└── scripts/
```

---

## Phase 1: Repository scaffold

### 1.1 Tooling

- [x] `.golangci.toml` — copied from Mayfly with the gci/goimports local prefix swapped to
      `github.com/MeKo-Christian/dragonfly`, and **without** Mayfly's per-file complexity
      exemptions
- [x] `treefmt.toml` — gofumpt → gci for Go, prettier for md/json/yaml, taplo for toml,
      shellcheck → shfmt for shell
- [x] `.gitignore` — Mayfly's, minus the WebAssembly demo entries
- [x] `LICENSE` — MIT, "Copyright (c) 2026 Christian-W. Budde"
- [x] `.github/workflows/test.yml` — format / lint / test (Go 1.23 + 1.24 matrix) / benchmark
- [x] `.github/workflows/release.yml` — semver validation, metadata checks, module-path
      assertion against `github.com/MeKo-Christian/dragonfly`
- [x] `justfile` — Mayfly's recipes minus the wasm ones
- [x] `go.mod` — `module github.com/MeKo-Christian/dragonfly`, `go 1.23.3`

**Rationale**: Getting the linter and formatter contract in place *before* the first line
of algorithm code means the house style is enforced from commit one, rather than being
retrofitted across thirty files later. Dropping Mayfly's complexity exemptions is
deliberate: they exist there to grandfather a 1200-line `OptimizeContext`, and inheriting
them would grandfather a problem we do not have yet.

### 1.2 Documents

- [x] `PLAN.md` — this file
- [x] `README.md` — skeleton following Mayfly's section order
- [x] `CHANGELOG.md` — Keep a Changelog 1.1.0 + SemVer, `## [Unreleased]` only
- [x] `CLAUDE.md` — development guide: commands, architecture, conventions, pitfalls
- [x] `AGENTS.md` — the short six-section agent brief
- [x] `git init`, initial commit

**Gate**: `just check` runs and passes (trivially, with no Go files yet).

---

## Phase 2: Core standard DA

- [x] `types.go` — `Config`, `Dragonfly`, `Best`, `Result`, `ConvergenceConfig`,
      `TerminationReason`, `BoundaryMethod`, `WeightAuto` sentinel
- [x] `helpers.go` — `unifrnd`, `randn`, `maxVec`, `minVec`, `sanitizeVec`, `sanitizeCost`,
      `effectiveXxx(config)` resolvers, `validateXxx(config) error`
- [x] `levy.go` + `levy_test.go`
- [x] `weights.go` + `weights_test.go`
- [x] `swarm.go` + `swarm_test.go`
- [x] `config.go` — `NewDefaultConfig` and friends
- [x] `functions.go` + `functions_test.go` — ported from Mayfly
- [x] `dragonfly.go` — package doc, `Optimize`, sequential `OptimizeContext`
- [x] `types_test.go`, `helpers_test.go`

**Gate**: converges on Sphere, Rastrigin, Ackley, and Rosenbrock at d = 10 within the
tolerances that will be recorded in `regression_test.go`; two runs with the same seed
produce identical `Result` values.

**Rationale**: The swarm primitives and the weight schedules are the parts a reader will
check against the paper, so they get their own files and their own hand-computed unit
tests. `swarm_test.go` in particular should build a 3-dragonfly, 2-dimensional swarm and
compare S, A, C, F, E and one full ΔX step against values worked out by hand from `DA.m`
— that single test is what catches a Euclidean-vs-per-dimension neighbour mistake or a
sign error in the enemy term, and neither of those shows up as an obvious failure in an
end-to-end convergence test.

---

## Phase 3: Lifecycle, constraints, convergence, monitoring

- [x] `lifecycle.go` — `RunOption` as a struct wrapping a private error-returning apply
      func, `resolveRunOptions` rejecting the zero value, `WithInitialPopulation`,
      `WithProgressObserver`, `WithPopulationObserver`, `WithLogger`
- [x] `constraints.go` — ported from Mayfly with unchanged semantics
- [x] `convergence.go` — `convergenceTracker.observe`
- [x] `monitoring.go` — slog events, `ExportConvergenceCSV`, `ExportConvergenceJSON`
- [x] `config_loader.go` — JSON load/save, `ValidateConfig`, presets, `AutoTuneConfig`
- [x] Matching `_test.go` files for each

**Gate**: the target-cost stop refuses to fire on an infeasible incumbent; observers
receive deep copies and run synchronously on the caller's goroutine; a cancelled context
returns `ctx.Err()`.

**Rationale**: These four files are close to pure ports — the semantics do not depend on
which metaheuristic sits underneath. Doing them as a batch, before parallelism, means the
parallel work in Phase 4 has real observers and real cancellation to test against.

---

## Phase 4: Deterministic parallelism

- [ ] `parallel.go` — `evaluationPool`, `batchBest` (stable-index selection),
      `mergeBest`, `effectiveMaxWorkers`
- [ ] `parallel_compute.go` — `parallelFor`
- [ ] `parallel_phases.go` — `prepareSwarmStep`, `prepareLevyStep`
- [ ] `parallel_variants.go` — `evaluateParallelStep`
- [ ] `parallel_test.go` including `TestParallelIsDeterministicForSeedAcrossSchedules`
      and `TestParallelCancellationDoesNotCommitPartialBatch`
- [ ] `BenchmarkNeighbourScan`

**Gate**: a seeded parallel run is bit-identical to the sequential one; cancellation
commits no partial batch; the neighbour-scan benchmark shows a real speedup at n ≥ 100.

**Rationale**: DA has a wrinkle Mayfly does not. The radius, the food source, the enemy,
and every neighbour set depend on the whole swarm, so the prepare phase is two passes:
first compute `r`, the weights, food and enemy; then, per dragonfly, scan for neighbours
and build ΔX. Both passes touch the RNG and stay sequential; only the objective evaluation
of the resulting positions fans out. Separately, the neighbour scan is O(n²·d) and is the
actual hot spot for large swarms — it can be parallelised safely precisely *because* it
draws no random numbers, and `BenchmarkNeighbourScan` exists to prove that is worth doing
before the complexity is added.

---

## Phase 5: BDA — binary variant

- [x] `binary.go` — `TransferFunction` type, the v1–v4 / s1–s4 registry, bit-flip update
- [x] `NewBinaryConfig()`
- [ ] `evaluateParallelBinary`
- [x] `binary_test.go` — transfer-function curves at sampled points, bit-flip statistics
- [x] `examples/feature_selection/` — a worked feature-selection problem

**Gate**: each transfer function matches its analytic form at sampled points; a
knapsack-style toy problem reaches the known optimum.

---

## Phase 6: MODA — multi-objective variant

- [ ] `multiobjective.go` — ported archive plus the hypercube grid
- [ ] `MultiObjectiveConfig`, `MultiObjectiveResult`, `OptimizeMultiObjective`
- [ ] `ExportParetoCSV`, `ExportParetoJSON`
- [ ] ZDT1–ZDT3 and Schaffer test problems added to `functions.go`
- [ ] `multiobjective_test.go`
- [ ] `examples/multiobjective/`

**Gate**: recovers the known Pareto fronts of ZDT1 and ZDT3; the archive never exceeds
`ArchiveSize`; every archived solution is mutually non-dominated.

**Rationale**: The non-domination invariant is the one that silently breaks during archive
maintenance, so it gets asserted on every archive mutation in tests rather than only at the
end of a run.

---

## Phase 7: Framework layer

- [ ] `variants.go` — `AlgorithmVariant` interface, `DAVariant` / `BDAVariant` /
      `MODAVariant`, the registry, `NewVariant`, `ListVariants`, `GetAllVariants`,
      `VariantBuilder`
- [ ] `selector.go` — `AlgorithmSelector`, `ProblemCharacteristics`, `ClassifyProblem`,
      `RecommendForBenchmark`
- [ ] `comparison.go` — `ComparisonRunner`, Wilcoxon signed-rank, Friedman, CSV/JSON export
- [ ] `variants_test.go`, `selector_test.go`, `comparison_test.go`

**Gate**: `ComparisonRunner` derives paired, identical seeds in sequential and parallel
mode; `GetAllVariants()` returns a stable canonical order.

**Rationale**: With only three variants this layer is thinner than Mayfly's seven-variant
version, but it is what makes the two libraries comparable head-to-head — the eventual
"DA vs MA on the same benchmark" table needs both sides to speak the same
`ComparisonResult`.

---

## Phase 8: BDD, regression, benchmarks

- [ ] `features/boundary_handling.feature` — wrap vs clamp vs reflect
- [ ] `features/configuration_validation.feature`
- [ ] `features/constraint_handling.feature`
- [ ] `features/optimization_convergence.feature`
- [ ] `features/variant_execution.feature`
- [ ] `integration_test.go` — godog wiring, `TestFeatures`
- [ ] `regression_test.go` — `RegressionBaseline` entries with tolerated degradation factors
- [ ] `benchmark_test.go` — `BenchmarkOptimize<Function>_<Variant>`,
      `BenchmarkDimensionScaling`, `BenchmarkPopulationSize`
- [ ] `performance_benchmark_test.go` — `BenchmarkOptimizeBaseline`, the profiling anchor
- [ ] `example_test.go` — `ExampleOptimize`, `ExampleOptimizeContext`, `ExampleNewBuilder`
- [ ] `go.mod`: add `github.com/cucumber/godog`

**Rationale**: Baselines encode *tolerated degradation factors*, not exact expected values.
A stochastic optimizer's output is not a golden file; the question a regression test can
usefully answer is "did this change make the algorithm meaningfully worse", and that is a
statistical question with a tolerance attached.

---

## Phase 9: Documentation

- [ ] `README.md` in full: Overview → Quick Start → Algorithm Variants table → Intelligent
      Selection → Statistical Comparison → Benchmark Functions → Documentation index →
      Running Examples → Build Commands → Research & Citations (with an Algorithm
      Implementation Map: File | Algorithm/Operator | Reference) → Performance → Development
      Status → Contributing → License
- [ ] `docs/README.md` — documentation hub with a navigation guide
- [ ] `docs/algorithms/{standard-da,bda,moda}.md` on Mayfly's fixed skeleton
      (Research Reference → Overview → Key Innovations → Usage Examples → Parameters →
      Benefits → Performance → When to Use → Parameter Tuning Guide → vs Other Variants →
      Related Documentation)
- [ ] `docs/api/{quick-reference,configuration,run-lifecycle,comparison-framework}.md`
- [ ] `docs/benchmarks.md`, `docs/research.md` (with BibTeX), `docs/performance.md`,
      `docs/releasing.md`
- [ ] `examples/` — each subdirectory its own module with a `replace` directive

---

## Phase 10: Release preparation

- [ ] 80%+ statement coverage
- [ ] `just security` (nancy) clean
- [ ] `CHANGELOG.md` `## [0.1.0]` entry
- [ ] `just release-check 0.1.0` passes
- [ ] Tag `v0.1.0`; the Go module proxy handles publication

---

## Deferred — recorded so the roadmap stays honest

### Inherited benchmark defects (shared with Mayfly)

`functions.go` was ported verbatim from Mayfly, which means it inherited two edge-case
defects. They are tracked in both repositories and should be fixed in both together, so the
two benchmark suites stay numerically comparable. See the note at the top of
`../Mayfly/PLAN.md` for the full write-up.

- [ ] `Levy([])` panics with `index out of range [0] with length 0` — the only benchmark
      function that panics rather than returning a value
- [ ] Empty-input handling is inconsistent: 12 functions return `0`, `Ackley` and `HappyCat`
      return `NaN` (division by `n`), `Levy` panics. Choose one convention, document it in
      the package comment, and assert it for all 15 functions in one table-driven test

**Not defects — checked and cleared 2026-08-23**, recorded so they are not "fixed" into real
bugs later: `Levy`'s `sin(π·wᵢ + 1)` is the standard definition (the `+1` is correct; it is
not a mistranscription of `π/4`), and `ExpandedSchafferF6`'s wrap-around pair
`g(x[n-1], x[0])` is part of the CEC definition of the expanded function.

- [ ] CEC2017 / CEC2020 benchmark suites
- [ ] WebAssembly browser demo and the GitHub Pages workflow
- [ ] `CONTRIBUTING.md`, issue and PR templates
- [ ] Hybrid and improved DA variants: memory-based MDA, hybrid HDA, chaotic DA, quantum DA
- [ ] A head-to-head `dragonfly` vs `mayfly` comparison harness and results table

---

## References

- Mirjalili, S. (2016). Dragonfly algorithm: a new meta-heuristic optimization technique
  for solving single-objective, discrete, and multi-objective problems.
  *Neural Computing and Applications*, 27(4), 1053–1073.
  doi:[10.1007/s00521-015-1920-1](https://doi.org/10.1007/s00521-015-1920-1)
- Reynolds, C. W. (1987). Flocks, herds and schools: A distributed behavioral model.
  *ACM SIGGRAPH Computer Graphics*, 21(4), 25–34. — the origin of separation, alignment
  and cohesion.
- Mantegna, R. N. (1994). Fast, accurate algorithm for numerical simulation of Lévy stable
  stochastic processes. *Physical Review E*, 49(5), 4677–4683.
- Deb, K. (2000). An efficient constraint handling method for genetic algorithms.
  *Computer Methods in Applied Mechanics and Engineering*, 186(2–4), 311–338.
  — the feasibility rules used in `constraints.go`.
- Coello Coello, C. A., Pulido, G. T., & Lechuga, M. S. (2004). Handling multiple objectives
  with particle swarm optimization. *IEEE Transactions on Evolutionary Computation*, 8(3),
  256–279. — the hypercube archive MODA borrows.
