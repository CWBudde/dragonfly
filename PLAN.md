# Dragonfly Algorithm (Go) — Implementation Plan

Module: `github.com/CWBudde/dragonfly`
Package: `dragonfly` (flat, at the repository root)
Status: **Phases 1–10 complete**, released as `v0.1.0`; **Phase 11 is open**. The unchecked
boxes below the phases are the deferred and open items.

This document is the roadmap. It is organised the same way as the sibling
[Mayfly](https://github.com/cwbudde/mayfly) project's `PLAN.md`: numbered phases,
`- [ ]` / `- [x]` task checkboxes, and a bolded `**Rationale**:` paragraph after each
subsection explaining _why_ the task exists. Update the checkboxes as work lands.

---

## 0. Project Overview

A dependency-free Go implementation of Seyedali Mirjalili's **Dragonfly Algorithm (DA)**,
covering all three variants from the original paper:

| Variant  | Problem class                       | Entry point                                |
| -------- | ----------------------------------- | ------------------------------------------ |
| **DA**   | Single-objective, continuous        | `Optimize` / `OptimizeContext`             |
| **BDA**  | Single-objective, binary / discrete | `OptimizeContext` with a transfer function |
| **MODA** | Multi-objective, continuous         | `OptimizeMultiObjective`                   |

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

Primary source: Mirjalili, S. (2016). _"Dragonfly algorithm: a new meta-heuristic
optimization technique for solving single-objective, discrete, and multi-objective
problems."_ **Neural Computing and Applications** 27(4), 1053–1073.
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

| Name | Form                                |
| ---- | ----------------------------------- |
| `v1` | \|erf(√π/2 · Δx)\|                  |
| `v2` | \|tanh(Δx)\|                        |
| `v3` | \|Δx / √(Δx²+1)\| _(paper default)_ |
| `v4` | \|(2/π)·arctan((π/2)·Δx)\|          |
| `s1` | 1/(1+e^(-2Δx))                      |
| `s2` | 1/(1+e^(-Δx))                       |
| `s3` | 1/(1+e^(-Δx/2))                     |
| `s4` | 1/(1+e^(-Δx/3))                     |

The objective signature stays `func([]float64) float64` with 0/1-valued input so the
benchmark, comparison, and constraint machinery is reused unchanged.

### 1.7 MODA — multi-objective variant (`multiobjective.go`)

- Start from `Mayfly/multiobjective.go`, which already provides `MultiObjectiveFunction`,
  `ParetoSolution`, `ParetoArchive`, `NewParetoArchive`, `Add`, and
  `UpdateFromPopulation`. Port and extend; do not rewrite.
- Add the **hypercube grid** MODA needs on top of the archive: partition objective space
  into `NGrid` hypercubes per objective, then
  - **food** = roulette draw from the _least_ populated occupied hypercube, weight `1/N^β`
  - **enemy** = roulette draw from the _most_ populated hypercube, weight `N^γ`
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
      `github.com/CWBudde/dragonfly`, and **without** Mayfly's per-file complexity
      exemptions
- [x] `treefmt.toml` — gofumpt → gci for Go, prettier for md/json/yaml, taplo for toml,
      shellcheck → shfmt for shell
- [x] `.gitignore` — Mayfly's, minus the WebAssembly demo entries
- [x] `LICENSE` — MIT, "Copyright (c) 2026 Christian-W. Budde"
- [x] `.github/workflows/test.yml` — format / lint / test (Go 1.23 + 1.24 matrix) / benchmark
- [x] `.github/workflows/release.yml` — semver validation, metadata checks, module-path
      assertion against `github.com/CWBudde/dragonfly`
- [x] `justfile` — Mayfly's recipes minus the wasm ones
- [x] `go.mod` — `module github.com/CWBudde/dragonfly`, `go 1.23.3`

**Rationale**: Getting the linter and formatter contract in place _before_ the first line
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

- [x] `parallel.go` — `evaluationPool`, `batchBest` (stable-index selection),
      `mergeBest`, `effectiveMaxWorkers`
- [x] `parallel_compute.go` — `parallelFor`
- [x] `parallel_phases.go` — `prepareSwarmStep`, `prepareLevyStep`
- [x] `parallel_variants.go` — `evaluateParallelStep`
- [x] `parallel_test.go` including `TestParallelIsDeterministicForSeedAcrossSchedules`
      and `TestParallelCancellationDoesNotCommitPartialBatch`
- [x] `BenchmarkNeighbourScan`

**Gate**: a seeded parallel run is bit-identical to the sequential one; cancellation
commits no partial batch; the neighbour-scan benchmark shows a real speedup at n ≥ 100.

**Rationale**: DA has a wrinkle Mayfly does not. The radius, the food source, the enemy,
and every neighbour set depend on the whole swarm, so the prepare phase is two passes:
first compute `r`, the weights, food and enemy; then, per dragonfly, scan for neighbours
and build ΔX. Both passes touch the RNG and stay sequential; only the objective evaluation
of the resulting positions fans out. Separately, the neighbour scan is O(n²·d) and is the
actual hot spot for large swarms — it can be parallelised safely precisely _because_ it
draws no random numbers, and `BenchmarkNeighbourScan` exists to prove that is worth doing
before the complexity is added.

---

## Phase 5: BDA — binary variant

- [x] `binary.go` — `TransferFunction` type, the v1–v4 / s1–s4 registry, bit-flip update
- [x] `NewBinaryConfig()`
- [x] `evaluateParallelBinary`
- [x] `binary_test.go` — transfer-function curves at sampled points, bit-flip statistics
- [x] `examples/feature_selection/` — a worked feature-selection problem

**Gate**: each transfer function matches its analytic form at sampled points; a
knapsack-style toy problem reaches the known optimum.

---

## Phase 6: MODA — multi-objective variant

- [x] `multiobjective.go` — ported archive plus the hypercube grid
- [x] `MultiObjectiveConfig`, `MultiObjectiveResult`, `OptimizeMultiObjective`
- [x] `ExportParetoCSV`, `ExportParetoJSON`
- [x] ZDT1–ZDT3 and Schaffer test problems added to `functions.go`
- [x] `multiobjective_test.go`
- [x] `examples/multiobjective/`

**Gate**: recovers the known Pareto fronts of ZDT1 and ZDT3; the archive never exceeds
`ArchiveSize`; every archived solution is mutually non-dominated.

**Rationale**: The non-domination invariant is the one that silently breaks during archive
maintenance, so it gets asserted on every archive mutation in tests rather than only at the
end of a run.

---

## Phase 7: Framework layer

- [x] `variants.go` — `AlgorithmVariant` interface, `DAVariant` / `BDAVariant` /
      `MODAVariant`, the registry, `NewVariant`, `ListVariants`, `GetAllVariants`,
      `VariantBuilder`
- [x] `selector.go` — `AlgorithmSelector`, `ProblemCharacteristics`, `ClassifyProblem`,
      `RecommendForBenchmark`
- [x] `comparison.go` — `ComparisonRunner`, Wilcoxon signed-rank, Friedman, CSV/JSON export
- [x] `variants_test.go`, `selector_test.go`, `comparison_test.go`

**Gate**: `ComparisonRunner` derives paired, identical seeds in sequential and parallel
mode; `GetAllVariants()` returns a stable canonical order.

**Rationale**: With only three variants this layer is thinner than Mayfly's seven-variant
version, but it is what makes the two libraries comparable head-to-head — the eventual
"DA vs MA on the same benchmark" table needs both sides to speak the same
`ComparisonResult`.

---

## Phase 8: BDD, regression, benchmarks

- [x] `features/boundary_handling.feature` — wrap vs clamp vs reflect
- [x] `features/configuration_validation.feature`
- [x] `features/constraint_handling.feature`
- [x] `features/optimization_convergence.feature`
- [x] `features/variant_execution.feature`
- [x] `integration_test.go` — godog wiring, `TestFeatures`
- [x] `regression_test.go` — `RegressionBaseline` entries with tolerated degradation factors
- [x] `benchmark_test.go` — `BenchmarkOptimize<Function>_<Variant>`,
      `BenchmarkDimensionScaling`, `BenchmarkPopulationSize`
- [x] `performance_benchmark_test.go` — `BenchmarkOptimizeBaseline`, the profiling anchor
- [x] `example_test.go` — `ExampleOptimize`, `ExampleOptimizeContext`, `ExampleNewBuilder`
- [x] `go.mod`: add `github.com/cucumber/godog`

**Rationale**: Baselines encode _tolerated degradation factors_, not exact expected values.
A stochastic optimizer's output is not a golden file; the question a regression test can
usefully answer is "did this change make the algorithm meaningfully worse", and that is a
statistical question with a tolerance attached.

---

## Phase 9: Documentation

- [x] `README.md` in full: Overview → Quick Start → Algorithm Variants table → Intelligent
      Selection → Statistical Comparison → Benchmark Functions → Documentation index →
      Running Examples → Build Commands → Research & Citations (with an Algorithm
      Implementation Map: File | Algorithm/Operator | Reference) → Performance → Development
      Status → Contributing → License
- [x] `docs/README.md` — documentation hub with a navigation guide
- [x] `docs/algorithms/{standard-da,bda,moda}.md` on Mayfly's fixed skeleton
      (Research Reference → Overview → Key Innovations → Usage Examples → Parameters →
      Benefits → Performance → When to Use → Parameter Tuning Guide → vs Other Variants →
      Related Documentation)
- [x] `docs/api/{quick-reference,configuration,run-lifecycle,comparison-framework}.md`
- [x] `docs/benchmarks.md`, `docs/research.md` (with BibTeX), `docs/performance.md`,
      `docs/releasing.md`
- [x] `examples/` — each subdirectory its own module with a `replace` directive

---

## Phase 10: Release preparation

- [x] 80%+ statement coverage — 96.3%
- [x] `just security` clean. Split into `just audit` (nancy, production tree) and
      `just vuln` (`govulncheck`, which also covers the test-only dependency tree and
      reports by reachability); `security` runs both, and `install-tools` now installs
      them. Worth knowing what the nancy half actually proves: the library is
      stdlib-only, so `go list -deps ./...` yields no third-party packages and nancy
      reports `Audited Dependencies: 0`. That is the intended state, not a passing scan
      — the recipe is there to catch the first real dependency that is ever added.
      `govulncheck` reports nothing reachable; its findings are all in the local Go
      toolchain's own stdlib, which is a property of the build machine rather than of
      this module. Documented in `docs/releasing.md`.
- [x] `CHANGELOG.md` `## [0.1.0]` entry, including a `Known limitations` section that
      names the unverified MODA and BDA constants rather than leaving them implied
- [x] `just release-check 0.1.0` passes
- [x] Tag `v0.1.0`, annotated and pushed to `github.com/CWBudde/dragonfly`; the Go
      module proxy handles publication from there.
- [x] Module path corrected to `github.com/CWBudde/dragonfly`. It was
      `github.com/MeKo-Christian/dragonfly`, which matched no repository that exists, so
      `go get` could not have resolved the module under either name — and `release.yml`
      asserted `go list -m` against that same wrong constant, so the release workflow
      agreed with the error instead of catching it. The repository was briefly
      `CWBudde/Dragonfly`, which forced an explicit `dragonfly` import alias everywhere
      (goimports requires one when the final path element and the package name differ);
      it has since been renamed to lowercase and the aliases are gone.
- [x] `github.com/CWBudde/Dragonfly@v0.1.0` — the capitalised path — was published to
      the module proxy before the rename and is cached there permanently, with its
      hashes recorded in the checksum database. Module paths are case-sensitive to the
      proxy, so that is a distinct module from the lowercase one and cannot be
      withdrawn. Harmless, but a search may surface it. The README now identifies the
      lowercase path as canonical and warns against depending on the cached capitalised path.

---

## Phase 11: Post-v0.1.0 correctness and fidelity remediation

An adversarial multi-agent audit on 2026-08-23 compared the implementation and tests with
the paper and the author's official `DA.m`, `BDA.m`, and `MODA.m`. The result is not a list
of cosmetic follow-ups: several passing tests encode behavior that matches neither source,
and the implementation currently mixes paper formulas, MATLAB control flow, and local
extensions while the documentation often says the sources agree.

The policy for this phase is **paper behavior by default, with an explicit MATLAB
compatibility mode** wherever the sources genuinely disagree. Locally improved behavior
remains available only when it is named and documented as an extension.

### 11.1 Canonical behavior and public contracts

- [x] Add `FidelityMode` with paper-default and MATLAB-compatible values; isolate the
      separation sign, one-neighbour fallback, food-distance branching, schedule, boundary,
      initialization, and evaluation-budget differences behind that explicit policy.
      The MATLAB path now follows the reference evaluate-before-move lifecycle in DA, BDA and
      MODA, consumes `NPop × iterations` objective evaluations, leaves the final movement
      unevaluated, preserves evaluated observer snapshots, uses the strict-interior DA movement
      enemy, and applies pre-move wrap/reset followed by post-move hard clamping. Its reference
      boundary order intentionally takes precedence over `BoundaryMethod`; paper mode retains
      the configurable boundary extension and evaluate-after-move lifecycle.
- [x] Add named MODA archive policies for the paper's segment probabilities, MATLAB's
      objective-span/20 density ranking, and the existing MOPSO-style grid. Keep the current
      `4/2/2`, `NGrid = 10` behavior only as an explicitly selected extension.
- [x] Replace false seed metadata with an explicit seed contract: library-created seeds are
      reproducible and reported as known; caller-supplied `*rand.Rand` state is reported as
      unknown unless the library itself constructed it from a configured seed.
- [x] Record the official-source hashes and replace the remaining unverified claims: Lévy
      `β = 1.5` and scale `0.01`, BDA's `±6` clamp, and MODA archive size 100 are verified;
      the current MODA grid/exponents are not reference values.

**Rationale**: A research implementation cannot call three conflicting behaviors jointly
faithful. The mode and policy names make every trajectory say which specification produced
it, while retaining the released extension for reproducibility.

### 11.2 DA and shared swarm mechanics

- [x] Correct neighbour exclusion to the promised component-wise `all(distance != 0)` rule
      and add the missing shared-coordinate fixture.
- [x] Gate the enemy term independently: an enemy outside the radius contributes exactly
      zero even when the food is inside it.
- [x] Replace the current paper/MATLAB hybrid for separation, exactly one neighbour, and the
      two update branches with hand-computed mode-specific transitions.
- [x] Feed schedules the iteration indices they actually specify and assert the first and
      final coefficients consumed by a real run, not the unused `t = T` helper value.
- [x] Validate the bound span itself as finite; replace iterative reflection with a
      constant-time fold whose step sign follows wall-crossing parity.
- [x] Never return the synthetic zero-position sentinel when every cost is non-finite;
      return a named error if no usable incumbent exists.
- [x] Check cancellation before initialization and after every sequential evaluation batch,
      including the final batch; remove hidden package-global RNG fallbacks.

**Rationale**: These faults change neighbourhood membership, acceleration, or the returned
solution. Convergence on Sphere does not exercise any of them.

### 11.3 BDA and MODA

- [x] Make BDA one swarm as required by the paper and `BDA.m`: every `i != j` is a neighbour
      and every iteration uses the unconditional five-factor step, without DA radius or
      Lévy branching.
- [x] Apply V-shaped transfers by complementing the current bit and S-shaped transfers by
      assigning zero or one from the sampled probability; add exact transition tests.
- [x] Make paper MODA selection proportional to `1/N` and `N`, implement MATLAB's direct
      reciprocal/density ranking in compatibility mode, and retain MOPSO exponents only in
      the named extension.
- [x] Copy multi-objective results immediately, establish objective arity before archive
      mutation, and reject mixed arity, non-finite objectives, or invalid violations at the
      public archive boundary.
- [x] Prevent grid-key overflow and normalize extreme roulette weights stably instead of
      silently falling back to uniform selection.
- [x] Count only durable archive changes as stagnation improvements; reject `UseBinary` and
      ignored `MinImprovement` settings in MODA and validate malformed export archives.

**Rationale**: The released BDA update is a continuous-DA hybrid, and the released MODA
selection constants match neither the paper nor `MODA.m`. These are variant-definition
errors, not tuning disagreements.

### 11.4 Configuration, statistics, exports, framework, and CI

- [x] Reject unknown and trailing JSON configuration data and document parallel safety for
      both objective and constraint callbacks.
- [x] Make comparison targets support zero and negative values; preserve setup errors,
      separate failed runs from numerical samples, and report statistical availability.
- [x] Add exact small-sample/tie-corrected Wilcoxon, tie-corrected Friedman, and adjusted
      pairwise significance; validate them against independent reference fixtures.
- [x] Give non-finite/unavailable JSON values an explicit wire representation and write every
      export through an atomic same-directory temporary file.
- [x] Reject MODA in the single-objective builder and unsupported discrete multi-objective
      recommendations rather than returning unusable configurations.
- [x] Enforce the 80% coverage gate, pinned lint/tool versions, nested example and WASM
      builds, race tests, and security checks in CI.
      Coverage, all formatter/linter/security tool versions, the Go 1.23/1.26 matrix, race
      tests and nested example/WASM builds are enforced. Security scanning runs in ordinary CI
      and in a separate weekly/manual workflow with pinned Nancy and `govulncheck` versions.

**Rationale**: A forgiving comparison must not turn failed runs into invented significance,
and an exporter must not destroy the previous report before discovering that encoding failed.

### 11.5 Verification and release

- [x] Add paper-mode and MATLAB-mode state-transition fixtures for all three variants,
      sequential/parallel bit-identity tests, hostile numeric/cancellation cases, and archive
      invariant checks for every mutation.
- [x] Re-run the multi-seed regression and 30-dimensional ZDT studies after correction;
      change degradation tolerances only from measured results, never merely to pass CI.
      The 15-seed DA/BDA regression suite passed every baseline without changing a tolerance.
      The paper-default five-dimensional MODA gate passed 14/15 ZDT1 seeds and 12/15 ZDT3
      seeds. The 30-dimensional, five-seed ZDT1–ZDT3 matrix found that none of the paper,
      MATLAB or named legacy profiles met the strict recovery bar; the valid non-dominated
      archives remain materially off-front. A separate 240-run study remeasured all eight BDA
      transfer functions in both fidelity modes. Raw deterministic CSV, generated summaries,
      exact configurations, source hashes and evaluation counts are retained under
      `docs/measurements/`.
- [ ] Update the algorithm/API/research documentation and `CHANGELOG.md`, then release the
      corrected contracts and changed stochastic trajectories as `v0.2.0` after
      `just check-race` and the release gates pass.
      **Progress:** the documentation, changelog, source-hashed studies and local v0.2.0
      release content are complete. Tagging, pushing and module-proxy publication intentionally
      remain open external release actions.

**Audit baseline**: before Phase 11, `go test ./...`, `go test -short -race ./...`, and
`go vet ./...` pass. Local lint could not start because the installed golangci-lint does not
recognize the configured linter, which is itself why the toolchain must be pinned. Phase 11
resolved that gap with exact version checks; the pinned linter reports zero issues.

---

## Deferred — recorded so the roadmap stays honest

### Found while documenting (2026-08-23)

- [x] `EnemyCutoffFraction` is dead at its default. `e = mc`, and `mc` already reaches
      zero at `t = T/2`, so the `0.75·T` cutoff can never fire -- only a fraction below
      `0.5` changes behaviour. This matches the paper (both rules are in `DA.m`), so it is
      not a defect, but the field advertised a control it does not have at its default.
      **Resolved 2026-08-23: documented as inert rather than dropped.** The field's godoc
      now says so outright, `scheduledEnemyWeight` repeats it at the call site, and
      `TestEnemyCutoffFractionIsInertAtDefault` pins the property -- every fraction from
      `0.5` up produces an identical enemy schedule. The cutoff is also inert whenever
      `EnemyWeight` is pinned off `WeightAuto`, which the godoc now states too. It stays
      because the paper and `DA.m` carry both rules and a reader checking the code against
      them should find it where they expect it.
- [x] MODA recovers the ZDT fronts only at low dimensionality. At the ZDT suite's original
      30 dimensions (NPop 100, 1000 iterations) the archive is non-dominated but its lowest
      `f2` is 1.39, well off the true front. The Phase 6 gate is met at the tests' `d = 5`
      (median distance to front 0.000). **Measured 2026-08-23; the figure is confirmed and
      the cause is a stall, not an unfinished run.** ZDT1 with NPop 100, five seeds, median
      distance to the analytic front: `d = 5` -> 0.000, `d = 10` -> 0.38-0.92,
      `d = 20` -> 1.00-1.51, `d = 30` -> 1.56-1.68 (lowest `f2` 1.26-1.38, so 1.39 is within
      seed noise). It is not a `d = 30` cliff -- the degradation is already visible at
      `d = 10`. Raising the budget to 10000 iterations improves `d = 30` to a median of
      0.38-0.46 and `d = 10` to 0.027-0.043, but instrumented trajectories show why that is
      not continued convergence: the running minimum of ZDT1's `g` term freezes after
      roughly the first half of every run (at `d = 30, T = 40000` the best `f2` is unchanged
      from iteration 2000 to iteration 40000), and a 4x population buys almost nothing. The
      schedules are `t/T`-relative, so a longer `T` only stretches the exploratory phase
      before the same stall. The likely mechanism is `convergenceFactor` (weights.go): `mc`
      reaches zero at `t = T/2`, zeroing `s`, `a`, `c` and `e`, after which only inertia and
      the food term move a dragonfly -- and the food is drawn from the _sparsest_ occupied
      hypercube, which seeks spread rather than convergence. The second half of a run can
      redistribute along the archive it has but cannot lower `g`. Recorded in the docs; no
      parity with the paper's 30-dimensional results is claimed, and the hypercube
      parameters were deliberately not tuned during the investigation since they are still
      unverified against `MODA.m`. **Remeasured 2026-08-24 after the fidelity correction:**
      the five-dimensional paper gate passes 14/15 ZDT1 and 12/15 ZDT3 seeds, but at 30
      dimensions all nine combinations of ZDT1–ZDT3 and paper/MATLAB/legacy profiles score
      0/5 against the strict GD/IGD/hypervolume recovery bar. Median best `g` remains
      1.825–2.804 instead of the true value 1. The source-hashed raw evidence is
      `docs/measurements/v0.2.0-quality.csv`; no 30-dimensional parity is claimed.
- [x] MODA honours neither `EnableParallel` nor `Config.Constraints` nor early stopping.
      All three are wired for DA and BDA only. **Resolved 2026-08-23: all three are wired,
      with two deliberate rejections.** Constraints go through `constrainedDominates`,
      Deb's rules lifted from the total order in `constraints.go` to the partial order an
      archive needs -- `BetterConstrainedCandidate` always names a winner and would
      collapse the front to a point. Early stopping counts an iteration as an improvement
      when the archive accepted a candidate, which is the multi-objective reading of "the
      incumbent improved" and needs no hypervolume machinery. `EnableParallel` fans out the
      objective calls only, through `parallelFor`, and a seeded run stays bit-identical
      with it set or not (`TestOptimizeMultiObjectiveParallelMatchesSequential`).
      `Convergence.TargetCost` and `ConstraintHandlingPenalty` have no multi-objective
      reading and are now rejected by `validateMultiObjectiveConfig` rather than silently
      ignored.
- [x] `SchafferN1` costs 6-10x the ZDT benchmarks despite a trivial 1-D objective -- that
      is archive maintenance at capacity, not search. Relevant to any `ArchiveSize` tuning.
      **Resolved 2026-08-23: 3.8x faster, 6% of the allocations, with the archive contents
      bit-identical.** SchafferN1 went from 75.8 ms / 379,024 allocs per op to 19.7 ms /
      22,534, and is now ~2.5x ZDT1 rather than 6-9x; ZDT1 improved from 11.3 ms / 33,245 to
      8.0 ms / 16,947. Three changes: `updateGrid` reuses each member's index array and
      skips the reassignment sweep entirely when the recomputed bounds are bit-identical,
      so only the newly inserted member is indexed; `occupiedCells` counting-sorts into
      reused buffers instead of building a map and comparison-sorting it, falling back to
      `sort.Slice` when the key space is too large; and `Add` compacts survivors in place
      rather than allocating a fresh slice of every member per accepted insert (documented
      consequence: a `Solutions` slice held across a mutation is not a snapshot). What this
      does _not_ change is the accept rate -- on SchafferN1 nothing dominates anything, so
      the archive is permanently at capacity, and that is a property of the problem's front.
      `TestMODAArchiveSnapshotIsExact` pins two runs element-for-element at exact float
      equality, which is what makes the optimization safe to believe.

### Inherited benchmark defects (shared with Mayfly)

`functions.go` was ported verbatim from Mayfly, which means it inherited two edge-case
defects. They are tracked in both repositories and should be fixed in both together, so the
two benchmark suites stay numerically comparable. See the note at the top of
`../Mayfly/PLAN.md` for the full write-up.

**Resolved 2026-08-23, in both repositories together.** The convention is now one line in
the `functions.go` file comment: every single-objective benchmark function returns `0` for
an empty position vector. `Ackley`, `HappyCat` and `Levy` gained the guard that gives them
that; the other twelve already did it. `TestBenchmarkFunctionsEmptyInput` asserts it for all
fifteen at once, against both a nil slice and an allocated empty one, so a sixteenth function
cannot land without answering the question. It replaces the `empty_vector` subtest that
logged the results and asserted nothing.

- [x] `Levy([])` panics with `index out of range [0] with length 0` — the only benchmark
      function that panics rather than returning a value
- [x] Empty-input handling is inconsistent: 12 functions return `0`, `Ackley` and `HappyCat`
      return `NaN` (division by `n`), `Levy` panics. Choose one convention, document it in
      the package comment, and assert it for all 15 functions in one table-driven test

**Not defects — checked and cleared 2026-08-23**, recorded so they are not "fixed" into real
bugs later: `Levy`'s `sin(π·wᵢ + 1)` is the standard definition (the `+1` is correct; it is
not a mistranscription of `π/4`), and `ExpandedSchafferF6`'s wrap-around pair
`g(x[n-1], x[0])` is part of the CEC definition of the expanded function.

- [ ] CEC2017 / CEC2020 benchmark suites
- [x] WebAssembly browser demo and the GitHub Pages workflow — `examples/wasm-demo/`,
      four pages (Swarm Lab, Pareto, Binary, Shootout), built by
      `scripts/build-wasm-demo.sh` and published by `.github/workflows/wasm-demo-pages.yml`.
      Building it surfaced a real gap and closed it: `OptimizeMultiObjective` took no
      `RunOption` at all, so no caller could watch a MODA run. It now accepts them, with
      `WithArchiveObserver` and `ParetoArchive.GridBounds()` added, and `NeighborhoodRadius`
      and `WithinRadius` exported so a caller reconstructing a neighbourhood asks the
      library rather than reimplementing the per-dimension test.
- [ ] `CONTRIBUTING.md`, issue and PR templates
- [ ] Hybrid and improved DA variants: memory-based MDA, hybrid HDA, chaotic DA, quantum DA
- [ ] A head-to-head `dragonfly` vs `mayfly` comparison harness and results table

---

## References

- Mirjalili, S. (2016). Dragonfly algorithm: a new meta-heuristic optimization technique
  for solving single-objective, discrete, and multi-objective problems.
  _Neural Computing and Applications_, 27(4), 1053–1073.
  doi:[10.1007/s00521-015-1920-1](https://doi.org/10.1007/s00521-015-1920-1)
- Reynolds, C. W. (1987). Flocks, herds and schools: A distributed behavioral model.
  _ACM SIGGRAPH Computer Graphics_, 21(4), 25–34. — the origin of separation, alignment
  and cohesion.
- Mantegna, R. N. (1994). Fast, accurate algorithm for numerical simulation of Lévy stable
  stochastic processes. _Physical Review E_, 49(5), 4677–4683.
- Deb, K. (2000). An efficient constraint handling method for genetic algorithms.
  _Computer Methods in Applied Mechanics and Engineering_, 186(2–4), 311–338.
  — the feasibility rules used in `constraints.go`.
- Coello Coello, C. A., Pulido, G. T., & Lechuga, M. S. (2004). Handling multiple objectives
  with particle swarm optimization. _IEEE Transactions on Evolutionary Computation_, 8(3),
  256–279. — the hypercube archive MODA borrows.
