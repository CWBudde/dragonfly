# Repository Guidelines

The contract for working in this repository, for contributors and coding agents alike.
`CLAUDE.md` is a pointer to this file; `AGENTS.md` is the one to edit.

## Project Overview

Go implementation of Seyedali Mirjalili's **Dragonfly Algorithm (DA)**, covering all three
variants of the original 2016 paper and three published improvements. A dependency-free
metaheuristic optimization library, written to research fidelity against the primary papers
and the author's reference MATLAB code (`DA.m`, `BDA.m`, `MODA.m`).

| Variant  | Problem class                       | Entry point                                |
| -------- | ----------------------------------- | ------------------------------------------ |
| **DA**   | Single-objective, continuous        | `Optimize` / `OptimizeContext`             |
| **BDA**  | Single-objective, binary / discrete | `OptimizeContext` with a transfer function |
| **MODA** | Multi-objective, continuous         | `OptimizeMultiObjective`                   |
| **MHDA** | Single-objective, continuous        | `OptimizeMemoryHybrid`                     |
| **CDA**  | Single-objective, continuous        | `OptimizeChaotic`                          |
| **QGDA** | Single-objective, continuous        | `OptimizeQuantum`                          |

**Current status**: `v0.1.0` is released. The Phase 11 correctness/fidelity remediation and
local `v0.2.0` release preparation are underway; `v0.2.0` is not released. Everything this
document describes under _Architecture & Core Concepts_ is implemented unless `PLAN.md` says
otherwise.

**PLAN.md is the single source of truth for progress.** Before starting work, read PLAN.md
and check which boxes are ticked — including its `Deferred` section, which records the known
gaps honestly. Do not infer status from this file.

**Module**: `github.com/CWBudde/dragonfly`, package `dragonfly`, flat at the repo root
(no `internal/`, no `pkg/`, no `cmd/`). Go 1.23.3. The only direct dependency is
`github.com/cucumber/godog`, and it is test-only. Everything else is standard library.
The lowercase repository component is canonical. The proxy-cached
`github.com/CWBudde/Dragonfly@v0.1.0` path is obsolete, distinct and receives no updates.

## Build & Development Commands

### Using Just (Task Runner)

```bash
just                    # list all recipes
just build              # go build -v ./...

# Tests
just test               # coverage -> coverage.out + coverage.html
just test-quick         # -short, quickest
just test-race          # -race -short, 5m timeout
just test-full          # everything incl. long suites, 10m, no race
just test-integration   # godog features only (go test -run TestFeatures)
just bench              # go test -bench=. -benchmem ./...
just profile-cpu        # pprof around BenchmarkOptimizeBaseline
just profile-mem        # the allocation counterpart

# Examples
just run                # cd examples && go run main.go
just compare            # examples as a comparison pass
just optimize func=Sphere size=30 iter=1000

# Formatting and linting
just setup-deps         # treefmt, golangci-lint v2, gofumpt, gci, shfmt, prettier, taplo
just install-tools      # setup-deps + godoc
just fmt                # treefmt (alias: just treefmt)
just lint               # golangci-lint run --config ./.golangci.toml
just lint-fix           # golangci-lint fmt, then run --fix

# Gates
just check-formatted    # treefmt --fail-on-change
just check-tidy         # go mod tidy -diff
just check              # check-formatted + check-tidy + lint + test
just check-race         # same, with test-race
just ci                 # quality + coverage + examples/WASM + security
just ci-race            # same, plus the short race suite

# Dependencies and misc
just tidy / just verify / just init / just clean
just docs               # godoc -http=:6060
just new-benchmark Name # append a benchmark-function skeleton to functions.go
just audit              # nancy over the production dependency tree
just vuln               # govulncheck, reports by reachability
just security           # both of the above

# Release
just release-check 0.2.0
just release 0.2.0
```

### Direct Go Commands

```bash
# Build
go build -v ./...

# Test with race detection
go test -v -race ./...

# Run a single test
go test -v -run TestNeighbourScan

# Skip the long statistical suites
go test -short ./...

# Benchmark one function
go test -bench=BenchmarkOptimizeSphere_DA -benchmem

# Coverage
go test -coverprofile=coverage.out ./... && go tool cover -html=coverage.out
```

### Module Management

```bash
go mod tidy
go mod verify
go mod tidy -diff        # what `just check-tidy` runs
go get -u ./...
```

## Architecture & Core Concepts

### Swarm Structure

A single population of `Dragonfly` records, each carrying a position `X`, a step vector `ΔX`
(the DA analogue of velocity), and a cost. Two swarm-wide references drive the search:
the **food source** `X⁺` (best position found so far) and the **enemy** `X⁻` (worst position
found so far). There is no per-individual personal best — DA has no memory beyond the
swarm-level food and enemy.

### The Five Swarming Primitives (`swarm.go`)

For dragonfly `i` with `N` neighbours inside radius `r`:

```
Separation  S_i = -Σ_j (X_i - X_j)          repel from crowding
Alignment   A_i = (Σ_j V_j) / N             match neighbour velocity
Cohesion    C_i = (Σ_j X_j) / N - X_i       move toward the local centroid
Food        F_i = X⁺ - X_i                  X⁺ = best position found so far
Enemy       E_i = X⁻ + X_i                  X⁻ = worst position found so far
```

Two details are easy to get wrong and must both carry dedicated unit tests:

- The **enemy term is a sum, not a difference**. `X⁻ + X_i` is what the paper and the
  reference code use. Writing `X⁻ - X_i` is the single most common porting bug.
- The **neighbourhood test is per-dimension, not Euclidean**. The reference code builds a
  component-wise distance vector and requires `all(dist <= r)`; self-neighbouring is excluded
  via `all(dist != 0)`. A Euclidean shortcut changes swarm dynamics and silently degrades
  convergence without failing an end-to-end test.

Separation, alignment and cohesion come from Reynolds' 1987 boids model; food and enemy are
DA's addition.

### Paper-Default Optimization Flow

```
1. Initialize the swarm: X_i uniform in [lb, ub], ΔX_i small random
2. Evaluate all X_i; set food X⁺ = best, enemy X⁻ = worst
3. For t = 1..T:
   a. Compute the schedules for this iteration (weights.go):
      w, mc, s, a, c, f, e, r, ΔX_max
   b. Update food X⁺ and enemy X⁻ from the current swarm
   c. For each dragonfly i:
      i.   Scan the swarm for neighbours: per-dimension, all(dist <= r), excluding self
      ii.  Build S_i, A_i, C_i, F_i, E_i from those neighbours
      iii. Paper-default step update:

           if neighbours > 0:
               use the full five-factor step; independently zero food/enemy when out of range
           else:
               use the Lévy walk

           MATLAB compatibility mode instead uses the reference food-distance branch:

           if any(dist2Food > r):            # food out of range -> local swarming only
               if neighbours > 1:
                   ΔX_i = w·ΔX_i + rand·A_i + rand·C_i + rand·S_i   # per-dimension rand
                   clamp ΔX to ±ΔX_max                              # no f, no e term
                   X_i += ΔX_i
               else:
                   X_i += Levy(d) ⊙ X_i                             # Levy random walk
                   ΔX_i = 0
           else:                             # food in range -> full five-factor step
               ΔX_i = (s·S_i + a·A_i + c·C_i + f·F_i + e·E_i) + w·ΔX_i
               clamp ΔX to ±ΔX_max
               X_i += ΔX_i

      iv.  Apply boundary handling to X_i (and ΔX_i, see below)
   d. Evaluate the new positions; update food/enemy; record convergence
   e. Check termination (max iterations, target cost, stagnation, ctx cancellation)
4. Return Result{GlobalBest, Convergence, FuncEvalCount, Seed, TerminationReason}
```

Paper mode evaluates initialization and every moved swarm, so a full run makes
`NPop × (MaxIterations + 1)` objective calls. MATLAB mode instead follows all three reference
loops: compute one-based schedules, evaluate the current population, update incumbents/archive
and choose MODA food/enemy, build the movement, then move. Its final moved swarm is deliberately
unevaluated, so a full run makes `NPop × MaxIterations` calls. Results and archives contain
evaluated candidates only.

For continuous DA/MODA the MATLAB movement order is primitives → pre-move wrap and step reset →
move → non-finite sanitization → post-move hard clamp. `FidelityMATLAB` overrides
`BoundaryMethod`. DA also keeps the reference's strict-interior movement enemy separate from
the actual worst evaluated candidate returned as `Result.Worst`; population snapshots expose
the movement enemy with an evaluated pre-move swarm copy.

`Config.FidelityMode` makes the disagreement explicit: paper behavior is the default;
`FidelityMATLAB` reproduces the reference separation sign, one-neighbor fallback and
food-distance branch. Never combine pieces of both into an unnamed hybrid.

### Boundary Handling

DA's default is **wrap-with-step-reset**, not the clamp Mayfly uses:

```
if x_j > ub_j { x_j = lb_j ; Δx_j = rand() }
if x_j < lb_j { x_j = ub_j ; Δx_j = rand() }
```

The step component is _reset to a fresh random draw_, not merely clamped — dropping that
half of the rule changes the exploration behaviour.

`Config.BoundaryMethod` selects the policy:

| Value       | Behaviour                                                          |
| ----------- | ------------------------------------------------------------------ |
| `"wrap"`    | Default. Paper behaviour: teleport to the opposite bound, reset Δx |
| `"clamp"`   | Mayfly's `maxVec`/`minVec` idiom: pin to the violated bound        |
| `"reflect"` | Mirror the overshoot back into the feasible interval               |

Wrapping is genuinely part of DA's exploration, but it interacts badly with some constrained
problems, which is why the alternatives exist as a named, documented paper-mode choice. In
continuous MATLAB mode the fixed reference boundary sequence above applies instead.

### Adaptive Weight Schedules (`weights.go`)

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

Keeping every schedule in one file lets `weights_test.go` assert the properties directly
rather than inferring them from optimization outcomes: `w` decreases monotonically, `mc`
reaches zero at the halfway point, `e` is exactly zero past `3T/4`, `r` grows with `t`.

### Lévy Flight (`levy.go`)

Mantegna's algorithm with β = 1.5:

```
σ = ( Γ(1+β)·sin(πβ/2) / ( Γ((1+β)/2)·β·2^((β-1)/2) ) )^(1/β)
Levy(x) = 0.01 · r₁·σ / |r₂|^(1/β)          r₁, r₂ ~ N(0,1)
```

The official DA/BDA `Levy.m` files verify β `1.5`, the σ formula and scale `0.01`.

### BDA — Binary Variant (`binary.go`)

BDA treats every `i != j` as a neighbor and applies one unconditional five-factor step. The
position update then depends on the transfer family:

```
V-shaped (paper default):  T(Δx) = | Δx / sqrt(Δx² + 1) |
x_j <- ¬x_j  if rand < T(Δx_j)   else   x_j
S-shaped: x_j <- 1 if rand < T(Δx_j), otherwise 0
```

A `TransferFunction` named-string type plus a registry ships the standard families:

| Name | Form                          |
| ---- | ----------------------------- |
| `v1` | \|erf(√π/2 · Δx)\|            |
| `v2` | \|tanh(Δx)\|                  |
| `v3` | \|Δx / √(Δx²+1)\| _(default)_ |
| `v4` | \|(2/π)·arctan((π/2)·Δx)\|    |
| `s1` | 1/(1+e^(-2Δx))                |
| `s2` | 1/(1+e^(-Δx))                 |
| `s3` | 1/(1+e^(-Δx/2))               |
| `s4` | 1/(1+e^(-Δx/3))               |

The objective signature stays `func([]float64) float64` with 0/1-valued input, so the
benchmark, comparison and constraint machinery is reused unchanged.

### MODA — Multi-objective Variant (`multiobjective.go`)

Ports Mayfly's `MultiObjectiveFunction`, `ParetoSolution`, `ParetoArchive`,
`NewParetoArchive`, `Add` and `UpdateFromPopulation`, then adds the **hypercube grid** MODA
needs on top. Three named archive policies are available:

- `ArchivePolicyPaperSegments` (default): food `1/N`, enemy `N`
- `ArchivePolicyMATLABDensity`: objective-span/20 density ranking from `MODA.m`
- `ArchivePolicyMOPSOGrid`: the v0.1 exponent-weighted grid extension

MATLAB-compatible MODA uses the `MODA.m` schedule rather than the shared paper schedule:
inertia decreases `0.9 → 0.2`, automatic S/A/C/E weights follow `mc` directly, and only the
automatic food term draws `2·rand`. Explicit pinned weights still override those defaults.

`ArchiveSize = 100` is verified against `MODA.m`. The MOPSO extension retains
`β = 4, γ = 2, δ = 2, NGrid = 10`, but those are not MODA reference constants.

`OptimizeMultiObjective` stays a separate entry point rather than overloading `Result`;
`MultiObjectiveResult` carries the final archive, with `ExportParetoCSV` / `ExportParetoJSON`
mirroring `monitoring.go`.

MODA honours the shared `Config` block, with two deliberate exceptions. Constraints go through
`constrainedDominates`, Deb's rules lifted from the total order in `constraints.go` to the
partial order an archive needs — `BetterConstrainedCandidate` must never be used here, it
always names a winner and would collapse the front to one point. Early stopping counts an
iteration as an improvement when the archive accepted a candidate. `EnableParallel` fans out
the objective calls only. The exceptions are `Convergence.TargetCost` and
`ConstraintHandlingPenalty`: neither has a multi-objective reading and both are rejected by
`validateMultiObjectiveConfig` rather than silently ignored.

### Configuration System

One **flat** `Config` struct — no nested option groups — with snake_case JSON tags
(`tagliatelle` enforces `json = 'snake'`). Field order is chosen for `govet`'s
`fieldalignment`: pointers/interfaces → strings → float64 → int → bool. That ordering is a
lint gate, not a style preference; adding a field in the "logical" place will fail `just lint`.

Layer presets with factory functions rather than hand-filling structs:

```go
config := dragonfly.NewDefaultConfig()   // standard continuous DA
config := dragonfly.NewBinaryConfig()    // BDA with the v3 transfer function
```

**Required fields** (must be set before calling `Optimize()`):

- `ObjectiveFunc` — function to minimize
- `ProblemSize` — number of dimensions
- `LowerBound` / `UpperBound` — search-space bounds

**The `WeightAuto = -1` sentinel**: every weight-schedule field (`W`, `S`, `A`, `C`, `F`, `E`,
`Radius`, `StepMax`) defaults to `WeightAuto`, which means "use the schedule from §1.4".
Setting a field to any other value pins it to a constant for the whole run. This mirrors
Mayfly's `NCAuto` / `AquilaWeightAuto` convention. Note the consequence: `0` is a _legitimate
pinned value_ (e.g. disabling the enemy term entirely), so code must test against
`WeightAuto`, never against zero.

`config_loader.go` provides JSON load/save, `ValidateConfig`, named presets and
`AutoTuneConfig`.

### File Layout

One concern per file, as in PLAN.md §2. A new file belongs in this shape or the shape needs a
reason to change.

```
Dragonfly/
├── go.mod  LICENSE  .gitignore
├── README.md  CHANGELOG.md  PLAN.md  CLAUDE.md  AGENTS.md
├── justfile  .golangci.toml  treefmt.toml
├── .github/workflows/{test.yml,release.yml,security.yml}
│
│  ── core ──
├── dragonfly.go        package doc, Optimize, OptimizeContext, the main loop
├── types.go            Config, Dragonfly, Best, Result, ConvergenceConfig, sentinels
├── config.go           NewDefaultConfig and the preset factories
├── config_loader.go    JSON load/save, ValidateConfig, presets, AutoTuneConfig
├── swarm.go            neighbourhood scan, S/A/C/F/E vectors, radius schedule
├── weights.go          adaptive w, s, a, c, f, e schedules
├── levy.go             Mantegna Levy flight
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
├── improved.go         shared improved-variant evaluation and lifecycle helpers
├── mhda.go             MHDA: personal/global memory and PSO exploitation
├── chaotic.go          CDA: continuous DA driven by one of ten chaotic maps
├── qgda.go             QGDA: Gaussian mutation and quantum rotation
├── variants.go         AlgorithmVariant interface, registry implementations, VariantBuilder
├── selector.go         AlgorithmSelector, ClassifyProblem, RecommendForBenchmark
├── comparison.go       ComparisonRunner, Wilcoxon, Friedman, CSV/JSON export
│
│  ── parallelism ──
├── parallel.go          evaluationPool, batchBest, mergeBest, effectiveMaxWorkers
├── parallel_compute.go  parallelFor
├── parallel_phases.go   prepareSwarmStep, prepareLevyStep (own the RNG, evaluate nothing)
├── parallel_variants.go evaluateParallelStep, evaluateParallelBinary
│
│  ── tests, siblings at root, package dragonfly ──
├── *_test.go            one per source file, plus benchmark/regression/integration
├── example_test.go      package dragonfly_test — the only black-box file
│
├── features/    *.feature, run by godog
├── docs/        README.md, algorithms/, api/, benchmarks.md, research.md
├── examples/    each subdirectory its own module, with a replace directive
└── scripts/
```

## Testing Strategy

### Conventions

- **White-box.** Test files live in `package dragonfly` and may exercise unexported helpers.
  `testpackage` is disabled in `.golangci.toml` for exactly this reason.
- **`example_test.go` is the sole black-box file** (`package dragonfly_test`). It holds
  `ExampleOptimize`, `ExampleOptimizeContext`, `ExampleNewBuilder` — compiled, runnable
  documentation, not a coverage vehicle.
- **No `t.Parallel()`.** Tests are deterministic and seed-driven; running them in parallel
  buys nothing and risks interleaved RNG or shared-config mutation. `paralleltest` is
  disabled in the lint config to make this explicit.
- **`testing.Short()` gating.** Long statistical suites (multi-seed convergence runs,
  regression baselines, comparison harnesses) must call `t.Skip` under `testing.Short()`.
  `just test-quick` and `just test-race` pass `-short`; `just test-full` does not. A test
  that cannot finish inside `just test-race`'s 5-minute budget must be gated.
- **godog features** live in `features/` and are wired through `TestFeatures` in
  `integration_test.go`. `godog` is the only direct dependency and stays test-only; run them
  with `just test-integration`.
- **Names** are `TestXxx`, `BenchmarkXxx`, `ExampleXxx`. Benchmarks live in `*_test.go` as
  `BenchmarkXxx(b *testing.B)` beside the tests, not in a directory of their own.
- **Keep `just check` green**, and read `coverage.html` locally rather than trusting the
  percentage alone.

### What each layer proves

- **Unit tests, hand-computed.** `swarm_test.go` builds a 3-dragonfly, 2-dimensional swarm and
  compares S, A, C, F, E and one full ΔX step against values worked out by hand from `DA.m`.
  That single test is what catches a Euclidean-vs-per-dimension neighbour mistake or a sign
  error in the enemy term — neither shows up as an obvious failure end-to-end.
- **Property tests.** `weights_test.go` asserts monotonicity and the exact zero crossings of
  the schedules directly, rather than inferring them from optimization outcomes.
- **Determinism tests.** `parallel_test.go` carries
  `TestParallelIsDeterministicForSeedAcrossSchedules` (a seeded parallel run must be
  bit-identical to the sequential one) and `TestParallelCancellationDoesNotCommitPartialBatch`.
- **Regression tests.** `RegressionBaseline` entries encode **tolerated degradation factors,
  not golden values.** A stochastic optimizer's output is not a golden file. The question a
  regression test can usefully answer is "did this change make the algorithm meaningfully
  worse", which is a statistical question with a tolerance attached. Never replace a baseline
  with an observed number to make a test pass.
- **Invariant tests.** MODA's non-domination invariant is asserted on _every_ archive mutation,
  not only at the end of a run — that is where it silently breaks.

### Coverage

80%+ statement coverage is the Phase 10 release gate.

## Development Guidelines

### Coding Style & Naming

- Go defaults: tabs, standard import grouping. Formatting is gofumpt followed by gci with the
  local prefix `github.com/CWBudde/dragonfly`; `just fmt` runs both through treefmt.
- Exported identifiers are `CamelCase`, internal ones `lowerCamel`, packages short and
  lowercase.
- Short mathematical identifiers that mirror the paper (`w`, `s`, `a`, `c`, `f`, `e`, `r`,
  `dX`) are preferred inside the algorithm files and kept out of public API names. `mnd` and
  `varnamelen` are disabled for exactly this reason: numeric literals _are_ the algorithm.
- Prefer small, cohesive files and pure functions — one concern per file, as in the layout
  above.
- Every stochastic helper takes `rng *rand.Rand` as its **last** parameter.
- Lint per `.golangci.toml`. Mayfly's per-file complexity exemptions are deliberately not
  inherited; if a function genuinely earns one, add it _and_ record the debt in PLAN.md.

### Adding a New Algorithm Variant

1. **Add the variant's parameters to the flat `Config` struct** in `types.go`, with a
   `Use<Variant>` boolean flag or a variant-selecting field. Respect `fieldalignment`
   ordering (pointers/interfaces → strings → float64 → int → bool) and give every field a
   snake_case JSON tag. Numeric schedule fields default to `WeightAuto`.
2. **Create a factory function** `New<Variant>Config()` in `config.go` that starts from
   `NewDefaultConfig()` and layers the preset on top. Add matching validation to
   `ValidateConfig` in `config_loader.go`.
3. **Implement the variant logic** in its own file (`binary.go`, `multiobjective.go`, …), not
   inline in `OptimizeContext`. Keep RNG threading explicit: every stochastic helper takes
   `rng *rand.Rand` as its **last** parameter. If the variant needs parallel evaluation, add
   a `prepare*` phase in `parallel_phases.go` and an `evaluateParallel*` in
   `parallel_variants.go`.
4. **Register it in `variants.go`**: implement `AlgorithmVariant`, add it to the registry, and
   keep `GetAllVariants()` in a stable canonical order. Teach `selector.go` when to recommend
   it.
5. **Test, example, document**: a hand-computed unit test for the new operator, a
   `_test.go` for the variant, an `examples/<variant>/` subdirectory with its own module and
   `replace` directive, a `docs/algorithms/<variant>.md` on the fixed skeleton, and a
   `RegressionBaseline` entry.

Do not add a per-file complexity exemption to `.golangci.toml` to make a new variant fit.
Mayfly carries those to grandfather a 1200-line `OptimizeContext`; this repo deliberately
starts without them. If a function genuinely earns an exemption, add it _and_ record the debt
in PLAN.md.

### Extending Benchmark Functions

Add to `functions.go` following Mayfly's doc-comment convention: one sentence naming the
function and characterising its landscape, then the global minimum, then non-obvious bounds.

```go
// Rastrigin is the Rastrigin benchmark function: highly multimodal with a regular lattice of local minima.
// Global minimum is at f(0, ..., 0) = 0.
func Rastrigin(x []float64) float64 {
	// ...
}

// Schwefel is the Schwefel benchmark function: deceptive, with the global minimum far from the next best local minima.
// Typical bounds: [-500, 500].
func Schwefel(x []float64) float64 {
	// ...
}
```

Rules:

- All functions are **minimization** problems.
- Signature is always `func(x []float64) float64` — including for the binary variant, where
  input is 0/1-valued.
- Add the "Typical bounds" line only when the function is not usable on generic symmetric
  bounds.
- `just new-benchmark <Name>` appends a skeleton in this shape.
- Add a matching case to `functions_test.go` asserting the value at the known optimum.
- If the name is used in a switch or table, consider adding it to
  `goconst.ignore-string-values` in `.golangci.toml` (Sphere, Rastrigin, Rosenbrock, Ackley,
  Griewank and Schwefel are already listed).

`mnd` and `varnamelen` are disabled: numeric literals _are_ the algorithm, and short math
identifiers (`x`, `w`, `r`, `s`, `a`, `c`, `f`, `e`) mirror the papers. Use them.

### Maximization Problems

The algorithm minimizes. For maximization, negate the objective:

```go
func maximizeProfit(x []float64) float64 {
	return -calculateProfit(x)
}
```

## Reproducible Results

Determinism is a hard requirement, not a nicety.

**RNG injection.** Prefer `Config.Seed` when the seed must be reported:

```go
config := dragonfly.NewDefaultConfig()
seed := int64(42)
config.Seed = &seed
result, _ := dragonfly.Optimize(config)
```

When both fields are nil, `OptimizeContext` creates a generator and reports its seed with
`Result.SeedKnown = true`. A directly supplied `Config.Rand` is honored, but its original seed
is not introspectable, so `SeedKnown` is false. Supplying both `Seed` and `Rand` is rejected.

**Explicit threading.** Every stochastic helper takes `rng *rand.Rand` as its last parameter.
No package-level `rand.Float64()`, no hidden `math/rand` global. (`gosec`'s G404 is excluded
precisely because `math/rand` is the point here — but that excludes the _weak-randomness_
warning, not the discipline.)

**The parallelism rule.** _All_ RNG draws happen on the calling goroutine during the
`prepare*` phase. Worker goroutines only evaluate the objective function — they never draw a
random number, and they never mutate shared swarm state. A seeded run must produce
**bit-identical** results with `EnableParallel` on or off, and
`TestParallelIsDeterministicForSeedAcrossSchedules` enforces exactly that.

DA has a wrinkle Mayfly does not: the radius, the food source, the enemy and every neighbour
set depend on the whole swarm, so the prepare phase is **two sequential passes** —

1. compute `r`, the weight schedules, the food source and the enemy;
2. per dragonfly, scan for neighbours and build ΔX.

Both passes touch the RNG and stay on the calling goroutine. Only the objective evaluation of
the resulting positions fans out. The neighbour scan itself is `O(n²·d)` and is the real hot
spot for large swarms; it can be parallelised safely _precisely because_ it draws no random
numbers, and `BenchmarkNeighbourScan` exists to prove that is worth doing before the
complexity is added.

Observers (`WithProgressObserver`, `WithPopulationObserver`, `WithArchiveObserver`) receive deep copies and run
synchronously on the caller's goroutine — they must not become an RNG or ordering back door.

## Common Pitfalls

1. **The enemy term is `X⁻ + X_i`, a SUM, not a difference.** Every other primitive is a
   difference, so `X⁻ - X_i` reads "correct" and compiles fine. It is wrong. The paper and the
   reference code both use the sum. Cover it with a hand-computed unit test.

2. **The neighbourhood test is per-dimension, not Euclidean.** Build a component-wise distance
   vector and require `all(dist <= r)`, excluding self via `all(dist != 0)`. A Euclidean
   `norm(X_i - X_j) <= r` shortcut is faster, obvious, and changes the swarm dynamics —
   convergence degrades silently and no end-to-end test flags it.

3. **Never draw random numbers inside worker goroutines.** All RNG draws belong on the calling
   goroutine in the `prepare*` phase. Workers evaluate the objective and nothing else. A single
   `rng.Float64()` inside a worker destroys bit-identical reproducibility and turns a
   deterministic test suite flaky in a way that is very hard to bisect.

4. **DA wraps at the boundary by default; it does not clamp.** A dragonfly that leaves the box
   teleports to the opposite bound _and_ has its step component reset to a fresh random draw.
   Users arriving from PSO, GA or Mayfly expect clamping and will read wrapping as a bug —
   and code ported from Mayfly will silently apply `maxVec`/`minVec` instead. Paper-mode fixes
   route through `Config.BoundaryMethod`; MATLAB mode deliberately bypasses it for the reference
   pre-wrap/reset and post-clamp sequence. The Δx reset is half the wrap rule.

5. **Do not present the legacy MOPSO exponents as MODA constants.** The audit verified Lévy
   β/σ/scale, BDA's `±6` clamp and MODA's archive size 100 against the official packages.
   `β = 4, γ = 2, δ = 2, NGrid = 10` belong only to `ArchivePolicyMOPSOGrid`; paper and MATLAB
   modes use their named policies instead.

6. **`WeightAuto` is `-1`, and `0` is a legitimate pinned value.** Test schedule fields against
   the sentinel, never against zero — `E = 0` means "pin the enemy weight to zero", not "use
   the default schedule".

7. **`fieldalignment` governs `Config` field order.** Adding a field where it reads best
   (grouped with its siblings) will fail `just lint`. Order is pointers/interfaces → strings →
   float64 → int → bool.

8. **Forgetting the required `Config` fields.** `ObjectiveFunc`, `ProblemSize`, `LowerBound`
   and `UpperBound` have no usable defaults. Always start from a factory function.

9. **Collapsing the two-branch step update.** The out-of-range branch uses only A, C, S with
   per-dimension random coefficients — no `f`, no `e` — and falls through to a Lévy walk when
   a dragonfly has ≤ 1 neighbour. A single unconditional five-factor step still converges on
   Sphere, which is what makes this bug survive.

## Commit & PR Guidelines

- Commits follow conventional style: `feat: ...`, `fix: ...`, `perf: ...`, `docs: ...`,
  `chore: ...`.
- A PR states the goal, the key changes and the before/after impact, and links its issues.
- Any behaviour change carries unit and, where it fits, feature tests, and ticks the PLAN.md
  checkboxes it completes. A change that closes a `Deferred` entry replaces that entry with
  what was measured, rather than deleting it.
- `just check` must be green before review; `just check-race` before anything touching the
  parallel paths.

## Security & Maintenance

- Dependencies: `just tidy` then `just verify`.
- `just security` runs both halves of the scan — `just audit` (nancy, production tree) and
  `just vuln` (govulncheck, which also covers the test-only tree and reports by reachability).
  The library is stdlib-only, so nancy reports zero audited dependencies; that is the intended
  state, and the recipe exists to catch the first real dependency that is ever added. Scanner
  versions are pinned and checked; ordinary CI, release validation and the weekly/manual
  security workflow run them under Go 1.26.
- Release: `just ci`, then `just release-check <semver>`, then `just release <semver>`.
  `docs/releasing.md` has the full procedure.

## Research Citations

- Mirjalili, S. (2016). Dragonfly algorithm: a new meta-heuristic optimization technique for
  solving single-objective, discrete, and multi-objective problems. _Neural Computing and
  Applications_, 27(4), 1053–1073. doi:10.1007/s00521-015-1920-1 — the primary source for DA,
  BDA and MODA.
- Reynolds, C. W. (1987). Flocks, herds and schools: A distributed behavioral model. _ACM
  SIGGRAPH Computer Graphics_, 21(4), 25–34. — the origin of separation, alignment and cohesion.
- Mantegna, R. N. (1994). Fast, accurate algorithm for numerical simulation of Lévy stable
  stochastic processes. _Physical Review E_, 49(5), 4677–4683. — `levy.go`.
- Deb, K. (2000). An efficient constraint handling method for genetic algorithms. _Computer
  Methods in Applied Mechanics and Engineering_, 186(2–4), 311–338. — the feasibility rules
  used in `constraints.go`.
- Coello Coello, C. A., Pulido, G. T., & Lechuga, M. S. (2004). Handling multiple objectives
  with particle swarm optimization. _IEEE Transactions on Evolutionary Computation_, 8(3),
  256–279. — the hypercube archive MODA borrows.

Reference implementations: the author's `DA.m`, `BDA.m` and `MODA.m`. Where the reference code
and a "cleaner" formulation disagree, **implement the paper** and expose the alternative behind
a config field.

## Key Files Reference

Source: `dragonfly.go` (main loop and entry points); `swarm.go`, `weights.go`, `levy.go` (the
parts a reader checks against the paper); `types.go`, `config.go`, `config_loader.go`
(configuration); `binary.go`, `multiobjective.go`, `mhda.go`, `chaotic.go`, `qgda.go`,
`variants.go` (the variants and framework layer); `parallel*.go` (deterministic parallelism).

Everything else:

- `PLAN.md` — the roadmap and the algorithm specification; source of truth for progress, and
  the place where known gaps are recorded rather than quietly carried
- `README.md` — project overview and quick start
- `CHANGELOG.md` — Keep a Changelog 1.1.0 + SemVer
- `docs/` — algorithm guides, API reference, benchmarks, performance, citations
- `justfile` — every build, test, lint, profile and release recipe
- `.golangci.toml` — the lint contract (golangci-lint v2, `default = 'all'` minus the
  documented exclusions)
- `treefmt.toml` — gofumpt → gci for Go, prettier for md/json/yaml, taplo for toml, shfmt for shell
- `.github/workflows/{test.yml,release.yml,security.yml}` — CI on Go 1.23 + 1.26; release
  asserts the module path and runs race, coverage, examples/WASM and security gates; the
  separate security workflow also runs weekly
- Tool versions are centralized in `justfile` and checked, not merely discovered on `PATH`:
  treefmt 2.5.0, golangci-lint 2.13.1, gofumpt 0.11.0, gci 0.14.0, shfmt 3.13.1,
  Taplo 0.10.0, Prettier 3.9.6, ShellCheck 0.11.0, Nancy 2.1.0 and govulncheck 1.1.4.
- `AGENTS.md` — this file. `CLAUDE.md` is a pointer to it
