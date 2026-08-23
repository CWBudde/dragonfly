# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Paper-default and MATLAB compatibility modes (`Config.FidelityMode`), plus named MODA
  archive policies for paper segments, MATLAB density ranking and the released MOPSO-grid
  extension.
- Explicit reproducibility metadata through `Config.Seed`, `Result.SeedKnown` and
  `MultiObjectiveResult.SeedKnown`.
- Exact small-sample Wilcoxon tests, tie corrections, Holm-adjusted pairwise p-values and
  failed-run accounting in comparison reports.

- A WebAssembly browser demo in `examples/wasm-demo`, published to GitHub Pages by
  `.github/workflows/wasm-demo-pages.yml` and built by `scripts/build-wasm-demo.sh`. Four
  pages — a Swarm Lab that colours each dragonfly by which branch of the two-branch step
  update it is about to take and draws the neighbourhood as the per-dimension box it is, a
  Pareto page that animates MODA's archive over its hypercube grid, a Binary page pairing
  BDA's bit matrix with the transfer function's own curve, and a Shootout that runs
  `ComparisonRunner` over DA's configurable choices. No optimization logic lives in the
  JavaScript: every number is computed by the library compiled to `js/wasm`. New justfile
  recipes `build-wasm-demo`, `run-wasm-demo` and `check-wasm-demo`.

- `OptimizeMultiObjective` now accepts `...RunOption`, and a new `WithArchiveObserver`
  reports an `ArchiveSnapshot` — deep copies of the archive, its per-objective grid extent
  and `NGrid` — once per completed iteration, at the same point in the loop where
  `WithPopulationObserver` fires for a single-objective run. Before this no caller could
  watch a multi-objective run at all; only the final archive and its size per iteration were
  observable. `WithInitialPopulation` now works on multi-objective runs too, seeding the
  leading slots without shifting the random stream. The signature change is variadic, so
  existing call sites compile unchanged.

- `ParetoArchive.GridBounds()` returns the archive's per-objective extent — the frame every
  solution's `GridIndex` is expressed in. Both slices are copies, because the archive
  rewrites its bounds through the existing backing arrays on every mutation.

- `NeighborhoodRadius` and `WithinRadius` are now exported. They are the radius schedule and
  the neighbour test the optimizer itself uses, so a caller reconstructing a neighbourhood
  from a `PopulationSnapshot` can ask the library rather than reimplement the rule — and a
  Euclidean reading of that per-dimension box test is the most common way to get this
  algorithm subtly wrong.

- Run options with no meaning for a problem class are now rejected rather than silently
  ignored, the rule `validateMultiObjectiveConfig` already applied to `Convergence.TargetCost`
  and `ConstraintHandlingPenalty`. `OptimizeMultiObjective` rejects `WithProgressObserver`,
  `WithPopulationObserver` and `WithLogger` — a Pareto front has no incumbent for any of them
  to report; `OptimizeContext` and `OptimizeBinaryContext` reject `WithArchiveObserver`. A nil
  observer registers nothing and is never rejected.

- MODA now honours the shared `Config` block. `Config.Constraints` reaches the archive
  through `constrainedDominates`, which lifts Deb's feasibility rules from the total order
  in `constraints.go` to the partial order a Pareto archive needs; `ParetoSolution` carries
  the aggregate violation and the CSV and JSON exports gained a column for it. Early
  stopping counts an iteration as an improvement when the archive accepted a candidate, so
  `Convergence.StagnationIterations` and `MinIterations` now shorten a MODA run and
  `MultiObjectiveResult.TerminationReason` can report `stagnation`. `EnableParallel` fans
  the objective calls out through `parallelFor`; every random draw stays on the calling
  goroutine, so a seeded run is bit-identical with parallelism on or off.

### Changed

- Corrected DA neighborhood exclusion, enemy-radius gating, schedule indexing, reflection,
  cancellation and non-finite-objective handling. Paper mode and MATLAB mode now have
  separately tested step transitions where their operators disagree.
- Corrected BDA to use the whole swarm and an unconditional five-factor step; S-shaped
  transfers now assign a sampled bit while V-shaped transfers complement the current bit.
- MODA now defaults to the paper's `1/N` and `N` archive selection, validates objective
  arity and numeric values before mutation, copies callback results, uses stable roulette
  arithmetic and rejects unsupported shared settings.
- Configuration loading rejects unknown fields and trailing JSON. Report exports now use an
  atomic same-directory replacement and serialize unavailable comparison costs as `null`.
- CI pins golangci-lint, tests the minimum and current Go releases, enforces 80% coverage and
  builds every nested example module and the WebAssembly target.

### Fixed

- Prevented synthetic unevaluated incumbents, false seed claims, comparison failures entering
  statistical samples, malformed Pareto archive exports and unusable MODA builder/selector
  recommendations.

- Pareto archive maintenance is substantially cheaper, with the archive contents
  bit-identical. `updateGrid` reuses each member's index array and skips the reassignment
  sweep when the bounds have not moved, `occupiedCells` counting-sorts into reused buffers
  instead of building and sorting a map, and `Add` compacts survivors in place. The
  SchafferN1 MODA benchmark dropped from 75.8 ms and 379,024 allocations per operation to
  19.7 ms and 22,534; ZDT1 from 11.3 ms and 33,245 to 8.0 ms and 16,947. One consequence is
  worth knowing: a `ParetoArchive.Solutions` slice held across a mutation is no longer a
  snapshot of the archive as it was.

### Fixed

- `Levy` no longer panics on an empty position vector, and `Ackley` and `HappyCat` no
  longer return `NaN` for one. The whole single-objective benchmark suite now follows one
  documented convention -- `f([]) == 0` -- asserted for all fifteen functions by
  `TestBenchmarkFunctionsEmptyInput`. The same fix landed in the sibling Mayfly library, so
  the two suites stay numerically comparable.

### Changed

- `validateMultiObjectiveConfig` rejects `Convergence.TargetCost` and
  `ConstraintHandlingPenalty` on a MODA config. Neither has a multi-objective reading -- a
  scalar target says nothing about a front, and penalizing every objective component is not
  a defensible default -- and both were previously accepted and then silently ignored.
- `Config.EnemyCutoffFraction` is documented as inert at its default. The enemy weight
  follows `mc`, which already reaches zero at half the run, so the `0.75` cutoff only ever
  replaces a zero with a zero; only a fraction below `0.5` changes anything, and nothing at
  all when `EnemyWeight` is pinned off `WeightAuto`. Behaviour is unchanged -- the field
  stays because the paper and `DA.m` carry both rules.

## [0.1.0] - 2026-08-23

First release. A dependency-free Go implementation of Mirjalili's Dragonfly Algorithm,
covering all three variants of the 2016 paper. The only direct dependency,
`github.com/cucumber/godog`, is test-only.

### Added

#### Core algorithm

- `Optimize` and `OptimizeContext`: the standard continuous Dragonfly Algorithm. The step
  update reproduces the reference implementation's two branches — local swarming with
  per-dimension random coefficients when the food source is out of the neighbourhood
  radius, the full five-factor step when it is in range, and a Lévy random walk for a
  dragonfly with at most one neighbour.
- `swarm.go`: the five swarming primitives. The neighbourhood test is per-dimension
  (Chebyshev) rather than Euclidean, self-neighbouring is excluded, and the enemy term is
  the paper's sum `X⁻ + X_i` — each covered by a hand-computed unit test rather than
  inferred from convergence behaviour.
- `weights.go`: the adaptive schedules for `w`, `s`, `a`, `c`, `f`, `e`, the neighbourhood
  radius and the step clamp, asserted directly for monotonicity and their zero crossings.
- `levy.go`: Mantegna's algorithm at β = 1.5. The σ constant is verified against the
  published value rather than inherited.
- Boundary handling as a named choice via `Config.BoundaryMethod`: `wrap` (the paper's
  teleport-to-the-opposite-bound with a fresh random step component, and the default),
  `clamp` and `reflect`.
- `functions.go`: 15 benchmark functions, shared with the sibling Mayfly project so the
  two suites stay numerically comparable.

#### Variants

- BDA, the binary variant: eight transfer functions (`v1`–`v4`, `s1`–`s4`) behind a named
  registry, `NewBinaryConfig`, and bit-flip position updates. The objective signature is
  unchanged, so the benchmark, constraint and comparison machinery is reused as is.
- MODA, the multi-objective variant: `OptimizeMultiObjective`, a Pareto archive with the
  hypercube grid the paper borrows from MOPSO — food drawn from the least-populated
  occupied cell, enemy from the most-populated, overflow evicted from the most crowded —
  and `ExportParetoCSV` / `ExportParetoJSON`.
- `variants.go`, `selector.go`, `comparison.go`: the `AlgorithmVariant` interface with a
  stable registry, `VariantBuilder`, problem classification and variant recommendation,
  and a `ComparisonRunner` with paired seeds plus Wilcoxon and Friedman tests.

#### Configuration and lifecycle

- One flat `Config` with snake_case JSON tags. Every weight-schedule field defaults to the
  `WeightAuto` sentinel, so `0` remains a legitimate pinned value.
- Factories `NewDefaultConfig`, `NewHighDimensionalConfig`, `NewFastConvergenceConfig` and
  `NewBinaryConfig`; `config_loader.go` adds JSON load/save, `ValidateConfig`, named
  presets and `AutoTuneConfig`.
- `RunOption` hooks: `WithInitialPopulation`, `WithProgressObserver`,
  `WithPopulationObserver` and `WithLogger`. Observers receive deep copies and run
  synchronously on the caller's goroutine.
- `constraints.go`: Deb's feasibility rules and linear/quadratic penalty methods behind a
  single `constraintEvaluator` that is the only authority on what "better" means.
- `convergence.go` and `monitoring.go`: target-cost and stagnation termination, structured
  `slog` lifecycle events, and convergence export to CSV and JSON.

#### Determinism and parallelism

- Optional parallel objective evaluation with a bounded worker pool. Every RNG draw
  happens on the calling goroutine during the prepare phase and workers only call the
  objective, so a seeded run is bit-identical with `EnableParallel` on or off —
  `TestParallelIsDeterministicForSeedAcrossSchedules` enforces exactly that.
- `Result.Seed` is always populated, so any run is reproducible after the fact.

#### Tests, examples and documentation

- 94% statement coverage: hand-computed unit tests for the operators, property tests for
  the schedules, determinism tests, 70 godog scenarios across six feature files, and
  regression baselines that encode tolerated degradation factors rather than golden
  values, because a stochastic optimizer has none.
- Six runnable examples, each its own module with a `replace` directive.
- A `docs/` hub with per-algorithm guides, an API quick reference, benchmark results and
  the research citations, including an explicit table separating verified constants from
  this implementation's judgement calls.

### Known limitations

Recorded in `PLAN.md` rather than hidden:

- MODA's hypercube parameters (`β = 4`, `γ = 2`, `δ = 2`, `NGrid = 10`) and BDA's step
  clamp (`MaxStepRatio = 6.0`) are this implementation's choices, not values read off the
  author's MATLAB, which was not available. Do not cite them as settled paper constants.
- MODA recovers the ZDT fronts at low dimensionality but not at the suite's original 30
  dimensions, and honours neither `EnableParallel` nor `Config.Constraints` nor early
  stopping.
- `EnemyCutoffFraction` is inert at its default: `e = mc` already reaches zero at `T/2`,
  so the `0.75·T` cutoff can never fire. This matches the paper, but the field advertises
  a control it does not have at that setting.
- `Levy([])` panics and empty-input handling is inconsistent across the benchmark suite.
  Neither affects a real optimization, where every position has at least one component.

[Unreleased]: https://github.com/CWBudde/dragonfly/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/CWBudde/dragonfly/releases/tag/v0.1.0
