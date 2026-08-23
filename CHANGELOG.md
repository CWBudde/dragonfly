# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/CWBudde/Dragonfly/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/CWBudde/Dragonfly/releases/tag/v0.1.0
