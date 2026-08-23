# Dragonfly Library Documentation

Everything written down about this implementation of the Dragonfly Algorithm lives here. The
[repository README](../README.md) is the tour; this folder is the reference.

## Quick Links

- **First run?** [API Quick Reference → Minimal optimization](api/quick-reference.md#minimal-optimization)
- **Tuning a run?** [Configuration Guide](api/configuration.md)
- **Which variant?** [Standard DA](algorithms/standard-da.md) · [BDA](algorithms/bda.md) · [MODA](algorithms/moda.md)
- **Picking a test function?** [Benchmark Functions](benchmarks.md)
- **Is it fast enough?** [Performance and Profiling](performance.md)
- **Where does this come from?** [Research References](research.md)

## Documentation Structure

### API Reference

- **[API Quick Reference](api/quick-reference.md)** — a task-oriented map of the package
  - Entry points and configuration factories
  - Run options and `Result` fields
  - Builder, variant registry and selector
  - Comparison framework, constraints, exports

- **[Configuration Guide](api/configuration.md)** — every `Config` field explained
  - Required fields and the factory functions
  - The `WeightAuto` sentinel and how to pin a weight
  - Boundary methods, radius and step schedules, Lévy parameters
  - Constraints, convergence, parallel evaluation, RNG
  - Validation, JSON persistence, presets, auto-tuning

- **[Run Lifecycle](api/run-lifecycle.md)** — controlling and watching a run
  - Context cancellation and what a cancelled run returns
  - Progress and population observers
  - Structured logging through `log/slog`
  - Convergence and Pareto-front export (CSV, JSON)
  - Seeding the initial population

- **[Comparison Framework](api/comparison-framework.md)** — head-to-head statistics
  - `ComparisonRunner` configuration and paired seeds
  - Wilcoxon signed-rank and Friedman tests
  - Reading and exporting a `ComparisonResult`

### Algorithm Documentation

Each variant follows the same skeleton: research reference, overview, key innovations, usage,
parameters, benefits, performance, when to use, tuning guide, comparison, related docs.

1. **[Standard DA](algorithms/standard-da.md)** — the paper's continuous algorithm
   - Best for: continuous single-objective problems
   - The baseline every other variant is measured against

2. **[BDA](algorithms/bda.md)** — the binary Dragonfly Algorithm
   - Best for: binary and discrete search spaces, feature selection, subset selection
   - DA's step vector read through a V- or S-shaped transfer function

3. **[MODA](algorithms/moda.md)** — the multi-objective Dragonfly Algorithm
   - Best for: Pareto front approximation, engineering design trade-offs
   - DA's swarm mechanics over a hypercube-gridded Pareto archive

### Reference Documentation

- **[Benchmark Functions](benchmarks.md)** — 15 single-objective and 4 multi-objective test
  functions, with bounds, optima, characteristics and the results this implementation actually
  reaches
- **[Performance and Profiling](performance.md)** — measured timings and allocation counts,
  dimension and population scaling, when parallel evaluation pays for itself, profiling recipes
- **[Research References](research.md)** — the primary and supporting papers, BibTeX entries,
  and an honest inventory of which constants are verified against the reference code
- **[Releasing](releasing.md)** — version policy, release checklist, validation workflow

## Navigation Guide

### By experience

**New to the library**

1. [README → Quick Start](../README.md#quick-start)
2. [API Quick Reference](api/quick-reference.md)
3. [Standard DA](algorithms/standard-da.md)

**Tuning a real problem**

1. [Configuration Guide](api/configuration.md)
2. [Run Lifecycle](api/run-lifecycle.md)
3. The tuning-guide section of your variant's page

**Evaluating or extending the library**

1. [Comparison Framework](api/comparison-framework.md)
2. [Performance and Profiling](performance.md)
3. [Research References](research.md) and [../PLAN.md](../PLAN.md)

### By task

| I want to…                       | Read                                                                                                   |
| -------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Minimize a function              | [Quick Reference](api/quick-reference.md#minimal-optimization)                                         |
| Maximize a function              | [Configuration Guide → Maximization](api/configuration.md#maximization-problems)                       |
| Add constraints                  | [Configuration Guide → Constraints](api/configuration.md#constraint-handling)                          |
| Stop early                       | [Configuration Guide → Convergence](api/configuration.md#convergence-detection)                        |
| Watch a run                      | [Run Lifecycle](api/run-lifecycle.md)                                                                  |
| Reproduce a run exactly          | [Configuration Guide → Random number generation](api/configuration.md#random-number-generation)        |
| Search a binary space            | [BDA](algorithms/bda.md)                                                                               |
| Trade off several objectives     | [MODA](algorithms/moda.md)                                                                             |
| Choose a variant automatically   | [Quick Reference → Selection](api/quick-reference.md#variant-selection)                                |
| Compare variants with statistics | [Comparison Framework](api/comparison-framework.md)                                                    |
| Speed a run up                   | [Performance](performance.md) and [Configuration → Parallel](api/configuration.md#parallel-evaluation) |
| Understand a surprising result   | [Pitfalls](#pitfalls-worth-reading-before-you-file-a-bug) below                                        |

## Pitfalls worth reading before you file a bug

These four surprise nearly everyone. Each is intentional, each is covered by a dedicated test,
and each is explained at length on the page linked beside it.

- **The enemy term is a SUM.** `E_i = X⁻ + X_i`, not `X⁻ - X_i`. Every other primitive is a
  difference, which is exactly why the difference form reads "correct" and compiles fine. The
  paper and the reference `DA.m` both add. [Standard DA](algorithms/standard-da.md#the-five-swarming-primitives)
- **The neighbourhood test is per-dimension, not Euclidean.** A neighbour is one whose distance
  is within the radius in _every_ component — a box test, not a ball test. The Euclidean
  shortcut is faster and obvious and silently shrinks every neighbourhood.
  [Standard DA](algorithms/standard-da.md#the-neighbourhood-scan)
- **DA wraps at the boundary; it does not clamp.** A dragonfly that leaves the box teleports to
  the opposite bound _and_ has that step component redrawn from `[0,1)`. The step reset is half
  the rule. Users arriving from PSO, GA or Mayfly expect clamping and read wrapping as a bug.
  [Configuration Guide → Boundary handling](api/configuration.md#boundary-handling)
- **`WeightAuto` is `-1`, and `0` is a legitimate pinned value.** `EnemyWeight = 0` means "pin
  the enemy weight to zero for the whole run", not "use the default schedule". Test against the
  sentinel, never against zero.
  [Configuration Guide → The WeightAuto sentinel](api/configuration.md#the-weightauto-sentinel)

## Contributing to documentation

1. Keep examples short, complete and _executed_ — no sample ships that has not been run
2. Prefer a source reference (`swarm.go`, `enemyVector`) over restating an implementation
3. Update this file when adding a document
4. Cross-reference rather than duplicating
5. Run `just fmt` before committing; prettier formats Markdown here

## External resources

- **Main README**: [../README.md](../README.md)
- **Roadmap and specification**: [../PLAN.md](../PLAN.md)
- **Development guide**: [../AGENTS.md](../AGENTS.md)
- **Examples**: [../examples/](../examples/)
- **Feature files**: [../features/](../features/)
- **Source**: the repository root — the package is flat
