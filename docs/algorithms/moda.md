# MODA — Multi-Objective Dragonfly Algorithm

## Research Reference

**Mirjalili, S. (2016). Dragonfly algorithm: a new meta-heuristic optimization technique for
solving single-objective, discrete, and multi-objective problems. _Neural Computing and
Applications_, 27(4), 1053–1073.** — §5 introduces MODA.

<https://doi.org/10.1007/s00521-015-1920-1>

The hypercube-gridded archive MODA uses to keep its front spread out comes from the MOPSO
lineage:

**Coello Coello, C. A., Pulido, G. T., & Lechuga, M. S. (2004). Handling multiple objectives
with particle swarm optimization. _IEEE Transactions on Evolutionary Computation_, 8(3),
256–279.**

Reference implementation: the author's `MODA.m`. It is **not available to this repository**,
which is why the archive's four numeric parameters are flagged as unverified below.

## Overview

MODA runs DA's swarm mechanics unchanged and replaces only what the swarm is attracted to.

A multi-objective problem has no single best position and no single worst one, so the food
source and the enemy cannot be "the best and worst seen". Instead the run maintains a
**Pareto archive** — a set of mutually non-dominated solutions — partitioned into a grid of
hypercubes over objective space, and each iteration draws:

- the **food source** from a _sparsely_ populated hypercube, weighting each occupied cell
  `1/N^β`, which pulls the swarm toward the thin parts of the front
- the **enemy** from a _crowded_ hypercube, weighting each cell `N^γ`, which pushes the swarm
  away from the parts already well covered

Both are drawn **once per iteration**, not once per dragonfly, as the reference MODA does: the
whole swarm is pulled toward the same sparse region and pushed away from the same crowded one
for the length of that iteration.

Everything else is the single-objective algorithm, called through the very same helper.
`moState.advance` builds a `runState` view over the swarm with the drawn food and enemy
substituted, then calls `prepareSwarmStep` — the same function `Optimize` calls. MODA changes
what the swarm is attracted to, not how it moves.

### The archive

`ParetoArchive` (`multiobjective.go`) keeps solutions in insertion order and restores its
central invariant on **every** mutation, not merely at the end of a run:

- `Add` rejects a candidate that an archived solution dominates, or that duplicates an existing
  objective vector exactly. Otherwise every solution the candidate dominates is removed and the
  candidate is appended.
- An insert past `MaxSize` evicts one member of the most crowded hypercube, chosen by a roulette
  weighted `N^δ`, so the archive never exceeds its capacity.
- `IsNonDominated()` re-checks the invariant in `O(n²·m)`. It is cheap enough that the tests
  assert it after every mutation.

The grid bounds are the exact per-objective minimum and maximum of the current contents, with
no inflation, so the extreme solutions land in the first and last bin. The bounds move whenever
the archive does — a solution's cell coordinates can change without the solution changing at
all. `occupiedCells` returns cells in ascending key order rather than in map order, because Go
randomizes map iteration and an unsorted listing would make every roulette draw depend on the
map's internal layout and break reproducibility.

### The archive parameters are UNVERIFIED

`DefaultArchiveBeta = 4`, `DefaultArchiveGamma = 2`, `DefaultArchiveDelta = 2`,
`DefaultArchiveNGrid = 10`, `DefaultArchiveSize = 100`.

These are the **MOPSO defaults** from Coello Coello et al. (2004), the lineage MODA borrows its
archive from. They have **not** been read off the author's `MODA.m`, which this repository does
not have. Do not cite them as settled values from the DA paper. The source says so at the
declaration, `PLAN.md` §1.7 says so, and this page says so: treat them as working defaults
until someone checks them against the reference.

What each one does, so you can judge a change to it:

- **β (food)** — larger β concentrates the food draw more sharply on the emptiest cell.
  β = 0 makes the draw uniform over occupied cells.
- **γ (enemy)** — larger γ concentrates the enemy draw more sharply on the fullest cell.
- **δ (eviction)** — larger δ makes overflow deletion more aggressive about the crowded cell.
- **NGrid** — hypercubes per objective. More bins mean a finer notion of "crowded", and with a
  small archive most cells hold one solution and the weighting stops discriminating.

A negative or non-finite exponent is raised to zero rather than used, because a negative
exponent inverts the preference the roulette exists to express.

### What MODA does not have

- **No early stopping.** `Result`'s convergence criteria are defined against a single best
  cost, which a multi-objective run does not have. `MultiObjectiveResult.TerminationReason` is
  therefore always `maximum_iterations`; it is reported anyway so a MODA result reads like a
  single-objective one.
- **No parallel evaluation.** `MultiObjectiveConfig.Swarm.EnableParallel` is not honoured;
  `moState.evaluateSwarm` scores the swarm on the calling goroutine.
- **No constraint handling.** The multi-objective path does not go through
  `constraintEvaluator`. Fold constraints into your objective vector if you need them.
- **No binary mode.** MODA is continuous.

`ArchiveSizeCurve` is the multi-objective analogue of `ConvergenceCurve`: the archive size after
each completed iteration. A curve that stops growing early is the usual sign of a stagnated run.

## Key Innovations

1. **The food source and the enemy become draws from a distribution rather than facts about the
   population.** This is the whole variant, and it is what turns "keep the non-dominated set"
   into "keep a non-dominated set that is spread out".
2. **The grid does double duty.** The same occupancy counts drive food selection, enemy
   selection and overflow eviction, with three exponents rather than three mechanisms.
3. **The invariant is maintained continuously.** Non-domination is restored on every insert,
   not repaired at the end. That is where an archive silently breaks, and where the tests
   therefore look.
4. **Reproducibility survives a map.** Every place a map could leak iteration order into the
   result is sorted first. A seeded MODA run reproduces exactly.

## Usage Examples

### ZDT1

```go
config := dragonfly.NewMultiObjectiveConfig()
config.ObjectiveFunc = dragonfly.ZDT1
config.Swarm.ProblemSize = 5
config.Swarm.LowerBound = 0
config.Swarm.UpperBound = 1
config.Swarm.MaxIterations = 400
config.Swarm.NPop = 60
config.ArchiveSize = 50
config.Swarm.Rand = rand.New(rand.NewSource(4242))

result, err := dragonfly.OptimizeMultiObjective(context.Background(), config)
if err != nil {
	log.Fatal(err)
}

fmt.Printf("archive size:  %d (capacity %d)\n", result.Archive.Len(), config.ArchiveSize)
fmt.Printf("non-dominated: %v\n", result.Archive.IsNonDominated())
fmt.Printf("evaluations:   %d\n", result.FuncEvalCount)

for _, solution := range result.Archive.Solutions[:3] {
	fmt.Printf("  f = %.4f\n", solution.ObjectiveValues)
}
```

Note the field name: `ParetoSolution.ObjectiveValues`, alongside `Position`, `GridIndex` and
`GridKey`.

### Exporting the front

```go
if err := result.ExportParetoCSV("front.csv"); err != nil {
	log.Fatal(err)
}
if err := result.ExportParetoJSON("front.json"); err != nil {
	log.Fatal(err)
}
```

The CSV carries one row per solution with an `index` column, one `objective_k` column per
objective and one `x_j` column per decision variable. The column count follows the archive's
contents, so an empty archive yields a header-only file rather than an error. The JSON document
adds the run summary: seed, evaluation count, iteration count, archive size and the
`archive_size_curve`.

### Through the variant layer

```go
variant := &dragonfly.MODAVariant{}
config := variant.GetMultiObjectiveConfig()
// ... fill in ObjectiveFunc and Swarm's ProblemSize and bounds ...
result, err := variant.RunMultiObjective(ctx, config)
```

`MODAVariant.Run` — the single-objective method every `AlgorithmVariant` has — always returns
`ErrMultiObjectiveVariant`. A MODA run has no single incumbent, so there is no honest `*Result`
to return, and returning a fabricated one would be worse than an error. For the same reason
`ComparisonRunner` defaults to `SingleObjectiveVariants()` and rejects a multi-objective
variant: every statistic it computes is defined over a scalar cost.

`MODAVariant.GetConfig()` returns the _swarm block_ of the default MODA configuration, so that
a caller inspecting the variant's mechanics through the common interface sees the schedules a
MODA run actually uses. It is not on its own runnable as MODA.

## Parameters

### `MultiObjectiveConfig`

| Field           | Default              | Meaning                                                             |
| --------------- | -------------------- | ------------------------------------------------------------------- |
| `ObjectiveFunc` | — (required)         | `func([]float64) []float64`, one value per objective, all minimized |
| `Swarm`         | `NewDefaultConfig()` | the shared mechanics; `Swarm.ObjectiveFunc` is ignored              |
| `Beta`          | `4`                  | food-selection exponent — **unverified**                            |
| `Gamma`         | `2`                  | enemy-selection exponent — **unverified**                           |
| `Delta`         | `2`                  | overflow-eviction exponent — **unverified**                         |
| `ArchiveSize`   | `100`                | archive capacity — **unverified**                                   |
| `NGrid`         | `10`                 | hypercubes per objective — **unverified**                           |

You must set `ObjectiveFunc` and `Swarm`'s `ProblemSize`, `LowerBound` and `UpperBound`.
`Beta`, `Gamma` and `Delta` must be non-negative and finite; `ArchiveSize` and `NGrid` must be
positive. The swarm block is validated by the same `validateConfig` a single-objective run uses.

### `MultiObjectiveResult`

| Field               | Meaning                                                    |
| ------------------- | ---------------------------------------------------------- |
| `Archive`           | the approximation of the Pareto front — this is the result |
| `ArchiveSizeCurve`  | archive size after each completed iteration                |
| `TerminationReason` | always `maximum_iterations`                                |
| `FuncEvalCount`     | calls to `ObjectiveFunc`                                   |
| `IterationCount`    | completed iterations                                       |
| `Seed`              | the recorded seed                                          |

## Benefits

- **One implementation of the search.** MODA calls the same `prepareSwarmStep` the continuous
  variant does, so a fix to the step update fixes both, and there is no second copy to drift.
- **A spread front, not a clustered one.** The sparse-cell food draw actively pushes the swarm
  toward under-covered regions, which is what the crowding heuristics in most multi-objective
  metaheuristics exist to achieve.
- **The invariant is checkable.** `IsNonDominated()` is exported, so an application can assert
  the property the archive promises rather than trusting it.
- **Export is built in.** CSV for plotting, JSON for keeping.

## Performance

All measured with seed 4242, `NPop` 60, 400 iterations, `ArchiveSize` 50, 24,060 evaluations —
the settings `multiobjective_test.go` uses for its front-recovery tests. "Median distance" is
the median vertical distance from an archived point to the analytic front; "spread" is the
range of `f1` the archive covers.

| Problem | Dimensions | Archive | Spread | Closest to front | Median distance |
| ------- | ---------: | ------: | -----: | ---------------: | --------------: |
| ZDT1    |          5 |      50 |  0.960 |            0.000 |           0.000 |
| ZDT3    |          5 |      21 |  0.846 |            0.000 |           0.324 |

At 5 dimensions MODA recovers ZDT1's front essentially exactly and covers most of it. ZDT3's
five disconnected pieces are harder to sit on, which is why its tolerances in the test suite
are looser.

Dimensionality is the binding constraint, and the honest number is less flattering. At the ZDT
suite's original 30 dimensions, with `NPop` 100 and 1000 iterations, the archive holds 44
mutually non-dominated ZDT1 solutions but its lowest `f2` is 1.39 — the true front has
`f2 ∈ [0, 1]`. The run has found the shape of the trade-off and not the front itself. This is
the same exploitation weakness [standard DA](standard-da.md#performance) has, seen from the
multi-objective side: the convergence factor reaches zero at the halfway mark and the swarm
stops closing the distance. SchafferN1, a one-variable problem, is solved outright.

Cost, on Linux/amd64 with Go 1.26.0 on an AMD Ryzen 5 4600H:
`BenchmarkOptimizeMultiObjectiveBaseline` — 30-dimensional ZDT1, 100 iterations, `NPop` 40 — is
55.7 ms and 59,133 allocations per run, against 49.8 ms and 24,927 for the single-objective
anchor. The extra allocations are the archive: every accepted candidate is stored as a deep
copy, and `updateGrid` reassigns every cell on every mutation. `MODAVariant.EstimatedOverhead()`
reports `1.2`, which the measurement supports.

One measured surprise worth knowing: `BenchmarkOptimizeSchafferN1_MODA` is 61.0 ms against
6–10 ms for the ZDT problems at the same settings. SchafferN1 is one-dimensional, so its front
is dense and the archive stays full, and a full archive makes `Add`'s domination sweep and
`updateGrid`'s reassignment the dominant cost. Archive maintenance, not the swarm, is what MODA
pays for.

## When to Use

**Use MODA when:**

- there are genuinely several objectives and you want the trade-off curve, not one compromise
- a weighted-sum scalarization would hide the shape of the trade-off, or you cannot defend the
  weights
- the decision variables are continuous
- the problem's dimensionality is modest. On this implementation, 5-dimensional ZDT problems
  are recovered well and 30-dimensional ones are not.

**Use something else when:**

- there is one objective — MODA on a single objective degenerates to an archive of one point,
  and `MODAVariant.ApplicableTo` scores it `0.1` accordingly
- the variables are binary — there is no multi-objective binary variant here
- you need constraints, early stopping or parallel evaluation — none is wired into the
  multi-objective path

## Parameter Tuning Guide

1. **`ArchiveSize` and `NGrid` together.** They are not independent: with `NGrid = 10` over two
   objectives there are up to 100 cells, so an archive of 100 averages one solution per cell and
   the occupancy weighting has almost nothing to discriminate on. Either raise `ArchiveSize` or
   lower `NGrid` until cells hold several solutions each.
2. **`Swarm.NPop` and `Swarm.MaxIterations`.** As with DA, these dominate everything else. The
   30-dimensional ZDT result above did not fail for want of tuning.
3. **`Beta`, if the front is clustered.** Raising it sharpens the pull toward empty cells.
   Because it is an exponent on a count, small changes matter: `4 → 6` is a large change in
   behaviour, not a nudge.
4. **`Gamma` and `Delta` last.** They shape where the swarm is pushed from and which solution
   is dropped on overflow, and both are secondary to how well the swarm converges at all.
5. **`Swarm.RadiusGrowth`**, for the same reason it matters in DA: it is the exploration and
   exploitation dial, and the multi-objective run needs exploration for longer.

Because all five archive parameters are unverified against the reference, a tuning study here
has more headroom than one on the continuous variant — but also less ground truth to compare
against.

## Compared with the Other Variants

| Aspect              | MODA                                   | Standard DA               | BDA                     |
| ------------------- | -------------------------------------- | ------------------------- | ----------------------- |
| Objectives          | several                                | one                       | one                     |
| Food source         | roulette draw from a sparse hypercube  | the best position seen    | the best position seen  |
| Enemy               | roulette draw from a crowded hypercube | the worst position seen   | the worst position seen |
| Result              | a Pareto archive                       | one incumbent + the enemy | one bit string          |
| Entry point         | `OptimizeMultiObjective`               | `Optimize`                | `OptimizeBinary`        |
| Early stopping      | not available                          | target cost, stagnation   | target cost, stagnation |
| Parallel evaluation | not available                          | yes                       | yes                     |
| Constraints         | not available                          | yes                       | yes                     |
| Overhead            | about 1.2x                             | 1.0x (baseline)           | 1.0x                    |

## Related Documentation

- [Standard DA](standard-da.md) — the step update MODA reuses verbatim
- [BDA](bda.md) — the binary variant
- [Configuration Guide](../api/configuration.md) — the swarm block MODA shares
- [Benchmark Functions](../benchmarks.md) — ZDT1–ZDT3 and SchafferN1
- [Performance and Profiling](../performance.md) — measured timings and allocation counts
- [Research References](../research.md) — citations, BibTeX, and the verification status of
  every borrowed constant
