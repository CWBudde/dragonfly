# Standard Dragonfly Algorithm (DA)

## Research Reference

**Mirjalili, S. (2016). Dragonfly algorithm: a new meta-heuristic optimization technique for
solving single-objective, discrete, and multi-objective problems. _Neural Computing and
Applications_, 27(4), 1053–1073.**

<https://doi.org/10.1007/s00521-015-1920-1>

The three swarming primitives DA shares with every boids-style model come from:

**Reynolds, C. W. (1987). Flocks, herds and schools: A distributed behavioral model. _ACM
SIGGRAPH Computer Graphics_, 21(4), 25–34.**

Reference implementation: the author's `DA.m`. Where the reference code and a cleaner
formulation disagree, this library implements the reference and exposes the alternative behind
a `Config` field. The three places it deliberately departs from `DA.m` are listed in the
[README](../../README.md#deviations-from-the-reference-matlab).

## Overview

DA maintains one population of dragonflies. Each carries a position `X` and a step `ΔX` — the
velocity analogue — and the swarm as a whole carries two reference points:

- the **food source** `X⁺`, the best position seen so far
- the **enemy** `X⁻`, the worst position seen so far

There is no per-individual personal best. DA has no memory beyond those two swarm-level
positions, which is what makes it a smaller algorithm than PSO despite having more terms.

Every iteration, each dragonfly scans the swarm for neighbours inside the current radius,
builds five vectors from them, weights those vectors with schedules that depend only on how far
the run has progressed, and adds the result to its position. Adaptive weights move the swarm
from exploration to exploitation; a dragonfly with no neighbours and no food in reach performs
a Lévy random walk instead.

### The five swarming primitives

For dragonfly `i` with `N` neighbours inside radius `r` (`swarm.go`):

```
Separation  S_i = -Σ_j (X_i - X_j)          repel from local crowding
Alignment   A_i = (Σ_j V_j) / N             match neighbour steps
Cohesion    C_i = (Σ_j X_j) / N - X_i       move toward the local centroid
Food        F_i = X⁺ - X_i                  attraction to the best position seen
Enemy       E_i = X⁻ + X_i                  distraction from the worst position seen
```

**The enemy term is a sum.** `E_i = X⁻ + X_i`, not `X⁻ - X_i`. Every other primitive is a
difference, so the difference form reads correct, compiles fine, and produces a convergence
curve that looks entirely plausible. It is nevertheless wrong: the paper and `DA.m` both add.
`enemyVector` in `swarm.go` says so in its doc comment, and `swarm_test.go` pins it with a
hand-computed case. Read the test before "fixing" it.

With no neighbours, `S_i` and `C_i` are the zero vector and `A_i` falls back to the dragonfly's
own step — all three are the reference implementation's fallbacks, reproduced on purpose.

### The neighbourhood scan

A neighbour is one whose distance is within the radius in **every component**:

```
all(|X_i,k - X_j,k| <= r)   and   any(X_i,k - X_j,k != 0)
```

This is a box test, not a ball test. A Euclidean `‖X_i - X_j‖ <= r` shortcut accepts only the
inscribed ball, silently shrinks every neighbourhood, and degrades convergence without failing
any end-to-end test. The second clause excludes a dragonfly from being its own neighbour, and
also excludes any dragonfly that happens to sit on exactly the same position — again the
reference behaviour, not an oversight.

The food-in-range test (`foodInRadius` in `dragonfly.go`) deliberately does **not** reuse this
helper: it has no all-zero exclusion, because a dragonfly sitting exactly on the food source is
at distance zero in every dimension and must still see the food as in range.

### The two-branch step update

This is the single most important fidelity detail in the algorithm (`parallel_phases.go`,
`prepareSwarmStep`):

```
if food is NOT within r in every dimension:      # local swarming only
    if neighbours > 1:
        ΔX_j = w·ΔX_j + rand·A_j + rand·C_j + rand·S_j    # per-dimension rand
        clamp ΔX to ±ΔX_max
        X += ΔX                                           # no food, no enemy term
    else:
        X += Levy(d) ⊙ X                                  # Lévy random walk
        ΔX = 0
else:                                            # the full five-factor step
    ΔX = (s·S + a·A + c·C + f·F + e·E) + w·ΔX
    clamp ΔX to ±ΔX_max
    X += ΔX
```

Collapsing this into one unconditional five-factor step is the classic porting bug. It still
converges on Sphere, which is exactly what lets it survive review. Two details inside the
branches matter as much as the branching itself: the three `rand` factors in the swarming
branch are drawn **per dimension** (drawing one per dragonfly makes the three primitives share
a scaling factor and changes the search, not just the random stream), and the Lévy branch
replaces the step rather than contributing to it, so `ΔX` is reset to zero.

### Adaptive weight schedules

All of DA's time dependence lives in `weights.go`, one struct per iteration:

```
w  = InertiaWeightStart - (t/T)·(start - end)          inertia, linearly decreasing
mc = max(0, 0.1·(1 - 2t/T))                            the shared convergence factor
s  = 2·rand·mc     a = 2·rand·mc     c = 2·rand·mc
f  = 2·rand
e  = mc, forced to exactly 0 once t > EnemyCutoffFraction·T
r  = (ub-lb)/RadiusInitialDivisor + (ub-lb)·(t/T)·RadiusGrowth
ΔX_max = (ub-lb)·MaxStepRatio
```

`mc` decays from `0.1` to zero over the **first half** of the run and stays there. That single
line explains most of DA's behaviour: past the halfway point, separation, alignment, cohesion
and the enemy term are all zero, and only the food term and inertia still move a dragonfly. The
swarm collapses onto the incumbent and stops improving. It is why DA explores well and exploits
badly, and why the numbers in [benchmarks.md](../benchmarks.md) are what they are.

One consequence worth knowing: because `e = mc` and `mc` is already zero at `t = T/2`, the
`EnemyCutoffFraction` cutoff at `0.75·T` never actually bites at its default value — the enemy
term has been zero for a quarter of the run by the time the cutoff arrives. The field only
changes behaviour when set below `0.5`, where it switches the enemy term off earlier than `mc`
would. It is kept because the paper states the cutoff as a separate rule.

`computeWeights` takes exactly four uniform draws per call — separation, alignment, cohesion,
food — whether or not those weights are pinned. A pinned weight discards its draw rather than
skipping it, so overriding one weight changes only that weight and leaves the rest of the run's
random stream aligned with a default-config run of the same seed.

### Boundary handling

DA's default is **wrap with step reset**, not clamp:

```
if x_j > ub { x_j = lb ; Δx_j = rand() }
if x_j < lb { x_j = ub ; Δx_j = rand() }
```

The redraw of the step component is half the rule. Dropping it changes the exploration
behaviour. `Config.BoundaryMethod` selects `"wrap"` (default), `"clamp"` or `"reflect"`; see
the [Configuration Guide](../api/configuration.md#boundary-handling).

## Key Innovations

Relative to the boids model DA extends, and to PSO, which it most resembles:

1. **Two swarm-level attractors instead of one.** The food source pulls, the enemy pushes. Both
   are recomputed from the population every iteration, and the enemy is reported in
   `Result.Worst` because every step of the run was computed against it.
2. **No personal best.** DA carries no per-individual memory. Everything an individual knows
   comes from its current neighbourhood and the two swarm-level positions.
3. **A neighbourhood radius that grows with the run.** Early on, neighbourhoods are local and
   the swarm searches as many small flocks — "static swarming". By the end, the radius covers
   the box and the swarm is one flock around the food source — "dynamic swarming". The
   transition is the algorithm's exploration-to-exploitation schedule, and it is geometric
   rather than a tuned coefficient.
4. **A Lévy walk as the fallback for isolation.** A dragonfly with no neighbours and no food in
   reach is not stalled: it takes a heavy-tailed multiplicative jump, which is what lets the
   swarm cross a basin it has no gradient information about.

## Usage Examples

### Minimal run

```go
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
fmt.Printf("evaluations: %d\n", result.FuncEvalCount)
fmt.Printf("enemy cost:  %.6g\n", result.Worst.Cost)
```

### Watching the run and stopping early

```go
target := 1e-4
config.Convergence = &dragonfly.ConvergenceConfig{
	TargetCost:           &target,
	MinImprovement:       1e-9,
	StagnationIterations: 20,
	MinIterations:        5,
}

result, err := dragonfly.OptimizeContext(ctx, config,
	dragonfly.WithInitialPopulation([][]float64{{1, 1, 1, 1, 1}}),
	dragonfly.WithProgressObserver(func(p dragonfly.Progress) {
		fmt.Printf("iteration %d: best %.6g\n", p.Iteration, p.Best.Cost)
	}),
)
// result.TerminationReason is one of maximum_iterations, target_cost, stagnation
```

### Pinning a weight

```go
config.EnemyWeight = 0 // switch the enemy term off entirely, for the whole run
// every other weight stays on its schedule, and the random stream stays aligned
```

Remember that `0` is a pinned value. `WeightAuto` (`-1`) is what means "use the schedule".

## Parameters

### Required

| Field                       | Meaning                                 |
| --------------------------- | --------------------------------------- |
| `ObjectiveFunc`             | the function to minimize                |
| `ProblemSize`               | number of decision variables            |
| `LowerBound` / `UpperBound` | the search box, same for all dimensions |

### Population and run length

| Field           | Default | Meaning            |
| --------------- | ------: | ------------------ |
| `NPop`          |      40 | swarm size         |
| `MaxIterations` |    1000 | hard iteration cap |

### Weight schedules

| Field                                                                                | Default      | Meaning                                               |
| ------------------------------------------------------------------------------------ | ------------ | ----------------------------------------------------- |
| `InertiaWeightStart` / `InertiaWeightEnd`                                            | 0.9 / 0.4    | brackets of the linearly decreasing `w`               |
| `SeparationWeight`, `AlignmentWeight`, `CohesionWeight`, `FoodWeight`, `EnemyWeight` | `WeightAuto` | pin a weight, or leave it on its schedule             |
| `EnemyCutoffFraction`                                                                | 0.75         | fraction of the run after which `e` is forced to zero |

### Geometry

| Field                  |  Default | Meaning                             |
| ---------------------- | -------: | ----------------------------------- |
| `RadiusInitialDivisor` |      4.0 | `r` starts at `(ub-lb)/divisor`     |
| `RadiusGrowth`         |      2.0 | `r` grows by `(ub-lb)·(t/T)·growth` |
| `MaxStepRatio`         |      0.1 | step clamp `ΔX_max = (ub-lb)·ratio` |
| `BoundaryMethod`       | `"wrap"` | `"wrap"`, `"clamp"` or `"reflect"`  |

### Lévy walk

| Field         | Default | Meaning                                                         |
| ------------- | ------: | --------------------------------------------------------------- |
| `UseLevyWalk` |  `true` | `false` leaves an isolated dragonfly still for that iteration   |
| `LevyBeta`    |     1.5 | Mantegna stability index; must be in the open interval `(0, 2)` |
| `LevyScale`   |    0.01 | multiplicative step scale                                       |

The full field-by-field reference, including constraints, convergence, parallelism and RNG, is
in the [Configuration Guide](../api/configuration.md).

## Benefits

- **Strong exploration.** The growing radius and the Lévy fallback keep the swarm spread out
  far longer than a PSO of comparable size, which is what makes DA competitive on rugged
  landscapes despite its weak endgame.
- **Very few tuned constants.** Everything that changes between iterations is a function of
  `t/T` and the box width. There is no learning rate, no temperature, no crossover probability.
- **Cheap per individual.** No personal-best bookkeeping, no archive, no sorting. The only
  quadratic cost is the neighbour scan.
- **Fully deterministic.** A seeded run reproduces bit-for-bit, sequentially or in parallel.

## Performance

Measured on this implementation: 10 dimensions, 500 iterations, `NPop` 40, default config,
seeds 1000–1014, median and mean of 15 runs. Lower is better; the global optimum is 0 for every
function except Michalewicz.

| Function           | Bounds            |    Median |      Mean |      Best |
| ------------------ | ----------------- | --------: | --------: | --------: |
| Sphere             | [-100, 100]       |     87.17 |     87.33 |     15.11 |
| Rastrigin          | [-5.12, 5.12]     |     27.88 |     29.78 |     17.81 |
| Rosenbrock         | [-5, 10]          |     152.4 |     175.7 |     30.62 |
| Ackley             | [-32.768, 32.768] |     5.837 |     5.861 |     3.263 |
| Griewank           | [-600, 600]       |     1.801 |     1.983 |     1.034 |
| Schwefel           | [-500, 500]       |      1590 |      1591 |     936.4 |
| Levy               | [-10, 10]         |    0.8646 |     1.095 |    0.1709 |
| Zakharov           | [-10, 10]         |     6.842 |     7.556 |      1.73 |
| Michalewicz        | [0, π]            |    -5.859 |    -5.961 |    -7.201 |
| DixonPrice         | [-10, 10]         |        16 |     18.62 |     1.675 |
| BentCigar          | [-100, 100]       | 6.564e+07 | 7.816e+07 | 5.045e+06 |
| Discus             | [-100, 100]       |      5719 |      5479 |     435.8 |
| Weierstrass        | [-0.5, 0.5]       |     3.883 |     4.023 |     2.693 |
| HappyCat           | [-2, 2]           |    0.2386 |    0.3023 |   0.08158 |
| ExpandedSchafferF6 | [-100, 100]       |     2.888 |     2.713 |     1.965 |

These are unflattering, and they are correct. `regression_test.go` watches five of them with a
uniform 3x tolerance against round reference means chosen from exactly this kind of block, and
its header comment makes the same point: DA's convergence factor reaches zero at the halfway
mark, so the paper's algorithm stalls well short of the optimum. Costs also scale with the box
— Sphere over `[-10, 10]` lands near `1` rather than near `87`, because the schedules are all
written in units of `(ub-lb)`.

Timing on Linux/amd64, Go 1.26.0, AMD Ryzen 5 4600H: a 30-dimensional Sphere over 100
iterations with `NPop` 40 costs about 49.8 ms and 24,927 allocations per run. See
[performance.md](../performance.md).

## When to Use

**Use standard DA when:**

- the problem is continuous and single-objective
- you want a well-explored search rather than a tightly converged one
- you need a reproducible baseline to compare something else against
- the objective is cheap, and you can afford `NPop × MaxIterations` evaluations

**Use something else when:**

- the search space is binary or discrete → [BDA](bda.md)
- there are several objectives to trade off → [MODA](moda.md)
- you need the last few digits of the optimum — DA will get you to the right basin and then
  stop. Consider running DA to locate the basin and a local method to polish it.

## Parameter Tuning Guide

Tune in this order. The first two matter far more than the rest.

1. **`MaxIterations` and `NPop` first.** More of both always helps and nothing else can
   substitute. `NPop` is the expensive one: the neighbour scan is `O(n²·d)`, so doubling the
   swarm roughly triples the wall-clock cost per iteration. Start from `NewDefaultConfig`
   (40 × 1000), or call `AutoTuneConfig` after setting `ProblemSize`.
2. **`RadiusGrowth`, for the exploration/exploitation balance.** The default `2.0` has the
   radius covering the box well before the run ends. Lower it (`NewHighDimensionalConfig` uses
   `1.0`) when the swarm collapses too early — in high dimensions a radius that reaches the
   whole box makes every dragonfly a neighbour of every other. Raise it (`NewFastConvergenceConfig`
   uses `4.0`) when you want a single flock exploiting sooner.
3. **`MaxStepRatio`, when the swarm overshoots.** The default clamps a step at a tenth of the
   box. Lower it on ill-conditioned problems with a narrow valley (Rosenbrock, BentCigar,
   Discus), where a tenth of the box jumps clean over the basin. Raise it, with a matching
   `RadiusGrowth` increase, for a short run.
4. **`BoundaryMethod`, if the wrap surprises you.** Wrapping is genuine exploration on an
   unconstrained box and actively unhelpful when the bounds encode a real constraint. On a
   10-dimensional Rosenbrock over `[-5, 10]`, seed 17, 300 iterations, the three rules gave
   114.1 (wrap), 630.4 (clamp) and 219.5 (reflect) — the ordering is problem-dependent, so
   measure rather than assume.
5. **`EnemyWeight = 0`, to test whether the enemy term is helping.** It is the cheapest
   ablation the library offers, and on some problems the answer is that it is not.
6. **`InertiaWeightStart` / `InertiaWeightEnd` last.** The `0.9 → 0.4` bracket is inherited
   from PSO practice and rarely repays attention before the four settings above.

Two settings that are usually a mistake to touch: `RadiusInitialDivisor` (the radius schedule's
shape is better controlled through `RadiusGrowth`) and `LevyBeta` (β = 1.5 is the value the
σ constant is verified against; other values are valid but untested).

## Compared with the Other Variants

| Aspect             | Standard DA                      | Consider instead                            |
| ------------------ | -------------------------------- | ------------------------------------------- |
| Search space       | continuous, one box for all dims | [BDA](bda.md) for 0/1 vectors               |
| Objectives         | one                              | [MODA](moda.md) for several                 |
| Result             | a single incumbent + the enemy   | MODA returns a Pareto archive               |
| Per-iteration cost | baseline (1.0x)                  | BDA 1.0x, MODA about 1.2x                   |
| Boundary handling  | wrap / clamp / reflect           | BDA ignores it — a bit cannot leave `[0,1]` |
| Lévy walk          | yes, for isolated dragonflies    | BDA has no counterpart; MODA keeps it       |
| Exploitation       | weak past the halfway point      | no variant fixes this; it is the algorithm  |

## Related Documentation

- [BDA — the binary variant](bda.md)
- [MODA — the multi-objective variant](moda.md)
- [Configuration Guide](../api/configuration.md) — every field, and how it is resolved
- [Run Lifecycle](../api/run-lifecycle.md) — cancellation, observers, export
- [Benchmark Functions](../benchmarks.md) — bounds, optima and measured results
- [Performance and Profiling](../performance.md) — timings, scaling, profiling recipes
- [Research References](../research.md) — citations and BibTeX
