# Configuration Guide

`Config` is one flat struct — no nested option groups — with snake_case JSON tags. Field order
in the source is chosen for `govet`'s `fieldalignment` (pointers/interfaces → strings → float64
→ int → bool), which is a lint gate rather than a style preference; this page groups the fields
by what they mean instead.

Always start from a factory function. `ObjectiveFunc`, `ProblemSize`, `LowerBound` and
`UpperBound` have no usable defaults, and everything else does.

## Complete field index

| Field                  | JSON                     | Type                 | Default (`NewDefaultConfig`) |
| ---------------------- | ------------------------ | -------------------- | ---------------------------- |
| `ObjectiveFunc`        | not serialized           | `ObjectiveFunction`  | — (required)                 |
| `Rand`                 | not serialized           | `*rand.Rand`         | `nil` → drawn and recorded   |
| `Convergence`          | `convergence`            | `*ConvergenceConfig` | `nil` (no early stopping)    |
| `Constraints`          | `constraints`            | `*ConstraintConfig`  | `nil` (unconstrained)        |
| `BoundaryMethod`       | `boundary_method`        | `BoundaryMethod`     | `"wrap"`                     |
| `TransferFunc`         | `transfer_function`      | `TransferFunction`   | `""` → `v3` (BDA only)       |
| `LowerBound`           | `lower_bound`            | `float64`            | — (required)                 |
| `UpperBound`           | `upper_bound`            | `float64`            | — (required)                 |
| `InertiaWeightStart`   | `inertia_weight_start`   | `float64`            | `0.9`                        |
| `InertiaWeightEnd`     | `inertia_weight_end`     | `float64`            | `0.4`                        |
| `SeparationWeight`     | `separation_weight`      | `float64`            | `WeightAuto`                 |
| `AlignmentWeight`      | `alignment_weight`       | `float64`            | `WeightAuto`                 |
| `CohesionWeight`       | `cohesion_weight`        | `float64`            | `WeightAuto`                 |
| `FoodWeight`           | `food_weight`            | `float64`            | `WeightAuto`                 |
| `EnemyWeight`          | `enemy_weight`           | `float64`            | `WeightAuto`                 |
| `RadiusInitialDivisor` | `radius_initial_divisor` | `float64`            | `4.0`                        |
| `RadiusGrowth`         | `radius_growth`          | `float64`            | `2.0`                        |
| `MaxStepRatio`         | `max_step_ratio`         | `float64`            | `0.1`                        |
| `EnemyCutoffFraction`  | `enemy_cutoff_fraction`  | `float64`            | `0.75`                       |
| `LevyBeta`             | `levy_beta`              | `float64`            | `1.5`                        |
| `LevyScale`            | `levy_scale`             | `float64`            | `0.01`                       |
| `ProblemSize`          | `problem_size`           | `int`                | — (required)                 |
| `NPop`                 | `npop`                   | `int`                | `40`                         |
| `MaxIterations`        | `max_iterations`         | `int`                | `1000`                       |
| `MaxWorkers`           | `max_workers`            | `int`                | `runtime.NumCPU()`           |
| `UseLevyWalk`          | `use_levy_walk`          | `bool`               | `true`                       |
| `EnableParallel`       | `enable_parallel`        | `bool`               | `false`                      |
| `UseBinary`            | `use_binary`             | `bool`               | `false`                      |

## Problem parameters

| Field           | Meaning                                                                        |
| --------------- | ------------------------------------------------------------------------------ |
| `ObjectiveFunc` | `func([]float64) float64`. The library **minimizes**.                          |
| `ProblemSize`   | Number of decision variables. Must be positive.                                |
| `LowerBound`    | Lower bound, the same for every dimension.                                     |
| `UpperBound`    | Upper bound, the same for every dimension. Must be finite and above the lower. |

There is one box for the whole problem — the search space is a hypercube, not a per-dimension
list of ranges. If your variables have genuinely different ranges, rescale them inside your
objective: take a position in `[0, 1]^d` and map each component to its own range before
evaluating. Equal bounds are rejected along with inverted ones, because a zero-width box makes
every schedule that divides by `(ub-lb)` degenerate.

Almost every schedule in the algorithm is written in units of `(ub-lb)`, so the box width is not
a neutral choice: it sets the neighbourhood radius, the step clamp and, through them, the scale
of the whole search. It also sets the scale of the reported cost — Sphere over `[-100, 100]`
lands near 87 where the same run over `[-10, 10]` lands near 1.

### Example

```go
config := dragonfly.NewDefaultConfig()
config.ObjectiveFunc = func(x []float64) float64 {
	return x[0]*x[0] + math.Sin(x[1])
}
config.ProblemSize = 2
config.LowerBound = -5
config.UpperBound = 5
```

## Population and run length

| Field           | Default | Meaning                               |
| --------------- | ------: | ------------------------------------- |
| `NPop`          |      40 | Swarm size. Must be positive.         |
| `MaxIterations` |    1000 | Hard iteration cap. Must be positive. |

Objective evaluations total `NPop × (MaxIterations + 1)` — the extra population is the initial
swarm, which is scored before the first iteration. A default run therefore costs 40,040
evaluations.

`NPop` is the expensive one. The neighbourhood scan is `O(n²·d)`, so cost grows roughly as
`n^1.8` in practice: 10 → 40 → 100 → 250 dragonflies cost 1.27 → 8.76 → 39.5 → 207 ms per run
in the measurements in [../performance.md](../performance.md). `MaxIterations` is linear.

## The `WeightAuto` sentinel

```go
const WeightAuto = -1.0
```

Every one of the five swarming weights defaults to `WeightAuto`, which means "follow the
paper's schedule". Any other finite value pins that weight to a constant for the whole run.

**`0` is a legitimate pinned value.** `config.EnemyWeight = 0` means "switch the enemy term off
for the whole run", not "use the default". Code that decides whether a weight is automatic must
test against `WeightAuto`, never against zero. This bites hardest with JSON: absent fields
decode as Go zero values, so a hand-authored partial configuration file silently pins every
weight it omits to zero. Write configuration files with `SaveConfig`, which always emits every
field.

Pinning a weight does not shift the random stream. `computeWeights` takes exactly four uniform
draws per call — separation, alignment, cohesion, food, in that order — whether or not those
weights are pinned; a pinned weight discards its draw rather than skipping it. So a run with one
weight overridden consumes the same random numbers everywhere else as a default-config run of
the same seed, and the override can be assessed in isolation.

## Weight schedules

| Field                                     |      Default | Constraint             |
| ----------------------------------------- | -----------: | ---------------------- |
| `InertiaWeightStart` / `InertiaWeightEnd` |    0.9 / 0.4 | finite                 |
| `SeparationWeight`                        | `WeightAuto` | `WeightAuto` or finite |
| `AlignmentWeight`                         | `WeightAuto` | `WeightAuto` or finite |
| `CohesionWeight`                          | `WeightAuto` | `WeightAuto` or finite |
| `FoodWeight`                              | `WeightAuto` | `WeightAuto` or finite |
| `EnemyWeight`                             | `WeightAuto` | `WeightAuto` or finite |
| `EnemyCutoffFraction`                     |         0.75 | in `[0, 1]`            |

The schedules themselves, with `t` the 0-based iteration and `T` = `MaxIterations`:

```
w  = InertiaWeightStart - (t/T)·(InertiaWeightStart - InertiaWeightEnd)
mc = max(0, 0.1·(1 - 2t/T))
s  = 2·rand·mc     a = 2·rand·mc     c = 2·rand·mc
f  = 2·rand
e  = mc, forced to exactly 0 once t > EnemyCutoffFraction·T
```

`mc` is the shared convergence factor and reaches zero at the **halfway point** of the run,
after which separation, alignment, cohesion and the enemy term are all zero and only the food
term and inertia still move a dragonfly. That is the single line that most shapes DA's
behaviour; see [standard-da.md](../algorithms/standard-da.md#adaptive-weight-schedules).

Because `e = mc`, `EnemyCutoffFraction` has no effect at its default of `0.75` — the enemy term
has already been zero since `t = T/2`. Values below `0.5` do have an effect.

A run of one iteration, or a nonsensical non-positive `T`, divides by one, so the schedules
degenerate to their starting values rather than to `NaN`.

## Boundary handling

| Value       | Behaviour                                                                                                             |
| ----------- | --------------------------------------------------------------------------------------------------------------------- |
| `"wrap"`    | **Default.** A component past a bound reappears at the opposite bound and its step component is redrawn from `[0,1)`. |
| `"clamp"`   | The component is pinned to the bound it crossed; the step is left alone.                                              |
| `"reflect"` | The component is mirrored back into the box and its step sign is flipped, repeatedly if needed.                       |

An empty `BoundaryMethod` resolves to `"wrap"`; anything else is rejected by validation. The
value is never written back, so a `Config` keeps meaning what its author wrote.

**The step reset is half the wrap rule.** Users arriving from PSO, GA or Mayfly expect clamping
and read wrapping as a bug, and code ported from Mayfly will silently apply a `maxVec`/`minVec`
clamp instead. Wrapping is genuinely part of DA's exploration behaviour on an unconstrained box.
It is unhelpful when the bounds encode a real constraint — a solution teleporting from one end
of a feasible range to the other is not a useful move — which is why the alternatives exist as a
named, documented choice.

Which one wins is problem-dependent. On a 10-dimensional Rosenbrock over `[-5, 10]`, seed 17,
300 iterations, the three gave best costs of 114.1 (wrap), 630.4 (clamp) and 219.5 (reflect).
Measure rather than assume.

Boundary repair runs **after** the position update in this implementation, where `DA.m` runs it
before. See the [README](../../README.md#deviations-from-the-reference-matlab) for why.

In binary mode `BoundaryMethod` is ignored entirely: a 0/1 vector cannot leave `[0, 1]`.

## Neighbourhood radius and the step clamp

| Field                  | Default | Constraint              |
| ---------------------- | ------: | ----------------------- |
| `RadiusInitialDivisor` |     4.0 | positive and finite     |
| `RadiusGrowth`         |     2.0 | non-negative and finite |
| `MaxStepRatio`         |     0.1 | positive and finite     |

```
r      = (ub-lb)/RadiusInitialDivisor + (ub-lb)·(t/T)·RadiusGrowth
ΔX_max = (ub-lb)·MaxStepRatio
```

`RadiusGrowth` is the exploration/exploitation dial. At the default `2.0` the radius reaches the
whole box well before the run ends, turning the swarm into one flock around the food source.
Lower it when the swarm collapses too early — in high dimensions a radius that covers the box
makes every dragonfly a neighbour of every other. `NewHighDimensionalConfig` uses `1.0`;
`NewFastConvergenceConfig` uses `4.0`.

`MaxStepRatio` bounds how far one iteration can move a dragonfly. Lower it on ill-conditioned
problems with a narrow valley, where a tenth of the box jumps clean over the basin.
`NewBinaryConfig` raises it to `6.0` for reasons specific to the transfer functions — see
[bda.md](../algorithms/bda.md#the-step-clamp-is-not-a-paper-constant).

## Lévy walk

| Field         | Default | Constraint                     |
| ------------- | ------: | ------------------------------ |
| `UseLevyWalk` |  `true` | —                              |
| `LevyBeta`    |     1.5 | open interval `(0, 2)`, finite |
| `LevyScale`   |    0.01 | non-negative and finite        |

A dragonfly with no food in range and at most one neighbour takes a Lévy step instead of a
swarming step:

```
X += Levy(d) ⊙ X       then ΔX = 0
Levy = LevyScale · r₁·σ / |r₂|^(1/β)          r₁, r₂ ~ N(0,1)
σ    = ( Γ(1+β)·sin(πβ/2) / ( Γ((1+β)/2)·β·2^((β-1)/2) ) )^(1/β)
```

Setting `UseLevyWalk = false` leaves such a dragonfly where it is for that iteration and only
resets its step. The Lévy branch is ignored entirely in binary mode.

`σ` and the `0.01` scale are **verified**: `levySigma(1.5)` evaluates to `0.6965745026`, the
accepted Mantegna value for β = 1.5. β outside `(0, 2)` is not a heavy-tailed distribution at
all — it is a division by zero or a negative exponent — which is why validation rejects it.

Because the Lévy step is multiplicative and heavy-tailed, a position can overflow to ±Inf in a
single iteration. `sanitizeVec` redraws any non-finite component uniformly from the box before
the boundary rule runs, so one unlucky draw cannot poison the run.

## Constraint handling

```go
config.Constraints = &dragonfly.ConstraintConfig{
	Handling:          dragonfly.ConstraintHandlingFeasibility,
	Inequalities:      []dragonfly.ConstraintFunction{ /* satisfied when g(x) <= 0 */ },
	Equalities:        []dragonfly.ConstraintFunction{ /* satisfied when |h(x)| <= tolerance */ },
	EqualityTolerance: 1e-6,
}
```

| Field               | Default         | Meaning                                                                    |
| ------------------- | --------------- | -------------------------------------------------------------------------- |
| `Handling`          | `"feasibility"` | `"feasibility"` (Deb's rules) or `"penalty"`                               |
| `Inequalities`      | none            | each contributes `max(0, g(x))` to the aggregate violation                 |
| `Equalities`        | none            | each contributes `max(0, \|h(x)\| - tolerance)`                            |
| `EqualityTolerance` | `0` (exact)     | must be finite and non-negative                                            |
| `PenaltyMethod`     | `"quadratic"`   | `"linear"` adds `factor·violation`; `"quadratic"` adds `factor·violation²` |
| `PenaltyFactor`     | `0`             | finite and non-negative; must be **positive** under penalty handling       |

**Deb's feasibility rules** are the default because they need no factor to tune:

1. a feasible candidate always beats an infeasible one
2. two feasible candidates are ranked by cost
3. two infeasible candidates are ranked by aggregate violation

**Penalty handling** ranks by penalized score instead, and consults the feasibility rules only
to break an exact tie. Quadratic is the default shape: it leaves small violations almost free
and makes large ones prohibitive, so the swarm can cross a thin infeasible ridge without
settling on the far side of a thick one. A penalty factor of zero would turn the policy into
plain cost minimization and silently ignore the constraints you wrote, so validation rejects it.

Three details worth knowing:

- `Result.GlobalBest.Cost` is always the **raw** cost your objective returned, never a penalized
  score. The violation is reported separately in `GlobalBest.ConstraintViolation`.
- A nil constraint function, or one returning a non-finite value, produces an infinite
  violation rather than an error: an unusable constraint has to lose every comparison.
- The target-cost stop refuses to fire on an infeasible incumbent. An infeasible position could
  otherwise undercut any target simply by ignoring the constraints.
- The constraint functions carry `json:"-"`; a JSON round-trip preserves the policy only.

The multi-objective path does not go through the constraint evaluator at all. Fold constraints
into your objective vector for MODA.

## Convergence detection

```go
target := 1e-6
config.Convergence = &dragonfly.ConvergenceConfig{
	TargetCost:           &target,
	MinImprovement:       1e-9,
	StagnationIterations: 50,
	MinIterations:        10,
}
```

| Field                  | Meaning                                                                                                                        |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `TargetCost`           | `*float64` — stop when the best cost is at or below it. A pointer so that a target of zero is distinguishable from "disabled". |
| `MinImprovement`       | The absolute reduction required to reset the stagnation counter. Non-negative.                                                 |
| `StagnationIterations` | Stop after this many consecutive iterations without a sufficient improvement. `0` disables.                                    |
| `MinIterations`        | Neither criterion may fire before this many completed iterations. Must be in `[0, MaxIterations]`; `0` behaves as `1`.         |

`MaxIterations` remains the hard upper bound. `Result.TerminationReason` says which criterion
ended the run.

The stagnation counter is maintained from the first observation regardless of `MinIterations`,
so a run that stalls during the warm-up stops as soon as the gate opens. What counts as an
improvement follows the active constraint policy: the penalized score under penalty handling,
the raw cost between two feasible candidates, the aggregate violation between two infeasible
ones — and crossing into feasibility always counts, whatever the margin.

MODA has no early stopping; its criteria are all defined against a single best cost.

## Parallel evaluation

| Field            | Default            | Meaning                                              |
| ---------------- | ------------------ | ---------------------------------------------------- |
| `EnableParallel` | `false`            | fan objective calls out across a bounded worker pool |
| `MaxWorkers`     | `runtime.NumCPU()` | worker bound; non-positive resolves to one per CPU   |

When `EnableParallel` is true, `ObjectiveFunc` may be called concurrently with distinct position
vectors and **must be safe for concurrent use**.

Parallelism never changes the answer. Every random draw an iteration makes — the weight
schedules, the neighbourhood scan, the step update, the boundary repair, and in BDA the bit-flip
tests — happens on the calling goroutine during the prepare phase. Workers read a finished
position, call the objective and write one evaluation each; they never draw a random number and
never touch shared swarm state. A seeded run is bit-identical with `EnableParallel` on or off,
and `TestParallelIsDeterministicForSeedAcrossSchedules` enforces it.

A cancelled batch commits nothing: the swarm carries either every cost from this iteration or
every cost from the previous one, never a mixture.

### When parallel evaluation helps

Only when the objective is expensive. Measured on Linux/amd64, Go 1.26.0, AMD Ryzen 5 4600H:

| Workload                                      | Sequential | Parallel |
| --------------------------------------------- | ---------: | -------: |
| 30-D Sphere, 100 iterations, `NPop` 40        |    49.8 ms |  57.3 ms |
| 10-D Rastrigin ×200, 50 iterations, `NPop` 30 |    37.3 ms |  22.3 ms |

On Sphere the pool is about 15% slower than the sequential path — dispatch and synchronization
cost more than a ten-nanosecond objective call. On an objective 200 times more expensive the
same workload is 1.67x faster. As a rule of thumb, leave `EnableParallel` off below roughly a
microsecond per evaluation, and measure your own objective rather than guessing.

The neighbourhood scan, not the objective, is what dominates large-swarm runs, and it is _not_
parallelized. `BenchmarkNeighborScan` exists to establish whether it is worth the complexity;
at `NPop` 500 with a late-run radius it costs 20.2 ms per scan.

MODA does not honour `EnableParallel` at all.

## Random number generation

```go
config.Rand = rand.New(rand.NewSource(42))
```

`Config.Rand` is the single injection point. When it is nil, `OptimizeContext` draws a seed from
the clock, creates a generator, **writes it back into the config**, and records the seed in
`Result.Seed`. Capture `Result.Seed`, feed it back, and you get the same trajectory:

```go
first, _ := dragonfly.Optimize(newConfig())

replay := newConfig()
replay.Rand = rand.New(rand.NewSource(first.Seed))

second, _ := dragonfly.Optimize(replay)
// second.GlobalBest.Cost == first.GlobalBest.Cost, exactly
```

One caveat, inherited from Mayfly's convention: when you supplied your own `*rand.Rand`, it is
that generator and not the recorded seed that drove the run. `Result.Seed` is then the unused
fallback, and reproducing the run means reusing your generator with the same starting state.

Because the config is mutated, a `Config` value is **not** reusable across runs as a template
once it has been through `Optimize` — the second run continues the first run's stream. Build a
fresh config per run, as the benchmarks and the comparison runner do.

Observers receive deep copies and run synchronously on the caller's goroutine. They must not
draw random numbers or mutate what they are handed; an observer reaching back into the swarm
would be a back door around reproducibility.

## Binary parameters

| Field          | Meaning                                                                                                  |
| -------------- | -------------------------------------------------------------------------------------------------------- |
| `UseBinary`    | Marks a configuration as BDA's. What `NewBinaryConfig` sets and what the variant registry dispatches on. |
| `TransferFunc` | `v1`…`v4`, `s1`…`s4`. Empty means `v3`, the paper's default. Ignored by the continuous entry points.     |

When `UseBinary` is set, validation additionally requires `LowerBound == 0` and
`UpperBound == 1`, and that `TransferFunc` names a registered entry. A continuous config
carrying a stray `TransferFunc` is not an error; a binary one on the wrong bounds is.

`OptimizeBinary` and `OptimizeBinaryContext` run BDA whether or not `UseBinary` is set.
`Optimize` ignores the field entirely, which is why `DAVariant.Run` refuses a config that has
it — see [bda.md](../algorithms/bda.md#through-the-variant-layer).

## Multi-objective configuration

`MultiObjectiveConfig` wraps a `Config` rather than duplicating it:

```go
config := dragonfly.NewMultiObjectiveConfig()
config.ObjectiveFunc = dragonfly.ZDT1 // func([]float64) []float64
config.Swarm.ProblemSize = 5
config.Swarm.LowerBound = 0
config.Swarm.UpperBound = 1
```

`Swarm` carries every shared mechanic — bounds, population, iterations, weight schedules,
boundary rule, Lévy parameters, RNG — so each means exactly what it means for a single-objective
run. `Swarm.ObjectiveFunc` is ignored. The archive parameters (`Beta`, `Gamma`, `Delta`,
`ArchiveSize`, `NGrid`) are documented, with their verification status, in
[moda.md](../algorithms/moda.md#the-archive-parameters-are-unverified).

## Factory functions

| Factory                      | `NPop` | `MaxIterations` | Other changes                                         |
| ---------------------------- | -----: | --------------: | ----------------------------------------------------- |
| `NewDefaultConfig()`         |     40 |            1000 | the baseline                                          |
| `NewHighDimensionalConfig()` |    100 |            3000 | `RadiusGrowth` 1.0                                    |
| `NewFastConvergenceConfig()` |     30 |             300 | `RadiusGrowth` 4.0, `MaxStepRatio` 0.2                |
| `NewBinaryConfig()`          |     40 |            1000 | bounds `[0,1]`, `v3`, `MaxStepRatio` 6.0, `UseBinary` |

Each returns a freshly allocated `Config`, so presets are never shared and the result can be
mutated freely. `NewPresetConfig(preset)` selects one by name — useful for a command-line flag
or a field in a larger configuration file.

## Auto-tuning

```go
config.ProblemSize = 80
dragonfly.AutoTuneConfig(config)
// npop=100 iterations=3000 radius_growth=1.0
```

A handful of coarse heuristics keyed on `ProblemSize`, not a search — a tuned configuration for
a particular objective will beat it. A nil config, or one whose `ProblemSize` is not yet
positive, is left alone.

| `ProblemSize` | `NPop` | `MaxIterations` | `RadiusGrowth` |
| ------------- | -----: | --------------: | -------------: |
| `< 10`        |     30 |             500 |      unchanged |
| `10`–`49`     |     40 |            1000 |      unchanged |
| `>= 50`       |    100 |            3000 |            1.0 |

It deliberately never touches the five swarming weights: leaving them at `WeightAuto` keeps the
paper's schedules, and pinning one is a choice a heuristic must not override.

## Validation

`ValidateConfig(config)` is the exported face of the checks `Optimize` runs, so a configuration
that passes here is accepted by `Optimize`. It reports the first problem it finds, naming the
field by its JSON name, and the checks are ordered from the required fields a caller most likely
forgot to the tuning parameters they most likely mistyped.

```go
if err := dragonfly.ValidateConfig(config); err != nil {
	log.Fatalf("bad config: %v", err)
}
```

Rejected: a nil config; a missing `ObjectiveFunc`; a non-positive `ProblemSize`, `NPop` or
`MaxIterations`; a negative `MaxWorkers`; non-finite, inverted or equal bounds; a non-finite
inertia bracket; a weight that is neither `WeightAuto` nor finite; a non-positive
`RadiusInitialDivisor` or `MaxStepRatio`; a negative `RadiusGrowth` or `LevyScale`; an
`EnemyCutoffFraction` outside `[0, 1]`; a `LevyBeta` outside the open interval `(0, 2)`; an
unrecognized `BoundaryMethod`; and any invalid convergence, constraint or binary sub-block.

## Configuration files

```go
_ = dragonfly.SaveConfig(config, "config.json") // 0600, indented
loaded, err := dragonfly.LoadConfig("config.json")
loaded.ObjectiveFunc = myObjective              // must be restored in code
```

`ObjectiveFunc`, `Rand` and the constraint function slices carry `json:"-"` because functions
and random sources cannot be serialized. Everything else — bounds, swarm size, weights,
schedules, the convergence block and the constraint policy — round-trips exactly, including a
weight left at `WeightAuto` and a weight deliberately pinned to zero, which are distinct
settings and stay distinct.

`LoadConfig` validates on the way in, against a copy carrying a stand-in objective, so a
malformed or contradictory file fails at load rather than at the start of a run.

**Do not hand-author a partial file.** Absent JSON fields decode as Go zero values, and zero is
a legitimate pinned weight. Write files with `SaveConfig`, which always emits every field.

## Configuration tips

### Quick testing

```go
config := dragonfly.NewFastConvergenceConfig()
config.MaxIterations = 100
config.NPop = 20
config.Rand = rand.New(rand.NewSource(42)) // reproducible while you iterate
```

### High-dimensional problems

```go
config := dragonfly.NewHighDimensionalConfig()
config.ProblemSize = 100
// or let the heuristics decide:
dragonfly.AutoTuneConfig(config)
```

Watch the `O(n²·d)` neighbourhood scan: at `NPop` 100 and `d` 100 a run is measured in tens of
seconds, not milliseconds.

### Expensive function evaluations

```go
config.EnableParallel = true
config.MaxWorkers = runtime.NumCPU()
```

Only above roughly a microsecond per evaluation; below that the pool costs more than it saves.

### Maximization problems

The library minimizes. Negate:

```go
func maximizeProfit(x []float64) float64 { return -calculateProfit(x) }
// ... and negate again when reporting:
fmt.Printf("best profit: %.6f\n", -result.GlobalBest.Cost)
```

### Ablating a term

```go
config.EnemyWeight = 0 // switch the enemy term off; every other weight stays automatic
```

The random stream stays aligned with an un-ablated run of the same seed, so the two are directly
comparable.

## Related documentation

- [API Quick Reference](quick-reference.md) — the task-oriented map
- [Run Lifecycle](run-lifecycle.md) — cancellation, observers, logging, export
- [Standard DA](../algorithms/standard-da.md) — what each schedule does to the search
- [BDA](../algorithms/bda.md) and [MODA](../algorithms/moda.md) — the variant-specific fields
- [Performance and Profiling](../performance.md) — the measurements quoted here
