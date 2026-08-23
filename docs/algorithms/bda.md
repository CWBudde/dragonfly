# BDA — Binary Dragonfly Algorithm

## Research Reference

**Mirjalili, S. (2016). Dragonfly algorithm: a new meta-heuristic optimization technique for
solving single-objective, discrete, and multi-objective problems. _Neural Computing and
Applications_, 27(4), 1053–1073.** — §4 introduces BDA.

<https://doi.org/10.1007/s00521-015-1920-1>

The V-shaped and S-shaped transfer-function families BDA draws on are the standard ones from
the binary-PSO literature that the paper builds on.

Reference implementation: the author's `BDA.m`. It is **not available to this repository**, so
one BDA constant here is this implementation's judgement rather than a quoted value — see
[The step clamp](#the-step-clamp-is-not-a-paper-constant) below.

## Overview

BDA differs from [standard DA](standard-da.md) in the position update and nowhere else.

`ΔX` is built by the same five primitives, the same two-branch gating and the same clamp the
continuous variant uses — `binary.go` calls straight into the shared step builders in
`parallel_phases.go`. The continuous position update is then discarded, and each bit is flipped
with a probability read off the step by a **transfer function**:

```
T(Δx_j) = |Δx_j / sqrt(Δx_j² + 1)|              the paper's default, v3
x_j <- ¬x_j   if rand < T(Δx_j)   else   x_j
```

So `ΔX` no longer means "how far to move". It means "how unsettled is this bit". A large step
magnitude is a high flip probability; a step near zero leaves the bit alone.

The objective keeps the ordinary `func([]float64) float64` signature and is handed a vector
whose components are exactly `0` or `1`. That is deliberate: the benchmark suite, the
constraint machinery, the observers and the comparison framework are reused unchanged.
`BinaryPositionsValid` is exported so a caller inspecting a `Result` or a population snapshot
can assert the invariant too.

### Transfer functions

Eight are registered. The V-shaped family is symmetric about zero — it reads the magnitude of a
step and flips regardless of sign. The S-shaped family is monotone increasing — a positive step
pushes a bit towards one and a negative step towards zero, in the sense that the flip
probability crosses one half at zero.

| Name | Form                          | `T(0)` | `T(1)` | `T(6)` |
| ---- | ----------------------------- | -----: | -----: | -----: |
| `v1` | \|erf(√π/2 · Δx)\|            |  0.000 |  0.790 |  1.000 |
| `v2` | \|tanh(Δx)\|                  |  0.000 |  0.762 |  1.000 |
| `v3` | \|Δx / √(Δx²+1)\| _(default)_ |  0.000 |  0.707 |  0.986 |
| `v4` | \|(2/π)·arctan((π/2)·Δx)\|    |  0.000 |  0.639 |  0.933 |
| `s1` | 1/(1+e^(-2Δx))                |  0.500 |  0.881 |  1.000 |
| `s2` | 1/(1+e^(-Δx))                 |  0.500 |  0.731 |  0.998 |
| `s3` | 1/(1+e^(-Δx/2))               |  0.500 |  0.622 |  0.953 |
| `s4` | 1/(1+e^(-Δx/3))               |  0.500 |  0.583 |  0.881 |

`LookupTransferFunction(name)` returns the function itself; `TransferFunctionNames()` returns
every registered name in this stable order. An unknown name is an error, never a silent
fallback to the default — a misspelling in a JSON configuration would otherwise run a different
algorithm than the one you wrote.

### The step clamp is not a paper constant

`NewBinaryConfig` sets `MaxStepRatio = 6.0`, which on the unit search box makes `ΔX_max = 6`.
This is **this implementation's choice, not a value read off `BDA.m`**. The reasoning is
written out in `config.go`: the transfer functions saturate by `|Δx| ≈ 6`, so clamping there is
what makes the whole range of flip probabilities reachable. The continuous default of `0.1`
would cap every flip probability at about a tenth and freeze the swarm. Treat `6.0` as a
working default until someone checks it against the reference source.

### What binary mode ignores

Two continuous-only mechanisms are deliberately switched off, both explained in
`OptimizeBinaryContext`:

- **`Config.BoundaryMethod`.** A 0/1 vector cannot leave `[0, 1]`, so there is nothing to
  repair. Applying the wrap rule anyway would be worse than useless: its `Δx` reset would
  overwrite the very step the next bit-flip decision is made from. The field is left alone
  rather than validated away, so one `Config` can be handed to both entry points.
- **`Config.UseLevyWalk` and the Lévy branch.** A Lévy walk is a multiplicative displacement of
  a real-valued position and has no binary counterpart. The food-out-of-range branch is
  therefore the local-swarming step for _every_ dragonfly, isolated or not. `swarm.go`'s
  documented empty-neighbourhood fallbacks reduce that step, for an isolated dragonfly, to a
  decaying carry of `ΔX` — which the transfer function reads as a diminishing random flip
  probability. That is the exploration role the Lévy walk plays in the continuous variant.

### What binary mode enforces

`validateBinaryConfig` rejects any bounds other than `[0, 1]`. Every schedule that scales with
`(ub-lb)` is written for that box, so a different one would silently rescale the step clamp and
the neighbourhood radius. A seeded initial population must be 0/1-valued as well; rounding it
silently would hand you a different starting swarm than the one you wrote.

Initial positions are fair coin flips, not uniform draws from `[0, 1]` — the position space is
the corners of the unit cube. Initial steps _are_ drawn uniformly from `[0, 1]`, so the first
iteration's flip probabilities are `T` of a real value rather than `T(0)`.

## Key Innovations

1. **The step becomes a probability, not a displacement.** This is the whole variant. It also
   means the step clamp is now the parameter that controls how erratic the search is.
2. **Nothing else changes.** BDA reuses DA's neighbourhood scan, its five primitives, its
   two-branch gating and its weight schedules. Every fidelity property proved for DA holds
   here; `binary.go` contains no second copy of the step rules.
3. **Two transfer-function families with genuinely different behaviour.** The V family is
   sign-blind and flips on magnitude alone; the S family biases towards one or zero. On
   problems where "how many bits should be set" is not known in advance, the V family's
   symmetry matters — see the measurements below.
4. **Determinism survives the extra draw.** `flipBits` takes a uniform draw for _every_
   component whether or not the flip happens, so the random stream does not depend on the
   outcome of earlier flips. That, plus keeping every draw on the calling goroutine, is what
   keeps a seeded binary run bit-identical with `EnableParallel` on or off.

## Usage Examples

### OneMax

```go
// oneMax counts the zero bits, so an all-ones string costs zero.
func oneMax(bits []float64) float64 {
	zeros := 0.0
	for _, bit := range bits {
		zeros += 1 - bit
	}

	return zeros
}

config := dragonfly.NewBinaryConfig() // bounds are already [0, 1]
config.ObjectiveFunc = oneMax
config.ProblemSize = 30
config.MaxIterations = 300
config.NPop = 30
config.TransferFunc = dragonfly.TransferV3 // the paper's default
config.Rand = rand.New(rand.NewSource(2000))

result, err := dragonfly.OptimizeBinary(config)
if err != nil {
	log.Fatal(err)
}

fmt.Printf("zero bits left: %.0f\n", result.GlobalBest.Cost)
fmt.Printf("still binary:   %v\n",
	dragonfly.BinaryPositionsValid(result.GlobalBest.Position))
```

Output: `zero bits left: 0`, `still binary: true`.

### Feature selection

The worked example in [`examples/feature_selection`](../../examples/feature_selection) solves a
wrapper-style problem: a 12-bit mask over candidate features, scored by a linear model's
training error plus a per-feature price, so a feature has to earn its place.

```
cost(mask) = MSE(mask) + lambda * |mask| / d
```

The per-feature price is what stops the search from simply selecting everything, and it is the
usual shape of a feature-selection objective for any wrapper method.

### Through the variant layer

```go
result, err := dragonfly.NewBuilder("bda").
	ForProblem(oneMax, 30, 0, 1). // bounds are ignored; BDA fixes them
	WithIterations(500).
	Optimize()
```

`DAVariant.Run` refuses a configuration with `UseBinary` set
(`ErrBinaryConfigOnContinuousVariant`), because nothing in `dragonfly.go` dispatches on that
field: handing a binary config to the continuous entry point would run the continuous algorithm
on a swarm confined to `[0, 1]` and return a real-valued "solution" that is not a bit string.
That is a silently wrong answer, so the variant layer — the one place that knows which
algorithm you asked for — rejects it.

## Parameters

BDA takes the same `Config` as DA. What differs:

| Field            | BDA value      | Notes                                                      |
| ---------------- | -------------- | ---------------------------------------------------------- |
| `LowerBound`     | `0` — enforced | rejected if anything else                                  |
| `UpperBound`     | `1` — enforced | rejected if anything else                                  |
| `MaxStepRatio`   | `6.0`          | this implementation's choice; see above                    |
| `TransferFunc`   | `""` → `v3`    | one of `v1`…`v4`, `s1`…`s4`                                |
| `UseBinary`      | `true`         | what `NewBinaryConfig` sets and the registry dispatches on |
| `BoundaryMethod` | ignored        | a bit cannot leave the box                                 |
| `UseLevyWalk`    | ignored        | no binary counterpart                                      |

`OptimizeBinary` and `OptimizeBinaryContext` run BDA whether or not `UseBinary` is set; the
flag exists for the variant registry and for validation.

Everything else — `NPop`, `MaxIterations`, the five weights, `RadiusInitialDivisor`,
`RadiusGrowth`, the inertia bracket, constraints, convergence, parallelism, RNG — means exactly
what it means for DA.

## Benefits

- **No encoding layer.** The objective signature is unchanged, so a binary problem reuses the
  same constraint handling, the same observers, the same exporters and the same comparison
  runner as a continuous one.
- **No decoding ambiguity.** Positions are exactly `0` or `1` from initialization onwards.
  There is no rounding step, and therefore no question of what the reported solution means.
- **Zero extra overhead.** `EstimatedOverhead()` is `1.0`: the bit-flip update replaces the
  continuous position update rather than adding to it.
- **All eight transfer functions in one registry**, so trying a different one is a one-line
  change and a `ComparisonRunner` sweep away.

## Performance

OneMax, 30 bits, 300 iterations, `NPop` 30, seeds 2000–2014, median/mean/worst of 15 runs.
Lower is better; zero means every bit was set.

| Transfer | Median | Mean | Worst |
| -------- | -----: | ---: | ----: |
| `v1`     |      0 | 0.00 |     0 |
| `v2`     |      0 | 0.00 |     0 |
| `v3`     |      0 | 0.00 |     0 |
| `v4`     |      0 | 0.00 |     0 |
| `s1`     |      1 | 1.13 |     2 |
| `s2`     |      2 | 2.13 |     3 |
| `s3`     |      4 | 3.80 |     4 |
| `s4`     |      4 | 4.07 |     5 |

The V family solves this problem outright; the S family does not, and gets steadily worse as
its slope flattens from `s1` to `s4`. The reason is structural rather than incidental: an
S-shaped function has `T(0) = 0.5`, so a settled bit whose step has decayed to zero still flips
half the time. A V-shaped function has `T(0) = 0`, and a settled bit stays settled. Unless you
have a specific reason to want direction-sensitive flipping, stay on the V family.

Cost, on Linux/amd64 with Go 1.26.0 on an AMD Ryzen 5 4600H: 30 bits, 50 iterations, `NPop` 30
costs 17.3 ms (`v3`) to 21.9 ms (`s4`) per run. The spread across the eight is about 25%, all
of it one transcendental call per component per iteration — `v3` (a square root) is the
cheapest and `s1`/`s4` (an exponential) the dearest. `BenchmarkOptimizeBinaryBaseline`, the
BDA profiling anchor, is 45.8 ms per run at 30 bits over 100 iterations with `NPop` 40.

## When to Use

**Use BDA when:**

- decision variables are genuinely binary: include or exclude, on or off, assign or do not
- the problem is subset selection — feature selection, knapsack, sensor placement, unit
  commitment
- a continuous relaxation followed by rounding would produce an answer you cannot defend

**Do not use BDA when:**

- the variables are continuous. `BDAVariant.ApplicableTo` scores a continuous problem at `0.05`
  — close to unusable, not merely a poor fit — because reaching one would need an encoding the
  library does not supply.
- the variables are integer-valued over a range wider than `{0, 1}`. Nothing here does binary
  encoding of integers for you; you would have to write the encoding into your objective.
- there are several objectives. BDA is single-objective; MODA is continuous. The combination is
  not implemented.

## Parameter Tuning Guide

1. **Transfer function first, and start with the V family.** `v3` is the paper's default and a
   good one. `v2` (tanh) saturates faster and makes the search more decisive; `v4` (arctan)
   saturates slowest and keeps more churn late in the run. Switch to the S family only when the
   sign of the step carries meaning for your encoding.
2. **`MaxStepRatio` second, and treat it as the exploration dial.** It is the one parameter
   whose default here is not from the paper. Lower it towards `1.0` to make the search calmer —
   the flip probability then tops out around `T(1) ≈ 0.71` for `v3` instead of `0.99`. Raise it
   above `6` and you gain almost nothing, because the transfer functions are already saturated.
3. **`NPop` and `MaxIterations` third.** Same trade-off as DA, and the same `O(n²·d)` neighbour
   scan.
4. **`RadiusGrowth`.** On a binary problem the "distance" between two dragonflies is a Hamming
   distance in disguise, and the box is only one unit wide, so the default radius schedule
   already reaches the whole space early. Lower it if the swarm converges to one bit string too
   fast.
5. **Weights.** As with DA, pin one at a time and leave the rest on their schedules. Pinning
   `EnemyWeight = 0` is the cheapest ablation available.

## Compared with the Other Variants

| Aspect          | BDA                          | Standard DA            | MODA                   |
| --------------- | ---------------------------- | ---------------------- | ---------------------- |
| Position space  | corners of the unit cube     | a real box             | a real box             |
| Position update | per-bit flip through `T(Δx)` | `X += ΔX`              | `X += ΔX`              |
| Step meaning    | flip probability             | displacement           | displacement           |
| Bounds          | fixed at `[0, 1]`            | caller's               | caller's               |
| Boundary rule   | not applicable               | wrap / clamp / reflect | wrap / clamp / reflect |
| Lévy walk       | none                         | yes                    | yes                    |
| Result          | one bit string               | one position           | a Pareto archive       |
| Overhead        | 1.0x                         | 1.0x (baseline)        | about 1.2x             |

## Related Documentation

- [Standard DA](standard-da.md) — the step builder BDA shares, and the primitives behind it
- [MODA](moda.md) — the multi-objective variant
- [Configuration Guide](../api/configuration.md) — every field, including the binary block
- [Run Lifecycle](../api/run-lifecycle.md) — observers, logging, export
- [`examples/feature_selection`](../../examples/feature_selection) — a worked BDA problem
- [Research References](../research.md) — citations and BibTeX
