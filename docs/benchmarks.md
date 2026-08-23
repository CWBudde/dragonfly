# Benchmark Functions Reference

`functions.go` ships 15 single-objective and 4 multi-objective test problems. The suite is
algorithm-agnostic — every function is pure maths over a `[]float64` — and was ported from the
sibling [Mayfly](https://github.com/cwbudde/mayfly) library so that results are directly
comparable between the two.

Two rules hold everywhere:

- **Everything is a minimization problem.** For a maximization problem of your own, negate.
- **The single-objective signature is always `func([]float64) float64`**, including for the
  binary variant, where the input is 0/1-valued. That is what lets BDA reuse the whole suite.

## Function categories

**Classic (5)** — Sphere, Rastrigin, Rosenbrock, Ackley, Griewank

**CEC-style (10)** — Schwefel, Levy, Zakharov, Michalewicz, DixonPrice, BentCigar, Discus,
Weierstrass, HappyCat, ExpandedSchafferF6

**Multi-objective (4)** — ZDT1, ZDT2, ZDT3, SchafferN1

## Quick reference table

`Characteristics` are the hand-classified values `BenchmarkCharacteristics(name)` returns; the
selector reads this table rather than sampling, because these landscapes are known from the
literature and a sampled estimate of a known answer is strictly worse than the known answer.

| Function             | Typical bounds    | Optimum               | Modality          | Landscape     |
| -------------------- | ----------------- | --------------------- | ----------------- | ------------- |
| `Sphere`             | [-100, 100]       | `f(0,…,0) = 0`        | unimodal          | smooth        |
| `Zakharov`           | [-10, 10]         | `f(0,…,0) = 0`        | unimodal          | smooth        |
| `Rosenbrock`         | [-5, 10]          | `f(1,…,1) = 0`        | unimodal          | narrow valley |
| `BentCigar`          | [-100, 100]       | `f(0,…,0) = 0`        | unimodal          | narrow valley |
| `Discus`             | [-100, 100]       | `f(0,…,0) = 0`        | unimodal          | narrow valley |
| `DixonPrice`         | [-10, 10]         | `f(x*) = 0`           | unimodal          | narrow valley |
| `Ackley`             | [-32.768, 32.768] | `f(0,…,0) = 0`        | multimodal        | rugged        |
| `HappyCat`           | [-2, 2]           | `f(-1,…,-1) = 0`      | multimodal        | rugged        |
| `Rastrigin`          | [-5.12, 5.12]     | `f(0,…,0) = 0`        | highly multimodal | rugged        |
| `Griewank`           | [-600, 600]       | `f(0,…,0) = 0`        | highly multimodal | rugged        |
| `Weierstrass`        | [-0.5, 0.5]       | `f(0,…,0) = 0`        | highly multimodal | rugged        |
| `ExpandedSchafferF6` | [-100, 100]       | `f(0,…,0) = 0`        | highly multimodal | rugged        |
| `Levy`               | [-10, 10]         | `f(1,…,1) = 0`        | highly multimodal | rugged        |
| `Schwefel`           | [-500, 500]       | `f(420.9687,…) = 0`   | highly multimodal | deceptive     |
| `Michalewicz`        | [0, π]            | ≈ `-9.66` at `d = 10` | highly multimodal | deceptive     |

`BenchmarkNames()` lists every name in the table; `BenchmarkCharacteristics(name)` returns the
characteristics and whether the name was known.

## Function details

### Sphere

```
f(x) = Σ xᵢ²
```

A smooth, convex, unimodal bowl — the simplest possible test. Every optimizer solves it; the
question is how fast, and whether a change broke something basic. Global minimum `f(0,…,0) = 0`.

### Rastrigin

```
f(x) = 10n + Σ (xᵢ² - 10·cos(2π·xᵢ))
```

A parabola with a cosine lattice on top: many local minima at regular spacing, all of them
shallow relative to the global basin. The standard test of whether an optimizer can escape a
regular structure. Global minimum `f(0,…,0) = 0`, typical bounds `[-5.12, 5.12]`.

### Rosenbrock

```
f(x) = Σᵢ₌₁ⁿ⁻¹ [100·(xᵢ₊₁ - xᵢ²)² + (1 - xᵢ)²]
```

The "banana" function: a long, narrow, curved valley. Finding the valley is easy; following it
to the optimum is not, and a step clamp sized for the whole box will jump straight over it.
Global minimum `f(1,…,1) = 0`, typical bounds `[-5, 10]`.

### Ackley

```
f(x) = -20·exp(-0.2·√(Σxᵢ²/n)) - exp(Σcos(2π·xᵢ)/n) + 20 + e
```

A nearly flat outer region with a deep, narrow central basin, and a fine ripple everywhere. The
flat region carries almost no gradient information, so an optimizer that relies on it stalls
far from the optimum. Global minimum `f(0,…,0) = 0`, typical bounds `[-32.768, 32.768]`.

### Griewank

```
f(x) = Σxᵢ²/4000 - Π cos(xᵢ/√i) + 1
```

A product term creates many regularly spaced local minima on top of a very shallow bowl. It
gets _easier_ in higher dimensions, which makes it a useful counterweight to the rest of the
suite. Global minimum `f(0,…,0) = 0`, typical bounds `[-600, 600]`.

### Schwefel

```
f(x) = 418.9829·n - Σ xᵢ·sin(√|xᵢ|)
```

Deceptive: the global minimum sits near a corner of the box at `xᵢ ≈ 420.9687`, far from the
second-best local minima, and the gradient near the centre leads away from it. Global minimum
`f(420.9687,…) = 0`, typical bounds `[-500, 500]`. This is the function that separates
optimizers with a real escape mechanism from ones that only look like they have one.

### Levy

```
wᵢ = 1 + (xᵢ - 1)/4
f(x) = sin²(π·w₁) + Σᵢ₌₁ⁿ⁻¹ (wᵢ-1)²·[1 + 10·sin²(π·wᵢ + 1)] + (wₙ-1)²·[1 + sin²(2π·wₙ)]
```

Multimodal with a strongly oscillating surface. Global minimum `f(1,…,1) = 0`, typical bounds
`[-10, 10]`. The `+1` inside `sin(π·wᵢ + 1)` is the standard definition, not a mistranscription
— it has been checked.

### Zakharov

```
f(x) = Σxᵢ² + (Σ0.5·i·xᵢ)² + (Σ0.5·i·xᵢ)⁴
```

Unimodal with no local minima besides the global one, but the quartic term makes it
ill-scaled: the cost changes by orders of magnitude across the box. Global minimum
`f(0,…,0) = 0`, typical bounds `[-5, 10]` or `[-10, 10]`.

### Michalewicz

```
f(x) = -Σ sin(xᵢ)·sin²ᵐ(i·xᵢ²/π),   m = 10
```

Steep valleys and ridges, with `m` controlling how steep. Large `m` makes the search almost
gradient-free: the function is flat nearly everywhere and drops sharply in narrow channels. The
optimum depends on dimension — about `-9.66` at `d = 10`. Typical bounds `[0, π]`.

### DixonPrice

```
f(x) = (x₁ - 1)² + Σᵢ₌₂ⁿ i·(2xᵢ² - xᵢ₋₁)²
```

A valley-shaped, unimodal landscape whose optimum has a different value in every dimension,
which catches an implementation that assumes a symmetric optimum. Typical bounds `[-10, 10]`.

### BentCigar

```
f(x) = x₁² + 10⁶·Σᵢ₌₂ⁿ xᵢ²
```

Unimodal and severely ill-conditioned: a condition number of a million between the first
direction and all the others. Global minimum `f(0,…,0) = 0`, typical bounds `[-100, 100]`.
Absolute costs here are enormous by construction — `10⁶ · x²` — so read them relative to the
box, not as a distance.

### Discus

```
f(x) = 10⁶·x₁² + Σᵢ₌₂ⁿ xᵢ²
```

BentCigar's mirror image: ill-conditioned along a single direction rather than all but one.
Global minimum `f(0,…,0) = 0`, typical bounds `[-100, 100]`.

### Weierstrass

```
f(x) = Σᵢ Σₖ₌₀²⁰ [0.5ᵏ·cos(2π·3ᵏ·(xᵢ + 0.5))] - n·Σₖ₌₀²⁰ [0.5ᵏ·cos(π·3ᵏ)]
```

Continuous everywhere and differentiable nowhere — structure at every scale, which is what
defeats any method that assumes a smooth local model. It is also by far the most expensive
function in the suite: 21 cosines per dimension. Global minimum `f(0,…,0) = 0`, typical bounds
`[-0.5, 0.5]`.

### HappyCat

```
f(x) = |Σxᵢ² - n|^0.25 + (0.5·Σxᵢ² + Σxᵢ)/n + 0.5
```

Multimodal with a curved, thin optimal region on a sphere of radius `√n`. Global minimum
`f(-1,…,-1) = 0`, typical bounds `[-2, 2]`.

### ExpandedSchafferF6

```
g(x, y) = 0.5 + (sin²(√(x²+y²)) - 0.5) / (1 + 0.001·(x²+y²))²
f(x)    = Σᵢ₌₁ⁿ⁻¹ g(xᵢ, xᵢ₊₁) + g(xₙ, x₁)
```

Concentric ripples around the origin that decay very slowly, so the local structure persists
across the whole box. The wrap-around pair `g(xₙ, x₁)` is part of the CEC definition of the
expanded function, not a bug. Global minimum `f(0,…,0) = 0`, typical bounds `[-100, 100]`.

## Multi-objective problems

All four have signature `func([]float64) []float64` and return two objectives, both minimized.

### ZDT1, ZDT2, ZDT3

They share a structure:

```
f1 = x₁
g  = 1 + 9·Σᵢ₌₂ⁿ xᵢ / (n-1)      — exactly 1 on the Pareto front
```

and differ only in `f2`:

| Problem | `f2`                                   | Front shape              |
| ------- | -------------------------------------- | ------------------------ |
| ZDT1    | `g·(1 - √(f1/g))`                      | convex and continuous    |
| ZDT2    | `g·(1 - (f1/g)²)`                      | concave                  |
| ZDT3    | `g·(1 - √(f1/g) - (f1/g)·sin(10π·f1))` | five disconnected pieces |

The front is reached when every `xᵢ` for `i ≥ 2` is zero, so `g = 1`. Typical bounds `[0, 1]` in
every dimension, 30 dimensions in the original suite.

ZDT3 is the one an archive that collapses onto a single region fails first: its five
disconnected pieces cannot be covered by a front approximation that has clustered.

The 30 dimensions of the original suite are beyond what this implementation recovers. In the
v0.2 paper-default five-dimensional, 15-seed gate, ZDT1 passed 14/15 seeds and ZDT3 passed
12/15. At 30 dimensions, none of the paper, MATLAB or named legacy profile/problem pairs met
the strict recovery bar in four of five seeds. The archive stays valid and non-dominated while
sitting well off the front. The metrics and raw evidence are in
[MODA's performance section](algorithms/moda.md#performance).

### SchafferN1

```
f1 = x₁²,   f2 = (x₁ - 2)²
```

The classic one-variable, two-objective problem: two shifted parabolas pulling in opposite
directions, with a convex front that is the image of `x ∈ [0, 2]`. Only the first component of
the position is read. Typical bounds `[-10, 10]`.

Because the front is one-dimensional and dense, the archive fills immediately and stays full,
which makes SchafferN1 a good stress test of archive maintenance rather than of search — see
[performance.md](performance.md).

## Measured results

The broad table below is the v0.1 trajectory retained as historical context: Standard DA, 10
dimensions, 500 iterations, `NPop` 40, default configuration, seeds 1000–1014,
median/mean/best/worst of 15 runs. The corrected v0.2 regression measurements follow it.
Every function's global minimum is 0 except Michalewicz.

| Function             | Bounds            |    Median |      Mean |      Best |     Worst |
| -------------------- | ----------------- | --------: | --------: | --------: | --------: |
| `Sphere`             | [-100, 100]       |     87.17 |     87.33 |     15.11 |     195.9 |
| `Rastrigin`          | [-5.12, 5.12]     |     27.88 |     29.78 |     17.81 |     48.65 |
| `Rosenbrock`         | [-5, 10]          |     152.4 |     175.7 |     30.62 |     454.8 |
| `Ackley`             | [-32.768, 32.768] |     5.837 |     5.861 |     3.263 |     8.725 |
| `Griewank`           | [-600, 600]       |     1.801 |     1.983 |     1.034 |     3.651 |
| `Schwefel`           | [-500, 500]       |      1590 |      1591 |     936.4 |      2034 |
| `Levy`               | [-10, 10]         |    0.8646 |     1.095 |    0.1709 |      2.96 |
| `Zakharov`           | [-10, 10]         |     6.842 |     7.556 |      1.73 |     15.99 |
| `Michalewicz`        | [0, π]            |    -5.859 |    -5.961 |    -7.201 |    -4.979 |
| `DixonPrice`         | [-10, 10]         |        16 |     18.62 |     1.675 |     44.02 |
| `BentCigar`          | [-100, 100]       | 6.564e+07 | 7.816e+07 | 5.045e+06 | 2.821e+08 |
| `Discus`             | [-100, 100]       |      5719 |      5479 |     435.8 |      9249 |
| `Weierstrass`        | [-0.5, 0.5]       |     3.883 |     4.023 |     2.693 |      5.85 |
| `HappyCat`           | [-2, 2]           |    0.2386 |    0.3023 |   0.08158 |    0.6747 |
| `ExpandedSchafferF6` | [-100, 100]       |     2.888 |     2.713 |     1.965 |     3.348 |

**Read these as a baseline, not as a quality claim.** DA's shared convergence factor reaches
zero at the halfway point of a run, after which only the food term and inertia still move a
dragonfly, so the paper's algorithm stops closing the distance long before the iteration cap.
These are the numbers a faithful port produces. See
[standard-da.md](algorithms/standard-da.md#adaptive-weight-schedules) for the mechanism.

Two reading notes:

- **Costs scale with the box.** Almost every schedule is written in units of `(ub-lb)`, so the
  same Sphere run over `[-10, 10]` lands near `1` rather than near `87`. Compare like with like.
- **Absolute magnitudes are not comparable across functions.** BentCigar's `10⁶` factor and
  Schwefel's `418.98·n` offset put them on entirely different scales.

## Regression baselines

`regression_test.go` watches six of these under `RegressionBaseline` entries. Each pairs a
**reference mean** — a round number of the right order of magnitude — with a **tolerance
factor**, and asserts only the product, plus a `SuccessThreshold` on the fraction of individual
runs that land under it.

| Baseline            | Dimensions | Iterations | `NPop` | Reference mean | Tolerance | Success threshold |
| ------------------- | ---------: | ---------: | -----: | -------------: | --------: | ----------------: |
| `DA_Sphere_10D`     |         10 |        500 |     40 |            100 |        3x |               80% |
| `DA_Rastrigin_10D`  |         10 |        500 |     40 |             30 |        3x |               80% |
| `DA_Ackley_10D`     |         10 |        500 |     40 |              6 |        3x |               80% |
| `DA_Rosenbrock_10D` |         10 |        500 |     40 |            200 |        3x |               80% |
| `DA_Griewank_10D`   |         10 |        500 |     40 |              2 |        3x |               80% |
| `BDA_OneMax_30bit`  |         30 |        300 |     30 |              1 |        3x |               80% |

The corrected v0.2 suite re-ran those exact 15 deterministic seeds without changing a
threshold. All six baselines passed 15/15 runs:

| Baseline            | Observed mean | Observed median | Pass rate |
| ------------------- | ------------: | --------------: | --------: |
| `DA_Sphere_10D`     |       94.5466 |         76.8328 |      100% |
| `DA_Rastrigin_10D`  |       25.0785 |         22.5307 |      100% |
| `DA_Ackley_10D`     |        5.0877 |          5.3935 |      100% |
| `DA_Rosenbrock_10D` |       81.1692 |         83.7892 |      100% |
| `DA_Griewank_10D`   |        1.4674 |          1.3960 |      100% |
| `BDA_OneMax_30bit`  |        0.0000 |          0.0000 |      100% |

A stochastic optimizer has no golden output: a change that merely shifts the random stream —
one extra draw, two draws reordered — changes every measured number without changing the
algorithm. Pinning observed values would produce a suite that fails on refactors and passes on
real regressions.

**Never paste a measured number into a baseline to make a test pass.** A run that exceeds
`ReferenceMean × Tolerance` is a finding: either the change degraded the algorithm, or the
reference was wrong to begin with and moving it needs its own justification.

## Using the suite

```go
config := dragonfly.NewDefaultConfig()
config.ObjectiveFunc = dragonfly.Rastrigin
config.ProblemSize = 10
config.LowerBound = -5.12
config.UpperBound = 5.12
config.MaxIterations = 500

result, err := dragonfly.Optimize(config)
```

Progressive testing — start easy and work up:

1. **Sphere** — does it converge at all?
2. **Rosenbrock** — can it follow a valley?
3. **Rastrigin** — can it escape a regular lattice?
4. **Ackley** — can it cross a flat region?
5. **Schwefel** — can it resist a deceptive gradient?

For a statistically defensible comparison, run each through
[`ComparisonRunner`](api/comparison-framework.md) with 30 paired seeds rather than eyeballing
single runs.

## Edge-case policy

Every single-objective benchmark now follows the same empty-input convention: `f([]) == 0`.
`Levy(nil)` no longer panics, and `Ackley(nil)` and `HappyCat(nil)` no longer return `NaN`.
The convention is asserted for all fifteen functions and matches the sibling Mayfly library.

Two things that look like defects and are not, checked and cleared: `Levy`'s
`sin(π·wᵢ + 1)` is the standard definition, and `ExpandedSchafferF6`'s wrap-around pair
`g(xₙ, x₁)` is part of the CEC definition of the expanded function.

## Adding a function

Follow the doc-comment convention: one sentence naming the function and characterising its
landscape, then the global minimum, then the bounds if they are not generic and symmetric.

```go
// Rastrigin is the Rastrigin benchmark function: highly multimodal with a regular lattice of local minima.
// Global minimum is at f(0, ..., 0) = 0.
func Rastrigin(x []float64) float64 {
	// ...
}
```

`just new-benchmark <Name>` appends a skeleton in this shape. Then:

- add a case to `functions_test.go` asserting the value at the known optimum
- add an entry to `benchmarkCharacteristics` in `selector.go` if the landscape is known
- add the name to `goconst.ignore-string-values` in `.golangci.toml` if it appears in a switch
  or a table

## Related documentation

- [Standard DA](algorithms/standard-da.md) — why the measured numbers look as they do
- [MODA](algorithms/moda.md) — the ZDT and SchafferN1 results
- [Comparison Framework](api/comparison-framework.md) — comparing variants on these problems
- [Performance and Profiling](performance.md) — how long each function costs to evaluate
- [Configuration Guide](api/configuration.md) — bounds, and why the box width matters
