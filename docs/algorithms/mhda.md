# Memory-based Hybrid Dragonfly Algorithm (MHDA)

## Research reference

Sree Ranjini, K. S. & Murugan, S. (2017). Memory based Hybrid Dragonfly Algorithm for
numerical optimization problems. _Expert Systems with Applications_, 83, 63–78.
<https://doi.org/10.1016/j.eswa.2017.04.033>

## Overview

Standard DA remembers only the swarm-wide food and enemy. MHDA adds one personal-best record
per dragonfly and follows every DA movement with a PSO movement initialized from those saved
positions. DA supplies exploration; the PSO stage exploits the promising region retained by
memory.

For dragonfly `i`, the second movement is:

```
V_i = w*V_i + c1*r1*(pbest_i-X_i) + c2*r2*(gbest-X_i)
X_i = X_i + V_i
```

`c1` and `c2` are `PSOCognitiveWeight` and `PSOSocialWeight`, both `2` by default. The random
factors are drawn per coordinate. A PSO candidate replaces its DA parent only when it is better
under the configured constraint policy.

## Usage

```go
config := dragonfly.NewMemoryHybridConfig()
config.ObjectiveFunc = dragonfly.Sphere
config.ProblemSize = 10
config.LowerBound = -100
config.UpperBound = 100

result, err := dragonfly.OptimizeMemoryHybrid(config)
```

A full run makes `NPop*(2*MaxIterations+1)` objective calls. Parallel evaluation is
bit-identical to sequential evaluation for the same seed. MHDA supports `FidelityPaper` only;
combining it with the separate `DA.m` compatibility lifecycle is rejected.

## When to use

Use MHDA when standard DA explores promising regions but stalls before exploiting them. Avoid
it when each objective call is expensive: it evaluates two populations per iteration.

See also [standard DA](standard-da.md), [QGDA](qgda.md), and the
[configuration guide](../api/configuration.md).
