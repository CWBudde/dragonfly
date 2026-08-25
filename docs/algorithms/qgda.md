# Quantum-behaved and Gaussian Mutational DA (QGDA)

## Research reference

Yu, C., Cai, Z., Ye, X., Wang, M., Zhao, X., Liang, G., Chen, H. & Li, C. (2020).
Quantum-like mutation-induced dragonfly-inspired optimization approach. _Mathematics and
Computers in Simulation_, 178, 259–289.
<https://doi.org/10.1016/j.matcom.2020.06.012>

## Overview

QGDA adds two greedy improvement stages after each continuous DA movement:

1. Gaussian mutation forms `X' = X*(1+k*N(0,1))`, with the paper's `k=1` default.
2. Quantum rotation treats each coordinate and the current global-best coordinate as a
   two-state vector. The sign of the rotation follows the paper's fitness/product lookup table,
   and the evaluated rotated candidate is retained greedily.

```
[alpha']   [ cos(theta) -sin(theta) ] [alpha]
[ beta'] = [ sin(theta)  cos(theta) ] [ beta]
```

The paper's rotation magnitude is `0.005*pi`. `QuantumRotate` exposes the norm-preserving
operator independently.

## Usage

```go
config := dragonfly.NewQuantumConfig()
config.ObjectiveFunc = dragonfly.Rastrigin
config.ProblemSize = 10
config.LowerBound = -5.12
config.UpperBound = 5.12

result, err := dragonfly.OptimizeQuantum(config)
```

Tune the operators with `GaussianMutationWeight` and `QuantumRotationAngle`. Both must be
positive and finite. A full run makes `NPop*(3*MaxIterations+1)` objective calls, so this
variant is intended for cheap objectives. Parallel and sequential seeded runs are
bit-identical. QGDA supports `FidelityPaper` only.

## When to use

Use QGDA on continuous multimodal landscapes where extra mutation and local rotation are worth
two additional evaluated populations per iteration. Prefer DA or MHDA for expensive
objectives.

See also [standard DA](standard-da.md), [MHDA](mhda.md), and the
[configuration guide](../api/configuration.md).
