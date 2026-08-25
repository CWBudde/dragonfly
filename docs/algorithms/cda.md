# Chaotic Dragonfly Algorithm (CDA)

## Research reference

Sayed, G. I., Tharwat, A. & Hassanien, A. E. (2019). Chaotic dragonfly algorithm: an improved
metaheuristic algorithm for feature selection. _Applied Intelligence_, 49, 188–205.
<https://doi.org/10.1007/s10489-018-1261-8>

## Overview

CDA is a continuous DA variant developed for feature selection. At iteration `i`, it computes
one chaotic value `B(i)` and uses it as every coefficient in the standard step equation:

```
DeltaX(t+1) = B(i)*(S + A + C + F + E + DeltaX(t))
```

This is the paper's modified equation, rather than five independent chaotic draws or a chaotic
population initializer. Explicitly pinned S/A/C/F/E fields still override their coefficient;
the CDA inertia coefficient is always `B(i)`.

Ten maps are available: `chebyshev`, `circle`, `gauss`, `iterative`, `logistic`, `piecewise`,
`sine`, `singer`, `sinusoidal`, and `tent`. `NewChaoticConfig` selects the paper's
best-performing Gauss profile at initial condition `0.7`.

## Usage

```go
config := dragonfly.NewChaoticConfig()
config.ObjectiveFunc = dragonfly.Rastrigin
config.ProblemSize = 10
config.LowerBound = -5.12
config.UpperBound = 5.12

result, err := dragonfly.OptimizeChaotic(config)
```

Choose a different recurrence with `config.ChaosMap` and `config.ChaosSeed`. Use
`ChaosMapNames` to list the stable registry and `ChaoticMapValue` to inspect a recurrence.
The normal `Seed` controls swarm initialization and boundary/Levy draws; `ChaosSeed` controls
only the chaotic coefficient stream.

CDA has DA's `NPop*(MaxIterations+1)` evaluation count and supports `FidelityPaper` only.

## When to use

Use CDA on continuous landscapes when deterministic, non-uniform coefficient sequences are
worth comparing with DA. Feature-selection callers can decode continuous coordinates into a
mask inside their objective. Start with Gauss/0.7; changing maps changes the algorithm rather
than merely reseeding it.

See also [BDA](bda.md), [MHDA](mhda.md), and the
[configuration guide](../api/configuration.md).
