# Research References

Everything this library implements, where it comes from, and — for the constants that could not
be checked — an explicit statement of what is verified and what is not.

## Primary source

**Mirjalili, S. (2016). Dragonfly algorithm: a new meta-heuristic optimization technique for
solving single-objective, discrete, and multi-objective problems. _Neural Computing and
Applications_, 27(4), 1053–1073.**

DOI: [10.1007/s00521-015-1920-1](https://doi.org/10.1007/s00521-015-1920-1)

One paper introduces all three variants. Its reference implementations are the author's `DA.m`,
`BDA.m` and `MODA.m`.

### Key contributions

- **Two swarm-level attractors.** Where PSO has a global best, DA has a food source _and_ an
  enemy: the best and worst positions seen so far. MATLAB-compatible DA separately preserves
  the reference's strictly-interior movement enemy while reporting the actual evaluated worst.
- **Five primitives instead of three.** Reynolds' separation, alignment and cohesion, plus food
  attraction and enemy distraction. The enemy term is `X⁻ + X_i`, a sum.
- **Static and dynamic swarming as one schedule.** The neighbourhood radius grows with the run,
  so the swarm searches as many small flocks early and as one flock late. The transition is
  geometric rather than a tuned coefficient, which is the paper's central idea.
- **A Lévy walk as the fallback for isolation.** A dragonfly with no neighbours and no food in
  reach takes a heavy-tailed multiplicative jump instead of stalling.
- **One step vector, three variants.** BDA reads the same `ΔX` as a bit-flip probability; MODA
  draws the food and enemy from a Pareto archive. Neither changes how the step is built.

### The variants

| Variant  | Paper section | What changes                                                                    |
| -------- | ------------- | ------------------------------------------------------------------------------- |
| **DA**   | §3            | the baseline: five primitives, two-branch step, Lévy fallback                   |
| **BDA**  | §4            | the position update becomes a per-bit flip through a transfer function          |
| **MODA** | §5            | the food source and the enemy are drawn from a hypercube-gridded Pareto archive |

## Supporting work

### Reynolds (1987) — separation, alignment, cohesion

**Reynolds, C. W. (1987). Flocks, herds and schools: A distributed behavioral model. _ACM
SIGGRAPH Computer Graphics_, 21(4), 25–34.**

DOI: [10.1145/37402.37406](https://doi.org/10.1145/37402.37406)

The boids model. Three local rules — steer away from crowding, match your neighbours' heading,
move toward your neighbours' centre — produce flocking without a leader or a global plan. DA's
first three primitives are these three rules, and its contribution is what it adds to them.

Implemented in `swarm.go`: `separationVector`, `alignmentVector`, `cohesionVector`.

### Mantegna (1994) — the Lévy flight

**Mantegna, R. N. (1994). Fast, accurate algorithm for numerical simulation of Lévy stable
stochastic processes. _Physical Review E_, 49(5), 4677–4683.**

DOI: [10.1103/PhysRevE.49.4677](https://doi.org/10.1103/PhysRevE.49.4677)

A method for drawing Lévy-stable steps from two Gaussians:

```
σ    = ( Γ(1+β)·sin(πβ/2) / ( Γ((1+β)/2)·β·2^((β-1)/2) ) )^(1/β)
Levy = scale · r₁·σ / |r₂|^(1/β),      r₁, r₂ ~ N(0,1)
```

Implemented in `levy.go`. **Verified**: `levySigma(1.5)` evaluates to `0.6965745026`, the
accepted Mantegna value for β = 1.5, and `0.01` is the step scale the DA reference
implementation uses. This is the one borrowed constant that has been checked rather than
assumed.

Two implementation notes. `|r₂|` is clamped to `1e-10` rather than redrawn, because a redraw
loop would consume a seed-dependent number of random values and desynchronize every subsequent
draw in the population; the clamp is unbiased in sign, affects a region of probability about
`8e-11`, and still admits a step of about `1e15 · scale · σ`, so the heavy tail survives. And a
degenerate β falls back to a plain Gaussian step rather than returning `NaN`.

### Deb (2000) — constraint feasibility rules

**Deb, K. (2000). An efficient constraint handling method for genetic algorithms. _Computer
Methods in Applied Mechanics and Engineering_, 186(2–4), 311–338.**

DOI: [10.1016/S0045-7825(99)00389-8](https://doi.org/10.1016/S0045-7825%2899%2900389-8)

Three rules for ranking constrained candidates without a penalty parameter to tune:

1. a feasible candidate always beats an infeasible one
2. two feasible candidates are ranked by objective value
3. two infeasible candidates are ranked by aggregate constraint violation

Implemented in `constraints.go` as `betterByFeasibilityRules`, and the default policy because
it needs no factor to tune. Penalty ranking is available as the alternative.

### Coello Coello, Pulido & Lechuga (2004) — the hypercube archive

**Coello Coello, C. A., Pulido, G. T., & Lechuga, M. S. (2004). Handling multiple objectives
with particle swarm optimization. _IEEE Transactions on Evolutionary Computation_, 8(3),
256–279.**

DOI: [10.1109/TEVC.2004.826067](https://doi.org/10.1109/TEVC.2004.826067)

MOPSO. Objective space is divided into hypercubes; a leader is drawn from a sparsely populated
one to spread the front out, and archive overflow deletes from a crowded one. MODA borrows this
mechanism wholesale and adds an enemy drawn from a crowded cell.

Implemented in `multiobjective.go` as the explicitly named `ArchivePolicyMOPSOGrid` extension.
It is not the paper or MATLAB default.

## Verification status of every borrowed constant

This section exists so a reader can tell a paper constant from a judgement call without reading
the source.

| Constant                          | Value                         | Status                                                                                       |
| --------------------------------- | ----------------------------- | -------------------------------------------------------------------------------------------- |
| Inertia bracket                   | `0.9 → 0.4`                   | Paper. `DA.m`.                                                                               |
| Convergence factor `mc`           | `max(0, 0.1·(1-2t/T))`        | Paper. `DA.m`.                                                                               |
| Radius schedule                   | `(ub-lb)/4 + (ub-lb)·(t/T)·2` | Paper. `DA.m`.                                                                               |
| Continuous step clamp             | `(ub-lb)/10`                  | Paper. `DA.m`.                                                                               |
| Enemy cutoff                      | `3T/4`                        | Paper. Note that it never bites at the default, because `mc` is already zero at `T/2`.       |
| Enemy term                        | `X⁻ + X_i`, a sum             | Paper and `DA.m`. Pinned by a hand-computed test.                                            |
| Neighbourhood test                | per-dimension box             | `DA.m`. Pinned by a hand-computed test.                                                      |
| Boundary rule                     | wrap, with a step redraw      | Paper default. MATLAB mode reproduces the reference pre-wrap/reset and post-move clamp.      |
| Lévy β                            | `1.5`                         | Paper.                                                                                       |
| Lévy σ                            | `0.6965745026`                | **Verified** against Mantegna (1994) for β = 1.5.                                            |
| Lévy scale                        | `0.01`                        | **Verified** as the DA reference implementation's value.                                     |
| BDA default transfer function     | `v3`                          | Paper, §4.                                                                                   |
| BDA step clamp `MaxStepRatio`     | `6.0`                         | **Verified** against official `BDA.m`.                                                       |
| MODA paper selection              | `1/N` food, `N` enemy         | Paper default, implemented by `ArchivePolicyPaperSegments`.                                  |
| MODA MATLAB selection             | span/20 density ranking       | **Verified** against `RankingProcess.m`, exposed as `ArchivePolicyMATLABDensity`.            |
| MODA MATLAB inertia               | `0.9 → 0.2`                   | **Verified** against `MODA.m`; automatic S/A/C/E follow `mc` directly.                       |
| MODA MOPSO `β`, `γ`, `δ`, `NGrid` | `4`, `2`, `2`, `10`           | Legacy extension only, exposed as `ArchivePolicyMOPSOGrid`; not DA/MODA reference constants. |
| MODA `ArchiveSize`                | `100`                         | **Verified** against official `MODA.m`.                                                      |

Official-source SHA-256 checksums used for this audit: `DA.zip`
`b3123fcea9fb35d2ed0ff123c2241263ceec44b9d9a774da9bd17ef036475ddc`, `BDA.zip`
`1801dac86c3e8c68cd404904b75ae200815555ceaefccba96cb19598d97cc1c6`, and `MODA.zip`
`81fc0096d5e552845743ebabc45d5cf81445f604dffc00d7e243b03e9cdf915f`.

## MATLAB fidelity and deliberate extensions

`FidelityMATLAB` follows the reference generation lifecycle: schedules, evaluation, incumbent
or archive update, then movement. Consequently its final moved population is intentionally
unevaluated and a full `NPop = N`, `T`-generation run makes `N·T` evaluations. Paper mode
evaluates initialization and every movement, for `N·(T+1)`.

Continuous MATLAB movement uses this exact order: calculate primitives, pre-wrap the current
dragonfly and reset violated step components, move using the already-calculated primitives,
sanitize non-finite values, then hard-clamp. `BoundaryMethod` is therefore a paper-mode choice
and is ignored by MATLAB-compatible DA/MODA. The final moved positions are repaired; earlier
claims that they were left out of bounds described an implementation gap that is now closed.

DA's reference movement enemy has a strict-interior update guard. The implementation preserves
that reference separately while still ranking every evaluated candidate for the public
`Result.Worst`. Population snapshots expose the movement enemy with a pre-movement copy of the
evaluated swarm, so their costs always describe their positions.

The remaining deliberate extensions are:

1. **Binary mode ignores `BoundaryMethod` and the Lévy branch** (`binary.go`). A 0/1 vector
   cannot leave `[0, 1]`, and the wrap rule's step reset would overwrite the step the next
   bit-flip decision reads. A Lévy walk is a multiplicative displacement of a real-valued
   position with no binary counterpart.
2. **The landscape classifier is scale-free** (`selector.go`). Mayfly's gradient-magnitude
   heuristic was deliberately not ported: it called Sphere over `[-5, 5]` rugged and Sphere over
   `[-1, 1]` smooth, which says more about the bounds than about the function. It is replaced by
   direction changes per line scan and total variation in units of that line's own value range,
   both of which mean the same thing whatever the box.

## Implementation notes

**Fidelity first.** Where the reference code and a cleaner formulation disagree, the reference
wins through the explicit `FidelityMATLAB` mode rather than an unnamed hybrid. `BoundaryMethod`
selects paper-mode wrapping, clamping or reflection because wrapping interacts badly with bounds
that encode a real constraint; MATLAB mode uses its fixed reference sequence.

**Determinism is a hard requirement.** Every stochastic helper takes `rng *rand.Rand` as its
last parameter; there is no package-level `math/rand` use. Every random draw an iteration makes
happens on the calling goroutine, and worker goroutines only evaluate the objective. A seeded
run is bit-identical with parallel evaluation on or off.

**Validation approach.** Four layers, described in `AGENTS.md`:

- **Hand-computed unit tests.** `swarm_test.go` builds a 3-dragonfly, 2-dimensional swarm and
  compares S, A, C, F, E and one full `ΔX` step against values worked out by hand from `DA.m`.
  That one test is what catches a Euclidean-versus-per-dimension neighbour mistake or a sign
  error in the enemy term — neither of which shows up as an obvious end-to-end failure.
- **Property tests.** `weights_test.go` asserts monotonicity and exact zero crossings of the
  schedules directly, rather than inferring them from optimization outcomes.
- **Determinism tests.** `TestParallelIsDeterministicForSeedAcrossSchedules` requires a seeded
  parallel run to be bit-identical to the sequential one.
- **Regression baselines with tolerances.** `RegressionBaseline` entries encode tolerated
  degradation factors, never golden values — a stochastic optimizer's output is not a golden
  file.

**An honest note on results.** DA's convergence factor reaches zero at the halfway point of a
run, after which only the food term and inertia still move a dragonfly. The consequence is
visible throughout [benchmarks.md](benchmarks.md): the algorithm explores well and finishes
poorly. Those numbers are what a faithful port produces, not what a tuned optimizer could, and
they are reported rather than dressed up.

## BibTeX

```bibtex
@article{mirjalili2016dragonfly,
  author  = {Mirjalili, Seyedali},
  title   = {Dragonfly algorithm: a new meta-heuristic optimization technique for
             solving single-objective, discrete, and multi-objective problems},
  journal = {Neural Computing and Applications},
  volume  = {27},
  number  = {4},
  pages   = {1053--1073},
  year    = {2016},
  doi     = {10.1007/s00521-015-1920-1}
}

@article{ranjini2017memory,
  author  = {Sree Ranjini, K. S. and Murugan, S.},
  title   = {Memory based Hybrid Dragonfly Algorithm for numerical optimization problems},
  journal = {Expert Systems with Applications},
  volume  = {83},
  pages   = {63--78},
  year    = {2017},
  doi     = {10.1016/j.eswa.2017.04.033}
}

@article{sayed2019chaotic,
  author  = {Sayed, Gehad Ismail and Tharwat, Aboul Ella and Hassanien, Aboul Ella},
  title   = {Chaotic dragonfly algorithm: an improved metaheuristic algorithm for feature selection},
  journal = {Applied Intelligence},
  volume  = {49},
  pages   = {188--205},
  year    = {2019},
  doi     = {10.1007/s10489-018-1261-8}
}

@article{yu2020quantum,
  author  = {Yu, Caiyang and Cai, Zhennao and Ye, Xiaojia and Wang, Mingjing and Zhao, Xuehua and
             Liang, Guoxi and Chen, Huiling and Li, Chengye},
  title   = {Quantum-like mutation-induced dragonfly-inspired optimization approach},
  journal = {Mathematics and Computers in Simulation},
  volume  = {178},
  pages   = {259--289},
  year    = {2020},
  doi     = {10.1016/j.matcom.2020.06.012}
}

@article{reynolds1987flocks,
  author  = {Reynolds, Craig W.},
  title   = {Flocks, herds and schools: A distributed behavioral model},
  journal = {ACM SIGGRAPH Computer Graphics},
  volume  = {21},
  number  = {4},
  pages   = {25--34},
  year    = {1987},
  doi     = {10.1145/37402.37406}
}

@article{mantegna1994fast,
  author  = {Mantegna, Rosario N.},
  title   = {Fast, accurate algorithm for numerical simulation of
             {L}{\'e}vy stable stochastic processes},
  journal = {Physical Review E},
  volume  = {49},
  number  = {5},
  pages   = {4677--4683},
  year    = {1994},
  doi     = {10.1103/PhysRevE.49.4677}
}

@article{deb2000efficient,
  author  = {Deb, Kalyanmoy},
  title   = {An efficient constraint handling method for genetic algorithms},
  journal = {Computer Methods in Applied Mechanics and Engineering},
  volume  = {186},
  number  = {2--4},
  pages   = {311--338},
  year    = {2000},
  doi     = {10.1016/S0045-7825(99)00389-8}
}

@article{coello2004handling,
  author  = {Coello Coello, Carlos A. and Pulido, Gregorio Toscano and
             Lechuga, Maximino Salazar},
  title   = {Handling multiple objectives with particle swarm optimization},
  journal = {IEEE Transactions on Evolutionary Computation},
  volume  = {8},
  number  = {3},
  pages   = {256--279},
  year    = {2004},
  doi     = {10.1109/TEVC.2004.826067}
}

@article{zitzler2000comparison,
  author  = {Zitzler, Eckart and Deb, Kalyanmoy and Thiele, Lothar},
  title   = {Comparison of multiobjective evolutionary algorithms:
             Empirical results},
  journal = {Evolutionary Computation},
  volume  = {8},
  number  = {2},
  pages   = {173--195},
  year    = {2000},
  doi     = {10.1162/106365600568202}
}

@inproceedings{schaffer1985multiple,
  author    = {Schaffer, J. David},
  title     = {Multiple objective optimization with vector evaluated
               genetic algorithms},
  booktitle = {Proceedings of the 1st International Conference on
               Genetic Algorithms},
  pages     = {93--100},
  year      = {1985}
}

@article{wilcoxon1945individual,
  author  = {Wilcoxon, Frank},
  title   = {Individual comparisons by ranking methods},
  journal = {Biometrics Bulletin},
  volume  = {1},
  number  = {6},
  pages   = {80--83},
  year    = {1945},
  doi     = {10.2307/3001968}
}

@article{friedman1937use,
  author  = {Friedman, Milton},
  title   = {The use of ranks to avoid the assumption of normality implicit
             in the analysis of variance},
  journal = {Journal of the American Statistical Association},
  volume  = {32},
  number  = {200},
  pages   = {675--701},
  year    = {1937},
  doi     = {10.1080/01621459.1937.10503522}
}

@techreport{awad2016cec2017,
  author      = {Awad, N. H. and Ali, M. Z. and Liang, J. J. and Qu, B. Y. and
                 Suganthan, P. N.},
  title       = {Problem Definitions and Evaluation Criteria for the {CEC 2017}
                 Special Session and Competition on Single Objective Bound
                 Constrained Real-Parameter Numerical Optimization},
  institution = {Nanyang Technological University},
  year        = {2016},
  month       = nov,
  url         = {https://github.com/P-N-Suganthan/CEC2017-BoundContrained}
}

@techreport{yue2019cec2020,
  author      = {Yue, C. T. and Price, K. V. and Suganthan, P. N. and Liang,
                 J. J. and Ali, M. Z. and Qu, B. Y. and Awad, N. H. and
                 Biswas, P. P.},
  title       = {Problem Definitions and Evaluation Criteria for the {CEC 2020}
                 Special Session and Competition on Single Objective Bound
                 Constrained Numerical Optimization},
  institution = {Zhengzhou University and Nanyang Technological University},
  year        = {2019},
  month       = nov,
  url         = {https://github.com/P-N-Suganthan/2020-Bound-Constrained-Opt-Benchmark}
}

@misc{budde2026dragonfly,
  author = {Budde, Christian-W.},
  title  = {Dragonfly: A Go implementation of the Dragonfly Algorithm},
  year   = {2026},
  note   = {\url{https://github.com/CWBudde/dragonfly}}
}
```

The numbered CEC suites follow Awad et al. (2016) and Yue et al. (2019), together with the
organizers' released evaluators and transformation data. The ZDT problems come from Zitzler,
Deb & Thiele (2000); SchafferN1 from Schaffer (1985); the Wilcoxon signed-rank and Friedman
tests in `comparison.go` from Wilcoxon (1945) and Friedman (1937). The standalone benchmark
functions are standard definitions with no single canonical citation.

## Further reading

**Related swarm metaheuristics**, for context on where DA sits:

- Kennedy & Eberhart (1995), Particle Swarm Optimization — the algorithm DA most resembles, and
  the one whose inertia-weight practice DA's `0.9 → 0.4` bracket comes from
- Mirjalili, Mirjalili & Lewis (2014), Grey Wolf Optimizer — the same author's earlier
  leader-following design
- Zervoudakis & Tsafarakis (2020), Mayfly Algorithm — implemented in the
  [sibling library](https://github.com/cwbudde/mayfly), which shares this one's API and
  benchmark suite, so the two are directly comparable

**Deferred here**, recorded in [PLAN.md](../PLAN.md): a head-to-head `dragonfly` versus
`mayfly` comparison harness and results table.

## Related documentation

- [Standard DA](algorithms/standard-da.md), [BDA](algorithms/bda.md), [MODA](algorithms/moda.md),
  [MHDA](algorithms/mhda.md), [CDA](algorithms/cda.md), and [QGDA](algorithms/qgda.md)
  — each variant's page opens with its own research reference
- [Benchmark Functions](benchmarks.md) — the test problems and what this implementation reaches
- [../README.md](../README.md#research--citations) — the Algorithm Implementation Map
- [../PLAN.md](../PLAN.md) — the specification these citations were implemented against
