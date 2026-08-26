# Cross-library head-to-head

This executable compares the standard continuous Dragonfly Algorithm (DA), the standard
Mayfly Algorithm (MA), and full-covariance CMA-ES under one objective-evaluation budget. CMA-ES
is a calibration baseline; the roadmap item being answered is DA versus MA.

The module intentionally uses local replacements. Check out the three sibling repositories in
this layout before running it:

```text
libs/
├── Dragonfly/
├── Mayfly/
└── go-cma-es/
```

From the Dragonfly root, `just head-to-head` runs the recorded study and replaces
`docs/measurements/v0.2.0-head-to-head.{csv,md}` atomically. `just test-head-to-head` runs the
harness's unit and exact-budget integration tests.

## Recorded protocol

- 30 dimensions and 30 paired seeds (`20260827` through `20260856`)
- Sphere `[-100,100]`, Rosenbrock `[-5,10]`, Rastrigin `[-5.12,5.12]`, Ackley
  `[-32.768,32.768]`, and Griewank `[-600,600]`
- exactly 20,000 real objective calls per optimizer, per seed, per problem
- DA: paper-default configuration and one 40-member swarm
- MA: standard defaults, including 20 male and 20 female mayflies
- CMA-ES: full active covariance, dimension-derived population, a seeded uniform initial mean,
  and initial sigma `0.3*(upper-lower)`
- no early stopping, inner evaluation parallelism, or ranking by elapsed time

The physical boxes are deliberate. DA's published enemy term is `X^- + X_i`, so translating a
symmetric box to `[0,1]^D` changes the algorithm rather than merely changing coordinates.

DA's paper lifecycle spends `NPop*(iterations+1)` calls, so 499 iterations consume exactly
20,000 calls at `NPop=40`. MA and CMA-ES may reach the cap partway through their final
generation. The harness calls the real objective exactly to the cap, scores later candidates
as `+Inf`, and verifies every completed run's accounting before writing either report.

## Custom run

```bash
cd examples/head-to-head
go run . -runs 10 -dimensions 20 -budget 10000 -seed 42 -out trial.csv
```

Pass `-cma=false` for a DA/MA-only experiment. Run `go run . -help` for provenance flags and
the Markdown output override.
