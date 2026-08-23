# Dragonfly

A dependency-free Go implementation of the **Dragonfly Algorithm (DA)**, a swarm
metaheuristic introduced by Seyedali Mirjalili in 2016. It models the static and dynamic
swarming behaviour of dragonflies — separation, alignment, cohesion, attraction to food,
and distraction from enemies — and covers all three variants from the original paper.

Sibling project: [mayfly](https://github.com/cwbudde/mayfly), which shares this library's
API style, tooling, and conventions.

> **Status: pre-release.** The scaffold and roadmap are in place; the algorithm is being
> implemented. See [PLAN.md](PLAN.md) for the phased plan and current progress.

## Overview

### Key features (planned)

- **Three variants** — continuous DA, binary BDA, and multi-objective MODA
- **Standard library only** — the sole direct dependency is godog, and it is test-only
- **Deterministic** — a seeded run reproduces exactly, with parallel evaluation on or off
- **Constraint handling** — Deb's feasibility rules or linear/quadratic penalties
- **Observable** — progress and population observers, `slog` integration, CSV/JSON export
- **Benchmark suite** — 15 classic and CEC-style test functions
- **Statistical comparison** — paired-seed runs with Wilcoxon and Friedman tests

## Quick Start

### Installation

```sh
go get github.com/MeKo-Christian/dragonfly
```

### Basic usage

```go
config := dragonfly.NewDefaultConfig()
config.ObjectiveFunc = dragonfly.Sphere
config.ProblemSize = 30
config.LowerBound = -100
config.UpperBound = 100
config.MaxIterations = 500

result, err := dragonfly.Optimize(config)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("best cost: %.10f\n", result.GlobalBest.Cost)
```

## Algorithm Variants

| Variant  | Best for                            | Documentation |
| -------- | ----------------------------------- | ------------- |
| **DA**   | Single-objective continuous problems | [docs/algorithms/standard-da.md](docs/algorithms/standard-da.md) |
| **BDA**  | Binary and discrete problems, feature selection | [docs/algorithms/bda.md](docs/algorithms/bda.md) |
| **MODA** | Multi-objective problems             | [docs/algorithms/moda.md](docs/algorithms/moda.md) |

## Build Commands

```sh
just build     # go build ./...
just test      # tests with coverage
just check     # format + tidy + lint + test
just bench     # benchmarks
just lint-fix  # format and auto-fix lint findings
```

Run `just --list` for the full recipe list.

## Research & Citations

Mirjalili, S. (2016). Dragonfly algorithm: a new meta-heuristic optimization technique for
solving single-objective, discrete, and multi-objective problems. *Neural Computing and
Applications*, 27(4), 1053–1073.
doi:[10.1007/s00521-015-1920-1](https://doi.org/10.1007/s00521-015-1920-1)

Go implementation by Christian-W. Budde.

## Development Status

See [PLAN.md](PLAN.md).

## Contributing

1. Fork the repository and create a feature branch
2. Follow the conventions in [AGENTS.md](AGENTS.md) and [CLAUDE.md](CLAUDE.md)
3. Add tests for new behaviour
4. Run `just check` and make sure it is green
5. Open a pull request describing the goal, the key changes, and the impact

## License

MIT — see [LICENSE](LICENSE).
