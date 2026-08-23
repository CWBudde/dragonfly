# Performance and Profiling

Every number on this page was measured on this machine:

- **CPU** AMD Ryzen 5 4600H with Radeon Graphics, 6 cores / 12 threads
- **OS** Linux 7.0.0-30-generic, `linux/amd64`
- **Go** 1.26.0
- **Command** `go test -bench=. -benchmem -run='^$' ./...`, plus `-count=5` for the medians in
  the [end-to-end table](#end-to-end-baselines)

Absolute timings are machine-specific and the timing noise on this box is about ±10%; the
allocation counts are exact and reproduce run to run. Re-measure on your own hardware before
using anything here as a threshold. What does carry across machines is the _shape_ of the
scaling, which is what the sections below are actually about.

## Reproducible baseline

```sh
just bench
# or, more precisely:
go test -run '^$' -bench '^Benchmark' -benchmem -count=10 . > before.bench
# apply the change
go test -run '^$' -bench '^Benchmark' -benchmem -count=10 . > after.bench
benchstat before.bench after.bench
```

The `.bench` extension is ignored by `.gitignore`, so local measurements do not enter commits.

Every optimizer benchmark builds a **fresh, fixed-seed configuration per operation**. That
matters twice: it keeps the measurement reproducible, and it stops one operation's optimizer
mutations — `OptimizeContext` writes a generator back into a nil `Config.Rand` — from carrying
into the next.

## End-to-end baselines

`BenchmarkOptimizeBaseline` is the profiling anchor: a 30-dimensional Sphere, `NPop` 40, 100
iterations, seed 42, 4,040 objective evaluations. `just profile-cpu` and `just profile-mem`
both select it by exactly that name.

Median of five 1 s samples:

| Benchmark                                 | Time/op |  Bytes/op | Allocs/op |
| ----------------------------------------- | ------: | --------: | --------: |
| `BenchmarkOptimizeBaseline` (DA)          | 49.8 ms | 6,309,931 |    24,927 |
| `BenchmarkOptimizeBaselineParallel` (DA)  | 57.3 ms | 6,463,409 |    27,050 |
| `BenchmarkOptimizeBinaryBaseline` (BDA)   | 45.8 ms | 7,069,214 |    28,093 |
| `BenchmarkOptimizeMultiObjectiveBaseline` | 55.7 ms | 8,968,818 |    59,133 |

BDA costs about what DA does — the bit-flip update replaces the continuous position update
rather than adding to it — with slightly more allocation from the per-dragonfly position save
and restore in `buildBinaryStep`. MODA costs about 12% more time and **2.4x the allocations**:
every accepted candidate is stored in the archive as a deep copy, and `updateGrid` reassigns
every solution's cell on every mutation. `MODAVariant.EstimatedOverhead()` reports `1.2`, which
the timing supports.

## Parallel evaluation only pays on an expensive objective

This is the single most useful measured finding in the library.

| Workload                                            | Sequential | Parallel |   Change |
| --------------------------------------------------- | ---------: | -------: | -------: |
| 30-D Sphere, 100 iterations, `NPop` 40 (the anchor) |    49.8 ms |  57.3 ms | **+15%** |
| 10-D Rastrigin ×200, 50 iterations, `NPop` 30       |    37.3 ms |  22.3 ms | **-40%** |

`BenchmarkParallelEvaluation` is the second row: a deliberately expensive objective that
evaluates Rastrigin 200 times per call. With it the pool is 1.67x faster on 12 threads. On
Sphere — about 10 ns per call, measured below — dispatch and synchronization cost more than the
work, and `EnableParallel` makes the run measurably slower.

The rule of thumb: leave `EnableParallel` off below roughly a microsecond per evaluation, and
measure your own objective rather than guessing. `BenchmarkOptimizeBaselineParallel` exists
precisely to document the floor of what the pool costs before an expensive objective starts to
earn it back.

Parallelism never changes the answer. A seeded run is bit-identical with `EnableParallel` on or
off — verified end to end:

```
sequential: 11.3276456009 in 35.1898ms
parallel:   11.3276456009 in 13.459845ms
bit-identical: true
```

## Population scaling is super-linear

The neighbourhood scan is `O(n²·d)` and it dominates for large swarms. 10-D Sphere, 50
iterations:

| `NPop` | Time/op |   Bytes/op | Allocs/op | vs. previous row     |
| -----: | ------: | ---------: | --------: | -------------------- |
|     10 | 1.27 ms |    255,600 |     3,115 | —                    |
|     40 | 8.76 ms |  1,474,945 |    12,332 | 4x swarm → 6.9x time |
|    100 | 39.5 ms |  6,545,833 |    30,670 | 2.5x → 4.5x          |
|    250 |  207 ms | 30,731,872 |    76,372 | 2.5x → 5.2x          |

Roughly `n^1.8` over this range. Allocations grow linearly — they are per-dragonfly, per
iteration — so it really is the scan, not the bookkeeping.

Budget accordingly: doubling `NPop` costs about three times as much wall clock, and it is not
obvious that a swarm of 250 buys three times the search of a swarm of 100.

## Dimension and iteration scaling are linear

`BenchmarkDimensionScaling`, Sphere, 50 iterations, `NPop` 30:

| Dimensions | Time/op |  Bytes/op | Allocs/op |
| ---------: | ------: | --------: | --------: |
|          2 | 1.78 ms |   441,504 |     6,074 |
|         10 | 5.43 ms |   990,464 |     9,284 |
|         30 | 14.6 ms | 2,261,109 |     9,395 |
|        100 | 50.9 ms | 7,531,996 |     9,504 |

Ten times the dimensions for 9.4 times the time: every primitive is a vector operation over `d`
components, so the shape is linear in `d` on top of the objective's own cost. Allocation
_counts_ barely move past `d = 10` — the same number of slices, each wider.

`BenchmarkIterationScaling`, Sphere, 10-D, `NPop` 30:

| Iterations | Time/op | Allocs/op |
| ---------: | ------: | --------: |
|         25 | 2.90 ms |     4,685 |
|        100 | 10.4 ms |    18,450 |
|        400 | 47.7 ms |    73,361 |

Linear, as it should be: a run that is slower than expected is slower per iteration rather than
running more of them.

## The neighbourhood scan in isolation

`BenchmarkNeighborScan` scans the whole swarm once — every dragonfly against every other — at 30
dimensions over a box of width 20, at an early-run radius (small neighbourhoods) and a late-run
one (the radius exceeds the box width, so every dragonfly is everyone's neighbour):

| `NPop` | Early radius | Late radius |
| -----: | -----------: | ----------: |
|     50 |      48.4 µs |      208 µs |
|    100 |       335 µs |      708 µs |
|    200 |      1.26 ms |     2.93 ms |
|    500 |      7.92 ms |     20.2 ms |

Quadratic in both columns, and three to four times more expensive late in the run than early,
because a full neighbourhood means every `withinRadius` call runs to completion instead of
short-circuiting on the first out-of-range component.

The scan draws no random numbers, which is exactly what would make it safe to parallelize
without breaking reproducibility. It has deliberately **not** been parallelized: this benchmark
exists to establish whether the complexity would be worth it before anyone adds it.

`BenchmarkSwarmStep` measures the whole prepare phase at the same 30 dimensions — scan plus the
five primitives plus the step update and the boundary repair — for one iteration: 243 µs at
`NPop` 50, rising to 23.4 ms at `NPop` 500. Read the scan table as a fraction of these.

## Objective function cost

`func([]float64) float64` on a 30-dimensional vector, no allocations except where noted:

| Function      |                  Time/op |
| ------------- | -----------------------: |
| `Rosenbrock`  |                  9.11 ns |
| `Sphere`      |                  10.0 ns |
| `Rastrigin`   |                   309 ns |
| `Ackley`      |                   343 ns |
| `Griewank`    |                   426 ns |
| `Schwefel`    |                   630 ns |
| `Levy`        | 1.03 µs (240 B, 1 alloc) |
| `Weierstrass` |                  40.1 µs |

Weierstrass is four thousand times more expensive than Sphere — 21 cosines per dimension — and
is the one bundled function where `EnableParallel` is clearly worth switching on. `Levy` is the
only one that allocates, for its `w` scratch slice.

Use this table to decide where your own objective sits relative to the microsecond threshold in
[the parallel section](#parallel-evaluation-only-pays-on-an-expensive-objective).

## Transfer functions

`BenchmarkTransferFunctions`, BDA on 30-bit OneMax, 50 iterations, `NPop` 30. All eight
allocate identically (10,573 allocs/op); they differ by one transcendental call per component
per iteration:

| Transfer | Time/op | Transcendental |
| -------- | ------: | -------------- |
| `v3`     | 17.3 ms | `sqrt`         |
| `v1`     | 18.8 ms | `erf`          |
| `v2`     | 18.9 ms | `tanh`         |
| `s3`     | 19.6 ms | `exp`          |
| `s2`     | 19.9 ms | `exp`          |
| `v4`     | 20.1 ms | `atan`         |
| `s1`     | 21.7 ms | `exp`          |
| `s4`     | 21.9 ms | `exp`          |

A 25% spread from cheapest to dearest. The paper's default `v3` is also the fastest, so there
is no cost argument for moving off it — pick a transfer function on
[behaviour](algorithms/bda.md#performance), not on speed.

## Boundary methods

`BenchmarkBoundaryMethods`, 10-D Rosenbrock, 50 iterations, `NPop` 30:

| Method    | Time/op | Allocs/op |
| --------- | ------: | --------: |
| `clamp`   | 5.55 ms |     9,270 |
| `reflect` | 5.55 ms |     9,267 |
| `wrap`    | 6.32 ms |     9,274 |

Wrapping costs about 14% more, which is the uniform draw per repaired component that the other
two do not take. That is the price of the paper's default, and it is small enough that it should
never be the reason to switch — switch on
[behaviour](api/configuration.md#boundary-handling) instead.

## Observer overhead

30-D Sphere, 100 iterations, `NPop` 40, median of five samples:

| Observers  | Time/op |  Bytes/op | Allocs/op |
| ---------- | ------: | --------: | --------: |
| none       | 45.0 ms | 6,309,920 |    24,927 |
| progress   | 48.8 ms | 6,333,920 |    25,027 |
| population | 51.3 ms | 8,546,720 |    33,227 |

The timings are within the noise of each other; the allocation counts are not, and they are the
number to plan against. A progress observer costs exactly 100 extra allocations over 100
iterations — one `Best` clone each, about 240 bytes. A population observer costs 8,300 extra
allocations and 2.2 MB: 83 per iteration, which is 40 dragonflies × 2 vectors, plus the swarm
slice and the two `Best` clones.

That factor of 83 is why the two are separate options rather than one snapshot. No copying
happens at all unless an observer is registered.

## MODA and the archive

Four multi-objective benchmarks, 10-D, 50 iterations, `NPop` 30, `ArchiveSize` 100, median of
five samples, before and after the archive hot path was reworked:

| Problem      | Time/op before |    after | Bytes/op before |     after | Allocs/op before |  after |
| ------------ | -------------: | -------: | --------------: | --------: | ---------------: | -----: |
| `ZDT2`       |        8.66 ms |  7.14 ms |       1,505,850 | 1,435,459 |           18,102 | 16,061 |
| `ZDT3`       |       10.13 ms |  7.18 ms |       1,722,906 | 1,444,970 |           21,492 | 16,134 |
| `ZDT1`       |       11.27 ms |  7.99 ms |       2,131,404 | 1,498,490 |           33,245 | 16,947 |
| `SchafferN1` |       75.76 ms | 19.72 ms |      17,921,297 | 1,972,058 |          379,024 | 22,534 |

SchafferN1 used to cost six to ten times what the ZDT problems cost, on a **one-dimensional**
problem with a trivial objective. The reason was the archive, not the search: SchafferN1's front
is one-dimensional and dense, so almost every candidate is non-dominated, the archive fills and
stays full, and every insert at capacity ran the grid maintenance twice — once from `Add` and
once more from the eviction — with a fresh cell index allocated per member each time, and
grouped the archive by hypercube through a freshly built and sorted map.

Three changes removed that work without changing a single archived value:
`updateGrid` reuses each member's index array and skips the reassignment sweep entirely when the
recomputed bounds are bit-identical, which they usually are; `occupiedCells` groups the archive
with a counting sort over the grid keys into reused buffers instead of a map; and `Add` compacts
the survivors in place rather than allocating a new slice of every member. Together they cut
SchafferN1 to a quarter of its time and a sixteenth of its allocations, and the ZDT problems
gain too, in proportion to how often their archives sit at capacity.

What this does not change is the accept rate. Nearly every SchafferN1 candidate is still
non-dominated, and the archive is still full from the first few iterations — that is a property
of the problem's front, not something the archive can optimize away. `Add`'s domination sweep is
still `O(archive)` per candidate, which is why SchafferN1 remains the most expensive of the four.

The lesson for a real multi-objective problem is unchanged: **archive maintenance, not the swarm,
is what MODA pays for**, and `ArchiveSize` is therefore a cost parameter as well as a quality one.

## CPU and memory profiles

```sh
just profile-cpu   # cpu.pprof
just profile-mem   # memory.pprof
```

Equivalently:

```sh
go test -run '^$' -bench '^BenchmarkOptimizeBaseline$' -benchtime=5s \
  -cpuprofile=cpu.pprof -memprofile=memory.pprof .
```

Inspect:

```sh
go tool pprof -top cpu.pprof
go tool pprof -top -alloc_space memory.pprof
go tool pprof -http=:0 cpu.pprof
```

Profile files and the test binary are ignored by Git. Remove them after analysis:
`rm -f cpu.pprof memory.pprof dragonfly.test`.

Where to look first, given the scaling above: `findNeighbors` and `withinRadius` for CPU on a
large swarm, and `separationVector` / `alignmentVector` / `cohesionVector` / `foodVector` /
`enemyVector` for allocations — each returns a freshly allocated vector per dragonfly per
iteration, which is five allocations per dragonfly per iteration and accounts for most of the
24,927 in the anchor.

## Measuring solution quality rather than speed

The benchmarks on this page measure wall-clock cost. They say nothing about whether the answer
is any good, which is `regression_test.go`'s job and is a statistical question with a tolerance
attached. See [benchmarks.md](benchmarks.md#regression-baselines) for the baselines and
[comparison-framework.md](api/comparison-framework.md) for comparing two configurations
properly.

## Related documentation

- [Configuration Guide](api/configuration.md#parallel-evaluation) — when to enable the pool
- [Benchmark Functions](benchmarks.md) — the functions timed above, and their measured quality
- [Standard DA](algorithms/standard-da.md) — the `O(n²·d)` scan and why it exists
- [MODA](algorithms/moda.md) — the archive that dominates the multi-objective cost
