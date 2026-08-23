# The Dragonfly WebAssembly demo

Four pages that run this library in a browser, published to
<https://cwbudde.github.io/Dragonfly/>.

The organising rule: **no optimization logic lives in JavaScript.** Every number
the pages draw is computed by `github.com/CWBudde/dragonfly` compiled to
`js/wasm`. JavaScript reads typed arrays and puts pixels on a canvas; it never
decides anything about the algorithm. A demo that reimplemented Rastrigin, or
the neighbourhood test, or a transfer function would be a plausible-looking lie
the moment either definition drifted — and the neighbourhood test in particular
is the one people get wrong.

## Running it

```bash
just run-wasm-demo      # build into ./dist and serve at http://localhost:8090
just check-wasm-demo    # compile-check both build tags, no output
```

**An HTTP server is required.** `file://` breaks the wasm fetch on every page and
the worker on the shootout. `just run-wasm-demo` is a one-line `python3 -m
http.server`; anything equivalent works.

`wasm_exec.js` is copied from the toolchain at build time and is deliberately
**not** committed: it is version-locked to the compiler that produced the
`.wasm`, and a stale copy fails at runtime in ways that look like demo bugs.
The build script probes both `$GOROOT/lib/wasm` (Go ≥ 1.24) and
`$GOROOT/misc/wasm`.

## The pages

| Page             | What it exercises                                                                             |
| ---------------- | --------------------------------------------------------------------------------------------- |
| `index.html`     | **Swarm Lab.** `OptimizeContext` over a benchmark landscape, replayed against a scrubber.     |
| `pareto.html`    | **Pareto.** `OptimizeMultiObjective` with the archive animated through `WithArchiveObserver`. |
| `binary.html`    | **Binary.** `OptimizeBinaryContext` as a bit matrix, beside the transfer function's curve.    |
| `benchmark.html` | **Shootout.** `ComparisonRunner` over DA's configurable choices, with Wilcoxon and Friedman.  |

What each is really for:

- The **Swarm Lab** exists to make the two-branch step update visible. Each
  dragonfly is coloured by the branch it is about to take — the five-factor
  step, local swarming on S/A/C alone, or a Lévy walk — and the neighbourhood is
  drawn as a _square_, because the neighbour test is per-dimension.
- The **Pareto** page exists because MODA's hypercube grid is the whole
  mechanism that keeps a front from collapsing to a point, and it is invisible
  in a plain scatter. Cells are shaded by occupancy: the food source is drawn
  from the sparsest, the enemy and every eviction from the most crowded.
- The **Binary** page pairs the bit matrix with the transfer function's own
  curve, so "why did that bit flip" has an answer on screen.
- The **Shootout** compares configurations rather than variants, because the
  library has two single-objective variants and BDA searches `{0,1}ᵈ` — putting
  it beside DA on Rastrigin compares two different problems.

## Layout

| File                                              | Role                                                           |
| ------------------------------------------------- | -------------------------------------------------------------- |
| `main.go` / `main_stub.go`                        | the export table; the stub keeps non-wasm builds working       |
| `bridge.go`                                       | `guard()`, `errorResult`, the tolerant option readers          |
| `marshal.go`                                      | `Float32Array` sinks over JS-owned `ArrayBuffer`s              |
| `limits.go`                                       | the caps every export clamps to                                |
| `benchmarks.go`                                   | bounds and minimisers for the library's benchmark functions    |
| `variants.go` / `rng.go`                          | small shared helpers                                           |
| `info.go`                                         | the capability table every `<select>` is built from            |
| `run.go` / `landscape.go`                         | the Swarm Lab's two exports                                    |
| `pareto.go`                                       | the MODA export                                                |
| `binary.go`                                       | the BDA export, plus the demo's own 0/1 objectives             |
| `compare.go`                                      | the shootout's export and its configuration contenders         |
| `boot.js`                                         | wasm loading, safe calls, buffer reuse, the transport — shared |
| `render.js`                                       | every pixel any page draws                                     |
| `app.js` / `pareto.js` / `binary.js` / `bench.js` | one controller per page                                        |
| `bench-worker.js`                                 | the shootout's wasm instance, off the main thread              |
| `bench-chart.js`                                  | the grouped bar chart                                          |
| `style.css`                                       | one stylesheet, shared                                         |

No build tooling, no npm, no bundler: plain `<script src>` files, no ES modules.

## Two things that look odd and are not

**`guard()` wraps every export in a `recover()`.** A Go panic that unwinds out
of a `js.Func` aborts the whole wasm instance, and every later call then fails —
a single bad request would permanently brick the page until a reload. Returning
the failure as data costs one deferred `recover` per call.

**The shootout's sweep loop is in JavaScript, not Go.** A call into Go blocks
its thread's event loop for its whole duration, so a Stop button cannot be
serviced while one is in flight. Chunking the sweep at one benchmark per call,
with a `setTimeout(0)` between calls, _is_ the cancellation mechanism. It is
also why the shootout's instance lives in a classic worker: `wasm_exec.js`
assigns `globalThis.Go` and a module worker cannot `importScripts()` it.

## Reading the numbers

Timings under `js/wasm` are single-threaded and without SIMD, and every run sets
`EnableParallel = false` — goroutines here are cooperatively scheduled onto one
browser thread, so a worker pool would cost coordination and buy nothing. Treat
any duration on these pages as a relative cost within the page, never as a
measure of the library's speed.

The Swarm Lab and the Pareto page are **record-then-replay**: Go computes the
whole history and JavaScript replays it. That is not a performance trick, it is
what makes the scrubber possible — you cannot scrub backwards through a live
computation. It also makes reproducibility demonstrable: the same seed produces
the same history, and _Same seed_ proves it.

Above two dimensions the Swarm Lab's heatmap is a **slice**, with the unplotted
axes pinned at the known minimiser. Where no minimiser is known — Michalewicz
above 2-D — the slice is taken through the middle of the domain and the page
says so, rather than claiming to pass through an optimum nobody can name.
