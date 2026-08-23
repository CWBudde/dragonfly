# Repository Guidelines

## Project Structure & Modules

- Root module: `github.com/MeKo-Christian/dragonfly`, package `dragonfly`, Go 1.23.3.
- Flat layout: source and tests are siblings at the repo root (`*.go`, `*_test.go`). No `internal/`, `pkg/`, or `cmd/`.
- Docs in `docs/`; runnable examples in `examples/` (each subdirectory its own module with a `replace` directive); Gherkin features in `features/`.
- Tooling configs: `.golangci.toml` (TOML, golangci-lint v2), `treefmt.toml`, `justfile`.
- Dependencies: standard library only. The sole planned direct dependency is `github.com/cucumber/godog`, and it is test-only.
- **No Go source exists yet.** `PLAN.md` is the source of truth for the roadmap, the algorithm specification (§1), the target file layout (§2), and current progress — check its checkboxes before starting work.

## Build, Test, and Dev Commands

- `just build`: Compile all packages (`go build -v ./...`).
- `just test`: Unit tests with coverage; writes `coverage.{out,html}`.
- `just test-quick` / `just test-race` / `just test-full`: Short, race-checked, or long-running runs.
- `just test-integration`: Godog-backed feature tests (`go test -run TestFeatures`).
- `just bench`: Benchmarks with memory stats; `just profile-cpu` / `just profile-mem` around `BenchmarkOptimizeBaseline`.
- `just run`: Run the example app in `examples/`.
- `just fmt` (alias `just treefmt`) / `just lint` / `just lint-fix`: Format via treefmt, lint via golangci-lint using `.golangci.toml`.
- `just check`: `check-formatted`, `check-tidy`, `lint`, `test` — use in PRs/CI. `just check-race` for the race variant.
- `just setup-deps` / `just install-tools`: Install treefmt, golangci-lint v2, gofumpt, gci, shfmt, prettier, taplo.

## Coding Style & Naming

- Go defaults: tabs, standard import grouping. Formatting is gofumpt followed by gci (local prefix `github.com/MeKo-Christian/dragonfly`).
- Naming: exported identifiers `CamelCase`, internal `lowerCamel`, packages short and lowercase.
- Short mathematical identifiers that mirror the paper (`w`, `s`, `a`, `c`, `f`, `e`, `r`, `dX`) are fine and preferred in the algorithm files; keep them out of public API names.
- Prefer small, cohesive files and pure functions. One concern per file, as in `PLAN.md` §2.
- Every stochastic helper takes `rng *rand.Rand` as its last parameter.
- Linting per `.golangci.toml`. Mayfly's per-file complexity exemptions are deliberately not inherited.

## Testing Guidelines

- Framework: standard `go test`; BDD via `godog` over `features/*.feature` (`just test-integration`).
- Test files are `*_test.go` siblings at the repo root in `package dragonfly` — white-box, free to exercise unexported helpers. `example_test.go` is the only `package dragonfly_test` file.
- Names: `TestXxx`, `BenchmarkXxx`, `ExampleXxx`. Benchmarks live in `*_test.go` as `BenchmarkXxx(b *testing.B)`.
- Do **not** use `t.Parallel()`: tests are deterministic and seed-driven, and parallel execution buys nothing while making failures harder to reproduce.
- Keep `just check` green; inspect `coverage.html` locally.

## Commit & PR Guidelines

- Commits: conventional style — `feat: ...`, `fix: ...`, `chore: ...`.
- PRs: state the goal, the key changes, and the before/after impact; link issues.
- Add or adjust unit and feature tests for any behavior change, and update the `PLAN.md` checkboxes the change completes.

## Security & Maintenance

- Dependencies: `just tidy` then `just verify`.
- Scan: `just security` (uses Nancy) before a release.
- Release prep: `just ci`, then `just release-check <semver>`, then tag with `just release version=<semver>`.
