# Releasing Dragonfly

This module uses Go module versioning and Semantic Versioning. Release tags have the form
`vMAJOR.MINOR.PATCH`, and the module stays in the unstable `v0` series until its public API is
declared stable.

Go modules are published from repository tags rather than uploaded anywhere. Pushing a tag and
fetching it once through the public proxy is the whole publication step.

## Version policy

- Increment **PATCH** for backward-compatible fixes and documentation changes.
- Increment **MINOR** for backward-compatible features. While the module is at `v0`, use a minor
  release for any intentional public API break as well, and call it out prominently in the
  changelog.
- Use a prerelease suffix such as `v0.2.0-rc.1` for release candidates.
- From `v1.0.0` onwards, increment **MAJOR** for breaking changes. A future v2 must also change
  the module path to `github.com/MeKo-Christian/dragonfly/v2`.
- **Never move or replace a published tag.** The Go module proxy caches tags immutably; publish
  a new version for corrections.

### What counts as a breaking change here

Beyond the obvious signature changes, three things specific to this library:

- **Changing a default that alters search behaviour** — `BoundaryMethod`, `MaxStepRatio`,
  `RadiusGrowth`, `NPop`, the transfer function. A caller's results change without their code
  changing.
- **Changing the number or order of RNG draws per iteration.** A seeded run that used to produce
  one trajectory now produces another. This is invisible to the compiler and very visible to
  anyone who recorded a seed. `computeWeights` takes four draws whether or not the weights are
  pinned precisely so that pinning one is _not_ such a change.
- **Resolving one of the unverified constants** — MODA's `β`, `γ`, `δ`, `NGrid`, `ArchiveSize`,
  or BDA's `MaxStepRatio`. If checking against the reference code moves a default, say so in the
  changelog under its own heading, not as a bullet among fixes.

## Release checklist

1. Choose the next version from the changes since the latest tag.
2. Move the relevant entries from `## [Unreleased]` in `CHANGELOG.md` into a dated version
   section and update its comparison links.
3. Run `just release-check version=MAJOR.MINOR.PATCH`.
4. Confirm the package overview reads correctly with `go doc .`, and check that the README and
   `docs/` links resolve in the repository browser.
5. Commit the release preparation as `chore: prepare vMAJOR.MINOR.PATCH`.
6. Run `just release version=MAJOR.MINOR.PATCH` to create the annotated tag locally.
7. Push the commit and the tag: `git push origin main vMAJOR.MINOR.PATCH`.
8. Confirm the GitHub release-validation workflow succeeds.
9. Ask the public Go proxy to fetch the immutable tag:
   `GOPROXY=proxy.golang.org go list -m github.com/MeKo-Christian/dragonfly@vMAJOR.MINOR.PATCH`
10. Verify the version and the rendered package documentation at
    <https://pkg.go.dev/github.com/MeKo-Christian/dragonfly>.

## What `just release-check` verifies

```sh
just release-check 0.1.0
```

- the argument is a valid semantic version
- `CHANGELOG.md` contains a `## [0.1.0]` section
- `LICENSE` and `README.md` are non-empty
- `go list -m` reports exactly `github.com/MeKo-Christian/dragonfly`
- `just verify` — `go mod verify`
- `just check-formatted` — `treefmt --fail-on-change`
- `just check-tidy` — `go mod tidy -diff`
- `just lint` — `golangci-lint run --config ./.golangci.toml`
- `go vet ./...`
- `go test -timeout 20m ./...` — the full suite, not the `-short` one

`just release version=0.1.0` runs all of that, then additionally requires a clean worktree and
an unused tag name before creating the annotated tag. It does not push.

## Release gates beyond the checklist

Phase 10 of [PLAN.md](../PLAN.md) adds two the tooling does not enforce:

- **80%+ statement coverage.** `just test` writes `coverage.out` and `coverage.html`.
- **`just security` clean.** This is two scans, and they cover different things.
  `just audit` pipes `go list -json -deps ./...` into `nancy sleuth`, which sees only the
  production build — and because the library is stdlib-only, that build has no third-party
  packages at all, so nancy reports `Audited Dependencies: 0`. That is the intended state
  rather than a passing scan, and the recipe exists to catch the first real dependency that
  is ever added. `just vuln` runs `govulncheck ./...`, which does cover the test-only
  dependency tree and reports by reachability, so it is the one that would actually find
  something today.

  Note that `govulncheck` also reports vulnerabilities in the Go toolchain the scan runs on.
  Those are a property of the local Go installation, not of this module; check whether a
  finding names `stdlib` or `os@go1.x` before treating it as a release blocker.

Before tagging a release, also re-measure [performance.md](performance.md) on the release
machine if any of its numbers are quoted as thresholds anywhere, and re-run the long regression
suite without `-short`:

```sh
just test-full
```

The regression baselines are the ones that catch a change that compiled, passed the unit tests
and quietly made the optimizer worse.

## Validation workflow

`.github/workflows/release.yml` runs for SemVer-shaped tags and can also be started manually. It
validates the version, the module metadata, the licence and the changelog, then runs static
analysis and the complete test suite. It does **not** create tags or GitHub releases; those stay
deliberate maintainer actions.

`.github/workflows/test.yml` is the ordinary CI: format, lint, test on a Go 1.23 and 1.24
matrix, and benchmarks.

## Changelog conventions

`CHANGELOG.md` follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/): an
`## [Unreleased]` section at the top, then dated version sections, each with `Added`, `Changed`,
`Deprecated`, `Removed`, `Fixed` and `Security` subheadings as needed.

Write entries for the person upgrading, not for the person who made the change. "The enemy
weight now defaults to `WeightAuto` instead of `0`, so runs that relied on the enemy term being
disabled must now pin it explicitly" is useful; "fix enemy weight default" is not.

## Related documentation

- [../PLAN.md](../PLAN.md) — the phase gates, including the Phase 10 release criteria
- [../CHANGELOG.md](../CHANGELOG.md) — the changelog itself
- [Performance and Profiling](performance.md) — what to re-measure, and on what
- [Benchmark Functions](benchmarks.md#regression-baselines) — the quality gates
