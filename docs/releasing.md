# Releasing Dragonfly

This module uses Go module versioning and Semantic Versioning. Release tags have the form
`vMAJOR.MINOR.PATCH`, and the module stays in the unstable `v0` series until its public API is
declared stable.

Go modules are published from repository tags rather than uploaded anywhere. Pushing a tag and
fetching it once through the public proxy is the whole publication step.

Use only `github.com/CWBudde/dragonfly`, with the lowercase repository component. The
capitalized `github.com/CWBudde/Dragonfly@v0.1.0` path was cached before the repository rename;
module paths are case-sensitive, so it is a distinct obsolete module and receives no updates.

## Version policy

- Increment **PATCH** for backward-compatible fixes and documentation changes.
- Increment **MINOR** for backward-compatible features. While the module is at `v0`, use a minor
  release for any intentional public API break as well, and call it out prominently in the
  changelog.
- Use a prerelease suffix such as `v0.2.0-rc.1` for release candidates.
- From `v1.0.0` onwards, increment **MAJOR** for breaking changes. A future v2 must also change
  the module path to `github.com/CWBudde/dragonfly/v2`.
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
- **Changing a verified reference constant or archive policy** — including BDA's `±6` clamp,
  MODA's archive size 100, or which named archive policy is the default. Call it out under its
  own changelog heading, not as a bullet among fixes.

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
   `GOPROXY=proxy.golang.org go list -m github.com/CWBudde/dragonfly@vMAJOR.MINOR.PATCH`
10. Verify the version and the rendered package documentation at
    <https://pkg.go.dev/github.com/CWBudde/dragonfly>.

## What `just release-check` verifies

```sh
just release-check 0.2.0
```

- the argument is a valid semantic version
- `CHANGELOG.md` contains the matching version section
- `LICENSE` and `README.md` are non-empty
- `go list -m` reports exactly `github.com/CWBudde/dragonfly`
- `go vet ./...`
- exact pinned formatter, linter and scanner versions
- verified and tidy modules, formatting and golangci-lint
- the short race suite and the complete suite through the 80% coverage gate
- every nested example module and both native and js/wasm demo builds
- Nancy's production dependency audit and govulncheck's reachable-code scan

`just release version=0.2.0` runs all of that, then additionally requires a clean worktree and
an unused tag name before creating the annotated tag. It does not push.

## Security gate details

`just release-check`, `just ci` and `just ci-race` all require `just security` to pass. It is
two scans, and they cover different things:

- **`just audit`.** This pipes `go list -json -deps ./...` into `nancy sleuth`, which sees only the
  production build — and because the library is stdlib-only, that build has no third-party
  packages at all, so nancy reports `Audited Dependencies: 0`. That is the intended state
  rather than a passing scan, and the recipe exists to catch the first real dependency that
  is ever added.
- **`just vuln`.** This runs `govulncheck ./...`, which does cover the test-only
  dependency tree and reports by reachability, so it is the one that would actually find
  something today.

Note that `govulncheck` also reports vulnerabilities in the Go toolchain the scan runs on.
Those are a property of the local Go installation, not of this module; check whether a finding
names `stdlib` or `os@go1.x` before treating it as a release blocker.

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
analysis, race and full coverage tests, examples/WASM and security under Go 1.26. It does
**not** create tags or GitHub releases; those stay deliberate maintainer actions.

`.github/workflows/test.yml` is the ordinary CI: format, lint, test on a Go 1.23 and 1.26
matrix, examples/WASM, security and benchmarks. `.github/workflows/security.yml` reruns the
pinned Nancy 2.1.0 and govulncheck 1.1.4 scans weekly and on manual dispatch under Go 1.26.

The rest of the pinned toolchain is treefmt 2.5.0, golangci-lint 2.13.1, gofumpt 0.11.0,
gci 0.14.0, shfmt 3.13.1, Taplo 0.10.0, Prettier 3.9.6 and ShellCheck 0.11.0. `just
check-tools` verifies the installed versions instead of accepting any same-named binary.

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
