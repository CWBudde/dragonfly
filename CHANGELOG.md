# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Repository scaffold: `go.mod` (`github.com/MeKo-Christian/dragonfly`), MIT license,
  `.gitignore`, `justfile`, `.golangci.toml`, `treefmt.toml`, and the format/lint/test and
  release-validation GitHub workflows, all mirroring the sibling Mayfly project so the two
  libraries share one set of conventions.
- `PLAN.md`: the phased implementation roadmap, including the algorithm specification the
  implementation is written against — the five swarming primitives, the two-branch step
  update from the reference implementation, the adaptive weight schedules, Lévy flight, the
  BDA transfer-function family, and MODA's hypercube-based Pareto archive.
