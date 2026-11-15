# 0005: Release the Rust crate before the Python tooling

- **Status:** accepted
- **Date:** 2026-11-15
- **Deciders:** maintainers

## Context

The repository contains one publishable Rust crate, `ruaa-dft-grids`, and a
repository-only Python preset reader under `python/`.  The Rust crate has
passed its pre-release audit and is ready for a first crates.io release.  The
Python reader is useful to maintainers and preset authors, but it has no
independent packaging pipeline, wheel matrix, versioning policy, or PyPI
project.

Releasing both together would add packaging and support work that the
numerical library does not depend on.  It would also blur what a `0.1.x`
release means: the Rust crate and the Python reader have different audiences
and different validation paths.

## Decision

1. The `0.1.x` release artifact is the Rust crate `ruaa-dft-grids` only.  It is
   published on crates.io; the repository tag and GitHub release describe that
   crate.
2. The `python/` preset reader remains repository tooling in this release
   cycle.  It is not published to PyPI, no wheel or sdist is built, and it
   carries no release version separate from the repository.
3. `release-plz` manages only the Cargo package in `Cargo.toml`.  It never
   publishes the Python directory, and `python/` remains excluded from the
   crate package.
4. A Python distribution is deferred until it has its own packaging checks,
   versioning and support policy.  Deferring it is not a blocker for `0.1.x`.
5. The binding crates remain outside this repository (decision
   [`0004`](0004-program-data-externalization.md)); their release is separate
   as well.

## Consequences

- `cargo publish` and the GitHub release contain only the Rust crate.
- The Python tests continue to run in CI, but a Python failure does not change
  the crate's release version or artifact.
- Maintainers must not describe the repository as shipping a Python package
  in `0.1.x` release notes.
- Revisit this decision when the Python reader gets an installable,
  independently tested distribution.

## Alternatives considered

- **Publish both now.**  Rejected: the Python reader has no release pipeline
  and would need PyPI trusted publishing, a wheel/sdist policy and its own
  support commitment.
- **Move `python/` out of the repository.**  Deferred: it shares the canonical
  preset fixture and is useful alongside the crate.
