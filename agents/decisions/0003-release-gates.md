# 0003: Release gates

- **Status:** proposed
- **Date:** 2026-11-13
- **Deciders:** maintainers

## Context

The tracked tree passed the local Rust gates when this decision was recorded.
The project is close to a first public release, but several publication
concerns are not covered by `cargo test`:

* the git history predates the current content-hygiene policy and may contain
  material that is no longer in the working tree;
* the core CI workflow is new; it does not yet cover the binding crates or the
  declared MSRV;
* the repository URL is still a TODO in `Cargo.toml`;
* the redistribution terms for bundled reference data and parameter tables
  need a recorded review;
* the public API has not had a final pre-0.1 pass;
* the core crate and the binding crates have different packaging and release
  readiness.

Publishing the current tree with the current history would expose more than the
current files. The history is part of the published repository.

## Decision

The first public release is gated on all of the following.

### 1. History review

Review the complete history that will be published, including commit messages
and deleted files. If it contains material excluded by
[`../rules/provenance.md`](../rules/provenance.md), choose one of:

* publish from a fresh, clean history (for example an orphan initial commit or
  a squashed release commit), keeping the development history private;
* rewrite the history to remove the excluded material;
* do not publish the history.

This is a maintainer decision. An agent must not rewrite or discard history
unilaterally. Archive the development history outside the published repository
before any rewrite.

### 2. Provenance and licence review

Record the redistribution decision for bundled parameter tables and reference
data in a decision record. A public Apache-2.0 licence on the code does not by
itself settle the terms of third-party data. Do not publish a table whose
terms are unresolved.

### 3. CI

The core workflow in `.github/workflows/ci.yml` must stay green and run, at
minimum:

* `cargo fmt --check`;
* `cargo clippy --all-targets -- -D warnings`;
* `cargo test` (offline);
* the docs build with warnings denied;
* `cargo package --list` (or an equivalent package audit).

The core test job must not require ORCA, PySCF, Gaussian, `strace`, or network
access. Binding jobs may be separate. See
[`../skills/ci/SKILL.md`](../skills/ci/SKILL.md). Add the binding and MSRV jobs
before the first release that claims support for them.

### 4. Package audit

`cargo package --list` must exclude `agents/`, `AGENTS.md`, local
configuration, scratch directories, raw output, and build artifacts. The
validation probes stay in because the comparison tests need them. The package
size and file count are reviewed, and `cargo publish --dry-run` succeeds.

### 5. API audit

Before 0.1:

* audit public enums and add `#[non_exhaustive]` where future variants are
  expected;
* remove or implement dead public variants;
* check the error variants, builder types, and public fields against the
  pre-1.0 policy in [`../rules/api-stability.md`](../rules/api-stability.md);
* keep the repository URL and the rest of the publish metadata current.

### 6. Documentation

* `cargo doc --no-deps` with warnings denied;
* README and `docs/` links resolve;
* the validation and performance pages describe the current state;
* no stale test counts, file paths, or "not yet implemented" claims that are
  no longer true.

### 7. Release record

Record the release version, tag, and the human approval to publish. The
release skill is in [`../skills/release/SKILL.md`](../skills/release/SKILL.md).

## Consequences

* The first release is blocked until the history and data-licence decisions are
  made. Those are maintainer decisions, not coding tasks.
* A fresh-history publication is the lowest-risk route if the development
  history contains excluded material and there is no need to preserve public
  commit history.
* The CI and package gates are repeatable and can run before the legal and
  history decisions are complete.
* The binding crates can follow the core release; they do not need to block
  it, but they need their own packaging checks before they are published.

## Alternatives considered

* **Publish the current tree and history as-is.** Rejected while the history
  review is open; a clean working tree does not clean old commits.
* **Rewrite history immediately.** Deferred to the maintainer; it is
  destructive and needs a private archive of the current history.
* **Publish the bindings with the core crate.** Deferred; the binding
  toolchains and packaging have a different readiness bar.

## Status update 2026-11-14

- Decision [`0004`](0004-program-data-externalization.md) supersedes the
  bundled-data part of this record: ORCA/Gaussian parameter tables,
  reference probes, producer scripts and program-specific construction code
  are no longer part of the published package.  `validation/` and the
  Python/C binding crates were removed from the package.
- The package audit now rejects any `validation/`, `crates/`, ORCA or
  Gaussian file; the documented PySCF tables carry the PySCF notice.
- The Lebedev generator was replaced with the Apache-2.0 `rstsr` source and
  the NOTICE was updated.

## Status update 2026-11-15

- Decision [`0005`](0005-defer-python-release.md) narrows the first release
  to the Rust crate.  The Python preset reader is repository tooling; its
  distribution is deferred and does not block `0.1.x`.
- Release automation is configured in
  [`.github/workflows/release-plz.yml`](../../.github/workflows/release-plz.yml)
  and
  [`.github/workflows/release-plz-pr.yml`](../../.github/workflows/release-plz-pr.yml).
  Both workflows are manual-dispatch only.  A public repository and the
  `CARGO_REGISTRY_TOKEN` secret remain prerequisites for the publish step.
