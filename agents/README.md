# Agent and maintainer material

This directory is the repository home for operational knowledge that is not
part of the library API, the user guide, or the numerical algorithm pages. The
repository is developed inside a larger local workspace; only the repository is
published, so process knowledge kept outside it would be lost.

## Read order

1. [`../AGENTS.md`](../AGENTS.md) — the normative repository rules. Read it
   before changing anything; this directory supplements it and never overrides
   it.
2. This file — the map, the boundaries, and the publication rules for
   `agents/`.
3. [`development.md`](development.md) — the architecture and maintenance
   sketch, including the non-CPU and extension strategy.
4. [`rules/`](rules/) — review procedures that apply to every change.
5. [`skills/`](skills/) — repeatable workflows for validation, CI, release,
   extension, and benchmarking.
6. [`decisions/`](decisions/) — accepted or proposed decisions and their
   consequences.

## What belongs where

| Material | Home |
|---|---|
| Public API, formulas, user guide, test fixtures | `docs/`, rustdoc, `tests/` |
| Agent procedures, maintainer rules, release gates, architecture decisions | `agents/` |
| Machine-specific notes, raw reference output, unpublished provenance notes | not tracked in this repository |

`agents/` is deliberately not a second copy of the user documentation. If a
fact is useful to a consumer of the crate, it belongs in `docs/` or a
docstring. If it is useful to someone changing or releasing the crate, it
belongs here.

## Publication boundary

* `agents/` is tracked and visible in the repository.
* It is excluded from the crates.io package through the `exclude` list in
  `Cargo.toml`; the root `AGENTS.md` is excluded for the same reason. The
  directory is therefore not rendered on docs.rs and is not part of the
  published crate.
* Links in this directory are repository-relative. Do not add links to local
  files, local reports, or absolute paths.

## Rules for adding material

* **No local paths.** Repository-relative paths are fine; machine-specific
  paths, scratch directories, and untracked configuration names are not.
* **No provenance narratives.** Do not describe how compatibility data was
  obtained. State behaviour and formulas factually; see
  [`rules/provenance.md`](rules/provenance.md).
* **No benchmark fragments.** Put measured tables in
  [`docs/performance.md`](../docs/performance.md), never in code comments or
  skill files. See [`rules/performance.md`](rules/performance.md).
* **Link, do not duplicate.** Prefer a link to the user guide, validation
  README, or a decision record over repeating the same instructions.
* **Keep skills runnable.** A skill must name its prerequisites, the commands
  it runs, and the result that means success.
* **Decisions are append-only.** Supersede a decision with a new record
  instead of rewriting its history. Mark the old one `superseded by NNNN`.

## Index

### Rules

* [`rules/provenance.md`](rules/provenance.md) — tracked-content hygiene and
  the release-time history check.
* [`rules/api-stability.md`](rules/api-stability.md) — pre-1.0 compatibility,
  `#[non_exhaustive]`, and public-API review.
* [`rules/validation.md`](rules/validation.md) — golden data, tolerances, and
  reference refresh.
* [`rules/performance.md`](rules/performance.md) — measurement and
  optimization rules.

### Skills

* [`skills/validate/SKILL.md`](skills/validate/SKILL.md) — run the offline
  suite and refresh a cross-code reference.
* [`skills/ci/SKILL.md`](skills/ci/SKILL.md) — add or maintain the CI gate.
* [`skills/release/SKILL.md`](skills/release/SKILL.md) — prepare and publish
  a release (the publish command is human-only).
* [`skills/extend/SKILL.md`](skills/extend/SKILL.md) — recipes for a new
  radial map, pruning scheme, radii table, partition scheme, or backend.
* [`skills/benchmark/SKILL.md`](skills/benchmark/SKILL.md) — measure a change
  and record it in the curated performance page.

### Decisions

* [`decisions/0001-record-agent-material-in-the-repository.md`](decisions/0001-record-agent-material-in-the-repository.md)
  — why this directory exists and what is excluded.
* [`decisions/0002-extension-and-accelerator-strategy.md`](decisions/0002-extension-and-accelerator-strategy.md)
  — no big-bang refactor before 0.1; how non-CPU backends fit.
* [`decisions/0003-release-gates.md`](decisions/0003-release-gates.md) — what
  must be true before the first public release.
* [`decisions/0004-program-data-externalization.md`](decisions/0004-program-data-externalization.md)
  — why program data and the binding crates live outside the package.

## Current status

The current tree passes the local Rust and Python gates (`cargo fmt --check`,
`cargo clippy --all-targets -- -D warnings`, `cargo test`,
`cargo doc --no-deps`, and `python -m unittest discover -s python/tests`).
The workflow in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs
core, Python-preset-tooling, MSRV 1.87, package verification and a manual
publish dry-run.  The repository URL is not yet live, and the publishable
history was replaced by a fresh initial import; the remaining release gates are
in [`decisions/0003-release-gates.md`](decisions/0003-release-gates.md).
