# 0001: Record agent and maintainer material in the repository

- **Status:** accepted
- **Date:** 2026-11-13
- **Deciders:** maintainers

## Context

The repository is developed inside a larger local workspace. The workspace
contains scratch probes, raw output, and maintainer notes that are not part of
the repository and will not be published. Process knowledge that currently
lives only in that workspace - how to validate, how to release, what the
extension boundaries are - would be lost when the repository is published on
its own.

The existing `AGENTS.md` is a compact policy file. It is the right entry point,
but it is not a good home for runnable procedures, long-form architecture
notes, or release gates. The user-facing `docs/` tree is also the wrong home:
it is rendered on docs.rs and is written for consumers of the crate.

## Decision

Add a tracked `agents/` directory containing:

* a `README.md` that maps the directory and states its publication boundary;
* `rules/` for review procedures that apply to changes;
* `skills/` for repeatable workflows (validation, CI, release, extension,
  benchmark);
* `decisions/` for architecture and process decisions;
* `development.md` for the architecture, extension seams, non-CPU strategy,
  refactor stance, and roadmap.

The root `AGENTS.md` remains the normative rule source. `agents/` supplements
it and must not contradict it. `agents/` is tracked in the repository but
excluded from the crates.io package through `Cargo.toml`; it is not part of
the public library documentation.

## Consequences

* Process knowledge survives the workspace and is visible to future
  contributors.
* The published crate stays lean, but the repository gains a maintenance
  surface that must be kept current.
* The directory does not replace `docs/`: consumer-facing facts still belong
  in `docs/` or a docstring.
* The root `AGENTS.md` must point at `agents/README.md`, otherwise agents that
  only read the root file will miss the procedures.

## Alternatives considered

* **Keep everything in the root `AGENTS.md`.** Rejected: it would become a
  long, mixed document and would still not hold runnable workflows well.
* **Put the material in `docs/`.** Rejected: it would be rendered on docs.rs
  and would mix maintainer process with consumer documentation.
* **Keep it in the local workspace.** Rejected: it would be lost on
  publication, which is the problem this decision addresses.
