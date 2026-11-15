# API stability and review

The crate is at `0.1.x`. The API is not frozen, but "not frozen" is not a
licence to churn: every public change costs downstream users a review. This
page defines the compatibility expectations and the checklist for a public
change.

## What counts as public API

* items exported from `src/lib.rs` and the public modules, including
  re-exports and type aliases;
* public fields, enum variants, and trait methods;
* generated builder types and their setters (they are part of the public
  surface, not an implementation detail);
* the `Error` variants and the `Display`/`source` behaviour that callers may
  match or log;
* feature names and default features;
* the MSRV and the crate's documented semver policy.

A change to a doc comment is not an API break. A change to the meaning of a
field, even without a type change, is.

## Pre-1.0 policy

* **Patch releases** should be bug fixes and documentation. Avoid new public
  API in a patch unless it is additive and low-risk.
* **Minor releases** may break the pre-1.0 API, but each break must be
  deliberate, documented in the changelog, and reflected in the migration
  notes.
* **Do not rely on `0.x` as permission for silent breaks.** If a rename or a
  semantic change is necessary, say so in the release notes and update the
  examples and docs in the same change.

## `#[non_exhaustive]`

Public enums that dispatch over an extensible set are the main forward
compatibility problem: adding a variant is a breaking change for downstream
code that matches exhaustively. `GridSpec`, `PartitionScheme`, `Radial`,
`Prune`, `RadiiTable`, `RadiiAdjust`, `PostPrune`, `PruneWarning`,
`HeavyFallback` and `Error` are already `#[non_exhaustive]`.  An enum that is
genuinely closed is documented as closed and stays exhaustively matchable. When adding a public enum:

* Decide whether it is genuinely closed. If it is expected to grow, mark it
  `#[non_exhaustive]` from the first release.
* Keep exhaustive matching for enums that are genuinely closed, and document
  why they are closed.
* Update the C ABI's unknown-variant branch when a non-exhaustive enum reaches
  that layer (`spec_kind` returns `-1`, setters return `UNSUPPORTED`).
* Do not add `#[non_exhaustive]` to a struct merely to avoid committing to its
  fields. Prefer private fields plus accessors for a type whose invariants
  matter.

Adding a variant to a non-exhaustive enum still needs a doc comment, a test,
and a note in the changelog. Downstream code is allowed to ignore it; the
crate is not.

## Adding a public item

1. Decide whether the item is consumer API or maintainer API. If only a
   backend or a test needs it, keep it `pub(crate)` or put it behind a
   documented extension trait.
2. Write the doc comment first. It must state units, conventions, errors, and
   whether the item is unstable.
3. Add a runnable doc example for a new entry point, or a test that exercises
   the public path.
4. Check the name against the existing vocabulary (Bohr, quadrature weight,
   partition fraction, parent atom, generating atom). Consistency is part of
   the API.
5. Add it to the module map or the relevant `docs/` page if a consumer would
   look for it there.
6. Run the gate in [`../skills/ci/SKILL.md`](../skills/ci/SKILL.md) and the
   package check in [`../skills/release/SKILL.md`](../skills/release/SKILL.md).

## Deprecation

* Prefer a deprecation with a working replacement over a silent removal.
* The deprecation message must name the replacement and, if relevant, the
  version in which the item will be removed.
* Keep the deprecated item working for at least one minor release.
* Update the examples, docs, and internal call sites in the same change, so the
  deprecation warning is visible only to downstream users.

## Feature flags

* New features must be additive and must not change the default build's
  behaviour.
* A feature that removes a dependency (for example a serial build without
  Rayon) must still pass the offline tests and the bitwise goldens.
* Do not make a feature part of a public type signature unless it is a
  deliberate, documented API decision.

## Review checklist for a public change

* [ ] The change is additive, or the break is documented and justified.
* [ ] New enum variants are on a non-exhaustive enum, or the enum is
      documented as closed.
* [ ] Builder changes preserve existing setter names, or the rename is in the
      changelog.
* [ ] Error variants have a clear message and a test for the new path.
* [ ] The doc comment states units, conventions, and errors.
* [ ] `cargo test`, `cargo clippy --all-targets -- -D warnings`,
      `cargo fmt --check`, and the docs build pass.
* [ ] `cargo package --list` still excludes repository-process material.
