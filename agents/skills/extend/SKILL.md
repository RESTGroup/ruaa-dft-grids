---
name: extend
description: Add a radial map, pruning policy, radii table, partition scheme, angular rule, or program backend without destabilizing the validated numerics.
---

# Extend the crate

Use this skill for a new construction or backend. It is the working recipe;
the review rules are in [`../../rules/api-stability.md`](../../rules/api-stability.md)
and [`../../rules/validation.md`](../../rules/validation.md).

## 1. Classify and scope

Identify the change:

| Change | Primary files | Reference needed |
|---|---|---|
| radial map | `src/radial/` (`RadialMap`, `RadialSpec`) | nodes/weights or moments |
| pruning policy | `src/prune/` (`Pruning`, `PruneContext`), backend policy module | region boundaries and sizes |
| radii table | `src/radii.rs`, `src/params.rs` | table plus element range |
| partition scheme | `src/partition/` (`SchemeOps`, `DerivScheme`) | weights, FD derivatives |
| backend preset | `src/grid.rs`, new backend module | point/weight probe |
| angular rule | `src/angular.rs`, `src/lebedev.rs` | exact directions and moments |
| ordering policy | `src/order.rs` | permutation invariants |

If the change is a new non-CPU execution path, stop and read
[`../../development.md`](../../development.md) section 4. Do not start by
adding a device generic to the core.

## 2. Protect the baseline

```bash
cargo test
cargo test --test orca_bitwise
```

If anything fails before your edit, stop and report it. Do not build on a red
baseline.

## 3. Add the reference test first

* For an external reference, add the producer case and the checked-in probe
  following `validation/README.md`.
* For an analytic construction, add a node/weight/moment or FD test.
* For a new partition scheme, add a fraction-sum test and a finite-difference
  derivative test where differentiable.
* For a new backend, add a point/weight comparison, not only a point count.

The test should fail for the missing implementation, not because of an
unrelated tolerance.

## 4. Implement the smallest change

* Add the enum variant or spec field with a doc comment.
* Keep the existing conventions: Bohr, sum-to-one angular weights,
  `4*pi*r^2*dr` radial measure, normalized partition fractions, explicit
  post-build pruning.
* Update every `match` the compiler reports, and decide each arm deliberately.
  Do not add a catch-all that silently maps the new variant to an old scheme.
* For a radial map, implement `RadialMap` and pair it with a
  `RadialScalePolicy` in `RadialSpec`; do not force a new map to define an
  element-scale rule.
* For a pruning policy, implement `Pruning` and consume `PruneContext`; the
  built-in `Prune` and `GridPruning` selectors are implementations, not
  separate contracts.
* For a partition scheme, add the `PartitionScheme` variant and a private
  `SchemeOps` implementation (plus the `ops_for` arm); implement `DerivScheme`
  or return a clear `UnsupportedDerivative`. Do not silently fall back to
  finite differences.
* For a backend, update `GridSpec::{partition, atomic_plan, atomic_plan_at,
  geometry_dependent}`, the `From` impls, and the C ABI's unknown-backend
  branch. `GridSpec` is `#[non_exhaustive]`.

## 5. Update the documentation

* `docs/algorithms.md` for a construction or formula.
* `docs/user-guide.md` for a consumer-facing selector.
* `docs/scheme-variants.md` for a partition scheme.
* `validation/` README for a new producer or case family.
* `agents/development.md` only if the architecture or extension procedure
  changed.

## 6. Validate and package

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo test --test <the-new-reference-test>
RUSTDOCFLAGS="--html-in-header katex-header.html -D warnings" cargo doc --no-deps
cargo package --list
```

For a performance-sensitive change, follow
[`../benchmark/SKILL.md`](../benchmark/SKILL.md). Do not put the measurement in
a code comment.

## 7. Stop conditions

Stop and ask the maintainer if:

* the new construction needs a parameter table whose redistribution is unclear;
* the change would rename or remove an existing public path;
* the change needs a new dependency or a feature that changes the default
  build;
* the change would add a backend boundary or a device abstraction;
* the reference differs in a way that is not explained by the construction.

## Completion checklist

* [ ] The new test fails without the change and passes with it.
* [ ] Existing goldens are unchanged or the diff is explained.
* [ ] Public API is documented and non-exhaustive where it should be.
* [ ] The relevant `docs/` page is updated.
* [ ] `cargo package --list` still excludes `agents/` and local material.
* [ ] `agents/development.md` is updated if the architecture changed.
