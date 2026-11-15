# Validation rules

The crate's correctness assets are the offline tests under `tests/` and the
optional cross-code checks under `examples/`.  Program-specific goldens,
producer scripts and reference trees live in the maintainer's local workspace
and are not part of this repository.

## Reference data is data

A checked-in fixture is validation, not clutter: do not delete rows, round
values, or drop a case because a test is inconvenient.  Review a fixture diff
like a code change and understand why the numbers moved.

## Offline tests

`cargo test` must remain offline:

* it must not require ORCA, PySCF, Gaussian, `strace`, network access, or a
  local scratch directory;
* it must read the checked-in data, not regenerate it;
* it must not silently skip a case when an external program is absent.

The optional PySCF cross-check (`examples/presets/check_against_pyscf.py`) is
manual, needs PySCF and a Rust toolchain, and is never part of `cargo test`.

## Tolerances

Do not weaken a tolerance to make a failing test pass.  First determine whether
the difference is a real defect, an intended arithmetic change, or a changed
reference.  An intended change updates the tolerance and the changelog in the
same commit, and the reason is stated in the commit message.

## Adding a case

* Prefer a mathematical invariant or a deterministic construction in `tests/`
  over another golden table.
* Keep the optional PySCF check in step with the shipped preset file: it
  builds the reference at run time and compares shell sizes, radii, radial
  weights and the full point set.
* A new preset-file field needs a case in `tests/preset_api.rs`,
  `tests/preset_catalog.rs` and the Python tooling under `python/tests/`;
  update the shared canonical fixture with `examples/preset_dump.rs`.
* A new program-specific reference belongs in the local workspace, described
  neutrally in the package ("checked against the reference construction
  maintained outside this package").

## What validation does not prove

The comparisons are at the grid level.  They do not prove that an SCF energy
is converged, that a functional is integrated accurately, or that a consumer's
basis and density are handled correctly.  Keep validation claims factual and
brief.
