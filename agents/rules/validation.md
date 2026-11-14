# Validation rules

The validation reference is the project's main correctness asset. This page
defines what may change and how to review a change. The format and case
matrices are documented in [`../../validation/README.md`](../../validation/README.md)
and the per-program READMEs.

## Golden data is data

The checked-in probes under `validation/` are measured reference data, not
clutter. Do not delete rows, round values, or remove a case because a test is
inconvenient. A golden diff is reviewed like a code change: understand why the
numbers moved before accepting them.

## Offline tests

`cargo test` must remain offline:

* it must not require ORCA, PySCF, Gaussian, `strace`, network access, or a
  local scratch directory;
* it must read the checked-in references, not regenerate them;
* it must not silently skip a case when an external program is absent.

The reference producers under `validation/{orca,pyscf,gaussian}/` are for
refreshing a reference, not for the normal test path.

## Tolerances

Do not weaken a tolerance to make a failing test pass. First determine whether
the difference is:

* an intended arithmetic or ordering change;
* a wrong constant, region boundary, angular size, or partition mapping;
* a legitimate last-bit difference from a different evaluation order.

Only the first and third justify a tolerance discussion, and the reason must be
recorded in the change description. The bitwise ORCA tests are stricter than
the probe tolerances on purpose; a last-bit change fails unless the hash is
intentionally refreshed.

## Running the suite

Use [`../skills/validate/SKILL.md`](../skills/validate/SKILL.md) for the quick
path, the reference-refresh commands, the single-thread derivative comparison,
and the ORCA hash-update procedure. This page only defines what may change.

## Refreshing a reference

A refresh is a maintainer-side action. Confirm that the producer version
matches its README, run it with the documented environment, review the probe
diff, and commit the probe with the code/test change. The raw work directory is
git-ignored and stays out of the commit. A refresh that is not explainable by
the code change or the producer's documented behaviour is a failure, not a
routine update.

The ORCA response producer needs `strace`; that is a maintainer-side tool, not
a CI dependency.

## Updating the ORCA hashes

Use the opt-in command in the validation skill only when the arithmetic change
is intentional and the other ORCA references still agree. A hash refresh is not
a substitute for understanding a probe failure.

## Adding a case or a new producer

* Add the case to the producer's input/case list, not to a duplicate table in
  Rust. The reference file is self-contained by design.
* Add only compact reference records. Do not check in full point tables or raw
  program output.
* Add a test to the matching `tests/<software>_reference.rs`, or reuse the
  shared harness with a new filter.
* For a new producer, follow the procedure in
  [`../../validation/README.md`](../../validation/README.md): README, producer,
  format extension if needed, test, and a note on the limits.
* A new scheme with no external reference needs an analytic or
  finite-difference test and a clear statement in the docs that it is not
  cross-code validated.

## What validation does not prove

The comparisons are at the grid level. They do not prove that an SCF energy is
converged, that a functional is integrated accurately, or that a consumer's
basis and density are handled correctly. Keep the README wording factual and
avoid turning a grid match into a broader claim.
