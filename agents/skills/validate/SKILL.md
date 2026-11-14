---
name: validate
description: Run the offline Rust/validation suite, add a reference case, or refresh a cross-code reference.
---

# Validate the crate

Use this skill for a normal correctness run, for a validation change, or when a
reference test fails. The normative rules are in
[`../../rules/validation.md`](../../rules/validation.md).

## Fast path

Run from the repository root:

```bash
cargo test
```

This is the offline core suite. It reads the checked-in references and must not
need ORCA, PySCF, Gaussian, `strace`, network access, or a local work directory.

For the binding crates:

```bash
cargo test --workspace
```

For a single reference family:

```bash
cargo test --test orca_reference
cargo test --test pyscf_reference
cargo test --test gaussian_reference
```

For derivative changes, also run the release-mode finite-difference suite and
the single-thread comparison:

```bash
cargo test --release --test partition_deriv
RAYON_NUM_THREADS=1 cargo test --release --test partition_deriv
```

## Refreshing a reference

Only do this when the external program and its version match the producer
README. The raw work directory is git-ignored; never commit it.

### ORCA

Prerequisites: ORCA 6.1.1, a job wrapper, Python 3.

```bash
export ORCARUN=/path/to/orca-wrapper.sh
export ORCA_HOME=/path/to/orca-6.1.1          # if the wrapper needs it
python3 validation/orca/reproduce.py
python3 validation/orca/reproduce_response.py # needs strace
cargo test --test orca_reference
```

Use `--only <case>` for one case, `--skip-run` to re-extract an existing work
directory, and `--clean` to start fresh. The response producer needs `strace`
because ORCA deletes its temporary response grids at normal termination.

### PySCF

```bash
python3 validation/pyscf/generate.py
cargo test --test pyscf_reference
```

### Gaussian 16

Prerequisites: Gaussian 16 and the `rwfdump` utility.

```bash
export G16RUN=/path/to/g16-job-wrapper     # receives absolute .gjf paths
export G16ROOT=/path/to/g16                # for the rwfdump utility
python3 validation/gaussian/reproduce.py
cargo test --test gaussian_reference
```

Review the generated probe diff before committing. A refresh should be
explainable by the code change or by the producer's documented behaviour.

## Updating the bitwise hashes

Only after the arithmetic change is understood and the other ORCA references
agree:

```bash
UPDATE_ORCA_HASHES=1 cargo test --test orca_bitwise -- --nocapture
```

A hash refresh is not a fix for a failing probe.

## When a test fails

1. Identify the layer: construction (`GridPlan`/atomic grid), pruning, angular
   directions, partition weights, or derivatives.
2. Compare against the reference producer's own values when available. Use
   `validation/tools/compare.py` for a manual case:
   ```bash
   mkdir -p validation/_work
   cargo run --quiet --example grid_probe -- \
       validation/orca/golden/atomic/H_ag4.txt > validation/_work/crate.txt
   python3 validation/tools/compare.py \
       validation/orca/golden/atomic/H_ag4.txt validation/_work/crate.txt
   ```
3. Check whether the failure is bitwise (arithmetic/order) or tolerance-level
   (constants/regions).
4. Do not weaken a tolerance or delete a case to get green. If the change is
   intended, document why the reference moved.

## Completion checklist

* [ ] `cargo test` passes offline.
* [ ] The relevant reference test passes.
* [ ] No raw program output or local path was added.
* [ ] A refreshed golden is reviewed, not blindly regenerated.
* [ ] Any intentional hash change is called out in the change description.
