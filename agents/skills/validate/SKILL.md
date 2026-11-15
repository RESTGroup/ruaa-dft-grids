---
name: validate
description: Run the offline Rust and Python preset suites, the optional PySCF cross-check, and the packaging gate.
---

# Validate the crate

Use this skill for a normal correctness run, for a preset-file change, or when
a test fails.  The normative rules are in
[`../../rules/validation.md`](../../rules/validation.md).

## Fast path

Run from the repository root:

```bash
cargo fmt --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --locked
RUSTDOCFLAGS="--html-in-header katex-header.html -D warnings" cargo doc --no-deps --locked
```

`cargo test` is the offline core suite (Rust unit, integration and doctest
targets).  It must not need ORCA, PySCF, Gaussian, `strace`, network access or
a local work directory.

The Python preset tooling has its own dependency-free suite:

```bash
python -m unittest discover -s python/tests -v
```

## Optional cross-code check

The shipped PySCF level-3 preset can be compared with PySCF's own construction.
It needs PySCF, NumPy and a Rust toolchain:

```bash
python3 examples/presets/check_against_pyscf.py --elements 1,8,26 --tolerance 1e-11
```

## Preset-file changes

The canonical JSON mapping is shared with the Python tooling.  After changing a
preset file or its serializer, regenerate the fixture and run both sides:

```bash
cargo run --example preset_dump -- examples/presets/pyscf-level-3.toml \
  > tests/data/pyscf-level-3-canonical.json
cargo test --locked --test preset_fixture
python -m unittest discover -s python/tests -v
```

## Derivative changes

```bash
cargo test --release --test partition_deriv
RAYON_NUM_THREADS=1 cargo test --release --test partition_deriv
```

## Packaging

With a clean working tree:

```bash
cargo package --list --locked
cargo test --manifest-path target/package/ruaa-dft-grids-0.1.0/Cargo.toml --locked
```

The package must not contain `agents/`, `python/`, `.github/`, local
configuration or program data.  `CI` runs the same checks; see
[`../ci/SKILL.md`](../ci/SKILL.md).

## Result

Success means every command above exits zero and the optional PySCF check
prints `all checks passed`.  A skipped reference family or a local work
directory is not a substitute for the offline suite.
