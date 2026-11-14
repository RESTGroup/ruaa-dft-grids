---
name: benchmark
description: Measure a weight, derivative, grid-build, or accelerator change and record the result in the curated performance page.
---

# Benchmark

Use this skill for a performance claim, a tuning change, or a non-CPU
prototype. Read [`../../rules/performance.md`](../../rules/performance.md)
first.

## 0. Check correctness

```bash
cargo test
cargo test --test orca_bitwise
```

Do not benchmark a change that has not passed the correctness suite. If the
change moves weights last-bit, treat it as a numerical change and validate it
first.

## 1. Build

Choose and record the target:

```bash
cargo build --release
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

The portable and host-specific builds are different claims. Say which one the
numbers use.

## 2. Run the matching example

```bash
# Partition weights: <molecule.xyz> <level> [batch] [becke|stratmann|mbecke|lko]
cargo run --release --example bench_weights -- molecule.xyz 3 4096 becke

# Partition derivatives: <molecule.xyz> <level> <grad|hess|both|combined> ...
cargo run --release --example bench_deriv -- molecule.xyz 3 grad

# Grid-native derivatives and thread scaling
cargo run --release --example bench_grid_deriv -- 4 3 3

# Random-order derivative stress case
cargo run --release --example bench_grid_permute -- molecule.xyz 3
```

Read the module comment at the top of the example for the full argument list.
Keep the geometry, level, batch, selection, and point count identical between
the baseline and the candidate.

## 3. Control the environment

* Fix `RAYON_NUM_THREADS` and state it.
* Pin the process to a CPU set when the OS supports it.
* Warm up, then repeat and interleave baseline/candidate runs.
* Report the minimum or median and say which. Label a single pass.
* Separate kernel time from transfer time for an accelerator run.

## 4. Record

Put the curated result in [`../../../docs/performance.md`](../../../docs/performance.md)
with:

* hardware and compiler version;
* target and thread count;
* workload (molecule, level, point count, scheme, batch);
* absolute time and the ratio to the baseline;
* whether the value is a minimum, median, or single pass;
* any correctness caveat (bitwise, tolerance, different algorithm).

Do not add timings, CPU names, or crossover tables to source comments or
commit messages.

## 5. Review

* [ ] Correctness suite passed before measuring.
* [ ] Same input and build settings as the baseline.
* [ ] Thread count and target stated.
* [ ] Ratios reported, not only absolute times.
* [ ] Transfer cost included for a non-CPU path.
* [ ] No benchmark fragment in code or commit text.

## Stop conditions

Stop if the measurement is not reproducible, if the baseline is noisy enough
that the difference is inside the noise, or if the optimization changes the
numerical result. Report the measurement and the uncertainty rather than
claiming a speedup.
