# Performance and measurement rules

The crate has measured performance work in `docs/performance.md` and runnable
benchmark examples under `examples/`. This page defines how to measure a
change and where the numbers go.

## Where numbers live

* **Curated results:** [`../../docs/performance.md`](../../docs/performance.md).
  It records the hardware, compiler, thread count, workload, and whether a
  value is a minimum, median, or single pass.
* **Raw worklogs and experiment output:** outside the repository. Do not check
  them in.
* **Code comments:** no timings, CPU names, crossover tables, or measured
  brackets. At most a one-line empirical rationale is acceptable when a
  constant is tuned; the measurement itself goes in the curated page.

## Correctness first

Never benchmark a change that has not passed the correctness suite. For a
numerical change:

1. Run `cargo test` and the relevant validation test.
2. Confirm the bitwise ORCA hashes and the PySCF/Gaussian references still
   pass, or that the intended diff is understood.
3. Only then measure.

An optimization that changes weights last-bit is a correctness change, not a
pure performance change. Treat it accordingly.

## Measurement protocol

Use [`../skills/benchmark/SKILL.md`](../skills/benchmark/SKILL.md) for the
build targets, the example commands, and the repetition protocol. The policy is
to state the target and thread count, keep the baseline and candidate inputs
identical, report ratios and uncertainty as well as absolute times, and never
compare a debug build with a release build or a full-grid run with a
partial-range run.

## Review checklist for a performance change

* [ ] The correctness suite passes before and after.
* [ ] The measurement uses the same input and build settings as the baseline.
* [ ] The claim states hardware, compiler, thread count, and target.
* [ ] The result is in `docs/performance.md`, not in a code comment.
* [ ] The change does not add a benchmark fragment to a commit message.
* [ ] If the change is a new backend or a non-CPU path, the CPU path is still
      the reference and the parity test is included.

## Non-CPU measurements

For an accelerator path, report the CPU baseline and the accelerator result on
the same case set, with the transfer cost included. A kernel-only timing that
excludes host-device transfers is not a user-visible speedup. State the
precision (f64 or f32) and whether the result is bitwise equal, within
tolerance, or a different algorithm.
