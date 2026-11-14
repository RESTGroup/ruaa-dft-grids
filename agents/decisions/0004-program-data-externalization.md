# 0004: Program data live outside the published crate

- **Status:** accepted
- **Date:** 2026-11-14
- **Deciders:** maintainers

## Context

The 0.1.0 release candidate bundled ORCA and Gaussian parameter tables, their
reference probes and the code that consumed those tables.  The redistribution
terms of that material are not settled, so a published Apache-2.0 package
cannot ship it.  The crate also embedded in-tree PyO3 and C-ABI binding
crates, which duplicated the same program-specific surface.

The numerical constructions are useful independently of the data: radial
node maps, angular rules, pruning policies and partition weights are
mathematical machinery.  A caller who has the shell/scale/cutoff data can
still reproduce an exact grid.

## Decision

1. The published crate is program-agnostic.  A topology is a data-driven
   `GridPreset` (shell counts, radial scales, cutoffs and angular layouts) or
   the built-in `PyscfGridSpec`.
2. ORCA/Gaussian parameter tables, reference probes, producer scripts and
   program-specific construction code are maintained in a separate local
   workspace and are not part of the package.
3. Program presets are supplied to the crate as `GridPreset` data.  The
   partition and radii configuration are supplied separately by the caller.
4. PySCF's Apache-2.0 tables and formulas remain in the crate with the PySCF
   notice reproduced in `NOTICE`.  The Lebedev generator is adapted from the
   Apache-2.0 `rstsr-sci-traits` implementation and is credited in `NOTICE`.
5. The in-tree Python/C binding crates are removed from the published
   package; they are archived in the local workspace until they are ported to
   the new specification.

## Consequences

- The crate ships no ORCA/Gaussian parameter data, goldens or producer
  scripts; `cargo test` is offline and self-contained.
- A user who needs an exact program construction loads a `GridPreset` file
  and supplies the corresponding partition/radii configuration.
- The local workspace is the place for program checks, parameter data and
  goldens; it is not published and is not referenced from tracked content.
- The public API changes: `OrcaGridSpec`, `GaussianGridSpec`, `GaussianGrid`,
  `GaussianPreset`, `DefGrid`, `GridPruning`, `CosxGrid`, `W_CUT` and the
  associated error/helper surface are gone.  This is a pre-1.0 breaking
  change.
