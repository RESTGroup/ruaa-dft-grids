# Changelog

All notable user-visible changes are recorded here.  The crate version is
still pre-1.0; breaking changes are allowed in minor releases and are called
out explicitly.

## 0.1.0 (unreleased)

### Grid specification

- The spec is now `GridSpec`: a topology plus a partition.  A topology is
  either a data-driven `GridPreset`, the built-in `PyscfGridSpec`, or a
  caller-assembled `AtomicGrid` set (`GridSpec::custom`).  The three
  program-specific spec types (`OrcaGridSpec`, `PyscfGridSpec`,
  `GaussianGridSpec` as a `GridSpec` variant) were replaced by this one
  surface.
- `GridPreset`/`ElementPreset`/`AngularLayout` supply shell counts, radial
  scales, cutoffs, pruning policies and explicit per-shell angular sizes.
  The types implement `serde`; the file form uses `[[element]]`, `[[range]]`
  and `default` blocks.  An element entry can also be one inline object in a
  top-level `element` array (for example `{ z = 8, radial_scale = 0.9 }`); the
  shipped PySCF preset uses that compact spelling.  A range supplies shared
  parameters for a contiguous charge range, and element entries may override
  only the fields that differ (the missing fields are inherited).  This keeps
  a full-periodic-table preset compact.
- `AngularLayout::ProductShells(Vec<usize>)` adds a per-shell adaptive-phi
  Gauss–Legendre product rule (the construction used by ORCA `AngularGrid 0`
  for H–Ne).
- `build_molecular_grid` and `GridPlan::build` accept anything convertible
  into `GridSpec`; `PyscfGridSpec::from_level` remains the short PySCF path.
- `build_molecular_grid_from_atomic_grids` (or
  `build_molecular_grid_from_plan` with `GridSpec::custom`) is the extension
  point for caller-built atomic grids.

- `GridPreset` carries an explicit element domain (`z_min`, `z_max`,
  `with_element_range`, `supports`).  `element(z)` and `atomic_grid(z)` return
  `MissingElementData { table: "grid preset" }` outside it instead of applying
  the preset default to an uncovered charge; `GridPreset::validate` rejects an
  empty/reversed domain or an element entry outside it.  This lets a preset
  state a Z = 1–112 coverage limit.
- New `catalog` module: `PresetCatalog`, `ProgramCatalog`, `PresetEntry` and
  `PresetKind` describe a JSON index over a directory of preset files.  The
  loader is data-free: `PresetCatalog::from_json_path` parses and validates
  the index, `from_json_str` parses only, and `PresetEntry::load` reads one
  referenced JSON `GridPreset` and applies the entry's `elements` range.
  Entry paths must be relative and may not contain `..`; names and paths must
  be unique and every grid entry must carry an element range.  The loader
  refuses a preset whose name differs from the entry and a catalog range
  wider than the preset's own domain.  The crate adds `serde_json` for this
  caller-data format; it still ships no catalog or program data.
- The preset and catalog file forms reject unknown keys at every level, so a
  typo is an error instead of a silently ignored field.  A preset file carries
  `format = 1` (an absent field means 1; other versions are rejected), and
  `GridPreset::validate` rejects a `[[range]]` block outside the preset's
  element domain.
- New `python/` tooling: a dependency-free reader/validator for preset and
  catalog files that mirrors the Rust file form.  It is repository tooling,
  not part of the published crate, and shares
  `tests/data/pyscf-level-3-canonical.json` with `tests/preset_fixture.rs`.

### PySCF path

- `pyscf::angular_count` resolves a PySCF `LEBEDEV_ORDER` key (for example
  order `11` to 50 points) and the 0/1 trivial rule.  `PyscfGridSpec::atom_grid`
  now accepts both a point count and an order key for `n_ang`.
- `angular::angular_grid(1)` returns the trivial one-point rule (origin,
  weight one), and `is_supported_size(1)` is true.  PySCF's own
  `MakeAngularGrid(1)`/`gen_atomic_grids` path is not used as a reference for
  this rule because PySCF 2.15 stores that grid as a one-dimensional array.

### Removed program data

- Removed the ORCA and Gaussian parameter tables, presets, goldens and
  program-specific construction modules from the published crate.  They live
  in the separate local preset workspace and are supplied to the crate as
  `GridPreset` data.  This is a licensing/provenance decision: the core
  crate distributes only the program-independent machinery, the PySCF
  open-source tables and the mathematical rules.
- Removed `OrcaGridSpec`, `GaussianGridSpec`, `GaussianGrid`,
  `GaussianPreset`, `DefGrid`, `GridPruning`, `CosxGrid`, `W_CUT` and the
  corresponding error variants/helpers.  Exact constructions are reproduced
  by loading the preset data into `GridPreset` and applying a
  caller-supplied partition/radii configuration.
- The framework no longer defaults a partition from a program name.
  `PartitionConfig::pyscf()`, `gaussian_ss()` and `gaussian_becke()` remain
  as explicit constructors; the caller chooses one.

### Numerical changes

- Partition normalization is scale-free: the product sum is divided whenever
  it is finite and non-zero, so a symmetric cage whose unnormalized products
  are tiny now gets the uniform partition instead of a first-centre unit
  mass.  The blocked, sparse and scalar kernels share the policy.
- `GridPreset` pruning uses the exact per-shell `r/xi` ratios from the full
  nominal shell generation; truncated (cutoff-limited) grids now classify
  the retained shells correctly.
- Preset JSON files are parsed with `serde_json`'s `float_roundtrip` feature,
  so a full-precision radial scale or cutoff keeps its exact bits instead of
  shifting by one ulp.
- The Lebedev generator was replaced with the Apache-2.0
  `rstsr-sci-traits` implementation; the point order and rule values are
  unchanged.

### Documentation

- New presets page (`docs/presets.md`) walks through PySCF's
  `Grids.level = 3` preset: the built-in `PyscfGridSpec`, the same topology as
  a data-driven `GridPreset`, the complete
  `examples/presets/pyscf-level-3.toml` file (seven ranges, 103 element
  entries in the compact `element` array), a per-element override, and the
  optional PySCF cross-check.
- New `examples/user_presets.rs` and `tests/preset_api.rs` cover the TOML
  round trip, the `element` > `range` > `default` precedence, and the equality
  of the preset file with the built-in PySCF atomic grids for Z = 1–103.
- `examples/presets/check_against_pyscf.py` compares the preset's atomic grids
  with PySCF's `gen_atomic_grids` (shell sizes, radii, radial weights and the
  full point set).  It is an optional manual check, not a `cargo test` target.

### Packaging

- The PyO3 and C-ABI binding crates were moved out of the published crate.  The package now contains only the
  core crate, its tests, examples and docs.
- `NOTICE` now records the Apache-2.0 `rstsr` and PySCF attributions and
  states that ORCA/Gaussian tables and probes are not part of the package.
  PySCF's notice is reproduced verbatim, including its closing notes.
- `cargo test` is offline and self-contained.  Cross-code reference goldens
  are maintained outside the package; the optional PySCF cross-check in
  `examples/presets/` builds its reference at run time and is not part of the
  test suite.
