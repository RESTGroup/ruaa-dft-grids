# ruaa_dft_grids

[![Crates.io](https://img.shields.io/crates/v/ruaa-dft-grids.svg)](https://crates.io/crates/ruaa-dft-grids)
[![docs.rs](https://docs.rs/ruaa-dft-grids/badge.svg)](https://docs.rs/ruaa-dft-grids)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/RESTGroup/ruaa-dft-grids/blob/main/LICENSE)

Molecular DFT numerical-integration grids for Rust.  The crate generates
atom-centred radial and angular quadratures, applies angular pruning,
assembles molecular grids, and evaluates Becke-type partition weights with
analytic gradients and Hessians.

The crate is program-agnostic.  A grid is described by a `GridSpec`: an atomic
topology plus a partition configuration.  The topology is either a
**data-driven preset** — shell counts, radial scales, cutoffs and per-shell
angular sizes supplied by the caller or read from a preset file — or the
built-in **PySCF** level preset.  Program-specific presets are data, not
hidden tables: a caller who has the relevant shell data supplies it through
the same API and gets the same points, pruning and weights.

## Highlights

* **Two topology sources, one API.** `PyscfGridSpec::from_level` covers the
  open-source PySCF `Grids.level` construction; `GridPreset` covers any
  caller-supplied construction, including explicit per-shell angular sizes.
* **Scientific constructions, not a generic spherical grid.** Radial node
  maps include Treutler–Ahlrichs, Delley, Mura–Knowles, Gauss–Chebyshev,
  Becke, the GC-corrected Chebyshev map and Euler–Maclaurin.  Angular rules
  include the Lebedev ladder, the 12-point icosahedral and 18-point
  octahedral rules, and Gauss–Legendre product grids.
* **Partition derivatives.** Classic Becke, Stratmann–Scuseria–Frisch, SS,
  LKO and mBecke (at the differentiable default clamp), with a streaming
  gradient/Hessian evaluator that never allocates an
  `n_atom x n_grid` object.
* **Full and streamed output.** `build_molecular_grid` materialises the whole
  grid in parallel; `GridPlan` + `PartitionEvaluator` + `BatchBuffer`
  evaluate a caller-chosen point range for large systems.
* **Explicit post-build pruning and ordering.** The build never applies a
  point cutoff and never reorders points implicitly; `PostPrune` and
  `GridOrder` make both policies explicit and testable.
* **Offline tests.** The test suite uses mathematical invariants and
  deterministic constructions; it needs no external program.

## Installation

```toml
[dependencies]
ruaa-dft-grids = "0.1"
```

The crate is pure Rust and has no C, Fortran or BLAS dependency.  It uses
Rayon for parallelism.  The 2024 edition is required (Rust 1.87 or newer).

## Quick start

```rust
use ruaa_dft_grids::grid::PyscfGridSpec;
use ruaa_dft_grids::moleculegrid::build_molecular_grid;

fn main() -> ruaa_dft_grids::Result<()> {
    // Water, Cartesian coordinates in Bohr.
    let water = [
        (8u32, [0.0, 0.0, 0.0]),
        (1, [0.0, 1.430, 1.110]),
        (1, [0.0, -1.430, 1.110]),
    ];

    let spec = PyscfGridSpec::from_level(1)?;
    let grid = build_molecular_grid(&water, &spec)?;

    println!("{} points for {} atoms", grid.len(), grid.atom_count());
    // Per point: coords, quad_weights, partition_fractions, weights, atom_idx.
    Ok(())
}
```

## Data-driven presets

A `GridPreset` is a plain data structure.  Build it in Rust:

```rust
use ruaa_dft_grids::grid::GridSpec;
use ruaa_dft_grids::preset::{ElementPreset, GridPreset};
use ruaa_dft_grids::radial::Radial;

fn main() -> ruaa_dft_grids::Result<()> {
    let h_xi = Radial::TreutlerAhlrichs.xi_for_element(1)?;
    let o_xi = Radial::TreutlerAhlrichs.xi_for_element(8)?;
    let preset = GridPreset::new(Radial::TreutlerAhlrichs)
        .with_element(1, ElementPreset::fixed(35, h_xi, 50))
        .with_element(8, ElementPreset::fixed(35, o_xi, 302));
    let spec = GridSpec::from_preset(preset);
    assert!(spec.preset().is_some());
    Ok(())
}
```

or read it from a TOML/JSON preset file:

```toml
name = "example"
radial = "treutler_ahlrichs"
molecular_cutoffs = false

[default]
n_rad = 50
radial_scale = 1.2
angular = { pruned = { base = 194, prune = "nwchem" } }

[[range]]
z_min = 3
z_max = 10
n_rad = 75
radial_scale = 0.9
angular = { shells = [50, 110, 194, 302] }

[[element]]
z = 8
radial_scale = 0.888
cutoff = 12.0
```

A range supplies shared parameters for a contiguous charge range; partial
element entries inherit the remaining fields (an element wins over its range,
which wins over `default`).  This keeps a full-periodic-table preset compact:
one block per row plus the per-element radial scale.

The file form also accepts a compact top-level `element = [...]` array with
one inline object per element, such as `{ z = 8, radial_scale = 0.9 }`.  The
shipped PySCF preset uses that spelling.  The array must precede the
`[[range]]` headers, because TOML scopes bare keys to the last table header.

The [presets page](https://docs.rs/ruaa-dft-grids/latest/ruaa_dft_grids/doc/presets/index.html)
opens PySCF's level 3 preset as data, ships the complete
`examples/presets/pyscf-level-3.toml` (seven ranges, 103 element entries), and
includes an optional `examples/presets/check_against_pyscf.py` cross-check
against PySCF's own grid construction.

A preset can declare the element charges it covers with `z_min`/`z_max`
(default 1–118).  `element(z)` outside that domain returns
`MissingElementData` instead of applying the shared `default` block, which lets
a program preset state, for example, a Z = 1–112 coverage limit.

A directory of preset files can be indexed by a JSON catalog
(`catalog::PresetCatalog`).  The catalog records the program, category, file
path, grid description and element range; `from_json_path` validates the index,
and `PresetEntry::load` reads the referenced JSON preset and applies the range.
The crate ships the catalog model and loader, not the catalog or any program
data:

```rust,no_run
use std::path::Path;
use ruaa_dft_grids::catalog::PresetCatalog;

fn main() -> ruaa_dft_grids::Result<()> {
    let dir = Path::new("data");
    let catalog = PresetCatalog::from_json_path(&dir.join("manifest.json"))?;
    let preset = catalog.load_preset("demo", "demo-grid", dir)?;
    let grid = preset.atomic_grid(8)?;
    assert!(grid.nr() > 0);
    Ok(())
}
```

`AngularLayout` covers a fixed rule, a pruning policy, an explicit per-shell
size list, a uniform product rule and a per-shell adaptive-phi product rule.  A construction outside those layouts can be
assembled as an `AtomicGrid` and passed to
`build_molecular_grid_from_atomic_grids`.

The partition is part of the full spec:

```rust
use ruaa_dft_grids::grid::{GridSpec, PyscfGridSpec};
use ruaa_dft_grids::partition::{PartitionConfig, PartitionScheme, RadiiConfig};

fn main() -> ruaa_dft_grids::Result<()> {
    let spec = GridSpec::from(PyscfGridSpec::from_level(1)?)
        .with_partition(PartitionConfig::new(
            PartitionScheme::MBecke,
            RadiiConfig::pyscf(),
        ));
    assert_eq!(spec.partition().scheme, PartitionScheme::MBecke);
    Ok(())
}
```

## Pipeline

```text
GridSpec -> GridPlan -> PartitionEvaluator -> BatchBuffer
```

* `GridSpec` is the full specification (topology plus partition).
* `GridPlan` is the topology: flat local coordinates, parent atoms and
  unpartitioned quadrature weights.  It is geometry-independent unless the
  preset enables molecular radial cutoffs.
* For a custom radial, angular or pruning construction, build an
  `AtomicGrid` (including its `z`) and pass the grids to
  `GridPlan::from_atomic_grids`; the result feeds the same streaming
  evaluator.
* `PartitionEvaluator` fills only a requested point range into a
  `BatchBuffer`, so the crate never stores a full `n_atom x n_grid` object.
* `MolecularGrid` is the convenience full-grid output;
  `build_molecular_grid` is the parallel one-call builder.

All coordinates are in **Bohr**.  Angular weights sum to one; the `4*pi`
factor is carried by the radial weight, so the quadrature weight is
`q_g = 4*pi*r^2 * dr * w_ang` and the total grid weight is `w_g = q_g P_g`.
The unpartitioned weight `quad_weight` and the normalized partition fraction
`partition_fraction` are stored separately.  Some Lebedev rules (74, 230,
266 points) carry negative quadrature weights; the arrays are still the exact
quadrature data.

## Partition derivatives

`MolecularGrid::weight_derivatives` is the one-call analytic gradient.
Response code can select atoms and contract in the same streaming pass:

```rust
use ruaa_dft_grids::grid::PyscfGridSpec;
use ruaa_dft_grids::moleculegrid::build_molecular_grid;
use ruaa_dft_grids::partition::{
    DerivContract, DerivOptions, DerivOutput, DerivPoints, DerivRequest,
};

fn main() -> ruaa_dft_grids::Result<()> {
    let atoms = [
        (8u32, [0.0, 0.0, 0.0]),
        (1, [0.0, 1.430, 1.110]),
        (1, [0.0, -1.430, 1.110]),
    ];
    let grid = build_molecular_grid(&atoms, &PyscfGridSpec::from_level(1)?)?;
    let evaluator = grid.partition_deriv_evaluator(DerivOptions {
        atoms: Some(vec![0, 2]),
        ..DerivOptions::default()
    })?;
    let coefficients = vec![1.0; grid.len()];
    let mut out = DerivOutput::new();
    evaluator.fill(
        DerivPoints::all_grid(&grid),
        &DerivRequest {
            weights: true,
            dw: true,
            dc: Some(DerivContract::exact(&coefficients, 1)),
            ..DerivRequest::default()
        },
        &mut out,
    )?;
    assert_eq!(out.dc().unwrap().dc().len(), 6);
    Ok(())
}
```

The derivative pages cover the tensor layouts, atom selection, sorted grids
and the generating-atom translation correction, and derive the classic Becke
gradient/Hessian.  `fill(DerivPoints::grid(..))` evaluates a point interval of
a materialised grid, `fill_serial` is its workspace-reusing serial form, and
`fill_serial_visit` visits one corrected Hessian at a time.

## Post-build pruning and ordering

The builder returns every angular point.  `MolecularGrid::split` applies a
`PostPrune` policy and returns the retained and pruned grids.
`GridOrder` applies a validated point permutation; totals, per-atom
quadrature weights and analytic derivatives are invariant.

## Documentation

The long-form documentation is part of the API docs:

| Page | Content |
|---|---|
| [User guide](https://docs.rs/ruaa-dft-grids/latest/ruaa_dft_grids/doc/guide/index.html) | consumer path with runnable examples |
| [Presets](https://docs.rs/ruaa-dft-grids/latest/ruaa_dft_grids/doc/presets/index.html) | the built-in PySCF level, the full preset file, the element domain, the preset catalog, the PySCF cross-check |
| [Numerical algorithms](https://docs.rs/ruaa-dft-grids/latest/ruaa_dft_grids/doc/algorithms/index.html) | radial maps, angular rules, pruning, partition formulas |
| [Partition derivative API](https://docs.rs/ruaa-dft-grids/latest/ruaa_dft_grids/doc/partition_derivatives/index.html) | tensor layouts, streaming, selection, sorted grids |
| [Becke derivatives](https://docs.rs/ruaa-dft-grids/latest/ruaa_dft_grids/doc/becke_derivatives/index.html) | the classic derivation |
| [Scheme variants](https://docs.rs/ruaa-dft-grids/latest/ruaa_dft_grids/doc/scheme_variants/index.html) | SSF, SS, LKO and mBecke differences |
| [Performance](https://docs.rs/ruaa-dft-grids/latest/ruaa_dft_grids/doc/performance/index.html) | measured throughput and build targets |

The API documentation uses KaTeX for the formulas.  To build it locally with
the same rendering as docs.rs:

```bash
RUSTDOCFLAGS="--html-in-header katex-header.html" cargo doc --open
```

The repository also contains runnable examples:

```text
cargo run --release --example user_build
cargo run --release --example user_presets
cargo run --release --example preset_catalog -- /path/to/data/manifest.json demo demo-grid 8
cargo run --release --example user_derivatives
cargo run --release --example user_sorting
```

## Development

This repository is fully driven by LLM agents: the code, tests, examples,
documentation and commit history are produced by AI coding agents; a human
maintainer sets the direction and reviews every change.  Each commit records
the agent and the model that produced it in its `Co-authored-by` trailers.

## Status

The crate is at `0.1.x`.  The API is not frozen yet; lower-level types may be
refined as the consumer API stabilises.

## References

* A. D. Becke, *J. Chem. Phys.* **88**, 2547 (1988).
* O. Treutler, R. Ahlrichs, *J. Chem. Phys.* **102**, 346 (1995).
* M. Krack, A. M. Köster, *J. Chem. Phys.* **108**, 3226 (1998).
* H. Stratmann, G. E. Scuseria, M. J. Frisch, *Chem. Phys. Lett.* **257**,
  213 (1996).
* H. Laqua, J. Kussmann, C. Ochsenfeld, *J. Chem. Phys.* **149**, 204111
  (2018).
* PySCF 2.x `pyscf.dft.radi` and `pyscf.dft.gen_grid`.
* V. I. Lebedev, D. N. Laikov, *Doklady Mathematics* **59**, 477 (1999).

## License

Licensed under the Apache License, Version 2.0.  See the `LICENSE` and
`NOTICE` files.
