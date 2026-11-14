# User guide: from molecular geometry to grid weights

This guide is the consumer path through `ruaa_dft_grids`:

```text
molecular geometry (Bohr)
        + a grid specification (PySCF preset or data-driven preset)
        -> MolecularGrid                 full grid, built in parallel
        -> analytic weight derivatives   gradient, optional Hessian
        -> consumer-defined point order  sorting, blocking, screening
```

The lower-level streaming API (`GridPlan`, `PartitionEvaluator`,
`BatchBuffer`) is described in the [crate overview](crate); most users do not
need it.  The formula-level background is in the
[numerical algorithms](crate::doc::algorithms) page.

All coordinates and lengths are in **Bohr**.  A geometry is a slice of
`(Z, [x, y, z])` pairs.  Convert Ångström input with
`x_bohr = x_angstrom * 1.889_726_125_457_828_1`.

## 1. Quick start

```rust
use ruaa_dft_grids::grid::PyscfGridSpec;
use ruaa_dft_grids::moleculegrid::build_molecular_grid;

fn main() -> ruaa_dft_grids::Result<()> {
    // Water in Bohr.
    let water = [
        (8u32, [0.0, 0.0, 0.0]),
        (1, [0.0, 1.430, 1.110]),
        (1, [0.0, -1.430, 1.110]),
    ];

    let spec = PyscfGridSpec::from_level(1)?;       // a PySCF level preset
    let grid = build_molecular_grid(&water, &spec)?;

    println!("{} points for {} atoms", grid.len(), grid.atom_count());
    println!("total weight {:.6}", grid.total_weight());

    assert_eq!(grid.coords().len(), grid.len());
    assert_eq!(grid.weights().len(), grid.len());
    assert_eq!(grid.atom_idx().len(), grid.len());
    Ok(())
}
```

`build_molecular_grid` is the parallel full-grid builder.  It materialises
every point and evaluates the partition weights on the ambient Rayon thread
pool.  Angular pruning is a topology input; the builder never applies a
post-build cutoff, and point selection is an explicit operation (§5).  Use
`RAYON_NUM_THREADS` or a caller-installed Rayon pool to control the worker
count.

The grid is self-describing: `grid.spec()` is the exact `GridSpec` used, and
`grid.atom_coords()` / `grid.atom_charges()` are the geometry.

### What the per-point arrays mean

| Field | Shape | Meaning |
|---|---|---|
| `coords` | `[G, 3]` | Cartesian coordinates (Bohr) |
| `quad_weights` | `[G]` | Unpartitioned quadrature weight $q$ |
| `partition_fractions` | `[G]` | Normalized partition fraction $P$ of the generating atom |
| `weights` | `[G]` | Total weight $w = qP$ |
| `atom_idx` | `[G]` | Generating atom (`u32`) |

Some Lebedev rules (74, 230, 266 points) carry negative quadrature weights;
the weights and fractions are still the exact quadrature data.  Do not assume
`q > 0` or `w > 0`; use `|w|` or the partition fraction when a cutoff must be
positive.

## 2. Specifications

### 2.1 PySCF presets

`PyscfGridSpec::from_level(level)` is the shortest form.  For more control use
the builder:

```rust
use ruaa_dft_grids::grid::{GridSpec, PyscfGridSpecBuilder};

fn main() -> ruaa_dft_grids::Result<()> {
    let spec: GridSpec = PyscfGridSpecBuilder::default()
        .level(3)                    // Grids.level
        // .radial(Radial::TreutlerAhlrichs) // Grids.radi_method
        // .prune(Prune::Treutler)           // Grids.prune
        .build()?
        .into();
    assert!(!spec.geometry_dependent());
    Ok(())
}
```

The level selects the radial and angular sizes from PySCF's published
tables.  A level above 9 is rejected at spec construction as
`Error::UnsupportedLevel`.

### 2.2 Data-driven presets

A `GridPreset` states the topology directly:

```rust
use ruaa_dft_grids::grid::GridSpec;
use ruaa_dft_grids::preset::{ElementPreset, GridPreset};
use ruaa_dft_grids::radial::Radial;

fn main() -> ruaa_dft_grids::Result<()> {
    let h_xi = Radial::TreutlerAhlrichs.xi_for_element(1)?;
    let o_xi = Radial::TreutlerAhlrichs.xi_for_element(8)?;
    let preset = GridPreset::new(Radial::TreutlerAhlrichs)
        .with_name("water")
        .with_element(1, ElementPreset::fixed(35, h_xi, 50))
        .with_element(8, ElementPreset::fixed(35, o_xi, 302));
    let spec = GridSpec::from_preset(preset);
    assert!(spec.preset().is_some());
    Ok(())
}
```

Each `ElementPreset` carries the nominal radial count, the radial length
scale, an optional cutoff, whether the cutoff keeps the full nominal shell
set, and an `AngularLayout`:

| `AngularLayout` | Meaning |
|---|---|
| `Fixed { n_ang }` | the same angular rule on every shell |
| `Pruned { base, prune }` | a base size plus a `Prune` scheme |
| `Shells(Vec<usize>)` | an explicit size for every active shell |
| `Product { theta }` | Gauss–Legendre theta x uniform-phi product rule |
| `ProductShells(Vec<usize>)` | one theta-ring count per shell, with the per-ring phi count chosen adaptively |

Presets implement `serde`.  The file form uses one element entry per
element — an `[[element]]` block, or one object in a top-level
`element = [...]` array; a `[[range]]` entry supplies shared parameters for a
contiguous charge range, and an element entry may override only the fields
that differ:

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
n_rad = 75
radial_scale = 1.1
cutoff = 12.0
angular = { shells = [50, 110, 194, 302] }
```

An explicit element entry wins over a matching `range`, which wins over
`default`; partial entries inherit the remaining fields.  Read the file with
the `serde`/TOML/JSON stack of your choice.  The core crate does not
ship a file loader or a file-location convention; the data types and their
invariants are the API.  A preset that sets
`molecular_cutoffs = true` extends each atom's cutoff by the pairwise rule

$$
R(A) = \max\left(R_\mathrm{iso}(A),\ \max_B\left[R_\mathrm{iso}(B) + d(A,B)\right]\right),
$$

so its point set depends on the full geometry.  Such a preset produces a
geometry-dependent `GridPlan`; rebuild the plan when the geometry moves.

The [presets page](crate::doc::presets) builds the complete PySCF level-3
preset as data, ships its TOML file and checks it against PySCF.

### 2.3 Combining topology and partition

The partition is part of the full `GridSpec`, not of the topology:

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

The built-in `PartitionConfig::pyscf()` is classic Becke with PySCF's
Bragg/Treutler radii.  `gaussian_ss()` and `gaussian_becke()` are the
published SS/BWeights formulas with equality radii.  A caller can supply any
`RadiiTable::Custom` values for a different program's radii.

## 3. From `MolecularGrid` to weight derivatives

The quadrature weight $q$ is fixed in each atom's local frame.  Only the
normalized partition fraction $P$ depends on the nuclear coordinates, so the
analytic derivative API provides

$$
w_g = q_g P_g,
\qquad
\frac{\partial w_g}{\partial \mathbf R} =
q_g \frac{\partial P_g}{\partial \mathbf R}.
$$

The one-call convenience method selects every atom, requests bare partition
weights and raw first derivatives, and runs in parallel in the grid's
current slot order:

```rust
use ruaa_dft_grids::grid::PyscfGridSpec;
use ruaa_dft_grids::moleculegrid::build_molecular_grid;

fn main() -> ruaa_dft_grids::Result<()> {
    let water = [
        (8u32, [0.0, 0.0, 0.0]),
        (1, [0.0, 1.430, 1.110]),
        (1, [0.0, -1.430, 1.110]),
    ];
    let grid = build_molecular_grid(&water, &PyscfGridSpec::from_level(1)?)?;
    let deriv = grid.weight_derivatives()?;

    assert_eq!(deriv.partition_weights().len(), grid.len());
    assert_eq!(deriv.dw().len(), grid.atom_count() * 3 * grid.len());
    assert!(deriv.dw_atom(0).is_some());
    Ok(())
}
```

Flattening is `(atom * 3 + component) * G + point`, with `component` in
`x, y, z` order.  To recover the derivative of the bare partition fraction,
divide by the point's quadrature weight.

### 3.1 Selected rows and contractions

Differentiating a subset of atoms and contracting over points avoids
materialising a larger tensor than needed:

```rust
use ruaa_dft_grids::grid::PyscfGridSpec;
use ruaa_dft_grids::moleculegrid::build_molecular_grid;
use ruaa_dft_grids::partition::{
    DerivContract, DerivOptions, DerivOutput, DerivPoints, DerivRequest,
};

fn main() -> ruaa_dft_grids::Result<()> {
    let water = [
        (8u32, [0.0, 0.0, 0.0]),
        (1, [0.0, 1.430, 1.110]),
        (1, [0.0, -1.430, 1.110]),
    ];
    let grid = build_molecular_grid(&water, &PyscfGridSpec::from_level(1)?)?;
    let g = grid.len();
    let coefficients = vec![1.0; g];

    let evaluator = grid.partition_deriv_evaluator(DerivOptions {
        atoms: Some(vec![0, 2]),
        ..DerivOptions::default()
    })?;

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

    let dc = out.dc().unwrap().dc_set(0);
    assert_eq!(dc.len(), 2 * 3);
    Ok(())
}
```

`dc` is $\sum_g c_g \partial w_g/\partial R$; `ddc` and the raw Hessian
`ddw` are available from the same call.  For a point interval use
`fill(DerivPoints::grid(&grid, range))`; for a custom raw-Hessian reduction
use `fill_serial_visit`.  The full tensor layouts, windowing and
finite-difference workflow are in the
[partition derivative guide](crate::doc::partition_derivatives).

### 3.2 Which partitions can be differentiated?

| Partition | Derivative support |
|---|---|
| classic Becke, `becke_early_exit = false` | yes |
| Stratmann–Scuseria (SSF) | yes (either flag) |
| SS | yes (either flag) |
| LKO | yes |
| mBecke at the default clamp `1.0` | yes (classic Becke backend) |
| classic Becke with early exit | no — `Error::UnsupportedDerivative` |
| `mbecke_clamp < 1.0` | no — clamped/truncated weight is discontinuous |

A geometry-dependent preset with molecular cutoffs has a **moving point
set**; the analytic derivative covers the partition fractions of the current
point set, not the moving cutoff.  Disable `molecular_cutoffs` when a
fixed-topology grid is required for derivative work.

### 3.3 Parallelism

`weight_derivatives` and `fill(DerivPoints::all_grid(..))` use the ambient
Rayon pool and evaluate points in the grid's current slot order.
`fill(DerivPoints::grid(..))` is the grid-native interval variant and is the
way to stream a large raw Hessian in bounded batches; `fill_serial` is the
workspace-reusing single-threaded form.  `fill_serial_visit` visits one point
at a time and never allocates a batch-sized `ddw`.

### 3.4 Partition weights at custom points

Points that did not come from a grid plan or a materialised grid still need
partition weights.  `PartitionEvaluator::from_parts` (or `new` with a
`Molecule`) builds a reusable evaluator with the molecule tables cached; its
`fill_points`/`fill_points_par` methods write row-major `[npoints, natom]`
weights (each row sums to one), and `weights`/`weights_at` are the allocating
convenience forms.

## 4. Sorting molecular grids

Ordering is a consumer policy, not part of the grid definition.  A
`GridOrder` is a validated permutation with its inverse; applying it moves
every per-point array together.  **Totals, per-atom quadrature weights and
analytic derivatives are invariant.**

```rust
use ruaa_dft_grids::grid::PyscfGridSpec;
use ruaa_dft_grids::moleculegrid::build_molecular_grid;

fn main() -> ruaa_dft_grids::Result<()> {
    let water = [
        (8u32, [0.0, 0.0, 0.0]),
        (1, [0.0, 1.430, 1.110]),
        (1, [0.0, -1.430, 1.110]),
    ];
    let mut grid = build_molecular_grid(&water, &PyscfGridSpec::from_level(1)?)?;
    let order = grid.reorder_boxes(1.2)?;   // sorts and returns the permutation
    for block in order.blocks(64) {         // blocks of sorted slots
        assert!(!block.is_empty());
        break;
    }
    Ok(())
}
```

`order.perm()[slot]` is the pre-sort source index, and
`order.inverse()[source]` is the new slot.  `reorder_morton(cell)` is the
Z-order variant.  A caller that already has a significance signature or
another key can sort with it directly:

```rust
use ruaa_dft_grids::grid::PyscfGridSpec;
use ruaa_dft_grids::moleculegrid::build_molecular_grid;
use ruaa_dft_grids::order::GridOrder;

fn main() -> ruaa_dft_grids::Result<()> {
    let water = [
        (8u32, [0.0, 0.0, 0.0]),
        (1, [0.0, 1.430, 1.110]),
        (1, [0.0, -1.430, 1.110]),
    ];
    let mut grid = build_molecular_grid(&water, &PyscfGridSpec::from_level(1)?)?;
    let keys = grid.atom_idx().iter().map(|&atom| u64::from(atom));
    let order = GridOrder::from_keys(keys)?; // stable for equal keys
    grid.reorder(&order)?;
    Ok(())
}
```

`GridOrder::from_permutation(&perm)` accepts a complete permutation computed
outside the crate.  `GridOrder::identity(n)` is a no-op.  Sorting applies to a
**materialised** grid, not to a `GridPlan`: the value evaluator walks
atom-contiguous runs.  The derivative evaluator is point-order agnostic, so a
sorted grid derives correctly in its current order.

After `MolecularGrid::split`, apply an order to each resulting grid
separately; the retained/pruned pair no longer shares one index space.

## 5. Post-build point selection

`MolecularGrid::split` returns the retained and pruned grids for a
`PostPrune` policy:

```rust
use ruaa_dft_grids::grid::PyscfGridSpec;
use ruaa_dft_grids::moleculegrid::{PostPrune, build_molecular_grid};

fn main() -> ruaa_dft_grids::Result<()> {
    let water = [
        (8u32, [0.0, 0.0, 0.0]),
        (1, [0.0, 1.430, 1.110]),
        (1, [0.0, -1.430, 1.110]),
    ];
    let grid = build_molecular_grid(&water, &PyscfGridSpec::from_level(1)?)?;
    let n = grid.len();
    let split = grid.split(PostPrune::PartitionCut(1e-14))?;
    assert_eq!(split.retained.len() + split.pruned.len(), n);
    Ok(())
}
```

`KeepIndices` and `KeepMask` fix a point set exactly and are the
differentiable choice.  `TotalWeightCut` and `PartitionCut` choose a subset
from the geometry, so their boundary is not differentiable.

## 6. Custom atomic grids

A construction that no preset layout expresses can be assembled directly:

```rust
use ruaa_dft_grids::atomgrid::AtomicGrid;
use ruaa_dft_grids::molecule::Molecule;
use ruaa_dft_grids::moleculegrid::build_molecular_grid_from_atomic_grids;
use ruaa_dft_grids::partition::PartitionConfig;

fn main() -> ruaa_dft_grids::Result<()> {
    let molecule = Molecule::new([(1u32, [0.0, 0.0, 0.0])])?;
    let atom = AtomicGrid::new(
        1,
        vec![[0.0, 0.0, 0.0]],
        vec![1.0],
        vec![1],
    )?;
    let grid = build_molecular_grid_from_atomic_grids(
        &molecule,
        vec![atom],
        PartitionConfig::pyscf(),
    )?;
    assert_eq!(grid.len(), 1);
    assert!(grid.spec().is_custom());
    Ok(())
}
```

This is the extension point for specialized angular rules and program-specific
simple grids.  The atomic grids must be in molecule order and carry the
matching nuclear charges.

## 7. Errors and common pitfalls

* Consumer construction and evaluation return `Result<T>` with one crate-wide
  `Error`; configuration, geometry and length errors are matchable, and the
  generated builder errors convert into `Error`.
* A few programmer-error paths panic rather than returning `Error`: malformed
  `DerivContract` tensors and out-of-range `GridPlan`/`MolecularGrid` index
  accessors.
* A missing element/radii entry is `Error::MissingElementData` with the table
  name; an unsupported scheme/level is `Error::UnsupportedScheme` /
  `Error::UnsupportedLevel`.
* `Error::UnsupportedDerivative` means the selected partition has no analytic
  backend, not that the grid or geometry is invalid.
* A preset without an entry for an element, a matching `range`, or a
  `default` returns `Error::MissingElementData { table: "grid preset" }`.
* Coordinates are Bohr.  Convert Ångström input with `x * (1.0 / BOHR)`.

## 8. Examples and tests

```text
cargo run --release --example user_build
cargo run --release --example user_derivatives
cargo run --release --example user_sorting
cargo test --test user_workflows
```

| Resource | Shows |
|---|---|
| `examples/user_build.rs` | PySCF preset and a data-driven preset -> `MolecularGrid`, post-prune, custom atomic grids |
| `examples/user_derivatives.rs` | `weight_derivatives`, shapes, selection, contraction, finite-difference check, unsupported early exit |
| `examples/user_sorting.rs` | box/Morton/key orders, invariance, blocks, derivative after sorting |
| `tests/user_workflows.rs` | the same contracts as offline assertions |

For deeper material:

* [crate overview](crate) — pipeline, streaming API.
* [numerical algorithms](crate::doc::algorithms) — radial and angular rules,
  pruning, partition formulas.
* [partition derivatives](crate::doc::partition_derivatives) — tensor
  layouts, streaming, selection, sorted-grid semantics and validation.
* [Becke derivative derivation](crate::doc::becke_derivatives).
* [scheme variants](crate::doc::scheme_variants) — SSF, SS, LKO and mBecke.
