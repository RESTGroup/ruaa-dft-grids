# Presets: the built-in PySCF level and data-driven grids

A grid specification is a **topology** plus a **partition**.  The topology is
the atomic construction: one radial node map, a radial length scale for each
element, an optional radial cutoff, and an angular layout for every radial
shell.  The crate ships one complete topology — PySCF's open-source
`Grids.level` preset — and a data model, `GridPreset`, for caller-supplied
topologies.  This page uses PySCF's level 3 preset to show both forms, the
complete preset file, how to change one element, and how to check the result
against PySCF.

All coordinates and lengths are in Bohr.  The partition is not part of the
topology; §6 shows how to attach one.

## 1. Use the built-in PySCF preset

`PyscfGridSpec::from_level(level)` selects PySCF's `Grids.level` construction:
the radial node map (`Grids.radi_method`, Treutler–Ahlrichs by default), the
pruning scheme (`Grids.prune`, NWChem by default), and the per-period
`(n_rad, n_ang)` pair from `gen_grid.RAD_GRIDS` and `gen_grid.ANG_ORDER`.
Level 3 is PySCF's default:

| Periodic row | Elements | `n_rad` | Angular points |
|---|---|---|---|
| 1 | H, He | 50 | 302 |
| 2 | Li–Ne | 75 | 302 |
| 3 | Na–Ar | 80 | 434 |
| 4 | K–Kr | 90 | 434 |
| 5 | Rb–Xe | 95 | 434 |
| 6 | Cs–Rn | 100 | 434 |
| 7 | Fr–Og | 105 | 434 |

The tables are also available piecewise: `pyscf::level_defaults(z, level)`
returns `(n_rad, n_ang)` for one element, `pyscf::angular_count(n)` resolves a
point count or a PySCF `LEBEDEV_ORDER` key (order `11` is 50 points),
`pyscf::lebedev_points_for_order(order)` maps an order directly, and
`Radial::TreutlerAhlrichs.xi_for_element(z)` gives the radial scale.  A level
above 9 is `Error::UnsupportedLevel`.  The size tables cover Z = 1–118; the
default Treutler scale table ends at Z = 103.

Use the builder for per-element overrides:

```rust
use ruaa_dft_grids::grid::{GridSpec, PyscfGridSpecBuilder};
use ruaa_dft_grids::prune::Prune;
use ruaa_dft_grids::radial::Radial;

fn main() -> ruaa_dft_grids::Result<()> {
    let mut overrides = std::collections::BTreeMap::new();
    overrides.insert(6u32, (99, 590)); // C: 99 shells, 590-point rule

    let spec: GridSpec = PyscfGridSpecBuilder::default()
        .level(3)                          // Grids.level
        .atom_grid(overrides)              // per-Z (n_rad, n_ang) overrides
        .radial(Radial::TreutlerAhlrichs)  // Grids.radi_method
        .prune(Prune::Nwchem)              // Grids.prune
        .build()?
        .into();
    assert!(!spec.geometry_dependent());
    Ok(())
}
```

`atom_grid` accepts either a point count or a `LEBEDEV_ORDER` key for `n_ang`;
the builder resolves the latter through `angular_count`.  A level preset is
geometry-independent: its point set depends only on the elements, not on
their positions.

## 2. Open the same preset as data

`GridPreset` states the topology as data: a radial node map, an optional
`default` element block, contiguous `range` blocks, and per-element entries.
An element entry may omit fields; the missing values come from the matching
range, or from `default` when no range matches.

PySCF's level tables map onto this model directly: one range per periodic
row, and one element entry for the Treutler scale.  For H–Ne at level 3:

```rust
use ruaa_dft_grids::grid::PyscfGridSpec;
use ruaa_dft_grids::preset::{ElementPreset, GridPreset};
use ruaa_dft_grids::prune::Prune;
use ruaa_dft_grids::pyscf::level_defaults;
use ruaa_dft_grids::radial::Radial;

fn pyscf_level_3() -> ruaa_dft_grids::Result<GridPreset> {
    let radial = Radial::TreutlerAhlrichs;
    let mut preset = GridPreset::new(radial).with_name("pyscf-level-3");
    // One range per periodic row; the full preset has seven.
    for &(z_min, z_max) in &[(1u32, 2u32), (3, 10)] {
        let (n_rad, n_ang) = level_defaults(z_min, 3)?;
        // The range scale is a placeholder: every element sets its own.
        let base = ElementPreset::pruned(n_rad, 1.0, n_ang, Prune::Nwchem);
        preset = preset.with_default_range(z_min, z_max, base.clone());
        for z in z_min..=z_max {
            preset = preset.with_element(
                z,
                ElementPreset {
                    radial_scale: radial.xi_for_element(z)?,
                    ..base.clone()
                },
            );
        }
    }
    Ok(preset)
}

fn main() -> ruaa_dft_grids::Result<()> {
    let preset = pyscf_level_3()?;
    let pyscf = PyscfGridSpec::from_level(3)?;
    for z in 1..=10 {
        assert_eq!(preset.atomic_grid(z)?, pyscf.atomic_grid(z)?);
    }
    Ok(())
}
```

The range carries the level sizes and the pruning layout; the element entry
carries the radial scale and inherits the rest.  The assertion is exact: the
data path builds the same points and quadrature weights as
`PyscfGridSpec::from_level(3)`.  `examples/user_presets.rs` builds the same
loop over all seven rows, and `tests/preset_api.rs` checks the result against
the built-in atomic grids for Z = 1–103, the domain of the Treutler scale
table.

`with_default_range` and `with_element` are the Rust builders.
`GridPreset::element(z)` applies the precedence explicitly: an element entry
wins over a matching range, which wins over `default`.  A preset with no entry
for an element and no matching range/default returns
`Error::MissingElementData { table: "grid preset" }`.

## 3. The complete preset file

`examples/presets/pyscf-level-3.toml` is the level-3 preset above for the
whole table: seven `[[range]]` blocks and 103 `[[element]]` entries.  It is
the file loaded by `examples/user_presets.rs` and checked by the script in §5.
An excerpt:

```toml
name = "pyscf-level-3"
radial = "treutler_ahlrichs"

# H, He
[[range]]
z_min = 1
z_max = 2
n_rad = 50
radial_scale = 1.0
angular = { pruned = { base = 302, prune = "nwchem" } }

# Li-Ne
[[range]]
z_min = 3
z_max = 10
n_rad = 75
radial_scale = 1.0
angular = { pruned = { base = 302, prune = "nwchem" } }

# ... five more ranges, then one entry per element ...

[[element]]
z = 1
radial_scale = 0.8

[[element]]
z = 6
radial_scale = 1.1

[[element]]
z = 8
radial_scale = 0.9
```

The range `radial_scale = 1.0` is a placeholder: every element in the range
has its own entry, and a range value applies only to an element without an
entry.  A `[[range]]` or `[default]` block is a complete element block; only
`[[element]]` entries are partial.  The partial entry is what keeps a
full-periodic-table preset compact: seven ranges plus one scale per element.

Read the file with the `serde` format crate of your choice.  With
`toml = "0.8"` in your own `Cargo.toml`:

```rust
// The repository ships this file; a consumer would use its own path.
use ruaa_dft_grids::preset::GridPreset;

fn main() -> ruaa_dft_grids::Result<()> {
    let text = std::fs::read_to_string("examples/presets/pyscf-level-3.toml")
        .expect("the preset file is in the repository");
    let preset: GridPreset = toml::from_str(&text).expect("the preset is valid TOML");
    assert_eq!(preset.elements.len(), 103);
    assert_eq!(preset.defaults.len(), 7);
    Ok(())
}
```

The core crate ships no file loader and no file-location convention; the data
types and their invariants are the API.  Writing a preset back with `serde`
emits complete element entries, which are valid input, just longer than the
compact form above.  `tests/preset_api.rs` checks the TOML round trip, the
inheritance order, and that this file equals the loop-built preset.

The file covers Z = 1–103 because the Treutler scale table ends there.  PySCF
itself has no default Treutler scale for Z = 104–118 either; add an explicit
`[[range]]` or `[[element]]` block with a `radial_scale` if you have one.

## 4. Change one element

`with_element(z, element)` replaces the whole element entry, so pass a complete
`ElementPreset` (start from the range block and override the fields you need).
Here a default block stands in for the whole table and one element is
replaced:

```rust
use ruaa_dft_grids::preset::{ElementPreset, GridPreset};
use ruaa_dft_grids::prune::Prune;
use ruaa_dft_grids::radial::Radial;

fn main() -> ruaa_dft_grids::Result<()> {
    let radial = Radial::TreutlerAhlrichs;
    let shared = ElementPreset::pruned(75, radial.xi_for_element(6)?, 302, Prune::Nwchem);
    let oxygen = ElementPreset::pruned(90, radial.xi_for_element(8)?, 590, Prune::Nwchem);
    let preset = GridPreset::new(radial)
        .with_name("promote oxygen")
        .with_default(shared)
        .with_element(8, oxygen);
    assert_eq!(preset.element(6)?.n_rad, 75); // default block
    assert_eq!(preset.element(8)?.n_rad, 90); // explicit entry
    Ok(())
}
```

For a standard PySCF level, take the new sizes from `level_defaults(z, level)`
and the scale from `Radial::xi_for_element(z)` as in §2.  In the file form,
write a partial entry instead:

```toml
[[element]]
z = 8
n_rad = 90
radial_scale = 0.9
angular = { pruned = { base = 590, prune = "nwchem" } }
```

`AngularLayout::Shells` is the escape hatch for pruning that is not one of the
built-in `Prune` policies.  An explicit shell list must cover the *active*
shells, so it cannot follow a moving molecular cutoff:
`molecular_cutoffs = true` together with `Shells`/`ProductShells` and a cutoff
is rejected by `GridPreset::validate` as `Error::UnsupportedGrid`.

A preset with `molecular_cutoffs = true` extends each atom's cutoff by the
pairwise geometry rule, so its point set depends on the full geometry; rebuild
the plan when the geometry moves.  See the [user guide](crate::doc::guide)
§2 for the formula and the derivative consequence.

## 5. Check the preset against PySCF

`examples/presets/check_against_pyscf.py` runs `examples/user_presets.rs` in
dump mode and compares the preset's atomic grids with PySCF's own
`gen_atomic_grids` at level 3.  For every selected element it checks the shell
count, the per-shell angular size, radius and radial volume element, the total
quadrature weight, and the full point set (coordinates and weights).

```text
python3 examples/presets/check_against_pyscf.py
python3 examples/presets/check_against_pyscf.py --elements all
python3 examples/presets/check_against_pyscf.py --elements 1,8,26,54,86 --tolerance 1e-11
```

The script needs PySCF, NumPy and a Rust toolchain.  It prints the maximum
deviation per element and exits non-zero on a mismatch.  PySCF groups points
by angular size and the crate keeps radial-shell order; the script compares
the sorted point sets.  The check is optional and manual: `cargo test` never
invokes it and stays offline.  `tests/preset_api.rs` is the offline part of
the same contract, asserting that the file equals the loop-built preset and
that the loop-built preset equals the built-in atomic grids for Z = 1–103.

## 6. Combining the topology with a partition

The partition is a separate part of the full `GridSpec`:

```rust
use ruaa_dft_grids::grid::{GridSpec, PyscfGridSpec};
use ruaa_dft_grids::partition::{PartitionConfig, PartitionScheme, RadiiConfig};

fn main() -> ruaa_dft_grids::Result<()> {
    let spec = GridSpec::from(PyscfGridSpec::from_level(3)?)
        .with_partition(PartitionConfig::new(
            PartitionScheme::MBecke,
            RadiiConfig::pyscf(),
        ));
    assert_eq!(spec.partition().scheme, PartitionScheme::MBecke);
    Ok(())
}
```

For a data preset, use `GridSpec::from_preset(preset)` (which defaults to
`PartitionConfig::pyscf()`) or
`GridSpec::from_preset_with_partition(preset, partition)`.  The topology data
does not carry a partition or radii choice.

## 7. Errors and pitfalls

* A missing element entry is `MissingElementData { table: "grid preset" }`.
  The shipped file returns it for Z = 104–118; the built-in path reports the
  missing Treutler table instead.
* A level above 9 is `UnsupportedLevel`; a charge outside Z = 1–118 is
  `UnsupportedElement`.
* A partial `[[element]]` entry with no `[[range]]`/`[default]` to inherit
  from is a deserialization error naming the missing field.
* `Shells`/`ProductShells` with `molecular_cutoffs = true` and a cutoff is
  `UnsupportedGrid`; a shell list that does not cover the active shells is
  `LengthMismatch` when the grid is built.
* `cutoff` and `generate_all` are inherited from a range or `default` when an
  element entry omits them.  A partial entry cannot clear an inherited cutoff;
  split the range so that element is covered by a range without a cutoff, or
  use a preset that has no range cutoff.

## 8. See also

* `examples/user_presets.rs` — the built-in preset, the data preset, the file,
  the override and the dump mode.
* `examples/presets/pyscf-level-3.toml` — the complete level-3 preset file.
* `examples/presets/check_against_pyscf.py` — the optional PySCF cross-check.
* `tests/preset_api.rs` — preset file, inheritance and equivalence checks.
* [User guide](crate::doc::guide) §2 — specifications and preset basics.
* [Numerical algorithms](crate::doc::algorithms) — radial maps, angular rules,
  pruning and partition formulas.
* PySCF `pyscf.dft.gen_grid` and `pyscf.dft.radi` — the published level,
  pruning and radial tables.
