//! Preset API checks: the built-in PySCF level as data, the file form, the
//! `element` > `range` > `default` inheritance, and the topology errors.
//!
//! The equivalence test is the contract behind the preset page and
//! [`../examples/user_presets.rs`](../examples/user_presets.rs): a caller can
//! take PySCF's published level tables, express them as a [`GridPreset`], and
//! obtain exactly the built-in atomic grids.

use ruaa_dft_grids::Error;
use ruaa_dft_grids::grid::PyscfGridSpec;
use ruaa_dft_grids::preset::{AngularLayout, ElementPreset, GridPreset};
use ruaa_dft_grids::prune::Prune;
use ruaa_dft_grids::pyscf::level_defaults;
use ruaa_dft_grids::radial::Radial;

/// PySCF periodic rows: `(Z_min, Z_max)` of each column of `RAD_GRIDS` and
/// `ANG_ORDER`.
const PERIODS: [(u32, u32); 7] = [
    (1, 2),
    (3, 10),
    (11, 18),
    (19, 36),
    (37, 54),
    (55, 86),
    (87, 118),
];

/// The PySCF Treutler scale table ends at Z = 103.
const MAX_TREUTLER_Z: u32 = 103;

/// A PySCF `Grids(level)` topology as data: one range per periodic row and
/// one per-element Treutler scale.
fn pyscf_level_preset(level: u8) -> ruaa_dft_grids::Result<GridPreset> {
    let radial = Radial::TreutlerAhlrichs;
    let mut preset = GridPreset::new(radial).with_name(format!("pyscf-level-{level}"));
    for &(z_min, z_max) in &PERIODS {
        let z_max = z_max.min(MAX_TREUTLER_Z);
        if z_min > z_max {
            continue;
        }
        let (n_rad, n_ang) = level_defaults(z_min, level)?;
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

#[test]
fn pyscf_level_three_as_data_matches_the_built_in_preset() {
    let pyscf = PyscfGridSpec::from_level(3).unwrap();
    let preset = pyscf_level_preset(3).unwrap();

    // Every element the Treutler scale table covers (PySCF's default radius
    // map), not just a spot check: the data path is exact.
    for z in 1..=MAX_TREUTLER_Z {
        assert_eq!(
            preset.atomic_grid(z).unwrap(),
            pyscf.atomic_grid(z).unwrap(),
            "Z = {z}"
        );
    }

    // The range carries the level sizes; the element entry carries the
    // Treutler scale and inherits the rest.
    let oxygen = preset.element(8).unwrap();
    let (n_rad, n_ang) = level_defaults(8, 3).unwrap();
    assert_eq!(oxygen.n_rad, n_rad);
    assert_eq!(
        oxygen.radial_scale,
        Radial::TreutlerAhlrichs.xi_for_element(8).unwrap()
    );
    assert_eq!(oxygen.angular, AngularLayout::pruned(n_ang, Prune::Nwchem));
    assert_eq!(oxygen.cutoff, None);
    assert_eq!(
        preset.element(11).unwrap().n_rad,
        level_defaults(11, 3).unwrap().0
    );
}

#[test]
fn the_preset_stops_where_the_treutler_table_stops() {
    let preset = pyscf_level_preset(3).unwrap();
    assert!(matches!(
        preset.atomic_grid(104),
        Err(Error::MissingElementData {
            z: 104,
            table: "grid preset"
        })
    ));
    // The built-in path has the same domain, reported by the radial table.
    assert!(matches!(
        PyscfGridSpec::from_level(3).unwrap().atomic_grid(104),
        Err(Error::MissingElementData {
            z: 104,
            table: "PySCF Treutler xi"
        })
    ));
}

#[test]
fn the_committed_preset_file_matches_the_programmatic_preset() {
    // `examples/presets/pyscf-level-3.toml` is the full periodic-table file
    // used by the preset page and the Python cross-check.  It uses the compact
    // top-level `element = [...]` array; parse it and require the same
    // topology as the loop-built preset, so the file cannot drift from the
    // PySCF tables.
    let text = include_str!("../examples/presets/pyscf-level-3.toml");
    let file: GridPreset = toml::from_str(text).unwrap();
    let programmatic = pyscf_level_preset(3).unwrap();

    assert_eq!(file, programmatic);
    assert_eq!(file.name, "pyscf-level-3");
    assert_eq!(file.defaults.len(), PERIODS.len());
    assert_eq!(file.elements.len(), MAX_TREUTLER_Z as usize);
    assert!(file.default.is_none());
}

#[test]
fn preset_file_round_trips_through_toml() {
    let preset = pyscf_level_preset(3).unwrap();
    let text = toml::to_string(&preset).unwrap();
    let loaded: GridPreset = toml::from_str(&text).unwrap();

    assert_eq!(loaded, preset);
    assert_eq!(loaded.defaults.len(), PERIODS.len());
    assert_eq!(loaded.elements.len(), MAX_TREUTLER_Z as usize);
    assert_eq!(loaded.radial, Radial::TreutlerAhlrichs);
    assert_eq!(loaded.element(8).unwrap(), preset.element(8).unwrap());

    // The serializer writes `[[element]]` blocks (the canonical output
    // form); the committed file uses the compact `element` array instead.
    // Both spellings parse to the same `GridPreset`.
    let range_at = text.find("[[range]]").unwrap();
    let element_at = text.find("[[element]]").unwrap();
    assert!(range_at < element_at, "ranges are written before elements");
    assert!(text.contains("z_min = 1"));
    assert!(text.contains("z_max = 2"));
}

#[test]
fn element_entries_override_range_entries_which_override_the_default() {
    let text = r#"
name = "precedence"
radial = "treutler_ahlrichs"

[default]
n_rad = 40
radial_scale = 1.0
angular = { pruned = { base = 194, prune = "nwchem" } }

[[range]]
z_min = 3
z_max = 10
n_rad = 75
radial_scale = 0.9
angular = { pruned = { base = 302, prune = "nwchem" } }

[[element]]
z = 8
radial_scale = 0.888

[[element]]
z = 6
angular = { fixed = { n_ang = 50 } }
"#;
    let preset: GridPreset = toml::from_str(text).unwrap();

    // The element sets only `radial_scale`; the range supplies the rest.
    let oxygen = preset.element(8).unwrap();
    assert_eq!(oxygen.n_rad, 75);
    assert_eq!(oxygen.radial_scale, 0.888);
    assert_eq!(oxygen.angular, AngularLayout::pruned(302, Prune::Nwchem));

    // The element sets only `angular`; the range supplies the rest.
    let carbon = preset.element(6).unwrap();
    assert_eq!(carbon.n_rad, 75);
    assert_eq!(carbon.radial_scale, 0.9);
    assert_eq!(carbon.angular, AngularLayout::fixed(50));

    // An element outside every range falls back to the default block.
    let hydrogen = preset.element(1).unwrap();
    assert_eq!(hydrogen.n_rad, 40);
    assert_eq!(hydrogen.radial_scale, 1.0);
    assert_eq!(hydrogen.angular, AngularLayout::pruned(194, Prune::Nwchem));
    assert_eq!(preset.element(18).unwrap(), hydrogen);
}

#[test]
fn a_missing_element_without_a_default_is_reported() {
    let preset = GridPreset::new(Radial::TreutlerAhlrichs)
        .with_element(8, ElementPreset::fixed(20, 0.9, 50));
    assert!(matches!(
        preset.element(1),
        Err(Error::MissingElementData {
            z: 1,
            table: "grid preset"
        })
    ));
}

#[test]
fn a_partial_element_entry_without_a_default_is_rejected() {
    let text = r#"
radial = "treutler_ahlrichs"

[[element]]
z = 8
radial_scale = 0.9
"#;
    let error = toml::from_str::<GridPreset>(text).unwrap_err().to_string();
    assert!(
        error.contains("Z = 8") && error.contains("n_rad"),
        "unexpected message: {error}"
    );
}

#[test]
fn a_moving_cutoff_rejects_an_explicit_shell_list() {
    let text = r#"
radial = "treutler_ahlrichs"
molecular_cutoffs = true

[[element]]
z = 8
n_rad = 75
radial_scale = 0.9
cutoff = 10.0
angular = { shells = [50, 110, 194, 302] }
"#;
    let preset: GridPreset = toml::from_str(text).unwrap();
    assert!(matches!(
        preset.validate(),
        Err(Error::UnsupportedGrid { name, .. }) if name == "preset element Z=8"
    ));
    assert!(preset.atomic_grid(8).is_err());
}

#[test]
fn unknown_keys_are_rejected() {
    // Top-level typo.
    let text = r#"
radial = "treutler_ahlrichs"
molecular_cutoff = true

[[element]]
z = 8
n_rad = 20
radial_scale = 0.9
angular = { fixed = { n_ang = 50 } }
"#;
    let error = toml::from_str::<GridPreset>(text).unwrap_err().to_string();
    assert!(
        error.contains("molecular_cutoff"),
        "unexpected error: {error}"
    );

    // Element-entry typo.
    let text = r#"
radial = "treutler_ahlrichs"

[[element]]
z = 8
n_rad = 20
radial_scale = 0.9
cuttoff = 9.0
angular = { fixed = { n_ang = 50 } }
"#;
    let error = toml::from_str::<GridPreset>(text).unwrap_err().to_string();
    assert!(error.contains("cuttoff"), "unexpected error: {error}");

    // Range-entry typo.
    let text = r#"
radial = "treutler_ahlrichs"

[[range]]
z_min = 3
z_max = 10
n_rad = 20
radial_scale = 0.9
extra = 1
angular = { fixed = { n_ang = 50 } }
"#;
    let error = toml::from_str::<GridPreset>(text).unwrap_err().to_string();
    assert!(error.contains("extra"), "unexpected error: {error}");

    // Default-block typo.
    let text = r#"
radial = "treutler_ahlrichs"

[default]
n_rad = 20
radial_scale = 0.9
cuttoff = 9.0
angular = { fixed = { n_ang = 50 } }
"#;
    let error = toml::from_str::<GridPreset>(text).unwrap_err().to_string();
    assert!(error.contains("cuttoff"), "unexpected error: {error}");

    // The old `[elements.Z]` table form is rejected, not silently ignored.
    let text = r#"
radial = "treutler_ahlrichs"

[default]
n_rad = 20
radial_scale = 0.9
angular = { fixed = { n_ang = 50 } }

[elements.8]
n_rad = 75
"#;
    let error = toml::from_str::<GridPreset>(text).unwrap_err().to_string();
    assert!(error.contains("elements"), "unexpected error: {error}");
}
