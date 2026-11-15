//! Preset catalog loading.
//!
//! The catalog and the preset files are caller data.  The in-repo tests use a
//! tiny temporary catalog; the external test runs only when
//! `RUAA_PRESET_DATA` points at a data directory with a `manifest.json`, so
//! the repository carries no program data or local path.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use ruaa_dft_grids::Error;
use ruaa_dft_grids::catalog::{PresetCatalog, PresetKind};

const TINY_PRESET: &str = r#"{
  "name": "tiny",
  "z_min": 1,
  "z_max": 2,
  "radial": "treutler_ahlrichs",
  "default": {
    "n_rad": 20,
    "radial_scale": 0.8,
    "angular": {"fixed": {"n_ang": 50}}
  }
}"#;

fn catalog_json() -> String {
    r#"{
      "format": 1,
      "description": "test catalog",
      "programs": {
        "demo": {
          "software": "Demo 1.0",
          "reference_basis": "none",
          "presets": [
            {
              "name": "tiny",
              "category": "xc",
              "file": "tiny.json",
              "kind": "grid",
              "grid": "tiny grid",
              "elements": [1, 2]
            },
            {
              "name": "escape",
              "category": "xc",
              "file": "../escape.json",
              "elements": [1, 2]
            },
            {
              "name": "noe",
              "category": "xc",
              "file": "tiny.json"
            },
            {
              "name": "schedule",
              "category": "response",
              "file": "schedule.json",
              "kind": "schedule"
            }
          ]
        }
      }
    }"#
    .to_string()
}

fn temp_dir() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "ruaa-dft-grids-catalog-{}-{nanos}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_catalog(dir: &Path) -> PathBuf {
    std::fs::write(dir.join("tiny.json"), TINY_PRESET).unwrap();
    let path = dir.join("manifest.json");
    std::fs::write(&path, catalog_json()).unwrap();
    path
}

#[test]
fn catalog_entries_carry_the_metadata() {
    let catalog = PresetCatalog::from_json_str(&catalog_json()).unwrap();
    let entry = catalog.entry("demo", "tiny").unwrap();
    assert_eq!(entry.kind, PresetKind::Grid);
    assert_eq!(entry.element_range(), Some((1, 2)));
    assert!(entry.is_grid());
    assert_eq!(catalog.grid_entries("demo").count(), 3);
    assert!(catalog.entry("demo", "missing").is_none());
}

#[test]
fn catalog_loads_a_grid_and_applies_the_element_range() {
    let dir = temp_dir();
    let path = write_catalog(&dir);
    let catalog = PresetCatalog::from_json_path(&path).unwrap();
    let preset = catalog.load_preset("demo", "tiny", &dir).unwrap();
    assert_eq!(preset.z_min, 1);
    assert_eq!(preset.z_max, 2);
    assert!(preset.atomic_grid(1).unwrap().nr() > 0);
    assert!(matches!(
        preset.atomic_grid(3),
        Err(Error::MissingElementData {
            z: 3,
            table: "grid preset"
        })
    ));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn grid_entries_must_declare_an_element_range() {
    let dir = temp_dir();
    let path = write_catalog(&dir);
    let catalog = PresetCatalog::from_json_path(&path).unwrap();
    assert!(matches!(
        catalog.load_preset("demo", "noe", &dir),
        Err(Error::Catalog { .. })
    ));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn schedule_entries_are_not_grids() {
    let dir = temp_dir();
    let path = write_catalog(&dir);
    let catalog = PresetCatalog::from_json_path(&path).unwrap();
    assert!(matches!(
        catalog.load_preset("demo", "schedule", &dir),
        Err(Error::UnsupportedGrid { .. })
    ));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn entry_paths_cannot_escape_the_data_directory() {
    let dir = temp_dir();
    let path = write_catalog(&dir);
    let catalog = PresetCatalog::from_json_path(&path).unwrap();
    assert!(matches!(
        catalog.load_preset("demo", "escape", &dir),
        Err(Error::Catalog { .. })
    ));
    // Even if the target exists above the data directory, the loader refuses.
    std::fs::write(dir.parent().unwrap().join("escape.json"), TINY_PRESET).unwrap();
    assert!(matches!(
        catalog.load_preset("demo", "escape", &dir),
        Err(Error::Catalog { .. })
    ));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn missing_entries_and_bad_formats_are_errors() {
    let catalog = PresetCatalog::from_json_str(&catalog_json()).unwrap();
    assert!(matches!(
        catalog.load_preset("demo", "missing", Path::new(".")),
        Err(Error::CatalogEntryNotFound { .. })
    ));
    let bad = catalog_json().replace("\"format\": 1", "\"format\": 2");
    assert!(matches!(
        PresetCatalog::from_json_str(&bad),
        Err(Error::Catalog { .. })
    ));
}

/// Optional external data check.  Run it explicitly with
/// `RUAA_PRESET_DATA=/path/to/data cargo test --test preset_catalog`.
#[test]
fn external_catalog_loads_when_configured() {
    let Ok(dir) = std::env::var("RUAA_PRESET_DATA") else {
        return;
    };
    let dir = Path::new(&dir);
    let catalog = PresetCatalog::from_json_path(&dir.join("manifest.json")).unwrap();
    let mut checked = 0usize;
    for (program, program_catalog) in &catalog.programs {
        for entry in &program_catalog.presets {
            if !entry.is_grid() {
                continue;
            }
            let preset = entry.load(dir).unwrap();
            let [z_min, z_max] = entry.elements.expect("grid entry has an element range");
            for z in z_min..=z_max {
                preset
                    .atomic_grid(z)
                    .unwrap_or_else(|error| panic!("{program}/{}: Z={z}: {error}", entry.name));
            }
            if z_max < 118 {
                assert!(
                    preset.atomic_grid(z_max + 1).is_err(),
                    "{program}/{}: domain not enforced above Z={z_max}",
                    entry.name
                );
            }
            checked += 1;
        }
    }
    assert!(checked > 0, "external catalog has no grid entries");
    println!("external catalog: {checked} grid presets");
}
