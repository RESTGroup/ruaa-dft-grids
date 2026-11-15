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

const TINY_ENTRY: &str =
    r#"{"name": "tiny", "category": "xc", "file": "tiny.json", "elements": [1, 2]}"#;
const SCHEDULE_ENTRY: &str =
    r#"{"name": "schedule", "category": "response", "file": "schedule.json", "kind": "schedule"}"#;

fn catalog_json(entries: &str) -> String {
    format!(
        r#"{{"format": 1, "description": "test catalog", "programs": {{"demo": {{"software": "Demo 1.0", "presets": [{entries}]}}}}}}"#
    )
}

fn valid_catalog_json() -> String {
    catalog_json(&format!("{TINY_ENTRY}, {SCHEDULE_ENTRY}"))
}

/// A catalog with every index problem: an unsafe path, a duplicate file and a
/// grid entry without an element range.  Used through `from_json_str` so the
/// load path can be tested entry by entry.
fn bad_catalog_json() -> String {
    let escape =
        r#"{"name": "escape", "category": "xc", "file": "../escape.json", "elements": [1, 2]}"#;
    let noe = r#"{"name": "noe", "category": "xc", "file": "tiny.json"}"#;
    catalog_json(&format!("{TINY_ENTRY}, {escape}, {noe}, {SCHEDULE_ENTRY}"))
}

fn temp_dir() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "ruaa-dft-grids-catalog-{}-{nanos}",
        std::process::id()
    ));
    let data = root.join("data");
    std::fs::create_dir_all(&data).unwrap();
    data
}

fn cleanup(dir: &Path) {
    std::fs::remove_dir_all(dir.parent().unwrap()).unwrap();
}

fn write_catalog(dir: &Path, manifest: &str) -> PathBuf {
    std::fs::write(dir.join("tiny.json"), TINY_PRESET).unwrap();
    let path = dir.join("manifest.json");
    std::fs::write(&path, manifest).unwrap();
    path
}

#[test]
fn catalog_entries_carry_the_metadata() {
    let catalog = PresetCatalog::from_json_str(&valid_catalog_json()).unwrap();
    let entry = catalog.entry("demo", "tiny").unwrap();
    assert_eq!(entry.kind, PresetKind::Grid);
    assert_eq!(entry.element_range(), Some((1, 2)));
    assert!(entry.is_grid());
    assert_eq!(catalog.grid_entries("demo").count(), 1);
    assert!(catalog.entry("demo", "missing").is_none());
}

#[test]
fn catalog_loads_a_grid_and_applies_the_element_range() {
    let dir = temp_dir();
    let path = write_catalog(&dir, &valid_catalog_json());
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
    cleanup(&dir);
}

#[test]
fn from_json_path_validates_the_index() {
    let dir = temp_dir();

    let duplicate_name = catalog_json(&format!(
        r#"{TINY_ENTRY}, {{"name": "tiny", "category": "xc", "file": "other.json", "elements": [1, 2]}}"#
    ));
    let error = PresetCatalog::from_json_path(&write_catalog(&dir, &duplicate_name))
        .unwrap_err()
        .to_string();
    assert!(error.contains("duplicate preset name"), "{error}");

    let duplicate_file = catalog_json(&format!(
        r#"{TINY_ENTRY}, {{"name": "other", "category": "xc", "file": "tiny.json", "elements": [1, 2]}}"#
    ));
    let error = PresetCatalog::from_json_path(&write_catalog(&dir, &duplicate_file))
        .unwrap_err()
        .to_string();
    assert!(error.contains("reuses file"), "{error}");

    let unsafe_path = catalog_json(&format!(
        r#"{TINY_ENTRY}, {{"name": "escape", "category": "xc", "file": "../escape.json", "elements": [1, 2]}}"#
    ));
    let error = PresetCatalog::from_json_path(&write_catalog(&dir, &unsafe_path))
        .unwrap_err()
        .to_string();
    assert!(error.contains("may not escape"), "{error}");

    let no_elements = catalog_json(&format!(
        r#"{TINY_ENTRY}, {{"name": "noe", "category": "xc", "file": "other.json"}}"#
    ));
    let error = PresetCatalog::from_json_path(&write_catalog(&dir, &no_elements))
        .unwrap_err()
        .to_string();
    assert!(error.contains("no element range"), "{error}");

    cleanup(&dir);
}

#[test]
fn grid_entries_must_declare_an_element_range() {
    let dir = temp_dir();
    std::fs::write(dir.join("tiny.json"), TINY_PRESET).unwrap();
    let catalog = PresetCatalog::from_json_str(&bad_catalog_json()).unwrap();
    let error = catalog
        .load_preset("demo", "noe", &dir)
        .unwrap_err()
        .to_string();
    assert!(error.contains("no element range"), "{error}");
    cleanup(&dir);
}

#[test]
fn schedule_entries_are_not_grids() {
    let dir = temp_dir();
    let path = write_catalog(&dir, &valid_catalog_json());
    let catalog = PresetCatalog::from_json_path(&path).unwrap();
    assert!(matches!(
        catalog.load_preset("demo", "schedule", &dir),
        Err(Error::UnsupportedGrid { .. })
    ));
    cleanup(&dir);
}

#[test]
fn entry_paths_cannot_escape_the_data_directory() {
    let dir = temp_dir();
    let catalog = PresetCatalog::from_json_str(&bad_catalog_json()).unwrap();
    assert!(matches!(
        catalog.load_preset("demo", "escape", &dir),
        Err(Error::Catalog { .. })
    ));
    // Even if the target exists above the data directory, the loader refuses.
    let above = dir.parent().unwrap().join("escape.json");
    std::fs::write(&above, TINY_PRESET).unwrap();
    assert!(matches!(
        catalog.load_preset("demo", "escape", &dir),
        Err(Error::Catalog { .. })
    ));
    cleanup(&dir);
}

#[test]
fn missing_entries_and_bad_formats_are_errors() {
    let catalog = PresetCatalog::from_json_str(&valid_catalog_json()).unwrap();
    assert!(matches!(
        catalog.load_preset("demo", "missing", Path::new(".")),
        Err(Error::CatalogEntryNotFound { .. })
    ));
    let bad = valid_catalog_json().replace("\"format\": 1", "\"format\": 2");
    assert!(matches!(
        PresetCatalog::from_json_str(&bad),
        Err(Error::Catalog { .. })
    ));
}

#[test]
fn unknown_keys_are_rejected() {
    let bad_program = valid_catalog_json().replace("\"software\"", "\"softwre\"");
    let error = PresetCatalog::from_json_str(&bad_program)
        .unwrap_err()
        .to_string();
    assert!(error.contains("softwre"), "{error}");

    let bad_entry = valid_catalog_json().replace("\"elements\"", "\"elemnts\"");
    let error = PresetCatalog::from_json_str(&bad_entry)
        .unwrap_err()
        .to_string();
    assert!(error.contains("elemnts"), "{error}");

    let bad_preset = TINY_PRESET.replace(
        "\"radial_scale\": 0.8",
        "\"radial_scale\": 0.8, \"cuttoff\": 9.0",
    );
    let error = serde_json::from_str::<ruaa_dft_grids::preset::GridPreset>(&bad_preset)
        .unwrap_err()
        .to_string();
    assert!(error.contains("cuttoff"), "{error}");
}

#[test]
fn catalog_range_cannot_widen_the_preset_domain() {
    let dir = temp_dir();
    let wide = catalog_json(
        r#"{"name": "tiny", "category": "xc", "file": "tiny.json", "elements": [1, 4]}"#,
    );
    let path = write_catalog(&dir, &wide);
    let catalog = PresetCatalog::from_json_path(&path).unwrap();
    let error = catalog
        .load_preset("demo", "tiny", &dir)
        .unwrap_err()
        .to_string();
    assert!(error.contains("exceeds the preset domain"), "{error}");
    cleanup(&dir);
}

#[test]
fn preset_name_must_match_the_entry_name() {
    let dir = temp_dir();
    let mismatch = catalog_json(
        r#"{"name": "other", "category": "xc", "file": "tiny.json", "elements": [1, 2]}"#,
    );
    let path = write_catalog(&dir, &mismatch);
    let catalog = PresetCatalog::from_json_path(&path).unwrap();
    let error = catalog
        .load_preset("demo", "other", &dir)
        .unwrap_err()
        .to_string();
    assert!(error.contains("does not match catalog entry"), "{error}");
    cleanup(&dir);
}

#[test]
fn validation_errors_name_the_entry_and_file() {
    let dir = temp_dir();
    let path = write_catalog(&dir, &valid_catalog_json());
    let bad_preset = TINY_PRESET.replace("\"n_ang\": 50", "\"n_ang\": 999");
    std::fs::write(dir.join("tiny.json"), bad_preset).unwrap();
    let catalog = PresetCatalog::from_json_path(&path).unwrap();
    let error = catalog
        .load_preset("demo", "tiny", &dir)
        .unwrap_err()
        .to_string();
    assert!(error.contains("tiny") && error.contains("999"), "{error}");
    cleanup(&dir);
}

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

const NARROW_PRESET: &str = r#"{
  "name": "narrow",
  "z_min": 1,
  "z_max": 4,
  "radial": "treutler_ahlrichs",
  "default": {
    "n_rad": 20,
    "radial_scale": 1.0,
    "angular": {"fixed": {"n_ang": 50}}
  },
  "range": [
    {"z_min": 1, "z_max": 4, "n_rad": 30, "radial_scale": 1.0,
     "angular": {"fixed": {"n_ang": 50}}}
  ],
  "element": [
    {"z": 1, "radial_scale": 1.1},
    {"z": 4, "radial_scale": 1.4}
  ]
}"#;

#[test]
fn a_narrower_catalog_range_clips_file_data() {
    let dir = temp_dir();
    std::fs::write(dir.join("narrow.json"), NARROW_PRESET).unwrap();
    let manifest = catalog_json(
        r#"{"name": "narrow", "category": "xc", "file": "narrow.json", "elements": [2, 3]}"#,
    );
    let path = dir.join("manifest.json");
    std::fs::write(&path, manifest).unwrap();
    let catalog = PresetCatalog::from_json_path(&path).unwrap();
    let preset = catalog.load_preset("demo", "narrow", &dir).unwrap();
    assert_eq!((preset.z_min, preset.z_max), (2, 3));
    assert_eq!(preset.defaults.len(), 1);
    assert_eq!((preset.defaults[0].z_min, preset.defaults[0].z_max), (2, 3));
    assert!(!preset.elements.contains_key(&1));
    assert!(!preset.elements.contains_key(&4));
    assert_eq!(preset.element(2).unwrap().n_rad, 30);
    assert!(preset.atomic_grid(2).is_ok());
    cleanup(&dir);
}

#[cfg(unix)]
#[test]
fn a_symlinked_preset_cannot_escape_the_data_directory() {
    let dir = temp_dir();
    let outside = dir.parent().unwrap().join("outside.json");
    std::fs::write(&outside, TINY_PRESET).unwrap();
    std::os::unix::fs::symlink(&outside, dir.join("link.json")).unwrap();
    let manifest = catalog_json(
        r#"{"name": "tiny", "category": "xc", "file": "link.json", "elements": [1, 2]}"#,
    );
    let path = dir.join("manifest.json");
    std::fs::write(&path, manifest).unwrap();
    let catalog = PresetCatalog::from_json_path(&path).unwrap();
    let error = catalog
        .load_preset("demo", "tiny", &dir)
        .unwrap_err()
        .to_string();
    assert!(error.contains("escapes the data directory"), "{error}");
    cleanup(&dir);
}
