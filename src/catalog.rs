//! Caller-supplied preset catalogs.
//!
//! A [`GridPreset`] describes one atomic topology.  A **catalog** describes a
//! directory tree of preset files: the program, the grid family, the file
//! path, the element domain and free-form provenance labels.  The catalog is
//! ordinary caller data; this crate ships no catalog and no program tables.
//!
//! The file format is JSON.  A catalog has this shape (the values are the
//! caller's):
//!
//! ```json
//! {
//!   "format": 1,
//!   "programs": {
//!     "demo": {
//!       "software": "Demo 1.0",
//!       "reference_basis": "example",
//!       "parameters": "demo/parameters.toml",
//!       "presets": [
//!         {
//!           "name": "demo-grid",
//!           "category": "xc",
//!           "file": "demo/xc/demo-grid.json",
//!           "kind": "grid",
//!           "grid": "default grid",
//!           "elements": [1, 10]
//!         }
//!       ]
//!     }
//!   }
//! }
//! ```
//!
//! [`PresetEntry::load`] reads one referenced file and returns a
//! [`GridPreset`].  The entry's `elements` range is authoritative: it is
//! copied into the preset's [`GridPreset::z_min`] / [`GridPreset::z_max`]
//! domain, so the catalog cannot accidentally apply a shared default block to
//! an element outside the program's coverage.  A `"kind": "schedule"` entry is
//! metadata (for example a response-grid schedule) and is not a
//! [`GridPreset`]; loading it returns [`Error::UnsupportedGrid`].
//!
//! The catalog loader never embeds a path: the caller supplies the catalog
//! path and the directory that the entry paths are relative to.  Entry paths
//! must be relative and may not contain `..`, so a catalog cannot escape its
//! data directory.
//!
//! # Example
//!
//! ```
//! use ruaa_dft_grids::catalog::PresetCatalog;
//!
//! # fn main() -> ruaa_dft_grids::Result<()> {
//! let text = r#"{
//!   "format": 1,
//!   "programs": {
//!     "demo": {
//!       "software": "Demo 1.0",
//!       "presets": [
//!         {"name": "grid", "category": "xc", "file": "grid.json",
//!          "elements": [1, 10]}
//!       ]
//!     }
//!   }
//! }"#;
//! let catalog = PresetCatalog::from_json_str(text)?;
//! let entry = catalog.entry("demo", "grid").expect("catalog entry");
//! assert_eq!(entry.elements, Some([1, 10]));
//! assert!(entry.is_grid());
//! # Ok(())
//! # }
//! ```

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

use crate::error::{Error, Result};
use crate::preset::GridPreset;

/// One preset catalog.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PresetCatalog {
    /// Catalog format version.  Version 1 is the only version currently
    /// understood.
    pub format: u32,
    /// Free-form catalog description.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// Programs, keyed by a caller-chosen program key (for example `demo`).
    pub programs: BTreeMap<String, ProgramCatalog>,
}

/// One program's preset entries.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ProgramCatalog {
    /// Program name and version for humans.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub software: String,
    /// Reference basis of the grid tables.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub reference_basis: String,
    /// Optional path to the raw parameter table used to build the presets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<String>,
    /// Preset entries.
    #[serde(default)]
    pub presets: Vec<PresetEntry>,
}

/// What a catalog entry refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum PresetKind {
    /// A [`GridPreset`] file.
    #[default]
    Grid,
    /// A metadata file (for example a response-grid schedule), not a grid.
    Schedule,
}

/// One catalog entry.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PresetEntry {
    /// Preset name, unique within the program.
    pub name: String,
    /// Grid family (for example `xc`, `cosx`, `response`).
    pub category: String,
    /// Path relative to the catalog's data directory.
    pub file: String,
    /// Entry kind.  Defaults to [`PresetKind::Grid`].
    #[serde(default)]
    pub kind: PresetKind,
    /// Human-readable grid description.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub grid: String,
    /// Optional program grid scheme number.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scheme: Option<u8>,
    /// Element domain `[z_min, z_max]`.  A grid entry must set it; a
    /// schedule entry omits it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elements: Option<[u32; 2]>,
    /// Reference basis of this entry, when it differs from the program's.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub reference_basis: String,
    /// Optional basis label (free-form caller metadata).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub basis: Option<String>,
    /// Optional basis-profile label (free-form caller metadata).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub basis_profile: Option<String>,
    /// Free-form notes.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub notes: String,
}

impl PresetCatalog {
    /// Parse a catalog from JSON text.
    ///
    /// # Errors
    /// [`Error::Catalog`] for malformed JSON or an unsupported format
    /// version.
    pub fn from_json_str(text: &str) -> Result<Self> {
        let catalog: Self = serde_json::from_str(text).map_err(|error| Error::Catalog {
            message: format!("catalog JSON: {error}"),
        })?;
        if catalog.format != 1 {
            return Err(Error::Catalog {
                message: format!("unsupported catalog format {} (expected 1)", catalog.format),
            });
        }
        Ok(catalog)
    }

    /// Read and parse a catalog file.
    ///
    /// # Errors
    /// [`Error::Catalog`] for an I/O or parse failure.
    pub fn from_json_path(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|error| Error::Catalog {
            message: format!("{}: {error}", path.display()),
        })?;
        Self::from_json_str(&text).map_err(|error| match error {
            Error::Catalog { message } => Error::Catalog {
                message: format!("{}: {message}", path.display()),
            },
            other => other,
        })
    }

    /// One program's catalog.
    #[must_use]
    pub fn program(&self, program: &str) -> Option<&ProgramCatalog> {
        self.programs.get(program)
    }

    /// One entry, looked up by program key and preset name.
    #[must_use]
    pub fn entry(&self, program: &str, name: &str) -> Option<&PresetEntry> {
        self.program(program)
            .and_then(|catalog| catalog.entry(name))
    }

    /// Load one referenced [`GridPreset`].
    ///
    /// `base_dir` is the directory that the entry's `file` is relative to.
    /// The entry's `elements` range is applied to the loaded preset.
    ///
    /// # Errors
    /// [`Error::CatalogEntryNotFound`] for an unknown name, and
    /// [`Error::Catalog`] for a missing/unreadable/malformed preset file.
    /// A non-grid entry returns [`Error::UnsupportedGrid`].
    pub fn load_preset(&self, program: &str, name: &str, base_dir: &Path) -> Result<GridPreset> {
        self.entry(program, name)
            .ok_or_else(|| Error::CatalogEntryNotFound {
                program: program.to_string(),
                name: name.to_string(),
            })?
            .load(base_dir)
    }

    /// Every grid entry of one program.
    pub fn grid_entries(&self, program: &str) -> impl Iterator<Item = &PresetEntry> {
        self.programs
            .get(program)
            .into_iter()
            .flat_map(|catalog| catalog.presets.iter())
            .filter(|entry| entry.is_grid())
    }
}

impl ProgramCatalog {
    /// One entry by preset name.
    #[must_use]
    pub fn entry(&self, name: &str) -> Option<&PresetEntry> {
        self.presets.iter().find(|entry| entry.name == name)
    }
}

impl PresetEntry {
    /// Whether this entry is a grid preset.
    #[must_use]
    pub fn is_grid(&self) -> bool {
        self.kind == PresetKind::Grid
    }

    /// The entry's element domain, when it declares one.
    #[must_use]
    pub fn element_range(&self) -> Option<(u32, u32)> {
        self.elements.map(|[z_min, z_max]| (z_min, z_max))
    }

    /// Load this entry as a [`GridPreset`].
    ///
    /// # Errors
    /// [`Error::UnsupportedGrid`] for a non-grid entry, [`Error::Catalog`]
    /// for an I/O or parse failure, and the preset validation errors after
    /// the element range is applied.
    pub fn load(&self, base_dir: &Path) -> Result<GridPreset> {
        if !self.is_grid() {
            return Err(Error::UnsupportedGrid {
                name: self.name.clone(),
                reason: "the catalog entry is not a grid preset".to_string(),
            });
        }
        let path = resolve_entry_path(base_dir, &self.file)?;
        let text = std::fs::read_to_string(&path).map_err(|error| Error::Catalog {
            message: format!("{}: {error}", path.display()),
        })?;
        let mut preset: GridPreset =
            serde_json::from_str(&text).map_err(|error| Error::Catalog {
                message: format!("{}: {error}", path.display()),
            })?;
        let Some([z_min, z_max]) = self.elements else {
            return Err(Error::Catalog {
                message: format!(
                    "{}: grid entry `{}` has no element range",
                    path.display(),
                    self.name
                ),
            });
        };
        if z_min == 0 || z_min > z_max || z_max > 118 {
            return Err(Error::Catalog {
                message: format!(
                    "{}: invalid element range [{z_min}, {z_max}]",
                    path.display()
                ),
            });
        }
        preset.z_min = z_min;
        preset.z_max = z_max;
        preset.validate()?;
        Ok(preset)
    }
}

/// Resolve an entry path below `base_dir`, rejecting absolute paths and
/// parent-directory components.
fn resolve_entry_path(base_dir: &Path, file: &str) -> Result<PathBuf> {
    let relative = Path::new(file);
    if relative.is_absolute() {
        return Err(Error::Catalog {
            message: format!("preset path `{file}` must be relative"),
        });
    }
    if relative.components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    }) {
        return Err(Error::Catalog {
            message: format!("preset path `{file}` may not escape the data directory"),
        });
    }
    Ok(base_dir.join(relative))
}
