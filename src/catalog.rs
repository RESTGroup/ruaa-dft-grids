//! Caller-supplied preset catalogs.
//!
//! A [`GridPreset`] describes one atomic topology.  A **catalog** describes a
//! directory tree of preset files: the program, the grid family, the file
//! path, the element domain and free-form provenance labels.  The catalog is
//! ordinary caller data; this crate ships no catalog and no program tables.
//!
//! Both the catalog and the preset files it references are JSON.  TOML is
//! the human-authoring form described on the
//! [presets page](crate::doc::presets); a catalog entry that points at a
//! `.toml` file fails to load.  Unknown keys are rejected at every level.
//!
//! A catalog has this shape (the values are the caller's):
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
//! an element outside the program's coverage.  A narrower range drops element
//! and range data outside it; a wider range is rejected.  A `"kind":
//! "schedule"` entry is metadata (for example a response-grid schedule) and is
//! not a [`GridPreset`]; loading it returns [`Error::UnsupportedGrid`].
//!
//! The catalog loader never embeds a path: the caller supplies the catalog
//! path and the directory that the entry paths are relative to.  Entry paths
//! must be relative and may not contain `..`; the canonical target, including
//! symlinks, must stay under the data directory.
//! [`PresetCatalog::from_json_path`] validates the index
//! (unique names and paths, a valid element range on every grid entry);
//! [`PresetCatalog::from_json_str`] parses only, so an embedded or partially
//! built catalog can be inspected with [`PresetCatalog::validate`] separately.
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
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
    /// Parse a catalog from JSON text without validating the index.
    ///
    /// # Errors
    /// [`Error::Catalog`] for malformed JSON, an unknown key or an
    /// unsupported format version.
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

    /// Read, parse and validate a catalog file.
    ///
    /// # Errors
    /// [`Error::Catalog`] for an I/O or parse failure, or for a failed
    /// [`Self::validate`] check.
    pub fn from_json_path(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|error| Error::Catalog {
            message: format!("{}: {error}", path.display()),
        })?;
        let catalog = Self::from_json_str(&text).map_err(|error| match error {
            Error::Catalog { message } => Error::Catalog {
                message: format!("{}: {message}", path.display()),
            },
            other => other,
        })?;
        catalog.validate().map_err(|error| match error {
            Error::Catalog { message } => Error::Catalog {
                message: format!("{}: {message}", path.display()),
            },
            other => other,
        })?;
        Ok(catalog)
    }

    /// Validate the catalog index without reading the preset files.
    ///
    /// Checks that every grid entry declares a valid element range, that
    /// entry names and file paths are unique within a program, and that every
    /// path is relative and contained.  Use [`Self::load_preset`] (or
    /// [`PresetEntry::load`]) to check the referenced files themselves.
    ///
    /// [`Self::from_json_path`] runs this check; [`Self::from_json_str`] does
    /// not, so tests and embedded catalogs can be inspected before they are
    /// complete.
    ///
    /// # Errors
    /// [`Error::Catalog`] for every failed check.
    pub fn validate(&self) -> Result<()> {
        if self.format != 1 {
            return Err(Error::Catalog {
                message: format!("unsupported catalog format {} (expected 1)", self.format),
            });
        }
        for (program, catalog) in &self.programs {
            let mut names = std::collections::BTreeSet::new();
            let mut files = std::collections::BTreeSet::new();
            for entry in &catalog.presets {
                if !names.insert(entry.name.as_str()) {
                    return Err(Error::Catalog {
                        message: format!(
                            "program `{program}`: duplicate preset name `{}`",
                            entry.name
                        ),
                    });
                }
                if !files.insert(entry.file.as_str()) {
                    return Err(Error::Catalog {
                        message: format!(
                            "program `{program}`: preset `{}` reuses file `{}`",
                            entry.name, entry.file
                        ),
                    });
                }
                resolve_entry_path(Path::new("."), &entry.file)?;
                if entry.is_grid() {
                    let Some([z_min, z_max]) = entry.elements else {
                        return Err(Error::Catalog {
                            message: format!(
                                "program `{program}`: grid entry `{}` has no element range",
                                entry.name
                            ),
                        });
                    };
                    if z_min == 0 || z_min > z_max || z_max > 118 {
                        return Err(Error::Catalog {
                            message: format!(
                                "program `{program}`: grid entry `{}` has invalid element range [{z_min}, {z_max}]",
                                entry.name
                            ),
                        });
                    }
                }
            }
        }
        Ok(())
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
    /// [`Error::Catalog`] for a missing, unreadable or malformed preset
    /// file, a name/domain mismatch, or a failed preset validation.
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
    ///
    /// An unknown program yields an empty iterator; use [`Self::program`] when
    /// the caller needs to distinguish a missing program from an empty one.
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
    /// [`Error::UnsupportedGrid`] for a non-grid entry, and [`Error::Catalog`]
    /// for an I/O or parse failure, an escaping path or symlink, a preset
    /// name that differs from the entry name, a catalog range that exceeds
    /// the preset's own domain, or a failed preset validation (the message
    /// carries the file path and entry name).  A narrower catalog range
    /// clips the preset's element and range data before validation.
    pub fn load(&self, base_dir: &Path) -> Result<GridPreset> {
        if !self.is_grid() {
            return Err(Error::UnsupportedGrid {
                name: self.name.clone(),
                reason: "the catalog entry is not a grid preset".to_string(),
            });
        }
        let path = resolve_entry_path(base_dir, &self.file)?;
        ensure_contained(base_dir, &path)?;
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
        let text = std::fs::read_to_string(&path).map_err(|error| Error::Catalog {
            message: format!("{}: {error}", path.display()),
        })?;
        let mut preset: GridPreset =
            serde_json::from_str(&text).map_err(|error| Error::Catalog {
                message: format!("{}: {error}", path.display()),
            })?;
        if !preset.name.is_empty() && preset.name != self.name {
            return Err(Error::Catalog {
                message: format!(
                    "{}: preset name `{}` does not match catalog entry `{}`",
                    path.display(),
                    preset.name,
                    self.name
                ),
            });
        }
        // Check the file on its own terms before applying the catalog window.
        preset.validate().map_err(|error| Error::Catalog {
            message: format!("{}: preset `{}`: {error}", path.display(), self.name),
        })?;
        // The catalog cannot widen the coverage the file itself declares:
        // that would silently apply the file's default block beyond its domain.
        if z_min < preset.z_min || z_max > preset.z_max {
            return Err(Error::Catalog {
                message: format!(
                    "{}: catalog range [{z_min}, {z_max}] exceeds the preset domain {}..={}",
                    path.display(),
                    preset.z_min,
                    preset.z_max
                ),
            });
        }
        if z_min > preset.z_min || z_max < preset.z_max {
            // A narrower catalog window is authoritative: drop the element and
            // range data that falls outside it, then validate the restricted
            // preset.  Clipping (rather than rejecting) lets one shared preset
            // serve programs with different coverage.
            preset.defaults = std::mem::take(&mut preset.defaults)
                .into_iter()
                .filter_map(|mut range| {
                    if range.z_max < z_min || range.z_min > z_max {
                        return None;
                    }
                    range.z_min = range.z_min.max(z_min);
                    range.z_max = range.z_max.min(z_max);
                    Some(range)
                })
                .collect();
            preset.elements.retain(|z, _| (z_min..=z_max).contains(z));
        }
        preset.z_min = z_min;
        preset.z_max = z_max;
        preset.validate().map_err(|error| Error::Catalog {
            message: format!("{}: preset `{}`: {error}", path.display(), self.name),
        })?;
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

/// Reject a resolved path whose canonical target leaves `base_dir`.
///
/// `resolve_entry_path` is lexical; this check also covers symlinked files and
/// directories.  A path that does not exist is left to the subsequent read,
/// which reports the I/O error with the file name.
fn ensure_contained(base_dir: &Path, path: &Path) -> Result<()> {
    let (Ok(base), Ok(target)) = (base_dir.canonicalize(), path.canonicalize()) else {
        return Ok(());
    };
    if target.starts_with(&base) {
        return Ok(());
    }
    Err(Error::Catalog {
        message: format!(
            "{}: preset path escapes the data directory {}",
            path.display(),
            base.display()
        ),
    })
}
