//! Data-driven grid presets.
//!
//! A [`GridPreset`] is the program-independent description of an atomic
//! topology: a radial node map, one [`ElementPreset`] per nuclear charge, and
//! an optional default for elements without an explicit entry.  It contains
//! no program-specific tables.  Callers either construct the data in Rust or
//! deserialize it from a preset file (the preset types implement `serde`), and
//! combine it with a [`crate::partition::PartitionConfig`] to form a
//! [`crate::grid::GridSpec`].
//!
//! The [presets page](crate::doc::presets) has a complete worked example:
//! PySCF's level-3 preset as data, its full TOML file and a PySCF
//! cross-check.
//!
//! # Why data-driven
//!
//! The radial node map is a mathematical construction, but the shell counts,
//! radial scales, cutoffs and per-shell angular sizes of a production grid are
//! a *data set*.  Making those values explicit keeps the core crate free of
//! program-specific parameter tables while still allowing an exact
//! reproduction: a caller who has the parameter data supplies it as a
//! [`GridPreset`] and gets the same points, pruning and weights.
//!
//! # Preset files
//!
//! With the `serde` feature (always enabled by this crate) the preset types
//! can be read with any `serde` format:
//!
//! ```toml
//! name = "example"
//! radial = "treutler_ahlrichs"
//! molecular_cutoffs = false
//!
//! [default]
//! n_rad = 50
//! radial_scale = 1.2
//! angular = { pruned = { base = 194, prune = "nwchem" } }
//!
//! [elements.8]
//! n_rad = 75
//! radial_scale = 1.1
//! cutoff = 12.0
//! angular = { shells = [50, 110, 194, 302] }
//! ```
//!
//! The exact file layout is the caller's: `ruaa_dft_grids` exposes the data
//! types and their invariants, not a file-location convention.  Program
//! presets use this format as ordinary caller data.
//!
//! # Generated grids
//!
//! [`GridPreset::atomic_grid`] generates one element's [`AtomicGrid`].  The
//! radial part uses [`crate::radial::shells`] with the preset scale and
//! cutoff; the angular part is one of:
//!
//! * a fixed rule size for every shell ([`AngularLayout::Fixed`]);
//! * a uniform base plus a [`crate::prune::Prune`] scheme
//!   ([`AngularLayout::Pruned`]);
//! * an explicit per-shell size list ([`AngularLayout::Shells`]), for grids
//!   whose pruning is not one of the built-in policies;
//! * a Gauss–Legendre x uniform-phi product rule
//!   ([`AngularLayout::Product`]).
//!
//! A construction that is not expressible this way can still use
//! [`crate::atomgrid::AtomicGrid`] directly and feed the assembled grids to
//! [`crate::grid::GridPlan::from_atomic_grids`].

use std::collections::BTreeMap;

use crate::angular::{
    angular_grid, is_supported_size, product_angular_grid, product_angular_grid_adaptive,
};
use crate::atomgrid::{AtomicGrid, assemble_atomic_grid};
use crate::error::{Error, Result};
use crate::molecule::Molecule;
use crate::prune::{Prune, PruneContext};
use crate::radial::{Radial, RadialShell, shells};

fn default_true() -> bool {
    true
}

fn default_z_min() -> u32 {
    1
}

fn default_z_max() -> u32 {
    118
}

fn is_default_z_min(value: &u32) -> bool {
    *value == 1
}

fn is_default_z_max(value: &u32) -> bool {
    *value == 118
}

fn is_true(value: &bool) -> bool {
    *value
}

fn is_false(value: &bool) -> bool {
    !*value
}

fn is_empty_string(value: &str) -> bool {
    value.is_empty()
}

/// Angular layout of the shells of one atom.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum AngularLayout {
    /// The same Lebedev (or special 12/18-point) rule on every shell.
    Fixed {
        /// Number of angular points per shell.
        n_ang: usize,
    },
    /// A uniform base size plus a pruning policy.
    Pruned {
        /// Unpruned base angular size.
        base: usize,
        /// Pruning policy that maps each shell radius to an angular size.
        prune: Prune,
    },
    /// An explicit angular size for every active radial shell.
    Shells(Vec<usize>),
    /// A Gauss–Legendre theta x uniform-phi product rule.
    Product {
        /// Number of theta rings; each ring carries `2 * theta` phi points.
        theta: usize,
    },
    /// One Gauss–Legendre theta-ring count per shell, with the per-ring phi
    /// count chosen adaptively (smallest `p >= 2` with
    /// `sin(theta)^p <= 1e-8`, capped at `2 * theta`).
    ///
    /// This is the product construction used by ORCA's `AngularGrid 0`
    /// (`SimpleGrid`); it is a generic variable-phi product rule, not a
    /// program table.
    ProductShells(Vec<usize>),
}

impl AngularLayout {
    /// Fixed rule size on every shell.
    #[must_use]
    pub fn fixed(n_ang: usize) -> Self {
        Self::Fixed { n_ang }
    }

    /// Uniform base with a pruning policy.
    #[must_use]
    pub fn pruned(base: usize, prune: Prune) -> Self {
        Self::Pruned { base, prune }
    }

    /// Explicit per-shell sizes.
    #[must_use]
    pub fn shells(sizes: impl Into<Vec<usize>>) -> Self {
        Self::Shells(sizes.into())
    }

    /// Product rule with `theta` rings.
    #[must_use]
    pub fn product(theta: usize) -> Self {
        Self::Product { theta }
    }

    /// Per-shell adaptive-phi product rule.
    #[must_use]
    pub fn product_shells(theta: impl Into<Vec<usize>>) -> Self {
        Self::ProductShells(theta.into())
    }

    fn validate(&self) -> Result<()> {
        match self {
            Self::Fixed { n_ang } => validate_size(*n_ang),
            Self::Pruned { base, prune } => {
                validate_size(*base)?;
                if *prune == Prune::None && *base == 0 {
                    return Err(Error::InvalidArgument {
                        argument: "angular layout base",
                    });
                }
                Ok(())
            }
            Self::Shells(sizes) => {
                if sizes.is_empty() {
                    return Err(Error::InvalidArgument {
                        argument: "angular layout shells",
                    });
                }
                for &size in sizes {
                    validate_size(size)?;
                }
                Ok(())
            }
            Self::Product { theta } => {
                if !(2..=999).contains(theta) {
                    return Err(Error::InvalidArgument {
                        argument: "angular layout product theta",
                    });
                }
                Ok(())
            }
            Self::ProductShells(theta) => {
                if theta.is_empty() {
                    return Err(Error::InvalidArgument {
                        argument: "angular layout product shells",
                    });
                }
                if theta.iter().any(|&n| !(2..=999).contains(&n)) {
                    return Err(Error::InvalidArgument {
                        argument: "angular layout product theta",
                    });
                }
                Ok(())
            }
        }
    }
}

fn validate_size(n_ang: usize) -> Result<()> {
    if is_supported_size(n_ang) {
        Ok(())
    } else {
        Err(Error::UnsupportedLebedevGrid { points: n_ang })
    }
}

/// Radial and angular parameters for one element.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ElementPreset {
    /// Number of nominal radial shells.
    pub n_rad: usize,
    /// Radial length scale in Bohr (the `xi` passed to
    /// [`crate::radial::Radial::grid_xi`]).
    pub radial_scale: f64,
    /// Outer radial cutoff in Bohr.  `None` keeps every nominal shell.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cutoff: Option<f64>,
    /// With a cutoff, generate the full nominal shell set and mark the outer
    /// tail inactive (`true`), or stop after the first inactive shell
    /// (`false`).  Ignored without a cutoff.
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub generate_all: bool,
    /// Angular layout.
    pub angular: AngularLayout,
}

impl ElementPreset {
    /// A fixed angular rule on every shell.
    #[must_use]
    pub fn fixed(n_rad: usize, radial_scale: f64, n_ang: usize) -> Self {
        Self {
            n_rad,
            radial_scale,
            cutoff: None,
            generate_all: true,
            angular: AngularLayout::fixed(n_ang),
        }
    }

    /// A uniform base plus a pruning scheme.
    #[must_use]
    pub fn pruned(n_rad: usize, radial_scale: f64, base: usize, prune: Prune) -> Self {
        Self {
            n_rad,
            radial_scale,
            cutoff: None,
            generate_all: true,
            angular: AngularLayout::pruned(base, prune),
        }
    }

    /// Explicit per-shell angular sizes.
    #[must_use]
    pub fn shells(n_rad: usize, radial_scale: f64, sizes: impl Into<Vec<usize>>) -> Self {
        Self {
            n_rad,
            radial_scale,
            cutoff: None,
            generate_all: true,
            angular: AngularLayout::shells(sizes),
        }
    }

    /// A per-shell adaptive-phi product rule.
    #[must_use]
    pub fn product_shells(n_rad: usize, radial_scale: f64, theta: impl Into<Vec<usize>>) -> Self {
        Self {
            n_rad,
            radial_scale,
            cutoff: None,
            generate_all: true,
            angular: AngularLayout::product_shells(theta),
        }
    }

    /// A product rule.
    #[must_use]
    pub fn product(n_rad: usize, radial_scale: f64, theta: usize) -> Self {
        Self {
            n_rad,
            radial_scale,
            cutoff: None,
            generate_all: true,
            angular: AngularLayout::product(theta),
        }
    }

    /// Attach an outer radial cutoff (Bohr).
    #[must_use]
    pub fn with_cutoff(mut self, cutoff: f64) -> Self {
        self.cutoff = Some(cutoff);
        self
    }

    /// Choose whether a cutoff keeps the nominal shell set or stops early.
    #[must_use]
    pub fn with_generate_all(mut self, generate_all: bool) -> Self {
        self.generate_all = generate_all;
        self
    }

    fn validate(&self) -> Result<()> {
        if self.n_rad == 0 {
            return Err(Error::InvalidArgument {
                argument: "radial shell count",
            });
        }
        if !self.radial_scale.is_finite() || self.radial_scale <= 0.0 {
            return Err(Error::InvalidArgument {
                argument: "radial scale",
            });
        }
        if let Some(cutoff) = self.cutoff {
            if !cutoff.is_finite() || cutoff <= 0.0 {
                return Err(Error::InvalidArgument {
                    argument: "radial cutoff",
                });
            }
        }
        self.angular.validate()
    }
}

/// Shared element parameters for a contiguous range of nuclear charges.
///
/// Ranges are the compact way to express the "one default per periodic row"
/// pattern of a preset: one full [`ElementPreset`] covers `z_min..=z_max`,
/// and `element` entries only need the values that differ (usually just the
/// radial scale).
#[derive(Debug, Clone, PartialEq)]
pub struct ElementRangeDefault {
    /// First nuclear charge covered by the range (inclusive).
    pub z_min: u32,
    /// Last nuclear charge covered by the range (inclusive).
    pub z_max: u32,
    /// Shared element parameters.
    pub element: ElementPreset,
}

impl ElementRangeDefault {
    /// A range default for `z_min..=z_max`.
    #[must_use]
    pub fn new(z_min: u32, z_max: u32, element: ElementPreset) -> Self {
        Self {
            z_min,
            z_max,
            element,
        }
    }

    /// Whether the range contains `z`.
    #[must_use]
    pub fn contains(&self, z: u32) -> bool {
        self.z_min <= z && z <= self.z_max
    }
}

/// Program-independent description of a molecular grid topology.
///
/// The Serde representation uses one `[[element]]` array-table entry per
/// element, which keeps the file readable in TOML; a `[[range]]` entry
/// supplies shared parameters for a contiguous range of charges:
///
/// ```toml
/// radial = "treutler_ahlrichs"
///
/// [default]
/// n_rad = 50
/// radial_scale = 1.2
/// angular = { pruned = { base = 194, prune = "nwchem" } }
///
/// [[range]]
/// z_min = 3
/// z_max = 10
/// n_rad = 75
/// radial_scale = 0.9
/// angular = { shells = [50, 110, 194, 302] }
///
/// [[element]]
/// z = 8
/// radial_scale = 0.888
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct GridPreset {
    /// Informational preset name.
    pub name: String,
    /// Lowest nuclear charge the preset covers (inclusive).  Default `1`.
    pub z_min: u32,
    /// Highest nuclear charge the preset covers (inclusive).  Default `118`.
    pub z_max: u32,
    /// Radial node mapping shared by every element.
    pub radial: Radial,
    /// Whether the molecular radial cutoff
    /// `R(A) = max(R_iso(A), max_B(R_iso(B) + d(A,B)))` applies.
    pub molecular_cutoffs: bool,
    /// Element parameters for charges without an explicit entry or range.
    pub default: Option<ElementPreset>,
    /// Parameter blocks shared by contiguous charge ranges.  A range takes
    /// precedence over [`Self::default`] and is overridden by an explicit
    /// [`Self::elements`] entry.
    pub defaults: Vec<ElementRangeDefault>,
    /// Per-element parameters.  Entries take precedence over ranges and
    /// [`Self::default`].
    pub elements: BTreeMap<u32, ElementPreset>,
}

/// One element entry in the preset file representation.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ElementEntry {
    /// Nuclear charge.
    pub z: u32,
    /// Element parameters.
    #[serde(flatten)]
    pub element: ElementPreset,
}

/// One range entry in the preset file representation.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ElementRangeEntry {
    /// First nuclear charge covered by the range.
    pub z_min: u32,
    /// Last nuclear charge covered by the range.
    pub z_max: u32,
    /// Shared element parameters.
    #[serde(flatten)]
    pub element: ElementPreset,
}

#[derive(serde::Serialize)]
struct GridPresetRepr {
    #[serde(default, skip_serializing_if = "is_empty_string")]
    name: String,
    #[serde(default = "default_z_min", skip_serializing_if = "is_default_z_min")]
    z_min: u32,
    #[serde(default = "default_z_max", skip_serializing_if = "is_default_z_max")]
    z_max: u32,
    radial: Radial,
    #[serde(default, skip_serializing_if = "is_false")]
    molecular_cutoffs: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    default: Option<ElementPreset>,
    #[serde(default, rename = "range", skip_serializing_if = "Vec::is_empty")]
    defaults: Vec<ElementRangeEntry>,
    #[serde(default, rename = "element", skip_serializing_if = "Vec::is_empty")]
    elements: Vec<ElementEntry>,
}

/// Partial element entry accepted by the deserializer.  Missing fields are
/// inherited from [`GridPreset::default`], so a preset can list only the
/// values that differ between elements (for example a per-element radial
/// scale) while the shared shell counts and angular layout live in one
/// `default` block.
#[derive(serde::Deserialize)]
struct ElementPatch {
    z: u32,
    #[serde(flatten)]
    fields: ElementFields,
}

#[derive(serde::Deserialize, Default)]
struct ElementFields {
    #[serde(default)]
    n_rad: Option<usize>,
    #[serde(default)]
    radial_scale: Option<f64>,
    #[serde(default)]
    cutoff: Option<f64>,
    #[serde(default)]
    generate_all: Option<bool>,
    #[serde(default)]
    angular: Option<AngularLayout>,
}

#[derive(serde::Deserialize)]
struct GridPresetDe {
    #[serde(default)]
    name: String,
    #[serde(default = "default_z_min")]
    z_min: u32,
    #[serde(default = "default_z_max")]
    z_max: u32,
    radial: Radial,
    #[serde(default)]
    molecular_cutoffs: bool,
    #[serde(default)]
    default: Option<ElementPreset>,
    #[serde(default, rename = "range")]
    defaults: Vec<ElementRangeEntry>,
    #[serde(default, rename = "element")]
    elements: Vec<ElementPatch>,
}

impl serde::Serialize for GridPreset {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        let repr = GridPresetRepr {
            name: self.name.clone(),
            z_min: self.z_min,
            z_max: self.z_max,
            radial: self.radial,
            molecular_cutoffs: self.molecular_cutoffs,
            default: self.default.clone(),
            defaults: self
                .defaults
                .iter()
                .map(|range| ElementRangeEntry {
                    z_min: range.z_min,
                    z_max: range.z_max,
                    element: range.element.clone(),
                })
                .collect(),
            elements: self
                .elements
                .iter()
                .map(|(&z, element)| ElementEntry {
                    z,
                    element: element.clone(),
                })
                .collect(),
        };
        repr.serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for GridPreset {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let repr = GridPresetDe::deserialize(deserializer)?;
        let defaults: Vec<ElementRangeDefault> = repr
            .defaults
            .into_iter()
            .map(|range| ElementRangeDefault::new(range.z_min, range.z_max, range.element))
            .collect();
        let mut elements = BTreeMap::new();
        for patch in repr.elements {
            let base = defaults
                .iter()
                .find(|range| range.contains(patch.z))
                .map(|range| &range.element)
                .or(repr.default.as_ref());
            let element = patch.fields.merge(base).map_err(|missing| {
                serde::de::Error::custom(format!(
                    "element Z = {} is missing `{missing}` and the preset has no default",
                    patch.z
                ))
            })?;
            if elements.insert(patch.z, element).is_some() {
                return Err(serde::de::Error::custom(format!(
                    "duplicate element Z = {} in preset",
                    patch.z
                )));
            }
        }
        Ok(Self {
            name: repr.name,
            z_min: repr.z_min,
            z_max: repr.z_max,
            radial: repr.radial,
            molecular_cutoffs: repr.molecular_cutoffs,
            default: repr.default,
            defaults,
            elements,
        })
    }
}

impl ElementFields {
    fn merge(
        self,
        default: Option<&ElementPreset>,
    ) -> std::result::Result<ElementPreset, &'static str> {
        let Some(mut element) = default.cloned() else {
            return Ok(ElementPreset {
                n_rad: self.n_rad.ok_or("n_rad")?,
                radial_scale: self.radial_scale.ok_or("radial_scale")?,
                cutoff: self.cutoff,
                generate_all: self.generate_all.unwrap_or(true),
                angular: self.angular.ok_or("angular")?,
            });
        };
        if let Some(value) = self.n_rad {
            element.n_rad = value;
        }
        if let Some(value) = self.radial_scale {
            element.radial_scale = value;
        }
        if let Some(value) = self.cutoff {
            element.cutoff = Some(value);
        }
        if let Some(value) = self.generate_all {
            element.generate_all = value;
        }
        if let Some(value) = self.angular {
            element.angular = value;
        }
        Ok(element)
    }
}

impl GridPreset {
    /// An empty preset for `radial`.
    #[must_use]
    pub fn new(radial: Radial) -> Self {
        Self {
            name: String::new(),
            z_min: 1,
            z_max: 118,
            radial,
            molecular_cutoffs: false,
            default: None,
            defaults: Vec::new(),
            elements: BTreeMap::new(),
        }
    }

    /// Set the informational name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the element domain `z_min..=z_max` (both inclusive).
    ///
    /// A preset with no explicit element entry for a charge outside the
    /// domain returns [`Error::MissingElementData`] from
    /// [`Self::element`] instead of falling back to the preset default.
    #[must_use]
    pub fn with_element_range(mut self, z_min: u32, z_max: u32) -> Self {
        self.z_min = z_min;
        self.z_max = z_max;
        self
    }

    /// Whether `z` is inside the preset's element domain.
    #[must_use]
    pub fn supports(&self, z: u32) -> bool {
        self.z_min <= z && z <= self.z_max
    }

    /// Set the fallback element parameters.
    #[must_use]
    pub fn with_default(mut self, default: ElementPreset) -> Self {
        self.default = Some(default);
        self
    }

    /// Add a shared parameter block for `z_min..=z_max`.
    #[must_use]
    pub fn with_default_range(mut self, z_min: u32, z_max: u32, element: ElementPreset) -> Self {
        self.defaults
            .push(ElementRangeDefault::new(z_min, z_max, element));
        self
    }

    /// Add or replace one element's parameters.
    #[must_use]
    pub fn with_element(mut self, z: u32, element: ElementPreset) -> Self {
        self.elements.insert(z, element);
        self
    }

    /// Enable or disable the molecular cutoff extension.
    #[must_use]
    pub fn with_molecular_cutoffs(mut self, enabled: bool) -> Self {
        self.molecular_cutoffs = enabled;
        self
    }

    /// Parameters for nuclear charge `z`.
    ///
    /// # Errors
    /// [`Error::UnsupportedElement`] for `z` outside `1..=118`, and
    /// [`Error::MissingElementData`] when neither an explicit entry nor a
    /// default exists.
    pub fn element(&self, z: u32) -> Result<&ElementPreset> {
        if !(1..=118).contains(&z) {
            return Err(Error::UnsupportedElement { z });
        }
        if !self.supports(z) {
            return Err(Error::MissingElementData {
                z,
                table: "grid preset",
            });
        }
        self.elements
            .get(&z)
            .or_else(|| {
                self.defaults
                    .iter()
                    .find(|range| range.contains(z))
                    .map(|range| &range.element)
            })
            .or(self.default.as_ref())
            .ok_or(Error::MissingElementData {
                z,
                table: "grid preset",
            })
    }

    /// Whether the point set depends on the full molecular geometry.
    #[must_use]
    pub fn geometry_dependent(&self) -> bool {
        self.molecular_cutoffs
            && (self
                .default
                .as_ref()
                .is_some_and(|element| element.cutoff.is_some())
                || self
                    .defaults
                    .iter()
                    .any(|range| range.element.cutoff.is_some())
                || self
                    .elements
                    .values()
                    .any(|element| element.cutoff.is_some()))
    }

    /// Validate every element entry and the topology invariants.
    ///
    /// # Errors
    /// [`Error::UnsupportedElement`] for an out-of-range charge,
    /// [`Error::UnsupportedLebedevGrid`] for an unsupported angular size, and
    /// [`Error::InvalidArgument`] for a malformed shell/scale/cutoff value.
    pub fn validate(&self) -> Result<()> {
        if self.z_min == 0 || self.z_min > self.z_max || self.z_max > 118 {
            return Err(Error::InvalidArgument {
                argument: "preset element range",
            });
        }
        for &z in self.elements.keys() {
            if !(1..=118).contains(&z) {
                return Err(Error::UnsupportedElement { z });
            }
            if !self.supports(z) {
                return Err(Error::InvalidArgument {
                    argument: "preset element range",
                });
            }
        }
        for (index, range) in self.defaults.iter().enumerate() {
            if !(1..=118).contains(&range.z_min) || !(1..=118).contains(&range.z_max) {
                return Err(Error::UnsupportedElement {
                    z: if (1..=118).contains(&range.z_min) {
                        range.z_max
                    } else {
                        range.z_min
                    },
                });
            }
            if range.z_min > range.z_max {
                return Err(Error::InvalidArgument {
                    argument: "preset default range",
                });
            }
            range.element.validate()?;
            if self.molecular_cutoffs
                && matches!(
                    range.element.angular,
                    AngularLayout::Shells(_) | AngularLayout::ProductShells(_)
                )
                && range.element.cutoff.is_some()
            {
                return Err(Error::UnsupportedGrid {
                    name: format!("preset range {}..={}", range.z_min, range.z_max),
                    reason: "an explicit per-shell list cannot follow a moving molecular cutoff"
                        .to_string(),
                });
            }
            for other in &self.defaults[..index] {
                if range.z_min <= other.z_max && other.z_min <= range.z_max {
                    return Err(Error::InvalidArgument {
                        argument: "overlapping preset default ranges",
                    });
                }
            }
        }
        for (&z, element) in &self.elements {
            if !(1..=118).contains(&z) {
                return Err(Error::UnsupportedElement { z });
            }
            element.validate()?;
            if self.molecular_cutoffs
                && matches!(
                    element.angular,
                    AngularLayout::Shells(_) | AngularLayout::ProductShells(_)
                )
                && element.cutoff.is_some()
            {
                return Err(Error::UnsupportedGrid {
                    name: format!("preset element Z={z}"),
                    reason: "an explicit per-shell list cannot follow a moving molecular cutoff"
                        .to_string(),
                });
            }
        }
        if let Some(default) = &self.default {
            default.validate()?;
            if self.molecular_cutoffs
                && matches!(
                    default.angular,
                    AngularLayout::Shells(_) | AngularLayout::ProductShells(_)
                )
                && default.cutoff.is_some()
            {
                return Err(Error::UnsupportedGrid {
                    name: "preset default".to_string(),
                    reason: "an explicit per-shell list cannot follow a moving molecular cutoff"
                        .to_string(),
                });
            }
        }
        Ok(())
    }

    /// Build one element's atomic grid with the preset's isolated cutoff.
    ///
    /// The returned grid is centered at the origin and carries the
    /// unpartitioned quadrature weights.
    ///
    /// # Errors
    /// Propagates [`Self::validate`], the missing element data, and the radial
    /// or angular construction errors.
    pub fn atomic_grid(&self, z: u32) -> Result<AtomicGrid> {
        self.validate()?;
        let element = self.element(z)?;
        self.build_atomic_grid(z, element, element.cutoff)
    }

    /// Build one atom's atomic grid at its molecular position, applying the
    /// molecular cutoff when enabled.
    ///
    /// # Errors
    /// [`Error::InvalidAtom`] for an out-of-range `center`, plus the errors of
    /// [`Self::atomic_grid`].
    pub fn atomic_grid_at(&self, molecule: &Molecule, center: usize) -> Result<AtomicGrid> {
        if center >= molecule.len() {
            return Err(Error::InvalidAtom {
                atom: center,
                natm: molecule.len(),
            });
        }
        if !self.molecular_cutoffs {
            return self.atomic_grid(molecule.charges()[center]);
        }
        self.validate()?;
        let z = molecule.charges()[center];
        let element = self.element(z)?;
        let Some(mut cutoff) = element.cutoff else {
            return self.build_atomic_grid(z, element, None);
        };
        for (other, &other_z) in molecule.charges().iter().enumerate() {
            if other == center {
                continue;
            }
            let Some(other_cutoff) = self.element(other_z)?.cutoff else {
                continue;
            };
            let distance = distance(molecule.centers()[center], molecule.centers()[other]);
            cutoff = cutoff.max(other_cutoff + distance);
        }
        self.build_atomic_grid(z, element, Some(cutoff))
    }

    fn build_atomic_grid(
        &self,
        z: u32,
        element: &ElementPreset,
        cutoff: Option<f64>,
    ) -> Result<AtomicGrid> {
        let effective_cutoff = cutoff.unwrap_or(f64::INFINITY);
        let generated: Vec<RadialShell> = shells(
            self.radial,
            element.n_rad,
            element.radial_scale,
            effective_cutoff,
            element.generate_all,
        );
        let active: Vec<&RadialShell> = generated.iter().filter(|shell| shell.active).collect();
        if active.is_empty() {
            return Err(Error::InvalidArgument {
                argument: "radial cutoff",
            });
        }
        let rads: Vec<f64> = active.iter().map(|shell| shell.r).collect();
        let wrads: Vec<f64> = active.iter().map(|shell| shell.w).collect();

        let angular = &element.angular;
        if let AngularLayout::Product { theta } = angular {
            return assemble_atomic_grid(z, &rads, &wrads, |_| product_angular_grid(*theta));
        }
        if let AngularLayout::ProductShells(theta) = angular {
            if theta.len() != rads.len() {
                return Err(Error::LengthMismatch {
                    expected: rads.len(),
                    got: theta.len(),
                });
            }
            return assemble_atomic_grid(z, &rads, &wrads, |shell| {
                product_angular_grid_adaptive(theta[shell])
            });
        }

        let sizes: Vec<usize> = match angular {
            AngularLayout::Fixed { n_ang } => vec![*n_ang; rads.len()],
            AngularLayout::Shells(sizes) => {
                if sizes.len() != rads.len() {
                    return Err(Error::LengthMismatch {
                        expected: rads.len(),
                        got: sizes.len(),
                    });
                }
                sizes.clone()
            }
            AngularLayout::Pruned { base, prune } => {
                // Use the ratios from the full nominal shell generation; the
                // Euler–Maclaurin ratio depends on `n_rad`, not on the number
                // of shells retained by the cutoff.
                let ratios: Vec<f64> = if prune.needs_ratios() {
                    active.iter().map(|shell| shell.ratio).collect()
                } else {
                    Vec::new()
                };
                let context = PruneContext::for_preset(z, &rads, &ratios, *base);
                crate::prune::sizes_checked(prune, &context, rads.len())?.sizes
            }
            AngularLayout::Product { .. } | AngularLayout::ProductShells(_) => {
                unreachable!("handled above")
            }
        };
        assemble_atomic_grid(z, &rads, &wrads, |shell| angular_grid(sizes[shell]))
    }
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hydrogen() -> ElementPreset {
        ElementPreset::fixed(20, 1.0, 50)
    }

    #[test]
    fn builds_fixed_grid() {
        let preset = GridPreset::new(Radial::TreutlerAhlrichs)
            .with_name("test")
            .with_default(hydrogen());
        let grid = preset.atomic_grid(1).unwrap();
        assert_eq!(grid.nr(), 20);
        assert!(grid.shell_angular.iter().all(|&n| n == 50));
        assert_eq!(grid.z, 1);
        assert!(grid.total_quad_weight() > 0.0);
    }

    #[test]
    fn builds_shell_list() {
        let element = ElementPreset::shells(3, 1.0, [6, 26, 50]);
        let preset = GridPreset::new(Radial::Delley).with_element(6, element);
        let grid = preset.atomic_grid(6).unwrap();
        assert_eq!(grid.shell_angular, vec![6, 26, 50]);
    }

    #[test]
    fn shell_list_must_cover_the_active_shells() {
        let element = ElementPreset::shells(3, 1.0, [6, 26]);
        let preset = GridPreset::new(Radial::Delley).with_element(6, element);
        assert!(matches!(
            preset.atomic_grid(6),
            Err(Error::LengthMismatch { .. })
        ));
    }

    #[test]
    fn builds_product_grid() {
        let element = ElementPreset::product(10, 1.0, 8);
        let preset = GridPreset::new(Radial::GaussChebyshev).with_element(8, element);
        let grid = preset.atomic_grid(8).unwrap();
        assert_eq!(grid.shell_angular, vec![2 * 8 * 8; 10]);
    }

    #[test]
    fn pruned_layout_matches_direct_prune() {
        let element = ElementPreset::pruned(40, 1.0, 194, Prune::Nwchem);
        let preset = GridPreset::new(Radial::TreutlerAhlrichs).with_element(6, element);
        let grid = preset.atomic_grid(6).unwrap();
        assert!(
            grid.shell_angular
                .iter()
                .all(|&n| { crate::params::PYSCF_LEBEDEV_NGRID.contains(&n) })
        );
    }

    #[test]
    fn missing_element_is_an_error() {
        let preset = GridPreset::new(Radial::TreutlerAhlrichs);
        assert!(matches!(
            preset.atomic_grid(8),
            Err(Error::MissingElementData { z: 8, .. })
        ));
    }

    #[test]
    fn molecular_cutoff_extends_the_outer_shells() {
        let element = ElementPreset::fixed(60, 1.0, 50).with_cutoff(4.0);
        let preset = GridPreset::new(Radial::EulerMaclaurin)
            .with_element(1, element)
            .with_molecular_cutoffs(true);
        let near = Molecule::new([(1u32, [0.0; 3]), (1, [0.0, 0.0, 1.0])]).unwrap();
        let far = Molecule::new([(1u32, [0.0; 3]), (1, [0.0, 0.0, 10.0])]).unwrap();
        let near_grid = preset.atomic_grid_at(&near, 0).unwrap();
        let far_grid = preset.atomic_grid_at(&far, 0).unwrap();
        assert!(far_grid.len() > near_grid.len());
        assert!(preset.geometry_dependent());
    }

    #[test]
    fn range_defaults_fill_partial_entries() {
        let base = ElementPreset::shells(3, 1.0, [6, 26, 50]);
        let preset = GridPreset::new(Radial::Delley)
            .with_default_range(1, 10, base)
            .with_element(
                8,
                ElementPreset {
                    radial_scale: 0.5,
                    ..ElementPreset::shells(3, 1.0, [6, 26, 50])
                },
            );
        let oxygen = preset.element(8).unwrap();
        assert_eq!(oxygen.n_rad, 3);
        assert_eq!(oxygen.radial_scale, 0.5);
        assert_eq!(oxygen.angular, AngularLayout::shells([6, 26, 50]));
        assert_eq!(preset.element(6).unwrap().radial_scale, 1.0);
        assert!(matches!(
            preset.element(11),
            Err(Error::MissingElementData { z: 11, .. })
        ));
        preset.validate().unwrap();
    }

    #[test]
    fn range_defaults_must_not_overlap() {
        let base = ElementPreset::fixed(3, 1.0, 6);
        let preset = GridPreset::new(Radial::Delley)
            .with_default_range(1, 10, base.clone())
            .with_default_range(5, 20, base);
        assert!(matches!(
            preset.validate(),
            Err(Error::InvalidArgument {
                argument: "overlapping preset default ranges"
            })
        ));
    }

    #[test]
    fn product_shells_build() {
        let element = ElementPreset::product_shells(3, 1.0, [5, 7, 14]);
        let preset = GridPreset::new(Radial::GaussChebyshev).with_element(1, element);
        let grid = preset.atomic_grid(1).unwrap();
        assert_eq!(grid.shell_angular, vec![55, 105, 350]);
        assert_eq!(grid.points.len(), 55 + 105 + 350);
    }

    #[test]
    fn element_range_is_enforced() {
        let preset = GridPreset::new(Radial::TreutlerAhlrichs)
            .with_element_range(1, 2)
            .with_element(1, hydrogen());
        assert!(preset.supports(1));
        assert!(preset.supports(2));
        assert!(!preset.supports(3));
        assert!(matches!(
            preset.element(3),
            Err(Error::MissingElementData {
                z: 3,
                table: "grid preset"
            })
        ));
        assert!(matches!(
            preset.atomic_grid(3),
            Err(Error::MissingElementData {
                z: 3,
                table: "grid preset"
            })
        ));
    }

    #[test]
    fn element_range_round_trips_through_the_file_form() {
        let preset = GridPreset::new(Radial::TreutlerAhlrichs)
            .with_element_range(3, 10)
            .with_element(3, hydrogen())
            .with_element(10, hydrogen());
        let text = toml::to_string(&preset).unwrap();
        assert!(text.contains("z_min = 3"), "{text}");
        assert!(text.contains("z_max = 10"), "{text}");
        let reloaded: GridPreset = toml::from_str(&text).unwrap();
        assert_eq!(reloaded.z_min, 3);
        assert_eq!(reloaded.z_max, 10);
        assert_eq!(preset, reloaded);
    }

    #[test]
    fn element_range_rejects_an_entry_outside_the_domain() {
        let preset = GridPreset::new(Radial::TreutlerAhlrichs)
            .with_element_range(1, 2)
            .with_element(6, hydrogen());
        assert!(matches!(
            preset.validate(),
            Err(Error::InvalidArgument {
                argument: "preset element range"
            })
        ));
    }

    #[test]
    fn validation_rejects_bad_element_data() {
        let mut preset = GridPreset::new(Radial::TreutlerAhlrichs);
        preset.elements.insert(0, hydrogen());
        assert!(matches!(
            preset.validate(),
            Err(Error::UnsupportedElement { z: 0 })
        ));
        let mut preset = GridPreset::new(Radial::TreutlerAhlrichs);
        preset.elements.insert(6, ElementPreset::fixed(0, 1.0, 50));
        assert!(preset.validate().is_err());
        let mut preset = GridPreset::new(Radial::TreutlerAhlrichs);
        preset.elements.insert(6, ElementPreset::fixed(20, 0.0, 50));
        assert!(preset.validate().is_err());
    }
}
