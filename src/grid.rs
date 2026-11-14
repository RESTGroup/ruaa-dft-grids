//! Geometry-independent grid topology and the public batch API.
//!
//! A [`GridSpec`] is the complete build specification: an atomic topology
//! (a data-driven [`crate::preset::GridPreset`], the built-in PySCF preset, or
//! a caller-supplied plan) plus a partition configuration. [`GridPlan`] turns
//! it into per-atom atomic grids and a flat point table (local coordinates,
//! quadrature weights, parent atom).
//!
//! The topology is independent of the molecular geometry except for a preset
//! with molecular radial cutoffs: such a preset extends each atom's cutoff by
//! the pairwise rule
//! `R(A) = max(R_iso(A), max_B(R_iso(B) + d(A,B)))`, so its point set depends
//! on the full geometry. See [`GridSpec::geometry_dependent`] and
//! [`GridPlan::geometry_dependent`].
//!
//! The public evaluation unit is a **range** of global grid points. Callers
//! request `start..end` and receive a [`BatchBuffer`] holding only that
//! range's coordinates, quadrature weights, partition weights and parent
//! atoms. The crate never materialises a `G x A` object internally.
//!
//! Analytic partition derivatives live in
//! [`crate::partition::PartitionDerivEvaluator`], a separate streaming
//! evaluator with its own output types:
//!
//! * [`crate::partition::DerivOutput`] holds a row-major `(nsel, 3, B)` `dw`
//!   for the requested range (or every atom when no selection is given), and
//!   may also hold a batch-sized `(nsel, 3, natm, 3, B)` `ddw`;
//! * [`crate::partition::GradientContract`] and
//!   [`crate::partition::HessianContract`] accumulate the streamed
//!   contractions `sum_g c_g dw` and `sum_g c_g ddw`, never a dense
//!   `natm x 3 x natm x 3 x G` object;
//! * [`crate::partition::PartitionDerivEvaluator::fill_serial_visit`] exposes
//!   one point's full corrected Hessian at a time for either a `GridPlan` or
//!   a materialised-grid range;
//! * [`crate::partition::PartitionDerivEvaluator::fill`] evaluates a
//!   [`crate::partition::DerivPoints`] interval, so a large grid can stream
//!   raw Hessians in bounded batches while accumulating contractions;
//!   [`crate::partition::PartitionDerivEvaluator::fill_serial`] is the
//!   workspace-reusing serial form.
//!
//! All derivative kernels use the point's generating atom from
//! [`GridPlan::parent_of`] (or [`crate::moleculegrid::MolecularGrid::atom_idx`]
//! for a materialised grid), so a sorted or otherwise permuted point order
//! does not change the per-point derivative. See the
//! [partition derivative API](crate::doc::partition_derivatives) for the
//! tensor layouts and the sorted-grid analysis.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;

use derive_builder::Builder;

use crate::atomgrid::AtomicGrid;
use crate::error::{Error, Result};
use crate::molecule::Molecule;
use crate::partition::{PartitionConfig, PartitionScheme, RadiiConfig};
use crate::preset::GridPreset;
use crate::prune::Prune;
use crate::pyscf::build_pyscf_atom_grid;
use crate::radial::Radial;

/// PySCF grid topology: level sizes, radial map, pruning and per-Z overrides.
///
/// This is the built-in preset for PySCF's open-source `Grids.level`
/// construction. It is a convenience source of an atomic topology; the
/// partition is supplied by the enclosing [`GridSpec`].
///
/// Build with [`PyscfGridSpecBuilder`].
///
/// ```
/// use ruaa_dft_grids::grid::{GridSpec, PyscfGridSpecBuilder};
///
/// # fn main() -> ruaa_dft_grids::Result<()> {
/// let spec: GridSpec = PyscfGridSpecBuilder::default()
///     .level(3)
///     .build()?
///     .into();
/// assert!(!spec.geometry_dependent());
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq, Builder)]
#[builder(build_fn(skip))]
#[non_exhaustive]
pub struct PyscfGridSpec {
    /// PySCF `Grids.level`.
    #[builder(default = "3")]
    pub level: u8,
    /// Per-Z `(n_rad, n_ang)` overrides.
    ///
    /// `n_ang` may be a Lebedev point count or a PySCF `LEBEDEV_ORDER` key;
    /// the builder resolves it through [`crate::pyscf::angular_count`], which
    /// also maps 0 and 1 to the trivial one-point rule.
    #[builder(default)]
    pub atom_grid: BTreeMap<u32, (usize, usize)>,
    /// Radial method.
    #[builder(default)]
    pub radial: Radial,
    /// Pruning scheme.
    #[builder(default)]
    pub prune: Prune,
    /// Partition configuration (used when this type is converted into a
    /// [`GridSpec`]).
    #[builder(default = "default_pyscf_partition()")]
    pub partition: PartitionConfig,
}

impl PyscfGridSpecBuilder {
    /// Build and validate the specification.
    ///
    /// # Errors
    /// The errors of [`PyscfGridSpec::validate`].
    pub fn build(&self) -> Result<PyscfGridSpec> {
        let spec = PyscfGridSpec {
            level: self.level.unwrap_or(3),
            atom_grid: self.atom_grid.clone().unwrap_or_default(),
            radial: self.radial.unwrap_or_default(),
            prune: self.prune.unwrap_or_default(),
            partition: self
                .partition
                .clone()
                .unwrap_or_else(default_pyscf_partition),
        };
        spec.validate()?;
        Ok(spec)
    }
}

fn default_pyscf_partition() -> PartitionConfig {
    PartitionConfig::new(PartitionScheme::Becke, RadiiConfig::pyscf())
}

impl Default for PyscfGridSpec {
    fn default() -> Self {
        PyscfGridSpecBuilder::default()
            .build()
            .expect("PySCF grid-spec defaults are valid")
    }
}

/// The atomic-topology source of a [`GridSpec`].
#[derive(Debug, Clone, PartialEq)]
enum GridSource {
    /// A data-driven preset.
    Preset(GridPreset),
    /// The built-in PySCF level preset.
    Pyscf(PyscfGridSpec),
    /// No topology: the caller supplies a [`GridPlan`] of hand-built atomic
    /// grids. Only the partition configuration is used.
    Custom,
}

/// Complete grid-build specification: topology plus partition.
///
/// A topology comes from one of three places:
///
/// * [`GridPreset`] — data-driven, program
///   independent, and serializable. This is the path for a caller who has the
///   shell counts, radial scales and angular layouts;
/// * [`PyscfGridSpec`] — the built-in PySCF `Grids.level` construction;
/// * [`GridSpec::custom`] — no topology, for a caller-assembled
///   [`GridPlan`].
///
/// The partition belongs to the full spec, not to the topology. Point pruning
/// after the build is a separate [`crate::moleculegrid::PostPrune`] operation;
/// angular pruning inside the atomic grid is part of the topology.
///
/// ```
/// use ruaa_dft_grids::grid::{GridSpec, PyscfGridSpec};
/// use ruaa_dft_grids::partition::PartitionConfig;
///
/// # fn main() -> ruaa_dft_grids::Result<()> {
/// let spec = GridSpec::from(PyscfGridSpec::from_level(1)?)
///     .with_partition(PartitionConfig::pyscf());
/// assert!(!spec.geometry_dependent());
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct GridSpec {
    source: GridSource,
    partition: PartitionConfig,
}

impl GridSpec {
    /// Build a spec from a data-driven preset with the default partition
    /// ([`PartitionConfig::pyscf`]).
    #[must_use]
    pub fn from_preset(preset: GridPreset) -> Self {
        Self::from_preset_with_partition(preset, PartitionConfig::pyscf())
    }

    /// Build a spec from a data-driven preset and an explicit partition.
    #[must_use]
    pub fn from_preset_with_partition(preset: GridPreset, partition: PartitionConfig) -> Self {
        Self {
            source: GridSource::Preset(preset),
            partition,
        }
    }

    /// A spec without a topology, for a caller-assembled [`GridPlan`].
    ///
    /// [`Self::atomic_grid`] and [`GridPlan::build`] cannot use this form;
    /// materialise the plan with
    /// [`crate::moleculegrid::build_molecular_grid_from_plan`] or
    /// [`crate::moleculegrid::build_molecular_grid_from_atomic_grids`].
    #[must_use]
    pub fn custom(partition: PartitionConfig) -> Self {
        Self {
            source: GridSource::Custom,
            partition,
        }
    }

    /// Replace the partition configuration.
    #[must_use]
    pub fn with_partition(mut self, partition: PartitionConfig) -> Self {
        if let GridSource::Pyscf(spec) = &mut self.source {
            spec.partition = partition.clone();
        }
        self.partition = partition;
        self
    }

    /// The partition configuration.
    #[must_use]
    pub fn partition(&self) -> &PartitionConfig {
        &self.partition
    }

    /// The data-driven preset, when this spec is built from one.
    #[must_use]
    pub fn preset(&self) -> Option<&GridPreset> {
        match &self.source {
            GridSource::Preset(preset) => Some(preset),
            _ => None,
        }
    }

    /// The built-in PySCF topology, when this spec is built from one.
    #[must_use]
    pub fn pyscf_spec(&self) -> Option<&PyscfGridSpec> {
        match &self.source {
            GridSource::Pyscf(spec) => Some(spec),
            _ => None,
        }
    }

    /// Whether this spec carries no atomic topology.
    #[must_use]
    pub fn is_custom(&self) -> bool {
        matches!(self.source, GridSource::Custom)
    }

    /// Validate the topology and the partition configuration.
    ///
    /// # Errors
    /// The topology's validation errors and [`PartitionConfig::validate`].
    pub fn validate(&self) -> Result<()> {
        match &self.source {
            GridSource::Preset(preset) => preset.validate()?,
            GridSource::Pyscf(spec) => spec.validate()?,
            GridSource::Custom => {}
        }
        self.partition.validate()
    }

    /// Build one element's atomic grid in isolation.
    ///
    /// For a preset with molecular cutoffs this uses the atom's isolated
    /// cutoff; use [`Self::atomic_grid_at`] for the molecular cutoff. The grid
    /// includes the nuclear charge in [`AtomicGrid::z`].
    ///
    /// # Errors
    /// Propagates the preset/PySCF element, scheme, level or data errors.
    pub fn atomic_grid(&self, z: u32) -> Result<AtomicGrid> {
        match &self.source {
            GridSource::Preset(preset) => preset.atomic_grid(z),
            GridSource::Pyscf(spec) => spec.atomic_grid(z),
            GridSource::Custom => Err(Error::UnsupportedGrid {
                name: "custom grid spec".to_string(),
                reason: "no atomic topology; supply a GridPlan or atomic grids".to_string(),
            }),
        }
    }

    /// Whether atomic grids depend on the atom's position (not just Z).
    /// [`GridPlan::build`] then computes one grid per atom instead of caching
    /// per element.
    #[must_use]
    pub fn geometry_dependent(&self) -> bool {
        match &self.source {
            GridSource::Preset(preset) => preset.geometry_dependent(),
            GridSource::Pyscf(_) => false,
            GridSource::Custom => false,
        }
    }

    /// Build one atom's grid at its position in `molecule`.
    ///
    /// `center` indexes `molecule`; for geometry-independent topologies this
    /// is the same as [`Self::atomic_grid`] on that atom's charge.
    ///
    /// # Errors
    /// [`Error::InvalidAtom`] for an out-of-range `center`, plus the topology
    /// builder's errors.
    pub fn atomic_grid_at(&self, molecule: &Molecule, center: usize) -> Result<AtomicGrid> {
        if center >= molecule.len() {
            return Err(Error::InvalidAtom {
                atom: center,
                natm: molecule.len(),
            });
        }
        match &self.source {
            GridSource::Preset(preset) => preset.atomic_grid_at(molecule, center),
            _ => self.atomic_grid(molecule.charges()[center]),
        }
    }
}

impl Default for GridSpec {
    fn default() -> Self {
        Self::from(PyscfGridSpec::default())
    }
}

impl From<PyscfGridSpec> for GridSpec {
    fn from(spec: PyscfGridSpec) -> Self {
        let partition = spec.partition.clone();
        Self {
            source: GridSource::Pyscf(spec),
            partition,
        }
    }
}

impl From<&PyscfGridSpec> for GridSpec {
    fn from(spec: &PyscfGridSpec) -> Self {
        spec.clone().into()
    }
}

impl From<GridPreset> for GridSpec {
    fn from(preset: GridPreset) -> Self {
        Self::from_preset(preset)
    }
}

impl From<&GridPreset> for GridSpec {
    fn from(preset: &GridPreset) -> Self {
        Self::from_preset(preset.clone())
    }
}

impl From<&GridSpec> for GridSpec {
    fn from(spec: &GridSpec) -> Self {
        spec.clone()
    }
}

impl PyscfGridSpec {
    /// PySCF `Grids.level` preset.
    ///
    /// This is a **full preset**, not a level-only switch: it sets `level`
    /// and leaves the builder defaults for the radial method, pruning,
    /// partition and radii. The level is validated here, even if every atom
    /// has an `atom_grid` override ([`crate::Error::UnsupportedLevel`] above
    /// level 9). Use [`PyscfGridSpecBuilder`] to override the other fields.
    ///
    /// # Errors
    /// [`crate::Error::UnsupportedLevel`] above level 9 and the partition
    /// validation errors.
    pub fn from_level(level: u8) -> Result<Self> {
        PyscfGridSpecBuilder::default().level(level).build()
    }

    /// Replace the partition configuration.
    #[must_use]
    pub fn with_partition(mut self, partition: PartitionConfig) -> Self {
        self.partition = partition;
        self
    }

    /// Use `radial` as the PySCF radial method.
    #[must_use]
    pub fn with_radial(mut self, radial: Radial) -> Self {
        self.radial = radial;
        self
    }

    /// Use `prune` as the PySCF pruning scheme.
    #[must_use]
    pub fn with_prune(mut self, prune: Prune) -> Self {
        self.prune = prune;
        self
    }

    /// Validate the level, per-element overrides and partition configuration.
    ///
    /// # Errors
    /// [`Error::UnsupportedLevel`] above 9, [`Error::UnsupportedElement`] or
    /// [`Error::InvalidArgument`] for a bad override, and the partition
    /// validation errors.
    pub fn validate(&self) -> Result<()> {
        if self.level > 9 {
            return Err(Error::UnsupportedLevel { level: self.level });
        }
        for (&z, &(n_rad, n_ang)) in &self.atom_grid {
            if !(1..=118).contains(&z) {
                return Err(Error::UnsupportedElement { z });
            }
            if n_rad == 0 {
                return Err(Error::InvalidArgument {
                    argument: "atom_grid n_rad",
                });
            }
            crate::pyscf::angular_count(n_ang)?;
        }
        self.partition.validate()
    }

    /// Build one element's atomic grid.
    ///
    /// # Errors
    /// Propagates [`Self::validate`] and the PySCF level / element / radial /
    /// pruning errors.
    pub fn atomic_grid(&self, z: u32) -> Result<AtomicGrid> {
        self.validate()?;
        build_pyscf_atom_grid(z, self)
    }
}

/// Molecular grid topology.
///
/// All point arrays are flat and indexed by a global point index. Points of an
/// atom are contiguous; use [`Self::atom_range`] to recover the interval. The
/// plan is independent of the atomic centres and of the partition weights for
/// geometry-independent topologies. A preset with `molecular_cutoffs`
/// computes the point set from the whole geometry; a plan built from such a
/// spec must not be reused after the geometry moves. Check
/// [`Self::geometry_dependent`] before caching a plan across geometries.
#[derive(Debug, Clone)]
pub struct GridPlan {
    atom_grids: Vec<Arc<AtomicGrid>>,
    atom_offsets: Vec<usize>,
    point_atom: Vec<u32>,
    local_coords: Vec<[f64; 3]>,
    quad_weights: Vec<f64>,
    geometry_dependent: bool,
}

impl GridPlan {
    /// Build the topology for `molecule` under `spec`.
    ///
    /// `spec` may be a [`GridSpec`], a [`GridPreset`],
    /// a [`PyscfGridSpec`], or a reference to any of them. Atomic grids are
    /// cached per element for geometry-independent specs. For a preset with
    /// molecular cutoffs, the coordinates are part of the topology and must
    /// not move after the plan is built.
    ///
    /// # Arguments
    ///
    /// * `molecule` — validated nuclear charges and positions in Bohr.
    /// * `spec` — grid build specification.
    ///
    /// ```
    /// use ruaa_dft_grids::grid::{GridPlan, PyscfGridSpec};
    /// use ruaa_dft_grids::molecule::Molecule;
    ///
    /// # fn main() -> ruaa_dft_grids::Result<()> {
    /// let water = Molecule::new([
    ///     (8u32, [0.0, 0.0, 0.0]),
    ///     (1, [0.0, 1.430, 1.110]),
    ///     (1, [0.0, -1.430, 1.110]),
    /// ])?;
    /// let plan = GridPlan::build(&water, PyscfGridSpec::from_level(0)?)?;
    /// assert_eq!(plan.atom_count(), 3);
    /// assert!(!plan.is_empty());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// [`Error::UnsupportedElement`], [`Error::MissingElementData`],
    /// [`Error::UnsupportedScheme`] or [`Error::UnsupportedLevel`] when the
    /// spec cannot produce an atomic grid for one of the charges.
    pub fn build<'a, M, G>(molecule: M, spec: G) -> Result<Self>
    where
        M: crate::molecule::AsMolecule<'a>,
        G: Into<GridSpec>,
    {
        let molecule = molecule.into_molecule()?;
        let spec = spec.into();
        spec.validate()?;
        let mut atom_grids = Vec::with_capacity(molecule.len());
        let mut cache: BTreeMap<u32, Arc<AtomicGrid>> = BTreeMap::new();

        for (ia, &z) in molecule.charges().iter().enumerate() {
            let grid = if spec.geometry_dependent() {
                Arc::new(spec.atomic_grid_at(molecule.as_ref(), ia)?)
            } else {
                match cache.get(&z) {
                    Some(grid) => Arc::clone(grid),
                    None => {
                        let grid = Arc::new(spec.atomic_grid(z)?);
                        cache.insert(z, Arc::clone(&grid));
                        grid
                    }
                }
            };
            atom_grids.push(grid);
        }

        Ok(Self::from_shared_grids(
            atom_grids,
            spec.geometry_dependent(),
        ))
    }

    /// Build a plan from a borrowed slice of `(Z, position)` pairs.
    ///
    /// Convenience shim around [`Self::build`] for code that does not keep a
    /// [`Molecule`] value.
    ///
    /// # Errors
    /// Same as [`Self::build`], plus [`Molecule::new`] for invalid atoms.
    pub fn build_from_atoms<G>(atoms: &[(u32, [f64; 3])], spec: G) -> Result<Self>
    where
        G: Into<GridSpec>,
    {
        Self::build(atoms, spec)
    }

    /// Build a plan from caller-supplied per-atom grids.
    ///
    /// This is the external extension point for a custom radial, angular or
    /// pruning construction: assemble each atom's [`AtomicGrid`] (or build the
    /// value struct directly) and pass them here. The resulting plan works
    /// with [`crate::partition::PartitionEvaluator`] and the range API;
    /// `build_molecular_grid` still requires a [`GridSpec`] because it also
    /// chooses the partition configuration.
    ///
    /// # Errors
    /// [`Error::GridArrayLength`] or [`Error::InvalidArgument`] if any grid is
    /// internally inconsistent.
    ///
    /// ```
    /// use ruaa_dft_grids::atomgrid::AtomicGrid;
    /// use ruaa_dft_grids::grid::GridPlan;
    ///
    /// # fn main() -> ruaa_dft_grids::Result<()> {
    /// let atom = AtomicGrid::new(
    ///     1,
    ///     vec![[0.0, 0.0, 0.0]],
    ///     vec![1.0],
    ///     vec![1],
    /// )?;
    /// let plan = GridPlan::from_atomic_grids(vec![atom])?;
    /// assert_eq!(plan.atom_count(), 1);
    /// assert_eq!(plan.len(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_atomic_grids(grids: Vec<AtomicGrid>) -> Result<Self> {
        let mut atom_grids = Vec::with_capacity(grids.len());
        for grid in grids {
            grid.validate()?;
            atom_grids.push(Arc::new(grid));
        }
        Ok(Self::from_shared_grids(atom_grids, false))
    }

    fn from_shared_grids(atom_grids: Vec<Arc<AtomicGrid>>, geometry_dependent: bool) -> Self {
        let mut atom_offsets = Vec::with_capacity(atom_grids.len() + 1);
        let mut point_atom = Vec::new();
        let mut local_coords = Vec::new();
        let mut quad_weights = Vec::new();

        atom_offsets.push(0);
        for (ia, grid) in atom_grids.iter().enumerate() {
            local_coords.extend_from_slice(&grid.points);
            quad_weights.extend_from_slice(&grid.quad_weights);
            point_atom.extend(std::iter::repeat_n(ia as u32, grid.points.len()));
            atom_offsets.push(local_coords.len());
        }

        Self {
            atom_grids,
            atom_offsets,
            point_atom,
            local_coords,
            quad_weights,
            geometry_dependent,
        }
    }

    /// Number of grid points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.local_coords.len()
    }

    /// Whether the plan has no points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.local_coords.is_empty()
    }

    /// Number of atoms.
    #[must_use]
    pub fn atom_count(&self) -> usize {
        self.atom_grids.len()
    }

    /// Whether this plan's point set depends on the full molecular geometry.
    ///
    /// `true` for a preset with `molecular_cutoffs`; such a plan is valid only
    /// for the geometry passed to [`Self::build`].  PySCF plans and plans from
    /// [`Self::from_atomic_grids`] are geometry-independent.
    #[must_use]
    pub fn geometry_dependent(&self) -> bool {
        self.geometry_dependent
    }

    /// The atomic grid of atom `ia`.
    ///
    /// # Panics
    /// If `ia >= self.atom_count()`.
    #[must_use]
    pub fn atom_grid(&self, ia: usize) -> &AtomicGrid {
        &self.atom_grids[ia]
    }

    /// Global point range owned by atom `ia`.
    ///
    /// # Panics
    /// If `ia >= self.atom_count()`.
    #[must_use]
    pub fn atom_range(&self, ia: usize) -> Range<usize> {
        self.atom_offsets[ia]..self.atom_offsets[ia + 1]
    }

    /// Parent atom of global point `index`.
    ///
    /// # Panics
    /// If `index >= self.len()`.
    #[must_use]
    pub fn parent_of(&self, index: usize) -> u32 {
        self.point_atom[index]
    }

    /// Local coordinate of global point `index`, shape `[3]`.
    ///
    /// # Panics
    /// If `index >= self.len()`.
    #[must_use]
    pub fn local_coord(&self, index: usize) -> [f64; 3] {
        self.local_coords[index]
    }

    /// Unpartitioned quadrature weight of global point `index`.
    ///
    /// # Panics
    /// If `index >= self.len()`.
    #[must_use]
    pub fn quad_weight(&self, index: usize) -> f64 {
        self.quad_weights[index]
    }

    /// Parent atom of every point in `range`, shape `[B]`
    /// (`B = range.len()`).
    ///
    /// # Panics
    /// If `range` is reversed or extends beyond [`Self::len`].
    #[must_use]
    pub fn point_atom(&self, range: Range<usize>) -> &[u32] {
        &self.point_atom[range]
    }

    /// Local coordinates in `range`, row-major `[B, 3]`
    /// (`B = range.len()`).
    ///
    /// # Panics
    /// If `range` is reversed or extends beyond [`Self::len`].
    #[must_use]
    pub fn local_coords(&self, range: Range<usize>) -> &[[f64; 3]] {
        &self.local_coords[range]
    }

    /// Unpartitioned quadrature weights in `range`, shape `[B]`
    /// (`B = range.len()`).
    ///
    /// # Panics
    /// If `range` is reversed or extends beyond [`Self::len`].
    #[must_use]
    pub fn quad_weights(&self, range: Range<usize>) -> &[f64] {
        &self.quad_weights[range]
    }

    /// Translate the points in `range` by their parent centres into `out`.
    ///
    /// # Arguments
    ///
    /// * `centers` — `[natm, 3]` row-major nuclear coordinates in Bohr;
    ///   `centers.len()` must equal [`Self::atom_count`].
    /// * `range` — global point interval `[start, end)`; `B = range.len()`.
    /// * `out` — `[B, 3]` row-major output coordinates; `out.len()` must
    ///   equal `range.len()`.
    ///
    /// # Errors
    /// [`Error::RangeOutOfBounds`] for a reversed or out-of-plan range, and
    /// [`Error::LengthMismatch`] if `centers` or `out` has the wrong length.
    pub fn coords_into(
        &self,
        centers: &[[f64; 3]],
        range: Range<usize>,
        out: &mut [[f64; 3]],
    ) -> Result<()> {
        self.check_range(&range)?;
        if centers.len() != self.atom_count() {
            return Err(Error::LengthMismatch {
                expected: self.atom_count(),
                got: centers.len(),
            });
        }
        if out.len() != range.len() {
            return Err(Error::LengthMismatch {
                expected: range.len(),
                got: out.len(),
            });
        }
        for (slot, g) in out.iter_mut().zip(range) {
            let ia = self.point_atom[g] as usize;
            let p = self.local_coords[g];
            let c = centers[ia];
            *slot = [c[0] + p[0], c[1] + p[1], c[2] + p[2]];
        }
        Ok(())
    }

    /// Allocate and translate the points in `range`.
    ///
    /// # Arguments
    ///
    /// * `centers` — `[natm, 3]` row-major nuclear coordinates in Bohr.
    /// * `range` — global point interval `[start, end)`.
    ///
    /// # Returns
    ///
    /// `[B, 3]` row-major translated coordinates, `B = range.len()`.
    ///
    /// # Errors
    /// [`Error::RangeOutOfBounds`] for a reversed or out-of-plan range, and
    /// [`Error::LengthMismatch`] if `centers.len() != self.atom_count()`.
    pub fn coords(&self, centers: &[[f64; 3]], range: Range<usize>) -> Result<Vec<[f64; 3]>> {
        self.check_range(&range)?;
        let mut out = vec![[0.0; 3]; range.len()];
        self.coords_into(centers, range, &mut out)?;
        Ok(out)
    }

    /// Contiguous ranges of at most `batch_size` points covering the plan.
    ///
    /// The ranges may cross atom boundaries; the evaluator splits them
    /// internally. `batch_size == 0` is treated as 1.
    pub fn batches(&self, batch_size: usize) -> impl Iterator<Item = Range<usize>> + '_ {
        let size = batch_size.max(1);
        let len = self.len();
        (0..len)
            .step_by(size)
            .map(move |start| start..(start + size).min(len))
    }

    fn check_range(&self, range: &Range<usize>) -> Result<()> {
        if range.start > range.end || range.end > self.len() {
            Err(Error::RangeOutOfBounds {
                start: range.start,
                end: range.end,
                len: self.len(),
            })
        } else {
            Ok(())
        }
    }
}

/// Reusable output buffer for one requested range.
///
/// The vectors are cleared, not deallocated, by [`Self::reset`]. For a values
/// request they contain `B = range.len()` entries.  The analytic derivative
/// API is separate ([`crate::partition::PartitionDerivEvaluator`]); this
/// buffer remains the value-only output, so its layout stays `[B]`/`[B, 3]`
/// and never grows a grid-sized derivative tensor.
///
/// [`Self::len`] is `weights.len()`; the evaluator writes all arrays together,
/// so callers must not mix arrays from different ranges.  Call [`Self::reset`]
/// before writing a new range, read the arrays through the accessors, and use
/// [`Self::check_shape`] to confirm they still agree with the range length.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchBuffer {
    /// Global range this buffer currently describes.
    pub(crate) range: Range<usize>,
    /// Translated point coordinates, `[B, 3]` row-major.
    pub(crate) coords: Vec<[f64; 3]>,
    /// Unpartitioned quadrature weights `q`, `[B]`.
    pub(crate) quad_weights: Vec<f64>,
    /// Normalized partition weights `P`, `[B]`.
    pub(crate) partition_weights: Vec<f64>,
    /// Total weights `q * P`, `[B]`.
    pub(crate) weights: Vec<f64>,
    /// Parent atom of each point, `[B]`.
    pub(crate) atom: Vec<u32>,
}

impl Default for BatchBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchBuffer {
    /// Empty buffer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            range: 0..0,
            coords: Vec::new(),
            quad_weights: Vec::new(),
            partition_weights: Vec::new(),
            weights: Vec::new(),
            atom: Vec::new(),
        }
    }

    /// Clear the buffer and record a new range, keeping the allocated capacity.
    pub fn reset(&mut self, range: Range<usize>) {
        self.range = range;
        self.coords.clear();
        self.quad_weights.clear();
        self.partition_weights.clear();
        self.weights.clear();
        self.atom.clear();
    }

    /// Number of points currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Whether the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Global range this buffer currently describes.
    #[must_use]
    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }

    /// Translated point coordinates, `[B, 3]` row-major.
    #[must_use]
    pub fn coords(&self) -> &[[f64; 3]] {
        &self.coords
    }

    /// Unpartitioned quadrature weights `q`, `[B]`.
    #[must_use]
    pub fn quad_weights(&self) -> &[f64] {
        &self.quad_weights
    }

    /// Normalized partition weights `P`, `[B]`.
    #[must_use]
    pub fn partition_weights(&self) -> &[f64] {
        &self.partition_weights
    }

    /// Total weights `q * P`, `[B]`.
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Parent atom of each point, `[B]`.
    #[must_use]
    pub fn atom(&self) -> &[u32] {
        &self.atom
    }

    /// Check that every point array has the length of the current range.
    ///
    /// # Errors
    /// [`Error::GridArrayLength`] naming the first inconsistent array.
    pub fn check_shape(&self) -> Result<()> {
        let expected = self.range.len();
        for (array, got) in [
            ("coords", self.coords.len()),
            ("quad_weights", self.quad_weights.len()),
            ("partition_weights", self.partition_weights.len()),
            ("weights", self.weights.len()),
            ("atom", self.atom.len()),
        ] {
            if got != expected {
                return Err(Error::GridArrayLength {
                    array,
                    expected,
                    got,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preset::ElementPreset;

    #[test]
    fn preset_spec_builds_and_carries_partition() {
        let preset = GridPreset::new(Radial::TreutlerAhlrichs)
            .with_default(ElementPreset::fixed(20, 1.0, 50));
        let spec = GridSpec::from_preset(preset.clone()).with_partition(PartitionConfig::pyscf());
        assert_eq!(spec.preset(), Some(&preset));
        assert!(!spec.is_custom());
        let grid = spec.atomic_grid(6).unwrap();
        assert_eq!(grid.nr(), 20);
        assert_eq!(spec.partition(), &PartitionConfig::pyscf());
    }

    #[test]
    fn custom_spec_has_no_topology() {
        let spec = GridSpec::custom(PartitionConfig::pyscf());
        assert!(spec.is_custom());
        assert!(matches!(
            spec.atomic_grid(6),
            Err(Error::UnsupportedGrid { .. })
        ));
        assert!(!spec.geometry_dependent());
        spec.validate().unwrap();
    }

    #[test]
    fn batch_buffer_check_shape_rejects_desynchronised_arrays() {
        let mut buffer = BatchBuffer::new();
        buffer.reset(0..2);
        buffer.coords.push([0.0; 3]);
        buffer.coords.push([1.0; 3]);
        buffer.weights.push(1.0);
        assert!(matches!(
            buffer.check_shape(),
            Err(Error::GridArrayLength {
                array: "quad_weights",
                expected: 2,
                got: 0,
            })
        ));
    }

    #[test]
    fn builder_defaults_are_classic_becke() {
        let pyscf = PyscfGridSpecBuilder::default()
            .build()
            .expect("valid PySCF defaults");
        assert_eq!(pyscf.level, 3);
        assert_eq!(
            pyscf.partition,
            PartitionConfig::new(PartitionScheme::Becke, RadiiConfig::pyscf())
        );

        let spec: GridSpec = pyscf.into();
        assert_eq!(
            spec.partition(),
            &PartitionConfig::new(PartitionScheme::Becke, RadiiConfig::pyscf())
        );
        assert_eq!(spec.pyscf_spec().unwrap().level, 3);
        assert!(!spec.geometry_dependent());
    }

    #[test]
    fn grid_plan_from_atomic_grids_preserves_parents() {
        let h = AtomicGrid::new(1, vec![[0.0, 0.0, 0.0]], vec![1.0], vec![1]).unwrap();
        let c = AtomicGrid::new(
            6,
            vec![[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            vec![2.0, 4.0],
            vec![1, 1],
        )
        .unwrap();

        let plan = GridPlan::from_atomic_grids(vec![h, c]).unwrap();
        assert_eq!(plan.atom_count(), 2);
        assert_eq!(plan.atom_range(0), 0..1);
        assert_eq!(plan.atom_range(1), 1..3);
        assert_eq!(plan.parent_of(0), 0);
        assert_eq!(plan.parent_of(1), 1);
        assert_eq!(plan.parent_of(2), 1);
        assert_eq!(plan.quad_weight(2), 4.0);
    }

    #[test]
    fn atomic_grid_validation_rejects_inconsistent_grids() {
        let bad = AtomicGrid {
            z: 1,
            points: Vec::new(),
            quad_weights: Vec::new(),
            shell_angular: vec![1],
        };
        assert!(matches!(
            bad.validate(),
            Err(Error::GridArrayLength {
                array: "atomic grid points",
                ..
            })
        ));
        assert!(GridPlan::from_atomic_grids(vec![bad]).is_err());

        let bad_weights = AtomicGrid {
            z: 1,
            points: vec![[0.0, 0.0, 0.0]],
            quad_weights: Vec::new(),
            shell_angular: vec![1],
        };
        assert!(matches!(
            bad_weights.validate(),
            Err(Error::GridArrayLength {
                array: "atomic grid quad_weights",
                ..
            })
        ));

        let bad_points = AtomicGrid {
            z: 1,
            points: vec![[0.0, 0.0, 0.0]],
            quad_weights: vec![1.0],
            shell_angular: vec![2],
        };
        assert!(matches!(
            bad_points.validate(),
            Err(Error::GridArrayLength {
                array: "atomic grid points",
                ..
            })
        ));
    }
}
