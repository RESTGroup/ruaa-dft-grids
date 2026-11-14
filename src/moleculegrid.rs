//! Full-materialisation molecular grid assembly and explicit post-build pruning.
//!
//! The structural pipeline lives in [`crate::grid`] and
//! [`crate::partition`]: a [`GridPlan`] holds the topology and a
//! [`PartitionEvaluator`] evaluates a caller-chosen point range into a
//! [`BatchBuffer`]. This module keeps a convenience [`MolecularGrid`] output
//! for callers that want the complete grid in memory.
//!
//! Every built [`MolecularGrid`] stores its [`GridSpec`] and, per point, the
//! Cartesian coordinate, the total weight `w = q * P`, the **unpartitioned
//! quadrature weight** `q`, the **normalized partition fraction** `P`, and the
//! source atom index.
//!
//! Pruning is deliberately not part of the build specification. The builders
//! materialise the full point set; [`MolecularGrid::split`] applies a
//! [`PostPrune`] policy afterwards and returns the retained and pruned grids.
//! The retained grid is a self-contained point set with its own [`len`],
//! [`points`], and [`total_weight`], so a consumer cannot accidentally keep
//! integrating over discarded points.
//!
//! Consumer shortcuts:
//!
//! * [`build_molecular_grid`] — the parallel one-call builder;
//! * [`MolecularGrid::weight_derivatives`] — analytic first derivatives of
//!   every total grid weight, all atoms, in the grid's current slot order;
//! * [`MolecularGrid::reorder_boxes`] / [`MolecularGrid::reorder_morton`] —
//!   apply a spatial [`GridOrder`] in one call.
//!
//! A post-prune policy is a point-set operation, not a differentiation
//! operation. `KeepIndices`/`KeepMask` fix a subset; `TotalWeightCut` and
//! `PartitionCut` choose a subset from the geometry, so their boundary is not
//! differentiable. A derivative consumer should treat a fixed retained mask as
//! the point set to differentiate.
//!
//! [`len`]: MolecularGrid::len
//! [`points`]: MolecularGrid::points
//! [`total_weight`]: MolecularGrid::total_weight

use crate::error::{Error, Result};
use crate::grid::{BatchBuffer, GridPlan, GridSpec};
use crate::molecule::Molecule;
use crate::order::GridOrder;
use crate::partition::{
    DerivOptions, DerivOutput, DerivPoints, DerivRequest, PartitionEvaluator, Workspace,
};

/// A molecular grid point with its total (quadrature x partition) weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridPoint {
    /// Cartesian coordinates in Bohr.
    pub xyz: [f64; 3],
    /// Total integration weight, `quad_weight * partition_fraction`.
    pub weight: f64,
    /// Unpartitioned atomic quadrature weight `q` (radial x angular).
    pub quad_weight: f64,
    /// Normalized partition fraction `P` of the generating atom; the
    /// all-atom fractions at the point sum to one.
    pub partition_fraction: f64,
    /// Index of the atom whose atomic grid generated this point.
    pub atom: u32,
}

/// A post-build point-selection policy for a [`MolecularGrid`].
///
/// The threshold variants compare strictly (`>`). The explicit-selection
/// variants keep the listed points verbatim.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum PostPrune {
    /// Keep points whose total-weight magnitude `|w| = |q * P|` is greater
    /// than `cutoff`.
    ///
    /// `w` has the same units as [`MolecularGrid::weights`] (Bohr^3 for a
    /// grid built in Bohr). The comparison uses the magnitude because
    /// quadrature weights are not guaranteed positive (the 74-point Lebedev
    /// rule has negative weights). This is a true grid-weight cutoff and has
    /// no ORCA keyword equivalent.
    TotalWeightCut(f64),

    /// Keep points whose normalized partition fraction `P` is greater than
    /// `cutoff`.
    ///
    /// `P` is dimensionless.  A program that cuts on its normalized
    /// partition fraction (for example ORCA's `WCut`) can be reproduced with
    /// this policy and that program's cutoff value.
    PartitionCut(f64),

    /// Keep exactly the listed point indices.
    ///
    /// Indices must be unique and in `0..len`. The retained grid preserves
    /// the original point order; use [`MolecularGrid::reorder`] afterwards if
    /// a different order is wanted.
    KeepIndices(Vec<u32>),

    /// Keep points where the mask is `true`.
    ///
    /// `mask.len()` must equal [`MolecularGrid::len`].
    KeepMask(Vec<bool>),
}

/// The two complementary grids produced by a post-build prune.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct GridSplit {
    /// Points that satisfy the post-prune policy.
    pub retained: MolecularGrid,
    /// Points rejected by the policy.
    ///
    /// This is a valid [`MolecularGrid`], but it is not an integration grid:
    /// its [`MolecularGrid::total_weight`] is the discarded weight.
    pub pruned: MolecularGrid,
}

/// A complete molecular grid: build spec, geometry, and (SoA) point data.
///
/// The fields are private so the per-point arrays stay parallel and every
/// source-atom index stays in range. Use [`Self::from_parts`] to construct a
/// custom grid, and the accessor methods to read it.
#[derive(Debug, Clone, PartialEq)]
pub struct MolecularGrid {
    spec: GridSpec,
    molecule: Molecule,
    coords: Vec<[f64; 3]>,
    weights: Vec<f64>,
    quad_weights: Vec<f64>,
    partition_fractions: Vec<f64>,
    atom_idx: Vec<u32>,
}

impl MolecularGrid {
    /// The exact build specification used.
    #[must_use]
    pub fn spec(&self) -> &GridSpec {
        &self.spec
    }

    /// The molecule this grid was built for.
    #[must_use]
    pub fn molecule(&self) -> &Molecule {
        &self.molecule
    }

    /// Nuclear coordinates in Bohr.
    #[must_use]
    pub fn atom_coords(&self) -> &[[f64; 3]] {
        self.molecule.centers()
    }

    /// Atomic numbers, one entry per atom.
    #[must_use]
    pub fn atom_charges(&self) -> &[u32] {
        self.molecule.charges()
    }

    /// Point coordinates in Bohr.
    #[must_use]
    pub fn coords(&self) -> &[[f64; 3]] {
        &self.coords
    }

    /// Total weights `q * P`.
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Unpartitioned quadrature weights `q`.
    #[must_use]
    pub fn quad_weights(&self) -> &[f64] {
        &self.quad_weights
    }

    /// Normalized partition fractions `P` of the generating atoms; the
    /// all-atom fractions at each point sum to one.
    #[must_use]
    pub fn partition_fractions(&self) -> &[f64] {
        &self.partition_fractions
    }

    /// Source atom index per point.
    #[must_use]
    pub fn atom_idx(&self) -> &[u32] {
        &self.atom_idx
    }

    /// Number of grid points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.coords.len()
    }

    /// Whether the grid has no points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }

    /// Number of atoms in the geometry.
    #[must_use]
    pub fn atom_count(&self) -> usize {
        self.molecule.len()
    }

    /// Iterate points as [`GridPoint`] values.
    pub fn points(&self) -> impl Iterator<Item = GridPoint> + '_ {
        self.coords
            .iter()
            .zip(&self.weights)
            .zip(&self.quad_weights)
            .zip(&self.partition_fractions)
            .zip(&self.atom_idx)
            .map(
                |((((&xyz, &weight), &quad_weight), &partition_fraction), &atom)| GridPoint {
                    xyz,
                    weight,
                    quad_weight,
                    partition_fraction,
                    atom,
                },
            )
    }

    /// Sum of the total (partitioned) weights.
    #[must_use]
    pub fn total_weight(&self) -> f64 {
        self.weights.iter().sum()
    }

    /// Sum of the unpartitioned quadrature weights.
    #[must_use]
    pub fn total_quad_weight(&self) -> f64 {
        self.quad_weights.iter().sum()
    }

    /// Sum of the quadrature weights generated by each atom.
    #[must_use]
    pub fn quad_weight_by_atom(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.atom_count()];
        for (&q, &ia) in self.quad_weights.iter().zip(&self.atom_idx) {
            out[ia as usize] += q;
        }
        out
    }

    /// Analytic first derivatives of every grid weight, for every atom.
    ///
    /// This is the one-call form of [`Self::partition_deriv_evaluator`] plus
    /// [`PartitionDerivEvaluator::fill`](crate::partition::PartitionDerivEvaluator::fill)
    /// for the common case: all atoms selected, bare partition weights and
    /// raw first derivatives requested, evaluated in parallel on the ambient
    /// Rayon pool in the grid's **current slot order**.  No [`GridPlan`] is
    /// needed, so a post-pruned or [`Self::reorder`]ed grid is evaluated as
    /// is.
    ///
    /// # Output (row-major, `G = self.len()`, `A = self.atom_count()`)
    ///
    /// * [`DerivOutput::partition_weights`] — `[G]`, `Pbar / Z` (no
    ///   quadrature factor).
    /// * [`DerivOutput::dw`] — `[A, 3, G]`, derivative of the total grid
    ///   weight `q * Pbar / Z` with respect to the nuclear coordinates,
    ///   flattened `(atom * 3 + component) * G + point`.
    /// * [`DerivOutput::dw_atom`] — the `[3, G]` block of one atom.
    ///
    /// The quadrature weight `q` is constant in the atomic frame, so divide
    /// `dw` by [`Self::quad_weights`] to obtain the derivative of the bare
    /// partition fraction.  For a selected set of atoms, contractions, or a
    /// Hessian, build the evaluator explicitly with
    /// [`Self::partition_deriv_evaluator`] and pass the wanted
    /// [`DerivRequest`].  For a point interval use
    /// [`DerivPoints::grid`](crate::partition::DerivPoints::grid) with
    /// [`PartitionDerivEvaluator::fill`](crate::partition::PartitionDerivEvaluator::fill)
    /// (or [`PartitionDerivEvaluator::fill_serial`](crate::partition::PartitionDerivEvaluator::fill_serial));
    /// for a custom raw-Hessian reduction use
    /// [`PartitionDerivEvaluator::fill_serial_visit`](crate::partition::PartitionDerivEvaluator::fill_serial_visit).
    ///
    /// **Moving cutoff.** For a preset with
    /// [`GridSpec::geometry_dependent`] true (molecular cutoffs enabled), this
    /// differentiates the partition fractions of the *current* point set; it
    /// does not differentiate the geometry-dependent point selection. Disable
    /// the molecular cutoffs or add the moving-support term separately.
    ///
    /// ```
    /// use ruaa_dft_grids::grid::PyscfGridSpec;
    /// use ruaa_dft_grids::moleculegrid::build_molecular_grid;
    ///
    /// # fn main() -> ruaa_dft_grids::Result<()> {
    /// let water = [
    ///     (8u32, [0.0, 0.0, 0.0]),
    ///     (1, [0.0, 1.430, 1.110]),
    ///     (1, [0.0, -1.430, 1.110]),
    /// ];
    /// let grid = build_molecular_grid(&water, &PyscfGridSpec::from_level(0)?)?;
    /// let deriv = grid.weight_derivatives()?;
    /// assert_eq!(deriv.dw().len(), grid.atom_count() * 3 * grid.len());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// [`Error::UnsupportedDerivative`] when the partition has no analytic
    /// derivative backend — notably classic Becke with the early exit enabled
    /// and `mbecke_clamp < 1.0`.  Both truncate or clamp the pair product, so
    /// the weight is discontinuous.  Use classic Becke without the early
    /// exit, SSF, SS, LKO, or mBecke at the default clamp.
    /// Also propagates the grid validation errors of
    /// [`PartitionDerivEvaluator::fill`](crate::partition::PartitionDerivEvaluator::fill).
    pub fn weight_derivatives(&self) -> Result<DerivOutput> {
        let evaluator = self.partition_deriv_evaluator(DerivOptions::default())?;
        let request = DerivRequest {
            weights: true,
            dw: true,
            ..DerivRequest::default()
        };
        let mut output = DerivOutput::new();
        evaluator.fill(DerivPoints::all_grid(self), &request, &mut output)?;
        Ok(output)
    }

    /// Reorder every per-point array under `order`.
    ///
    /// `coords`, `weights`, `quad_weights`, `partition_fractions` and
    /// `atom_idx` are permuted; `spec`, `atom_coords` and `atom_charges` are
    /// untouched. The point set
    /// and the total weight are invariant, so an order can be applied after
    /// the grid is built without changing what it integrates. `order` must
    /// cover this grid's current points (built from these `coords`, or over the
    /// same count); [`GridOrder::inverse`] maps a sorted slot back to its
    /// original index.
    ///
    /// After a [`Self::split`], apply an order to each resulting grid
    /// separately; the retained/pruned pair no longer shares one index space.
    ///
    /// # Errors
    /// [`Error::LengthMismatch`] if `order.len()` differs from [`Self::len`].
    pub fn reorder(&mut self, order: &GridOrder) -> Result<()> {
        if order.len() != self.len() {
            return Err(Error::LengthMismatch {
                expected: self.len(),
                got: order.len(),
            });
        }
        order.permute_in_place(&mut self.coords)?;
        order.permute_in_place(&mut self.weights)?;
        order.permute_in_place(&mut self.quad_weights)?;
        order.permute_in_place(&mut self.partition_fractions)?;
        order.permute_in_place(&mut self.atom_idx)?;
        Ok(())
    }

    /// Sort the points into uniform cubic boxes and apply the permutation.
    ///
    /// This is [`GridOrder::boxes`] followed by [`Self::reorder`], with the
    /// order returned so the caller can iterate its sorted blocks:
    ///
    /// ```
    /// use ruaa_dft_grids::grid::PyscfGridSpec;
    /// use ruaa_dft_grids::moleculegrid::build_molecular_grid;
    ///
    /// # fn main() -> ruaa_dft_grids::Result<()> {
    /// let water = [
    ///     (8u32, [0.0, 0.0, 0.0]),
    ///     (1, [0.0, 1.430, 1.110]),
    ///     (1, [0.0, -1.430, 1.110]),
    /// ];
    /// let mut grid = build_molecular_grid(&water, &PyscfGridSpec::from_level(0)?)?;
    /// let before = grid.total_weight();
    /// let order = grid.reorder_boxes(1.2)?;   // PySCF-style 1.2 Bohr boxes
    /// let scale = before.abs().max(1.0);
    /// assert!((grid.total_weight() - before).abs() < 1e-8 * scale);
    /// for block in order.blocks(64) {
    ///     assert!(!block.is_empty());
    ///     break;
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// The point set, every total, and the analytic derivatives are
    /// invariant; only the slot order changes.  `box_size` must be finite and
    /// positive (Bohr).
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] for a bad `box_size`, and
    /// [`Error::TooManyPoints`] when the grid exceeds the `u32` index space.
    pub fn reorder_boxes(&mut self, box_size: f64) -> Result<GridOrder> {
        let order = GridOrder::boxes(&self.coords, box_size)?;
        self.reorder(&order)?;
        Ok(order)
    }

    /// Sort the points along a Morton (Z-order) curve and apply the
    /// permutation.
    ///
    /// This is [`GridOrder::morton`] followed by [`Self::reorder`].  The
    /// curve keeps nearby points close in memory with a single sort; `cell`
    /// is the quantisation cell size in Bohr.  See [`Self::reorder_boxes`]
    /// for the simpler lexicographic variant.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] for a bad `cell`, and
    /// [`Error::TooManyPoints`] when the grid exceeds the `u32` index space.
    pub fn reorder_morton(&mut self, cell: f64) -> Result<GridOrder> {
        let order = GridOrder::morton(&self.coords, cell)?;
        self.reorder(&order)?;
        Ok(order)
    }

    /// Build the keep mask for `policy` without splitting the grid.
    ///
    /// Threshold policies compare strictly (`>`) against `|w|` for
    /// [`PostPrune::TotalWeightCut`] and against `P` for
    /// [`PostPrune::PartitionCut`].
    ///
    /// # Errors
    /// [`Error::InvalidCutoff`] for a non-finite or negative threshold;
    /// [`Error::PruneIndexOutOfRange`] / [`Error::PruneIndexDuplicate`] for a
    /// bad `KeepIndices` list; [`Error::LengthMismatch`] for a `KeepMask` of
    /// the wrong length; [`Error::GridArrayLength`] /
    /// [`Error::InvalidGridAtom`] if the grid's point arrays are inconsistent.
    pub fn keep_mask(&self, policy: &PostPrune) -> Result<Vec<bool>> {
        self.validate()?;
        let n = self.len();
        match policy {
            PostPrune::TotalWeightCut(cutoff) => {
                check_cutoff(*cutoff)?;
                Ok(self.weights.iter().map(|&w| w.abs() > *cutoff).collect())
            }
            PostPrune::PartitionCut(cutoff) => {
                check_cutoff(*cutoff)?;
                Ok(self
                    .partition_fractions
                    .iter()
                    .map(|&p| p > *cutoff)
                    .collect())
            }
            PostPrune::KeepIndices(indices) => {
                let mut mask = vec![false; n];
                for &index in indices {
                    let index = index as usize;
                    if index >= n {
                        return Err(Error::PruneIndexOutOfRange { index, len: n });
                    }
                    if mask[index] {
                        return Err(Error::PruneIndexDuplicate { index });
                    }
                    mask[index] = true;
                }
                Ok(mask)
            }
            PostPrune::KeepMask(mask) => {
                if mask.len() != n {
                    return Err(Error::LengthMismatch {
                        expected: n,
                        got: mask.len(),
                    });
                }
                Ok(mask.clone())
            }
        }
    }

    /// Split this grid into the points retained by `policy` and the rest.
    ///
    /// The two grids partition the original point set: their point counts sum
    /// to [`Self::len`], their per-point arrays are disjoint, and neither is
    /// renormalized. Both carry the same `spec` and `atoms`, so each is
    /// self-describing. The retained and pruned grids preserve the original
    /// relative point order.
    ///
    /// ```
    /// use ruaa_dft_grids::grid::PyscfGridSpec;
    /// use ruaa_dft_grids::moleculegrid::{PostPrune, build_molecular_grid};
    ///
    /// # fn main() -> ruaa_dft_grids::Result<()> {
    /// let water = [
    ///     (8u32, [0.0, 0.0, 0.0]),
    ///     (1, [0.0, 1.430, 1.110]),
    ///     (1, [0.0, -1.430, 1.110]),
    /// ];
    /// let grid = build_molecular_grid(&water, &PyscfGridSpec::from_level(0)?)?;
    /// let n = grid.len();
    /// let split = grid.split(PostPrune::KeepIndices(vec![0u32, (n - 1) as u32]))?;
    /// assert_eq!(split.retained.len(), 2);
    /// assert_eq!(split.retained.len() + split.pruned.len(), n);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// The same validation errors as [`Self::keep_mask`].
    pub fn split(self, policy: PostPrune) -> Result<GridSplit> {
        let keep = self.keep_mask(&policy)?;
        Ok(split_owned(self, &keep))
    }

    /// Split this grid in place: `self` becomes the retained set and the
    /// pruned grid is returned.
    ///
    /// This is the in-place form of [`Self::split`] when the caller only
    /// needs the discarded points for diagnostics; the retained arrays are
    /// moved back into `self` rather than copied.
    ///
    /// # Errors
    /// The same validation errors as [`Self::keep_mask`].
    pub fn split_off_pruned(&mut self, policy: PostPrune) -> Result<MolecularGrid> {
        let keep = self.keep_mask(&policy)?;
        let grid = MolecularGrid {
            spec: self.spec.clone(),
            molecule: self.molecule.clone(),
            coords: std::mem::take(&mut self.coords),
            weights: std::mem::take(&mut self.weights),
            quad_weights: std::mem::take(&mut self.quad_weights),
            partition_fractions: std::mem::take(&mut self.partition_fractions),
            atom_idx: std::mem::take(&mut self.atom_idx),
        };
        let split = split_owned(grid, &keep);
        self.coords = split.retained.coords;
        self.weights = split.retained.weights;
        self.quad_weights = split.retained.quad_weights;
        self.partition_fractions = split.retained.partition_fractions;
        self.atom_idx = split.retained.atom_idx;
        Ok(split.pruned)
    }

    /// Build a grid from its parts and validate the point arrays.
    ///
    /// This is the constructor for custom topologies. `coords`, `weights`,
    /// `quad_weights`, `partition_fractions` and `atom_idx` must all have the
    /// same length; every `atom_idx` entry must index `molecule`.
    ///
    /// # Errors
    /// [`Error::GridArrayLength`], [`Error::InvalidGridAtom`],
    /// [`Error::UnsupportedElement`] or [`Error::NonFiniteCoordinate`] for an
    /// inconsistent or invalid grid.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        spec: GridSpec,
        molecule: Molecule,
        coords: Vec<[f64; 3]>,
        weights: Vec<f64>,
        quad_weights: Vec<f64>,
        partition_fractions: Vec<f64>,
        atom_idx: Vec<u32>,
    ) -> Result<Self> {
        let grid = Self {
            spec,
            molecule,
            coords,
            weights,
            quad_weights,
            partition_fractions,
            atom_idx,
        };
        grid.validate()?;
        Ok(grid)
    }

    /// Validate the molecule, array lengths, atom indices and finiteness.
    ///
    /// Negative quadrature and total weights are allowed; the 74-, 230- and
    /// 266-point Lebedev rules have negative angular weights.
    ///
    /// # Errors
    /// [`Error::GridArrayLength`], [`Error::InvalidGridAtom`],
    /// [`Error::UnsupportedElement`], [`Error::NonFiniteCoordinate`] or
    /// [`Error::InvalidArgument`] for a malformed grid.
    pub fn validate(&self) -> Result<()> {
        self.spec.validate()?;
        self.molecule.validate()?;
        let expected = self.coords.len();
        for (array, got) in [
            ("weights", self.weights.len()),
            ("quad_weights", self.quad_weights.len()),
            ("partition_fractions", self.partition_fractions.len()),
            ("atom_idx", self.atom_idx.len()),
        ] {
            if got != expected {
                return Err(Error::GridArrayLength {
                    array,
                    expected,
                    got,
                });
            }
        }
        if self
            .coords
            .iter()
            .any(|p| p.iter().any(|value| !value.is_finite()))
            || self.weights.iter().any(|value| !value.is_finite())
            || self.quad_weights.iter().any(|value| !value.is_finite())
            || self
                .partition_fractions
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(Error::InvalidArgument {
                argument: "molecular grid values",
            });
        }
        for ((&weight, &quad), &fraction) in self
            .weights
            .iter()
            .zip(&self.quad_weights)
            .zip(&self.partition_fractions)
        {
            let product = quad * fraction;
            let scale = weight.abs().max(product.abs()).max(1.0);
            if (weight - product).abs() > 1e-12 * scale {
                return Err(Error::InvalidArgument {
                    argument: "molecular grid weight does not equal q * P",
                });
            }
        }
        for &atom in &self.atom_idx {
            if atom as usize >= self.atom_count() {
                return Err(Error::InvalidGridAtom {
                    atom,
                    natm: self.atom_count(),
                });
            }
        }
        Ok(())
    }

    fn with_capacity(spec: &GridSpec, molecule: &Molecule, capacity: usize) -> MolecularGrid {
        MolecularGrid {
            spec: spec.clone(),
            molecule: molecule.clone(),
            coords: Vec::with_capacity(capacity),
            weights: Vec::with_capacity(capacity),
            quad_weights: Vec::with_capacity(capacity),
            partition_fractions: Vec::with_capacity(capacity),
            atom_idx: Vec::with_capacity(capacity),
        }
    }

    fn push_point(
        &mut self,
        xyz: [f64; 3],
        weight: f64,
        quad_weight: f64,
        partition_fraction: f64,
        atom: u32,
    ) {
        self.coords.push(xyz);
        self.weights.push(weight);
        self.quad_weights.push(quad_weight);
        self.partition_fractions.push(partition_fraction);
        self.atom_idx.push(atom);
    }
}

fn check_cutoff(cutoff: f64) -> Result<()> {
    if cutoff.is_finite() && cutoff >= 0.0 {
        Ok(())
    } else {
        Err(Error::InvalidCutoff { cutoff })
    }
}

fn split_owned(grid: MolecularGrid, keep: &[bool]) -> GridSplit {
    let MolecularGrid {
        spec,
        molecule,
        coords,
        weights,
        quad_weights,
        partition_fractions,
        atom_idx,
    } = grid;
    let kept = keep.iter().filter(|&&k| k).count();
    let mut retained = MolecularGrid::with_capacity(&spec, &molecule, kept);
    let mut pruned = MolecularGrid::with_capacity(&spec, &molecule, keep.len() - kept);
    for (((((xyz, weight), quad_weight), partition_fraction), atom), &k) in coords
        .into_iter()
        .zip(weights)
        .zip(quad_weights)
        .zip(partition_fractions)
        .zip(atom_idx)
        .zip(keep)
    {
        let dst = if k { &mut retained } else { &mut pruned };
        dst.push_point(xyz, weight, quad_weight, partition_fraction, atom);
    }
    GridSplit { retained, pruned }
}

/// Assemble the complete molecular grid for `molecule` under `spec`.
///
/// This is the parallel one-call builder. It materialises every point; it
/// does not apply a cutoff. The partition weights are evaluated in parallel
/// on the ambient Rayon pool; use a Rayon thread pool (or
/// `RAYON_NUM_THREADS`) to control the worker count. Atomic grids are cached
/// per element where the topology is geometry-independent, which keeps the
/// serial part small for typical molecules. Use [`MolecularGrid::split`] with a [`PostPrune`]
/// policy afterwards, or [`GridPlan`] and [`PartitionEvaluator`] directly for
/// range-streaming builds.
///
/// # Arguments
///
/// * `molecule` — validated nuclear charges and positions in Bohr.
/// * `spec` — grid build specification.
///
/// # Returns
///
/// A [`MolecularGrid`] whose per-point arrays have shape `[G]` or `[G, 3]`
/// (`G = number of points`).
///
/// ```
/// use ruaa_dft_grids::grid::PyscfGridSpec;
/// use ruaa_dft_grids::molecule::Molecule;
/// use ruaa_dft_grids::moleculegrid::build_molecular_grid;
///
/// # fn main() -> ruaa_dft_grids::Result<()> {
/// let water = Molecule::new([
///     (8u32, [0.0, 0.0, 0.0]),
///     (1, [0.0, 1.430, 1.110]),
///     (1, [0.0, -1.430, 1.110]),
/// ])?;
/// let grid = build_molecular_grid(&water, &PyscfGridSpec::from_level(0)?)?;
/// assert_eq!(grid.atom_count(), 3);
/// assert_eq!(grid.coords().len(), grid.len());
/// # Ok(())
/// # }
/// ```
///
/// # Errors
/// Propagates [`GridPlan::build`]'s topology errors and
/// [`PartitionEvaluator::new`]'s radii errors.
pub fn build_molecular_grid<'a, M, G>(molecule: M, spec: G) -> Result<MolecularGrid>
where
    M: crate::molecule::AsMolecule<'a>,
    G: Into<GridSpec>,
{
    let spec = spec.into();
    let molecule = molecule.into_molecule()?;
    let plan = GridPlan::build(molecule.as_ref(), &spec)?;
    let evaluator = PartitionEvaluator::new(molecule.as_ref(), spec.partition().clone())?;
    let mut batch = BatchBuffer::new();
    let mut workspace = Workspace::new();
    evaluator.fill_plan_par(&plan, 0..plan.len(), &mut workspace, &mut batch)?;

    MolecularGrid::from_parts(
        spec,
        molecule.into_owned(),
        batch.coords,
        batch.weights,
        batch.quad_weights,
        batch.partition_weights,
        batch.atom,
    )
}

/// Build a molecular grid from a borrowed slice of `(Z, position)` pairs.
///
/// Convenience shim around [`build_molecular_grid`] for code that does not
/// keep a [`Molecule`] value.
///
/// # Errors
/// Same as [`build_molecular_grid`], plus [`Molecule::new`] for invalid atoms.
pub fn build_molecular_grid_from_atoms<G>(
    atoms: &[(u32, [f64; 3])],
    spec: G,
) -> Result<MolecularGrid>
where
    G: Into<GridSpec>,
{
    build_molecular_grid(atoms, spec)
}

/// Materialise a caller-built [`GridPlan`] with its molecule and spec.
///
/// This is the composition path for a custom atomic topology: build the plan
/// (for example with [`GridPlan::from_atomic_grids`]), then materialise it
/// without rebuilding the topology.  The evaluator is built from
/// `spec.partition()`.
///
/// # Errors
/// [`Error::LengthMismatch`] if the plan and molecule have different atom
/// counts, [`Error::ChargesMismatch`] if a plan grid's `z` differs from the
/// molecule, and the partition/grid validation errors.
pub fn build_molecular_grid_from_plan(
    molecule: &Molecule,
    plan: &GridPlan,
    spec: GridSpec,
) -> Result<MolecularGrid> {
    molecule.validate()?;
    if plan.atom_count() != molecule.len() {
        return Err(Error::LengthMismatch {
            expected: molecule.len(),
            got: plan.atom_count(),
        });
    }
    for (atom, (&expected, grid)) in molecule
        .charges()
        .iter()
        .zip((0..plan.atom_count()).map(|ia| plan.atom_grid(ia)))
        .enumerate()
    {
        if grid.z != expected {
            return Err(Error::ChargesMismatch {
                atom,
                expected,
                got: grid.z,
            });
        }
    }
    let evaluator = PartitionEvaluator::new(molecule, spec.partition().clone())?;
    let mut batch = BatchBuffer::new();
    let mut workspace = Workspace::new();
    evaluator.fill_plan_par(plan, 0..plan.len(), &mut workspace, &mut batch)?;
    MolecularGrid::from_parts(
        spec,
        molecule.clone(),
        batch.coords,
        batch.weights,
        batch.quad_weights,
        batch.partition_weights,
        batch.atom,
    )
}

/// Materialise a set of caller-built atomic grids into a molecular grid.
///
/// This is the convenience path for a construction that is not expressible
/// as a [`GridSpec`] topology — for example a specialized angular rule, a
/// program-specific simple grid, or a per-atom rule chosen outside the crate.
/// The atomic grids must be in molecule order and carry the matching nuclear
/// charges; the partition configuration is used as given.
///
/// # Errors
/// The [`GridPlan::from_atomic_grids`] validation errors, plus the errors of
/// [`build_molecular_grid_from_plan`].
pub fn build_molecular_grid_from_atomic_grids(
    molecule: &Molecule,
    grids: Vec<crate::atomgrid::AtomicGrid>,
    partition: crate::partition::PartitionConfig,
) -> Result<MolecularGrid> {
    let plan = GridPlan::from_atomic_grids(grids)?;
    build_molecular_grid_from_plan(molecule, &plan, GridSpec::custom(partition))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::PyscfGridSpecBuilder;
    use crate::partition::{PartitionConfig, PartitionConfigBuilder, PartitionScheme};

    const PARTITION_CUT: f64 = 1e-14;

    fn pyscf_spec(partition: PartitionConfig) -> GridSpec {
        PyscfGridSpecBuilder::default()
            .partition(partition)
            .build()
            .unwrap()
            .into()
    }

    fn test_partition() -> GridSpec {
        let partition = PartitionConfigBuilder::default()
            .scheme(PartitionScheme::MBecke)
            .radii(crate::partition::test_support::custom_radii())
            .becke_early_exit(true)
            .build()
            .unwrap();
        pyscf_spec(partition)
    }

    /// The metadata invariant that the future gradient path relies on:
    /// `weight == quad_weight * P`, every point names a real atom, and the
    /// unpartitioned weights are positive. The full grid may contain
    /// zero-weight points; a post-build cut removes them.
    #[test]
    fn point_metadata_invariants() {
        let atoms = [(1u32, [-0.7, 0.0, 0.0]), (1, [0.7, 0.0, 0.0])];
        let grid = build_molecular_grid(&atoms, test_partition()).unwrap();
        assert_eq!(grid.len(), grid.coords.len());
        assert!(grid.len() > 1000);
        for p in grid.points() {
            assert!((p.atom as usize) < atoms.len());
            assert!(p.weight.is_finite());
            assert!(p.quad_weight.is_finite() && p.quad_weight != 0.0);
            assert!(p.partition_fraction.is_finite());
            assert!((0.0..=1.0).contains(&p.partition_fraction));
            assert_eq!(
                p.weight.to_bits(),
                (p.quad_weight * p.partition_fraction).to_bits()
            );
        }
        assert!(grid.points().any(|p| p.weight > 0.0));
        assert!(grid.points().any(|p| p.partition_fraction == 0.0));
        let by_atom = grid.quad_weight_by_atom();
        assert!(by_atom.iter().all(|&w| w > 0.0));
    }

    #[test]
    fn custom_plan_materialises_without_rebuilding() {
        let atoms = [(8u32, [0.0, 0.0, 0.0]), (1, [0.0, 1.43, 1.11])];
        let molecule = Molecule::new(atoms).unwrap();
        let spec: GridSpec = crate::grid::PyscfGridSpec::from_level(0).unwrap().into();
        let plan = GridPlan::build(&molecule, &spec).unwrap();
        let grid = build_molecular_grid_from_plan(&molecule, &plan, spec).unwrap();
        assert_eq!(grid.len(), plan.len());
        assert_eq!(grid.atom_count(), molecule.len());
    }

    /// Building materialises every point; the cutoff is a post-build policy.
    #[test]
    fn build_does_not_filter_points() {
        let atoms = [(1u32, [-0.7, 0.0, 0.0]), (1, [0.7, 0.0, 0.0])];
        let spec = test_partition();
        let grid = build_molecular_grid(&atoms, &spec).unwrap();
        let plan = GridPlan::build(&atoms, &spec).unwrap();
        assert_eq!(grid.len(), plan.len());
        assert_eq!(grid.partition_fractions.len(), grid.len());
    }

    #[test]
    fn h2_grid_sane() {
        let atoms = [(1u32, [-0.7, 0.0, 0.0]), (1, [0.7, 0.0, 0.0])];
        let grid = build_molecular_grid(&atoms, test_partition()).unwrap();
        assert!(grid.len() > 1000, "only {} points", grid.len());
        assert!(grid.points().all(|p| p.weight.is_finite()));
    }

    /// Partition invariance: for a common density rho the partitioned sum
    /// equals the unpartitioned one, i.e. the grid reproduces
    /// `int rho d^3r = N * pi^{3/2}` for N unit s-Gaussians.
    #[test]
    fn partitioned_integration_partition_invariant() {
        let atoms = [(1u32, [0.0, 0.0, 0.0]), (1, [1.5, 0.2, -0.3])];
        let grid = build_molecular_grid(&atoms, test_partition()).unwrap();
        let val: f64 = grid
            .points()
            .map(|p| {
                let [x, y, z] = p.xyz;
                let rho: f64 = atoms
                    .iter()
                    .map(|&(_, [ax, ay, az])| {
                        (-(x - ax) * (x - ax) - (y - ay) * (y - ay) - (z - az) * (z - az)).exp()
                    })
                    .sum();
                p.weight * rho
            })
            .sum();
        let target = 2.0 * std::f64::consts::PI.powf(1.5);
        let err = (val / target - 1.0).abs();
        assert!(err < 2e-2, "val {val} vs {target}, rel err {err}");
    }

    /// mBecke (2022) partition reproduces classic Becke weights (within
    /// floating-point evaluation order) and integrates the same density.
    #[test]
    fn mbecke_grid_partition_invariant() {
        let atoms = [(1u32, [0.0, 0.0, 0.0]), (6, [1.5, 0.2, -0.3])];
        for partition in [
            PartitionConfigBuilder::default()
                .scheme(PartitionScheme::Becke)
                .radii(crate::partition::test_support::custom_radii())
                .becke_early_exit(true)
                .build()
                .unwrap(),
            PartitionConfigBuilder::default()
                .scheme(PartitionScheme::MBecke)
                .radii(crate::partition::test_support::custom_radii())
                .becke_early_exit(true)
                .build()
                .unwrap(),
        ] {
            let spec = pyscf_spec(partition);
            let grid = build_molecular_grid(&atoms, &spec).unwrap();
            let val: f64 = grid
                .points()
                .map(|p| {
                    let [x, y, z] = p.xyz;
                    let rho: f64 = atoms
                        .iter()
                        .map(|&(_, [ax, ay, az])| {
                            (-(x - ax) * (x - ax) - (y - ay) * (y - ay) - (z - az) * (z - az)).exp()
                        })
                        .sum();
                    p.weight * rho
                })
                .sum();
            let target = 2.0 * std::f64::consts::PI.powf(1.5);
            let err = (val / target - 1.0).abs();
            assert!(
                err < 2e-2,
                "{:?}: rel err {err}",
                grid.spec.partition().scheme
            );
        }
    }

    /// Cutoffs are validated by the post-prune policy, not by the builder.
    #[test]
    fn post_prune_validation() {
        let atoms = [(1u32, [0.0, 0.0, 0.0])];
        let grid = build_molecular_grid(&atoms, test_partition()).unwrap();
        for policy in [
            PostPrune::TotalWeightCut(f64::NAN),
            PostPrune::TotalWeightCut(-1.0),
            PostPrune::TotalWeightCut(f64::INFINITY),
            PostPrune::PartitionCut(f64::NAN),
            PostPrune::PartitionCut(-1.0),
            PostPrune::PartitionCut(f64::NEG_INFINITY),
        ] {
            assert!(matches!(
                grid.keep_mask(&policy).unwrap_err(),
                Error::InvalidCutoff { .. }
            ));
        }

        // Zero is a legal threshold; strict `>` keeps all positive weights.
        assert!(
            grid.keep_mask(&PostPrune::PartitionCut(0.0))
                .unwrap()
                .iter()
                .all(|&k| k)
        );
        // A lone atom has P = 1, so `> 1` keeps nothing.
        assert!(
            grid.keep_mask(&PostPrune::PartitionCut(1.0))
                .unwrap()
                .iter()
                .all(|&k| !k)
        );
    }

    #[test]
    fn post_prune_index_and_mask_validation() {
        let atoms = [(1u32, [0.0, 0.0, 0.0])];
        let grid = build_molecular_grid(&atoms, test_partition()).unwrap();
        let n = grid.len();

        assert_eq!(
            grid.keep_mask(&PostPrune::KeepIndices(vec![n as u32]))
                .unwrap_err(),
            Error::PruneIndexOutOfRange { index: n, len: n }
        );
        assert_eq!(
            grid.keep_mask(&PostPrune::KeepIndices(vec![0, 0]))
                .unwrap_err(),
            Error::PruneIndexDuplicate { index: 0 }
        );
        assert_eq!(
            grid.keep_mask(&PostPrune::KeepMask(Vec::new()))
                .unwrap_err(),
            Error::LengthMismatch {
                expected: n,
                got: 0
            }
        );
    }

    #[test]
    fn split_retained_and_pruned_partition_the_grid() {
        let atoms = [(1u32, [-0.7, 0.0, 0.0]), (1, [0.7, 0.0, 0.0])];
        let grid = build_molecular_grid(&atoms, test_partition()).unwrap();
        let full_len = grid.len();
        let full_weight = grid.total_weight();
        let full_quad = grid.total_quad_weight();

        let split = grid
            .clone()
            .split(PostPrune::PartitionCut(PARTITION_CUT))
            .unwrap();
        assert_eq!(split.retained.len() + split.pruned.len(), full_len);
        assert!(split.retained.len() < full_len);
        assert!(
            (split.retained.total_weight() + split.pruned.total_weight() - full_weight).abs()
                <= 1e-12 * full_weight.abs().max(1.0)
        );
        assert!(
            (split.retained.total_quad_weight() + split.pruned.total_quad_weight() - full_quad)
                .abs()
                <= 1e-12 * full_quad.abs().max(1.0)
        );

        let mut in_place = grid;
        let pruned = in_place
            .split_off_pruned(PostPrune::PartitionCut(PARTITION_CUT))
            .unwrap();
        assert_eq!(in_place, split.retained);
        assert_eq!(pruned, split.pruned);
    }
}
