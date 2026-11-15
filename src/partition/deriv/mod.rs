//! Analytic first and second derivatives of the partition weights.
//!
//! This module is the scheme-independent streaming evaluator.  It owns
//!
//! ```text
//! PartitionDerivEvaluator
//!   ├─ fill(DerivPoints, request, output)
//!   ├─ fill_serial(DerivPoints, workspace, request, output)
//!   └─ fill_serial_visit(DerivPoints, workspace, request, output, visit)
//! ```
//!
//! [`DerivPoints`] selects a [`GridPlan`] range or a materialised
//! [`MolecularGrid`] range; both share the same point-local kernels. The
//! materialised-grid form accepts post-pruned and reordered grids, and a
//! point interval lets a large grid stream raw Hessians in bounded batches.
//! A single run produces any subset of the raw first/second derivatives
//! (`dw`, `ddw`) and their contractions (`dc`, `ddc`).  `dc` and `ddc` may
//! use different [`DerivContract`] tensors, including different `nset`, and a
//! contract tensor may describe a larger batch than the current range.
//!
//! # Streaming and memory
//!
//! No `natm x ngrids` object is ever created.  A gradient-only run evaluates
//! a range in bounded blocks (configurable, with eight-point lanes where
//! profitable) and bounded per-thread scratch.  A run that requests a Hessian
//! streams bounded chunks: the classic backend uses eight-point lanes when
//! profitable and otherwise one point at a time, writing the Hessian,
//! contracting it, or handing it to the visitor immediately.
//!
//! Per-point Hessian scratch is `O(natm^2)` (the `dlogP` matrix), which is the
//! unavoidable size of an exact Becke Hessian.  The raw `ddw` buffer is a
//! **batch** tensor `[nsel, 3, natm, 3, B]`, not a grid-sized one; use a small
//! range (or the visitor) for large molecules.  The contracted `ddc` is
//! `O(nset * nsel * natm^2)`.
//!
//! # Atom selection
//!
//! [`DerivOptions::atoms`] selects the **differentiated rows** `A` of
//! `dw[A,t]` / `ddw[A,t,B,s]`; every atom still enters as a centre `B` (and
//! as a pair partner).  The generating-atom translation correction needs
//! sums over all rows; a scheme accumulates those directly, so an unselected
//! row is never materialised.
//!
//! # Schemes
//!
//! [`PartitionDerivEvaluator::new`] installs the built-in backend selected by
//! the configuration:
//!
//! * classic Becke (`PartitionScheme::Becke`, `becke_early_exit == false`);
//! * Stratmann–Scuseria–Frisch (either `becke_early_exit` value);
//! * Gaussian 16 SS (`PartitionScheme::GaussianSs`, either flag);
//! * LKO (saturation-function geometry);
//! * mBecke with the default clamp when ORCA's clamp cannot fire
//!   (`max |a_AB| <= 0.5`), delegated to the classic Becke backend.
//!
//! The ORCA classic early exit and `mbecke_clamp < 1.0` return
//! [`Error::UnsupportedDerivative`].  Other algorithms implement
//! [`DerivScheme`] and use [`PartitionDerivEvaluator::with_scheme`]; the
//! streaming loop, selection, contractions and parallel reduction are shared.
//!
//! The classic Becke derivation is in
//! [Becke derivatives](crate::doc::becke_derivatives); the SSF, Gaussian SS,
//! LKO and mBecke formula/cost differences are in
//! [scheme variants](crate::doc::scheme_variants), and the common API
//! conventions are in the
//! [partition derivative API](crate::doc::partition_derivatives).

mod contract;
mod output;
mod scheme;
mod source;
mod workspace;

pub use contract::{DerivContract, GradientContract, HessianContract};
pub use output::{DerivOutput, DerivRequest};
pub use scheme::{DerivScheme, DerivScratch};
pub use workspace::DerivWorkspace;

use std::ops::Range;

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::grid::GridPlan;
use crate::molecule::Molecule;
use crate::moleculegrid::MolecularGrid;
use crate::partition::becke::deriv::BeckeDeriv;
use crate::partition::lko::deriv::LkoDeriv;
use crate::partition::stratmann::deriv::{GaussianSsDeriv, StratmannDeriv};
use crate::partition::{PartitionConfig, PartitionScheme};
use source::DerivSource;
use workspace::{ensure_block, ensure_point};

/// A contiguous chunk returned by the parallel gradient path.
type GradChunk = (
    usize,
    usize,
    Option<Vec<f64>>,
    Option<Vec<f64>>,
    Option<GradientContract>,
);

/// A contiguous chunk returned by the parallel per-point path.
type PointChunk = (
    usize,
    usize,
    Option<Vec<f64>>,
    Option<Vec<f64>>,
    Option<Vec<f64>>,
    Option<GradientContract>,
    Option<HessianContract>,
);

/// Optional raw-Hessian visitor.
type HessianVisitor<'v> = Option<&'v mut dyn FnMut(usize, &[f64])>;

/// Hessian kernel selection for the built-in Becke backend.
///
/// Future accelerator/backend selections may be added; downstream code must
/// include a wildcard arm.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum HessianBackend {
    /// Use the automatic size/ISA gate.
    #[default]
    Auto,
    /// Force the fixed-lane block Hessian, regardless of the automatic gate.
    Lane,
    /// Force the scalar pointwise Hessian, regardless of the automatic gate.
    Pointwise,
}

/// Options for [`PartitionDerivEvaluator`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DerivOptions {
    /// Differentiated atoms (rows). `None` means every atom.
    ///
    /// Duplicates are removed and the list is sorted, so the output slot
    /// order is deterministic.
    pub atoms: Option<Vec<usize>>,
    /// Gradient block length in grid points. `0` picks a bounded default
    /// from the scratch budget; a larger value trades memory for fewer loop
    /// setups.  The value is rounded up to a multiple of eight.
    pub block_points: usize,
    /// Temporary-memory budget per parallel worker, in bytes. `0` selects
    /// [`DEFAULT_WORKER_MEMORY_BYTES`] (128 MiB).
    ///
    /// The parallel paths use this to cap chunk-local output buffers and the
    /// number of concurrent Hessian worker scratches.  Gradient-only runs
    /// have a much smaller footprint and remain fully parallel.
    pub worker_memory_bytes: usize,
    /// Hessian kernel selection.  `Auto` applies the automatic gate for the
    /// compiled ISA; `Lane` and `Pointwise` are explicit testing/benchmark
    /// overrides for the built-in Becke backend.  Custom schemes supplied to
    /// [`PartitionDerivEvaluator::with_scheme`] make their own choice.
    pub hessian_backend: HessianBackend,
}

/// Default per-worker temporary-memory budget (128 MiB).
pub const DEFAULT_WORKER_MEMORY_BYTES: usize = 128 << 20;

/// Streaming analytic derivative evaluator for one geometry and one
/// selection.
///
/// Build once per geometry; the borrowing rules mirror
/// [`crate::partition::PartitionEvaluator`].
#[derive(Debug)]
pub struct PartitionDerivEvaluator {
    molecule: Molecule,
    config: PartitionConfig,
    scheme: Box<dyn DerivScheme>,
    sel: Vec<usize>,
    block: usize,
    worker_memory_bytes: usize,
}

/// Point source for a derivative evaluation.
///
/// A [`GridPlan`] range is evaluated from local coordinates and the plan's
/// parent atoms; a [`MolecularGrid`] range is evaluated from its global point
/// store, so post-pruned and reordered grids need no special handling.
///
/// The enum is closed: a point source is either a plan range or a
/// materialised-grid range.
#[derive(Debug, Clone)]
pub enum DerivPoints<'g> {
    /// A global range of a plan.
    Plan {
        /// Topology supplying local coordinates, parent atoms and weights.
        plan: &'g GridPlan,
        /// Global point interval `[start, end)`.
        range: Range<usize>,
    },
    /// A global range of a materialised grid.
    Grid {
        /// Grid supplying global coordinates, parent atoms and weights.
        grid: &'g MolecularGrid,
        /// Point interval `[start, end)`.
        range: Range<usize>,
    },
}

impl<'g> DerivPoints<'g> {
    /// A plan range.
    #[must_use]
    pub fn plan(plan: &'g GridPlan, range: Range<usize>) -> Self {
        Self::Plan { plan, range }
    }

    /// A grid range.
    #[must_use]
    pub fn grid(grid: &'g MolecularGrid, range: Range<usize>) -> Self {
        Self::Grid { grid, range }
    }

    /// The whole materialised grid.
    #[must_use]
    pub fn all_grid(grid: &'g MolecularGrid) -> Self {
        Self::Grid {
            grid,
            range: 0..grid.len(),
        }
    }

    /// Number of points in the selected range.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Plan { range, .. } | Self::Grid { range, .. } => range.len(),
        }
    }

    /// Whether the selected range is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The selected range.
    #[must_use]
    pub fn range(&self) -> Range<usize> {
        match self {
            Self::Plan { range, .. } | Self::Grid { range, .. } => range.clone(),
        }
    }
}

/// Largest caller-requested gradient block.
pub const MAX_BLOCK_POINTS: usize = 1 << 20;

impl PartitionDerivEvaluator {
    /// Build the derivative backend and validate the configuration.
    ///
    /// # Arguments
    ///
    /// * `molecule` — validated nuclear charges and positions in Bohr.
    /// * `config` — partition configuration; Becke (no early exit),
    ///   Stratmann, LKO and default-clamp mBecke have built-in derivative
    ///   backends (see the module documentation).
    /// * `options` — `[nsel]` row selection (or all atoms) and gradient
    ///   block size; see [`DerivOptions`].
    ///
    /// # Errors
    /// * [`Error::UnsupportedDerivative`] if no built-in backend exists
    ///   for `config`;
    /// * [`Error::InvalidAtom`] for an out-of-range selected atom;
    /// * [`Error::InvalidArgument`] for an oversized block or memory request.
    pub fn new(
        molecule: &Molecule,
        config: PartitionConfig,
        options: DerivOptions,
    ) -> Result<Self> {
        molecule.validate()?;
        config.validate()?;
        let centers = molecule.centers();
        let charges = molecule.charges();
        let n = molecule.len();
        let mut sel = match options.atoms {
            Some(atoms) => atoms,
            None => (0..n).collect(),
        };
        for &atom in &sel {
            if atom >= n {
                return Err(Error::InvalidAtom { atom, natm: n });
            }
        }
        sel.sort_unstable();
        sel.dedup();

        // mBecke with the default clamp is the classic pair product up to
        // round-off, so the classic Becke derivative is its derivative, but
        // only when ORCA's clamp and shortcut are inert and the clamp is not
        // `< 1.0` (see `partition::mbecke`); checked before the match (builds tables).
        let mbecke_is_classic = config.scheme == PartitionScheme::MBecke
            && crate::partition::mbecke_classic_product_is_valid(&config, centers, charges)?;
        let backend: Box<dyn DerivScheme> = match (config.scheme, config.becke_early_exit) {
            (PartitionScheme::Becke, false) => Box::new(BeckeDeriv::new(
                config.clone(),
                centers,
                charges,
                sel,
                options.hessian_backend,
            )?),
            // The SSF switch has exact zero/one factors, so the classic
            // ORCA early-exit truncation has no meaning for it; both values
            // of the flag select the same sparse SSF backend.
            (PartitionScheme::Stratmann, _) => {
                Box::new(StratmannDeriv::new(config.clone(), centers, charges, sel)?)
            }
            // Gaussian SS has the same compact-support product structure with
            // its own switch; the analytic kernel is shared.
            (PartitionScheme::GaussianSs, _) => {
                Box::new(GaussianSsDeriv::new(config.clone(), centers, charges, sel)?)
            }
            (PartitionScheme::Lko, _) => {
                Box::new(LkoDeriv::new(config.clone(), centers, charges, sel)?)
            }
            (PartitionScheme::MBecke, _) if mbecke_is_classic => Box::new(BeckeDeriv::new(
                config.clone(),
                centers,
                charges,
                sel,
                options.hessian_backend,
            )?),
            _ => {
                return Err(Error::UnsupportedDerivative {
                    scheme: config.scheme,
                    early_exit: config.becke_early_exit,
                });
            }
        };
        Self::with_scheme_memory(
            molecule,
            config,
            backend,
            options.worker_memory_bytes,
            options.block_points,
        )
    }

    /// Build from parallel center/charge slices.
    ///
    /// Convenience shim around [`Self::new`] for callers that do not keep a
    /// [`Molecule`] value.
    ///
    /// # Errors
    /// Same as [`Self::new`], plus [`Error::LengthMismatch`] if the slices
    /// differ in length.
    pub fn from_parts(
        centers: &[[f64; 3]],
        charges: &[u32],
        config: PartitionConfig,
        options: DerivOptions,
    ) -> Result<Self> {
        if centers.len() != charges.len() {
            return Err(Error::LengthMismatch {
                expected: centers.len(),
                got: charges.len(),
            });
        }
        let molecule = Molecule::new(charges.iter().copied().zip(centers.iter().copied()))?;
        Self::new(&molecule, config, options)
    }

    /// Build an evaluator from a caller-provided [`DerivScheme`] backend.
    ///
    /// This is the extension point for future partition algorithms.  The
    /// backend owns its molecule-only tables and its selection; the evaluator
    /// only needs the molecule (to translate the local grid coordinate) and
    /// the partition config (for [`Self::config`]).
    ///
    /// # Arguments
    ///
    /// * `molecule` — validated nuclear charges and positions in Bohr.
    /// * `config` — partition configuration reported by [`Self::config`].
    /// * `scheme` — `[natm]`-atom backend with a strictly increasing
    ///   `[nsel]` selection.
    /// * `block_points` — gradient block length, `0` for the automatic value.
    ///
    /// # Errors
    /// [`Error::LengthMismatch`] if the backend's atom count differs from
    /// `molecule.len()`, [`Error::InvalidAtom`] for an out-of-range
    /// selection, and [`Error::InvalidSelectionOrder`] if the selection is
    /// not strictly increasing.
    pub fn with_scheme(
        molecule: &Molecule,
        config: PartitionConfig,
        scheme: Box<dyn DerivScheme>,
        block_points: usize,
    ) -> Result<Self> {
        Self::with_scheme_memory(molecule, config, scheme, 0, block_points)
    }

    /// Like [`Self::with_scheme`] with an explicit per-worker memory budget.
    ///
    /// * `worker_memory_bytes` — `0` selects
    ///   [`DEFAULT_WORKER_MEMORY_BYTES`]; otherwise the parallel chunk and
    ///   block sizes are capped by this budget.
    /// * `block_points` — `0` selects the automatic block; a positive value
    ///   is rounded up to a multiple of eight and capped at
    ///   [`MAX_BLOCK_POINTS`].
    pub fn with_scheme_memory(
        molecule: &Molecule,
        config: PartitionConfig,
        scheme: Box<dyn DerivScheme>,
        worker_memory_bytes: usize,
        block_points: usize,
    ) -> Result<Self> {
        molecule.validate()?;
        let n = molecule.len();
        if scheme.atom_count() != n {
            return Err(Error::LengthMismatch {
                expected: n,
                got: scheme.atom_count(),
            });
        }
        let sel = scheme.selection().to_vec();
        for (i, &atom) in sel.iter().enumerate() {
            if atom >= n {
                return Err(Error::InvalidAtom { atom, natm: n });
            }
            if i > 0 && atom <= sel[i - 1] {
                return Err(Error::InvalidSelectionOrder { atom });
            }
        }
        let nsel = sel.len();
        let block = if block_points == 0 {
            auto_gradient_block(n, nsel)
        } else {
            let rounded = block_points
                .max(1)
                .checked_add(7)
                .and_then(|value| (value / 8).checked_mul(8))
                .ok_or(Error::InvalidArgument {
                    argument: "block_points",
                })?;
            if rounded > MAX_BLOCK_POINTS {
                return Err(Error::InvalidArgument {
                    argument: "block_points",
                });
            }
            rounded
        };
        Ok(Self {
            molecule: molecule.clone(),
            config,
            scheme,
            sel,
            block,
            worker_memory_bytes: if worker_memory_bytes == 0 {
                DEFAULT_WORKER_MEMORY_BYTES
            } else {
                worker_memory_bytes
            },
        })
    }

    /// Build from parallel center/charge slices with a custom backend.
    ///
    /// # Errors
    /// Same as [`Self::with_scheme`], plus [`Error::LengthMismatch`] if the
    /// slices differ in length.
    pub fn with_scheme_from_parts(
        centers: &[[f64; 3]],
        charges: &[u32],
        config: PartitionConfig,
        scheme: Box<dyn DerivScheme>,
        block_points: usize,
    ) -> Result<Self> {
        if centers.len() != charges.len() {
            return Err(Error::LengthMismatch {
                expected: centers.len(),
                got: charges.len(),
            });
        }
        let molecule = Molecule::new(charges.iter().copied().zip(centers.iter().copied()))?;
        Self::with_scheme(&molecule, config, scheme, block_points)
    }

    /// Build from parallel center/charge slices with a custom backend and
    /// memory budget.
    ///
    /// # Errors
    /// Same as [`Self::with_scheme_memory`].
    pub fn with_scheme_memory_from_parts(
        centers: &[[f64; 3]],
        charges: &[u32],
        config: PartitionConfig,
        scheme: Box<dyn DerivScheme>,
        worker_memory_bytes: usize,
        block_points: usize,
    ) -> Result<Self> {
        if centers.len() != charges.len() {
            return Err(Error::LengthMismatch {
                expected: centers.len(),
                got: charges.len(),
            });
        }
        let molecule = Molecule::new(charges.iter().copied().zip(centers.iter().copied()))?;
        Self::with_scheme_memory(&molecule, config, scheme, worker_memory_bytes, block_points)
    }

    /// The molecule this evaluator was built for.
    #[must_use]
    pub fn molecule(&self) -> &Molecule {
        &self.molecule
    }

    /// The partition configuration this evaluator was built for.
    #[must_use]
    pub fn config(&self) -> &PartitionConfig {
        &self.config
    }

    /// The installed scheme backend.
    #[must_use]
    pub fn scheme(&self) -> &dyn DerivScheme {
        self.scheme.as_ref()
    }

    /// Selected (differentiated) atoms, sorted.
    #[must_use]
    pub fn atoms(&self) -> &[usize] {
        &self.sel
    }

    /// Number of atoms in the evaluated geometry.
    #[must_use]
    pub fn atom_count(&self) -> usize {
        self.scheme.atom_count()
    }

    /// Gradient block length in points.
    #[must_use]
    pub fn gradient_block(&self) -> usize {
        self.block
    }

    /// Per-worker temporary-memory budget in bytes.
    #[must_use]
    pub fn worker_memory_bytes(&self) -> usize {
        self.worker_memory_bytes
    }

    fn check_range(&self, len: usize, range: &Range<usize>) -> Result<()> {
        if range.start > range.end || range.end > len {
            return Err(Error::RangeOutOfBounds {
                start: range.start,
                end: range.end,
                len,
            });
        }
        Ok(())
    }

    /// Validate a materialised grid before evaluating it.
    ///
    /// Checks the per-point array lengths, the atom indices, and that the
    /// grid's molecule matches the evaluator's.  The evaluator's partition
    /// configuration is authoritative; the grid's spec partition is not
    /// compared, so build the evaluator through
    /// [`MolecularGrid::partition_deriv_evaluator`] to evaluate the grid's own
    /// configuration.
    fn check_grid(&self, grid: &MolecularGrid) -> Result<()> {
        grid.validate()?;
        if grid.molecule() != &self.molecule {
            if grid.atom_coords() != self.molecule.centers() {
                return Err(Error::GeometryMismatch);
            }
            for (atom, (&expected, &got)) in self
                .molecule
                .charges()
                .iter()
                .zip(grid.atom_charges())
                .enumerate()
            {
                if expected != got {
                    return Err(Error::ChargesMismatch {
                        atom,
                        expected,
                        got,
                    });
                }
            }
        }
        Ok(())
    }

    /// Weight and selected derivatives of one point (quadrature weight 1).
    ///
    /// # Arguments
    ///
    /// * `point` — `[3]` Cartesian grid coordinate in Bohr.
    /// * `parent` — generating atom of the point (`A_g`).
    ///
    /// # Returns
    ///
    /// `(Pbar / Z, dw)` where `dw` is row-major `[nsel, 3]`, flattened as
    /// `slot * 3 + t`.  No quadrature factor is included.
    ///
    /// This convenience call allocates a fresh workspace and output vector;
    /// use [`Self::fill_serial_visit`] for a streaming evaluation.
    ///
    /// # Panics
    /// If `parent >= self.atom_count()`.
    #[must_use]
    pub fn gradient_at(&self, point: [f64; 3], parent: usize) -> (f64, Vec<f64>) {
        assert!(parent < self.atom_count(), "parent atom out of range");
        let nsel = self.sel.len();
        let mut ws = DerivWorkspace::new();
        let mut w = [0.0];
        let mut dw = vec![0.0; nsel * 3];
        let scratch = ws.block_scratch(self.scheme.as_ref(), 1);
        self.scheme
            .gradient_block(&[point], &[parent as u32], &[1.0], &mut w, &mut dw, scratch);
        (w[0], dw)
    }

    /// Corrected Hessian of one point (quadrature weight 1).
    ///
    /// # Arguments
    ///
    /// * `point` — `[3]` Cartesian grid coordinate in Bohr.
    /// * `parent` — generating atom of the point (`A_g`).
    ///
    /// # Returns
    ///
    /// Row-major `[nsel, 3, natm, 3]`, flattened as
    /// `((slot * 3 + t) * natm + B) * 3 + s`; no quadrature factor.
    ///
    /// This convenience call allocates a fresh workspace and output vector;
    /// use [`Self::fill_serial_visit`] for a streaming evaluation.
    ///
    /// # Panics
    /// If `parent >= self.atom_count()`.
    #[must_use]
    pub fn hessian_at(&self, point: [f64; 3], parent: usize) -> Vec<f64> {
        assert!(parent < self.atom_count(), "parent atom out of range");
        let mut ws = DerivWorkspace::new();
        let (scratch, _grad) = ws.point_parts(self.scheme.as_ref());
        let (_weight, hess) = self.scheme.point_hessian(point, parent, 1.0, None, scratch);
        hess.to_vec()
    }

    /// Evaluate a plan range or a materialised grid in parallel.
    ///
    /// This is the single entry point for both point sources. A run that
    /// requests `ddw` or `ddc` evaluates the Hessian kernel, which also emits
    /// `dw`/`dc`; a gradient-only run uses the faster blocked kernel.
    ///
    /// A [`DerivContract`]'s `offset` is the column of the first evaluated
    /// point.  When streaming a sub-range of a full-batch tensor, set
    /// `offset = range.start` (for example with [`DerivContract::over`]).
    ///
    /// # Errors
    /// [`Error::RangeOutOfBounds`] for a reversed or out-of-range interval,
    /// [`Error::LengthMismatch`] / [`Error::ChargesMismatch`] if the plan does
    /// not match the evaluator's molecule, and the grid errors of
    /// [`Self::fill`] for a [`DerivPoints::Grid`] source.
    pub fn fill(
        &self,
        points: DerivPoints<'_>,
        req: &DerivRequest<'_>,
        out: &mut DerivOutput,
    ) -> Result<()> {
        self.check_points(&points)?;
        let range = points.range();
        out.prepare(&self.sel, self.scheme.atom_count(), &range, req)?;
        match points {
            DerivPoints::Plan { plan, .. } => {
                let source = DerivSource::Plan {
                    plan,
                    centers: self.molecule.centers(),
                };
                self.fill_parallel(source, range, req, out);
            }
            DerivPoints::Grid { grid, .. } => {
                self.fill_parallel(DerivSource::Grid(grid), range, req, out);
            }
        }
        Ok(())
    }

    /// Serial form of [`Self::fill`], reusing one [`DerivWorkspace`].
    ///
    /// # Errors
    /// Same as [`Self::fill`].
    pub fn fill_serial(
        &self,
        points: DerivPoints<'_>,
        ws: &mut DerivWorkspace,
        req: &DerivRequest<'_>,
        out: &mut DerivOutput,
    ) -> Result<()> {
        self.check_points(&points)?;
        let range = points.range();
        out.prepare(&self.sel, self.scheme.atom_count(), &range, req)?;
        match points {
            DerivPoints::Plan { plan, .. } => {
                let source = DerivSource::Plan {
                    plan,
                    centers: self.molecule.centers(),
                };
                if req.ddw || req.ddc.is_some() {
                    self.fill_point_serial(source, range, ws, req, out, None);
                } else {
                    self.fill_gradient_serial(source, range, ws, req, out);
                }
            }
            DerivPoints::Grid { grid, .. } => {
                let source = DerivSource::Grid(grid);
                if req.ddw || req.ddc.is_some() {
                    self.fill_point_serial(source, range, ws, req, out, None);
                } else {
                    self.fill_gradient_serial(source, range, ws, req, out);
                }
            }
        }
        Ok(())
    }

    /// Serial evaluation that hands one point's full corrected Hessian to
    /// `visit` instead of allocating [`DerivOutput::ddw`].
    ///
    /// `visit` receives `(global_point_index, block)` with `block` row-major
    /// `[nsel, 3, natm, 3]`, flattened
    /// `((slot * 3 + t) * natm + B) * 3 + s`, the Hessian of the total grid
    /// weight (quadrature included). The slice is valid only during the call.
    /// Contracted outputs (`dc`, `ddc`) and raw `dw` are still produced.
    ///
    /// # Errors
    /// Same as [`Self::fill`].
    pub fn fill_serial_visit<F>(
        &self,
        points: DerivPoints<'_>,
        ws: &mut DerivWorkspace,
        req: &DerivRequest<'_>,
        out: &mut DerivOutput,
        visit: F,
    ) -> Result<()>
    where
        F: FnMut(usize, &[f64]),
    {
        self.check_points(&points)?;
        let range = points.range();
        out.prepare(&self.sel, self.scheme.atom_count(), &range, req)?;
        let mut visit = visit;
        match points {
            DerivPoints::Plan { plan, .. } => {
                let source = DerivSource::Plan {
                    plan,
                    centers: self.molecule.centers(),
                };
                self.fill_point_serial(source, range, ws, req, out, Some(&mut visit));
            }
            DerivPoints::Grid { grid, .. } => {
                self.fill_point_serial(
                    DerivSource::Grid(grid),
                    range,
                    ws,
                    req,
                    out,
                    Some(&mut visit),
                );
            }
        }
        Ok(())
    }

    fn fill_parallel(
        &self,
        source: DerivSource<'_>,
        range: Range<usize>,
        req: &DerivRequest<'_>,
        out: &mut DerivOutput,
    ) {
        if req.ddw || req.ddc.is_some() {
            self.fill_point_par(source, range, req, out);
        } else {
            self.fill_gradient_par(source, range, req, out);
        }
    }

    fn check_points(&self, points: &DerivPoints<'_>) -> Result<()> {
        match points {
            DerivPoints::Plan { plan, range } => {
                self.check_range(plan.len(), range)?;
                if plan.atom_count() != self.molecule.len() {
                    return Err(Error::LengthMismatch {
                        expected: self.molecule.len(),
                        got: plan.atom_count(),
                    });
                }
                for (ia, &z) in self.molecule.charges().iter().enumerate() {
                    let got = plan.atom_grid(ia).z;
                    if got != z {
                        return Err(Error::ChargesMismatch {
                            atom: ia,
                            expected: z,
                            got,
                        });
                    }
                }
                Ok(())
            }
            DerivPoints::Grid { grid, range } => {
                self.check_grid(grid)?;
                self.check_range(grid.len(), range)
            }
        }
    }

    fn fill_gradient_serial(
        &self,
        source: DerivSource<'_>,
        range: Range<usize>,
        ws: &mut DerivWorkspace,
        req: &DerivRequest<'_>,
        out: &mut DerivOutput,
    ) {
        let DerivOutput {
            weights, dw, dc, ..
        } = out;
        let weights_out = req.weights.then_some(weights.as_mut_slice());
        let dw_out = req.dw.then_some(dw.as_mut_slice());
        self.eval_gradient_chunk(source, range, ws, req, 0, weights_out, dw_out, dc.as_mut());
    }

    fn fill_gradient_par(
        &self,
        source: DerivSource<'_>,
        range: Range<usize>,
        req: &DerivRequest<'_>,
        out: &mut DerivOutput,
    ) {
        let len = range.len();
        if len == 0 {
            return;
        }
        let nsel = self.sel.len();
        // Two chunks per worker give rayon a tail to steal without making the
        // per-chunk `dw` buffer large.  If a chunk-local `dw` would exceed the
        // worker budget, split further instead.
        let threads = rayon::current_num_threads().max(1);
        let per_point = 8 * (usize::from(req.weights) + if req.dw { nsel * 3 } else { 0 });
        let n_by_memory = if per_point == 0 {
            1
        } else {
            len.saturating_mul(per_point)
                .div_ceil(self.worker_memory_bytes.max(1))
                .max(1)
        };
        let nchunks = (2 * threads)
            .max(n_by_memory)
            .min(len.div_ceil(8).max(1))
            .min(len);
        let chunk = len.div_ceil(nchunks).max(1);
        let parts: Vec<GradChunk> = (0..nchunks)
            .into_par_iter()
            .map_init(DerivWorkspace::new, |ws, k| {
                let start = range.start + k * chunk;
                if start >= range.end {
                    return (start, start, None, None, None);
                }
                let end = (start + chunk).min(range.end);
                let sub = start..end;
                let mut weights = req.weights.then(|| vec![0.0; sub.len()]);
                let mut dw = req.dw.then(|| vec![0.0; nsel * 3 * sub.len()]);
                let mut dc = req.dc.map(|c| GradientContract::new(&self.sel, c.nset()));
                self.eval_gradient_chunk(
                    source,
                    sub,
                    ws,
                    req,
                    start - range.start,
                    weights.as_deref_mut(),
                    dw.as_deref_mut(),
                    dc.as_mut(),
                );
                (start, end, weights, dw, dc)
            })
            .collect();

        let DerivOutput {
            weights, dw, dc, ..
        } = out;
        let range_len = range.len();
        for (start, end, w, _d, dc_local) in &parts {
            let off = *start - range.start;
            let len_i = end.saturating_sub(*start);
            if len_i == 0 {
                continue;
            }
            if let Some(w) = w {
                weights[off..off + len_i].copy_from_slice(w);
            }
            if let (Some(dst), Some(src)) = (dc.as_mut(), dc_local.as_ref()) {
                dst.add(src);
            }
        }
        if dw.len() >= (1 << 18) {
            // Assembling the strided `[row, range]` output is a pure copy;
            // spread it over rows so the serial assembly does not dominate a
            // short gradient evaluation.
            dw.par_chunks_mut(range_len)
                .enumerate()
                .for_each(|(row, dst)| {
                    for (start, end, _, d, _) in &parts {
                        if let Some(d) = d {
                            let off = *start - range.start;
                            let len_i = end.saturating_sub(*start);
                            if len_i > 0 {
                                dst[off..off + len_i]
                                    .copy_from_slice(&d[row * len_i..(row + 1) * len_i]);
                            }
                        }
                    }
                });
        } else {
            for (start, end, _, d, _) in &parts {
                if let Some(d) = d {
                    let off = *start - range.start;
                    let len_i = end.saturating_sub(*start);
                    for row in 0..nsel * 3 {
                        let src = &d[row * len_i..(row + 1) * len_i];
                        let dst = row * range_len + off;
                        dw[dst..dst + len_i].copy_from_slice(src);
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_gradient_chunk(
        &self,
        source: DerivSource<'_>,
        range: Range<usize>,
        ws: &mut DerivWorkspace,
        req: &DerivRequest<'_>,
        column_base: usize,
        mut weights_out: Option<&mut [f64]>,
        mut dw_out: Option<&mut [f64]>,
        mut dc_out: Option<&mut GradientContract>,
    ) {
        let nsel = self.sel.len();
        let block = self.block;
        let range_len = range.len();
        if let Some(weights) = &weights_out {
            debug_assert_eq!(weights.len(), range_len);
        }
        if let Some(dw) = &dw_out {
            debug_assert_eq!(dw.len(), nsel * 3 * range_len);
        }

        let DerivWorkspace {
            block_scratch,
            coords,
            parents,
            quad,
            weights_block,
            dw_block,
            ..
        } = ws;
        weights_block.resize(block, 0.0);
        dw_block.resize(nsel * 3 * block, 0.0);

        let mut start = range.start;
        while start < range.end {
            let end = (start + block).min(range.end);
            let nb = end - start;
            let off = start - range.start;

            coords.clear();
            parents.clear();
            quad.clear();
            coords.reserve(nb);
            parents.reserve(nb);
            quad.reserve(nb);
            for g in start..end {
                let (coord, parent, q) = source.point(g);
                coords.push(coord);
                parents.push(parent);
                quad.push(q);
            }

            let weights_dst = match weights_out.as_deref_mut() {
                Some(w) => &mut w[off..off + nb],
                None => &mut weights_block[..nb],
            };
            let scratch = ensure_block(block_scratch, self.scheme.as_ref(), block);
            self.scheme
                .gradient_block(coords, parents, quad, weights_dst, dw_block, scratch);

            if let Some(dw_out) = dw_out.as_deref_mut() {
                for row in 0..nsel * 3 {
                    let src = &dw_block[row * block..row * block + nb];
                    let dst = row * range_len + off;
                    dw_out[dst..dst + nb].copy_from_slice(src);
                }
            }
            if let Some(dc) = dc_out.as_deref_mut() {
                // `DerivOutput::prepare` installs `out.dc` only when `req.dc`
                // is `Some`, so the contract is present by construction.
                let contract = req
                    .dc
                    .as_ref()
                    .expect("dc output requested without a contract");
                dc.add_strided(
                    contract,
                    contract.offset() + column_base + off,
                    dw_block,
                    block,
                    nb,
                );
            }
            start = end;
        }
    }

    fn fill_point_serial(
        &self,
        source: DerivSource<'_>,
        range: Range<usize>,
        ws: &mut DerivWorkspace,
        req: &DerivRequest<'_>,
        out: &mut DerivOutput,
        visitor: HessianVisitor<'_>,
    ) {
        let DerivOutput {
            weights,
            dw,
            ddw,
            dc,
            ddc,
            ..
        } = out;
        let weights_out = req.weights.then_some(weights.as_mut_slice());
        let dw_out = req.dw.then_some(dw.as_mut_slice());
        let ddw_out = req.ddw.then_some(ddw.as_mut_slice());
        if self.scheme.hessian_pointwise() {
            self.eval_pointwise_chunk(
                source,
                range,
                ws,
                req,
                0,
                weights_out,
                dw_out,
                ddw_out,
                dc.as_mut(),
                ddc.as_mut(),
                visitor,
            );
        } else {
            self.eval_point_chunk(
                source,
                range,
                ws,
                req,
                0,
                weights_out,
                dw_out,
                ddw_out,
                dc.as_mut(),
                ddc.as_mut(),
                visitor,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_point_chunk(
        &self,
        source: DerivSource<'_>,
        range: Range<usize>,
        ws: &mut DerivWorkspace,
        req: &DerivRequest<'_>,
        column_base: usize,
        mut weights_out: Option<&mut [f64]>,
        mut dw_out: Option<&mut [f64]>,
        mut ddw_out: Option<&mut [f64]>,
        mut dc_out: Option<&mut GradientContract>,
        mut ddc_out: Option<&mut HessianContract>,
        mut visitor: HessianVisitor<'_>,
    ) {
        let nsel = self.sel.len();
        let n3 = 3 * self.scheme.atom_count();
        let row_len = nsel * 3;
        let hess_len = row_len * n3;
        let range_len = range.len();
        let want_grad = dw_out.is_some() || dc_out.is_some();
        let DerivWorkspace {
            point_scratch,
            coords,
            parents,
            quad,
            weights_block,
            grad_block,
            ddw_block,
            ..
        } = ws;
        grad_block.resize(row_len * crate::partition::simd::LANES, 0.0);
        ddw_block.resize(hess_len * crate::partition::simd::LANES, 0.0);

        let mut start = range.start;
        while start < range.end {
            let end = (start + crate::partition::simd::LANES).min(range.end);
            let b = end - start;
            coords.clear();
            parents.clear();
            quad.clear();
            coords.reserve(b);
            parents.reserve(b);
            quad.reserve(b);
            for g in start..end {
                let (coord, parent, q) = source.point(g);
                coords.push(coord);
                parents.push(parent);
                quad.push(q);
            }

            weights_block.resize(b, 0.0);
            let scratch = ensure_point(point_scratch, self.scheme.as_ref());
            let grad = want_grad.then_some(&mut grad_block[..row_len * b]);
            self.scheme.hessian_block(
                coords,
                parents,
                quad,
                weights_block,
                grad,
                &mut ddw_block[..hess_len * b],
                scratch,
            );

            let off = start - range.start;
            if let Some(weights) = weights_out.as_deref_mut() {
                weights[off..off + b].copy_from_slice(&weights_block[..b]);
            }
            if let Some(dw) = dw_out.as_deref_mut() {
                for row in 0..row_len {
                    for l in 0..b {
                        dw[row * range_len + off + l] = grad_block[l * row_len + row];
                    }
                }
            }
            if let Some(ddw) = ddw_out.as_deref_mut() {
                for c in 0..hess_len {
                    for l in 0..b {
                        ddw[c * range_len + off + l] = ddw_block[l * hess_len + c];
                    }
                }
            }
            if let Some(dc) = dc_out.as_deref_mut() {
                // `DerivOutput::prepare` installs `out.dc` only when `req.dc`
                // is `Some`, so the contract is present by construction.
                let contract = req
                    .dc
                    .as_ref()
                    .expect("dc output requested without a contract");
                for l in 0..b {
                    let point = l * row_len;
                    dc.add_point(
                        contract,
                        contract.offset() + column_base + off + l,
                        &grad_block[point..point + row_len],
                    );
                }
            }
            if let Some(ddc) = ddc_out.as_deref_mut() {
                let contract = req
                    .ddc
                    .as_ref()
                    .expect("ddc output requested without a contract");
                for l in 0..b {
                    let point = l * hess_len;
                    ddc.add_point(
                        contract,
                        contract.offset() + column_base + off + l,
                        &ddw_block[point..point + hess_len],
                    );
                }
            }
            if let Some(visit) = visitor.as_mut() {
                for l in 0..b {
                    let point = l * hess_len;
                    (**visit)(start + l, &ddw_block[point..point + hess_len]);
                }
            }
            start = end;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_pointwise_chunk(
        &self,
        source: DerivSource<'_>,
        range: Range<usize>,
        ws: &mut DerivWorkspace,
        req: &DerivRequest<'_>,
        column_base: usize,
        mut weights_out: Option<&mut [f64]>,
        mut dw_out: Option<&mut [f64]>,
        mut ddw_out: Option<&mut [f64]>,
        mut dc_out: Option<&mut GradientContract>,
        mut ddc_out: Option<&mut HessianContract>,
        mut visitor: HessianVisitor<'_>,
    ) {
        let nsel = self.sel.len();
        let range_len = range.len();
        let want_grad = dw_out.is_some() || dc_out.is_some();
        let DerivWorkspace {
            point_scratch,
            grad_point,
            ..
        } = ws;
        if grad_point.len() != nsel * 3 {
            grad_point.resize(nsel * 3, 0.0);
        }

        for (idx, g) in range.enumerate() {
            let (point, parent, quad) = source.point(g);
            let scratch = ensure_point(point_scratch, self.scheme.as_ref());
            let grad = want_grad.then_some(grad_point.as_mut_slice());
            let (weight, hess) =
                self.scheme
                    .point_hessian(point, parent as usize, quad, grad, scratch);

            if let Some(weights) = weights_out.as_deref_mut() {
                weights[idx] = weight;
            }
            if let Some(dw) = dw_out.as_deref_mut() {
                for (row, &value) in grad_point.iter().enumerate() {
                    dw[row * range_len + idx] = value;
                }
            }
            if let Some(ddw) = ddw_out.as_deref_mut() {
                for (c, &value) in hess.iter().enumerate() {
                    ddw[c * range_len + idx] = value;
                }
            }
            if let Some(dc) = dc_out.as_deref_mut() {
                // `DerivOutput::prepare` installs `out.dc` only when `req.dc`
                // is `Some`, so the contract is present by construction.
                let contract = req
                    .dc
                    .as_ref()
                    .expect("dc output requested without a contract");
                dc.add_point(contract, contract.offset() + column_base + idx, grad_point);
            }
            if let Some(ddc) = ddc_out.as_deref_mut() {
                let contract = req
                    .ddc
                    .as_ref()
                    .expect("ddc output requested without a contract");
                ddc.add_point(contract, contract.offset() + column_base + idx, hess);
            }
            if let Some(visit) = visitor.as_mut() {
                (**visit)(g, hess);
            }
        }
    }

    fn fill_point_par(
        &self,
        source: DerivSource<'_>,
        range: Range<usize>,
        req: &DerivRequest<'_>,
        out: &mut DerivOutput,
    ) {
        let len = range.len();
        if len == 0 {
            return;
        }
        let n = self.scheme.atom_count();
        let n3 = 3 * n;
        let nsel = self.sel.len();
        let threads = rayon::current_num_threads().max(1);
        let nset_dc = req.dc.map_or(0, |c| c.nset());
        let nset_ddc = req.ddc.map_or(0, |c| c.nset());
        let scratch_bytes = self.scheme.hessian_scratch_bytes();
        let contract_bytes = 8 * (nset_ddc * nsel * 3 * n3 + nset_dc * nsel * 3);
        // Workspace block buffers are always allocated for the eight-point
        // Hessian stream; the per-point raw buffers are chunk-local.
        let pointwise = self.scheme.hessian_pointwise();
        let block_bytes = if pointwise {
            0
        } else {
            8 * (nsel * 3 * 8 + nsel * 3 * n3 * 8)
        };
        let per_point_local = 8
            * (usize::from(req.weights)
                + if req.dw { nsel * 3 } else { 0 }
                + if req.ddw { nsel * 3 * n3 } else { 0 });
        let base_bytes = scratch_bytes + block_bytes + contract_bytes;
        let mut chunk = len.div_ceil(threads).clamp(1, 32);
        while chunk > 1
            && base_bytes.saturating_add(chunk * per_point_local) > self.worker_memory_bytes
        {
            chunk = chunk.div_ceil(2);
        }
        let mut nchunks = len.div_ceil(chunk).max(1);
        if !req.ddw && (req.dc.is_some() || req.ddc.is_some()) {
            // Every chunk keeps a private contraction until the reduction.
            // Without raw `ddw`, cap the number of live contractions instead
            // of allocating one per 32-point chunk.  Sixteen chunks per thread
            // balances the workers; the worker-memory budget bounds the total
            // when a single contraction is large.
            let budget_chunks =
                self.worker_memory_bytes.saturating_mul(threads) / contract_bytes.max(1);
            let max_chunks = (16 * threads).max(1).min(budget_chunks.max(1));
            if nchunks > max_chunks {
                nchunks = max_chunks;
                chunk = len.div_ceil(nchunks);
            }
        }

        let parts: Vec<PointChunk> = (0..nchunks)
            .into_par_iter()
            .map_init(DerivWorkspace::new, |ws, k| {
                let start = range.start + k * chunk;
                if start >= range.end {
                    return (start, start, None, None, None, None, None);
                }
                let end = (start + chunk).min(range.end);
                let sub = start..end;
                let sub_len = sub.len();
                let mut weights = req.weights.then(|| vec![0.0; sub_len]);
                let mut dw = req.dw.then(|| vec![0.0; nsel * 3 * sub_len]);
                let mut ddw = req.ddw.then(|| vec![0.0; nsel * 3 * n3 * sub_len]);
                let mut dc = req.dc.map(|c| GradientContract::new(&self.sel, c.nset()));
                let mut ddc = req
                    .ddc
                    .map(|c| HessianContract::new(&self.sel, c.nset(), n));
                if pointwise {
                    self.eval_pointwise_chunk(
                        source,
                        sub,
                        ws,
                        req,
                        start - range.start,
                        weights.as_deref_mut(),
                        dw.as_deref_mut(),
                        ddw.as_deref_mut(),
                        dc.as_mut(),
                        ddc.as_mut(),
                        None,
                    );
                } else {
                    self.eval_point_chunk(
                        source,
                        sub,
                        ws,
                        req,
                        start - range.start,
                        weights.as_deref_mut(),
                        dw.as_deref_mut(),
                        ddw.as_deref_mut(),
                        dc.as_mut(),
                        ddc.as_mut(),
                        None,
                    );
                }
                (start, end, weights, dw, ddw, dc, ddc)
            })
            .collect();

        let DerivOutput {
            weights,
            dw,
            ddw,
            dc,
            ddc,
            ..
        } = out;
        let range_len = range.len();
        for (start, end, w, d, ddw_local, dc_local, ddc_local) in parts {
            let off = start - range.start;
            let len_i = end.saturating_sub(start);
            if len_i == 0 {
                continue;
            }
            if let Some(w) = w {
                weights[off..off + len_i].copy_from_slice(&w);
            }
            if let Some(d) = d {
                for row in 0..nsel * 3 {
                    let src = &d[row * len_i..(row + 1) * len_i];
                    let dst = row * range_len + off;
                    dw[dst..dst + len_i].copy_from_slice(src);
                }
            }
            if let Some(ddw_local) = ddw_local {
                for c in 0..nsel * 3 * n3 {
                    let src = &ddw_local[c * len_i..(c + 1) * len_i];
                    let dst = c * range_len + off;
                    ddw[dst..dst + len_i].copy_from_slice(src);
                }
            }
            if let (Some(dst), Some(src)) = (dc.as_mut(), dc_local.as_ref()) {
                dst.add(src);
            }
            if let (Some(dst), Some(src)) = (ddc.as_mut(), ddc_local.as_ref()) {
                dst.add(src);
            }
        }
    }
}

fn auto_gradient_block(n: usize, nsel: usize) -> usize {
    // dist + prod + drdist = 5n doubles per point; dz + dp = 6 nsel.
    // One fixed lane consumes 8 * (5n + 6nsel) doubles, so a 2048-element
    // (16 KB) target is conservative for the cache sizes of common x86-64
    // CPUs.  Large molecules necessarily exceed it at the eight-lane minimum.
    let per_point = 5 * n + 6 * nsel;
    let lanes = ((1 << 11) / (per_point * crate::partition::simd::LANES).max(1)).max(1);
    (lanes * crate::partition::simd::LANES).clamp(crate::partition::simd::LANES, 512)
}

impl MolecularGrid {
    /// Build a partition-derivative evaluator from this grid.
    ///
    /// The evaluator owns a clone of the grid's molecule and takes the
    /// partition configuration from [`MolecularGrid::spec`]. Evaluate the
    /// grid with [`PartitionDerivEvaluator::fill`] and
    /// [`DerivPoints::all_grid`] (all points) or [`DerivPoints::grid`] (a
    /// point interval); use [`PartitionDerivEvaluator::fill_serial_visit`]
    /// for a raw-Hessian visitor. The parallel fills use the ambient Rayon
    /// thread pool; a post-pruned or reordered grid is evaluated in its
    /// current point order.
    ///
    /// # Errors
    /// The same errors as [`PartitionDerivEvaluator::new`], in particular
    /// [`Error::UnsupportedDerivative`] for a configuration without a
    /// built-in derivative backend.
    pub fn partition_deriv_evaluator(
        &self,
        options: DerivOptions,
    ) -> Result<PartitionDerivEvaluator> {
        PartitionDerivEvaluator::new(self.molecule(), self.spec().partition().clone(), options)
    }
}
