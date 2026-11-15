//! Partition weights (classic Becke, SSF, Gaussian SS, mBecke, LKO) and
//! range-based evaluation.
//!
//! All schemes share one execution path.  A [`PartitionConfig`] describes the
//! partition completely — the cell-function variant ([`PartitionScheme`]) and
//! the radii handling ([`RadiiConfig`]) — and ORCA is just a configuration
//! built from the ORCA radii and the early-exit flag; there is no separate
//! ORCA numeric path.
//!
//! Execution is split into three stages:
//!
//! 1. `PartitionTables::new` precomputes everything that depends only on
//!    the molecule (per-atom radii, pair adjustment factors $a_{AB}$, and
//!    the interatomic distances $R_{AB}$) once;
//! 2. `PartitionConfig::center_weights_par` evaluates a whole atomic grid
//!    in blocks with rayon, reusing one distance pass per point and bounded
//!    per-worker scratch (the private `kernel` module);
//! 3. the evaluator's point and range entry points share the same tables.
//!
//! Within one dispatch, the classic Becke, compact-support SSF/Gaussian SS
//! and LKO blocked and per-point kernels agree bit-for-bit. The symmetric
//! SSF dense and sparse kernels agree to round-off (they use opposite
//! size-adjustment directions and round differently); the Gaussian SS dense
//! path evaluates the asymmetric partner direction explicitly. The mBecke
//! blocked path evaluates the algebraically
//! identical classic pair product for the default clamp and therefore agrees
//! to round-off only (a few ulp; see the `mbecke` kernel) — the single-point
//! API keeps ORCA's exact nearest-first order.
//!
//! Schemes:
//! * classic Becke (`becke`) and the Stratmann–Scuseria–Frisch switch
//!   (`stratmann`);
//! * Gaussian 16 `SSWeights` (`gaussian_ss`), the degree-13 `XScal = 0.86`
//!   switch with asymmetric support;
//! * ORCA "mBecke (2022)" nearest-first evaluation (`mbecke`);
//! * Laqua–Kussmann–Ochsenfeld saturation partition (`lko`).
//!
//! The classic Becke cell function, size adjustment, weight kernels and the
//! analytic gradient/Hessian kernels all live in the private `becke`
//! submodule.  [`PartitionDerivEvaluator`] is the public derivative entry
//! point; it is a separate evaluator, while the value path uses the same
//! fixed-array weight cells and remains guarded by bitwise regression tests.
//! Derivatives are implemented for classic Becke without the ORCA early
//! exit, for both compact-support schemes (SSF and Gaussian SS), for LKO and
//! for mBecke with the differentiable default clamp.  The ORCA classic
//! early-exit truncation and a literal mBecke clamp below one return
//! [`crate::Error::UnsupportedDerivative`]; a custom algorithm can provide
//! its own [`DerivScheme`] backend through
//! [`PartitionDerivEvaluator::with_scheme`].
//!
//! The public range API is [`PartitionEvaluator`]/[`Workspace`]:
//! a [`GridPlan`] supplies the topology and the evaluator fills only the
//! requested point range. For points that are not tied to a plan or a
//! materialised grid, [`PartitionEvaluator::fill_points`] /
//! [`PartitionEvaluator::fill_points_par`] write row-major
//! `[npoints, natom]` weights. The molecule-only `PartitionTables` and the
//! per-point kernels stay private to this module.
//!
//! Derivative API (streaming, selected rows, one fused run, no
//! $n_\mathrm{atom} \times n_\mathrm{grid}$ object):
//!
//! ```text
//! let ev = PartitionDerivEvaluator::new(&centers, &charges, config, options)?;
//! let mut ws = DerivWorkspace::new();
//! let mut out = DerivOutput::new();
//! let req = DerivRequest {
//!     weights: true,
//!     dw: true,
//!     ddw: false,
//!     dc: Some(DerivContract::exact(&c_dw, nset_dw)),
//!     ddc: Some(DerivContract::exact(&c_ddc, nset_ddc)),
//! };
//! ev.fill_serial(&plan, range, &mut ws, &req, &mut out)?;
//! // out.dw()  -> [nsel, 3, B]
//! // out.dc()  -> [nset_dw, nsel, 3]
//! // out.ddc() -> [nset_ddc, nsel, 3, natm, 3]
//! ```
//!
//! See the [partition derivative API](crate::doc::partition_derivatives) for
//! the API conventions, tensor layouts, sorted-grid analysis and validation
//! workflow; the Becke derivation is in
//! [Becke derivatives](crate::doc::becke_derivatives), and the formula and
//! cost differences of SSF, Gaussian SS, LKO and mBecke are in
//! [scheme variants](crate::doc::scheme_variants).

pub(crate) mod becke;
pub mod deriv;
mod kernel;
mod lko;
mod mbecke;
mod simd;
pub(crate) mod stratmann;

pub use deriv::{
    DEFAULT_WORKER_MEMORY_BYTES, DerivContract, DerivOptions, DerivOutput, DerivPoints,
    DerivRequest, DerivScheme, DerivScratch, DerivWorkspace, GradientContract, HessianBackend,
    HessianContract, PartitionDerivEvaluator,
};

use std::ops::Range;
use std::sync::Arc;

use stratmann::SsSwitch as _;

use crate::error::{Error, Result};
use crate::grid::{BatchBuffer, GridPlan};
use crate::molecule::Molecule;
use derive_builder::Builder;
use kernel::LaneScratch;
use mbecke::MBECKE_CLAMP;
use rayon::prelude::*;

/// Per-worker value-path scratch.
///
/// The classic and Stratmann pair-product kernels, the LKO kernels and the
/// mBecke fast path use fixed eight-point lanes; the SSF sparse kernel and
/// the exact mBecke nearest-first fallback keep per-point scratch.
enum Scratch {
    Lane(LaneScratch),
    Sparse(stratmann::Scratch),
    Lko(lko::Scratch),
    MBecke(mbecke::Scratch),
}

/// Internal value-path dispatch for one partition scheme.
///
/// This is the single registration point for a scheme's value kernel: add an
/// implementation and one arm in [`ops_for`]. The evaluator, the point
/// fallback and the blocked grid path all go through these methods, so a new
/// scheme has one value-kernel dispatch point.
///
/// The cell-function switch stays on [`PartitionConfig::switch_value`] because
/// it is called inside the pair loop; only the point/block kernels dispatch
/// through this trait.
trait SchemeOps: Sync {
    fn block_length(&self, _config: &PartitionConfig, num_centers: usize) -> usize {
        kernel::block_length(num_centers)
    }

    fn scratch(&self, _config: &PartitionConfig, tables: &PartitionTables, block: usize)
    -> Scratch;

    #[allow(clippy::too_many_arguments)]
    fn point_weights_all(
        &self,
        config: &PartitionConfig,
        tables: &PartitionTables,
        point: [f64; 3],
        out: &mut [f64],
        dist: &mut [f64],
        order: &mut [usize],
        prod: &mut [f64],
    );

    #[allow(clippy::too_many_arguments)]
    fn block_weights(
        &self,
        config: &PartitionConfig,
        tables: &PartitionTables,
        center: usize,
        grid: &[[f64; 3]],
        out: &mut [f64],
        scratch: &mut Scratch,
    );
}

struct BeckeOps;

impl SchemeOps for BeckeOps {
    fn scratch(
        &self,
        _config: &PartitionConfig,
        tables: &PartitionTables,
        block: usize,
    ) -> Scratch {
        Scratch::Lane(LaneScratch::new(tables.len(), block))
    }

    fn point_weights_all(
        &self,
        config: &PartitionConfig,
        tables: &PartitionTables,
        point: [f64; 3],
        out: &mut [f64],
        dist: &mut [f64],
        _order: &mut [usize],
        prod: &mut [f64],
    ) {
        becke::point_weights_all(config, tables, point, out, dist, prod);
    }

    fn block_weights(
        &self,
        config: &PartitionConfig,
        tables: &PartitionTables,
        center: usize,
        grid: &[[f64; 3]],
        out: &mut [f64],
        scratch: &mut Scratch,
    ) {
        let Scratch::Lane(scratch) = scratch else {
            unreachable!("classic Becke scratch type mismatch")
        };
        becke::block_weights(config, tables, center, grid, out, scratch);
    }
}

/// Compact-support SSF-family kernels, monomorphised over the switch.
struct SsOps<S: stratmann::SsSwitch>(std::marker::PhantomData<S>);

impl<S: stratmann::SsSwitch> SchemeOps for SsOps<S> {
    fn block_length(&self, _config: &PartitionConfig, num_centers: usize) -> usize {
        if use_sparse_ss(num_centers) {
            8
        } else {
            kernel::block_length(num_centers)
        }
    }

    fn scratch(
        &self,
        _config: &PartitionConfig,
        tables: &PartitionTables,
        block: usize,
    ) -> Scratch {
        if use_sparse_ss(tables.len()) {
            Scratch::Sparse(stratmann::Scratch::new(tables.len(), block))
        } else {
            Scratch::Lane(LaneScratch::new(tables.len(), block))
        }
    }

    fn point_weights_all(
        &self,
        _config: &PartitionConfig,
        tables: &PartitionTables,
        point: [f64; 3],
        out: &mut [f64],
        dist: &mut [f64],
        _order: &mut [usize],
        prod: &mut [f64],
    ) {
        if use_sparse_ss(tables.len()) {
            let mut sparse_order = vec![0u32; tables.len()];
            stratmann::point_weights_all::<S>(tables, point, out, dist, &mut sparse_order, prod);
        } else {
            stratmann::dense::point_weights_all::<S>(tables, point, out, dist, prod);
        }
    }

    fn block_weights(
        &self,
        _config: &PartitionConfig,
        tables: &PartitionTables,
        center: usize,
        grid: &[[f64; 3]],
        out: &mut [f64],
        scratch: &mut Scratch,
    ) {
        if use_sparse_ss(tables.len()) {
            let Scratch::Sparse(scratch) = scratch else {
                unreachable!("SSF scratch type mismatch")
            };
            stratmann::block_weights::<S>(tables, center, grid, out, scratch);
        } else {
            let Scratch::Lane(scratch) = scratch else {
                unreachable!("dense SSF scratch type mismatch")
            };
            stratmann::dense::block_weights::<S>(tables, center, grid, out, scratch);
        }
    }
}

struct LkoOps;

impl SchemeOps for LkoOps {
    fn block_length(&self, _config: &PartitionConfig, num_centers: usize) -> usize {
        if lko::use_sparse_lko(num_centers) {
            8
        } else {
            kernel::block_length(num_centers)
        }
    }

    fn scratch(
        &self,
        _config: &PartitionConfig,
        tables: &PartitionTables,
        block: usize,
    ) -> Scratch {
        Scratch::Lko(lko::Scratch::new(tables.len(), block))
    }

    fn point_weights_all(
        &self,
        config: &PartitionConfig,
        tables: &PartitionTables,
        point: [f64; 3],
        out: &mut [f64],
        dist: &mut [f64],
        _order: &mut [usize],
        prod: &mut [f64],
    ) {
        lko::point_weights_all(config, tables, point, out, dist, prod);
    }

    fn block_weights(
        &self,
        config: &PartitionConfig,
        tables: &PartitionTables,
        center: usize,
        grid: &[[f64; 3]],
        out: &mut [f64],
        scratch: &mut Scratch,
    ) {
        let Scratch::Lko(scratch) = scratch else {
            unreachable!("LKO scratch type mismatch")
        };
        lko::block_weights(config, tables, center, grid, out, scratch);
    }
}

struct MbeckeOps;

impl SchemeOps for MbeckeOps {
    fn scratch(&self, config: &PartitionConfig, tables: &PartitionTables, block: usize) -> Scratch {
        if mbecke_needs_exact(config, tables) {
            Scratch::MBecke(mbecke::Scratch::new(tables.len(), block))
        } else {
            Scratch::Lane(LaneScratch::new(tables.len(), block))
        }
    }

    fn point_weights_all(
        &self,
        config: &PartitionConfig,
        tables: &PartitionTables,
        point: [f64; 3],
        out: &mut [f64],
        dist: &mut [f64],
        order: &mut [usize],
        prod: &mut [f64],
    ) {
        mbecke::point_weights_all(config, tables, point, out, dist, order, prod);
    }

    fn block_weights(
        &self,
        config: &PartitionConfig,
        tables: &PartitionTables,
        center: usize,
        grid: &[[f64; 3]],
        out: &mut [f64],
        scratch: &mut Scratch,
    ) {
        if mbecke_needs_exact(config, tables) {
            let Scratch::MBecke(scratch) = scratch else {
                unreachable!("mBecke scratch type mismatch")
            };
            mbecke::block_weights(config, tables, center, grid, out, scratch);
        } else {
            let Scratch::Lane(scratch) = scratch else {
                unreachable!("mBecke classic scratch type mismatch")
            };
            becke::block_no_early_exit(config, tables, center, grid, out, scratch);
        }
    }
}

fn ops_for(scheme: PartitionScheme) -> &'static dyn SchemeOps {
    static BECKE: BeckeOps = BeckeOps;
    static STRATMANN: SsOps<stratmann::StratmannSwitch> = SsOps(std::marker::PhantomData);
    static GAUSSIAN_SS: SsOps<stratmann::GaussianSsSwitch> = SsOps(std::marker::PhantomData);
    static LKO: LkoOps = LkoOps;
    static MBECKE: MbeckeOps = MbeckeOps;

    match scheme {
        PartitionScheme::Becke => &BECKE,
        PartitionScheme::Stratmann => &STRATMANN,
        PartitionScheme::GaussianSs => &GAUSSIAN_SS,
        PartitionScheme::Lko => &LKO,
        PartitionScheme::MBecke => &MBECKE,
    }
}

/// Atomic-size-adjustment function.
///
/// The variants differ only in which radius enters the ratio and in the exact
/// expression; both formulas are documented on [`RadiiAdjust::factor`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RadiiAdjust {
    /// No size adjustment (`nu = mu`).
    None,
    /// Square-root radii.  PySCF's `treutler_atomic_radii_adjust` and ORCA's
    /// compatible size adjustment are the same expression; this uses ORCA's
    /// exact evaluation order so the ORCA preset stays bit-stable.
    Treutler,
    /// Raw radii (PySCF `becke_atomic_radii_adjust`).
    Becke,
}

impl RadiiAdjust {
    /// Size-adjustment factor `a_AB`, clamped to `[-0.5, 0.5]`.
    ///
    /// * `Treutler`: `x = sqrt(r_A/r_B)`, `u = (x-1)/(x+1)`,
    ///   `a = u/(u^2 - 1)` (ORCA-compatible expression, also PySCF's
    ///   Treutler adjustment up to floating-point evaluation order);
    /// * `Becke`: `a = 0.25 (r_B/r_A - r_A/r_B)`.
    ///
    /// Only the radius ratio enters, so a common unit cancels.  Pairs of
    /// (nearly) equal radii get `0`.
    #[inline]
    #[must_use]
    pub fn factor(self, r_a: f64, r_b: f64) -> f64 {
        match self {
            Self::None => 0.0,
            Self::Treutler => becke::size_adjust_factor(r_a, r_b),
            Self::Becke => {
                let ri = r_a.max(0.0) + 1e-200;
                let rj = r_b.max(0.0) + 1e-200;
                (0.25 * (rj / ri - ri / rj)).clamp(-0.5, 0.5)
            }
        }
    }
}

/// Radii handling: atomic-radii table + size-adjustment function.
#[derive(Clone, Debug, PartialEq)]
pub struct RadiiConfig {
    /// Atomic-radii table. Custom tables are indexed by nuclear charge
    /// (index 0 is a dummy); entries `1..` must be finite and positive.
    pub table: crate::radii::RadiiTable,
    /// Size-adjustment function.
    pub adjust: RadiiAdjust,
}

impl RadiiConfig {
    /// Explicit radii table + adjustment.
    #[must_use]
    pub fn new(table: crate::radii::RadiiTable, adjust: RadiiAdjust) -> Self {
        Self { table, adjust }
    }

    /// PySCF `radi.BRAGG_RADII` + Treutler adjustment (PySCF default).
    #[must_use]
    pub fn pyscf() -> Self {
        Self::new(crate::radii::RadiiTable::Bragg, RadiiAdjust::Treutler)
    }

    /// PySCF `radi.COVALENT_RADII` + Treutler adjustment.
    #[must_use]
    pub fn covalent() -> Self {
        Self::new(crate::radii::RadiiTable::Covalent, RadiiAdjust::Treutler)
    }

    /// PySCF BRAGG + `radi.becke_atomic_radii_adjust`.
    #[must_use]
    pub fn becke_adjust() -> Self {
        Self::new(crate::radii::RadiiTable::Bragg, RadiiAdjust::Becke)
    }

    /// PySCF BRAGG without a size adjustment.
    #[must_use]
    pub fn no_adjust() -> Self {
        Self::new(crate::radii::RadiiTable::Bragg, RadiiAdjust::None)
    }

    /// Uniform unit radii with no size adjustment (Gaussian SS cells).
    ///
    /// Every pair has `r_A / r_B = 1`, so the SSF switch sees no
    /// element-dependent cell asymmetry.
    #[must_use]
    pub fn uniform() -> Self {
        Self::new(
            crate::radii::RadiiTable::Custom(Arc::from(vec![1.0; 119])),
            RadiiAdjust::None,
        )
    }

    /// Validate the radii table.
    ///
    /// Preset tables are compile-time data. A custom table is indexed by
    /// nuclear charge (index 0 is a dummy); entries `1..` must be finite and
    /// positive. The table may be shorter than 119, but it must contain at
    /// least one real entry.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] for an empty, non-finite or non-positive
    /// custom table.
    pub fn validate(&self) -> Result<()> {
        if let crate::radii::RadiiTable::Custom(table) = &self.table {
            if table.len() < 2 {
                return Err(Error::InvalidArgument {
                    argument: "custom radii table",
                });
            }
            if table[1..]
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
            {
                return Err(Error::InvalidArgument {
                    argument: "custom radii table",
                });
            }
        }
        Ok(())
    }
}

/// Whether the sparse compact-support (SSF/Gaussian SS) kernel beats the
/// dense pair-product kernel.
///
/// The dense kernel is `O(natm^2)` per point but SIMD over eight points; the
/// sparse kernel is `O(natm)` distances + a sorted candidate prefix +
/// `O(n_survivors * natm)` products, with a large constant per point.  The
/// cut below is chosen empirically: it shifts with the compiled ISA because
/// the dense kernel benefits from AVX-512/AVX2 far more than the mostly
/// scalar sparse kernel, and a wrong call costs little on the affected tiny
/// sizes while the large-molecule gain is substantial.
#[inline]
fn use_sparse_ss(n: usize) -> bool {
    if cfg!(target_feature = "avx512f") {
        n >= 26
    } else if cfg!(target_feature = "avx2") {
        n >= 24
    } else {
        n >= 16
    }
}

/// Whether mBecke needs the exact ORCA nearest-first kernel instead of the
/// classic pair product (see [`mbecke::classic_product_is_valid`]).
#[inline]
fn mbecke_needs_exact(cfg: &PartitionConfig, tables: &PartitionTables) -> bool {
    !mbecke::classic_product_is_valid(cfg, tables)
}

/// Molecule-level version of [`mbecke::classic_product_is_valid`] for the
/// derivative dispatch, which does not otherwise build the weight tables.
#[inline]
pub(crate) fn mbecke_classic_product_is_valid(
    cfg: &PartitionConfig,
    centers: &[[f64; 3]],
    charges: &[u32],
) -> Result<bool> {
    if cfg.mbecke_clamp < 1.0 {
        return Ok(false);
    }
    let tables = PartitionTables::new(cfg, centers, charges)?;
    Ok(mbecke::classic_product_is_valid(cfg, &tables))
}

/// Partition scheme: the mathematical construction, independent of radii.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum PartitionScheme {
    /// Classic Becke cell functions, classic polynomial, pairwise products.
    #[default]
    Becke,
    /// Classic cell functions with the Stratmann–Scuseria–Frisch switch
    /// (PySCF `gen_grid.stratmann`).
    Stratmann,
    /// Gaussian 16 `SSWeights` switch (`XScal = 0.86`, degree 13,
    /// asymmetric support).  Use with equality radii for the Gaussian SS
    /// mapping.
    GaussianSs,
    /// Classic cell functions evaluated nearest-atom-first with the ORCA
    /// mBecke early exit / clamp / nearest-atom shortcut.
    MBecke,
    /// Laqua–Kussmann–Ochsenfeld saturation partition
    /// (PySCF `becke_lko`).
    Lko,
}

/// Largest accepted Becke step-function hardness.
///
/// This is a resource guard, not a physical parameter; classic Becke uses 3.
pub const MAX_HARDNESS: usize = 1024;

/// Full partition configuration: scheme + radii + evaluation constants.
///
/// ORCA and PySCF differ only in the values passed here (radii table and
/// adjustment), not in the code path.  [`PartitionConfig::new`] fills in the
/// defaults; use [`PartitionConfigBuilder`] (generated by `derive_builder`)
/// to override `hardness`, `mbecke_clamp` or `becke_early_exit`.
#[derive(Clone, Debug, PartialEq, Builder)]
#[builder(setter(into))]
pub struct PartitionConfig {
    /// Partition scheme.
    pub scheme: PartitionScheme,
    /// Radii handling.
    pub radii: RadiiConfig,
    /// Number of step-function iterations (Becke's "hardness"; 3 is classic).
    ///
    /// Must be in `1..=MAX_HARDNESS`; validated by [`PartitionConfig::validate`].
    #[builder(default = "3")]
    pub hardness: usize,
    /// mBecke clamp parameter `p` (ORCA keyword `%method GRIDA0`).
    ///
    /// Must be finite and non-negative; values `>= 1` are inert. Validated by
    /// [`PartitionConfig::validate`].
    #[builder(default = "MBECKE_CLAMP")]
    pub mbecke_clamp: f64,
    /// Classic Becke early exit: stop the pair sweep of atom `i` once
    /// `|1 - f| <= 1e-14` (ORCA's classic Becke pair sweep).  PySCF's
    /// reference implementation continues the sweep, so its presets disable
    /// this flag; the kernels are otherwise identical.
    #[builder(default = "false")]
    pub becke_early_exit: bool,
}

impl PartitionConfig {
    /// Explicit scheme + radii with the default optional fields.
    ///
    /// Use [`PartitionConfigBuilder`] to override `hardness`, `mbecke_clamp`
    /// or `becke_early_exit`.
    #[must_use]
    pub fn new(scheme: PartitionScheme, radii: RadiiConfig) -> Self {
        PartitionConfigBuilder::default()
            .scheme(scheme)
            .radii(radii)
            .build()
            .expect("PartitionConfig default fields are valid")
    }

    /// Validate the configuration invariants.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] for `hardness == 0` or above
    /// [`MAX_HARDNESS`], a non-finite/negative `mbecke_clamp`, or an invalid
    /// custom radii table.
    pub fn validate(&self) -> Result<()> {
        if self.hardness == 0 || self.hardness > MAX_HARDNESS {
            return Err(Error::InvalidArgument {
                argument: "hardness",
            });
        }
        if !self.mbecke_clamp.is_finite() || self.mbecke_clamp < 0.0 {
            return Err(Error::InvalidArgument {
                argument: "mbecke_clamp",
            });
        }
        self.radii.validate()
    }

    /// PySCF's program-default partition: classic Becke with PySCF
    /// Bragg/Treutler radii.
    #[must_use]
    pub fn pyscf() -> Self {
        Self::new(PartitionScheme::Becke, RadiiConfig::pyscf())
    }

    /// Gaussian 16 `SSWeights`: the degree-13 `XScal = 0.86` switch with
    /// equality radii and no size adjustment.
    ///
    /// This is the exact Gaussian SS cell mapping; the switch and its first
    /// two derivatives are implemented analytically.  Gaussian's positive
    /// support cutoff (`mu >= 0.852`) is reproduced; the optional
    /// `IOp(79)` cutoff and Gaussian's neighbour-shell truncation are not
    /// modelled.
    #[must_use]
    pub fn gaussian_ss() -> Self {
        Self::new(PartitionScheme::GaussianSs, RadiiConfig::uniform())
    }

    /// Approximation of Gaussian SS weights using the PySCF/SSF switch
    /// (`XScal = 0.64`, degree 7) with equality radii.
    ///
    /// In representative X–H profiles this differs from Gaussian 16's
    /// normalized fraction by at most about `2.3e-3` (RMS below `1e-3`);
    /// inner shells agree.  Kept for comparisons with the PySCF SSF form;
    /// [`gaussian_ss`](Self::gaussian_ss) is the exact mapping.
    #[must_use]
    pub fn gaussian_ss_mapping() -> Self {
        Self::new(PartitionScheme::Stratmann, RadiiConfig::uniform())
    }

    /// Gaussian `BWeights`: classic Becke cells with equality radii and no
    /// size adjustment, matching Gaussian 16's default B weights.
    ///
    /// Gaussian's optional `IOp(79)` Becke cell cutoff is not modelled; the
    /// default cutoff is inactive in the reference checks maintained outside
    /// this package, but a user who changes it must add the screening
    /// separately.
    #[must_use]
    pub fn gaussian_becke() -> Self {
        Self::new(PartitionScheme::Becke, RadiiConfig::uniform())
    }

    /// Precompute the molecule-only tables for a geometry.
    ///
    /// # Arguments
    ///
    /// * `centers` — `[natm, 3]` row-major nuclear coordinates in Bohr.
    /// * `charges` — `[natm]` atomic numbers.
    pub(crate) fn tables(&self, centers: &[[f64; 3]], charges: &[u32]) -> Result<PartitionTables> {
        self.validate()?;
        PartitionTables::new(self, centers, charges)
    }

    /// Compatibility shim: normalized weights of one point via a fresh
    /// [`PartitionEvaluator`].
    ///
    /// Prefer [`PartitionEvaluator::from_parts`] when evaluating more than
    /// one point.
    ///
    /// # Errors
    /// The evaluator construction and point-length errors.
    pub fn weights(
        &self,
        point: [f64; 3],
        centers: &[[f64; 3]],
        charges: &[u32],
    ) -> Result<Vec<f64>> {
        PartitionEvaluator::from_parts(centers, charges, self.clone())?.weights_at(point)
    }

    /// Compatibility shim: normalized weight of one center at one point via a
    /// fresh [`PartitionEvaluator`].
    ///
    /// # Errors
    /// [`Error::InvalidAtom`] for an out-of-range center, plus the evaluator
    /// construction errors.
    pub fn weight(
        &self,
        center: usize,
        point: [f64; 3],
        centers: &[[f64; 3]],
        charges: &[u32],
    ) -> Result<f64> {
        let evaluator = PartitionEvaluator::from_parts(centers, charges, self.clone())?;
        let weights = evaluator.weights_at(point)?;
        weights.get(center).copied().ok_or(Error::InvalidAtom {
            atom: center,
            natm: weights.len(),
        })
    }

    /// Compatibility shim: build a reusable [`PartitionEvaluator`] for
    /// caller-supplied points.
    ///
    /// # Errors
    /// The evaluator construction errors.
    pub fn weights_evaluator(
        &self,
        centers: &[[f64; 3]],
        charges: &[u32],
    ) -> Result<PartitionEvaluator> {
        PartitionEvaluator::from_parts(centers, charges, self.clone())
    }

    /// Normalized partition weight of one center at one point, reusing
    /// precomputed `PartitionTables`.
    ///
    /// # Errors
    /// [`Error::InvalidAtom`] when `center` is not an atom index.
    #[cfg(test)]
    pub(crate) fn weight_with_tables(
        &self,
        tables: &PartitionTables,
        center: usize,
        point: [f64; 3],
    ) -> Result<f64> {
        let n = tables.len();
        if center >= n {
            return Err(Error::InvalidAtom {
                atom: center,
                natm: n,
            });
        }
        let mut out = vec![0.0; n];
        let mut dist = vec![0.0; n];
        let mut order = vec![0usize; n];
        let mut prod = vec![0.0; n];
        self.evaluate_point(tables, point, &mut out, &mut dist, &mut order, &mut prod);
        Ok(out[center])
    }

    /// Normalized weights of `center` for every point of `grid`, serial.
    ///
    /// The grid is evaluated in blocks with bounded scratch; this is the
    /// reference the parallel version is tested against.
    ///
    /// # Arguments
    ///
    /// * `tables` — precomputed molecule tables.
    /// * `center` — global atom index whose weight is written.
    /// * `grid` — `[B, 3]` row-major global grid coordinates in Bohr.
    /// * `out` — `[B]` normalized weight of `center` per point.
    pub(crate) fn center_weights(
        &self,
        tables: &PartitionTables,
        center: usize,
        grid: &[[f64; 3]],
        out: &mut [f64],
    ) {
        assert_eq!(
            grid.len(),
            out.len(),
            "center_weights: grid and output length differ"
        );
        assert!(center < tables.len(), "center_weights: center out of range");
        if grid.is_empty() {
            return;
        }
        let block = self.block_length(tables.len());
        let mut scratch = self.scratch(tables, block);
        for (block_index, chunk) in out.chunks_mut(block).enumerate() {
            let start = block_index * block;
            let grid_block = &grid[start..start + chunk.len()];
            self.center_weights_block(tables, center, grid_block, chunk, &mut scratch);
        }
    }

    /// Normalized weights of `center` for every point of `grid`, parallel.
    ///
    /// Blocks of grid points are distributed over rayon workers; every worker
    /// keeps its scratch across blocks.  The result is bit-identical to
    /// [`Self::center_weights`] regardless of the thread count for the
    /// classic Becke, SSF, Gaussian SS and LKO schemes.  For mBecke with the default clamp
    /// the blocked path evaluates the algebraically identical classic pair
    /// product, which differs from the nearest-first reference order by
    /// round-off (see [`mbecke`]); the nearest-first order stays available
    /// through [`Self::weights`] / [`Self::weight`].  A clamp `< 1.0`, or a
    /// molecule whose `max |a_AB| > 0.5`, keeps the nearest-first kernel.
    /// Do not call from inside another rayon parallel region.
    ///
    /// # Arguments
    ///
    /// * `tables` — precomputed molecule tables.
    /// * `center` — global atom index whose weight is written.
    /// * `grid` — `[B, 3]` row-major global grid coordinates in Bohr.
    /// * `out` — `[B]` normalized weight of `center` per point.
    pub(crate) fn center_weights_par(
        &self,
        tables: &PartitionTables,
        center: usize,
        grid: &[[f64; 3]],
        out: &mut [f64],
    ) {
        assert_eq!(
            grid.len(),
            out.len(),
            "center_weights_par: grid and output length differ"
        );
        assert!(
            center < tables.len(),
            "center_weights_par: center out of range"
        );
        if grid.is_empty() {
            return;
        }
        let block = self.block_length(tables.len());
        out.par_chunks_mut(block).enumerate().for_each_init(
            || self.scratch(tables, block),
            |scratch, (block_index, chunk)| {
                let start = block_index * block;
                let grid_block = &grid[start..start + chunk.len()];
                self.center_weights_block(tables, center, grid_block, chunk, scratch);
            },
        );
    }

    fn block_length(&self, num_centers: usize) -> usize {
        self.ops().block_length(self, num_centers)
    }

    fn scratch(&self, tables: &PartitionTables, block: usize) -> Scratch {
        self.ops().scratch(self, tables, block)
    }

    fn evaluate_point(
        &self,
        tables: &PartitionTables,
        point: [f64; 3],
        out: &mut [f64],
        dist: &mut [f64],
        order: &mut [usize],
        prod: &mut [f64],
    ) {
        self.ops()
            .point_weights_all(self, tables, point, out, dist, order, prod);
    }

    fn center_weights_block(
        &self,
        tables: &PartitionTables,
        center: usize,
        grid: &[[f64; 3]],
        out: &mut [f64],
        scratch: &mut Scratch,
    ) {
        self.ops()
            .block_weights(self, tables, center, grid, out, scratch);
    }

    /// Cell-function switch of the pair-product schemes.
    #[inline]
    pub(crate) fn switch_value(&self, g: f64) -> f64 {
        match self.scheme {
            PartitionScheme::Stratmann => stratmann::StratmannSwitch::switch(g),
            PartitionScheme::GaussianSs => stratmann::GaussianSsSwitch::switch(g),
            _ => becke::step_function(g, self.hardness),
        }
    }

    fn ops(&self) -> &'static dyn SchemeOps {
        ops_for(self.scheme)
    }
}

/// Molecule-only data shared by every grid point: per-atom radii, pair
/// adjustment factors `a_AB`, and interatomic distances `R_AB`.
///
/// Build once per molecule ([`PartitionConfig::tables`]) and reuse it for
/// every atomic grid.  Storage is `O(n^2)` (three `f64` matrices: `a_ab`,
/// `r_ab`, `inv_r`, plus the LKO-only `fac` cache).
#[derive(Debug, Clone)]
pub(crate) struct PartitionTables {
    n: usize,
    centers: Vec<[f64; 3]>,
    a_ab: Vec<f64>,
    r_ab: Vec<f64>,
    /// `[i*n + j] = 1 / R_ij` (zero diagonal, zero for coincident centres).
    inv_r: Vec<f64>,
    /// LKO saturation factor `[i*n + j] = 1 / sat(R_ij)` (zero diagonal).
    ///
    /// Only populated for [`PartitionScheme::Lko`]; empty otherwise.  LKO
    /// consumes `fac` instead of `inv_r`, and evaluating `sat` once per pair
    /// (instead of once per pair per grid point) removes a 12-term
    /// polynomial and an `exp` from the hot loop.
    fac: Vec<f64>,
    /// SSF candidate-cut ratio: a candidate can only survive below
    /// `rcut * d_min` (see [`crate::partition::stratmann`]).
    rcut: f64,
    /// Largest `|a_AB|` in the molecule.
    a_max: f64,
}

/// Map a flat index into the triangular set of atom pairs `j < i`.
///
/// Row `i` (pairs `(i, 0..i)`) starts at flat index `i*(i-1)/2`, so every
/// parallel task handles exactly one pair and the work is evenly distributed.
/// The two correction loops absorb the rounding of the `sqrt` estimate.
#[inline]
fn pair_from_index(k: usize) -> (usize, usize) {
    let mut i = (((1.0 + 8.0 * k as f64).sqrt() + 1.0) / 2.0) as usize;
    while i * (i - 1) / 2 > k {
        i -= 1;
    }
    while (i + 1) * i / 2 <= k {
        i += 1;
    }
    (i, k - i * (i - 1) / 2)
}

/// Raw-pointer wrapper for the disjoint parallel writes of
/// `PartitionTables::new`.
///
/// The pointee is a flat `f64` matrix; the pointer is neither `Send` nor
/// `Sync` by itself, so it is wrapped here after auditing the access pattern.
/// Workers write only the entries of their own unordered pair, and the
/// allocation is not resized during the parallel region.
#[derive(Clone, Copy)]
struct SharedPtr(*mut f64);
// SAFETY: see the type comment and the `PartitionTables::new` write sites.
unsafe impl Send for SharedPtr {}
unsafe impl Sync for SharedPtr {}

impl SharedPtr {
    /// # Safety
    ///
    /// `index` must be in bounds of the allocation, and no other worker may
    /// write the same index.
    #[inline]
    unsafe fn write(self, index: usize, value: f64) {
        // SAFETY: the caller guarantees `index` is in bounds and unaliased by
        // any other worker's write.
        unsafe {
            self.0.add(index).write(value);
        }
    }
}

impl PartitionTables {
    /// Precompute the tables of `cfg` for a geometry (Bohr, element charges).
    ///
    /// The `O(n^2)` pair loop is evaluated in parallel over the flat
    /// triangular pair index; each task owns one unordered pair and writes
    /// both of its symmetric entries.
    ///
    /// # Arguments
    ///
    /// * `cfg` — partition configuration (radii table and adjustment).
    /// * `centers` — `[natm, 3]` row-major nuclear coordinates in Bohr.
    /// * `charges` — `[natm]` atomic numbers.
    /// # Errors
    /// [`Error::LengthMismatch`] if `centers.len() != charges.len()` and
    /// [`Error::MissingElementData`] when the radii table does not cover one
    /// of the charges.
    pub(crate) fn new(
        cfg: &PartitionConfig,
        centers: &[[f64; 3]],
        charges: &[u32],
    ) -> Result<Self> {
        cfg.validate()?;
        if centers.len() != charges.len() {
            return Err(Error::LengthMismatch {
                expected: centers.len(),
                got: charges.len(),
            });
        }
        let n = centers.len();
        for (ia, (&z, center)) in charges.iter().zip(centers).enumerate() {
            if !(1..=118).contains(&z) {
                return Err(Error::UnsupportedElement { z });
            }
            if center.iter().any(|value| !value.is_finite()) {
                return Err(Error::NonFiniteCoordinate { atom: ia });
            }
        }
        for i in 0..n {
            for j in 0..i {
                if centers[i] == centers[j] {
                    return Err(Error::CoincidentCenters { a: i, b: j });
                }
            }
        }
        let radii: Vec<f64> = charges
            .iter()
            .map(|&z| cfg.radii.table.bohr(z))
            .collect::<Result<_>>()?;
        if radii
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(Error::InvalidArgument {
                argument: "atomic radii",
            });
        }
        let mut a_ab = vec![0.0; n * n];
        let mut r_ab = vec![0.0; n * n];
        let mut inv_r = vec![0.0; n * n];
        let mut fac = if cfg.scheme == PartitionScheme::Lko {
            vec![0.0; n * n]
        } else {
            Vec::new()
        };
        let r_ptr = SharedPtr(r_ab.as_mut_ptr());
        let a_ptr = SharedPtr(a_ab.as_mut_ptr());
        let i_ptr = SharedPtr(inv_r.as_mut_ptr());
        let f_ptr = if fac.is_empty() {
            None
        } else {
            Some(SharedPtr(fac.as_mut_ptr()))
        };
        // Flat triangular index: one task per unordered pair, so every worker
        // does the same amount of work (unlike a `par_iter` over `i`, where
        // row `i` owns `i` pairs).
        let num_pairs = n.saturating_sub(1) * n / 2;
        (0..num_pairs)
            .into_par_iter()
            .with_min_len(64)
            .for_each(|k| {
                let (i, j) = pair_from_index(k);
                let r = dist3(centers[i], centers[j]);
                // Evaluate both orders instead of negating one: the two
                // roundings need not be exact negatives, and the schemes
                // consume `a_AB` in both directions.
                let a_ij = cfg.radii.adjust.factor(radii[i], radii[j]);
                let a_ji = cfg.radii.adjust.factor(radii[j], radii[i]);
                // SAFETY: each unordered pair `(i, j)`, `j < i`, is visited
                // exactly once with four distinct target indices; the vectors
                // are never reallocated and no reference aliases these
                // elements while the workers run.
                let inv = if r > 0.0 { 1.0 / r } else { 0.0 };
                unsafe {
                    r_ptr.write(i * n + j, r);
                    r_ptr.write(j * n + i, r);
                    a_ptr.write(i * n + j, a_ij);
                    a_ptr.write(j * n + i, a_ji);
                    i_ptr.write(i * n + j, inv);
                    i_ptr.write(j * n + i, inv);
                }
                if let Some(f_ptr) = f_ptr {
                    let f = 1.0 / lko::lko_sat_func(r);
                    // SAFETY: same disjoint-pair argument as above.
                    unsafe {
                        f_ptr.write(i * n + j, f);
                        f_ptr.write(j * n + i, f);
                    }
                }
            });
        let mut a_max = 0.0_f64;
        for &a in &a_ab {
            a_max = a_max.max(a.abs());
        }
        let mu_cut = match cfg.scheme {
            PartitionScheme::GaussianSs => stratmann::GAUSSIAN_MU_POS,
            _ => stratmann::A,
        };
        Ok(Self {
            n,
            centers: centers.to_vec(),
            a_ab,
            r_ab,
            inv_r,
            fac,
            rcut: ss_rcut(a_max, mu_cut),
            a_max,
        })
    }

    /// Number of centers.
    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.n
    }

    /// Coordinates of center `i` (Bohr).
    #[must_use]
    pub(crate) fn center(&self, i: usize) -> [f64; 3] {
        self.centers[i]
    }

    /// Size-adjustment factor `a_AB` (antisymmetric, zero diagonal).
    #[must_use]
    pub(crate) fn a_ab(&self, a: usize, b: usize) -> f64 {
        self.a_ab[a * self.n + b]
    }

    /// Interatomic distance `R_AB` (Bohr, zero diagonal).
    #[must_use]
    pub(crate) fn r_ab(&self, a: usize, b: usize) -> f64 {
        self.r_ab[a * self.n + b]
    }

    /// Contiguous `inv_r` slice starting at `offset`.
    #[must_use]
    pub(crate) fn inv_r_slice(&self, offset: usize, len: usize) -> &[f64] {
        &self.inv_r[offset..offset + len]
    }

    /// LKO saturation factor `fac_AB = 1 / sat(R_AB)` (zero diagonal).
    ///
    /// # Panics
    /// For a non-LKO table (the matrix is not built for other schemes).
    #[must_use]
    pub(crate) fn fac(&self, a: usize, b: usize) -> f64 {
        self.fac[a * self.n + b]
    }

    /// A single `fac` entry by flat row-major offset.
    #[must_use]
    pub(crate) fn fac_row(&self, offset: usize) -> f64 {
        self.fac[offset]
    }

    /// Contiguous `a_ab` slice starting at `offset`.
    #[must_use]
    pub(crate) fn a_slice(&self, offset: usize, len: usize) -> &[f64] {
        &self.a_ab[offset..offset + len]
    }

    /// A single `a_ab` entry by flat row-major offset.
    #[must_use]
    pub(crate) fn a_ab_row(&self, offset: usize) -> f64 {
        self.a_ab[offset]
    }

    /// A single `inv_r` entry by flat row-major offset.
    #[must_use]
    pub(crate) fn inv_r_row(&self, offset: usize) -> f64 {
        self.inv_r[offset]
    }

    /// Largest `|a_AB|` in the molecule.
    ///
    /// [`RadiiAdjust::factor`] clamps every pair factor to `[-0.5, 0.5]`, so
    /// this is at most `0.5`.  It is exposed because the mBecke fast path and
    /// the delegated mBecke derivative are only valid when the size-adjusted
    /// coordinate cannot leave `[-1, 1]`, which is exactly this bound.
    #[must_use]
    pub(crate) fn max_abs_a(&self) -> f64 {
        self.a_max
    }

    /// SSF candidate-cut ratio derived from [`Self::a_max`].
    ///
    /// For `mu >= mu_kill`, where `mu - a_max (1 - mu^2) = A`, every allowed
    /// size adjustment gives `nu >= A`, i.e. the cell factor is exactly zero.
    /// Since `mu = (d_i - d_j)/R_ij >= (d_i - d_j)/(d_i + d_j)`, an atom with
    /// `d_i >= rcut d_j` is certified dead by the closer atom `j`.
    #[must_use]
    pub(crate) fn ss_rcut(&self) -> f64 {
        self.rcut
    }
}

/// Compact-support candidate-cut ratio for a maximum size-adjustment
/// magnitude and a kill cutoff `mu_cut`.
#[must_use]
fn ss_rcut(a_max: f64, mu_cut: f64) -> f64 {
    let a = a_max.clamp(0.0, 0.5);
    let mu = if a > 0.0 {
        (-1.0 + (1.0 + 4.0 * a * (mu_cut + a)).sqrt()) / (2.0 * a)
    } else {
        mu_cut
    };
    ((1.0 + mu) / (1.0 - mu)).max(1.0)
}

/// Normalize unnormalized cell products.
///
/// The shared policy normalizes whenever the sum is finite and non-zero.
/// Only an exact zero or a non-finite sum takes the fallback that puts unit
/// mass on the first centre.  The guard is scale-free: a symmetric cage whose
/// products are all tiny but equal still normalizes to the uniform partition.
fn normalize(pa: &mut [f64]) {
    let total: f64 = pa.iter().sum();
    if total != 0.0 && total.is_finite() {
        for v in pa.iter_mut() {
            *v /= total;
        }
    } else {
        pa.fill(0.0);
        if let Some(first) = pa.first_mut() {
            *first = 1.0;
        }
    }
}

/// Euclidean distance between two `[x, y, z]` points (Bohr).
#[inline]
pub(crate) fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]))
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_with_tables_matches_weights() {
        let centers = [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 1.1, -0.4]];
        let charges = [1u32, 9, 6];
        for cfg in [
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
            PartitionConfig::new(PartitionScheme::Stratmann, RadiiConfig::pyscf()),
            PartitionConfig::new(PartitionScheme::Lko, RadiiConfig::pyscf()),
        ] {
            let tables = cfg.tables(&centers, &charges).unwrap();
            for &x in &[-2.0, -0.3, 0.0, 0.7, 1.8] {
                let point = [x, 0.1, -0.2];
                let all = cfg.weights(point, &centers, &charges).unwrap();
                for (center, &want) in all.iter().enumerate() {
                    let got = cfg.weight_with_tables(&tables, center, point).unwrap();
                    assert_eq!(got.to_bits(), want.to_bits(), "{cfg:?} center {center}");
                }
            }
        }
    }

    #[test]
    fn becke_scheme_is_independent_of_radii() {
        let orca = PartitionConfig::new(
            PartitionScheme::Becke,
            crate::partition::test_support::custom_radii(),
        );
        let pyscf = PartitionConfig::new(PartitionScheme::Becke, RadiiConfig::pyscf());
        assert_eq!(orca.scheme, PartitionScheme::Becke);
        assert_eq!(pyscf.scheme, PartitionScheme::Becke);
        assert_ne!(orca.radii, pyscf.radii);
        let centers = [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]];
        let charges = [2u32, 10];
        let pt = [0.3, 0.1, 1.0];
        let wa = orca.weight(0, pt, &centers, &charges).unwrap();
        let wb = pyscf.weight(0, pt, &centers, &charges).unwrap();
        assert!(
            (wa - wb).abs() > 1e-4,
            "radii handling did not change the weight"
        );
    }

    #[test]
    fn empty_molecule_is_accepted() {
        for cfg in [
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
            PartitionConfig::new(PartitionScheme::Lko, RadiiConfig::pyscf()),
        ] {
            let w = cfg.weights([0.0, 0.0, 0.0], &[], &[]).unwrap();
            assert!(w.is_empty(), "{cfg:?}: non-empty weights for no atoms");
        }
    }

    #[test]
    fn pair_index_is_a_triangular_bijection() {
        for n in 1..65 {
            let pairs = n * (n - 1) / 2;
            let mut seen = vec![false; n * n];
            for k in 0..pairs {
                let (i, j) = pair_from_index(k);
                assert!(j < i && i < n, "n={n} k={k}: ({i}, {j})");
                assert!(!seen[i * n + j], "n={n}: pair ({i}, {j}) repeated");
                seen[i * n + j] = true;
            }
            assert_eq!(seen.iter().filter(|&&hit| hit).count(), pairs);
        }
    }

    /// The parallel pair-table construction must fill both symmetric
    /// entries with exactly the serial values.
    #[test]
    fn partition_tables_are_symmetric() {
        let centers = [
            [0.0, 0.0, 0.0],
            [1.4, 1.1, 0.0],
            [-1.4, 1.1, 0.2],
            [0.0, -2.0, 0.1],
            [3.0, -0.4, 1.2],
        ];
        let charges = [8u32, 1, 1, 6, 7];
        let configs = [
            PartitionConfig::new(
                PartitionScheme::Becke,
                crate::partition::test_support::custom_radii(),
            ),
            PartitionConfig::new(PartitionScheme::Becke, RadiiConfig::pyscf()),
            PartitionConfig::new(PartitionScheme::Becke, RadiiConfig::becke_adjust()),
            PartitionConfig::new(PartitionScheme::Lko, RadiiConfig::pyscf()),
        ];
        for cfg in &configs {
            let tables = cfg.tables(&centers, &charges).unwrap();
            let radii: Vec<f64> = charges
                .iter()
                .map(|&z| cfg.radii.table.bohr(z).unwrap())
                .collect();
            for i in 0..centers.len() {
                assert_eq!(tables.center(i), centers[i]);
                assert_eq!(tables.r_ab(i, i), 0.0);
                assert_eq!(tables.a_ab(i, i), 0.0);
                for j in 0..i {
                    let r = super::dist3(centers[i], centers[j]);
                    assert_eq!(tables.r_ab(i, j).to_bits(), r.to_bits());
                    assert_eq!(tables.r_ab(j, i).to_bits(), r.to_bits());
                    let a_ij = cfg.radii.adjust.factor(radii[i], radii[j]);
                    let a_ji = cfg.radii.adjust.factor(radii[j], radii[i]);
                    assert_eq!(tables.a_ab(i, j).to_bits(), a_ij.to_bits());
                    assert_eq!(tables.a_ab(j, i).to_bits(), a_ji.to_bits());
                }
            }
        }
    }

    /// The sparse SSF branch must reproduce the per-point reference bit for
    /// bit, including above the dense/sparse crossover.
    #[test]
    fn sparse_blocked_matches_point_reference() {
        let n = 30;
        let mut centers = Vec::new();
        for k in 0..n {
            let t = k as f64;
            centers.push([2.1 * t.cos(), 0.8 * t, 1.3 * t.sin()]);
        }
        let charges: Vec<u32> = (0..n).map(|k| 1 + (k as u32 * 5) % 18).collect();
        let grid: Vec<[f64; 3]> = (0..257)
            .map(|k| {
                let t = k as f64 * 0.031;
                [
                    6.0 * t.cos() + 0.4 * t.sin(),
                    4.0 * t.sin() - 0.3,
                    3.0 * t.cos() + 0.2 * t,
                ]
            })
            .collect();
        let cfg = PartitionConfig::new(PartitionScheme::Stratmann, RadiiConfig::pyscf());
        assert!(
            use_sparse_ss(n),
            "test molecule is below the dense/sparse crossover"
        );
        let tables = cfg.tables(&centers, &charges).unwrap();
        for center in [0usize, n / 2, n - 1] {
            let mut want = vec![0.0; grid.len()];
            for (p, &point) in grid.iter().enumerate() {
                want[p] = cfg.weight(center, point, &centers, &charges).unwrap();
            }
            let mut serial = vec![0.0; grid.len()];
            cfg.center_weights(&tables, center, &grid, &mut serial);
            let mut parallel = vec![0.0; grid.len()];
            cfg.center_weights_par(&tables, center, &grid, &mut parallel);
            for p in 0..grid.len() {
                assert_eq!(
                    serial[p].to_bits(),
                    want[p].to_bits(),
                    "SSF center {center} point {p}: serial"
                );
                assert_eq!(
                    parallel[p].to_bits(),
                    want[p].to_bits(),
                    "SSF center {center} point {p}: parallel"
                );
            }
        }
    }

    /// The blocked/parallel kernels must reproduce the per-point reference
    /// bit-for-bit, except for the mBecke fast path.
    ///
    /// With ORCA's default clamp the mBecke product is mathematically the
    /// classic pair product; the public blocked kernel evaluates it in the
    /// classic multiplication order (eight-point lanes) while
    /// `PartitionConfig::weights_into` keeps the nearest-first reference
    /// order.  The two agree to round-off only.  The tolerance below is the
    /// maximum on this grid (see `mbecke_fast_path_matches_reference`).
    #[test]
    fn blocked_and_parallel_match_point_reference() {
        let centers = [
            [0.0, 0.0, 0.0],
            [1.43, 1.11, 0.0],
            [-1.43, 1.11, 0.2],
            [0.0, -2.0, 0.1],
        ];
        let charges = [8u32, 1, 1, 6];
        let grid: Vec<[f64; 3]> = (0..257)
            .map(|k| {
                let t = k as f64 * 0.027;
                [
                    2.5 * t.cos() + 0.3 * t.sin(),
                    2.0 * t.sin() - 0.2,
                    1.5 * t.cos() + 0.1 * t,
                ]
            })
            .collect();
        let configs = [
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
            // Non-default clamp: clamp branches fire, so the exact ORCA
            // nearest-first kernel must be used bit-for-bit.
            PartitionConfigBuilder::default()
                .scheme(PartitionScheme::MBecke)
                .radii(crate::partition::test_support::custom_radii())
                .becke_early_exit(true)
                .mbecke_clamp(0.5)
                .build()
                .unwrap(),
            PartitionConfig::new(PartitionScheme::Becke, RadiiConfig::pyscf()),
            PartitionConfig::new(PartitionScheme::Stratmann, RadiiConfig::pyscf()),
            PartitionConfig::new(PartitionScheme::Lko, RadiiConfig::pyscf()),
            PartitionConfig::new(PartitionScheme::Becke, RadiiConfig::no_adjust()),
        ];
        for cfg in &configs {
            let tables = cfg.tables(&centers, &charges).unwrap();
            for center in 0..centers.len() {
                let mut want = vec![0.0; grid.len()];
                for (p, &point) in grid.iter().enumerate() {
                    want[p] = cfg.weight(center, point, &centers, &charges).unwrap();
                }
                let mut serial = vec![0.0; grid.len()];
                cfg.center_weights(&tables, center, &grid, &mut serial);
                let mut parallel = vec![0.0; grid.len()];
                cfg.center_weights_par(&tables, center, &grid, &mut parallel);
                let mbecke_fast = cfg.scheme == PartitionScheme::MBecke
                    && mbecke::classic_product_is_valid(cfg, &tables);
                for p in 0..grid.len() {
                    if mbecke_fast {
                        assert!(
                            (serial[p] - want[p]).abs() <= 4e-15,
                            "{cfg:?} center {center} point {p}: serial {} vs {}",
                            serial[p],
                            want[p]
                        );
                        assert!(
                            (parallel[p] - want[p]).abs() <= 4e-15,
                            "{cfg:?} center {center} point {p}: parallel {} vs {}",
                            parallel[p],
                            want[p]
                        );
                        continue;
                    }
                    assert_eq!(
                        serial[p].to_bits(),
                        want[p].to_bits(),
                        "{cfg:?} center {center} point {p}: serial"
                    );
                    assert_eq!(
                        parallel[p].to_bits(),
                        want[p].to_bits(),
                        "{cfg:?} center {center} point {p}: parallel"
                    );
                }
            }
        }
    }
}

/// Per-thread scratch for batch evaluation.
///
/// Reuse one workspace across all ranges handled by a thread. The buffers grow
/// to the largest atom's point count and are not deallocated between batches.
#[derive(Debug, Default)]
pub struct Workspace {
    coords: Vec<[f64; 3]>,
    weights: Vec<f64>,
}

impl Workspace {
    /// Empty workspace.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Partition evaluator for one molecule and one partition configuration.
///
/// The evaluator owns the molecule and the precomputed `O(natm^2)` tables.
/// Build it once per geometry, then use [`Self::fill_points`] for arbitrary
/// point sets or [`Self::fill_plan`] for ranges of a [`GridPlan`]. The same
/// type serves both arbitrary-point and grid-range evaluation.
#[derive(Debug, Clone)]
pub struct PartitionEvaluator {
    molecule: Molecule,
    config: PartitionConfig,
    tables: PartitionTables,
}

impl PartitionEvaluator {
    /// Build the molecule tables.
    ///
    /// # Arguments
    ///
    /// * `molecule` — validated nuclear charges and positions in Bohr.
    /// * `config` — partition configuration.
    ///
    /// ```
    /// use ruaa_dft_grids::grid::{BatchBuffer, GridPlan, PyscfGridSpec};
    /// use ruaa_dft_grids::molecule::Molecule;
    /// use ruaa_dft_grids::partition::{PartitionEvaluator, Workspace};
    ///
    /// # fn main() -> ruaa_dft_grids::Result<()> {
    /// let water = Molecule::new([
    ///     (8u32, [0.0, 0.0, 0.0]),
    ///     (1, [0.0, 1.430, 1.110]),
    ///     (1, [0.0, -1.430, 1.110]),
    /// ])?;
    /// let spec = PyscfGridSpec::from_level(0)?;
    /// let plan = GridPlan::build(&water, &spec)?;
    /// let evaluator = PartitionEvaluator::new(&water, spec.partition.clone())?;
    ///
    /// let mut workspace = Workspace::new();
    /// let mut batch = BatchBuffer::new();
    /// let mut total = 0.0;
    /// for range in plan.batches(1024) {
    ///     evaluator.fill_plan(&plan, range, &mut workspace, &mut batch)?;
    ///     total += batch.weights().iter().sum::<f64>();
    /// }
    /// assert!(total > 0.0);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    /// [`Error::UnsupportedElement`] / [`Error::NonFiniteCoordinate`] for an
    /// invalid molecule, and [`Error::MissingElementData`] when the radii
    /// table does not cover one of the charges.
    pub fn new(molecule: &Molecule, config: PartitionConfig) -> Result<Self> {
        molecule.validate()?;
        let tables = config.tables(molecule.centers(), molecule.charges())?;
        Ok(Self {
            molecule: molecule.clone(),
            config,
            tables,
        })
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
    ) -> Result<Self> {
        if centers.len() != charges.len() {
            return Err(Error::LengthMismatch {
                expected: centers.len(),
                got: charges.len(),
            });
        }
        let molecule = Molecule::new(charges.iter().copied().zip(centers.iter().copied()))?;
        Self::new(&molecule, config)
    }

    /// The molecule this evaluator was built for.
    #[must_use]
    pub fn molecule(&self) -> &Molecule {
        &self.molecule
    }

    /// Number of atoms (columns of the point output).
    #[must_use]
    pub fn atom_count(&self) -> usize {
        self.tables.len()
    }

    /// The partition configuration.
    #[must_use]
    pub fn config(&self) -> &PartitionConfig {
        &self.config
    }

    /// Write the normalized weights of one point into `out` (`[natm]`).
    ///
    /// # Errors
    /// [`Error::LengthMismatch`] if `out.len() != self.atom_count()`.
    pub fn fill_point(&self, point: [f64; 3], out: &mut [f64]) -> Result<()> {
        let n = self.tables.len();
        if out.len() != n {
            return Err(Error::LengthMismatch {
                expected: n,
                got: out.len(),
            });
        }
        if n == 0 {
            return Ok(());
        }
        let mut dist = vec![0.0; n];
        let mut order = vec![0usize; n];
        let mut prod = vec![0.0; n];
        self.config
            .evaluate_point(&self.tables, point, out, &mut dist, &mut order, &mut prod);
        Ok(())
    }

    /// Normalized weights of one point (`[natm]`).
    ///
    /// # Errors
    /// Same as [`Self::fill_point`].
    pub fn weights_at(&self, point: [f64; 3]) -> Result<Vec<f64>> {
        let mut out = vec![0.0; self.tables.len()];
        self.fill_point(point, &mut out)?;
        Ok(out)
    }

    /// Write the weights of `points` into `out` (row-major
    /// `[npoints, natm]`), serial.
    ///
    /// # Errors
    /// [`Error::LengthMismatch`] if `out.len() != points.len() * natom`.
    pub fn fill_points(&self, points: &[[f64; 3]], out: &mut [f64]) -> Result<()> {
        let n = self.tables.len();
        let expected = points.len() * n;
        if out.len() != expected {
            return Err(Error::LengthMismatch {
                expected,
                got: out.len(),
            });
        }
        if n == 0 || points.is_empty() {
            return Ok(());
        }
        let mut dist = vec![0.0; n];
        let mut order = vec![0usize; n];
        let mut prod = vec![0.0; n];
        for (chunk, point) in out.chunks_exact_mut(n).zip(points) {
            self.config.evaluate_point(
                &self.tables,
                *point,
                chunk,
                &mut dist,
                &mut order,
                &mut prod,
            );
        }
        Ok(())
    }

    /// Parallel form of [`Self::fill_points`], using the ambient Rayon pool.
    ///
    /// # Errors
    /// Same as [`Self::fill_points`].
    pub fn fill_points_par(&self, points: &[[f64; 3]], out: &mut [f64]) -> Result<()> {
        let n = self.tables.len();
        let expected = points.len() * n;
        if out.len() != expected {
            return Err(Error::LengthMismatch {
                expected,
                got: out.len(),
            });
        }
        if n == 0 || points.is_empty() {
            return Ok(());
        }
        out.par_chunks_mut(n).zip(points.par_iter()).for_each_init(
            || (vec![0.0; n], vec![0usize; n], vec![0.0; n]),
            |(dist, order, prod), (chunk, point)| {
                self.config
                    .evaluate_point(&self.tables, *point, chunk, dist, order, prod);
            },
        );
        Ok(())
    }

    /// Weights of `points` (row-major `[npoints, natm]`), serial.
    ///
    /// # Errors
    /// Same as [`Self::fill_points`].
    pub fn weights(&self, points: &[[f64; 3]]) -> Result<Vec<f64>> {
        let mut out = vec![0.0; points.len() * self.tables.len()];
        self.fill_points(points, &mut out)?;
        Ok(out)
    }

    /// Evaluate the partition weights for the global point `range`.
    ///
    /// # Output (row-major)
    ///
    /// * `out.coords()` — `[B, 3]` translated grid coordinates.
    /// * `out.quad_weights()` — `[B]` unpartitioned quadrature weights `q`.
    /// * `out.partition_weights()` — `[B]` normalized partition weights
    ///   `Pbar / Z`.
    /// * `out.weights()` — `[B]` total weights `q * Pbar / Z`.
    /// * `out.atom()` — `[B]` generating atom per point.
    ///
    /// # Errors
    /// [`Error::LengthMismatch`] when the plan's atom count or charges differ
    /// from the evaluator's, and [`Error::RangeOutOfBounds`] for a reversed or
    /// out-of-plan range.
    pub fn fill_plan(
        &self,
        plan: &GridPlan,
        range: Range<usize>,
        ws: &mut Workspace,
        out: &mut BatchBuffer,
    ) -> Result<()> {
        self.fill_plan_impl(plan, range, ws, out, false)
    }

    /// Parallel variant of [`Self::fill_plan`].
    ///
    /// Do not call it from inside another Rayon parallel region: the
    /// underlying per-atom kernel parallelises blocks internally.
    ///
    /// # Errors
    /// Same as [`Self::fill_plan`].
    pub fn fill_plan_par(
        &self,
        plan: &GridPlan,
        range: Range<usize>,
        ws: &mut Workspace,
        out: &mut BatchBuffer,
    ) -> Result<()> {
        self.fill_plan_impl(plan, range, ws, out, true)
    }

    fn fill_plan_impl(
        &self,
        plan: &GridPlan,
        range: Range<usize>,
        ws: &mut Workspace,
        out: &mut BatchBuffer,
        parallel: bool,
    ) -> Result<()> {
        if plan.atom_count() != self.tables.len() {
            return Err(Error::LengthMismatch {
                expected: self.tables.len(),
                got: plan.atom_count(),
            });
        }
        for (ia, &z) in self.molecule.charges().iter().enumerate() {
            if plan.atom_grid(ia).z != z {
                return Err(Error::ChargesMismatch {
                    atom: ia,
                    expected: z,
                    got: plan.atom_grid(ia).z,
                });
            }
        }
        if range.start > range.end || range.end > plan.len() {
            return Err(Error::RangeOutOfBounds {
                start: range.start,
                end: range.end,
                len: plan.len(),
            });
        }

        out.reset(range.clone());
        let mut start = range.start;
        while start < range.end {
            let ia = plan.parent_of(start) as usize;
            let atom_end = plan.atom_range(ia).end.min(range.end);
            let n = atom_end - start;

            ws.coords.clear();
            ws.coords.reserve(n);
            for g in start..atom_end {
                let p = plan.local_coord(g);
                let c = self.molecule.centers()[ia];
                ws.coords.push([c[0] + p[0], c[1] + p[1], c[2] + p[2]]);
            }

            ws.weights.clear();
            ws.weights.resize(n, 0.0);
            if parallel {
                self.config
                    .center_weights_par(&self.tables, ia, &ws.coords, &mut ws.weights);
            } else {
                self.config
                    .center_weights(&self.tables, ia, &ws.coords, &mut ws.weights);
            }

            out.coords.extend_from_slice(&ws.coords);
            for g in start..atom_end {
                let q = plan.quad_weight(g);
                let p = ws.weights[g - start];
                out.quad_weights.push(q);
                out.partition_weights.push(p);
                out.weights.push(q * p);
            }
            out.atom.extend(std::iter::repeat_n(ia as u32, n));
            start = atom_end;
        }

        Ok(())
    }
}

/// Test-only radii used by the unit tests that need a table distinct from
/// PySCF's.  The values are a synthetic non-PySCF set, not a program table.
#[cfg(test)]
pub(crate) mod test_support {
    use super::{RadiiAdjust, RadiiConfig};
    use crate::radii::RadiiTable;

    /// A small asymmetric radii table with the same units (Bohr) as the
    /// PySCF presets.
    pub(crate) fn custom_radii() -> RadiiConfig {
        let mut table = vec![1.0; 119];
        table[0] = 0.0;
        for (z, radius) in [
            (1, 0.661),
            (2, 0.661),
            (3, 1.600),
            (4, 1.600),
            (5, 1.600),
            (6, 1.323),
            (7, 1.323),
            (8, 1.058),
            (9, 0.945),
            (10, 0.945),
            (11, 1.600),
            (12, 1.600),
            (17, 1.890),
            (18, 1.890),
            (19, 1.600),
            (26, 2.173),
            (36, 2.646),
            (54, 3.307),
        ] {
            table[z] = radius;
        }
        RadiiConfig::new(
            RadiiTable::custom(table).expect("test radii are valid"),
            RadiiAdjust::Treutler,
        )
    }
}
