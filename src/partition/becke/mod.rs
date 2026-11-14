//! Classic Becke partition: cell functions, weights, and analytic
//! first/second derivatives.
//!
//! * [`switch`] — the polynomial switch `p(x) = 3/2 x - 1/2 x^3`, the
//!   size-adjusted coordinate `nu(mu)`, and the value / first / second
//!   `nu`-derivatives used by the gradient and Hessian kernels;
//! * [`weights`] — the classic pair-product weight kernels (blocked and
//!   reference).  The dense SSF fallback lives in
//!   [`crate::partition::stratmann::dense`]; it shares the lane machinery
//!   in [`crate::partition::kernel`];
//! * [`gradient`] — analytic `dw/dR` for the exact classic scheme;
//! * [`hessian`] — analytic `d2w/dR2` (and optional `dw`) for the exact
//!   classic scheme, pointwise;
//! * [`hessian_lane`] — the fixed-array eight-point variant of the lane
//!   Hessian path;
//! * [`deriv`] — [`BeckeDeriv`], the [`crate::partition::DerivScheme`]
//!   adapter used by the streaming evaluator.
//!
//! Derivative support is deliberately limited to the exact classic scheme
//! (`PartitionScheme::Becke`, `becke_early_exit == false`).  The early-exit
//! weight is a truncated product and is discontinuous when a pair factor
//! crosses the cutoff, so it has no well-defined analytic derivative; the
//! public derivative evaluator rejects it instead of returning a value that
//! silently disagrees with finite differences.

pub(crate) mod deriv;
pub(crate) mod gradient;
pub(crate) mod hessian;
pub(crate) mod hessian_lane;
pub(crate) mod switch;
pub(crate) mod weights;

pub(crate) use switch::{size_adjust_factor, step_function};
pub(crate) use weights::{block_no_early_exit, block_weights, point_weights_all};

use crate::error::Result;
use crate::partition::{PartitionConfig, PartitionTables};

/// Molecule-only tables used by the analytic derivative kernels.
///
/// This is a superset of [`PartitionTables`]: the weight tables stay exactly
/// as they are (so the weight path is untouched), and the derivative-only
/// data is appended:
///
/// * `inv_r[i*n+j] = 1 / R_ij`;
/// * `u_ab[i*n+j] = (R_i - R_j) / R_ij`, the unit vector from `j` to `i`
///   (`u_ji = -u_ij`);
/// * `pu_ab[i*n+j] = (I - u u^T) / R_ij`, the second derivative of
///   `R_ij` with respect to the two centre coordinates (`pu` is symmetric
///   under `i <-> j`).
///
/// Storage is `O(n^2)` (about 13 `f64` per unordered pair), which fits the
/// "`natm x natm` matrices are allowed" budget; no grid-sized object is
/// ever created.
#[derive(Debug, Clone)]
pub(crate) struct DerivTables {
    base: PartitionTables,
    inv_r: Vec<f64>,
    u_ab: Vec<[f64; 3]>,
    pu_ab: Vec<[f64; 9]>,
}

impl DerivTables {
    /// Build the weight tables plus the derivative-only pair data.
    ///
    /// # Arguments
    ///
    /// * `cfg` — partition configuration (radii table and adjustment).
    /// * `centers` — `[natm, 3]` row-major nuclear coordinates in Bohr.
    /// * `charges` — `[natm]` atomic numbers.
    ///
    /// # Layout
    ///
    /// All pair arrays are `[natm, natm]` row-major; `pu_ab` is
    /// `[natm, natm, 3, 3]`.  The vectors are symmetric/antisymmetric as
    /// documented on the struct.
    /// # Errors
    /// Propagates [`PartitionTables::new`]'s length/radii errors.
    pub(crate) fn new(
        cfg: &PartitionConfig,
        centers: &[[f64; 3]],
        charges: &[u32],
    ) -> Result<Self> {
        let base = PartitionTables::new(cfg, centers, charges)?;
        let n = base.len();
        let mut inv_r = vec![0.0; n * n];
        let mut u_ab = vec![[0.0; 3]; n * n];
        let mut pu_ab = vec![[0.0; 9]; n * n];
        for i in 1..n {
            for j in 0..i {
                let r = base.r_ab(i, j);
                let inv = 1.0 / r;
                inv_r[i * n + j] = inv;
                inv_r[j * n + i] = inv;
                let mut u = [0.0; 3];
                for t in 0..3 {
                    u[t] = (centers[i][t] - centers[j][t]) * inv;
                }
                u_ab[i * n + j] = u;
                u_ab[j * n + i] = [-u[0], -u[1], -u[2]];
                let mut pu = [0.0; 9];
                for t in 0..3 {
                    for s in 0..3 {
                        let delta = if t == s { 1.0 } else { 0.0 };
                        pu[t * 3 + s] = (delta - u[t] * u[s]) * inv;
                    }
                }
                pu_ab[i * n + j] = pu;
                pu_ab[j * n + i] = pu;
            }
        }
        Ok(Self {
            base,
            inv_r,
            u_ab,
            pu_ab,
        })
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.base.len()
    }

    #[inline]
    pub(crate) fn center(&self, i: usize) -> [f64; 3] {
        self.base.center(i)
    }

    #[inline]
    pub(crate) fn a_ab(&self, i: usize, j: usize) -> f64 {
        self.base.a_ab(i, j)
    }

    #[inline]
    pub(crate) fn r_ab(&self, i: usize, j: usize) -> f64 {
        self.base.r_ab(i, j)
    }

    #[inline]
    pub(crate) fn inv_r(&self, i: usize, j: usize) -> f64 {
        self.inv_r[i * self.base.len() + j]
    }

    #[inline]
    pub(crate) fn u_ab(&self, i: usize, j: usize) -> [f64; 3] {
        self.u_ab[i * self.base.len() + j]
    }

    #[inline]
    pub(crate) fn pu_ab(&self, i: usize, j: usize) -> [f64; 9] {
        self.pu_ab[i * self.base.len() + j]
    }

    /// The underlying weight tables (radii, pair factors, distances).
    #[inline]
    pub(crate) fn base(&self) -> &PartitionTables {
        &self.base
    }

    /// Contiguous `a_ab` slice starting at flat offset `offset`.
    #[inline]
    pub(crate) fn a_slice(&self, offset: usize, len: usize) -> &[f64] {
        self.base.a_slice(offset, len)
    }

    /// Contiguous `inv_r` slice starting at flat offset `offset`.
    #[inline]
    pub(crate) fn inv_r_slice(&self, offset: usize, len: usize) -> &[f64] {
        self.base.inv_r_slice(offset, len)
    }
}
