//! Scheme-specific derivative kernels.
//!
//! [`PartitionDerivEvaluator`](super::PartitionDerivEvaluator) owns the
//! streaming loop, the atom selection, the contractions and the parallel
//! reduction.  A partition scheme only has to evaluate
//!
//! * a block of points at first order ([`DerivScheme::gradient_block`]);
//! * one point at second order, optionally together with its first
//!   derivative ([`DerivScheme::point_hessian`]).
//!
//! That is deliberately the whole interface: a future partition algorithm
//! with a completely different cell function still streams grid points
//! through the same evaluator by implementing this trait.  The built-in
//! implementation for the exact classic Becke scheme lives in
//! `crate::partition::becke::deriv`.
//!
//! # Derivative convention
//!
//! Every derivative returned by a [`DerivScheme`] is a **total** derivative
//! of the grid weight
//!
//! ```text
//! w_g = q_g Pbar_g / Z_g
//! ```
//!
//! with respect to the nuclear coordinates, *including* the motion of the
//! grid point with its generating atom (`parent`).  The evaluator does not
//! apply the generating-atom correction itself because only the scheme knows
//! how to accumulate the sums over unselected rows cheaply.  See the
//! [partition derivative API](crate::doc::partition_derivatives) §6 for the
//! correction and [Becke derivatives](crate::doc::becke_derivatives) for the
//! Becke-specific form.
//!
//! # Selection
//!
//! A scheme is built for one selection ([`DerivScheme::selection`]) and
//! produces only those rows.  The selection must be strictly increasing and
//! every entry must be `< atom_count()`; the evaluator validates this when
//! the scheme is installed.

use std::any::{Any, TypeId};

/// Opaque per-thread scratch owned by a [`DerivScheme`].
///
/// The blanket implementation means any `'static + Send` type can be used as
/// scratch; the scheme downcasts it back to its concrete type.  The evaluator
/// keeps two scratch slots (block and point), so implementing both methods
/// does not force a scheme to use the same type for the two evaluation modes.
pub trait DerivScratch: Any + Send {
    /// Upcast for the scheme-side downcast.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T: Any + Send> DerivScratch for T {
    #[inline]
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Scheme-specific analytic derivative kernel.
///
/// Implement this trait to add a new partition algorithm to
/// [`PartitionDerivEvaluator`](super::PartitionDerivEvaluator); build the
/// evaluator with
/// [`PartitionDerivEvaluator::with_scheme`](super::PartitionDerivEvaluator::with_scheme).
///
/// **Stability.** This is an unstable extension point: the `TypeId` scratch
/// protocol and the default `hessian_scratch_bytes` are not yet a semver
/// contract. It is public so external backends can be prototyped, but pin the
/// crate version if you implement it.
///
/// The evaluator never holds an `natm x ngrids` object: a scheme is called
/// with a bounded block (first derivatives) or with one point (second
/// derivatives), and the caller consumes or contracts the result immediately.
///
/// # Arguments and layouts (all row-major)
///
/// Let `B` be the number of points in the current block, `nsel` the number of
/// selected rows, `natm` the atom count and `n3 = 3 * natm`.
///
/// * [`Self::gradient_block`]
///   * `coords` — `[B, 3]`, global grid coordinates (already translated).
///   * `parents` — `[B]`, generating atom of each point.
///   * `quad` — `[B]`, quadrature weights `q_g`.
///   * `weights` — `[B]` output, bare partition weight `Pbar / Z`.
///   * `dw` — `[nsel, 3, block]` storage, flattened
///     `(slot * 3 + t) * block + point`; only the first `B <= block` points
///     of each row are written.  It stores the derivative of the **total**
///     weight `q_g Pbar / Z`.
/// * [`Self::point_hessian`]
///   * `point` — `[3]`, global grid coordinate.
///   * `parent` — generating atom `A_g`.
///   * `quad` — quadrature weight `q_g`.
///   * `grad` — optional `[nsel, 3]` output, flattened `slot * 3 + t`; the
///     total first derivative of `q_g Pbar / Z`.
///   * returned slice — `[nsel, 3, natm, 3]`, flattened
///     `((slot * 3 + t) * natm + B_atom) * 3 + s`; the total second
///     derivative of `q_g Pbar / Z`.
pub trait DerivScheme: std::fmt::Debug + Send + Sync {
    /// Number of atoms (`natm`).
    fn atom_count(&self) -> usize;

    /// Selected differentiated rows, strictly increasing and in `0..natm`.
    fn selection(&self) -> &[usize];

    /// Concrete type of [`Self::new_block_scratch`], used by the workspace to
    /// detect a scheme change.
    ///
    /// This must be the `TypeId` of the concrete type returned by
    /// `new_block_scratch`; the kernels downcast the cached scratch to it.
    fn block_scratch_kind(&self) -> TypeId;

    /// Concrete type of [`Self::new_point_scratch`], used by the workspace to
    /// detect a scheme change.
    ///
    /// This must be the `TypeId` of the concrete type returned by
    /// `new_point_scratch`; the kernels downcast the cached scratch to it.
    fn point_scratch_kind(&self) -> TypeId;

    /// Allocate block scratch for `block` points and the scheme's selection.
    fn new_block_scratch(&self, block: usize) -> Box<dyn DerivScratch>;

    /// Allocate per-point Hessian scratch for the scheme's selection.
    fn new_point_scratch(&self) -> Box<dyn DerivScratch>;

    /// Estimated bytes of one per-worker Hessian scratch.
    ///
    /// The evaluator uses this to cap the number of parallel Hessian chunks by
    /// the configured worker memory budget.  The default is `0`, meaning
    /// "unknown"; a scheme with a large Hessian state should override it with
    /// its actual scratch size.
    fn hessian_scratch_bytes(&self) -> usize {
        0
    }

    /// Whether the evaluator should stream Hessians one point at a time
    /// instead of assembling eight-point blocks.
    ///
    /// A scheme with a large per-point Hessian (for example an all-row
    /// `O(natm^3)` output) avoids the block-sized staging buffers this way.
    fn hessian_pointwise(&self) -> bool {
        false
    }

    /// Evaluate the weight and the selected first derivatives of one block.
    ///
    /// See the [trait documentation](Self) for the layouts.  The scheme may
    /// assume `coords.len() == parents.len() == quad.len() == weights.len()`
    /// and `dw.len() == nsel * 3 * block`, where `block` is the block size
    /// passed to [`Self::new_block_scratch`].  The last block of a range may
    /// be shorter than `block`; only its first `coords.len()` points are
    /// written.
    fn gradient_block(
        &self,
        coords: &[[f64; 3]],
        parents: &[u32],
        quad: &[f64],
        weights: &mut [f64],
        dw: &mut [f64],
        scratch: &mut dyn DerivScratch,
    );

    /// Evaluate a block of up to eight points at second order.
    ///
    /// The default implementation calls [`Self::point_hessian`] once per
    /// point.  A scheme with a fixed-array kernel should override it so the
    /// evaluator can stream eight points at a time.
    ///
    /// # Layout
    ///
    /// With `B = coords.len() <= 8`, `n3 = 3 * atom_count()` and
    /// `nsel = selection().len()`:
    ///
    /// * `weights` — `[B]` output;
    /// * `grad` — optional `[B, nsel, 3]` output;
    /// * `ddw` — `[B, nsel, 3, n3]` output, point-major, flattened
    ///   `(point * nsel * 3 + slot * 3 + t) * n3 + c`.
    ///
    /// The default implementation is correct for any `B` and is the fallback
    /// for custom schemes.
    #[allow(clippy::too_many_arguments)]
    fn hessian_block(
        &self,
        coords: &[[f64; 3]],
        parents: &[u32],
        quad: &[f64],
        weights: &mut [f64],
        mut grad: Option<&mut [f64]>,
        ddw: &mut [f64],
        scratch: &mut dyn DerivScratch,
    ) {
        let b = coords.len();
        let n3 = 3 * self.atom_count();
        let nsel = self.selection().len();
        debug_assert_eq!(parents.len(), b);
        debug_assert_eq!(quad.len(), b);
        debug_assert_eq!(weights.len(), b);
        debug_assert_eq!(ddw.len(), b * nsel * 3 * n3);
        if let Some(g) = grad.as_deref() {
            debug_assert_eq!(g.len(), b * nsel * 3);
        }
        for idx in 0..b {
            let grad_slice = grad
                .as_deref_mut()
                .map(|g| &mut g[idx * nsel * 3..(idx + 1) * nsel * 3]);
            let (weight, hess) = self.point_hessian(
                coords[idx],
                parents[idx] as usize,
                quad[idx],
                grad_slice,
                scratch,
            );
            weights[idx] = weight;
            let base = idx * nsel * 3 * n3;
            ddw[base..base + nsel * 3 * n3].copy_from_slice(hess);
        }
    }

    /// Evaluate one point at second order, optionally with its gradient.
    ///
    /// The returned slice borrows `scratch` and stays valid until the next
    /// call that uses the same scratch.  `grad`, when present, is written
    /// before the method returns.
    ///
    /// # Returns
    ///
    /// `(Pbar / Z, ddw)` where `ddw` is `[nsel, 3, natm, 3]` (see the
    /// [trait documentation](Self)).
    fn point_hessian<'s>(
        &self,
        point: [f64; 3],
        parent: usize,
        quad: f64,
        grad: Option<&mut [f64]>,
        scratch: &'s mut dyn DerivScratch,
    ) -> (f64, &'s [f64]);
}
