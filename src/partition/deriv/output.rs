//! One-pass request and output buffers.

use std::ops::Range;

use crate::error::Result;

use super::contract::{DerivContract, GradientContract, HessianContract};

/// Which derivatives one evaluation run should produce.
///
/// Raw and contracted outputs are independent: any subset may be requested,
/// and `dc` / `ddc` may use different [`DerivContract`] tensors (including
/// different `nset`).  A run that requests `ddw` or `ddc` evaluates the
/// Hessian kernel, which also produces `dw` / `dc` at almost no extra cost;
/// a gradient-only run uses the faster blocked kernel.
///
/// A raw per-point Hessian can also be consumed without allocating
/// [`DerivOutput::ddw`] by using
/// [`crate::partition::PartitionDerivEvaluator::fill_serial_visit`] with
/// [`DerivPoints::plan`](crate::partition::DerivPoints::plan) or
/// [`DerivPoints::grid`](crate::partition::DerivPoints::grid).
#[derive(Clone, Copy, Debug, Default)]
pub struct DerivRequest<'a> {
    /// Write the bare partition weights `Pbar / Z`, shape `[B]`.
    pub weights: bool,
    /// Write the raw first derivatives, shape `[nsel, 3, B]`.
    pub dw: bool,
    /// Write the raw second derivatives, shape `[nsel, 3, natm, 3, B]`.
    ///
    /// This is `O(nsel * natm^2 * B)` doubles; prefer a small streaming range
    /// (or [`crate::partition::PartitionDerivEvaluator::fill_serial_visit`] with
    /// [`DerivPoints`](crate::partition::DerivPoints)) when the full batch
    /// would be large.
    pub ddw: bool,
    /// Accumulate the contracted first derivatives, shape
    /// `[nset, nsel, 3]`.
    pub dc: Option<DerivContract<'a>>,
    /// Accumulate the contracted second derivatives, shape
    /// `[nset, nsel, 3, natm, 3]`.
    pub ddc: Option<DerivContract<'a>>,
}

/// Outputs of one evaluation range.
///
/// The raw arrays are **overwritten** by every `fill*` call; the contracted
/// `dc` / `ddc` arrays are **accumulated** across calls that request them.
/// Use [`Self::zero_contracts`] before the first range of a stream.  A
/// contract that is not requested by a call is cleared (and
/// [`Self::dc`]/[`Self::ddc`] then return `None`), so the output always
/// describes the current request.
#[derive(Debug, Default)]
pub struct DerivOutput {
    pub(crate) range: Range<usize>,
    pub(crate) atoms: Vec<usize>,
    pub(crate) ncols: usize,
    pub(crate) weights: Vec<f64>,
    pub(crate) dw: Vec<f64>,
    pub(crate) ddw: Vec<f64>,
    pub(crate) dw_requested: bool,
    pub(crate) ddw_requested: bool,
    pub(crate) dc: Option<GradientContract>,
    pub(crate) ddc: Option<HessianContract>,
}

impl DerivOutput {
    /// Empty output.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn prepare(
        &mut self,
        atoms: &[usize],
        natm: usize,
        range: &Range<usize>,
        req: &DerivRequest<'_>,
    ) -> Result<()> {
        let len = range.len();
        let nsel = atoms.len();
        if let Some(contract) = req.dc {
            contract.check_range(len)?;
        }
        if let Some(contract) = req.ddc {
            contract.check_range(len)?;
        }
        self.range = range.clone();
        self.atoms.clear();
        self.atoms.extend_from_slice(atoms);
        self.ncols = natm;

        if req.weights {
            self.weights.resize(len, 0.0);
        } else {
            self.weights.clear();
        }
        self.dw_requested = req.dw;
        if req.dw {
            self.dw.resize(nsel * 3 * len, 0.0);
        } else {
            self.dw.clear();
        }
        self.ddw_requested = req.ddw;
        if req.ddw {
            self.ddw.resize(nsel * 3 * natm * 3 * len, 0.0);
        } else {
            self.ddw.clear();
        }

        if let Some(contract) = req.dc {
            let reuse = self
                .dc
                .as_ref()
                .is_some_and(|out| out.nset() == contract.nset() && out.atoms() == atoms);
            if !reuse {
                self.dc = Some(GradientContract::new(atoms, contract.nset()));
            }
        } else {
            self.dc = None;
        }
        if let Some(contract) = req.ddc {
            let reuse = self.ddc.as_ref().is_some_and(|out| {
                out.nset() == contract.nset() && out.ncols() == natm && out.atoms() == atoms
            });
            if !reuse {
                self.ddc = Some(HessianContract::new(atoms, contract.nset(), natm));
            }
        } else {
            self.ddc = None;
        }
        Ok(())
    }

    /// Set the accumulated `dc` and `ddc` entries to zero.
    pub fn zero_contracts(&mut self) {
        if let Some(dc) = &mut self.dc {
            dc.zero();
        }
        if let Some(ddc) = &mut self.ddc {
            ddc.zero();
        }
    }

    /// Global point interval of the last `fill*` call.
    #[must_use]
    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }

    /// Selected row atoms of the last call, shape `[nsel]`.
    #[must_use]
    pub fn atoms(&self) -> &[usize] {
        &self.atoms
    }

    /// Bare partition weights `Pbar / Z` of the range, shape `[B]`.
    ///
    /// Empty unless [`DerivRequest::weights`] was set.
    #[must_use]
    pub fn partition_weights(&self) -> &[f64] {
        &self.weights
    }

    /// Raw selected first derivatives, row-major `[nsel, 3, B]`.
    ///
    /// Flattened `(slot * 3 + t) * B + point`; empty unless
    /// [`DerivRequest::dw`] was set.  The derivative is of the total weight
    /// `q_g Pbar / Z` (quadrature included).
    #[must_use]
    pub fn dw(&self) -> &[f64] {
        &self.dw
    }

    /// The `3 x B` gradient block of one selected atom, or `None`.
    ///
    /// # Arguments
    ///
    /// * `atom` — global atom index; `None` when it is not selected.
    ///
    /// # Returns
    ///
    /// `[3, B]` row-major (`t * B + point`).
    #[must_use]
    pub fn dw_atom(&self, atom: usize) -> Option<&[f64]> {
        if !self.dw_requested {
            return None;
        }
        let slot = self.atoms.iter().position(|&a| a == atom)?;
        let len = self.range.len();
        Some(&self.dw[slot * 3 * len..(slot * 3 + 3) * len])
    }

    /// Raw selected second derivatives, row-major `[nsel, 3, natm, 3, B]`.
    ///
    /// Flattened
    /// `(((slot * 3 + t) * natm + B_atom) * 3 + s) * B + point`; empty unless
    /// [`DerivRequest::ddw`] was set.  This is a streaming-batch tensor
    /// (`B` is the range length), not a grid-sized one.
    #[must_use]
    pub fn ddw(&self) -> &[f64] {
        &self.ddw
    }

    /// The `3 x natm x 3 x B` Hessian block of one selected atom, or `None`.
    ///
    /// # Arguments
    ///
    /// * `atom` — global atom index; `None` when it is not selected.
    ///
    /// # Returns
    ///
    /// `[3, natm, 3, B]` row-major
    /// (`((t * natm + B_atom) * 3 + s) * B + point`).
    #[must_use]
    pub fn ddw_atom(&self, atom: usize) -> Option<&[f64]> {
        if !self.ddw_requested {
            return None;
        }
        let slot = self.atoms.iter().position(|&a| a == atom)?;
        let len = self.range.len();
        let stride = 3 * self.ncols * 3 * len;
        Some(&self.ddw[slot * stride..(slot + 1) * stride])
    }

    /// Accumulated contracted gradient, if [`DerivRequest::dc`] was set.
    #[must_use]
    pub fn dc(&self) -> Option<&GradientContract> {
        self.dc.as_ref()
    }

    /// Accumulated contracted Hessian, if [`DerivRequest::ddc`] was set.
    #[must_use]
    pub fn ddc(&self) -> Option<&HessianContract> {
        self.ddc.as_ref()
    }
}
