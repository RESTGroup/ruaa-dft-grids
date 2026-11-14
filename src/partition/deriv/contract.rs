//! Contraction tensors and their accumulated outputs.
//!
//! The per-point coefficients are passed as [`DerivContract`], a row-major
//! `[nset, cols]` view.  `dc` and `ddc` can use different views, with
//! different `nset`, in the same evaluation run.

use std::ops::Range;

use crate::error::{Error, Result};

/// Per-point contraction coefficients for one output family.
///
/// # Layout (row-major)
///
/// `data` is `[nset, cols]`, flattened as `set * cols + column`.  A range of
/// `B` points starting at `offset` uses columns
/// `offset .. offset + B`; set `s` therefore reads
/// `data[s * cols + offset + point]`.
///
/// The `cols` dimension is the number of columns of the **underlying batch**;
/// it does not have to equal the current `range.len()`.  This lets a caller
/// keep one `[nset, nbatched-grids]` tensor and stream sub-ranges of it
/// without copying.  Use [`Self::exact`] when the tensor covers exactly the
/// range, or [`Self::window`] to select a sub-range.
#[derive(Clone, Copy, Debug)]
pub struct DerivContract<'a> {
    data: &'a [f64],
    nset: usize,
    cols: usize,
    offset: usize,
}

impl<'a> DerivContract<'a> {
    /// A tensor of shape `[nset, cols]`.
    ///
    /// # Arguments
    ///
    /// * `data` — `[nset, cols]` row-major coefficients.
    /// * `nset` — number of contraction sets (`>= 1`).
    /// * `cols` — number of point columns.
    ///
    /// # Panics
    /// If `nset == 0` or `data.len() != nset * cols`.
    #[must_use]
    pub fn new(data: &'a [f64], nset: usize, cols: usize) -> Self {
        assert!(nset > 0, "DerivContract: nset must be positive");
        assert_eq!(
            data.len(),
            nset * cols,
            "DerivContract: data length must be nset * cols"
        );
        Self {
            data,
            nset,
            cols,
            offset: 0,
        }
    }

    /// A tensor whose column count is inferred as `data.len() / nset`.
    ///
    /// This is the `[nset, B]` form most callers want for one range.
    ///
    /// # Arguments
    ///
    /// * `data` — `[nset, B]` row-major coefficients.
    /// * `nset` — number of contraction sets (`>= 1`); `data.len()` must be a
    ///   multiple of it.
    ///
    /// # Panics
    /// If `nset == 0` or `data.len() % nset != 0`.
    #[must_use]
    pub fn exact(data: &'a [f64], nset: usize) -> Self {
        assert!(nset > 0, "DerivContract: nset must be positive");
        assert_eq!(
            data.len() % nset,
            0,
            "DerivContract: data length must be a multiple of nset"
        );
        Self {
            data,
            nset,
            cols: data.len() / nset,
            offset: 0,
        }
    }

    /// A `[nset, cols]` tensor restricted to columns
    /// `offset .. offset + B`.
    ///
    /// # Arguments
    ///
    /// * `data` — `[nset, cols]` row-major coefficients.
    /// * `nset` — number of contraction sets (`>= 1`).
    /// * `cols` — number of columns in `data`.
    /// * `offset` — first column used by the next evaluation range.
    ///
    /// # Panics
    /// If `nset == 0`, `data.len() != nset * cols`, or `offset > cols`.
    #[must_use]
    pub fn window(data: &'a [f64], nset: usize, cols: usize, offset: usize) -> Self {
        assert!(nset > 0, "DerivContract: nset must be positive");
        assert_eq!(
            data.len(),
            nset * cols,
            "DerivContract: data length must be nset * cols"
        );
        assert!(offset <= cols, "DerivContract: offset exceeds cols");
        Self {
            data,
            nset,
            cols,
            offset,
        }
    }

    /// A `[nset, cols]` tensor whose first used column is `range.start`.
    ///
    /// Use this for a contract that covers a global grid and is evaluated
    /// over a sub-range.
    #[must_use]
    pub fn over(data: &'a [f64], nset: usize, cols: usize, range: Range<usize>) -> Self {
        Self::window(data, nset, cols, range.start)
    }

    /// Checked form of [`Self::new`].
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] for `nset == 0`, or
    /// [`Error::LengthMismatch`] when `data.len() != nset * cols`.
    pub fn try_new(data: &'a [f64], nset: usize, cols: usize) -> Result<Self> {
        if nset == 0 {
            return Err(Error::InvalidArgument {
                argument: "DerivContract nset",
            });
        }
        if data.len() != nset * cols {
            return Err(Error::LengthMismatch {
                expected: nset * cols,
                got: data.len(),
            });
        }
        Ok(Self {
            data,
            nset,
            cols,
            offset: 0,
        })
    }

    /// Checked form of [`Self::exact`].
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] for `nset == 0`, or
    /// [`Error::LengthMismatch`] when `data.len()` is not a multiple of `nset`.
    pub fn try_exact(data: &'a [f64], nset: usize) -> Result<Self> {
        if nset == 0 {
            return Err(Error::InvalidArgument {
                argument: "DerivContract nset",
            });
        }
        if !data.len().is_multiple_of(nset) {
            return Err(Error::LengthMismatch {
                expected: data.len().div_ceil(nset) * nset,
                got: data.len(),
            });
        }
        Ok(Self {
            data,
            nset,
            cols: data.len() / nset,
            offset: 0,
        })
    }

    /// Checked form of [`Self::window`].
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] for `nset == 0` or `offset > cols`, or
    /// [`Error::LengthMismatch`] when `data.len() != nset * cols`.
    pub fn try_window(data: &'a [f64], nset: usize, cols: usize, offset: usize) -> Result<Self> {
        if nset == 0 {
            return Err(Error::InvalidArgument {
                argument: "DerivContract nset",
            });
        }
        if data.len() != nset * cols {
            return Err(Error::LengthMismatch {
                expected: nset * cols,
                got: data.len(),
            });
        }
        if offset > cols {
            return Err(Error::InvalidArgument {
                argument: "DerivContract offset",
            });
        }
        Ok(Self {
            data,
            nset,
            cols,
            offset,
        })
    }

    /// Number of contraction sets.
    #[must_use]
    pub fn nset(&self) -> usize {
        self.nset
    }

    /// Number of columns of the underlying tensor.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// First column used by the next range.
    #[must_use]
    pub fn offset(&self) -> usize {
        self.offset
    }

    pub(crate) fn check_range(&self, len: usize) -> Result<()> {
        let Some(need) = self.offset.checked_add(len) else {
            return Err(Error::LengthMismatch {
                expected: usize::MAX,
                got: self.cols,
            });
        };
        if need > self.cols {
            return Err(Error::LengthMismatch {
                expected: need,
                got: self.cols,
            });
        }
        Ok(())
    }

    #[inline]
    fn coeff(&self, set: usize, column: usize) -> f64 {
        self.data[set * self.cols + column]
    }
}

/// Accumulated contracted gradient `dc`.
///
/// # Layout (row-major)
///
/// `dc` is `[nset, nsel, 3]`, flattened as
/// `(set * nsel + slot) * 3 + t`: the derivative of
/// `sum_g c[set, g] q_g Pbar_g / Z_g` with respect to the selected row atom
/// `A = atoms[slot]` and component `t`.
///
/// The contract is **accumulated** across ranges; call [`Self::zero`] before
/// the first range of a stream.
#[derive(Clone, Debug)]
pub struct GradientContract {
    atoms: Vec<usize>,
    nset: usize,
    dc: Vec<f64>,
}

impl GradientContract {
    /// A zeroed contract for `atoms` and `nset` sets.
    ///
    /// # Arguments
    ///
    /// * `atoms` — `[nsel]` selected global atom indices (rows of `dc`).
    /// * `nset` — number of contraction sets (`>= 1`).
    ///
    /// # Panics
    /// If `nset == 0`.
    #[must_use]
    pub(crate) fn new(atoms: &[usize], nset: usize) -> Self {
        assert!(nset > 0, "GradientContract: nset must be positive");
        Self {
            atoms: atoms.to_vec(),
            nset,
            dc: vec![0.0; nset * atoms.len() * 3],
        }
    }

    /// Set every accumulated entry to zero.
    pub fn zero(&mut self) {
        self.dc.fill(0.0);
    }

    /// Selected row atoms, shape `[nsel]`.
    #[must_use]
    pub fn atoms(&self) -> &[usize] {
        &self.atoms
    }

    /// Number of contraction sets.
    #[must_use]
    pub fn nset(&self) -> usize {
        self.nset
    }

    /// Row-major `[nset, nsel, 3]` accumulated gradient.
    #[must_use]
    pub fn dc(&self) -> &[f64] {
        &self.dc
    }

    /// One set of the accumulated gradient, shape `[nsel, 3]`, flattened
    /// `slot * 3 + t`.
    ///
    /// # Arguments
    ///
    /// * `set` — contraction set in `0..nset`.
    ///
    /// # Panics
    /// If `set >= nset`.
    #[must_use]
    pub fn dc_set(&self, set: usize) -> &[f64] {
        let stride = self.atoms.len() * 3;
        &self.dc[set * stride..(set + 1) * stride]
    }

    pub(crate) fn add_strided(
        &mut self,
        contract: &DerivContract<'_>,
        column: usize,
        block: &[f64],
        stride: usize,
        points: usize,
    ) {
        let n = self.atoms.len() * 3;
        debug_assert_eq!(self.nset, contract.nset());
        debug_assert!(column + points <= contract.cols());
        debug_assert!(n * stride <= block.len());
        for set in 0..self.nset {
            let cbase = set * contract.cols() + column;
            let obase = set * n;
            for row in 0..n {
                let mut acc = 0.0;
                let src = &block[row * stride..row * stride + points];
                for (p, &coef) in contract.data[cbase..cbase + points].iter().enumerate() {
                    acc += coef * src[p];
                }
                self.dc[obase + row] += acc;
            }
        }
    }

    pub(crate) fn add_point(
        &mut self,
        contract: &DerivContract<'_>,
        column: usize,
        values: &[f64],
    ) {
        let n = self.atoms.len() * 3;
        debug_assert_eq!(values.len(), n);
        for set in 0..self.nset {
            let coef = contract.coeff(set, column);
            if coef == 0.0 {
                continue;
            }
            let base = set * n;
            for (dst, &src) in self.dc[base..base + n].iter_mut().zip(values) {
                *dst += coef * src;
            }
        }
    }

    pub(crate) fn add(&mut self, other: &Self) {
        debug_assert_eq!(self.nset, other.nset);
        for (dst, &src) in self.dc.iter_mut().zip(&other.dc) {
            *dst += src;
        }
    }
}

/// Accumulated contracted Hessian `ddc`.
///
/// # Layout (row-major)
///
/// `ddc` is `[nset, nsel, 3, ncols, 3]`, flattened as
/// `(((set * nsel + slot) * 3 + t) * ncols + B) * 3 + s`: the second
/// derivative of `sum_g c[set, g] q_g Pbar_g / Z_g` with respect to the
/// selected row atom `A = atoms[slot]` (component `t`) and the column atom
/// `B` (component `s`).
///
/// The contract is **accumulated** across ranges; call [`Self::zero`] before
/// the first range of a stream.
#[derive(Clone, Debug)]
pub struct HessianContract {
    atoms: Vec<usize>,
    nset: usize,
    ncols: usize,
    ddc: Vec<f64>,
}

impl HessianContract {
    /// A zeroed contract for `atoms`, `nset` sets and `ncols` column atoms.
    ///
    /// # Arguments
    ///
    /// * `atoms` — `[nsel]` selected global atom indices (rows of `ddc`).
    /// * `nset` — number of contraction sets (`>= 1`).
    /// * `ncols` — number of column atoms, normally `natm`.
    ///
    /// # Panics
    /// If `nset == 0`.
    #[must_use]
    pub(crate) fn new(atoms: &[usize], nset: usize, ncols: usize) -> Self {
        assert!(nset > 0, "HessianContract: nset must be positive");
        Self {
            atoms: atoms.to_vec(),
            nset,
            ncols,
            ddc: vec![0.0; nset * atoms.len() * 3 * ncols * 3],
        }
    }

    /// Set every accumulated entry to zero.
    pub fn zero(&mut self) {
        self.ddc.fill(0.0);
    }

    /// Selected row atoms, shape `[nsel]`.
    #[must_use]
    pub fn atoms(&self) -> &[usize] {
        &self.atoms
    }

    /// Number of contraction sets.
    #[must_use]
    pub fn nset(&self) -> usize {
        self.nset
    }

    /// Number of column atoms (`ddc` fourth dimension).
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Row-major `[nset, nsel, 3, ncols, 3]` accumulated Hessian.
    #[must_use]
    pub fn ddc(&self) -> &[f64] {
        &self.ddc
    }

    /// One set of the accumulated Hessian, shape `[nsel, 3, ncols, 3]`,
    /// flattened `((slot * 3 + t) * ncols + B) * 3 + s`.
    ///
    /// # Arguments
    ///
    /// * `set` — contraction set in `0..nset`.
    ///
    /// # Panics
    /// If `set >= nset`.
    #[must_use]
    pub fn ddc_set(&self, set: usize) -> &[f64] {
        let stride = self.atoms.len() * 3 * self.ncols * 3;
        &self.ddc[set * stride..(set + 1) * stride]
    }

    pub(crate) fn add_point(
        &mut self,
        contract: &DerivContract<'_>,
        column: usize,
        values: &[f64],
    ) {
        let n = self.atoms.len() * 3 * self.ncols * 3;
        debug_assert_eq!(values.len(), n);
        for set in 0..self.nset {
            let coef = contract.coeff(set, column);
            if coef == 0.0 {
                continue;
            }
            let base = set * n;
            for (dst, &src) in self.ddc[base..base + n].iter_mut().zip(values) {
                *dst += coef * src;
            }
        }
    }

    pub(crate) fn add(&mut self, other: &Self) {
        debug_assert_eq!(self.nset, other.nset);
        debug_assert_eq!(self.ncols, other.ncols);
        for (dst, &src) in self.ddc.iter_mut().zip(&other.ddc) {
            *dst += src;
        }
    }
}
