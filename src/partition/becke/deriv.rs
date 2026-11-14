//! [`DerivScheme`] implementation for the exact classic Becke partition.
//!
//! This is a thin adapter over [`super::gradient`] and [`super::hessian`]; the
//! evaluator-facing layouts are documented on [`DerivScheme`].

use std::any::TypeId;

use super::DerivTables;
use super::gradient;
use super::hessian;
use super::hessian_lane;
use crate::error::Result;
use crate::partition::PartitionConfig;
use crate::partition::deriv::{DerivScheme, DerivScratch, HessianBackend};

/// Per-worker Hessian scratch: scalar for the general path, fixed-lane where
/// its eight-point working set stays small enough (see the lane limits
/// below).
enum PointScratch {
    Scalar(hessian::Scratch),
    Lane(hessian_lane::Scratch),
}

/// AVX-512 lane scratch limit (cut chosen empirically; see `AVX2_LANE_LIMIT`).
const AVX512_LANE_LIMIT: usize = 4 << 20;

/// AVX2 lane scratch limit (chosen empirically; larger than the AVX-512
/// limit because the two-register lowering tolerates a larger working set).
const AVX2_LANE_LIMIT: usize = 16 << 20;

fn use_lane_hessian(n: usize, nsel: usize, backend: HessianBackend) -> bool {
    if backend == HessianBackend::Pointwise {
        return false;
    }
    if backend == HessianBackend::Lane {
        return true;
    }
    let footprint = hessian_lane::Scratch::base_estimated_bytes(n, nsel);
    if cfg!(target_feature = "avx512f") {
        return footprint <= AVX512_LANE_LIMIT;
    }
    if !cfg!(target_feature = "avx2") {
        return false;
    }
    footprint <= AVX2_LANE_LIMIT
}

/// Becke derivative backend: molecule tables plus the selected row map.
#[derive(Debug)]
pub(crate) struct BeckeDeriv {
    config: PartitionConfig,
    tables: DerivTables,
    sel: Vec<usize>,
    slot_of: Vec<usize>,
    hessian_backend: HessianBackend,
}

impl BeckeDeriv {
    /// Build the pair tables and the global-atom -> output-slot map.
    ///
    /// # Arguments
    ///
    /// * `config` — partition configuration (exact classic Becke).
    /// * `centers` — `[natm, 3]` row-major nuclear coordinates in Bohr.
    /// * `charges` — `[natm]` atomic numbers.
    /// * `sel` — `[nsel]` strictly increasing selected rows.
    ///
    /// # Errors
    /// Propagates [`DerivTables::new`]'s length/radii errors.
    pub(crate) fn new(
        config: PartitionConfig,
        centers: &[[f64; 3]],
        charges: &[u32],
        sel: Vec<usize>,
        hessian_backend: HessianBackend,
    ) -> Result<Self> {
        let tables = DerivTables::new(&config, centers, charges)?;
        let n = tables.len();
        let mut slot_of = vec![usize::MAX; n];
        for (slot, &atom) in sel.iter().enumerate() {
            slot_of[atom] = slot;
        }
        Ok(Self {
            config,
            tables,
            sel,
            slot_of,
            hessian_backend,
        })
    }

    #[inline]
    fn block_scratch(scratch: &mut dyn DerivScratch) -> &mut gradient::Scratch {
        scratch
            .as_any_mut()
            .downcast_mut()
            .expect("Becke block scratch type mismatch")
    }

    #[inline]
    fn point_scratch(scratch: &mut dyn DerivScratch) -> &mut PointScratch {
        scratch
            .as_any_mut()
            .downcast_mut()
            .expect("Becke point scratch type mismatch")
    }

    #[allow(clippy::too_many_arguments)]
    fn scalar_hessian_block(
        &self,
        coords: &[[f64; 3]],
        parents: &[u32],
        quad: &[f64],
        weights: &mut [f64],
        mut grad: Option<&mut [f64]>,
        ddw: &mut [f64],
        scratch: &mut hessian::Scratch,
    ) {
        let b = coords.len();
        let n3 = 3 * self.tables.len();
        let row_len = self.sel.len() * 3;
        let hess_len = row_len * n3;
        for idx in 0..b {
            let grad_slice = grad
                .as_deref_mut()
                .map(|g| &mut g[idx * row_len..(idx + 1) * row_len]);
            let weight = hessian::point_hessian(
                &self.config,
                &self.tables,
                coords[idx],
                parents[idx] as usize,
                &self.sel,
                &self.slot_of,
                quad[idx],
                grad_slice,
                scratch,
            );
            weights[idx] = weight;
            ddw[idx * hess_len..(idx + 1) * hess_len].copy_from_slice(&scratch.out);
        }
    }
}

impl DerivScheme for BeckeDeriv {
    fn atom_count(&self) -> usize {
        self.tables.len()
    }

    fn selection(&self) -> &[usize] {
        &self.sel
    }

    fn block_scratch_kind(&self) -> TypeId {
        TypeId::of::<gradient::Scratch>()
    }

    fn point_scratch_kind(&self) -> TypeId {
        TypeId::of::<PointScratch>()
    }

    fn new_block_scratch(&self, block: usize) -> Box<dyn DerivScratch> {
        Box::new(gradient::Scratch::new(
            self.tables.len(),
            self.sel.len(),
            block,
        ))
    }

    fn hessian_scratch_bytes(&self) -> usize {
        let n = self.tables.len();
        let nsel = self.sel.len();
        if use_lane_hessian(n, nsel, self.hessian_backend) {
            hessian_lane::Scratch::estimated_bytes(n, nsel)
        } else {
            hessian::Scratch::estimated_bytes(n, nsel)
        }
    }

    fn hessian_pointwise(&self) -> bool {
        // The eight-point staging buffers only pay off for the fixed-lane
        // kernel.  The scalar fallback streams points directly and keeps the
        // per-worker working set small.
        !use_lane_hessian(self.tables.len(), self.sel.len(), self.hessian_backend)
    }

    fn new_point_scratch(&self) -> Box<dyn DerivScratch> {
        let n = self.tables.len();
        let nsel = self.sel.len();
        if use_lane_hessian(n, nsel, self.hessian_backend) {
            Box::new(PointScratch::Lane(hessian_lane::Scratch::new(n, nsel)))
        } else {
            Box::new(PointScratch::Scalar(hessian::Scratch::new(n, nsel)))
        }
    }

    fn gradient_block(
        &self,
        coords: &[[f64; 3]],
        parents: &[u32],
        quad: &[f64],
        weights: &mut [f64],
        dw: &mut [f64],
        scratch: &mut dyn DerivScratch,
    ) {
        let scratch = Self::block_scratch(scratch);
        gradient::block_gradient(
            &self.config,
            &self.tables,
            &self.sel,
            &self.slot_of,
            coords,
            parents,
            quad,
            weights,
            dw,
            scratch,
        );
    }

    fn hessian_block(
        &self,
        coords: &[[f64; 3]],
        parents: &[u32],
        quad: &[f64],
        weights: &mut [f64],
        grad: Option<&mut [f64]>,
        ddw: &mut [f64],
        scratch: &mut dyn DerivScratch,
    ) {
        let scratch = Self::point_scratch(scratch);
        match scratch {
            PointScratch::Lane(hessian_lane::Scratch { core, .. }) => {
                hessian_lane::block_hessian(
                    &self.config,
                    &self.tables,
                    coords,
                    parents,
                    quad,
                    &self.sel,
                    &self.slot_of,
                    weights,
                    grad,
                    ddw,
                    core,
                );
            }
            PointScratch::Scalar(scratch) => {
                self.scalar_hessian_block(coords, parents, quad, weights, grad, ddw, scratch);
            }
        }
    }

    fn point_hessian<'s>(
        &self,
        point: [f64; 3],
        parent: usize,
        quad: f64,
        grad: Option<&mut [f64]>,
        scratch: &'s mut dyn DerivScratch,
    ) -> (f64, &'s [f64]) {
        let scratch = Self::point_scratch(scratch);
        match scratch {
            PointScratch::Scalar(scratch) => {
                let weight = hessian::point_hessian(
                    &self.config,
                    &self.tables,
                    point,
                    parent,
                    &self.sel,
                    &self.slot_of,
                    quad,
                    grad,
                    scratch,
                );
                (weight, &scratch.out)
            }
            PointScratch::Lane(hessian_lane::Scratch { core, out }) => {
                let n3 = 3 * self.tables.len();
                let out_len = self.sel.len() * 3 * n3;
                if out.len() != out_len {
                    out.resize(out_len, 0.0);
                }
                let mut weights = [0.0];
                hessian_lane::block_hessian(
                    &self.config,
                    &self.tables,
                    &[point],
                    &[parent as u32],
                    &[quad],
                    &self.sel,
                    &self.slot_of,
                    &mut weights,
                    grad,
                    out.as_mut_slice(),
                    core,
                );
                (weights[0], out.as_slice())
            }
        }
    }
}
