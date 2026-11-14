//! [`DerivScheme`] implementation for the LKO partition.
//!
//! This is the [`crate::partition::becke::deriv::BeckeDeriv`] adapter with
//! the LKO pair geometry.  The gradient dispatches between the dense
//! eight-point lane kernel of [`super::gradient`] and the masked union-list
//! kernel of [`super::gradient_sparse`] above the size threshold (chosen
//! empirically).  The
//! Hessian is pointwise and dispatches between the dense kernel of
//! [`super::hessian`] (small molecules, small overhead) and the sparse
//! near-set kernel of [`super::hessian_sparse`].  Both pairs of kernels
//! produce the same values up to signed zero.

use std::any::TypeId;

use super::gradient;
use super::gradient_sparse;
use super::hessian;
use super::hessian_sparse;
use super::tables::DerivTables;
use super::use_sparse_lko_gradient;
use super::use_sparse_lko_hessian;
use crate::error::Result;
use crate::partition::PartitionConfig;
use crate::partition::deriv::{DerivScheme, DerivScratch};

/// Per-thread block gradient scratch, selected by the sparse/dense crossover.
#[derive(Debug)]
enum BlockScratch {
    Dense(Box<gradient::Scratch>),
    Sparse(Box<gradient_sparse::Scratch>),
}

/// Per-thread point Hessian scratch, selected by the sparse/dense crossover.
#[derive(Debug)]
enum PointScratch {
    Dense(hessian::Scratch),
    Sparse(hessian_sparse::Scratch),
}

/// LKO derivative backend: molecule tables plus the selected row map.
#[derive(Debug)]
pub(crate) struct LkoDeriv {
    config: PartitionConfig,
    tables: DerivTables,
    sel: Vec<usize>,
    slot_of: Vec<usize>,
}

impl LkoDeriv {
    /// Build the pair tables and the global-atom -> output-slot map.
    ///
    /// # Errors
    /// Propagates [`DerivTables::new`]'s length/radii errors.
    pub(crate) fn new(
        config: PartitionConfig,
        centers: &[[f64; 3]],
        charges: &[u32],
        sel: Vec<usize>,
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
        })
    }
}

impl DerivScheme for LkoDeriv {
    fn atom_count(&self) -> usize {
        self.tables.len()
    }

    fn selection(&self) -> &[usize] {
        &self.sel
    }

    fn block_scratch_kind(&self) -> TypeId {
        TypeId::of::<BlockScratch>()
    }

    fn point_scratch_kind(&self) -> TypeId {
        TypeId::of::<PointScratch>()
    }

    fn new_block_scratch(&self, block: usize) -> Box<dyn DerivScratch> {
        let n = self.tables.len();
        let nsel = self.sel.len();
        Box::new(if use_sparse_lko_gradient(n) {
            BlockScratch::Sparse(Box::new(gradient_sparse::Scratch::new(n, nsel, block)))
        } else {
            BlockScratch::Dense(Box::new(gradient::Scratch::new(n, nsel, block)))
        })
    }

    fn new_point_scratch(&self) -> Box<dyn DerivScratch> {
        let n = self.tables.len();
        let nsel = self.sel.len();
        Box::new(if use_sparse_lko_hessian(n) {
            PointScratch::Sparse(hessian_sparse::Scratch::new(n, nsel))
        } else {
            PointScratch::Dense(hessian::Scratch::new(n, nsel))
        })
    }

    fn hessian_scratch_bytes(&self) -> usize {
        let n = self.tables.len();
        let nsel = self.sel.len();
        if use_sparse_lko_hessian(n) {
            hessian_sparse::Scratch::estimated_bytes(n, nsel)
        } else {
            hessian::Scratch::estimated_bytes(n, nsel)
        }
    }

    fn hessian_pointwise(&self) -> bool {
        true
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
        let scratch = scratch
            .as_any_mut()
            .downcast_mut::<BlockScratch>()
            .expect("LKO block scratch type mismatch");
        match scratch {
            BlockScratch::Dense(scratch) => gradient::block_gradient(
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
            ),
            BlockScratch::Sparse(scratch) => gradient_sparse::block_gradient(
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
            ),
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
        let scratch = scratch
            .as_any_mut()
            .downcast_mut::<PointScratch>()
            .expect("LKO point scratch type mismatch");
        match scratch {
            PointScratch::Dense(scratch) => {
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
            PointScratch::Sparse(scratch) => {
                let weight = hessian_sparse::point_hessian(
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
        }
    }
}
