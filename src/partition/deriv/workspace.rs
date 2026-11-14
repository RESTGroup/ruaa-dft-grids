//! Reusable per-thread derivative workspace.
//!
//! One [`DerivWorkspace`] serves both evaluation modes: a gradient block and
//! a per-point Hessian (which can also emit the gradient).  The scratch is
//! allocated lazily and reused across ranges; switching schemes or block
//! sizes reallocates only the affected slot.

use std::any::TypeId;

use super::scheme::{DerivScheme, DerivScratch};

/// Cached scratch plus the parameters it was built for.
pub(crate) struct ScratchSlot {
    kind: TypeId,
    natm: usize,
    nsel: usize,
    block: usize,
    scratch: Box<dyn DerivScratch>,
}

impl std::fmt::Debug for ScratchSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScratchSlot")
            .field("natm", &self.natm)
            .field("nsel", &self.nsel)
            .field("block", &self.block)
            .finish_non_exhaustive()
    }
}

/// Reusable per-thread scratch for [`PartitionDerivEvaluator`].
///
/// Use one workspace per thread.  It contains both the blocked-gradient and
/// the per-point-Hessian scratch, so a combined gradient+Hessian run does not
/// need a second workspace.  Buffers grow to the largest range/block seen and
/// are not deallocated between calls.
///
/// [`PartitionDerivEvaluator`]: super::PartitionDerivEvaluator
#[derive(Default)]
pub struct DerivWorkspace {
    pub(crate) block_scratch: Option<ScratchSlot>,
    pub(crate) point_scratch: Option<ScratchSlot>,
    pub(crate) coords: Vec<[f64; 3]>,
    pub(crate) parents: Vec<u32>,
    pub(crate) quad: Vec<f64>,
    pub(crate) weights_block: Vec<f64>,
    pub(crate) dw_block: Vec<f64>,
    pub(crate) grad_point: Vec<f64>,
    pub(crate) grad_block: Vec<f64>,
    pub(crate) ddw_block: Vec<f64>,
}

impl std::fmt::Debug for DerivWorkspace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DerivWorkspace")
            .field("block_scratch", &self.block_scratch)
            .field("point_scratch", &self.point_scratch)
            .field("coords_capacity", &self.coords.capacity())
            .field("dw_block_capacity", &self.dw_block.capacity())
            .field("grad_point_capacity", &self.grad_point.capacity())
            .field("grad_block_capacity", &self.grad_block.capacity())
            .field("ddw_block_capacity", &self.ddw_block.capacity())
            .finish_non_exhaustive()
    }
}

impl DerivWorkspace {
    /// Empty workspace.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Block scratch for `scheme`, sized for `block` points.
    pub(crate) fn block_scratch(
        &mut self,
        scheme: &dyn DerivScheme,
        block: usize,
    ) -> &mut dyn DerivScratch {
        ensure_block(&mut self.block_scratch, scheme, block)
    }

    /// Point scratch plus the reusable `[nsel, 3]` gradient output buffer.
    ///
    /// The two borrows are disjoint, so a caller can pass
    /// `grad = Some(grad_buf)` and still obtain the Hessian from the returned
    /// scratch.
    pub(crate) fn point_parts<'s>(
        &'s mut self,
        scheme: &dyn DerivScheme,
    ) -> (&'s mut dyn DerivScratch, &'s mut [f64]) {
        let nsel = scheme.selection().len();
        if self.grad_point.len() != nsel * 3 {
            self.grad_point.resize(nsel * 3, 0.0);
        }
        let Self {
            point_scratch,
            grad_point,
            ..
        } = self;
        let scratch = ensure_point(point_scratch, scheme);
        (scratch, grad_point.as_mut_slice())
    }
}

pub(crate) fn ensure_block<'s>(
    slot: &'s mut Option<ScratchSlot>,
    scheme: &dyn DerivScheme,
    block: usize,
) -> &'s mut dyn DerivScratch {
    let kind = scheme.block_scratch_kind();
    let natm = scheme.atom_count();
    let nsel = scheme.selection().len();
    let rebuild = match slot {
        Some(cached) => {
            cached.kind != kind
                || cached.natm != natm
                || cached.nsel != nsel
                || cached.block != block
        }
        None => true,
    };
    if rebuild {
        *slot = Some(ScratchSlot {
            kind,
            natm,
            nsel,
            block,
            scratch: scheme.new_block_scratch(block),
        });
    }
    slot.as_mut()
        .expect("block scratch was just built")
        .scratch
        .as_mut()
}

pub(crate) fn ensure_point<'s>(
    slot: &'s mut Option<ScratchSlot>,
    scheme: &dyn DerivScheme,
) -> &'s mut dyn DerivScratch {
    let kind = scheme.point_scratch_kind();
    let natm = scheme.atom_count();
    let nsel = scheme.selection().len();
    let rebuild = match slot {
        Some(cached) => cached.kind != kind || cached.natm != natm || cached.nsel != nsel,
        None => true,
    };
    if rebuild {
        *slot = Some(ScratchSlot {
            kind,
            natm,
            nsel,
            block: 1,
            scratch: scheme.new_point_scratch(),
        });
    }
    slot.as_mut()
        .expect("point scratch was just built")
        .scratch
        .as_mut()
}
