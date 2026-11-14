//! Per-point access for derivative evaluation.
//!
//! The derivative kernels are point-local: they take a global coordinate, the
//! generating atom and the quadrature weight of every point, and never reduce
//! across points.  [`DerivSource`] exposes exactly those three values for the
//! two point stores the crate has:
//!
//! * [`GridPlan`] plus its nuclear centres (translating the stored local
//!   coordinates), for streaming without a materialised grid;
//! * [`MolecularGrid`], whose coordinates are already global and whose point
//!   order may be arbitrary (post-pruned or reordered).
//!
//! Neither variant assumes atom-contiguous points.

use crate::grid::GridPlan;
use crate::moleculegrid::MolecularGrid;

/// Point store viewed by the streaming derivative evaluator.
#[derive(Clone, Copy)]
pub(crate) enum DerivSource<'a> {
    /// Topology plan and the centres used to translate its local coordinates.
    Plan {
        /// Point topology.
        plan: &'a GridPlan,
        /// `[natm, 3]` nuclear coordinates in Bohr.
        centers: &'a [[f64; 3]],
    },
    /// Fully materialised grid; coordinates are global and order is arbitrary.
    Grid(&'a MolecularGrid),
}

impl<'a> DerivSource<'a> {
    /// Global coordinate, generating atom and quadrature weight of `index`.
    pub(crate) fn point(self, index: usize) -> ([f64; 3], u32, f64) {
        match self {
            Self::Plan { plan, centers } => {
                let parent = plan.parent_of(index) as usize;
                let local = plan.local_coord(index);
                let c = centers[parent];
                (
                    [c[0] + local[0], c[1] + local[1], c[2] + local[2]],
                    parent as u32,
                    plan.quad_weight(index),
                )
            }
            Self::Grid(grid) => (
                grid.coords()[index],
                grid.atom_idx()[index],
                grid.quad_weights()[index],
            ),
        }
    }
}
