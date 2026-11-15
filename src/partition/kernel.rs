//! Shared evaluation machinery for the partition kernels.
//!
//! The molecular data that does not depend on the grid point (atomic radii,
//! size-adjustment factors, interatomic distances) lives in
//! [`PartitionTables`](super::PartitionTables).  This module owns the
//! per-thread scratch and the block layout used by the grid kernels:
//!
//! * `dist` / `prod` are `[center * block + point]` tiles, so one block of
//!   grid points is evaluated with the atom-pair loop on the outside and the
//!   point loop on the inside (a blocked partition-kernel layout);
//! * `dmin` holds the nearest-center distance of each point (LKO screening);
//! * `p` / `order` are the per-point working set of the nearest-first mBecke
//!   kernel.
//!
//! The tile keeps the per-thread scratch bounded (`2 * SCRATCH_ELEMS * 8 B`
//! here) independently of the molecule size.

use super::PartitionTables;
use super::simd::{self, V};

/// Target size (in `f64` elements) of one scratch plane per worker.
///
/// The blocked weight kernel revisits the `dist`/`prod` tile for every atom
/// pair; the tile is `2 * SCRATCH_ELEMS * 8 B` in total.  Keeping it near
/// 16 KB keeps the tile cache-sized while allowing a long inner loop. See the
/// [performance notes](crate::doc::performance).
const SCRATCH_ELEMS: usize = 1 << 10;
/// Lower bound of the block length; one fixed lane of eight doubles.
const BLOCK_MIN: usize = simd::LANES;
/// Upper bound of the block length; keeps the scratch within the target size.
const BLOCK_MAX: usize = 1024;

/// Length of the grid block processed in one pass of a blocked kernel.
///
/// The result is a multiple of the fixed lane width so the classic kernel can
/// use whole `f64x8` lanes.  Rounding to the nearest lane keeps the tile near
/// the 16 KB target instead of leaving up to seven lanes unused.
#[must_use]
pub(crate) fn block_length(num_centers: usize) -> usize {
    let raw = (SCRATCH_ELEMS / num_centers.max(1)).clamp(BLOCK_MIN, BLOCK_MAX);
    let lanes = simd::LANES;
    ((raw + lanes / 2) / lanes * lanes).clamp(BLOCK_MIN, BLOCK_MAX)
}

/// Normalize the per-point cell products of a single-point evaluation into
/// `out` (all centers).
pub(crate) fn finish_all(prod: &[f64], out: &mut [f64]) {
    out.copy_from_slice(prod);
    super::normalize(out);
}

/// Per-worker scratch for the fixed-array (`f64x8`) pair-product kernels.
///
/// Shared by the classic Becke kernels and the dense SSF fallback: both keep
/// the same `[center * nlane + lane]` distance/product tile and update both
/// members of a pair in one lane pass.
#[derive(Debug, Clone)]
pub(crate) struct LaneScratch {
    /// Block length in grid points.
    pub(crate) block: usize,
    /// Number of eight-point lanes in one block.
    pub(crate) nlane: usize,
    /// `[center * nlane + lane]` grid-to-centre distances.
    pub(crate) dist: Vec<V>,
    /// `[center * nlane + lane]` running cell products.
    pub(crate) prod: Vec<V>,
    /// `[lane]` ORCA early-exit activity.
    pub(crate) active: Vec<V>,
    /// `[lane]` gathered coordinates.
    pub(crate) coords: Vec<[V; 3]>,
}

impl LaneScratch {
    pub(crate) fn new(num_centers: usize, block: usize) -> Self {
        let nlane = block.div_ceil(simd::LANES).max(1);
        Self {
            block,
            nlane,
            dist: vec![V::zero(); num_centers * nlane],
            prod: vec![V::zero(); num_centers * nlane],
            active: vec![V::zero(); nlane],
            coords: vec![[V::zero(); 3]; nlane],
        }
    }
}

/// Fill the lane distance tile and reset the product tile.
#[inline(always)]
pub(crate) fn init_lane_block(
    tables: &PartitionTables,
    grid: &[[f64; 3]],
    out: &mut [f64],
    scratch: &mut LaneScratch,
) -> (usize, usize, usize) {
    let n = tables.len();
    let nb = out.len();
    let nlane = scratch.nlane;
    let active_lanes = nb.div_ceil(simd::LANES);
    debug_assert!(nb <= scratch.block);
    for (lane, coord) in scratch.coords.iter_mut().take(active_lanes).enumerate() {
        *coord = simd::gather_coords(grid, lane);
    }
    for k in 0..n {
        let c = tables.center(k);
        let cx = simd::splat(c[0]);
        let cy = simd::splat(c[1]);
        let cz = simd::splat(c[2]);
        for lane in 0..active_lanes {
            let p = scratch.coords[lane];
            let dx = p[0] - cx;
            let dy = p[1] - cy;
            let dz = p[2] - cz;
            scratch.dist[k * nlane + lane] = (dx * dx + dy * dy + dz * dz).sqrt();
            scratch.prod[k * nlane + lane] = simd::one();
        }
    }
    (n, nb, active_lanes)
}

/// Normalize one lane block and write the selected centre's weights.
#[inline(always)]
pub(crate) fn finish_center_lane(
    prod: &[V],
    nlane: usize,
    num_centers: usize,
    center: usize,
    out: &mut [f64],
    active_lanes: usize,
) {
    for lane in 0..active_lanes {
        let mut total = simd::zero();
        for k in 0..num_centers {
            total += prod[k * nlane + lane];
        }
        let numerator = prod[center * nlane + lane];
        let value = numerator / total;
        // Match `normalize` in `partition::mod`: an exact-zero or non-finite
        // total puts the unit mass on centre 0, not on every centre.
        let fallback = if center == 0 {
            simd::one()
        } else {
            simd::zero()
        };
        let value = fallback.mask_select(simd::degenerate(total), value);
        let start = lane * simd::LANES;
        let len = simd::lane_len(out.len(), lane);
        simd::store(&mut out[start..start + len], value, len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn degenerate_normalization_puts_the_unit_mass_on_center_zero() {
        // A zero product total is the only case where the blocked kernel
        // does not simply divide by Z.  It must match `normalize`, which
        // returns [1, 0, 0], rather than writing 1 for every centre.
        let prod = [V::zero(); 3];
        let mut out = vec![f64::NAN; simd::LANES];
        for center in 0..3 {
            finish_center_lane(&prod, 1, 3, center, &mut out, 1);
            let want = if center == 0 { 1.0 } else { 0.0 };
            assert!(out.iter().all(|&v| v == want), "center {center}: {out:?}");
        }
    }
}
