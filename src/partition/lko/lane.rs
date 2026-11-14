//! Masked eight-point lane kernel for the LKO partition.
//!
//! This is the main LKO kernel.  It evaluates **eight adjacent grid points at
//! once** with a *union* candidate list and per-lane masks:
//!
//! 1. the eight point-to-centre distances are evaluated lane-wise (one
//!    `f64x8` per atom) and each point gets its own
//!    `S = {i : d_i <= d_min + 5}` / `T = {i : d_i <= d_min + 10}` masks
//!    (`+1e-9` margin, see [`super`]);
//! 2. the union `T` list of the eight points is the pair-iteration domain;
//!    for a pair `(i, j)` the mask `need_i = S_i * T_j` selects the lanes
//!    where atom `i`'s product is multiplied by the pair factor and
//!    `need_j = S_j * T_i` the lanes for atom `j`.  Pairs with an empty
//!    union are skipped;
//! 3. the pair factor is computed for all eight lanes and multiplied into
//!    the per-lane product tile through a mask whose inactive lanes select
//!    exactly `1.0`.  Multiplication by exactly `1.0` is exact, and the
//!    pairs are visited in the canonical `(i, j)` order of the reference,
//!    so **every lane reproduces the reference multiply sequence
//!    bit-for-bit**;
//! 4. `Z` is the lane-wise ascending sum over all atoms (dead products are
//!    exactly zero) and the weight is `P_center / Z`; only an exact-zero or
//!    non-finite `Z` takes the first-centre fallback.
//!
//! The union set is much smaller than the full pair set, so the kernel stays
//! both screen-sparse and vectorised.
//!
//! Lane masks are stored as `u8` bitsets (one bit per lane): a
//! `Simd<bool, 8>` per atom makes `any_true` and the bool-array round trip
//! lower to eight scalar branches.  The bitset is built from the comparison
//! with a fused multiply/reduce (one vector op per mask) and turned back into
//! a blend selector through a 256-entry table of eight-byte masks.
//!
//! One block passed to this kernel belongs to a **single** generating atom
//! (the caller always evaluates one atom's grid), which is what allows the
//! final `prod[center]` lane extraction.

use lightweight_simd::Simd;

use crate::partition::simd::{self, V, step_function, step3};
use crate::partition::{PartitionConfig, PartitionTables};

use super::RCUT_LKO;

/// Extra screening width (Bohr); see [`super`].
const MARGIN: f64 = 1e-9;

/// The boolean lane mask of eight points.
pub(crate) type Mask = Simd<bool, { simd::LANES }>;

/// `MASK_TABLE[bits]` is the boolean lane mask of the `u8` bitset `bits`.
///
/// A `Simd<bool, _>` built lane by lane from a runtime `u8` lowers to eight
/// scalar branches in `any_true` and in the blend; the table keeps the
/// conversion to one 8-byte load.
const fn mask_table() -> [[bool; simd::LANES]; 256] {
    let mut table = [[false; simd::LANES]; 256];
    let mut bits = 0usize;
    while bits < 256 {
        let mut l = 0;
        while l < simd::LANES {
            table[bits][l] = (bits >> l) & 1 == 1;
            l += 1;
        }
        bits += 1;
    }
    table
}

static MASK_TABLE: [[bool; simd::LANES]; 256] = mask_table();

/// Lane weights `1, 2, 4, ...` used to pack a comparison result into a bitset.
const LANE_BITS: [f64; simd::LANES] = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0];

/// The `Simd` mask of a bitset.
#[inline(always)]
pub(crate) fn mask_of(bits: u8) -> Mask {
    Simd(MASK_TABLE[bits as usize])
}

/// Pack the comparison `d <= cut` into a lane bitset.
#[inline(always)]
pub(crate) fn le_bits(d: V, cut: V) -> u8 {
    let sel = simd::mask_value(d.simd_le(cut)) * V::from(LANE_BITS);
    sel.reduce_sum() as u8
}

/// Per-worker scratch of the masked lane kernel.
#[derive(Debug, Clone)]
pub(crate) struct Scratch {
    /// Block length in grid points.
    pub(crate) block: usize,
    /// `[center]` lane-wise distances.
    dist: Vec<V>,
    /// `[center]` lane-wise cell products (exactly zero for dead centres).
    prod: Vec<V>,
    /// `[center]` bit-set of the lanes where the centre is live.
    s_mask: Vec<u8>,
    /// `[center]` bit-set of the lanes where the centre is a candidate.
    t_mask: Vec<u8>,
    /// `[candidate]` union of the eight `T` sets, ascending.
    t_list: Vec<u32>,
}

impl Scratch {
    pub(crate) fn new(num_centers: usize, block: usize) -> Self {
        Self {
            block,
            dist: vec![V::zero(); num_centers],
            prod: vec![V::zero(); num_centers],
            s_mask: vec![0; num_centers],
            t_mask: vec![0; num_centers],
            t_list: vec![0; num_centers],
        }
    }
}

/// LKO weight of `center` for one block of grid points.
pub(crate) fn block_weights(
    cfg: &PartitionConfig,
    tables: &PartitionTables,
    center: usize,
    grid: &[[f64; 3]],
    out: &mut [f64],
    scratch: &mut Scratch,
) {
    if cfg.hardness == 3 {
        block_impl(tables, center, grid, out, scratch, step3);
    } else {
        let hardness = cfg.hardness;
        block_impl(tables, center, grid, out, scratch, |g| {
            step_function(g, hardness)
        });
    }
}

#[inline]
fn block_impl<F: Fn(V) -> V>(
    tables: &PartitionTables,
    center: usize,
    grid: &[[f64; 3]],
    out: &mut [f64],
    scratch: &mut Scratch,
    switch: F,
) {
    let n = tables.len();
    let nb = grid.len();
    let nlane = nb.div_ceil(simd::LANES);
    debug_assert!(nb <= scratch.block);
    debug_assert!(center < n);
    let Scratch {
        dist,
        prod,
        s_mask,
        t_mask,
        t_list,
        ..
    } = scratch;
    let one = simd::one();
    let zero = simd::zero();
    let s_width = simd::splat(RCUT_LKO + MARGIN);
    let t_width = simd::splat(2.0 * RCUT_LKO + MARGIN);
    let half = simd::splat(0.5);
    let minus_one = simd::splat(-1.0);
    let dist = &mut dist[..n];
    let prod = &mut prod[..n];
    let s_mask = &mut s_mask[..n];
    let t_mask = &mut t_mask[..n];

    for l_lane in 0..nlane {
        let coords = simd::gather_coords(grid, l_lane);
        // --- distances, lane-wise --- //
        for (k, slot) in dist.iter_mut().enumerate() {
            let c = tables.center(k);
            let dx = coords[0] - simd::splat(c[0]);
            let dy = coords[1] - simd::splat(c[1]);
            let dz = coords[2] - simd::splat(c[2]);
            *slot = (dx * dx + dy * dy + dz * dz).sqrt();
        }
        // --- per-lane nearest distance and screening masks --- //
        let mut dmin = V::splat(f64::INFINITY);
        for &d in dist.iter() {
            dmin = dmin.min(d);
        }
        let s_cut = dmin + s_width;
        let t_cut = dmin + t_width;
        let mut nt = 0usize;
        for k in 0..n {
            let d = dist[k];
            let s = le_bits(d, s_cut);
            let t = le_bits(d, t_cut);
            s_mask[k] = s;
            t_mask[k] = t;
            if t != 0 {
                t_list[nt] = k as u32;
                nt += 1;
            }
            prod[k] = one.mask_select(mask_of(s), zero);
        }
        // --- union pair loop, canonical (i, j) order --- //
        for a in 0..nt {
            let i = t_list[a] as usize;
            debug_assert!(i < n);
            // SAFETY: `t_list` only contains atom indices in `0..n`.
            let di = unsafe { *dist.get_unchecked(i) };
            let s_i = unsafe { *s_mask.get_unchecked(i) };
            let t_i = unsafe { *t_mask.get_unchecked(i) };
            let row = i * n;
            let (prod_lo, prod_hi) = prod.split_at_mut(i);
            let ri = &mut prod_hi[0];
            for &jj in &t_list[..a] {
                let j = jj as usize;
                debug_assert!(j < n);
                // SAFETY: `t_list` only contains atom indices in `0..n`, and
                // `j != i`, so `prod[j]` does not alias `ri`.
                let need_i = unsafe { s_i & *t_mask.get_unchecked(j) };
                let need_j = unsafe { *s_mask.get_unchecked(j) & t_i };
                if need_i | need_j == 0 {
                    continue;
                }
                let fac = simd::splat(tables.fac_row(row + j));
                let a_ab = simd::splat(tables.a_ab_row(row + j));
                let dj = unsafe { *dist.get_unchecked(j) };
                let g = ((di - dj) * fac).max(minus_one).min(one);
                let nu = g + a_ab * (one - g * g);
                let f = switch(nu);
                let si = half * (one - f);
                let sj = half * (one + f);
                *ri *= si.mask_select(mask_of(need_i), one);
                // SAFETY: `j < i` (ascending `T` list), so this is a distinct
                // element of the other half of the split borrow.
                unsafe {
                    *prod_lo.get_unchecked_mut(j) *= sj.mask_select(mask_of(need_j), one);
                }
            }
        }
        // --- normalization and output --- //
        let mut z = zero;
        for &p in prod.iter() {
            z += p;
        }
        let q = prod[center] / z;
        // Match `normalize` in `partition::mod`: an exact-zero or non-finite
        // total puts the unit mass on centre 0, not on every centre.
        let fallback = if center == 0 { one } else { zero };
        let w = fallback.mask_select(simd::degenerate(z), q);
        let start = l_lane * simd::LANES;
        let len = simd::lane_len(nb, l_lane);
        simd::store(&mut out[start..start + len], w, len);
    }
}
