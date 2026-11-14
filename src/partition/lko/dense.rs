//! Dense eight-point lane kernel for the LKO partition.
//!
//! This is the small-molecule fallback: it is the classic Becke lane kernel
//! with `inv_r` replaced by the precomputed LKO `fac_ij = 1 / sat(R_ij)` and
//! an explicit clamp of `mu`, then the classic polynomial switch.  The pair
//! loop is over all `i > j`, so every lane point evaluates the same product
//! sequence as [`super::point_weights_all`] and the result is bitwise
//! identical.
//!
//! Selected by [`super::use_sparse_lko`]; it reuses
//! [`crate::partition::kernel::LaneScratch`].

use crate::partition::kernel::{LaneScratch, finish_center_lane, init_lane_block};
use crate::partition::simd;
use crate::partition::simd::{step_function, step3};
use crate::partition::{PartitionConfig, PartitionTables};

/// LKO cell functions of `center` for one block of grid points.
pub(crate) fn block_weights(
    cfg: &PartitionConfig,
    tables: &PartitionTables,
    center: usize,
    grid: &[[f64; 3]],
    out: &mut [f64],
    scratch: &mut LaneScratch,
) {
    if cfg.hardness == 3 {
        block_h3(tables, center, grid, out, scratch);
    } else {
        block_generic(cfg, tables, center, grid, out, scratch);
    }
}

fn block_h3(
    tables: &PartitionTables,
    center: usize,
    grid: &[[f64; 3]],
    out: &mut [f64],
    scratch: &mut LaneScratch,
) {
    let (n, _nb, active_lanes) = init_lane_block(tables, grid, out, scratch);
    let nlane = scratch.nlane;
    for i in 0..n {
        for j in 0..i {
            let fac = simd::splat(tables.fac(i, j));
            let a_ab = simd::splat(tables.a_ab(i, j));
            for lane in 0..active_lanes {
                let di = scratch.dist[i * nlane + lane];
                let dj = scratch.dist[j * nlane + lane];
                let g = ((di - dj) * fac).max(simd::splat(-1.0)).min(simd::one());
                let nu = g + a_ab * (simd::one() - g * g);
                let f = step3(nu);
                let si = simd::splat(0.5) * (simd::one() - f);
                let sj = simd::splat(0.5) * (simd::one() + f);
                let pi = scratch.prod[i * nlane + lane];
                let pj = scratch.prod[j * nlane + lane];
                scratch.prod[i * nlane + lane] = pi * si;
                scratch.prod[j * nlane + lane] = pj * sj;
            }
        }
    }
    finish_center_lane(&scratch.prod, nlane, n, center, out, active_lanes);
}

fn block_generic(
    cfg: &PartitionConfig,
    tables: &PartitionTables,
    center: usize,
    grid: &[[f64; 3]],
    out: &mut [f64],
    scratch: &mut LaneScratch,
) {
    let (n, _nb, active_lanes) = init_lane_block(tables, grid, out, scratch);
    let nlane = scratch.nlane;
    let hardness = cfg.hardness;
    for i in 0..n {
        for j in 0..i {
            let fac = simd::splat(tables.fac(i, j));
            let a_ab = simd::splat(tables.a_ab(i, j));
            for lane in 0..active_lanes {
                let di = scratch.dist[i * nlane + lane];
                let dj = scratch.dist[j * nlane + lane];
                let g = ((di - dj) * fac).max(simd::splat(-1.0)).min(simd::one());
                let nu = g + a_ab * (simd::one() - g * g);
                let f = step_function(nu, hardness);
                let si = simd::splat(0.5) * (simd::one() - f);
                let sj = simd::splat(0.5) * (simd::one() + f);
                let pi = scratch.prod[i * nlane + lane];
                let pj = scratch.prod[j * nlane + lane];
                scratch.prod[i * nlane + lane] = pi * si;
                scratch.prod[j * nlane + lane] = pj * sj;
            }
        }
    }
    finish_center_lane(&scratch.prod, nlane, n, center, out, active_lanes);
}
