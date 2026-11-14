//! Dense compact-support pair-product fallback for small molecules.
//!
//! Below the sparse-kernel crossover the classic `O(natm^2)` pair product is
//! faster: the sparse kernel pays a distance pass, a sort and a per-survivor
//! product even when there are only a handful of atoms.  This module keeps
//! the dense kernels next to the sparse one so the two cell-function
//! directions they use are documented in one place.
//!
//! The dense block kernel shares [`LaneScratch`], [`init_lane_block`] and
//! [`finish_center_lane`] with the classic Becke kernels, which is why that
//! machinery lives in `partition::kernel`.
//!
//! # Cell-function direction
//!
//! For a symmetric switch the dense kernel evaluates each *unordered* pair
//! once and gives the two products the complementary factors
//! `1/2 (1 - f)` / `1/2 (1 + f)`; the partner therefore sees the size
//! adjustment in the opposite direction.  The sparse kernel uses the
//! canonical per-product direction (see [`super`]).  The two are identical
//! because `a(B, A) = -a(A, B)` and differ only at round-off.  The Gaussian
//! switch is asymmetric, so the dense path evaluates the partner direction
//! explicitly instead of using the complement.

use crate::partition::kernel::{self, LaneScratch, finish_center_lane, init_lane_block};
use crate::partition::simd;
use crate::partition::stratmann::SsSwitch;
use crate::partition::{PartitionTables, dist3};

/// Dense cell products of all centres at one point, then normalize.
pub(crate) fn point_weights_all<S: SsSwitch>(
    tables: &PartitionTables,
    point: [f64; 3],
    out: &mut [f64],
    dist: &mut [f64],
    prod: &mut [f64],
) {
    let n = tables.len();
    if n == 0 {
        return;
    }
    for (i, d) in dist[..n].iter_mut().enumerate() {
        *d = dist3(point, tables.center(i));
    }
    prod[..n].fill(1.0);
    for i in 0..n {
        for j in 0..i {
            let mu = (dist[i] - dist[j]) / tables.r_ab(i, j);
            let nu = mu + tables.a_ab(i, j) * (1.0 - mu * mu);
            let f = S::switch(nu);
            prod[i] *= 0.5 * (1.0 - f);
            let f_j = if S::COMPLEMENTARY_FACTOR {
                -f
            } else {
                let mu_j = -mu;
                let nu_j = mu_j + tables.a_ab(j, i) * (1.0 - mu_j * mu_j);
                S::switch(nu_j)
            };
            prod[j] *= 0.5 * (1.0 - f_j);
        }
    }
    kernel::finish_all(&prod[..n], out);
}

/// Dense weights of `center` for one block of grid points.
pub(crate) fn block_weights<S: SsSwitch>(
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
            let a_ab = simd::splat(tables.a_ab(i, j));
            let r_ab = simd::splat(tables.r_ab(i, j));
            for lane in 0..active_lanes {
                let di = scratch.dist[i * nlane + lane];
                let dj = scratch.dist[j * nlane + lane];
                let mu = (di - dj) / r_ab;
                let nu = mu + a_ab * (simd::one() - mu * mu);
                let f = S::switch_vec(nu);
                let si = simd::splat(0.5) * (simd::one() - f);
                let fj = if S::COMPLEMENTARY_FACTOR {
                    -f
                } else {
                    let mu_j = -mu;
                    let nu_j = mu_j + simd::splat(tables.a_ab(j, i)) * (simd::one() - mu_j * mu_j);
                    S::switch_vec(nu_j)
                };
                let sj = simd::splat(0.5) * (simd::one() - fj);
                let pi = scratch.prod[i * nlane + lane];
                let pj = scratch.prod[j * nlane + lane];
                scratch.prod[i * nlane + lane] = pi * si;
                scratch.prod[j * nlane + lane] = pj * sj;
            }
        }
    }
    finish_center_lane(&scratch.prod, nlane, n, center, out, active_lanes);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::partition::{PartitionConfig, PartitionScheme, RadiiConfig};

    /// The dense fallback must agree with the sparse value kernel at
    /// round-off for a molecule below the sparse crossover.
    #[test]
    fn dense_matches_sparse_at_roundoff() {
        let centers = [
            [0.0, 0.0, 0.0],
            [2.3, 0.4, -0.2],
            [-1.1, 2.0, 0.7],
            [0.3, -2.2, 1.4],
        ];
        let charges = [8u32, 1, 6, 7];
        let cfg = PartitionConfig::new(PartitionScheme::Stratmann, RadiiConfig::pyscf());
        let tables = cfg.tables(&centers, &charges).unwrap();
        let n = centers.len();
        let mut dense = vec![0.0; n];
        let mut dist = vec![0.0; n];
        let mut prod = vec![0.0; n];
        let mut sparse = vec![0.0; n];
        let mut sdist = vec![0.0; n];
        let mut order = vec![0u32; n];
        let mut sprod = vec![0.0; n];
        for k in 0..40 {
            let t = k as f64 * 0.21;
            let point = [3.0 * t.cos(), 2.0 * t.sin(), 0.4 * t];
            point_weights_all::<crate::partition::stratmann::StratmannSwitch>(
                &tables, point, &mut dense, &mut dist, &mut prod,
            );
            super::super::point_weights_all::<crate::partition::stratmann::StratmannSwitch>(
                &tables,
                point,
                &mut sparse,
                &mut sdist,
                &mut order,
                &mut sprod,
            );
            for a in 0..n {
                assert!(
                    (dense[a] - sparse[a]).abs() <= 1e-13 * sparse[a].abs().max(1e-300) + 1e-300,
                    "point {point:?} atom {a}: dense {} sparse {}",
                    dense[a],
                    sparse[a]
                );
            }
        }
    }
}
