//! Classic Becke partition weights (A. D. Becke, JCP 88, 4571 (1988)).
//!
//! Cell-function products for all centers, pairwise-updated exactly as in
//! Becke's scheme (each unordered pair `{ia, ib}` updates both entries with
//! `0.5*(1-f)` and `0.5*(1+f)`), then normalized so the partition weights
//! sum to 1 at every grid point.
//!
//! The molecular tables are evaluated once by
//! [`PartitionTables::new`](super::PartitionTables::new); the kernels here
//! only consume point-to-center distances and the precomputed `a_AB` / `R_AB`.
//!
//! The blocked kernel is specialised on the hardness variant (3 or generic)
//! and on the ORCA early-exit flag, so the common PySCF path has neither a scheme match
//! nor an early-exit branch inside the point loop.  The arithmetic sequence is
//! unchanged: the hardness-3 specialisation is the same three composition
//! layers, and the non-early path multiplies each pair factor directly.
//! `tests/orca_bitwise.rs` guards the result bit-for-bit.

use crate::partition::kernel::{self, LaneScratch, finish_center_lane, init_lane_block};
use crate::partition::simd;
use crate::partition::{PartitionConfig, PartitionScheme, PartitionTables};

/// Classic pair-product weights of all centers at one point (Bohr).
///
/// # Arguments
///
/// * `cfg` — classic Becke partition configuration.
/// * `tables` — molecule-only weight tables.
/// * `point` — `[3]` Cartesian coordinate in Bohr.
/// * `out` — `[natm]` normalized partition weights (sum to 1).
/// * `dist` — `[natm]` scratch: point-to-centre distances.
/// * `prod` — `[natm]` scratch: running cell products.
pub(crate) fn point_weights_all(
    cfg: &PartitionConfig,
    tables: &PartitionTables,
    point: [f64; 3],
    out: &mut [f64],
    dist: &mut [f64],
    prod: &mut [f64],
) {
    let n = tables.len();
    for (i, d) in dist[..n].iter_mut().enumerate() {
        *d = crate::partition::dist3(point, tables.center(i));
    }
    prod.fill(1.0);
    // ORCA stops the pair sweep of an atom once `|1 - f| <= 1e-14` and
    // leaves the partner product untouched; PySCF's reference continues.
    for i in 0..n {
        for j in 0..i {
            let mu = (dist[i] - dist[j]) / tables.r_ab(i, j);
            let nu = mu + tables.a_ab(i, j) * (1.0 - mu * mu);
            let f = cfg.switch_value(nu);
            if cfg.becke_early_exit && (1.0 - f).abs() <= 1e-14 {
                prod[i] = 0.0;
                break;
            }
            prod[i] *= 0.5 * (1.0 - f);
            prod[j] *= 0.5 * (1.0 + f);
        }
    }
    kernel::finish_all(prod, out);
}

/// Classic pair-product weights of `center` for one block of grid points.
///
/// The block is evaluated as fixed eight-point lanes; the atom-pair loop is
/// outside the lane loop, so every pair updates eight points at once.
///
/// # Arguments
///
/// * `cfg` — classic Becke partition configuration (with or without early exit).
/// * `tables` — molecule-only weight tables.
/// * `center` — global atom index whose weight is written.
/// * `grid` — `[B, 3]` row-major global grid coordinates.
/// * `out` — `[B]` normalized partition weight of `center` per point.
/// * `scratch` — fixed-lane `dist`/`prod` tiles.
pub(crate) fn block_weights(
    cfg: &PartitionConfig,
    tables: &PartitionTables,
    center: usize,
    grid: &[[f64; 3]],
    out: &mut [f64],
    scratch: &mut LaneScratch,
) {
    match cfg.scheme {
        PartitionScheme::Becke if cfg.becke_early_exit => {
            block_becke_early(cfg, tables, center, grid, out, scratch)
        }
        PartitionScheme::Becke => block_no_early_exit(cfg, tables, center, grid, out, scratch),
        _ => unreachable!("Becke weights kernel called for a non-pair-product scheme"),
    }
}

/// Classic pair product without the ORCA early exit, for any hardness.
///
/// Shared by the classic Becke scheme and the mBecke fast path: mBecke's
/// nearest-first product with the default clamp is the classic product in a
/// different multiplication order, so the two agree to round-off (see
/// [`crate::partition::mbecke`]).
pub(crate) fn block_no_early_exit(
    cfg: &PartitionConfig,
    tables: &PartitionTables,
    center: usize,
    grid: &[[f64; 3]],
    out: &mut [f64],
    scratch: &mut LaneScratch,
) {
    if cfg.hardness == 3 {
        block_becke_h3(tables, center, grid, out, scratch);
    } else {
        block_generic(cfg, tables, center, grid, out, scratch);
    }
}

/// Hardness-3 classic Becke, no early exit (PySCF default path).
fn block_becke_h3(
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
                let f = simd::step3(nu);
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

/// Generic classic Becke without early exit (`hardness != 3`).
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
            let a_ab = simd::splat(tables.a_ab(i, j));
            let r_ab = simd::splat(tables.r_ab(i, j));
            for lane in 0..active_lanes {
                let di = scratch.dist[i * nlane + lane];
                let dj = scratch.dist[j * nlane + lane];
                let mu = (di - dj) / r_ab;
                let nu = mu + a_ab * (simd::one() - mu * mu);
                let f = simd::step_function(nu, hardness);
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

/// Classic Becke with the ORCA early exit.
///
/// `active[lane]` is 0 once atom `i` saturated for that point, after which
/// the remaining `j` of this `i` are skipped (partner products stay
/// untouched).  The lane arithmetic is the scalar arithmetic lane by lane.
fn block_becke_early(
    cfg: &PartitionConfig,
    tables: &PartitionTables,
    center: usize,
    grid: &[[f64; 3]],
    out: &mut [f64],
    scratch: &mut LaneScratch,
) {
    let (n, _nb, active_lanes) = init_lane_block(tables, grid, out, scratch);
    let nlane = scratch.nlane;
    for i in 0..n {
        for lane in 0..active_lanes {
            scratch.active[lane] = simd::one();
        }
        for j in 0..i {
            let a_ab = simd::splat(tables.a_ab(i, j));
            let r_ab = simd::splat(tables.r_ab(i, j));
            for lane in 0..active_lanes {
                let di = scratch.dist[i * nlane + lane];
                let dj = scratch.dist[j * nlane + lane];
                let mu = (di - dj) / r_ab;
                let nu = mu + a_ab * (simd::one() - mu * mu);
                let f = if cfg.hardness == 3 {
                    simd::step3(nu)
                } else {
                    simd::step_function(nu, cfg.hardness)
                };
                let sat = simd::mask_value(simd::abs_le(simd::one() - f, 1e-14));
                let keep = scratch.active[lane] * (simd::one() - sat);
                scratch.active[lane] = keep;
                let si = simd::splat(0.5) * (simd::one() - f);
                let sj = simd::splat(0.5) * (simd::one() + f);
                let pi = scratch.prod[i * nlane + lane];
                let pj = scratch.prod[j * nlane + lane];
                scratch.prod[i * nlane + lane] = pi * (si * keep);
                scratch.prod[j * nlane + lane] = pj * (sj * keep + (simd::one() - keep));
            }
        }
    }
    finish_center_lane(&scratch.prod, nlane, n, center, out, active_lanes);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::partition::{PartitionConfigBuilder, RadiiConfig};

    type Case = (Vec<(f64, f64, f64)>, Vec<u32>);

    const H2: [(f64, f64, f64); 2] = [(-0.7, 0.0, 0.0), (0.7, 0.0, 0.0)];

    /// Partition weights sum to 1 at every point (unit test of the cell
    /// functions on a heteronuclear CO-like pair as well).
    #[test]
    fn weights_sum_to_one() {
        let cases: Vec<Case> = vec![
            (H2.to_vec(), vec![1, 1]),
            (H2.to_vec(), vec![8, 6]),
            (
                vec![(0.0, 0.0, 0.0), (0.0, 0.0, 2.5), (1.5, 1.0, -1.0)],
                vec![1, 6, 17],
            ),
        ];
        let cfg = PartitionConfigBuilder::default()
            .scheme(PartitionScheme::Becke)
            .radii(crate::partition::test_support::custom_radii())
            .becke_early_exit(true)
            .build()
            .unwrap();
        for (centers, charges) in &cases {
            let centers3: Vec<[f64; 3]> = centers.iter().map(|&(x, y, z)| [x, y, z]).collect();
            for &x in &[-2.0, -0.7, 0.0, 0.35, 1.0, 2.5] {
                let pt = [x, 0.2 * (x + 1.0), -0.1];
                let total: f64 = (0..centers3.len())
                    .map(|a| cfg.weight(a, pt, &centers3, charges).unwrap())
                    .sum();
                assert!((total - 1.0).abs() < 1e-10, "sum = {total} at point {pt:?}");
            }
        }
    }

    /// The ORCA and PySCF radii presets use the same kernel but different
    /// tables, so the weights must differ.
    #[test]
    fn radii_presets_change_the_weight() {
        let orca = PartitionConfig::new(
            PartitionScheme::Becke,
            crate::partition::test_support::custom_radii(),
        );
        let pyscf = PartitionConfig::new(PartitionScheme::Becke, RadiiConfig::pyscf());
        let centers = [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]];
        let charges = [2u32, 10];
        let pt = [0.3, 0.1, 1.0];
        assert!(
            (orca.weight(0, pt, &centers, &charges).unwrap()
                - pyscf.weight(0, pt, &centers, &charges).unwrap())
            .abs()
                > 1e-4
        );
    }

    #[test]
    fn lane_block_matches_point_kernel_bitwise() {
        let centers = [[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]];
        let charges = [1u32, 1];
        for early in [false, true] {
            let cfg = PartitionConfigBuilder::default()
                .scheme(PartitionScheme::Becke)
                .radii(crate::partition::test_support::custom_radii())
                .becke_early_exit(early)
                .build()
                .unwrap();
            let tables = cfg.tables(&centers, &charges).unwrap();
            let grid: Vec<[f64; 3]> = (0..37)
                .map(|k| [-2.0 + 4.0 * k as f64 / 36.0, 0.13, -0.21])
                .collect();
            for center in 0..tables.len() {
                let mut lane = LaneScratch::new(tables.len(), 48);
                let mut got = vec![0.0; grid.len()];
                block_weights(&cfg, &tables, center, &grid, &mut got, &mut lane);
                for (k, pt) in grid.iter().enumerate() {
                    let mut want = vec![0.0; tables.len()];
                    let mut dist = vec![0.0; tables.len()];
                    let mut prod = vec![0.0; tables.len()];
                    point_weights_all(&cfg, &tables, *pt, &mut want, &mut dist, &mut prod);
                    assert_eq!(
                        got[k].to_bits(),
                        want[center].to_bits(),
                        "early={early} point={pt:?} center={center}: {} vs {}",
                        got[k],
                        want[center]
                    );
                }
            }
        }
    }
    #[test]
    fn early_four_atom_block_matches_point() {
        let centers = [
            [0.0, 0.0, 0.0],
            [1.43, 1.11, 0.0],
            [-1.43, 1.11, 0.2],
            [0.0, -2.0, 0.1],
        ];
        let charges = [8u32, 1, 1, 6];
        let cfg = PartitionConfigBuilder::default()
            .scheme(PartitionScheme::Becke)
            .radii(crate::partition::test_support::custom_radii())
            .becke_early_exit(true)
            .build()
            .unwrap();
        let tables = cfg.tables(&centers, &charges).unwrap();
        let grid: Vec<[f64; 3]> = (0..256)
            .map(|k| {
                let t = k as f64 * 0.027;
                [
                    2.5 * t.cos() + 0.3 * t.sin(),
                    2.0 * t.sin() - 0.2,
                    1.5 * t.cos() + 0.1 * t,
                ]
            })
            .collect();
        let center = 0;
        let mut lane = LaneScratch::new(tables.len(), 256);
        let mut got = vec![0.0; grid.len()];
        block_weights(&cfg, &tables, center, &grid, &mut got, &mut lane);
        for (k, pt) in grid.iter().enumerate() {
            let mut want = vec![0.0; tables.len()];
            let mut dist = vec![0.0; tables.len()];
            let mut prod = vec![0.0; tables.len()];
            point_weights_all(&cfg, &tables, *pt, &mut want, &mut dist, &mut prod);
            assert_eq!(
                got[k].to_bits(),
                want[center].to_bits(),
                "k={k} got={} want={}",
                got[k],
                want[center]
            );
        }
    }
}
