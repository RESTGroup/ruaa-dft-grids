//! Compact-support pair products: PySCF SSF and Gaussian 16 SS.
//!
//! The SSF and Gaussian SS partitions share the same cell-product kernel.
//! Their switches are piecewise polynomials that are **exactly** `-1`
//! (cell factor one) on the negative side and **exactly** `+1` (cell factor
//! zero) on the positive side outside their support:
//!
//! * PySCF/ORCA SSF (Stratmann, Scuseria and Frisch, 1996; PySCF
//!   `gen_grid.stratmann`): `A = 0.64`, degree 7, symmetric support
//!   `|nu| <= A`;
//! * Gaussian 16 `SSWeights`: `XScal = 0.86`, degree 13, asymmetric support
//!   `-0.86 <= nu <= 0.852` (see [`super::PartitionConfig::gaussian_ss`]).
//!
//! A cell factor is `s = 1/2 (1 - f)`, so it is **exactly zero** where
//! `f = 1` and **exactly one** where `f = -1`.  Multiplying by exactly one
//! is a no-op, so a cell product only depends on the "active" pairs.
//!
//! Two structural facts follow when the positive support bound is the
//! killing cutoff:
//!
//! * `P_M` can only be zeroed by a partner that is **closer** to the grid
//!   point than `M`.  For `d_j >= d_M` we have `mu <= 0`, hence
//!   `nu <= 1/2` for every `|a| <= 1/2`, so the factor is strictly positive.
//! * A partner `j0` closer than `M` certifies `P_M = 0` as soon as
//!   `d_M >= R_cut d_j0`, with `R_cut` derived once from the largest
//!   `|a_AB|` of the molecule and the positive support bound (see
//!   [`PartitionTables::ss_rcut`]).
//!
//! The kernel therefore evaluates one grid point as
//!
//! 1. all point-to-centre distances,
//! 2. the candidate prefix `d_i <= R_cut d_min`, sorted by distance,
//! 3. for every candidate, a scan over the already processed (closer)
//!    candidates; the first one that kills the product ends the scan,
//! 4. the cell product of the survivors only,
//! 5. `Z = sum P_M` and `Pbar / Z`.
//!
//! The pair work per point drops from `O(natm^2)` to `O(natm log natm)` in
//! the candidate prefix plus `O(n_survivors * natm)` for the products.  The
//! full product is evaluated as eight-partner `f64x8` lanes; a killed atom
//! is exactly `0.0`.
//!
//! # Cell-function convention
//!
//! This module evaluates the canonical per-product direction: for a product
//! `M` every partner `j` uses `a(M, j)`, including `j > M`.  The dense
//! fallback may evaluate each *unordered* pair once and give the partner the
//! complementary factor `1/2 (1 + f)`.  That is algebraically identical for
//! a symmetric switch with `a(B, A) = -a(A, B)`; the Gaussian switch is
//! asymmetric, so its dense path evaluates the partner direction explicitly.
//!
//! # Dense fallback
//!
//! Below an ISA-dependent crossover the classic `O(natm^2)` pair product is
//! faster than the sparse kernel (the cut is empirical; see
//! `use_sparse_stratmann`); [`dense`] keeps that fallback next to the sparse
//! one.  It shares the lane scratch with the classic Becke kernels through
//! [`crate::partition::kernel`].

use crate::partition::simd::{self, V};
use crate::partition::{PartitionTables, dist3};

mod switch;

pub(crate) use switch::{A, GAUSSIAN_MU_POS, GaussianSsSwitch, SsSwitch, StratmannSwitch, active};

/// Per-worker scratch for the sparse SSF kernels.
///
/// The layout is flat because the hot loops are per point; only the distance
/// plane is evaluated eight points at a time.
#[derive(Debug, Clone)]
pub(crate) struct Scratch {
    /// Block length in grid points.
    pub(crate) block: usize,
    /// `[center]` distance of the current eight-point lane.
    pub(crate) lane: Vec<V>,
    /// `[center]` distance of the current point.
    pub(crate) dist: Vec<f64>,
    /// `[candidate]` distance-sorted candidate atoms.
    pub(crate) order: Vec<u32>,
    /// `[center]` cell product (exactly zero for killed atoms).
    pub(crate) prod: Vec<f64>,
}

impl Scratch {
    pub(crate) fn new(num_centers: usize, block: usize) -> Self {
        Self {
            block,
            lane: vec![V::zero(); num_centers],
            dist: vec![0.0; num_centers],
            order: vec![0; num_centers],
            prod: vec![0.0; num_centers],
        }
    }
}

/// Point-to-centre distance field of one lane of eight grid points.
#[inline]
pub(crate) fn fill_lane(tables: &PartitionTables, coords: [V; 3], lane: &mut [V]) {
    let n = tables.len();
    for (k, slot) in lane.iter_mut().take(n).enumerate() {
        let c = tables.center(k);
        let dx = coords[0] - simd::splat(c[0]);
        let dy = coords[1] - simd::splat(c[1]);
        let dz = coords[2] - simd::splat(c[2]);
        *slot = (dx * dx + dy * dy + dz * dz).sqrt();
    }
}

/// Transpose one lane of distances into `dist`.
#[inline]
pub(crate) fn scatter_lane(lane: &[V], l: usize, dist: &mut [f64]) {
    for (d, v) in dist.iter_mut().zip(lane) {
        *d = v[l];
    }
}

/// Cell product of one survivor.
///
/// The self term has `mu = 0`, `a = 0`, hence `nu = 0` and factor exactly
/// `1/2`; it is included in the product and removed by the final `* 2`, which
/// is exact in binary floating point.  This keeps the partner loop branch-free.
#[inline]
pub(crate) fn product<S: SsSwitch>(tables: &PartitionTables, i: usize, dist: &[f64]) -> f64 {
    let n = tables.len();
    let row = i * n;
    let di = simd::splat(dist[i]);
    let mut acc = simd::one();
    let mut j = 0;
    while j + simd::LANES <= n {
        let dj = V::from_slice(&dist[j..j + simd::LANES]);
        let inv = V::from_slice(tables.inv_r_slice(row + j, simd::LANES));
        let a = V::from_slice(tables.a_slice(row + j, simd::LANES));
        let mu = (di - dj) * inv;
        let nu = mu + a * (simd::one() - mu * mu);
        acc *= simd::splat(0.5) * (simd::one() - S::switch_vec(nu));
        j += simd::LANES;
    }
    let mut tail = 1.0;
    while j < n {
        let mu = (dist[i] - dist[j]) * tables.inv_r_row(row + j);
        let nu = mu + tables.a_ab_row(row + j) * (1.0 - mu * mu);
        tail *= 0.5 * (1.0 - S::switch(nu));
        j += 1;
    }
    acc.reduce_product() * 2.0 * tail
}

/// Distances are already in `dist`; fill `order`/`prod` and return `Z`.
///
/// `prod` is exactly zero for every killed atom; only the survivors are
/// evaluated.
pub(crate) fn prepare_slices<S: SsSwitch>(
    tables: &PartitionTables,
    dist: &[f64],
    order: &mut [u32],
    prod: &mut [f64],
) -> f64 {
    let n = tables.len();
    debug_assert!(dist.len() >= n && order.len() >= n && prod.len() >= n);

    // A farther atom can never kill a product; a closer atom at distance
    // `<= d / R_cut` certifies the kill.  Only that prefix needs sorting.
    let rcut = tables.ss_rcut();
    let mut dmin = f64::INFINITY;
    for &d in &dist[..n] {
        dmin = dmin.min(d);
    }
    let cut = rcut * dmin;
    let mut nc = 0;
    for (i, &d) in dist[..n].iter().enumerate() {
        if d <= cut {
            order[nc] = i as u32;
            nc += 1;
        }
    }
    order[..nc].sort_unstable_by_key(|&i| dist[i as usize].to_bits());

    prod[..n].fill(0.0);
    for pos in 0..nc {
        let i = order[pos] as usize;
        let row = i * n;
        let di = dist[i];
        let mut killed = false;
        for &j in &order[..pos] {
            let j = j as usize;
            let mu = (di - dist[j]) * tables.inv_r_row(row + j);
            let nu = mu + tables.a_ab_row(row + j) * (1.0 - mu * mu);
            if nu >= S::MU_MAX {
                killed = true;
                break;
            }
        }
        if !killed {
            prod[i] = product::<S>(tables, i, dist);
        }
    }

    let mut z = 0.0;
    for &p in &prod[..n] {
        z += p;
    }
    z
}

/// Sparse SSF weights of all centres at one point, reusing the caller's
/// `dist`/`order`/`prod` scratch.
pub(crate) fn point_weights_all<S: SsSwitch>(
    tables: &PartitionTables,
    point: [f64; 3],
    out: &mut [f64],
    dist: &mut [f64],
    order: &mut [u32],
    prod: &mut [f64],
) {
    let n = tables.len();
    if n == 0 {
        return;
    }
    for (i, d) in dist[..n].iter_mut().enumerate() {
        *d = dist3(point, tables.center(i));
    }
    let z = prepare_slices::<S>(tables, dist, order, prod);
    if z != 0.0 && z.is_finite() {
        for (o, p) in out[..n].iter_mut().zip(&prod[..n]) {
            *o = p / z;
        }
    } else {
        out[..n].fill(0.0);
        out[0] = 1.0;
    }
}

/// Sparse SSF weights of `center` for one block of grid points.
pub(crate) fn block_weights<S: SsSwitch>(
    tables: &PartitionTables,
    center: usize,
    grid: &[[f64; 3]],
    out: &mut [f64],
    scratch: &mut Scratch,
) {
    let n = tables.len();
    let nb = grid.len();
    let nlane = nb.div_ceil(simd::LANES);
    debug_assert!(nb <= scratch.block);
    let Scratch {
        lane,
        dist,
        order,
        prod,
        ..
    } = scratch;
    for l_lane in 0..nlane {
        let coords = simd::gather_coords(grid, l_lane);
        fill_lane(tables, coords, &mut lane[..n]);
        let len = simd::lane_len(nb, l_lane);
        for l in 0..len {
            scatter_lane(&lane[..n], l, &mut dist[..n]);
            let z = prepare_slices::<S>(tables, &dist[..n], &mut order[..n], &mut prod[..n]);
            let pbar = prod[center];
            let fallback = if center == 0 { 1.0 } else { 0.0 };
            out[l_lane * simd::LANES + l] = if z != 0.0 && z.is_finite() {
                pbar / z
            } else {
                fallback
            };
        }
    }
}

pub(crate) mod dense;
pub(crate) mod deriv;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::partition::{PartitionConfig, PartitionScheme, RadiiConfig};

    #[test]
    fn stratmann_follows_pyscf_ssf() {
        // Unified `Stratmann` is PySCF's SSF switch, not ORCA's alias.
        let strat = PartitionConfig::new(PartitionScheme::Stratmann, RadiiConfig::pyscf());
        let becke = PartitionConfig::new(PartitionScheme::Becke, RadiiConfig::pyscf());
        let centers = [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let charges = [8u32, 6];
        let mut diff = 0.0_f64;
        for &x in &[-1.5, -0.5, 0.0, 0.5, 1.5] {
            let pt = [x, 0.1, -0.2];
            diff = diff.max(
                (strat.weight(0, pt, &centers, &charges).unwrap()
                    - becke.weight(0, pt, &centers, &charges).unwrap())
                .abs(),
            );
        }
        assert!(diff > 1e-6, "Stratmann did not use the SSF switch");
        assert_eq!(strat.scheme, PartitionScheme::Stratmann);
    }

    /// The sparse kernel (candidate cut + survivor scan + lane product) must
    /// reproduce the direct dense cell products for arbitrary geometries,
    /// radii adjustments and points, including points exactly on a centre.
    #[test]
    fn sparse_products_match_dense_reference() {
        let mut seed = 0x1234_5678_9abc_def0_u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut uniform = |lo: f64, hi: f64| lo + (hi - lo) * rnd();
        let configs = [
            PartitionConfig::new(PartitionScheme::Stratmann, RadiiConfig::pyscf()),
            PartitionConfig::new(
                PartitionScheme::Stratmann,
                crate::partition::test_support::custom_radii(),
            ),
            PartitionConfig::new(PartitionScheme::Stratmann, RadiiConfig::no_adjust()),
            PartitionConfig::new(PartitionScheme::Stratmann, RadiiConfig::becke_adjust()),
        ];
        let dense_product = |tables: &PartitionTables, point: [f64; 3], m: usize| {
            let n = tables.len();
            let mut p = 1.0;
            let dm = dist3(point, tables.center(m));
            for j in 0..n {
                if j == m {
                    continue;
                }
                let dj = dist3(point, tables.center(j));
                let inv = tables.inv_r_row(m * n + j);
                let mu = (dm - dj) * inv;
                let nu = mu + tables.a_ab_row(m * n + j) * (1.0 - mu * mu);
                p *= 0.5 * (1.0 - StratmannSwitch::switch(nu));
            }
            p
        };
        for case in 0..64 {
            let n = 1 + case % 9;
            let mut centers = Vec::with_capacity(n);
            for k in 0..n {
                let scale = if case % 3 == 0 { 8.0 } else { 3.0 };
                centers.push([
                    uniform(-scale, scale),
                    uniform(-scale, scale),
                    uniform(-scale, scale),
                ]);
                if k > 0 {
                    // keep a minimum separation
                    let prev = centers[k - 1];
                    if dist3(prev, centers[k]) < 0.4 {
                        centers[k][0] += 0.7;
                    }
                }
            }
            let charges: Vec<u32> = (0..n).map(|k| 1 + (k as u32 * 7) % 18).collect();
            for cfg in &configs {
                let tables = cfg.tables(&centers, &charges).unwrap();
                let mut dist = vec![0.0; n];
                let mut order = vec![0u32; n];
                let mut prod = vec![0.0; n];
                for p in 0..24 {
                    let point = match p % 6 {
                        0 => centers[p % n],
                        1 => {
                            let c = centers[p % n];
                            [c[0] + 1e-12, c[1] - 1e-12, c[2]]
                        }
                        2 => [uniform(-30.0, 30.0), 0.0, 0.0],
                        _ => [uniform(-6.0, 6.0), uniform(-6.0, 6.0), uniform(-6.0, 6.0)],
                    };
                    for (k, d) in dist.iter_mut().enumerate() {
                        *d = dist3(point, tables.center(k));
                    }
                    let z =
                        prepare_slices::<StratmannSwitch>(&tables, &dist, &mut order, &mut prod);
                    let mut want = vec![0.0; n];
                    for (m, w) in want.iter_mut().enumerate() {
                        *w = dense_product(&tables, point, m);
                    }
                    for m in 0..n {
                        if want[m] == 0.0 {
                            assert_eq!(
                                prod[m].to_bits(),
                                0.0_f64.to_bits(),
                                "case={case} cfg={cfg:?} point={point:?} atom={m}: \
                                 sparse product is non-zero for a killed atom"
                            );
                        } else {
                            // The SSF polynomial can exceed 1 by a few ulp at
                            // |nu| ~ A, so a product of many tiny factors may
                            // be a (physically meaningless) -1e-20; compare
                            // magnitudes.
                            assert!(
                                (prod[m] - want[m]).abs() <= 1e-12 * want[m].abs() + 1e-300,
                                "case={case} cfg={cfg:?} point={point:?} atom={m}: \
                                 sparse={} dense={}",
                                prod[m],
                                want[m]
                            );
                        }
                    }
                    let want_z: f64 = want.iter().sum();
                    assert!(
                        (z - want_z).abs() <= 1e-12 * want_z.abs().max(1e-300),
                        "case={case} cfg={cfg:?} point={point:?}: Z={z} want={want_z}"
                    );
                }
            }
        }
    }

    /// The Gaussian SS switch uses the same sparse structure with an
    /// asymmetric support; the sparse and dense products must agree for
    /// uniform radii and for a non-zero size adjustment (which shifts the
    /// kill cutoff).
    #[test]
    fn gaussian_ss_sparse_products_match_dense_reference() {
        let mut seed = 0x0dd0_1234_5678_9abc_u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut uniform = |lo: f64, hi: f64| lo + (hi - lo) * rnd();
        let configs = [
            PartitionConfig::gaussian_ss(),
            PartitionConfig::new(PartitionScheme::GaussianSs, RadiiConfig::pyscf()),
            PartitionConfig::new(
                PartitionScheme::GaussianSs,
                crate::partition::test_support::custom_radii(),
            ),
            PartitionConfig::new(PartitionScheme::GaussianSs, RadiiConfig::no_adjust()),
        ];
        let dense_product = |tables: &PartitionTables, point: [f64; 3], m: usize| {
            let n = tables.len();
            let mut p = 1.0;
            let dm = dist3(point, tables.center(m));
            for j in 0..n {
                if j == m {
                    continue;
                }
                let dj = dist3(point, tables.center(j));
                let inv = tables.inv_r_row(m * n + j);
                let mu = (dm - dj) * inv;
                let nu = mu + tables.a_ab_row(m * n + j) * (1.0 - mu * mu);
                p *= 0.5 * (1.0 - GaussianSsSwitch::switch(nu));
            }
            p
        };
        for case in 0..32 {
            let n = 1 + case % 9;
            let mut centers = Vec::with_capacity(n);
            for k in 0..n {
                let scale = if case % 3 == 0 { 8.0 } else { 3.0 };
                centers.push([
                    uniform(-scale, scale),
                    uniform(-scale, scale),
                    uniform(-scale, scale),
                ]);
                if k > 0 {
                    let prev = centers[k - 1];
                    if dist3(prev, centers[k]) < 0.4 {
                        centers[k][0] += 0.7;
                    }
                }
            }
            let charges: Vec<u32> = (0..n).map(|k| 1 + (k as u32 * 7) % 18).collect();
            for cfg in &configs {
                let tables = cfg.tables(&centers, &charges).unwrap();
                let mut dist = vec![0.0; n];
                let mut order = vec![0u32; n];
                let mut prod = vec![0.0; n];
                for p in 0..12 {
                    let point = match p % 6 {
                        0 => centers[p % n],
                        1 => {
                            let c = centers[p % n];
                            [c[0] + 1e-12, c[1] - 1e-12, c[2]]
                        }
                        2 => [uniform(-30.0, 30.0), 0.0, 0.0],
                        _ => [uniform(-6.0, 6.0), uniform(-6.0, 6.0), uniform(-6.0, 6.0)],
                    };
                    for (k, d) in dist.iter_mut().enumerate() {
                        *d = dist3(point, tables.center(k));
                    }
                    let z =
                        prepare_slices::<GaussianSsSwitch>(&tables, &dist, &mut order, &mut prod);
                    let mut want = vec![0.0; n];
                    for (m, w) in want.iter_mut().enumerate() {
                        *w = dense_product(&tables, point, m);
                    }
                    for m in 0..n {
                        if want[m] == 0.0 {
                            assert_eq!(
                                prod[m].to_bits(),
                                0.0_f64.to_bits(),
                                "case={case} cfg={cfg:?} point={point:?} atom={m}:                                  sparse product is non-zero for a killed atom"
                            );
                        } else {
                            assert!(
                                (prod[m] - want[m]).abs() <= 1e-12 * want[m].abs() + 1e-300,
                                "case={case} cfg={cfg:?} point={point:?} atom={m}:                                  sparse={} dense={}",
                                prod[m],
                                want[m]
                            );
                        }
                    }
                    let want_z: f64 = want.iter().sum();
                    assert!(
                        (z - want_z).abs() <= 1e-12 * want_z.abs().max(1e-300),
                        "case={case} cfg={cfg:?} point={point:?}: Z={z} want={want_z}"
                    );
                }
            }
        }
    }

    /// The candidate cut must never discard a non-zero product.
    #[test]
    fn candidate_cut_keeps_every_survivor() {
        // Long chain with strongly different radii: `rcut` is largest here.
        let n = 12;
        let mut centers = Vec::new();
        for k in 0..n {
            centers.push([1.7 * k as f64, 0.0, 0.0]);
        }
        let charges: Vec<u32> = (0..n).map(|k| if k % 2 == 0 { 1 } else { 35 }).collect();
        for cfg in [
            PartitionConfig::new(PartitionScheme::Stratmann, RadiiConfig::pyscf()),
            PartitionConfig::new(
                PartitionScheme::Stratmann,
                crate::partition::test_support::custom_radii(),
            ),
        ] {
            let tables = cfg.tables(&centers, &charges).unwrap();
            let mut dist = vec![0.0; n];
            let mut order = vec![0u32; n];
            let mut prod = vec![0.0; n];
            for p in 0..200 {
                let x = -2.0 + 30.0 * p as f64 / 199.0;
                let point = [x, 0.3 * (p as f64).sin(), 0.2];
                for (k, d) in dist.iter_mut().enumerate() {
                    *d = dist3(point, tables.center(k));
                }
                prepare_slices::<StratmannSwitch>(&tables, &dist, &mut order, &mut prod);
                let mut dmin = f64::INFINITY;
                for &d in &dist {
                    dmin = dmin.min(d);
                }
                assert!(tables.ss_rcut() >= 1.0);
                for k in 0..n {
                    if prod[k] != 0.0 {
                        assert!(
                            dist[k] <= tables.ss_rcut() * dmin * (1.0 + 1e-12),
                            "survivor {k} outside the candidate cut"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn switch_derivatives_match_finite_difference() {
        for k in 0..5000 {
            let x = -1.2 + 2.4 * k as f64 / 5000.0;
            let (f, df, ddf) = StratmannSwitch::switch_d2(x);
            assert_eq!(f.to_bits(), StratmannSwitch::switch(x).to_bits());
            let (f1, df1) = StratmannSwitch::switch_d1(x);
            assert_eq!(f.to_bits(), f1.to_bits());
            // `h = 1e-4` keeps the second difference above the rounding
            // noise of f ~ 1 near the cutoff.
            let h = 1e-4;
            let fd = (StratmannSwitch::switch(x + h) - StratmannSwitch::switch(x - h)) / (2.0 * h);
            assert!((df - fd).abs() < 1e-6, "x={x} df={df} fd={fd}");
            let fd2 = (StratmannSwitch::switch(x + h) - 2.0 * f + StratmannSwitch::switch(x - h))
                / (h * h);
            assert!((ddf - fd2).abs() < 1e-4, "x={x} ddf={ddf} fd2={fd2}");
            if x.abs() >= A {
                assert_eq!(df1.to_bits(), 0.0_f64.to_bits());
                assert_eq!(ddf.to_bits(), 0.0_f64.to_bits());
            }
        }
    }
}
