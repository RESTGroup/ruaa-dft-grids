//! Laqua–Kussmann–Ochsenfeld (LKO) saturation partition (PySCF
//! `gen_grid.becke_lko`; H. Laqua, J. Kussmann, C. Ochsenfeld, JCP 149,
//! 204111 (2018)).
//!
//! The LKO cell factors are the classic Becke factors with the interatomic
//! distance replaced by the saturating function
//!
//! ```text
//! sat(r) = 5 (1 - exp(-sum_{m=1..12} (r/5)^m / m)),   fac_ij = 1 / sat(R_ij),
//! mu_ij  = (d_i - d_j) fac_ij,   g = clamp(mu_ij, -1, 1),
//! nu_ij  = g + a_ij (1 - g^2),   s_ij = 1/2 (1 - f(nu_ij)),
//! s_ji   = 1/2 (1 + f(nu_ij)),
//! ```
//!
//! with the classic Becke polynomial `f` composed `hardness` times.
//!
//! # Exact locality
//!
//! Because `sat(r) < 5` for every finite `r`, we have `fac_ij > 1/5`, and for
//! a point with nearest-centre distance `d_min`:
//!
//! * if `d_i > d_min + 5` the pair `(i, j0)` with the nearest centre `j0`
//!   has `mu > 5 * (1/5) = 1`, so `g = 1`, `f = 1` and `s_ij = 0` **exactly**:
//!   `P_i` is exactly zero whatever the multiplication order;
//! * a partner `j` of a live atom `i` (`d_i <= d_min + 5`) changes `P_i`
//!   only if `d_j - d_i < 5`, i.e. only partners with
//!   `d_j <= d_min + 10` can matter; all other factors are exactly `1`.
//!
//! The sparse kernel below evaluates only the live set `S = {i : d_i <=
//! d_min + 5}` against the candidate set `T = {i : d_i <= d_min + 10}`, in
//! the canonical pair order of the reference (`i` ascending, `j` ascending)
//! so every product sees exactly the reference multiplication sequence and
//! the result is **bitwise** identical.  `S` and `T` are widened by a
//! 1e-9 Bohr margin so the exact-$\pm1$ clamp is guaranteed in the presence of
//! floating-point rounding of `d_i - d_j`.
//!
//! The scalar reference implementation [`point_weights_all`] is kept
//! unchanged (bitwise/round-off reference and the PySCF construction).
//! [`block_weights`] dispatches between the screened
//! eight-point lane kernel ([`lane`], the main kernel) and the dense
//! eight-point lane kernel ([`dense`], the small-molecule fallback) with the
//! [`use_sparse_lko`] crossover.
//!
//! Coincident centres (`R_ij = 0`) are a degenerate input: `sat(0) = 0`
//! makes `fac_ij` infinite and the pair factor has no defined limit, so the
//! kernels do not guarantee any particular value for such geometries.

use super::kernel;
use super::{PartitionConfig, PartitionTables};

/// PySCF `VXCgen_grid_lko` saturation function: `_lko_sat_func`
/// (`RCUT_LKO = 5`, `MC_LKO = 12`).
#[inline]
pub(crate) fn lko_sat_func(x: f64) -> f64 {
    const RCUT: f64 = 5.0;
    const MC: usize = 12;
    let x = x / RCUT;
    let mut tot = 0.0;
    let mut xm = 1.0;
    for m in 1..=MC {
        xm *= x;
        tot += xm / m as f64;
    }
    RCUT * (1.0 - (-tot).exp())
}

/// PySCF `VXCgen_grid_lko` pair screening threshold (`RCUT_LKO`).
pub(crate) const RCUT_LKO: f64 = 5.0;

/// LKO saturation function with its first two derivatives,
/// `(sat, sat', sat'', exp(-S))`.
///
/// `sat(r) = 5 (1 - exp(-S(r)))`, `S(r) = sum_{m=1..12} (r/5)^m / m`;
/// `S'(r) = (1/5) sum_{k=0..11} (r/5)^k`,
/// `S''(r) = (1/25) sum_{k=1..11} k (r/5)^(k-1)`, and
/// `sat' = 5 e^{-S} S'`, `sat'' = 5 e^{-S} (S'' - S'^2)`.
///
/// `e^{-S}` is recovered as `1 - sat/5`, so the plateau where `sat` rounds
/// to exactly `5` has `sat' = sat'' = 0` exactly (a constant `fac = 0.2`).
/// The value `sat` is bitwise [`lko_sat_func`].
#[inline]
pub(crate) fn lko_sat_d1_d2(x: f64) -> (f64, f64, f64) {
    const RCUT: f64 = 5.0;
    const MC: usize = 12;
    let sat = lko_sat_func(x);
    let e = 1.0 - sat / RCUT;
    let x = x / RCUT;
    // S'(r) and S''(r) with the same power recursion as `lko_sat_func`.
    let mut sp = 0.0; // sum_{k=0..11} x^k
    let mut spp = 0.0; // sum_{k=1..11} k x^{k-1}
    let mut xm = 1.0;
    for m in 1..=MC {
        sp += xm;
        if m >= 2 {
            spp += (m - 1) as f64 * xm / x;
        }
        xm *= x;
    }
    let sp = sp / RCUT;
    let spp = spp / (RCUT * RCUT);
    let d1 = RCUT * e * sp;
    let d2 = RCUT * e * (spp - sp * sp);
    (sat, d1, d2)
}

/// Whether the screened (masked-lane) LKO kernel beats the dense lane
/// kernel.
///
/// Both kernels vectorise over eight grid points.  The dense kernel visits
/// all `natm (natm-1)/2` pairs and needs no per-atom masks; the screened
/// kernel visits the union of the eight points' candidate pairs and pays for
/// the mask construction and the masked multiply.  The cut is chosen
/// empirically and shifts with the compiled ISA because the dense kernel
/// benefits more from the wider ISA; a wrong call costs little on the
/// affected sizes.
#[inline]
pub(crate) fn use_sparse_lko(n: usize) -> bool {
    if cfg!(target_feature = "avx512f") {
        n >= 40
    } else if cfg!(target_feature = "avx2") {
        n >= 32
    } else {
        n >= 24
    }
}

/// Whether the sparse lane gradient beats the dense lane kernel.
///
/// The dense kernel already vectorises eight points; the sparse kernel adds
/// the per-lane `S`/`T` masks and the union candidate list, which pays off
/// when the candidate set is much smaller than the atom set.  The cut is
/// chosen empirically.
#[inline]
pub(crate) fn use_sparse_lko_gradient(n: usize) -> bool {
    n >= 32
}

/// Whether the sparse Hessian kernel beats the dense pointwise kernel.
///
/// The dense kernel's vectorised arithmetic is cheaper per pair for a small
/// molecule, while the sparse kernel avoids the `O(natm^2)` scan and the
/// dense log-derivative table.  The cut is chosen empirically; both kernels
/// agree to signed zero.
#[inline]
pub(crate) fn use_sparse_lko_hessian(n: usize) -> bool {
    n >= 36
}

/// LKO cell functions of all centers at one point (Bohr).
///
/// This is the serial reference: it evaluates every pair and screens only
/// with PySCF's coarse `d_min + 5` rule.  `dist` and `prod` are per-center
/// scratch of length `n`.
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
        *d = super::dist3(point, tables.center(i));
    }
    let dmin = dist.iter().copied().fold(f64::INFINITY, f64::min);
    prod.fill(1.0);
    for i in 0..n {
        for j in 0..i {
            if dist[i] > dmin + RCUT_LKO && dist[j] > dmin + RCUT_LKO {
                continue;
            }
            let fac = 1.0 / lko_sat_func(tables.r_ab(i, j));
            let mut g = ((dist[i] - dist[j]) * fac).clamp(-1.0, 1.0);
            g += tables.a_ab(i, j) * (1.0 - g * g);
            let f = cfg.switch_value(g);
            prod[i] *= 0.5 * (1.0 - f);
            prod[j] *= 0.5 * (1.0 + f);
        }
    }
    kernel::finish_all(prod, out);
}

/// LKO cell functions of `center` for one block of grid points.
pub(crate) fn block_weights(
    cfg: &PartitionConfig,
    tables: &PartitionTables,
    center: usize,
    grid: &[[f64; 3]],
    out: &mut [f64],
    scratch: &mut Scratch,
) {
    match scratch {
        Scratch::Masked(s) => lane::block_weights(cfg, tables, center, grid, out, s),
        Scratch::Dense(s) => dense::block_weights(cfg, tables, center, grid, out, s),
    }
}

/// Per-worker scratch of the LKO kernels.
pub(crate) enum Scratch {
    /// Screened masked-lane kernel ([`lane`]).
    Masked(lane::Scratch),
    /// Dense eight-point lane fallback ([`dense`]).
    Dense(kernel::LaneScratch),
}

impl Scratch {
    pub(crate) fn new(num_centers: usize, block: usize) -> Self {
        if use_sparse_lko(num_centers) {
            Self::Masked(lane::Scratch::new(num_centers, block))
        } else {
            Self::Dense(kernel::LaneScratch::new(num_centers, block))
        }
    }
}

pub(crate) mod dense;
pub(crate) mod deriv;
pub(crate) mod gradient;
pub(crate) mod gradient_sparse;
pub(crate) mod hessian;
pub(crate) mod hessian_sparse;
pub(crate) mod lane;
pub(crate) mod tables;

#[cfg(test)]
mod tests {
    use super::{RCUT_LKO, lko_sat_func, use_sparse_lko_gradient, use_sparse_lko_hessian};
    use crate::partition::kernel::LaneScratch;
    use crate::partition::{
        PartitionConfig, PartitionConfigBuilder, PartitionScheme, PartitionTables, RadiiConfig,
    };

    #[test]
    fn sparse_dispatch_boundaries_are_stable() {
        assert!(!use_sparse_lko_gradient(31));
        assert!(use_sparse_lko_gradient(32));
        assert!(!use_sparse_lko_hessian(35));
        assert!(use_sparse_lko_hessian(36));
    }

    #[test]
    fn lko_saturation_endpoints() {
        assert_eq!(lko_sat_func(0.0), 0.0);
        assert!((lko_sat_func(1e-6) - 1e-6).abs() < 1e-12);
        assert!((lko_sat_func(100.0) - 5.0).abs() < 1e-12);
    }

    /// `sat(r) <= 5` (hence `fac >= 1/5`) for every finite `r`, which is what
    /// makes the `d_min + 5` locality argument exact.  `sat` reaches `5.0`
    /// exactly once the truncated series exceeds ~37 (`exp(-S)` drops below
    /// half an ulp of 5); `fac` is then exactly `0.2` and the argument only
    /// needs `fac >= 0.2`.
    #[test]
    fn saturation_is_below_or_at_cut() {
        for k in 0..200_000 {
            let r = 1e-6 + 200.0 * k as f64 / 200_000.0;
            let sat = lko_sat_func(r);
            assert!(sat <= RCUT_LKO, "sat({r}) = {sat}");
            assert!(1.0 / sat >= 0.2, "fac({r}) = {}", 1.0 / sat);
        }
        assert_eq!(lko_sat_func(7.5), 5.0);
        assert_eq!(1.0 / lko_sat_func(7.5), 0.2);
    }

    #[test]
    fn lko_weights_sum_to_one_and_differ_from_becke() {
        let centers = [[-4.0, 0.0, 0.0], [4.0, 0.0, 0.0]];
        let charges = [1u32, 9];
        let lko = PartitionConfig::new(PartitionScheme::Lko, RadiiConfig::pyscf());
        let becke = PartitionConfig::new(PartitionScheme::Becke, RadiiConfig::pyscf());
        let mut diff = 0.0_f64;
        for &x in &[-5.0, -2.0, -0.5, 0.0, 0.5, 2.0, 5.0] {
            let pt = [x, 0.1, -0.2];
            let w = lko.weights(pt, &centers, &charges).unwrap();
            let total: f64 = w.iter().sum();
            assert!((total - 1.0).abs() < 1e-12, "sum {total} at {pt:?}");
            let wb = becke.weights(pt, &centers, &charges).unwrap();
            diff = diff.max((w[0] - wb[0]).abs());
        }
        assert!(diff > 1e-6, "LKO did not differ from Becke");
    }

    /// Build the tables and scratch for a scheme and compare the blocked
    /// kernel with the per-point reference for three centers.
    fn check_block_bitwise(
        cfg: &PartitionConfig,
        centers: &[[f64; 3]],
        charges: &[u32],
        grid: &[[f64; 3]],
    ) {
        let tables = PartitionTables::new(cfg, centers, charges).unwrap();
        let block = 8;
        for center in [0usize, tables.len() / 2, tables.len() - 1] {
            let mut scratch = super::Scratch::new(tables.len(), block);
            let mut got = vec![0.0; grid.len()];
            for (chunk_index, chunk) in grid.chunks(block).enumerate() {
                let start = chunk_index * block;
                let out = &mut got[start..start + chunk.len()];
                super::block_weights(cfg, &tables, center, chunk, out, &mut scratch);
            }
            for (p, &point) in grid.iter().enumerate() {
                let mut want = vec![0.0; tables.len()];
                let mut dist = vec![0.0; tables.len()];
                let mut prod = vec![0.0; tables.len()];
                super::point_weights_all(cfg, &tables, point, &mut want, &mut dist, &mut prod);
                assert_eq!(
                    got[p].to_bits(),
                    want[center].to_bits(),
                    "center {center} point {p} ({point:?}): {} vs {}",
                    got[p],
                    want[center]
                );
            }
        }
    }

    /// The masked lane kernel must be bitwise identical to the reference on
    /// a molecule where screening actually fires.
    #[test]
    fn masked_kernel_is_bitwise_identical() {
        let mut centers = Vec::new();
        for k in 0..60 {
            let t = k as f64;
            centers.push([2.1 * t.cos(), 0.8 * t, 1.3 * t.sin()]);
        }
        let charges: Vec<u32> = (0..60).map(|k| 1 + (k as u32 * 5) % 18).collect();
        let grid: Vec<[f64; 3]> = (0..777)
            .map(|k| {
                let t = k as f64 * 0.0131;
                [
                    5.0 * t.cos() + 0.7 * t.sin(),
                    4.0 * t.sin() - 0.3,
                    3.5 * t.cos() + 0.4 * t,
                ]
            })
            .collect();
        let cfg = PartitionConfig::new(PartitionScheme::Lko, RadiiConfig::pyscf());
        assert!(super::use_sparse_lko(centers.len()));
        check_block_bitwise(&cfg, &centers, &charges, &grid);
    }

    /// The dense lane fallback must be bitwise identical as well.
    #[test]
    fn dense_kernel_is_bitwise_identical() {
        let centers = [[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0], [0.0, 1.5, 0.2]];
        let charges = [1u32, 1, 8];
        let grid: Vec<[f64; 3]> = (0..101)
            .map(|k| [-2.0 + 4.0 * k as f64 / 100.0, 0.13, -0.21])
            .collect();
        let cfg = PartitionConfig::new(PartitionScheme::Lko, RadiiConfig::pyscf());
        assert!(!super::use_sparse_lko(centers.len()));
        let tables = PartitionTables::new(&cfg, &centers, &charges).unwrap();
        let mut lane = LaneScratch::new(tables.len(), 104);
        let mut got = vec![0.0; grid.len()];
        super::dense::block_weights(&cfg, &tables, 1, &grid, &mut got, &mut lane);
        for (p, &point) in grid.iter().enumerate() {
            let mut want = vec![0.0; tables.len()];
            let mut dist = vec![0.0; tables.len()];
            let mut prod = vec![0.0; tables.len()];
            super::point_weights_all(&cfg, &tables, point, &mut want, &mut dist, &mut prod);
            assert_eq!(
                got[p].to_bits(),
                want[1].to_bits(),
                "dense LKO point {p}: {} vs {}",
                got[p],
                want[1]
            );
        }
    }

    /// Generic hardness must also stay bitwise identical.
    #[test]
    fn masked_kernel_generic_hardness_is_bitwise_identical() {
        let centers = [
            [0.0; 3],
            [2.4, 0.1, 0.0],
            [0.2, 2.6, 0.3],
            [-2.0, -1.0, 1.1],
        ];
        let charges = [7u32, 6, 8, 1];
        let grid: Vec<[f64; 3]> = (0..257)
            .map(|k| {
                let t = k as f64 * 0.027;
                [
                    3.5 * t.cos() + 0.3 * t.sin(),
                    2.5 * t.sin() - 0.2,
                    2.0 * t.cos() + 0.1 * t,
                ]
            })
            .collect();
        for hardness in [1usize, 2, 4] {
            let cfg = PartitionConfigBuilder::default()
                .scheme(PartitionScheme::Lko)
                .radii(RadiiConfig::pyscf())
                .hardness(hardness)
                .build()
                .unwrap();
            check_block_bitwise(&cfg, &centers, &charges, &grid);
        }
    }

    /// The tests above use the production gate; this one forces each kernel on
    /// a molecule that crosses the crossover so both paths are covered
    /// independent of the compiled ISA.
    #[test]
    fn both_kernels_are_bitwise_identical_across_the_gate() {
        let mut centers = Vec::new();
        for k in 0..48 {
            let t = k as f64;
            centers.push([2.3 * t.cos(), 0.9 * t, 1.1 * t.sin()]);
        }
        let charges: Vec<u32> = (0..48).map(|k| 1 + (k as u32 * 7) % 18).collect();
        let grid: Vec<[f64; 3]> = (0..321)
            .map(|k| {
                let t = k as f64 * 0.021;
                [
                    7.0 * t.cos() + 0.5 * t.sin(),
                    5.0 * t.sin() - 0.4,
                    4.0 * t.cos() + 0.3 * t,
                ]
            })
            .collect();
        let cfg = PartitionConfig::new(PartitionScheme::Lko, RadiiConfig::pyscf());
        let tables = PartitionTables::new(&cfg, &centers, &charges).unwrap();
        let block = 8;
        for center in [0usize, 7, 47] {
            for masked in [false, true] {
                let mut got = vec![0.0; grid.len()];
                match masked {
                    true => {
                        let mut s = super::lane::Scratch::new(tables.len(), block);
                        for (ci, chunk) in grid.chunks(block).enumerate() {
                            let start = ci * block;
                            let out = &mut got[start..start + chunk.len()];
                            super::lane::block_weights(&cfg, &tables, center, chunk, out, &mut s);
                        }
                    }
                    false => {
                        let mut s = crate::partition::kernel::LaneScratch::new(tables.len(), block);
                        for (ci, chunk) in grid.chunks(block).enumerate() {
                            let start = ci * block;
                            let out = &mut got[start..start + chunk.len()];
                            super::dense::block_weights(&cfg, &tables, center, chunk, out, &mut s);
                        }
                    }
                }
                for (p, &point) in grid.iter().enumerate() {
                    let mut want = vec![0.0; tables.len()];
                    let mut dist = vec![0.0; tables.len()];
                    let mut prod = vec![0.0; tables.len()];
                    super::point_weights_all(&cfg, &tables, point, &mut want, &mut dist, &mut prod);
                    assert_eq!(
                        got[p].to_bits(),
                        want[center].to_bits(),
                        "masked={masked} center {center} point {p}: {} vs {}",
                        got[p],
                        want[center]
                    );
                }
            }
        }
    }
}
