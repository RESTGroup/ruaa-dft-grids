//! Sparse eight-point lane gradient of the LKO partition.
//!
//! This is the derivative counterpart of [`super::gradient`]: the same
//! per-lattice-point quotient rule and generating-atom correction, but the
//! pair work follows the exact LKO locality
//!
//! ```text
//! S = {i : d_i <= d_min + 5},   T = {j : d_j <= d_min + 10}.
//! ```
//!
//! A pair can change a cell product only when at least one endpoint is in
//! `S` and the partner is in `T`; every other factor is exactly zero or one.
//! For each eight-point lane the kernel builds the per-point `S`/`T` masks
//! and the union `T` list exactly as the weight kernel does
//! ([`super::lane`]), multiplies the products through the same masks, and
//! then visits only the union pairs for the derivative sums.  The union list
//! is ascending, so each lane sees the canonical `(i, j)` order of the dense
//! kernel; pairs skipped between two inactive centres have exactly zero
//! coefficients.  The emitted weights and first derivatives are therefore
//! bitwise identical to the dense lane kernel up to signed zero.
//!
//! The product and derivative passes use [`super::tables::DerivTables`]
//! (`fac`, `k`, `a_ab`, `u_ab`), so no dense `O(natm^2)` state is kept.

use super::super::becke::gradient::{INVTOL, NOT_SELECTED};
use super::super::simd::{self, LaneAttrib, V};
use super::RCUT_LKO;
use super::lane::{le_bits, mask_of};
use super::tables::DerivTables;
use crate::partition::PartitionConfig;

/// Extra screening width (Bohr); see [`super`].
const MARGIN: f64 = 1e-9;

/// Per-thread scratch of the sparse lane gradient.
///
/// One lane (eight grid points) is processed at a time, so the per-atom
/// arrays are `O(natm)` rather than `O(natm * block)`.
#[derive(Debug, Clone)]
pub(crate) struct Scratch {
    pub(crate) block: usize,
    pub(crate) n: usize,
    pub(crate) nsel: usize,
    /// `[center]` lane-wise distances.
    dist: Vec<V>,
    /// `[center]` lane-wise cell products.
    prod: Vec<V>,
    /// `[(center * 3 + t)]` unit vectors `(R_center - r_g) / d`.
    drdist: Vec<V>,
    /// `[center]` bitset of live lanes.
    s_mask: Vec<u8>,
    /// `[center]` bitset of candidate lanes.
    t_mask: Vec<u8>,
    /// `[candidate]` union `T`, ascending.
    t_list: Vec<u32>,
    /// `[(slot * 3 + t)]` selected `dZ`.
    dz: Vec<V>,
    /// `[(slot * 3 + t)]` selected `dP`.
    dp: Vec<V>,
    /// `[t]` sum over all centres of `dZ`.
    dz_sum: [V; 3],
    /// `[t]` sum over all centres of `dP`.
    dp_sum: [V; 3],
}

impl Scratch {
    pub(crate) fn new(n: usize, nsel: usize, block: usize) -> Self {
        Self {
            block,
            n,
            nsel,
            dist: vec![V::zero(); n],
            prod: vec![V::zero(); n],
            drdist: vec![V::zero(); 3 * n],
            s_mask: vec![0; n],
            t_mask: vec![0; n],
            t_list: vec![0; n],
            dz: vec![V::zero(); nsel * 3],
            dp: vec![V::zero(); nsel * 3],
            dz_sum: [V::zero(); 3],
            dp_sum: [V::zero(); 3],
        }
    }
}

/// Evaluate the parent partition weights and selected gradients of one block.
#[allow(clippy::too_many_arguments)]
pub(crate) fn block_gradient(
    cfg: &PartitionConfig,
    tables: &DerivTables,
    sel: &[usize],
    slot_of: &[usize],
    coords: &[[f64; 3]],
    parents: &[u32],
    quad: &[f64],
    weights: &mut [f64],
    dw: &mut [f64],
    scratch: &mut Scratch,
) {
    let n = tables.len();
    let nb = coords.len();
    let block = scratch.block;
    let hardness = cfg.hardness;
    debug_assert_eq!(scratch.n, n);
    debug_assert_eq!(scratch.nsel, sel.len());
    debug_assert_eq!(parents.len(), nb);
    debug_assert_eq!(quad.len(), nb);
    debug_assert_eq!(weights.len(), nb);
    debug_assert_eq!(dw.len(), sel.len() * 3 * block);
    debug_assert!(nb <= block);

    let active_lanes = nb.div_ceil(simd::LANES);
    let one = simd::one();
    let zero = simd::zero();
    let half = simd::splat(0.5);
    let minus_one = simd::splat(-1.0);
    let s_width = simd::splat(RCUT_LKO + MARGIN);
    let t_width = simd::splat(2.0 * RCUT_LKO + MARGIN);
    let inv_tol = simd::splat(INVTOL);

    for lane in 0..active_lanes {
        let len = simd::lane_len(nb, lane);
        let start = lane * simd::LANES;
        let attr = LaneAttrib::from_parents(parents, lane, nb);
        let c = simd::gather_coords(coords, lane);
        let wq = simd::gather(quad, lane);

        // --- distances, unit vectors and screening masks --- //
        let mut dmin = V::splat(f64::INFINITY);
        for k in 0..n {
            let center = tables.center(k);
            let dx = simd::splat(center[0]) - c[0];
            let dy = simd::splat(center[1]) - c[1];
            let dz = simd::splat(center[2]) - c[2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            scratch.dist[k] = d;
            dmin = dmin.min(d);
            let inv = (one / d).mask_select(d.simd_gt(zero), zero);
            scratch.drdist[k * 3] = dx * inv;
            scratch.drdist[k * 3 + 1] = dy * inv;
            scratch.drdist[k * 3 + 2] = dz * inv;
        }
        let s_cut = dmin + s_width;
        let t_cut = dmin + t_width;
        let mut nt = 0usize;
        for k in 0..n {
            let d = scratch.dist[k];
            let s = le_bits(d, s_cut);
            let t = le_bits(d, t_cut);
            scratch.s_mask[k] = s;
            scratch.t_mask[k] = t;
            if t != 0 {
                scratch.t_list[nt] = k as u32;
                nt += 1;
            }
            scratch.prod[k] = one.mask_select(mask_of(s), zero);
        }

        // --- masked product pass over the union candidate list --- //
        for a in 0..nt {
            let i = scratch.t_list[a] as usize;
            let di = scratch.dist[i];
            let s_i = scratch.s_mask[i];
            let t_i = scratch.t_mask[i];
            for b in 0..a {
                let j = scratch.t_list[b] as usize;
                let need_i = s_i & scratch.t_mask[j];
                let need_j = scratch.s_mask[j] & t_i;
                if need_i | need_j == 0 {
                    continue;
                }
                let fac = simd::splat(tables.fac(i, j));
                let a_ab = simd::splat(tables.a_ab(i, j));
                let dj = scratch.dist[j];
                let g = ((di - dj) * fac).max(minus_one).min(one);
                let nu = g + a_ab * (one - g * g);
                let f = if hardness == 3 {
                    simd::step3(nu)
                } else {
                    simd::step_function(nu, hardness)
                };
                let si = half * (one - f);
                let sj = half * (one + f);
                scratch.prod[i] *= si.mask_select(mask_of(need_i), one);
                scratch.prod[j] *= sj.mask_select(mask_of(need_j), one);
            }
        }

        // --- normalisers --- //
        let mut z = zero;
        for &p in &scratch.prod {
            z += p;
        }
        let pg = attr.select_product(&scratch.prod);

        // --- derivative pass over the same union list --- //
        for v in scratch.dz.iter_mut() {
            *v = zero;
        }
        for v in scratch.dp.iter_mut() {
            *v = zero;
        }
        scratch.dz_sum = [zero; 3];
        scratch.dp_sum = [zero; 3];
        for a in 0..nt {
            let i = scratch.t_list[a] as usize;
            let di = scratch.dist[i];
            let s_i = scratch.s_mask[i];
            let t_i = scratch.t_mask[i];
            let prod_i = scratch.prod[i];
            let slot_i = slot_of[i];
            for b in 0..a {
                let j = scratch.t_list[b] as usize;
                let need_i = s_i & scratch.t_mask[j];
                let need_j = scratch.s_mask[j] & t_i;
                if need_i | need_j == 0 {
                    continue;
                }
                let slot_j = slot_of[j];
                let dj = scratch.dist[j];
                let fac = simd::splat(tables.fac(i, j));
                let kk = simd::splat(tables.k(i, j));
                let a_ab = simd::splat(tables.a_ab(i, j));
                let u = tables.u_ab(i, j);
                let u = [simd::splat(u[0]), simd::splat(u[1]), simd::splat(u[2])];
                let prod_j = scratch.prod[j];
                let g = ((di - dj) * fac).max(minus_one).min(one);
                let nu = g + a_ab * (one - g * g);
                let (f, df) = simd::value_d1_any(nu, hardness);
                let dnu = one - simd::splat(2.0) * a_ab * g;
                let si = half * (one - f);
                let sj = half * (one + f);
                let dlog_i = (-half * df * dnu) / si.max(inv_tol);
                let dlog_j = (half * df * dnu) / sj.max(inv_tol);
                let common_z = prod_i * dlog_i + prod_j * dlog_j;
                let coef_pg =
                    attr.select(i, prod_i * dlog_i, zero) + attr.select(j, prod_j * dlog_j, zero);

                let dr_i = [
                    scratch.drdist[i * 3],
                    scratch.drdist[i * 3 + 1],
                    scratch.drdist[i * 3 + 2],
                ];
                let dr_j = [
                    scratch.drdist[j * 3],
                    scratch.drdist[j * 3 + 1],
                    scratch.drdist[j * 3 + 2],
                ];
                let gk = g * kk;
                let gku = [gk * u[0], gk * u[1], gk * u[2]];
                let dmu_i = [
                    dr_i[0] * fac - gku[0],
                    dr_i[1] * fac - gku[1],
                    dr_i[2] * fac - gku[2],
                ];
                let dmu_j = [
                    gku[0] - dr_j[0] * fac,
                    gku[1] - dr_j[1] * fac,
                    gku[2] - dr_j[2] * fac,
                ];
                for t in 0..3 {
                    scratch.dz_sum[t] += common_z * (dmu_i[t] + dmu_j[t]);
                    scratch.dp_sum[t] += coef_pg * (dmu_i[t] + dmu_j[t]);
                    if slot_i != NOT_SELECTED {
                        scratch.dz[slot_i * 3 + t] += common_z * dmu_i[t];
                        scratch.dp[slot_i * 3 + t] += coef_pg * dmu_i[t];
                    }
                    if slot_j != NOT_SELECTED {
                        scratch.dz[slot_j * 3 + t] += common_z * dmu_j[t];
                        scratch.dp[slot_j * 3 + t] += coef_pg * dmu_j[t];
                    }
                }
            }
        }

        // --- normalisation and selected rows --- //
        let small = simd::abs_le(z, INVTOL);
        let inv_z = one / z;
        let q = pg / z;
        let w = one.mask_select(small, q);
        simd::store(&mut weights[start..start + len], w, len);
        for t in 0..3 {
            let sum_t = wq * inv_z * (scratch.dp_sum[t] - q * scratch.dz_sum[t]);
            for (slot, &atom) in sel.iter().enumerate() {
                let v = wq * inv_z * (scratch.dp[slot * 3 + t] - q * scratch.dz[slot * 3 + t]);
                let v = (v - sum_t).mask_select(attr.mask(atom), v);
                let v = zero.mask_select(small, v);
                let offset = (slot * 3 + t) * block + start;
                simd::store(&mut dw[offset..offset + len], v, len);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::gradient as dense;
    use super::*;
    use crate::partition::{PartitionConfigBuilder, PartitionScheme, RadiiConfig};

    fn configs() -> Vec<PartitionConfig> {
        let mut out = vec![PartitionConfig::new(
            PartitionScheme::Lko,
            RadiiConfig::pyscf(),
        )];
        for hardness in [1usize, 2, 4] {
            out.push(
                PartitionConfigBuilder::default()
                    .scheme(PartitionScheme::Lko)
                    .radii(RadiiConfig::pyscf())
                    .hardness(hardness)
                    .build()
                    .unwrap(),
            );
        }
        out
    }

    fn water() -> (Vec<u32>, Vec<[f64; 3]>) {
        (
            vec![8, 1, 1],
            vec![[0.0, 0.0, 0.0], [1.43, 0.0, 0.0], [-0.36, 1.38, 0.0]],
        )
    }

    fn benzene() -> (Vec<u32>, Vec<[f64; 3]>) {
        let mut charges = vec![6u32; 6];
        let mut centers = Vec::new();
        for k in 0..6 {
            let a = std::f64::consts::TAU * k as f64 / 6.0;
            centers.push([2.64 * a.cos(), 2.64 * a.sin(), 0.0]);
        }
        for k in 0..6 {
            let a = std::f64::consts::TAU * (k as f64 + 0.5) / 6.0;
            centers.push([4.72 * a.cos(), 4.72 * a.sin(), 0.0]);
            charges.push(1);
        }
        (charges, centers)
    }

    fn cluster() -> (Vec<u32>, Vec<[f64; 3]>) {
        let mut charges = Vec::new();
        let mut centers = Vec::new();
        for k in 0..24 {
            let t = k as f64;
            let c = [
                3.6 * (0.37 * t + 0.3).sin() + 0.4 * (t * 0.13).sin(),
                3.1 * (0.29 * t + 1.1).cos(),
                2.7 * (0.41 * t + 0.7).sin(),
            ];
            centers.push(c);
            charges.push(1 + (k as u32 * 7) % 18);
        }
        (charges, centers)
    }

    fn check_case(charges: &[u32], centers: &[[f64; 3]]) {
        let n = centers.len();
        for cfg in configs() {
            let tables = DerivTables::new(&cfg, centers, charges).unwrap();
            let mut selections: Vec<Vec<usize>> = vec![(0..n).collect(), vec![0], vec![n / 2]];
            if n > 2 {
                selections.push(vec![0, n / 2, n - 1]);
            }
            for sel in selections {
                let mut slot_of = vec![usize::MAX; n];
                for (slot, &atom) in sel.iter().enumerate() {
                    slot_of[atom] = slot;
                }
                for block in [64usize, 37] {
                    let nb = block.min(37);
                    let mut coords = Vec::with_capacity(nb);
                    let mut parents = Vec::with_capacity(nb);
                    let mut quad = Vec::with_capacity(nb);
                    for k in 0..nb {
                        let parent = (k * 7 + 3) % n;
                        let t = k as f64;
                        let local = if k % 11 == 0 {
                            [0.0, 0.0, 0.0]
                        } else {
                            [
                                3.4 * (0.31 * t + 0.1).sin(),
                                2.9 * (0.23 * t + 0.7).cos(),
                                3.2 * (0.17 * t + 1.3).sin(),
                            ]
                        };
                        coords.push([
                            centers[parent][0] + local[0],
                            centers[parent][1] + local[1],
                            centers[parent][2] + local[2],
                        ]);
                        parents.push(parent as u32);
                        quad.push(0.37 + 0.011 * k as f64);
                    }
                    let mut sd = dense::Scratch::new(n, sel.len(), block);
                    let mut ss = Scratch::new(n, sel.len(), block);
                    let mut wd = vec![0.0; nb];
                    let mut ws = vec![0.0; nb];
                    let mut dwd = vec![0.0; sel.len() * 3 * block];
                    let mut dws = vec![0.0; sel.len() * 3 * block];
                    dense::block_gradient(
                        &cfg, &tables, &sel, &slot_of, &coords, &parents, &quad, &mut wd, &mut dwd,
                        &mut sd,
                    );
                    block_gradient(
                        &cfg, &tables, &sel, &slot_of, &coords, &parents, &quad, &mut ws, &mut dws,
                        &mut ss,
                    );
                    let same =
                        |a: f64, b: f64| a.to_bits() == b.to_bits() || (a == 0.0 && b == 0.0);
                    for (idx, (&a, &b)) in wd.iter().zip(&ws).enumerate() {
                        assert!(same(a, b), "weight mismatch n={n} block={block} idx={idx}");
                    }
                    for (idx, (&a, &b)) in dwd.iter().zip(&dws).enumerate() {
                        assert!(
                            same(a, b),
                            "gradient mismatch n={n} sel={sel:?} block={block} idx={idx}: {a} vs {b}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn sparse_matches_the_dense_lane_reference() {
        for (charges, centers) in [water(), benzene(), cluster()] {
            check_case(&charges, &centers);
        }
    }
}
