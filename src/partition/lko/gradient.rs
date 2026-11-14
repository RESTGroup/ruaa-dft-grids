//! Dense analytic gradient of the LKO partition weights.
//!
//! This is the small-molecule lane kernel and the reference for
//! [`super::gradient_sparse`], which replaces the full pair scan with the
//! LKO near-set masks and union candidate list above the size threshold
//! (chosen empirically).
//! It is the classic Becke gradient kernel of
//! [`crate::partition::becke::gradient`] with the pair geometry replaced by
//! the LKO saturation function.  For one grid point `g` with generating atom
//! `A_g`, the weight is
//!
//! ```text
//! w_g = wq_g * Pbar_g / Z_g,
//! ```
//!
//! and the pair loop accumulates the logarithmic product derivatives
//!
//! ```text
//! dZ[A,t]   = sum_{pairs} (P_i dlog s_i + P_j dlog s_j) dmu/dR_At
//! dPbar[A,t] = sum_{pairs} (delta_{i,A_g} P_i dlog s_i
//!                         + delta_{j,A_g} P_j dlog s_j) dmu/dR_At
//! ```
//!
//! with the LKO chain rule (see [`super::tables`])
//!
//! ```text
//! dmu/dR_i = fac r_hat_i - mu k u_hat_ij
//! dmu/dR_j = -fac r_hat_j + mu k u_hat_ij.
//! ```
//!
//! Pairs whose pre-clamp `mu` lies outside `[-1, 1]` are **exactly constant**
//! in the cell functions (the factor is 0 or 1), so their derivative
//! contributions vanish: the kernel does not skip them explicitly because
//! `dmu` is zero there (the switch derivative $f^{\prime}(\pm1)$ vanishes). The
//! Hessian, whose `d^2mu` is formed from the unclamped `mu`, does skip them
//! explicitly.
//!
//! The quotient rule, the generating-atom translation correction and the
//! selection layout are identical to the classic scheme.

use super::super::becke::gradient::{INVTOL, NOT_SELECTED};
use super::super::simd::{self, LaneAttrib, V};
use super::tables::DerivTables;
use crate::partition::PartitionConfig;

/// Per-thread scratch for the gradient block kernel.
#[derive(Debug, Clone)]
pub(crate) struct Scratch {
    pub(crate) block: usize,
    pub(crate) nlane: usize,
    pub(crate) n: usize,
    pub(crate) nsel: usize,
    /// `[center * nlane + lane]` distances.
    pub(crate) dist: Vec<V>,
    /// `[center * nlane + lane]` cell products.
    pub(crate) prod: Vec<V>,
    /// `[(center * 3 + t) * nlane + lane]` unit vectors
    /// `(R_center - r_g) / |R_center - r_g|`.
    pub(crate) drdist: Vec<V>,
    /// `[(slot * 3 + t) * nlane + lane]` selected `dZ`.
    pub(crate) dz: Vec<V>,
    /// `[(slot * 3 + t) * nlane + lane]` selected `dPbar`.
    pub(crate) dp: Vec<V>,
    /// `[t * nlane + lane]` sum over all centres of `dZ`.
    pub(crate) dz_sum: Vec<V>,
    /// `[t * nlane + lane]` sum over all centres of `dPbar`.
    pub(crate) dp_sum: Vec<V>,
    /// Normalisation sum per lane.
    pub(crate) z: Vec<V>,
    /// Generating-atom numerator per lane.
    pub(crate) pg: Vec<V>,
    /// Gathered coordinates per lane.
    pub(crate) coords: Vec<[V; 3]>,
    /// Gathered quadrature weights per lane.
    pub(crate) quad: Vec<V>,
}

impl Scratch {
    pub(crate) fn new(n: usize, nsel: usize, block: usize) -> Self {
        let nlane = block.div_ceil(simd::LANES).max(1);
        Self {
            block,
            nlane,
            n,
            nsel,
            dist: vec![V::zero(); n * nlane],
            prod: vec![V::zero(); n * nlane],
            drdist: vec![V::zero(); 3 * n * nlane],
            dz: vec![V::zero(); nsel * 3 * nlane],
            dp: vec![V::zero(); nsel * 3 * nlane],
            dz_sum: vec![V::zero(); 3 * nlane],
            dp_sum: vec![V::zero(); 3 * nlane],
            z: vec![V::zero(); nlane],
            pg: vec![V::zero(); nlane],
            coords: vec![[V::zero(); 3]; nlane],
            quad: vec![V::zero(); nlane],
        }
    }
}

/// Evaluate the parent partition weight and the selected `dw` rows for one
/// fixed-lane block of points.
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
    let nlane = scratch.nlane;
    let active_lanes = nb.div_ceil(simd::LANES);
    debug_assert_eq!(scratch.nsel, sel.len());
    debug_assert_eq!(scratch.n, n);
    debug_assert_eq!(parents.len(), nb);
    debug_assert_eq!(quad.len(), nb);
    debug_assert_eq!(weights.len(), nb);
    debug_assert_eq!(dw.len(), sel.len() * 3 * block);
    debug_assert!(nb <= block);
    debug_assert!(active_lanes <= nlane);

    for lane in 0..active_lanes {
        scratch.coords[lane] = simd::gather_coords(coords, lane);
        scratch.quad[lane] = simd::gather(quad, lane);
    }
    let attrib: Vec<LaneAttrib> = (0..active_lanes)
        .map(|lane| LaneAttrib::from_parents(parents, lane, nb))
        .collect();

    fill_distances(tables, active_lanes, scratch);
    fill_products(cfg, tables, active_lanes, scratch);
    fill_normalisers(&attrib, scratch);

    scratch.dz.fill(V::zero());
    scratch.dp.fill(V::zero());
    scratch.dz_sum.fill(V::zero());
    scratch.dp_sum.fill(V::zero());

    let hardness = cfg.hardness;
    let one = simd::one();
    let zero = simd::zero();
    for i in 0..n {
        for j in 0..i {
            let fac = simd::splat(tables.fac(i, j));
            let kk = simd::splat(tables.k(i, j));
            let a_ab = simd::splat(tables.a_ab(i, j));
            let u = tables.u_ab(i, j);
            let u = [simd::splat(u[0]), simd::splat(u[1]), simd::splat(u[2])];
            let slot_i = slot_of[i];
            let slot_j = slot_of[j];
            for (lane, attr) in attrib.iter().enumerate() {
                let di = scratch.dist[i * nlane + lane];
                let dj = scratch.dist[j * nlane + lane];
                let prod_i = scratch.prod[i * nlane + lane];
                let prod_j = scratch.prod[j * nlane + lane];
                let mu = (di - dj) * fac;
                let g = mu.max(simd::splat(-1.0)).min(one);
                let nu = g + a_ab * (one - g * g);
                let (f, df) = simd::value_d1_any(nu, hardness);
                let dnu = one - simd::splat(2.0) * a_ab * g;
                let si = simd::splat(0.5) * (one - f);
                let sj = simd::splat(0.5) * (one + f);
                let dlog_i = (-simd::splat(0.5) * df * dnu) / si.max(simd::splat(INVTOL));
                let dlog_j = (simd::splat(0.5) * df * dnu) / sj.max(simd::splat(INVTOL));
                let common_z = prod_i * dlog_i + prod_j * dlog_j;
                let coef_pg =
                    attr.select(i, prod_i * dlog_i, zero) + attr.select(j, prod_j * dlog_j, zero);

                let dr_i0 = scratch.drdist[(i * 3) * nlane + lane];
                let dr_i1 = scratch.drdist[(i * 3 + 1) * nlane + lane];
                let dr_i2 = scratch.drdist[(i * 3 + 2) * nlane + lane];
                let dr_j0 = scratch.drdist[(j * 3) * nlane + lane];
                let dr_j1 = scratch.drdist[(j * 3 + 1) * nlane + lane];
                let dr_j2 = scratch.drdist[(j * 3 + 2) * nlane + lane];

                // dmu = fac * r_hat -+ mu k u_hat.  Clamped lanes need no explicit
                // mask: the switch derivative `df` is exactly zero at `nu = ±1`
                // (`p'(±1) = 0`), so every contribution is exactly zero.  (The
                // Hessian must and does mask them, because `f''(±1) != 0`.)
                let gk = g * kk;
                let gku = [gk * u[0], gk * u[1], gk * u[2]];
                let dmu_i = [
                    dr_i0 * fac - gku[0],
                    dr_i1 * fac - gku[1],
                    dr_i2 * fac - gku[2],
                ];
                let dmu_j = [
                    gku[0] - dr_j0 * fac,
                    gku[1] - dr_j1 * fac,
                    gku[2] - dr_j2 * fac,
                ];
                for t in 0..3 {
                    scratch.dz_sum[t * nlane + lane] += common_z * (dmu_i[t] + dmu_j[t]);
                    scratch.dp_sum[t * nlane + lane] += coef_pg * (dmu_i[t] + dmu_j[t]);
                    if slot_i != NOT_SELECTED {
                        scratch.dz[(slot_i * 3 + t) * nlane + lane] += common_z * dmu_i[t];
                        scratch.dp[(slot_i * 3 + t) * nlane + lane] += coef_pg * dmu_i[t];
                    }
                    if slot_j != NOT_SELECTED {
                        scratch.dz[(slot_j * 3 + t) * nlane + lane] += common_z * dmu_j[t];
                        scratch.dp[(slot_j * 3 + t) * nlane + lane] += coef_pg * dmu_j[t];
                    }
                }
            }
        }
    }

    for (lane, attr) in attrib.iter().enumerate() {
        let z = scratch.z[lane];
        let small = simd::abs_le(z, INVTOL);
        let inv_z = one / z;
        let q = scratch.pg[lane] / z;
        let w = one.mask_select(small, q);
        let start = lane * simd::LANES;
        let len = simd::lane_len(weights.len(), lane);
        simd::store(&mut weights[start..start + len], w, len);

        let wq = scratch.quad[lane];
        for t in 0..3 {
            let sum_t = wq
                * inv_z
                * (scratch.dp_sum[t * nlane + lane] - q * scratch.dz_sum[t * nlane + lane]);
            for (slot, &a) in sel.iter().enumerate() {
                let dp = scratch.dp[(slot * 3 + t) * nlane + lane];
                let dz = scratch.dz[(slot * 3 + t) * nlane + lane];
                let v = wq * inv_z * (dp - q * dz);
                let v = (v - sum_t).mask_select(attr.mask(a), v);
                let v = zero.mask_select(small, v);
                let offset = (slot * 3 + t) * block + start;
                simd::store(&mut dw[offset..offset + len], v, len);
            }
        }
    }
}

/// One distance pass plus the `dR_dist` unit vectors.
fn fill_distances(tables: &DerivTables, active_lanes: usize, scratch: &mut Scratch) {
    let n = tables.len();
    let nlane = scratch.nlane;
    for k in 0..n {
        let c = tables.center(k);
        let cx = simd::splat(c[0]);
        let cy = simd::splat(c[1]);
        let cz = simd::splat(c[2]);
        for lane in 0..active_lanes {
            let p = scratch.coords[lane];
            let dx = cx - p[0];
            let dy = cy - p[1];
            let dz = cz - p[2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            let inv = (simd::one() / d).mask_select(d.simd_gt(simd::zero()), simd::zero());
            scratch.dist[k * nlane + lane] = d;
            scratch.drdist[(k * 3) * nlane + lane] = dx * inv;
            scratch.drdist[(k * 3 + 1) * nlane + lane] = dy * inv;
            scratch.drdist[(k * 3 + 2) * nlane + lane] = dz * inv;
        }
    }
}

/// The LKO pair-product pass (clamp + classic switch).
fn fill_products(
    cfg: &PartitionConfig,
    tables: &DerivTables,
    active_lanes: usize,
    scratch: &mut Scratch,
) {
    let n = tables.len();
    let nlane = scratch.nlane;
    let one = simd::one();
    for k in 0..n {
        for lane in 0..active_lanes {
            scratch.prod[k * nlane + lane] = one;
        }
    }
    let hardness = cfg.hardness;
    for i in 0..n {
        for j in 0..i {
            let fac = simd::splat(tables.fac(i, j));
            let a_ab = simd::splat(tables.a_ab(i, j));
            for lane in 0..active_lanes {
                let di = scratch.dist[i * nlane + lane];
                let dj = scratch.dist[j * nlane + lane];
                let g = ((di - dj) * fac).max(simd::splat(-1.0)).min(simd::one());
                let nu = g + a_ab * (one - g * g);
                let f = if hardness == 3 {
                    simd::step3(nu)
                } else {
                    simd::step_function(nu, hardness)
                };
                let si = simd::splat(0.5) * (one - f);
                let sj = simd::splat(0.5) * (one + f);
                let pi = scratch.prod[i * nlane + lane];
                let pj = scratch.prod[j * nlane + lane];
                scratch.prod[i * nlane + lane] = pi * si;
                scratch.prod[j * nlane + lane] = pj * sj;
            }
        }
    }
}

/// `Z` and `Pbar` per lane.
fn fill_normalisers(attrib: &[LaneAttrib], scratch: &mut Scratch) {
    let n = scratch.n;
    let nlane = scratch.nlane;
    for (lane, attr) in attrib.iter().enumerate() {
        let mut z = simd::zero();
        for k in 0..n {
            z += scratch.prod[k * nlane + lane];
        }
        let mut pg = simd::zero();
        for k in 0..n {
            pg = attr.select(k, scratch.prod[k * nlane + lane], pg);
        }
        scratch.z[lane] = z;
        scratch.pg[lane] = pg;
    }
}
