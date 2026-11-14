//! Analytic first and second derivatives of the compact-support
//! (SSF and Gaussian SS) partitions.
//!
//! The value kernel ([`super`]) already returns the survivors and their cell
//! products.  This module adds the pair passes on top of that structure:
//!
//! ```text
//! P_M         = prod_{j != M} 1/2 (1 - f(nu_Mj))
//! d log P_M   = sum_j l_Mj dmu_Mj
//! d^2 log P_M = sum_j (l_Mj d^2mu_Mj + ll_Mj dmu_Mj (x) dmu_Mj)
//! d P_M       = P_M d log P_M
//! d^2 P_M     = P_M [ d^2 log P_M + d log P_M (x) d log P_M ]
//! ```
//!
//! Only pairs strictly inside the switch support have a non-zero `l`/`ll`
//! (the support is symmetric for SSF and asymmetric for Gaussian SS); every
//! other pair has a constant factor (exactly one, or a zero that already
//! killed the product).  The pair work is therefore proportional to the
//! active pairs incident to the survivors, not to `natm^2`.
//!
//! The generating-atom (grid shift) correction is the same as for the
//! classic scheme: all partial derivatives are taken at fixed grid
//! coordinate, and the parent row is corrected with the sums over all rows,
//! which are accumulated directly (see the [partition derivative
//! API](crate::doc::partition_derivatives) §6).
//!
//! # Sparse second-order structure
//!
//! `d^2 Z` is never materialised as `3natm x 3natm`.  The kernel keeps
//! * `dz`, `dp`, `dq` — the first derivatives of `Z`, `Pbar` and `q`;
//! * `ds`, `dt` — the summed pairwise + cross second derivatives
//!   `sum_A d^2Z[A,t,B,s]` and `sum_A d^2Pbar[A,t,B,s]` (`9 natm` each);
//! * one dense `3 natm` row at a time for a selected derivative row.
//!
//! A selected row is assembled from the survivors whose support contains
//! that row atom, so a partial selection costs the rank-one row plus a
//! handful of pair re-evaluations instead of the full `O(natm^3)` sweep.

use std::any::TypeId;
use std::marker::PhantomData;

use super::{GaussianSsSwitch, SsSwitch, StratmannSwitch, active};
use crate::error::Result;
use crate::partition::PartitionConfig;
use crate::partition::becke::DerivTables;
use crate::partition::deriv::{DerivScheme, DerivScratch};
use crate::partition::simd::{self, V};

/// Relative threshold of the existing derivative kernels.
const INVTOL: f64 = 1e-14;

/// Per-worker derivative scratch.  One type serves the blocked gradient and
/// the pointwise Hessian; every buffer is bounded by `O(natm)` or
/// `O(nsel * natm^2)` (the public raw Hessian output itself).
#[derive(Debug, Clone)]
pub(crate) struct Scratch {
    block: usize,
    n: usize,
    /// Distance field of one eight-point lane.
    lane: Vec<V>,
    /// `[center]` distances of the current point.
    dist: Vec<f64>,
    /// `[candidate]` distance-sorted candidate atoms.
    order: Vec<u32>,
    /// `[center]` cell products (exact zeros for killed atoms).
    prod: Vec<f64>,
    /// `[center * 3 + t]` unit vectors `(R_center - r_g) / d`.
    r: Vec<f64>,
    /// `[center * 9 + t * 3 + s]` = `(I - r r^T) / d`.
    prm: Vec<f64>,
    /// `[atom * 3 + t]` first derivative of `Z`.
    dz: Vec<f64>,
    /// `[atom * 3 + t]` first derivative of `Pbar`.
    dp: Vec<f64>,
    /// `[atom * 3 + t]` first derivative of `q = Pbar / Z`.
    dq: Vec<f64>,
    /// `[atom * 3 + t]` log-derivative of the current product.
    g: Vec<f64>,
    /// Survivor atoms, ascending.
    surv: Vec<u32>,
    /// Survivor index of an atom, `u32::MAX` otherwise.
    si_of: Vec<u32>,
    /// Concatenated supports (`M` first, then its active partners ascending).
    sup_atoms: Vec<u32>,
    /// Log-derivative values at [`Self::sup_atoms`].
    sup_vals: Vec<[f64; 3]>,
    /// Offset of each survivor's support in [`Self::sup_atoms`].
    sup_start: Vec<usize>,
    /// Column sums `sum_A g_M[A,t]` per survivor.
    g_sum: Vec<[f64; 3]>,
    /// `sum_A` of the pairwise+cross `d^2 Z` (`(t*n + B)*3 + s`).
    ds: Vec<f64>,
    /// `sum_A` of the pairwise+cross `d^2 Pbar`.
    dt: Vec<f64>,
    /// `sum_A d^2 w[A,t,B,s]` (`(B*3+t)*3+s`).
    full_a: Vec<f64>,
    /// `sum_B full_a[B,t,s]`.
    full_ab: [f64; 9],
    /// Pairwise `d^2 Z` rows of the selected atom, `[t, c]`.
    ddz_row: Vec<f64>,
    /// Pairwise `d^2 Pbar` rows of the selected atom, `[t, c]`.
    ddp_row: Vec<f64>,
    /// Rank-one cross rows `sum_M P_M l_M[At] l_M[c]`, `[t, c]`.
    c_row: Vec<f64>,
    /// Per-point gradient row `[nsel, 3]`.
    dw_row: Vec<f64>,
    /// Corrected raw Hessian rows, `[nsel, 3, 3n]`.
    out: Vec<f64>,
    /// Generating atom of the current point (Hessian row assembly).
    parent: usize,
}

impl Scratch {
    pub(crate) fn new(n: usize, nsel: usize, block: usize) -> Self {
        let n3 = 3 * n;
        Self {
            block,
            n,
            lane: vec![V::zero(); n],
            dist: vec![0.0; n],
            order: vec![0; n],
            prod: vec![0.0; n],
            r: vec![0.0; n3],
            prm: vec![0.0; 9 * n],
            dz: vec![0.0; n3],
            dp: vec![0.0; n3],
            dq: vec![0.0; n3],
            g: vec![0.0; n3],
            surv: Vec::with_capacity(n),
            si_of: vec![u32::MAX; n],
            sup_atoms: Vec::with_capacity(n),
            sup_vals: Vec::with_capacity(n),
            sup_start: Vec::with_capacity(n),
            g_sum: Vec::with_capacity(n),
            ds: vec![0.0; 9 * n],
            dt: vec![0.0; 9 * n],
            full_a: vec![0.0; 9 * n],
            full_ab: [0.0; 9],
            ddz_row: vec![0.0; 3 * n3],
            ddp_row: vec![0.0; 3 * n3],
            c_row: vec![0.0; 3 * n3],
            dw_row: vec![0.0; nsel * 3],
            out: vec![0.0; nsel * 3 * n3],
            parent: usize::MAX,
        }
    }

    /// Estimated bytes of one per-worker Hessian scratch.
    pub(crate) fn estimated_bytes(n: usize, nsel: usize) -> usize {
        let n3 = 3 * n;
        (n3 + 9 * n + 4 * n3 + 3 * 9 * n + 4 * n3 + n) * 8 + nsel * 3 * n3 * 8
    }
}

/// Visit the active partners `j` of product `i`.
///
/// The `nu` test is evaluated eight partners at a time; only the active
/// lanes enter the scalar `visitor`.  Active means strictly inside the
/// switch support; the support may be asymmetric.
#[inline]
fn for_each_active<S: SsSwitch, F: FnMut(usize)>(
    tables: &DerivTables,
    dist: &[f64],
    i: usize,
    mut visitor: F,
) {
    let n = dist.len();
    let row = i * n;
    let di = V::splat(dist[i]);
    let lo = V::splat(S::MU_MIN);
    let hi = V::splat(S::MU_MAX);
    let mut j = 0;
    while j + simd::LANES <= n {
        let dj = V::from_slice(&dist[j..j + simd::LANES]);
        let inv = V::from_slice(tables.inv_r_slice(row + j, simd::LANES));
        let a = V::from_slice(tables.a_slice(row + j, simd::LANES));
        let mu = (di - dj) * inv;
        let nu = mu + a * (V::one() - mu * mu);
        let mask = nu.simd_gt(lo) & nu.simd_lt(hi);
        if mask.any_true() {
            let lanes = mask.0;
            for (l, &hit) in lanes.iter().enumerate() {
                if hit {
                    visitor(j + l);
                }
            }
        }
        j += simd::LANES;
    }
    while j < n {
        let inv = tables.inv_r(i, j);
        if inv != 0.0 {
            let mu = (dist[i] - dist[j]) * inv;
            let nu = mu + tables.a_ab(i, j) * (1.0 - mu * mu);
            if active::<S>(nu) {
                visitor(j);
            }
        }
        j += 1;
    }
}

/// One active pair with its first-order log-derivative data.
struct PairFirst {
    l: f64,
    dmu_i: [f64; 3],
    dmu_j: [f64; 3],
}

/// One active pair with its second-order log-derivative data.
struct PairSecond {
    l: f64,
    ll: f64,
    dmu_i: [f64; 3],
    dmu_j: [f64; 3],
    d2_ii: [[f64; 3]; 3],
    d2_ij: [[f64; 3]; 3],
    d2_jj: [[f64; 3]; 3],
}

/// Value and first `mu`-derivative contribution of one pair, or `None` when
/// the pair is inactive.
#[inline]
fn pair_first<S: SsSwitch>(
    tables: &DerivTables,
    dist: &[f64],
    r: &[f64],
    i: usize,
    j: usize,
) -> Option<PairFirst> {
    let inv = tables.inv_r(i, j);
    if inv == 0.0 {
        return None;
    }
    let a = tables.a_ab(i, j);
    let mu = (dist[i] - dist[j]) * inv;
    let nu = mu + a * (1.0 - mu * mu);
    if !active::<S>(nu) {
        return None;
    }
    let (f, df) = S::switch_d1(nu);
    let s = 0.5 * (1.0 - f);
    if s <= 0.0 || !s.is_finite() {
        return None;
    }
    let kappa = 1.0 - 2.0 * a * mu;
    let l = -0.5 * df * kappa / s;
    let u = tables.u_ab(i, j);
    let ri = [r[i * 3], r[i * 3 + 1], r[i * 3 + 2]];
    let rj = [r[j * 3], r[j * 3 + 1], r[j * 3 + 2]];
    Some(PairFirst {
        l,
        dmu_i: [
            (ri[0] - mu * u[0]) * inv,
            (ri[1] - mu * u[1]) * inv,
            (ri[2] - mu * u[2]) * inv,
        ],
        dmu_j: [
            (-rj[0] + mu * u[0]) * inv,
            (-rj[1] + mu * u[1]) * inv,
            (-rj[2] + mu * u[2]) * inv,
        ],
    })
}

/// Value, first and second `mu`-derivative contribution of one pair.
#[inline]
fn pair_second<S: SsSwitch>(
    tables: &DerivTables,
    dist: &[f64],
    r: &[f64],
    prm: &[f64],
    i: usize,
    j: usize,
) -> Option<PairSecond> {
    let inv_r = tables.inv_r(i, j);
    if inv_r == 0.0 {
        return None;
    }
    let r_ab = tables.r_ab(i, j);
    let a = tables.a_ab(i, j);
    let di = dist[i];
    let dj = dist[j];
    let mu = (di - dj) * inv_r;
    let nu = mu + a * (1.0 - mu * mu);
    if !active::<S>(nu) {
        return None;
    }
    let (f, df, ddf) = S::switch_d2(nu);
    let s = 0.5 * (1.0 - f);
    if s <= 0.0 || !s.is_finite() {
        return None;
    }
    let kappa = 1.0 - 2.0 * a * mu;
    let dkappa = -2.0 * a;
    let ds_mu = -0.5 * df * kappa;
    let dds_mu = -0.5 * (ddf * kappa * kappa + df * dkappa);
    let l = ds_mu / s;
    let ll = dds_mu / s - l * l;

    let u = tables.u_ab(i, j);
    let inv_g2 = inv_r * inv_r;
    let inv_g3 = inv_g2 * inv_r;
    let ri = [r[i * 3], r[i * 3 + 1], r[i * 3 + 2]];
    let rj = [r[j * 3], r[j * 3 + 1], r[j * 3 + 2]];
    let pri = prm3(prm, i);
    let prj = prm3(prm, j);
    let pu = pu33(tables.pu_ab(i, j));
    let nrj = neg3(&rj);
    let nu_vec = neg3(&u);
    let npu = neg33(&pu);
    let zero33 = [[0.0; 3]; 3];
    let f_scalar = di - dj;
    let d2_ii = d2mu(&ri, &ri, &pri, &u, &u, &pu, f_scalar, r_ab, inv_g2, inv_g3);
    let d2_ij = d2mu(
        &ri, &nrj, &zero33, &u, &nu_vec, &npu, f_scalar, r_ab, inv_g2, inv_g3,
    );
    let d2_jj = d2mu(
        &nrj, &nrj, &prj, &nu_vec, &nu_vec, &pu, f_scalar, r_ab, inv_g2, inv_g3,
    );
    Some(PairSecond {
        l,
        ll,
        dmu_i: [
            (ri[0] - mu * u[0]) * inv_r,
            (ri[1] - mu * u[1]) * inv_r,
            (ri[2] - mu * u[2]) * inv_r,
        ],
        dmu_j: [
            (-rj[0] + mu * u[0]) * inv_r,
            (-rj[1] + mu * u[1]) * inv_r,
            (-rj[2] + mu * u[2]) * inv_r,
        ],
        d2_ii,
        d2_ij,
        d2_jj,
    })
}

#[inline(always)]
fn pu33(pu: [f64; 9]) -> [[f64; 3]; 3] {
    [
        [pu[0], pu[1], pu[2]],
        [pu[3], pu[4], pu[5]],
        [pu[6], pu[7], pu[8]],
    ]
}

#[inline(always)]
fn prm3(prm: &[f64], k: usize) -> [[f64; 3]; 3] {
    [
        [prm[k * 9], prm[k * 9 + 1], prm[k * 9 + 2]],
        [prm[k * 9 + 3], prm[k * 9 + 4], prm[k * 9 + 5]],
        [prm[k * 9 + 6], prm[k * 9 + 7], prm[k * 9 + 8]],
    ]
}

#[inline(always)]
fn neg3(v: &[f64; 3]) -> [f64; 3] {
    [-v[0], -v[1], -v[2]]
}

#[inline(always)]
fn neg33(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [-m[0][0], -m[0][1], -m[0][2]],
        [-m[1][0], -m[1][1], -m[1][2]],
        [-m[2][0], -m[2][1], -m[2][2]],
    ]
}

/// Second derivative of a quotient `f / g` with respect to two role
/// directions (same formula as the classic kernel).
#[allow(clippy::too_many_arguments)]
#[inline]
fn d2mu(
    fx: &[f64; 3],
    fy: &[f64; 3],
    fxy: &[[f64; 3]; 3],
    gx: &[f64; 3],
    gy: &[f64; 3],
    gxy: &[[f64; 3]; 3],
    f: f64,
    g: f64,
    inv_g2: f64,
    inv_g3: f64,
) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for t in 0..3 {
        for s in 0..3 {
            let ofg = fx[t] * gy[s] + gx[t] * fy[s];
            let ogg = gx[t] * gy[s];
            out[t][s] = (fxy[t][s] * g - ofg - f * gxy[t][s]) * inv_g2 + 2.0 * f * ogg * inv_g3;
        }
    }
    out
}

/// Unit vectors from distances that were already evaluated (lane path).
fn fill_unit_from_dist(tables: &DerivTables, point: [f64; 3], scratch: &mut Scratch) {
    let n = scratch.n;
    for k in 0..n {
        let c = tables.center(k);
        let d = scratch.dist[k];
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        scratch.r[k * 3] = (c[0] - point[0]) * inv;
        scratch.r[k * 3 + 1] = (c[1] - point[1]) * inv;
        scratch.r[k * 3 + 2] = (c[2] - point[2]) * inv;
    }
}

/// Point-to-centre distances plus unit vectors of one point.
fn fill_distances_and_unit(tables: &DerivTables, point: [f64; 3], scratch: &mut Scratch) {
    let n = scratch.n;
    for k in 0..n {
        let c = tables.center(k);
        let dx = c[0] - point[0];
        let dy = c[1] - point[1];
        let dz = c[2] - point[2];
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        scratch.dist[k] = d;
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        scratch.r[k * 3] = dx * inv;
        scratch.r[k * 3 + 1] = dy * inv;
        scratch.r[k * 3 + 2] = dz * inv;
    }
}

/// Distances, unit vectors and projection matrices of one point.
fn fill_geometry(tables: &DerivTables, point: [f64; 3], scratch: &mut Scratch) {
    let n = scratch.n;
    for k in 0..n {
        let c = tables.center(k);
        let dx = c[0] - point[0];
        let dy = c[1] - point[1];
        let dz = c[2] - point[2];
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        scratch.dist[k] = d;
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        let r = [dx * inv, dy * inv, dz * inv];
        scratch.r[k * 3] = r[0];
        scratch.r[k * 3 + 1] = r[1];
        scratch.r[k * 3 + 2] = r[2];
        for t in 0..3 {
            for s in 0..3 {
                let delta = if t == s { 1.0 } else { 0.0 };
                scratch.prm[k * 9 + t * 3 + s] = (delta - r[t] * r[s]) * inv;
            }
        }
    }
}

/// Distances, unit vectors and products of one point; returns `Z`.
fn prepare_point<S: SsSwitch>(tables: &DerivTables, scratch: &mut Scratch) -> f64 {
    super::prepare_slices::<S>(
        tables.base(),
        &scratch.dist,
        &mut scratch.order,
        &mut scratch.prod,
    )
}

/// Sparse SSF gradients of one point.
///
/// `dw_row` is `[nsel, 3]` for this point; the quadrature weight is folded
/// into it.  The returned weight is the bare `Pbar / Z`.
#[allow(clippy::too_many_arguments)]
fn gradient_point<S: SsSwitch>(
    tables: &DerivTables,
    point: [f64; 3],
    parent: usize,
    quad: f64,
    sel: &[usize],
    dist_ready: bool,
    scratch: &mut Scratch,
) -> f64 {
    let n = scratch.n;
    if dist_ready {
        fill_unit_from_dist(tables, point, scratch);
    } else {
        fill_distances_and_unit(tables, point, scratch);
    }
    let z = prepare_point::<S>(tables, scratch);
    if z.abs() <= INVTOL {
        scratch.dw_row.fill(0.0);
        return 1.0;
    }
    let pbar = scratch.prod[parent];
    let weight = pbar / z;
    if pbar == 0.0 {
        scratch.dw_row.fill(0.0);
        return weight;
    }
    let inv_z = 1.0 / z;
    let q = pbar * inv_z;
    scratch.dz.fill(0.0);
    scratch.dp.fill(0.0);

    {
        let dist = &scratch.dist[..];
        let r = &scratch.r[..];
        let prod = &scratch.prod[..];
        let dz = &mut scratch.dz[..];
        let dp = &mut scratch.dp[..];
        for (i, &pi) in prod.iter().enumerate() {
            if pi == 0.0 {
                continue;
            }
            let is_parent = i == parent;
            let m = i * 3;
            for_each_active::<S, _>(tables, dist, i, |j| {
                let Some(pair) = pair_first::<S>(tables, dist, r, i, j) else {
                    return;
                };
                let c = pi * pair.l;
                let jm = j * 3;
                for t in 0..3 {
                    dz[m + t] += c * pair.dmu_i[t];
                    dz[jm + t] += c * pair.dmu_j[t];
                    if is_parent {
                        dp[m + t] += c * pair.dmu_i[t];
                        dp[jm + t] += c * pair.dmu_j[t];
                    }
                }
            });
        }
    }

    let mut dz_sum = [0.0; 3];
    let mut dp_sum = [0.0; 3];
    for a in 0..n {
        for t in 0..3 {
            dz_sum[t] += scratch.dz[a * 3 + t];
            dp_sum[t] += scratch.dp[a * 3 + t];
        }
    }

    for (slot, &a) in sel.iter().enumerate() {
        for t in 0..3 {
            let r = a * 3 + t;
            let mut v = quad * inv_z * (scratch.dp[r] - q * scratch.dz[r]);
            if a == parent {
                v -= quad * inv_z * (dp_sum[t] - q * dz_sum[t]);
            }
            scratch.dw_row[slot * 3 + t] = v;
        }
    }
    weight
}

/// Sparse compact-support Hessian of one point (and optionally its gradient).
#[allow(clippy::too_many_arguments)]
fn hessian_point<S: SsSwitch>(
    tables: &DerivTables,
    point: [f64; 3],
    parent: usize,
    quad: f64,
    sel: &[usize],
    grad: Option<&mut [f64]>,
    scratch: &mut Scratch,
) -> f64 {
    let n = scratch.n;
    let n3 = 3 * n;
    fill_geometry(tables, point, scratch);
    let z = prepare_point::<S>(tables, scratch);
    if z.abs() <= INVTOL {
        scratch.out.fill(0.0);
        if let Some(g) = grad {
            g.fill(0.0);
        }
        return 1.0;
    }
    let pbar = scratch.prod[parent];
    let weight = pbar / z;
    if pbar == 0.0 {
        // The weight is identically zero in a neighbourhood of the point:
        // every derivative (including the row sums) vanishes.
        scratch.out.fill(0.0);
        if let Some(g) = grad {
            g.fill(0.0);
        }
        return weight;
    }
    scratch.parent = parent;
    // The summed `dS`/`dT` correction row is only needed when the generating
    // atom itself is a selected row; otherwise every correction comes from
    // the selected row's own sum.
    let need_full = sel.binary_search(&parent).is_ok();
    let inv_z = 1.0 / z;
    let q = pbar * inv_z;
    scratch.dz.fill(0.0);
    scratch.dp.fill(0.0);
    if need_full {
        scratch.ds.fill(0.0);
        scratch.dt.fill(0.0);
    }
    scratch.surv.clear();
    scratch.sup_atoms.clear();
    scratch.sup_vals.clear();
    scratch.sup_start.clear();
    scratch.g_sum.clear();

    // --- pass 1: per survivor, log derivatives, pairwise + cross sums --- //
    for i in 0..n {
        let pi = scratch.prod[i];
        if pi == 0.0 {
            continue;
        }
        scratch.si_of[i] = scratch.surv.len() as u32;
        let is_parent = i == parent;
        scratch.g.fill(0.0);
        let start = scratch.sup_atoms.len();
        scratch.sup_atoms.push(i as u32);
        {
            let dist = &scratch.dist[..];
            let r = &scratch.r[..];
            let prm = &scratch.prm[..];
            let g = &mut scratch.g[..];
            let dz = &mut scratch.dz[..];
            let dp = &mut scratch.dp[..];
            let ds = &mut scratch.ds[..];
            let dt = &mut scratch.dt[..];
            let sup_atoms = &mut scratch.sup_atoms;
            let m = i * 3;
            for_each_active::<S, _>(tables, dist, i, |j| {
                if need_full {
                    let Some(pair) = pair_second::<S>(tables, dist, r, prm, i, j) else {
                        return;
                    };
                    let jm = j * 3;
                    let c0 = pi * pair.l;
                    for t in 0..3 {
                        g[m + t] += pair.l * pair.dmu_i[t];
                        g[jm + t] += pair.l * pair.dmu_j[t];
                        dz[m + t] += c0 * pair.dmu_i[t];
                        dz[jm + t] += c0 * pair.dmu_j[t];
                        if is_parent {
                            dp[m + t] += c0 * pair.dmu_i[t];
                            dp[jm + t] += c0 * pair.dmu_j[t];
                        }
                    }
                    let c1 = c0;
                    let cdd = pi * pair.ll;
                    for t in 0..3 {
                        for s in 0..3 {
                            let blk_ii =
                                cdd * pair.dmu_i[t] * pair.dmu_i[s] + c1 * pair.d2_ii[t][s];
                            let blk_ij =
                                cdd * pair.dmu_i[t] * pair.dmu_j[s] + c1 * pair.d2_ij[t][s];
                            let blk_ji =
                                cdd * pair.dmu_j[t] * pair.dmu_i[s] + c1 * pair.d2_ij[s][t];
                            let blk_jj =
                                cdd * pair.dmu_j[t] * pair.dmu_j[s] + c1 * pair.d2_jj[t][s];
                            let si = (t * n + i) * 3 + s;
                            let sj = (t * n + j) * 3 + s;
                            ds[si] += blk_ii + blk_ji;
                            ds[sj] += blk_ij + blk_jj;
                            if is_parent {
                                dt[si] += blk_ii + blk_ji;
                                dt[sj] += blk_ij + blk_jj;
                            }
                        }
                    }
                } else {
                    let Some(pair) = pair_first::<S>(tables, dist, r, i, j) else {
                        return;
                    };
                    let jm = j * 3;
                    let c0 = pi * pair.l;
                    for t in 0..3 {
                        g[m + t] += pair.l * pair.dmu_i[t];
                        g[jm + t] += pair.l * pair.dmu_j[t];
                        dz[m + t] += c0 * pair.dmu_i[t];
                        dz[jm + t] += c0 * pair.dmu_j[t];
                        if is_parent {
                            dp[m + t] += c0 * pair.dmu_i[t];
                            dp[jm + t] += c0 * pair.dmu_j[t];
                        }
                    }
                }
                sup_atoms.push(j as u32);
            });
        }
        let end = scratch.sup_atoms.len();
        for a in start..end {
            let atom = scratch.sup_atoms[a] as usize;
            scratch.sup_vals.push([
                scratch.g[atom * 3],
                scratch.g[atom * 3 + 1],
                scratch.g[atom * 3 + 2],
            ]);
        }
        scratch.sup_start.push(start);
        let mut gs = [0.0; 3];
        if need_full {
            for a in start..end {
                for (g, v) in gs.iter_mut().zip(scratch.sup_vals[a]) {
                    *g += v;
                }
            }
        }
        scratch.g_sum.push(gs);
        scratch.surv.push(i as u32);

        if need_full {
            for a in start..end {
                let atom = scratch.sup_atoms[a] as usize;
                let val = scratch.sup_vals[a];
                for (t, &gst) in gs.iter().enumerate() {
                    let x = pi * gst;
                    if x == 0.0 {
                        continue;
                    }
                    for (s, &v) in val.iter().enumerate() {
                        let c = (t * n + atom) * 3 + s;
                        scratch.ds[c] += x * v;
                        if is_parent {
                            scratch.dt[c] += x * v;
                        }
                    }
                }
            }
        }
    }
    scratch.sup_start.push(scratch.sup_atoms.len());

    // --- first-order completion --- //
    let need_sums = need_full || grad.is_some();
    let mut dz_sum = [0.0; 3];
    let mut dp_sum = [0.0; 3];
    for c in 0..n3 {
        scratch.dq[c] = (scratch.dp[c] - q * scratch.dz[c]) * inv_z;
    }
    if need_sums {
        for a in 0..n {
            for (t, (zs, ps)) in dz_sum.iter_mut().zip(dp_sum.iter_mut()).enumerate() {
                *zs += scratch.dz[a * 3 + t];
                *ps += scratch.dp[a * 3 + t];
            }
        }
    }

    // --- summed correction row --- //
    if need_full {
        let mut dq_sum = [0.0; 3];
        for a in 0..n {
            for (t, q) in dq_sum.iter_mut().enumerate() {
                *q += scratch.dq[a * 3 + t];
            }
        }
        for b in 0..n {
            for t in 0..3 {
                for s in 0..3 {
                    let c = b * 3 + s;
                    let idx = (t * n + b) * 3 + s;
                    scratch.full_a[b * 9 + t * 3 + s] =
                        (scratch.dt[idx] - scratch.dq[c] * dz_sum[t] - q * scratch.ds[idx]) * inv_z
                            - dq_sum[t] * scratch.dz[c] * inv_z;
                }
            }
        }
        scratch.full_ab = [0.0; 9];
        for b in 0..n {
            for t in 0..3 {
                for s in 0..3 {
                    scratch.full_ab[t * 3 + s] += scratch.full_a[b * 9 + t * 3 + s];
                }
            }
        }
    }

    // --- selected rows --- //
    for (slot, &a) in sel.iter().enumerate() {
        scratch.ddz_row.fill(0.0);
        scratch.ddp_row.fill(0.0);
        scratch.c_row.fill(0.0);
        if scratch.prod[a] != 0.0 {
            row_pair_i_side::<S>(tables, scratch, a);
        }
        row_pair_j_side::<S>(tables, scratch, a);
        row_cross(scratch, a);
        for t in 0..n3 {
            scratch.ddz_row[t] += scratch.c_row[t];
        }
        for t in 0..3 {
            let r = a * 3 + t;
            let dz_r = scratch.dz[r];
            let dq_r = scratch.dq[r];
            let out_row = (slot * 3 + t) * n3;
            let row = t * n3;
            let cst = (
                simd::splat(dz_r),
                simd::splat(dq_r),
                simd::splat(q),
                simd::splat(inv_z),
            );
            let mut c = 0;
            while c + simd::LANES <= n3 {
                let ddp = V::from_slice(&scratch.ddp_row[row + c..row + c + simd::LANES]);
                let dq = V::from_slice(&scratch.dq[c..c + simd::LANES]);
                let ddz = V::from_slice(&scratch.ddz_row[row + c..row + c + simd::LANES]);
                let dz = V::from_slice(&scratch.dz[c..c + simd::LANES]);
                let v = (ddp - dq * cst.0 - cst.2 * ddz) * cst.3 - cst.1 * dz * cst.3;
                v.store_slice(&mut scratch.out[out_row + c..out_row + c + simd::LANES]);
                c += simd::LANES;
            }
            while c < n3 {
                let v = (scratch.ddp_row[row + c]
                    - scratch.dq[c] * dz_r
                    - q * scratch.ddz_row[row + c])
                    * inv_z
                    - dq_r * scratch.dz[c] * inv_z;
                scratch.out[out_row + c] = v;
                c += 1;
            }
            let mut fs = [0.0; 3];
            for c in (0..n3).step_by(3) {
                fs[0] += scratch.out[out_row + c];
                fs[1] += scratch.out[out_row + c + 1];
                fs[2] += scratch.out[out_row + c + 2];
            }
            if a == parent {
                for b in 0..n {
                    for (s, &fsv) in fs.iter().enumerate() {
                        let c = b * 3 + s;
                        let idx = out_row + c;
                        let v = scratch.out[idx];
                        scratch.out[idx] = if b == parent {
                            scratch.full_ab[t * 3 + s]
                                - fsv
                                - scratch.full_a[parent * 9 + t * 3 + s]
                                + v
                        } else {
                            -scratch.full_a[b * 9 + t * 3 + s] + v
                        };
                    }
                }
            } else {
                for (s, &fsv) in fs.iter().enumerate() {
                    scratch.out[out_row + parent * 3 + s] -= fsv;
                }
            }
            for c in 0..n3 {
                scratch.out[out_row + c] *= quad;
            }
        }
    }

    // --- optional first derivative from the already computed dP / dZ --- //
    if let Some(g) = grad {
        for (slot, &a) in sel.iter().enumerate() {
            for t in 0..3 {
                let mut v = quad * inv_z * (scratch.dp[a * 3 + t] - q * scratch.dz[a * 3 + t]);
                if a == parent {
                    v -= quad * inv_z * (dp_sum[t] - q * dz_sum[t]);
                }
                g[slot * 3 + t] = v;
            }
        }
    }
    weight
}

/// Find `atom` in survivor `si`'s support; returns the support offset.
#[inline]
fn support_offset(scratch: &Scratch, si: usize, atom: usize) -> Option<usize> {
    let start = scratch.sup_start[si];
    let end = scratch.sup_start[si + 1];
    if scratch.sup_atoms[start] as usize == atom {
        return Some(start);
    }
    scratch.sup_atoms[start + 1..end]
        .binary_search_by_key(&(atom as u32), |&x| x)
        .ok()
        .map(|p| start + 1 + p)
}

/// Pairwise `d^2 Z` (and `d^2 Pbar` for the parent) with the selected atom as
/// the first derivative index.  All three `t` components are accumulated in
/// one pair pass.
fn row_pair_i_side<S: SsSwitch>(tables: &DerivTables, scratch: &mut Scratch, a: usize) {
    let n3 = scratch.n * 3;
    let si = scratch.si_of[a] as usize;
    let start = scratch.sup_start[si];
    let end = scratch.sup_start[si + 1];
    let pi = scratch.prod[a];
    let is_parent = a == scratch.parent;
    for idx in start + 1..end {
        let j = scratch.sup_atoms[idx] as usize;
        let Some(pair) = pair_second::<S>(tables, &scratch.dist, &scratch.r, &scratch.prm, a, j)
        else {
            continue;
        };
        let c1 = pi * pair.l;
        let cdd = pi * pair.ll;
        for t in 0..3 {
            let row = t * n3;
            let ai = a * 3;
            let ji = j * 3;
            for s in 0..3 {
                let blk_ii = cdd * pair.dmu_i[t] * pair.dmu_i[s] + c1 * pair.d2_ii[t][s];
                let blk_ij = cdd * pair.dmu_i[t] * pair.dmu_j[s] + c1 * pair.d2_ij[t][s];
                scratch.ddz_row[row + ai + s] += blk_ii;
                scratch.ddz_row[row + ji + s] += blk_ij;
                if is_parent {
                    scratch.ddp_row[row + ai + s] += blk_ii;
                    scratch.ddp_row[row + ji + s] += blk_ij;
                }
            }
        }
    }
}

/// Pairwise `d^2 Z` / `d^2 Pbar` with the selected atom as the second
/// derivative index: all pairs `(M, a)` whose product support contains `a`.
fn row_pair_j_side<S: SsSwitch>(tables: &DerivTables, scratch: &mut Scratch, a: usize) {
    let n3 = scratch.n * 3;
    let nsurv = scratch.surv.len();
    for si in 0..nsurv {
        let m = scratch.surv[si] as usize;
        if m == a || support_offset(scratch, si, a).is_none() {
            continue;
        }
        let pm = scratch.prod[m];
        let Some(pair) = pair_second::<S>(tables, &scratch.dist, &scratch.r, &scratch.prm, m, a)
        else {
            continue;
        };
        let c1 = pm * pair.l;
        let cdd = pm * pair.ll;
        let is_parent = m == scratch.parent;
        for t in 0..3 {
            let row = t * n3;
            let mi = m * 3;
            let ai = a * 3;
            for s in 0..3 {
                let blk_ji = cdd * pair.dmu_j[t] * pair.dmu_i[s] + c1 * pair.d2_ij[s][t];
                let blk_jj = cdd * pair.dmu_j[t] * pair.dmu_j[s] + c1 * pair.d2_jj[t][s];
                scratch.ddz_row[row + mi + s] += blk_ji;
                scratch.ddz_row[row + ai + s] += blk_jj;
                if is_parent {
                    scratch.ddp_row[row + mi + s] += blk_ji;
                    scratch.ddp_row[row + ai + s] += blk_jj;
                }
            }
        }
    }
}

/// Rank-one cross rows `sum_M P_M l_M[At] l_M[c]` for `d^2 Z`, and the
/// parent's analogue for `d^2 Pbar`, for all three `t`.
fn row_cross(scratch: &mut Scratch, a: usize) {
    let n3 = scratch.n * 3;
    for si in 0..scratch.surv.len() {
        let start = scratch.sup_start[si];
        let end = scratch.sup_start[si + 1];
        let Some(off) = support_offset(scratch, si, a) else {
            continue;
        };
        let pm = scratch.prod[scratch.surv[si] as usize];
        for t in 0..3 {
            let va = scratch.sup_vals[off][t];
            if va == 0.0 {
                continue;
            }
            let x = pm * va;
            let row = t * n3;
            for idx in start..end {
                let atom = scratch.sup_atoms[idx] as usize;
                let val = scratch.sup_vals[idx];
                for (s, &v) in val.iter().enumerate() {
                    scratch.c_row[row + atom * 3 + s] += x * v;
                }
            }
        }
    }
    // d^2 Pbar cross rows: Pbar l_parent[At] l_parent[c].
    let parent = scratch.parent;
    let psi = scratch.si_of[parent];
    if psi != u32::MAX {
        let si = psi as usize;
        if let Some(off) = support_offset(scratch, si, a) {
            let start = scratch.sup_start[si];
            let end = scratch.sup_start[si + 1];
            let pbar = scratch.prod[parent];
            for t in 0..3 {
                let va = scratch.sup_vals[off][t];
                if va == 0.0 {
                    continue;
                }
                let x = pbar * va;
                let row = t * n3;
                for idx in start..end {
                    let atom = scratch.sup_atoms[idx] as usize;
                    let val = scratch.sup_vals[idx];
                    for (s, &v) in val.iter().enumerate() {
                        scratch.ddp_row[row + atom * 3 + s] += x * v;
                    }
                }
            }
        }
    }
}

/// [`DerivScheme`] adapter for the compact-support pair products.
#[derive(Debug)]
pub(crate) struct SsDeriv<S: SsSwitch> {
    tables: DerivTables,
    sel: Vec<usize>,
    switch: PhantomData<S>,
}

/// SSF derivative backend.
pub(crate) type StratmannDeriv = SsDeriv<StratmannSwitch>;
/// Gaussian SS derivative backend.
pub(crate) type GaussianSsDeriv = SsDeriv<GaussianSsSwitch>;

impl<S: SsSwitch> SsDeriv<S> {
    pub(crate) fn new(
        config: PartitionConfig,
        centers: &[[f64; 3]],
        charges: &[u32],
        sel: Vec<usize>,
    ) -> Result<Self> {
        let tables = DerivTables::new(&config, centers, charges)?;
        Ok(Self {
            tables,
            sel,
            switch: PhantomData,
        })
    }

    #[inline]
    fn scratch(scratch: &mut dyn DerivScratch) -> &mut Scratch {
        scratch
            .as_any_mut()
            .downcast_mut()
            .expect("SSF scratch type mismatch")
    }
}

impl<S: SsSwitch> DerivScheme for SsDeriv<S> {
    fn atom_count(&self) -> usize {
        self.tables.len()
    }

    fn selection(&self) -> &[usize] {
        &self.sel
    }

    fn block_scratch_kind(&self) -> TypeId {
        TypeId::of::<Scratch>()
    }

    fn point_scratch_kind(&self) -> TypeId {
        TypeId::of::<Scratch>()
    }

    fn new_block_scratch(&self, block: usize) -> Box<dyn DerivScratch> {
        Box::new(Scratch::new(self.tables.len(), self.sel.len(), block))
    }

    fn new_point_scratch(&self) -> Box<dyn DerivScratch> {
        Box::new(Scratch::new(self.tables.len(), self.sel.len(), 1))
    }

    fn hessian_scratch_bytes(&self) -> usize {
        Scratch::estimated_bytes(self.tables.len(), self.sel.len())
    }

    fn hessian_pointwise(&self) -> bool {
        true
    }

    fn gradient_block(
        &self,
        coords: &[[f64; 3]],
        parents: &[u32],
        quad: &[f64],
        weights: &mut [f64],
        dw: &mut [f64],
        scratch: &mut dyn DerivScratch,
    ) {
        let scratch = Self::scratch(scratch);
        let nb = coords.len();
        let block = scratch.block;
        let nsel = self.sel.len();
        debug_assert_eq!(weights.len(), nb);
        debug_assert_eq!(dw.len(), nsel * 3 * block);
        let mut start = 0;
        while start < nb {
            let len = (nb - start).min(simd::LANES);
            let lane = &coords[start..start + len];
            if len == simd::LANES {
                let c = simd::gather_coords(lane, 0);
                super::fill_lane(
                    self.tables.base(),
                    c,
                    &mut scratch.lane[..self.tables.len()],
                );
            }
            for l in 0..len {
                let p = start + l;
                if len == simd::LANES {
                    super::scatter_lane(
                        &scratch.lane[..self.tables.len()],
                        l,
                        &mut scratch.dist[..self.tables.len()],
                    );
                }
                let weight = gradient_point::<S>(
                    &self.tables,
                    coords[p],
                    parents[p] as usize,
                    quad[p],
                    &self.sel,
                    len == simd::LANES,
                    scratch,
                );
                weights[p] = weight;
                for slot in 0..nsel {
                    for t in 0..3 {
                        dw[(slot * 3 + t) * block + p] = scratch.dw_row[slot * 3 + t];
                    }
                }
            }
            start += len;
        }
    }

    fn point_hessian<'s>(
        &self,
        point: [f64; 3],
        parent: usize,
        quad: f64,
        grad: Option<&mut [f64]>,
        scratch: &'s mut dyn DerivScratch,
    ) -> (f64, &'s [f64]) {
        let scratch = Self::scratch(scratch);
        let weight =
            hessian_point::<S>(&self.tables, point, parent, quad, &self.sel, grad, scratch);
        (weight, &scratch.out)
    }
}
