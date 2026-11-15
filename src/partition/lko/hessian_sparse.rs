//! Sparse analytic Hessian of the LKO partition weights.
//!
//! This is the sparse counterpart of [`super::hessian`]: the arithmetic is the
//! same classic Becke quotient/rank-one structure with the LKO saturation
//! geometry, but the pair work follows the exact LKO locality of
//! [`super`](super::mod):
//!
//! ```text
//! S = {i : d_i <= d_min + 5},   T = {j : d_j <= d_min + 10}
//! ```
//!
//! * a centre outside `S` has an exactly zero cell product;
//! * a partner outside `T` has an exactly unit factor for every live centre.
//!
//! Only pairs in `T x T` can carry a derivative, and only pairs with at least
//! one surviving endpoint contribute.  The candidate list `T` therefore
//! replaces the full `natm (natm - 1) / 2` scan, and the dense
//! `3 natm x 3 natm` log-derivative table is replaced by one sorted support
//! row per survivor.  The pair loop keeps the canonical ascending `(i, j)`
//! order of the dense kernel, so every accumulated sum (`dZ`, `dP`, the
//! summed quotient rows and the selected raw rows) has the same addition
//! order and the emitted gradient and Hessian are bitwise identical to the
//! dense reference up to signed zero.
//!
//! # Kernel structure
//!
//! The public contract is the same as [`super::hessian::point_hessian`]:
//! `out[(slot * 3 + t) * n3 + c] = d^2 (q_g Pbar / Z) / dR_At dR_c`, with
//! `c = B * 3 + s`.  Pairs are visited in `T x T` ascending, which is a
//! subsequence of the dense pair order; a pair is skipped when both cell
//! products are zero or when the pre-clamp `|mu| >= 1` makes both cell
//! factors locally constant.  `S/T` are widened by a small floating-point
//! margin so the clamp is guaranteed despite rounding.
//!
//! The pair math is [`super::hessian`]'s `d2mu` quotient rule with
//! `fac = 1/sat(R)`, `k = sat'/sat` and `k' = sat''/sat - k^2`.

use super::super::becke::gradient::INVTOL;
use super::super::becke::switch;
use super::RCUT_LKO;
use super::tables::DerivTables;
use crate::partition::PartitionConfig;

/// Floating-point widening of the `S` / `T` cuts (Bohr).
const CUT_MARGIN: f64 = 1e-9;

/// One off-diagonal entry of a survivor's log-derivative row.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Support {
    atom: u32,
    val: [f64; 3],
}

/// Per-thread scratch for the sparse point Hessian kernel.
#[derive(Debug, Clone)]
pub(crate) struct Scratch {
    pub(crate) n: usize,
    pub(crate) n3: usize,
    pub(crate) nsel: usize,
    pub(crate) dist: Vec<f64>,
    pub(crate) drdist: Vec<f64>,
    /// `[M * 9 + t * 3 + s]` = `Proj(r_M) / |r_M|`.
    pub(crate) prm: Vec<f64>,
    /// `[M]` cell products.
    pub(crate) p: Vec<f64>,
    /// `[cand]` candidate atoms `T`, ascending.
    pub(crate) cand: Vec<u32>,
    /// `[surv]` surviving atoms (`P != 0`), ascending.
    pub(crate) surv: Vec<u32>,
    /// `[M]` off-diagonal entries of the log-derivative row of `M`, sorted by
    /// partner atom.  Only survivors have entries.
    pub(crate) rows: Vec<Vec<Support>>,
    /// `[M * 3 + t]` diagonal of the log-derivative row of `M`.
    pub(crate) diag: Vec<f64>,
    /// `[M * 3 + t]` column sum of the log-derivative row of `M`.
    pub(crate) dlogp_sum: Vec<f64>,
    /// `[(t * n + B) * 3 + s]` = `sum_A` of the pairwise `ddZ`.
    pub(crate) ds_pair: Vec<f64>,
    /// `[(t * n + B) * 3 + s]` = `sum_A` of the pairwise `ddPbar`.
    pub(crate) dt_pair: Vec<f64>,
    /// Cross-completed `dS`.
    pub(crate) ds_full: Vec<f64>,
    /// Cross-completed `dT`.
    pub(crate) dt_full: Vec<f64>,
    /// `[A * 3 + t]` first derivative of `Z`.
    pub(crate) dz: Vec<f64>,
    /// `[A * 3 + t]` first derivative of `Pbar`.
    pub(crate) dp: Vec<f64>,
    /// `[A * 3 + t]` first derivative of `q = Pbar / Z`.
    pub(crate) dq: Vec<f64>,
    /// `[B * 9 + t * 3 + s]` = `sum_A ddw[A,t,B,s]`.
    pub(crate) full_a: Vec<f64>,
    /// `[(slot * 3 + t) * n3 + c]` pairwise `ddZ` of selected rows.
    pub(crate) ddz: Vec<f64>,
    /// `[(slot * 3 + t) * n3 + c]` pairwise `ddPbar` of selected rows.
    pub(crate) ddp: Vec<f64>,
    /// Row workspace for the rank-one cross term.
    pub(crate) c_row: Vec<f64>,
    /// Corrected selected rows.
    pub(crate) out: Vec<f64>,
}

impl Scratch {
    /// Estimated bytes of one scalar Hessian scratch.
    pub(crate) fn estimated_bytes(n: usize, nsel: usize) -> usize {
        let n3 = 3 * n;
        let doubles = n
            + 3 * n
            + 9 * n
            + n
            + n
            + n
            + n
            + 3 * n
            + 3 * n
            + 4 * (9 * n)
            + 3 * n3
            + 9 * n
            + 2 * (nsel * 3 * n3)
            + n3
            + nsel * 3 * n3;
        // One worst-case support entry per ordered pair, plus the row Vec
        // headers.  The live support is far smaller; this is a budget bound.
        let support_entries = n * n;
        8 * doubles + support_entries * std::mem::size_of::<Support>() + n * 64
    }

    pub(crate) fn new(n: usize, nsel: usize) -> Self {
        let n3 = 3 * n;
        Self {
            n,
            n3,
            nsel,
            dist: vec![0.0; n],
            drdist: vec![0.0; 3 * n],
            prm: vec![0.0; 9 * n],
            p: vec![0.0; n],
            cand: Vec::with_capacity(n),
            surv: Vec::with_capacity(n),
            rows: (0..n).map(|_| Vec::new()).collect(),
            diag: vec![0.0; 3 * n],
            dlogp_sum: vec![0.0; 3 * n],
            ds_pair: vec![0.0; 9 * n],
            dt_pair: vec![0.0; 9 * n],
            ds_full: vec![0.0; 9 * n],
            dt_full: vec![0.0; 9 * n],
            dz: vec![0.0; n3],
            dp: vec![0.0; n3],
            dq: vec![0.0; n3],
            full_a: vec![0.0; 9 * n],
            ddz: vec![0.0; nsel * 3 * n3],
            ddp: vec![0.0; nsel * 3 * n3],
            c_row: vec![0.0; n3],
            out: vec![0.0; nsel * 3 * n3],
        }
    }
}

/// Second-order pair data of one active `(i, j)` pair (`i > j`).
#[derive(Clone, Copy)]
struct PairSecond {
    dlog_i: f64,
    dlog_j: f64,
    ddlog_i: f64,
    ddlog_j: f64,
    dmu_i: [f64; 3],
    dmu_j: [f64; 3],
    d2_aa: [[f64; 3]; 3],
    d2_ab: [[f64; 3]; 3],
    d2_bb: [[f64; 3]; 3],
}

/// LKO log-derivative and `dmu`/`d2mu` data of one non-clamped pair.
#[allow(clippy::too_many_arguments)]
#[inline]
fn pair_second(
    tables: &DerivTables,
    dist: &[f64],
    drdist: &[f64],
    prm: &[f64],
    i: usize,
    j: usize,
    mu: f64,
    hardness: usize,
) -> PairSecond {
    let fac = tables.fac(i, j);
    let sat = tables.sat(i, j);
    let kk = tables.k(i, j);
    let kp = tables.kprime(i, j);
    // g = sat(R): g' = sat', g'' = sat''.
    let gp = kk * sat;
    let gpp = sat * (kp + kk * kk);
    let inv_g2 = fac * fac;
    let inv_g3 = inv_g2 * fac;
    let a_ab = tables.a_ab(i, j);
    let u = tables.u_ab(i, j);
    let pu_flat = tables.pu_ab(i, j);
    let pu = [
        [pu_flat[0], pu_flat[1], pu_flat[2]],
        [pu_flat[3], pu_flat[4], pu_flat[5]],
        [pu_flat[6], pu_flat[7], pu_flat[8]],
    ];
    let di = dist[i];
    let dj = dist[j];
    let nu = mu + a_ab * (1.0 - mu * mu);
    let (f, df, ddf) = switch::value_d2_any(nu, hardness);
    let dnu = 1.0 - 2.0 * a_ab * mu;
    let ddnu = -2.0 * a_ab;
    let si = 0.5 * (1.0 - f);
    let sj = 0.5 * (1.0 + f);
    let dsi = -0.5 * df * dnu;
    let dsj = 0.5 * df * dnu;
    let ddsi = -0.5 * (ddf * dnu * dnu + df * ddnu);
    let ddsj = -ddsi;
    let dlog_i = dsi / si.max(INVTOL);
    let dlog_j = dsj / sj.max(INVTOL);
    let ddlog_i = ddsi / si.max(INVTOL) - dlog_i * dlog_i;
    let ddlog_j = ddsj / sj.max(INVTOL) - dlog_j * dlog_j;

    let ri = [drdist[i * 3], drdist[i * 3 + 1], drdist[i * 3 + 2]];
    let rj = [drdist[j * 3], drdist[j * 3 + 1], drdist[j * 3 + 2]];
    let pri = [
        [prm[i * 9], prm[i * 9 + 1], prm[i * 9 + 2]],
        [prm[i * 9 + 3], prm[i * 9 + 4], prm[i * 9 + 5]],
        [prm[i * 9 + 6], prm[i * 9 + 7], prm[i * 9 + 8]],
    ];
    let prj = [
        [prm[j * 9], prm[j * 9 + 1], prm[j * 9 + 2]],
        [prm[j * 9 + 3], prm[j * 9 + 4], prm[j * 9 + 5]],
        [prm[j * 9 + 6], prm[j * 9 + 7], prm[j * 9 + 8]],
    ];
    let nrj = neg3(&rj);
    let zero33 = [[0.0; 3]; 3];
    let f_scalar = di - dj;
    let gx = [gp * u[0], gp * u[1], gp * u[2]];
    let gy = [-gx[0], -gx[1], -gx[2]];
    let mut gxy_aa = [[0.0; 3]; 3];
    for t in 0..3 {
        for s in 0..3 {
            gxy_aa[t][s] = gpp * u[t] * u[s] + gp * pu[t][s];
        }
    }
    let gxy_ab = neg33(&gxy_aa);
    let d2_aa = d2mu(
        &ri, &ri, &pri, &gx, &gx, &gxy_aa, f_scalar, sat, inv_g2, inv_g3,
    );
    let d2_ab = d2mu(
        &ri, &nrj, &zero33, &gx, &gy, &gxy_ab, f_scalar, sat, inv_g2, inv_g3,
    );
    let d2_bb = d2mu(
        &nrj, &nrj, &prj, &gy, &gy, &gxy_aa, f_scalar, sat, inv_g2, inv_g3,
    );

    let gk = mu * kk;
    let dmu_i = [
        ri[0] * fac - gk * u[0],
        ri[1] * fac - gk * u[1],
        ri[2] * fac - gk * u[2],
    ];
    let dmu_j = [
        -rj[0] * fac + gk * u[0],
        -rj[1] * fac + gk * u[1],
        -rj[2] * fac + gk * u[2],
    ];
    PairSecond {
        dlog_i,
        dlog_j,
        ddlog_i,
        ddlog_j,
        dmu_i,
        dmu_j,
        d2_aa,
        d2_ab,
        d2_bb,
    }
}

/// Add `x * value` to a summed second-order row with a transposed layout
/// `[(t * n + B) * 3 + s]`.
#[inline(always)]
fn add_row_t(dst: &mut [f64], n: usize, t: usize, atom: usize, value: &[f64; 3], x: f64) {
    let base = (t * n + atom) * 3;
    for s in 0..3 {
        dst[base + s] += x * value[s];
    }
}

/// Value of the log-derivative row of `m` at the component `t` of column `a`.
#[inline]
fn row_value(scratch: &Scratch, m: usize, a: usize, t: usize) -> f64 {
    if m == a {
        return scratch.diag[m * 3 + t];
    }
    let row = &scratch.rows[m];
    match row.binary_search_by_key(&(a as u32), |e| e.atom) {
        Ok(idx) => row[idx].val[t],
        Err(_) => 0.0,
    }
}

/// Evaluate the sparse selected Hessian rows for one grid point.
///
/// The contract, arguments and return value are those of
/// [`super::hessian::point_hessian`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn point_hessian(
    cfg: &PartitionConfig,
    tables: &DerivTables,
    point: [f64; 3],
    parent: usize,
    sel: &[usize],
    slot_of: &[usize],
    quad_weight: f64,
    grad: Option<&mut [f64]>,
    scratch: &mut Scratch,
) -> f64 {
    let n = tables.len();
    let n3 = 3 * n;
    let hardness = cfg.hardness;
    debug_assert_eq!(scratch.n, n);
    debug_assert_eq!(scratch.n3, n3);
    debug_assert_eq!(slot_of.len(), n);

    // --- distances, unit vectors, projection matrices, candidate set --- //
    let mut dmin = f64::INFINITY;
    for k in 0..n {
        let c = tables.center(k);
        let dx = c[0] - point[0];
        let dy = c[1] - point[1];
        let dz = c[2] - point[2];
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        scratch.dist[k] = d;
        dmin = dmin.min(d);
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        let r = [dx * inv, dy * inv, dz * inv];
        scratch.drdist[k * 3] = r[0];
        scratch.drdist[k * 3 + 1] = r[1];
        scratch.drdist[k * 3 + 2] = r[2];
        for t in 0..3 {
            for s in 0..3 {
                let delta = if t == s { 1.0 } else { 0.0 };
                scratch.prm[k * 9 + t * 3 + s] = (delta - r[t] * r[s]) * inv;
            }
        }
    }
    let cand_cut = dmin + 2.0 * RCUT_LKO + CUT_MARGIN;
    scratch.cand.clear();
    for k in 0..n {
        if scratch.dist[k] <= cand_cut {
            scratch.cand.push(k as u32);
        }
    }

    // --- running products over the candidate set (pass 1) --- //
    scratch.p.fill(0.0);
    for &k in &scratch.cand {
        scratch.p[k as usize] = 1.0;
    }
    for a in 0..scratch.cand.len() {
        let i = scratch.cand[a] as usize;
        for b in 0..a {
            let j = scratch.cand[b] as usize;
            let fac = tables.fac(i, j);
            let a_ab = tables.a_ab(i, j);
            let g = ((scratch.dist[i] - scratch.dist[j]) * fac).clamp(-1.0, 1.0);
            let nu = g + a_ab * (1.0 - g * g);
            let f = if hardness == 3 {
                switch::step3(nu)
            } else {
                switch::step_function(nu, hardness)
            };
            scratch.p[i] *= 0.5 * (1.0 - f);
            scratch.p[j] *= 0.5 * (1.0 + f);
        }
    }
    let z: f64 = scratch.p[..n].iter().sum();
    if z.abs() <= INVTOL {
        scratch.out.fill(0.0);
        if let Some(grad) = grad {
            grad.fill(0.0);
        }
        return 1.0;
    }
    let inv_z = 1.0 / z;
    let pg = scratch.p[parent];
    let q = pg / z;

    // --- survivors and support rows --- //
    scratch.surv.clear();
    for &k in &scratch.cand {
        let k = k as usize;
        scratch.rows[k].clear();
        scratch.diag[k * 3] = 0.0;
        scratch.diag[k * 3 + 1] = 0.0;
        scratch.diag[k * 3 + 2] = 0.0;
        if scratch.p[k] != 0.0 {
            scratch.surv.push(k as u32);
        }
    }
    let need_full = sel.binary_search(&parent).is_ok();
    let need_pbar = !sel.is_empty();

    // --- zeroing discipline --- //
    if need_full {
        scratch.ds_pair.fill(0.0);
        scratch.dt_pair.fill(0.0);
    }
    scratch.dz.fill(0.0);
    scratch.dp.fill(0.0);
    for slot in 0..scratch.nsel {
        for t in 0..3 {
            let base = (slot * 3 + t) * n3;
            scratch.ddz[base..base + n3].fill(0.0);
            scratch.ddp[base..base + n3].fill(0.0);
        }
    }

    // --- pair derivative pass over the candidate set (deriv 2) --- //
    for a in 0..scratch.cand.len() {
        let i = scratch.cand[a] as usize;
        for b in 0..a {
            let j = scratch.cand[b] as usize;
            let fac = tables.fac(i, j);
            let di = scratch.dist[i];
            let dj = scratch.dist[j];
            let mu = (di - dj) * fac;
            // Clamped pairs have exactly constant cell factors: no derivative.
            if mu.abs() >= 1.0 {
                continue;
            }
            let pi = scratch.p[i];
            let pj = scratch.p[j];
            if pi == 0.0 && pj == 0.0 {
                continue;
            }
            let pair = pair_second(
                tables,
                &scratch.dist,
                &scratch.drdist,
                &scratch.prm,
                i,
                j,
                mu,
                hardness,
            );
            let coef1 = pi * pair.dlog_i + pj * pair.dlog_j;
            let coefdd = pi * pair.ddlog_i + pj * pair.ddlog_j;
            let is_parent_i = if i == parent { 1.0 } else { 0.0 };
            let is_parent_j = if j == parent { 1.0 } else { 0.0 };
            let c1_pg = is_parent_i * pi * pair.dlog_i + is_parent_j * pj * pair.dlog_j;
            let cdd_pg = is_parent_i * pi * pair.ddlog_i + is_parent_j * pj * pair.ddlog_j;

            for t in 0..3 {
                scratch.dz[i * 3 + t] += coef1 * pair.dmu_i[t];
                scratch.dz[j * 3 + t] += coef1 * pair.dmu_j[t];
                scratch.dp[i * 3 + t] += c1_pg * pair.dmu_i[t];
                scratch.dp[j * 3 + t] += c1_pg * pair.dmu_j[t];
            }

            // Off-diagonal log-derivative rows, only for survivors.
            if pi != 0.0 {
                for t in 0..3 {
                    scratch.diag[i * 3 + t] += pair.dlog_i * pair.dmu_i[t];
                }
                scratch.rows[i].push(Support {
                    atom: j as u32,
                    val: [
                        pair.dlog_i * pair.dmu_j[0],
                        pair.dlog_i * pair.dmu_j[1],
                        pair.dlog_i * pair.dmu_j[2],
                    ],
                });
            }
            if pj != 0.0 {
                for t in 0..3 {
                    scratch.diag[j * 3 + t] += pair.dlog_j * pair.dmu_j[t];
                }
                scratch.rows[j].push(Support {
                    atom: i as u32,
                    val: [
                        pair.dlog_j * pair.dmu_i[0],
                        pair.dlog_j * pair.dmu_i[1],
                        pair.dlog_j * pair.dmu_i[2],
                    ],
                });
            }

            let slot_i = slot_of[i];
            let slot_j = slot_of[j];
            let i3 = i * 3;
            let j3 = j * 3;
            for t in 0..3 {
                for s in 0..3 {
                    let oo_ii = pair.dmu_i[t] * pair.dmu_i[s];
                    let oo_ij = pair.dmu_i[t] * pair.dmu_j[s];
                    let oo_ji = pair.dmu_j[t] * pair.dmu_i[s];
                    let oo_jj = pair.dmu_j[t] * pair.dmu_j[s];
                    let blk_ii = coefdd * oo_ii + coef1 * pair.d2_aa[t][s];
                    let blk_ij = coefdd * oo_ij + coef1 * pair.d2_ab[t][s];
                    let blk_ji = coefdd * oo_ji + coef1 * pair.d2_ab[s][t];
                    let blk_jj = coefdd * oo_jj + coef1 * pair.d2_bb[t][s];

                    if need_full {
                        let ds_i = (t * n + i) * 3 + s;
                        let ds_j = (t * n + j) * 3 + s;
                        scratch.ds_pair[ds_i] += blk_ii + blk_ji;
                        scratch.ds_pair[ds_j] += blk_ij + blk_jj;
                    }
                    if need_pbar {
                        let pblk_ii = cdd_pg * oo_ii + c1_pg * pair.d2_aa[t][s];
                        let pblk_ij = cdd_pg * oo_ij + c1_pg * pair.d2_ab[t][s];
                        let pblk_ji = cdd_pg * oo_ji + c1_pg * pair.d2_ab[s][t];
                        let pblk_jj = cdd_pg * oo_jj + c1_pg * pair.d2_bb[t][s];
                        if need_full {
                            let dt_i = (t * n + i) * 3 + s;
                            let dt_j = (t * n + j) * 3 + s;
                            scratch.dt_pair[dt_i] += pblk_ii + pblk_ji;
                            scratch.dt_pair[dt_j] += pblk_ij + pblk_jj;
                        }
                        if slot_i != usize::MAX {
                            let row = (slot_i * 3 + t) * n3;
                            scratch.ddp[row + i3 + s] += pblk_ii;
                            scratch.ddp[row + j3 + s] = pblk_ij;
                        }
                        if slot_j != usize::MAX {
                            let row = (slot_j * 3 + t) * n3;
                            scratch.ddp[row + i3 + s] = pblk_ji;
                            scratch.ddp[row + j3 + s] += pblk_jj;
                        }
                    }
                    if slot_i != usize::MAX {
                        let row = (slot_i * 3 + t) * n3;
                        scratch.ddz[row + i3 + s] += blk_ii;
                        scratch.ddz[row + j3 + s] = blk_ij;
                    }
                    if slot_j != usize::MAX {
                        let row = (slot_j * 3 + t) * n3;
                        scratch.ddz[row + i3 + s] = blk_ji;
                        scratch.ddz[row + j3 + s] += blk_jj;
                    }
                }
            }
        }
    }

    // --- log-derivative column sums --- //
    for &m in &scratch.surv {
        let m = m as usize;
        let row = &scratch.rows[m];
        debug_assert!(
            row.windows(2).all(|w| w[0].atom < w[1].atom),
            "support row of atom {m} is not strictly ascending"
        );
        let diag = [
            scratch.diag[m * 3],
            scratch.diag[m * 3 + 1],
            scratch.diag[m * 3 + 2],
        ];
        let mut sum = [0.0; 3];
        let mut diagonal_done = false;
        for entry in row {
            if !diagonal_done && entry.atom as usize > m {
                for (s, d) in sum.iter_mut().zip(diag) {
                    *s += d;
                }
                diagonal_done = true;
            }
            for (s, v) in sum.iter_mut().zip(entry.val) {
                *s += v;
            }
        }
        if !diagonal_done {
            for (s, d) in sum.iter_mut().zip(diag) {
                *s += d;
            }
        }
        scratch.dlogp_sum[m * 3..m * 3 + 3].copy_from_slice(&sum);
    }

    // --- first-order completion --- //
    let mut dz_sum = [0.0; 3];
    let mut dp_sum = [0.0; 3];
    for b in 0..n {
        for t in 0..3 {
            dz_sum[t] += scratch.dz[b * 3 + t];
            dp_sum[t] += scratch.dp[b * 3 + t];
        }
    }
    for c in 0..n3 {
        scratch.dq[c] = (scratch.dp[c] - q * scratch.dz[c]) * inv_z;
    }
    let dq_sum: [f64; 3] = [
        (dp_sum[0] - q * dz_sum[0]) * inv_z,
        (dp_sum[1] - q * dz_sum[1]) * inv_z,
        (dp_sum[2] - q * dz_sum[2]) * inv_z,
    ];

    // --- optional first derivative from the already-computed dP / dZ --- //
    if let Some(grad) = grad {
        debug_assert_eq!(grad.len(), sel.len() * 3);
        for (slot, &a) in sel.iter().enumerate() {
            for t in 0..3 {
                let mut v =
                    quad_weight * inv_z * (scratch.dp[a * 3 + t] - q * scratch.dz[a * 3 + t]);
                if a == parent {
                    v -= quad_weight * inv_z * (dp_sum[t] - q * dz_sum[t]);
                }
                grad[slot * 3 + t] = v;
            }
        }
    }

    // --- cross-completed dS / dT --- //
    if need_full {
        scratch.ds_full.copy_from_slice(&scratch.ds_pair);
        scratch.dt_full.copy_from_slice(&scratch.dt_pair);
        for &m in &scratch.surv {
            let m = m as usize;
            let pm = scratch.p[m];
            let row = &scratch.rows[m];
            for t in 0..3 {
                let x = pm * scratch.dlogp_sum[m * 3 + t];
                if x == 0.0 {
                    continue;
                }
                let mut diagonal_done = false;
                for entry in row {
                    if !diagonal_done && entry.atom as usize > m {
                        let diag = [
                            scratch.diag[m * 3],
                            scratch.diag[m * 3 + 1],
                            scratch.diag[m * 3 + 2],
                        ];
                        add_row_t(&mut scratch.ds_full, n, t, m, &diag, x);
                        diagonal_done = true;
                    }
                    add_row_t(
                        &mut scratch.ds_full,
                        n,
                        t,
                        entry.atom as usize,
                        &entry.val,
                        x,
                    );
                }
                if !diagonal_done {
                    let diag = [
                        scratch.diag[m * 3],
                        scratch.diag[m * 3 + 1],
                        scratch.diag[m * 3 + 2],
                    ];
                    add_row_t(&mut scratch.ds_full, n, t, m, &diag, x);
                }
            }
        }
        if pg != 0.0 {
            let parent_row = &scratch.rows[parent];
            for t in 0..3 {
                let x = pg * scratch.dlogp_sum[parent * 3 + t];
                if x == 0.0 {
                    continue;
                }
                let mut diagonal_done = false;
                for entry in parent_row {
                    if !diagonal_done && entry.atom as usize > parent {
                        let diag = [
                            scratch.diag[parent * 3],
                            scratch.diag[parent * 3 + 1],
                            scratch.diag[parent * 3 + 2],
                        ];
                        add_row_t(&mut scratch.dt_full, n, t, parent, &diag, x);
                        diagonal_done = true;
                    }
                    add_row_t(
                        &mut scratch.dt_full,
                        n,
                        t,
                        entry.atom as usize,
                        &entry.val,
                        x,
                    );
                }
                if !diagonal_done {
                    let diag = [
                        scratch.diag[parent * 3],
                        scratch.diag[parent * 3 + 1],
                        scratch.diag[parent * 3 + 2],
                    ];
                    add_row_t(&mut scratch.dt_full, n, t, parent, &diag, x);
                }
            }
        }
    }

    // --- fullA from the summed quotient rule --- //
    let mut full_ab = [0.0; 9];
    if need_full {
        for b in 0..n {
            for t in 0..3 {
                for s in 0..3 {
                    let c = b * 3 + s;
                    let v = (scratch.dt_full[(t * n + b) * 3 + s]
                        - scratch.dq[c] * dz_sum[t]
                        - q * scratch.ds_full[(t * n + b) * 3 + s])
                        * inv_z
                        - dq_sum[t] * scratch.dz[c] * inv_z;
                    scratch.full_a[b * 9 + t * 3 + s] = v;
                }
            }
        }
        for b in 0..n {
            for t in 0..3 {
                for s in 0..3 {
                    full_ab[t * 3 + s] += scratch.full_a[b * 9 + t * 3 + s];
                }
            }
        }
    }

    // --- selected-row finalisation and translation correction --- //
    for (slot, &a) in sel.iter().enumerate() {
        for t in 0..3 {
            let r = a * 3 + t;
            scratch.c_row.fill(0.0);
            for &m in &scratch.surv {
                let m = m as usize;
                let va = row_value(scratch, m, a, t);
                let x = scratch.p[m] * va;
                if x == 0.0 {
                    continue;
                }
                let row = &scratch.rows[m];
                let mut diagonal_done = false;
                for entry in row {
                    let atom = entry.atom as usize;
                    if !diagonal_done && atom > m {
                        for s in 0..3 {
                            scratch.c_row[m * 3 + s] += x * scratch.diag[m * 3 + s];
                        }
                        diagonal_done = true;
                    }
                    for s in 0..3 {
                        scratch.c_row[atom * 3 + s] += x * entry.val[s];
                    }
                }
                if !diagonal_done {
                    for s in 0..3 {
                        scratch.c_row[m * 3 + s] += x * scratch.diag[m * 3 + s];
                    }
                }
            }
            let dlog_parent_r = row_value(scratch, parent, a, t);
            let dq_r = scratch.dq[r];
            let dz_r = scratch.dz[r];
            let out_row = (slot * 3 + t) * n3;
            if pg != 0.0 && dlog_parent_r != 0.0 {
                let scale = pg * dlog_parent_r;
                let parent_row = &scratch.rows[parent];
                for entry in parent_row {
                    let atom = entry.atom as usize;
                    for s in 0..3 {
                        scratch.ddp[out_row + atom * 3 + s] += scale * entry.val[s];
                    }
                }
                for s in 0..3 {
                    scratch.ddp[out_row + parent * 3 + s] += scale * scratch.diag[parent * 3 + s];
                }
            }
            let mut fs = [0.0; 3];
            for b in 0..n {
                for (s, fsv) in fs.iter_mut().enumerate() {
                    let c = b * 3 + s;
                    let ddz = scratch.ddz[out_row + c] + scratch.c_row[c];
                    let ddp = scratch.ddp[out_row + c];
                    let v = (ddp - scratch.dq[c] * dz_r - q * ddz) * inv_z
                        - dq_r * scratch.dz[c] * inv_z;
                    scratch.out[out_row + c] = v;
                    *fsv += v;
                }
            }
            if a == parent {
                for b in 0..n {
                    for (s, &fsv) in fs.iter().enumerate() {
                        let c = b * 3 + s;
                        let idx = out_row + c;
                        let v = scratch.out[idx];
                        scratch.out[idx] = if b == parent {
                            full_ab[t * 3 + s] - fsv - scratch.full_a[parent * 9 + t * 3 + s] + v
                        } else {
                            -scratch.full_a[b * 9 + t * 3 + s] + v
                        };
                    }
                }
            } else {
                for (s, &fsv) in fs.iter().enumerate() {
                    let idx = out_row + parent * 3 + s;
                    scratch.out[idx] -= fsv;
                }
            }
            for c in 0..n3 {
                scratch.out[out_row + c] *= quad_weight;
            }
        }
    }
    q
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
/// directions, `X` and `Y`.
///
/// ```text
/// (f/g)_{XY} = [f_XY g - (f_X g_Y + g_X f_Y) - f g_XY] / g^2
///              + 2 f g_X g_Y / g^3
/// ```
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

#[cfg(test)]
mod tests {
    use super::super::hessian as dense;
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
                3.6 * (0.37 * t + 0.3).sin(),
                3.1 * (0.29 * t + 1.1).cos(),
                2.7 * (0.41 * t + 0.7).sin(),
            ];
            // Keep the nearest-neighbour distance comfortably above zero.
            let c = [c[0] + 0.4 * (t * 0.13).sin(), c[1] - 0.3, c[2]];
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
                let mut sd = dense::Scratch::new(n, sel.len());
                let mut ss = Scratch::new(n, sel.len());
                for k in 0..240 {
                    let parent = k % n;
                    let t = k as f64;
                    let local = if k == 0 {
                        [0.0, 0.0, 0.0]
                    } else {
                        [
                            3.1 * (0.31 * t + 0.1).sin(),
                            2.9 * (0.23 * t + 0.7).cos(),
                            3.3 * (0.17 * t + 1.3).sin(),
                        ]
                    };
                    let point = [
                        centers[parent][0] + local[0],
                        centers[parent][1] + local[1],
                        centers[parent][2] + local[2],
                    ];
                    let mut gd = vec![0.0; sel.len() * 3];
                    let mut gs = vec![0.0; sel.len() * 3];
                    let wd = dense::point_hessian(
                        &cfg,
                        &tables,
                        point,
                        parent,
                        &sel,
                        &slot_of,
                        0.734,
                        Some(&mut gd),
                        &mut sd,
                    );
                    let ws = point_hessian(
                        &cfg,
                        &tables,
                        point,
                        parent,
                        &sel,
                        &slot_of,
                        0.734,
                        Some(&mut gs),
                        &mut ss,
                    );
                    assert_eq!(
                        wd.to_bits(),
                        ws.to_bits(),
                        "weight mismatch n={n} parent={parent} point={point:?}"
                    );
                    assert_eq!(gd.len(), sel.len() * 3, "inconsistent selection length");
                    // Signed zeros are numerically equal and can legitimately
                    // differ because the sparse kernel skips exact zero terms.
                    let same =
                        |a: f64, b: f64| a.to_bits() == b.to_bits() || (a == 0.0 && b == 0.0);
                    for (idx, (&a, &b)) in gd.iter().zip(&gs).enumerate() {
                        assert!(
                            same(a, b),
                            "gradient mismatch n={n} parent={parent} point={point:?} idx={idx}: {a} vs {b}"
                        );
                    }
                    for (idx, (&a, &b)) in sd.out.iter().zip(&ss.out).enumerate() {
                        assert!(
                            same(a, b),
                            "Hessian mismatch n={n} parent={parent} point={point:?} idx={idx}: {a} vs {b}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn sparse_matches_the_dense_reference() {
        for (charges, centers) in [water(), benzene(), cluster()] {
            check_case(&charges, &centers);
        }
    }
}
