//! Analytic Hessian of the exact classic Becke weights.
//!
//! This is the second derivative of [`super::gradient`], evaluated for a
//! **selected set of rows** `A` against **all** columns `B`.  The public
//! kernel contract is:
//!
//! ```text
//! out[(slot * 3 + t) * n3 + B * 3 + s] = d^2 w_g / dR_At dR_Bs
//! ```
//!
//! with `n3 = 3 * natm` and `w_g = wq_g Pbar_g / Z_g`.  The storage is
//! `O(nsel * natm^2)` per point and `O(natm^2)` scratch (`dlogP`), never
//! `O(natm * ngrids)`: the public evaluator streams one block of points at a
//! time and contracts or visits the block immediately.
//!
//! ## Structure
//!
//! * the pair loop builds the first-order logarithmic derivative of every
//!   running product `dlogP[M, C, t]`, the pairwise part of `ddZ`/`ddPbar`
//!   for the selected rows, and the **sums over the first index**
//!   `dS[t,B,s] = sum_A ddZ[A,t,B,s]`, `dT[t,B,s] = sum_A ddPbar[A,t,B,s]`;
//! * the rank-one cross terms are added after the pair loop:
//!   `C[A,t,B,s] = sum_M P_M dlogP[M,A,t] dlogP[M,B,s]` and
//!   `Pbar dlogP[A_g,A,t] dlogP[A_g,B,s]`; the lane kernel evaluates `C` in
//!   cache-sized `12 x 12` tiles with the `M` loop innermost, and mirrors the
//!   upper triangle for an all-row Hessian;
//! * `fullA[B,t,s] = sum_A ddw[A,t,B,s]` is obtained from the summed
//!   quotient rule, so the translation correction of the generating atom
//!   does **not** require materialising the unselected rows;
//! * the selected rows are finalised, corrected, and written to `out`.
//!
//! This is what makes `atoms = [A]` cheap: a single row costs
//! `O(natm^2)` instead of the full `O(natm^3)`, and the all-atom case is
//! exactly the dense Hessian.
//!
//! ## Derivation of the summed correction
//!
//! With partial derivatives taken at fixed grid coordinate,
//!
//! ```text
//! ddw[A,t,B,s] = wq { [ddP[A,t,B,s] - dq[B,s] dZ[A,t]
//!                     - q ddZ[A,t,B,s]] / Z - dq[A,t] dZ[B,s] / Z }.
//! ```
//!
//! Summing over `A` and using `dq[A,t] = (dP[A,t] - q dZ[A,t]) / Z` gives
//!
//! ```text
//! fullA[B,t,s] = [ dT[t,B,s] - dq[B,s] S[t] - q dS[t,B,s] ] / Z
//!                - Q[t] dZ[B,s] / Z,
//! S[t]  = sum_A dZ[A,t],   Q[t] = sum_A dq[A,t],
//! dT[t,B,s] = sum_A ddP[A,t,B,s],  dS[t,B,s] = sum_A ddZ[A,t,B,s].
//! ```
//!
//! The pair parts of `dS`/`dT` are accumulated per pair; their cross parts
//! are `sum_M P_M (sum_A dlogP[M,A,t]) dlogP[M,B,s]` and
//! `Pbar (sum_A dlogP[A_g,A,t]) dlogP[A_g,B,s]`.

use super::DerivTables;
use super::gradient::{INVTOL, NOT_SELECTED};
use crate::partition::PartitionConfig;
use crate::partition::simd::{self, LaneAttrib, V};

/// Tile width for the cache-blocked rank-one cross term.  A
/// `CROSS_CB x CROSS_CB` accumulator is cache-sized for the lane path.
const CROSS_CB: usize = 12;

/// Below this many output rows the tiled cross-term materialisation costs more
/// in stores than it saves in `dlogp` re-reads.
const CROSS_MIN_ROWS: usize = 8;

/// Per-lane Hessian core scratch (all fields are `f64x8` vectors).
#[derive(Debug, Clone)]
pub(crate) struct Core {
    pub(crate) n: usize,
    pub(crate) n3: usize,
    pub(crate) dist: Vec<V>,
    pub(crate) drdist: Vec<V>,
    /// `[M * 9 + t * 3 + s]` = `Proj(r_M) / |r_M|`.
    pub(crate) prm: Vec<V>,
    pub(crate) p: Vec<V>,
    /// `[M * n3 + C * 3 + t]` first log-derivative of the running product.
    pub(crate) dlogp: Vec<V>,
    /// `[M * 3 + t]` = `sum_A dlogP[M, A, t]`.
    pub(crate) dlogp_sum: Vec<V>,
    /// `[(slot * 3 + t) * n3 + c]` pairwise `ddZ` for the selected rows.
    pub(crate) ddz: Vec<V>,
    /// `[(slot * 3 + t) * n3 + c]` pairwise `ddPbar` for the selected rows.
    pub(crate) ddp: Vec<V>,
    /// `[(t * n + B) * 3 + s]` = `sum_A` of the pairwise `ddZ`.
    pub(crate) ds_pair: Vec<V>,
    /// `[(t * n + B) * 3 + s]` = `sum_A` of the pairwise `ddPbar`.
    pub(crate) dt_pair: Vec<V>,
    /// Cross-completed `dS` (reuses `ds_pair` layout).
    pub(crate) ds_full: Vec<V>,
    /// Cross-completed `dT` (reuses `dt_pair` layout).
    pub(crate) dt_full: Vec<V>,
    /// `[A * 3 + t]` first derivative of `Z`.
    pub(crate) dz: Vec<V>,
    /// `[A * 3 + t]` first derivative of `Pbar`.
    pub(crate) dp: Vec<V>,
    /// `[A * 3 + t]` first derivative of `q = Pbar / Z`.
    pub(crate) dq: Vec<V>,
    /// `[B * 9 + t * 3 + s]` = `sum_A ddw[A,t,B,s]`.
    pub(crate) full_a: Vec<V>,
    /// Row workspace for the rank-one cross term when it is not materialised.
    pub(crate) c_row: Vec<V>,
    /// Tiled cross terms for all selected rows, `[row * n3 + c]`; empty when
    /// the row count is too small to amortise the materialisation.
    pub(crate) c_full: Vec<V>,
}

impl Core {
    pub(crate) fn new(n: usize, nsel: usize) -> Self {
        let n3 = 3 * n;
        let rows = 3 * nsel;
        let c_full = if rows >= CROSS_MIN_ROWS {
            vec![V::zero(); rows * n3]
        } else {
            Vec::new()
        };
        Self {
            n,
            n3,
            dist: vec![V::zero(); n],
            drdist: vec![V::zero(); 3 * n],
            prm: vec![V::zero(); 9 * n],
            p: vec![V::zero(); n],
            dlogp: vec![V::zero(); n * n3],
            dlogp_sum: vec![V::zero(); 3 * n],
            ddz: vec![V::zero(); nsel * 3 * n3],
            ddp: vec![V::zero(); nsel * 3 * n3],
            ds_pair: vec![V::zero(); 9 * n],
            dt_pair: vec![V::zero(); 9 * n],
            ds_full: vec![V::zero(); 9 * n],
            dt_full: vec![V::zero(); 9 * n],
            dz: vec![V::zero(); n3],
            dp: vec![V::zero(); n3],
            dq: vec![V::zero(); n3],
            full_a: vec![V::zero(); 9 * n],
            c_row: vec![V::zero(); n3],
            c_full,
        }
    }
}

/// Per-thread Hessian scratch plus the single-point output buffer.
#[derive(Debug, Clone)]
pub(crate) struct Scratch {
    pub(crate) core: Core,
    /// Single-point `[nsel, 3, natm, 3]` output for `point_hessian`.
    pub(crate) out: Vec<f64>,
}

impl Scratch {
    /// Estimated bytes of the fixed-lane scratch without the tiled cross
    /// terms.  This is the scratch-size estimate used to select the lane
    /// Hessian; the extra `c_full` buffer does not change which molecules
    /// take the lane path.
    pub(crate) fn base_estimated_bytes(n: usize, nsel: usize) -> usize {
        let n3 = 3 * n;
        let rows = 3 * nsel;
        let vectors = 3 * n * n + 18 * nsel * n + 74 * n;
        64 * vectors + 8 * rows * n3
    }

    /// Estimated bytes of one fixed-lane Hessian scratch, including the
    /// materialised cross terms.
    pub(crate) fn estimated_bytes(n: usize, nsel: usize) -> usize {
        let rows = 3 * nsel;
        let cross = if rows >= CROSS_MIN_ROWS {
            64 * rows * 3 * n
        } else {
            0
        };
        Self::base_estimated_bytes(n, nsel) + cross
    }

    pub(crate) fn new(n: usize, nsel: usize) -> Self {
        Self {
            core: Core::new(n, nsel),
            out: vec![0.0; nsel * 3 * 3 * n],
        }
    }
}

/// Evaluate the corrected selected Hessian rows for one fixed lane.
///
/// `coords`, `parents` and `quad` have `B <= 8` entries.  Output `ddw` is
/// point-major `[B, nsel, 3, natm, 3]`; `weights` is `[B]` and `grad`, when
/// present, is `[B, nsel, 3]`.  The evaluator transposes this into the public
/// point-last `ddw` layout.
#[allow(clippy::too_many_arguments)]
pub(crate) fn block_hessian(
    cfg: &PartitionConfig,
    tables: &DerivTables,
    coords: &[[f64; 3]],
    parents: &[u32],
    quad: &[f64],
    sel: &[usize],
    slot_of: &[usize],
    weights: &mut [f64],
    grad: Option<&mut [f64]>,
    ddw: &mut [f64],
    scratch: &mut Core,
) {
    let n = tables.len();
    let n3 = 3 * n;
    let b = coords.len();
    let nsel = sel.len();
    let hardness = cfg.hardness;
    debug_assert!(b <= simd::LANES);
    debug_assert_eq!(parents.len(), b);
    debug_assert_eq!(quad.len(), b);
    debug_assert_eq!(weights.len(), b);
    debug_assert_eq!(scratch.n, n);
    debug_assert_eq!(scratch.n3, n3);
    debug_assert_eq!(ddw.len(), nsel * 3 * n3 * b);
    if let Some(g) = grad.as_deref() {
        debug_assert_eq!(g.len(), nsel * 3 * b);
    }

    let attrib = LaneAttrib::from_parents(parents, 0, b);
    let xyz = simd::gather_coords(coords, 0);
    let wq = simd::gather(quad, 0);

    // --- distances, unit vectors, projection matrices --- //
    for k in 0..n {
        let c = tables.center(k);
        let dx = simd::splat(c[0]) - xyz[0];
        let dy = simd::splat(c[1]) - xyz[1];
        let dz = simd::splat(c[2]) - xyz[2];
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        let inv = (simd::one() / d).mask_select(d.simd_gt(simd::zero()), simd::zero());
        scratch.dist[k] = d;
        let r = [dx * inv, dy * inv, dz * inv];
        scratch.drdist[k * 3] = r[0];
        scratch.drdist[k * 3 + 1] = r[1];
        scratch.drdist[k * 3 + 2] = r[2];
        for t in 0..3 {
            for s in 0..3 {
                let delta = if t == s { simd::one() } else { simd::zero() };
                scratch.prm[k * 9 + t * 3 + s] = (delta - r[t] * r[s]) * inv;
            }
        }
    }

    // --- running products (pass 1) --- //
    scratch.p[..n].fill(simd::one());
    for i in 0..n {
        for j in 0..i {
            let a_ab = simd::splat(tables.a_ab(i, j));
            let r_ab = simd::splat(tables.r_ab(i, j));
            let mu = (scratch.dist[i] - scratch.dist[j]) / r_ab;
            let nu = mu + a_ab * (simd::one() - mu * mu);
            let f = if hardness == 3 {
                simd::step3(nu)
            } else {
                simd::step_function(nu, hardness)
            };
            let si = simd::splat(0.5) * (simd::one() - f);
            let sj = simd::splat(0.5) * (simd::one() + f);
            scratch.p[i] *= si;
            scratch.p[j] *= sj;
        }
    }
    let z = scratch
        .p
        .iter()
        .take(n)
        .fold(simd::zero(), |acc, &value| acc + value);
    let small = simd::abs_le(z, INVTOL);
    let inv_z = simd::one() / z;
    let pg = attrib.select_product(&scratch.p);
    let q = pg / z;

    // --- zeroing discipline --- //
    for m in 0..n {
        let base = m * n3 + m * 3;
        scratch.dlogp[base..base + 3].fill(simd::zero());
    }
    scratch.ds_pair.fill(simd::zero());
    scratch.dt_pair.fill(simd::zero());
    scratch.dz.fill(simd::zero());
    scratch.dp.fill(simd::zero());
    for slot in 0..nsel {
        for t in 0..3 {
            let base = (slot * 3 + t) * n3;
            scratch.ddz[base..base + n3].fill(simd::zero());
            scratch.ddp[base..base + n3].fill(simd::zero());
        }
    }

    // --- pair derivative pass (deriv 2) --- //
    for i in 0..n {
        for j in 0..i {
            let inv_r = tables.inv_r(i, j);
            let r_ab = tables.r_ab(i, j);
            let inv_g2 = inv_r * inv_r;
            let inv_g3 = inv_g2 * inv_r;
            let inv_g2_v = simd::splat(inv_g2);
            let inv_g3_v = simd::splat(inv_g3);
            let a_ab = simd::splat(tables.a_ab(i, j));
            let u = tables.u_ab(i, j);
            let u = [simd::splat(u[0]), simd::splat(u[1]), simd::splat(u[2])];
            let pu_flat = tables.pu_ab(i, j);
            let pu = [
                [
                    simd::splat(pu_flat[0]),
                    simd::splat(pu_flat[1]),
                    simd::splat(pu_flat[2]),
                ],
                [
                    simd::splat(pu_flat[3]),
                    simd::splat(pu_flat[4]),
                    simd::splat(pu_flat[5]),
                ],
                [
                    simd::splat(pu_flat[6]),
                    simd::splat(pu_flat[7]),
                    simd::splat(pu_flat[8]),
                ],
            ];
            let di = scratch.dist[i];
            let dj = scratch.dist[j];
            let inv_r_v = simd::splat(inv_r);
            let mu = (di - dj) * inv_r_v;
            let nu = mu + a_ab * (simd::one() - mu * mu);
            let (f, df, ddf) = simd::value_d2_any(nu, hardness);
            let dnu = simd::one() - simd::splat(2.0) * a_ab * mu;
            let ddnu = simd::splat(-2.0) * a_ab;
            let si = simd::splat(0.5) * (simd::one() - f);
            let sj = simd::splat(0.5) * (simd::one() + f);
            let dsi = -simd::splat(0.5) * df * dnu;
            let dsj = simd::splat(0.5) * df * dnu;
            let ddsi = -simd::splat(0.5) * (ddf * dnu * dnu + df * ddnu);
            let ddsj = -ddsi;
            let dlog_i = dsi / si.max(simd::splat(INVTOL));
            let dlog_j = dsj / sj.max(simd::splat(INVTOL));
            let ddlog_i = ddsi / si.max(simd::splat(INVTOL)) - dlog_i * dlog_i;
            let ddlog_j = ddsj / sj.max(simd::splat(INVTOL)) - dlog_j * dlog_j;

            let ri = [
                scratch.drdist[i * 3],
                scratch.drdist[i * 3 + 1],
                scratch.drdist[i * 3 + 2],
            ];
            let rj = [
                scratch.drdist[j * 3],
                scratch.drdist[j * 3 + 1],
                scratch.drdist[j * 3 + 2],
            ];
            let pri = [
                [
                    scratch.prm[i * 9],
                    scratch.prm[i * 9 + 1],
                    scratch.prm[i * 9 + 2],
                ],
                [
                    scratch.prm[i * 9 + 3],
                    scratch.prm[i * 9 + 4],
                    scratch.prm[i * 9 + 5],
                ],
                [
                    scratch.prm[i * 9 + 6],
                    scratch.prm[i * 9 + 7],
                    scratch.prm[i * 9 + 8],
                ],
            ];
            let prj = [
                [
                    scratch.prm[j * 9],
                    scratch.prm[j * 9 + 1],
                    scratch.prm[j * 9 + 2],
                ],
                [
                    scratch.prm[j * 9 + 3],
                    scratch.prm[j * 9 + 4],
                    scratch.prm[j * 9 + 5],
                ],
                [
                    scratch.prm[j * 9 + 6],
                    scratch.prm[j * 9 + 7],
                    scratch.prm[j * 9 + 8],
                ],
            ];
            let nrj = neg3(&rj);
            let nu_vec = neg3(&u);
            let npu = neg33(&pu);
            let zero33 = [[simd::zero(); 3]; 3];

            let f_scalar = di - dj;
            let d2mu_aa = d2mu(
                &ri, &ri, &pri, &u, &u, &pu, f_scalar, r_ab, inv_g2_v, inv_g3_v,
            );
            let d2mu_ab = d2mu(
                &ri, &nrj, &zero33, &u, &nu_vec, &npu, f_scalar, r_ab, inv_g2_v, inv_g3_v,
            );
            let d2mu_bb = d2mu(
                &nrj, &nrj, &prj, &nu_vec, &nu_vec, &pu, f_scalar, r_ab, inv_g2_v, inv_g3_v,
            );

            let dmu_i = [
                (ri[0] - mu * u[0]) * inv_r_v,
                (ri[1] - mu * u[1]) * inv_r_v,
                (ri[2] - mu * u[2]) * inv_r_v,
            ];
            let dmu_j = [
                (-rj[0] + mu * u[0]) * inv_r_v,
                (-rj[1] + mu * u[1]) * inv_r_v,
                (-rj[2] + mu * u[2]) * inv_r_v,
            ];

            let p_i = scratch.p[i];
            let p_j = scratch.p[j];
            let coef1 = p_i * dlog_i + p_j * dlog_j;
            let coefdd = p_i * ddlog_i + p_j * ddlog_j;
            let c1_pg = attrib.select(i, p_i * dlog_i, simd::zero())
                + attrib.select(j, p_j * dlog_j, simd::zero());
            let cdd_pg = attrib.select(i, p_i * ddlog_i, simd::zero())
                + attrib.select(j, p_j * ddlog_j, simd::zero());

            for t in 0..3 {
                scratch.dz[i * 3 + t] += coef1 * dmu_i[t];
                scratch.dz[j * 3 + t] += coef1 * dmu_j[t];
                scratch.dp[i * 3 + t] += c1_pg * dmu_i[t];
                scratch.dp[j * 3 + t] += c1_pg * dmu_j[t];
            }

            for t in 0..3 {
                scratch.dlogp[i * n3 + i * 3 + t] += dlog_i * dmu_i[t];
                scratch.dlogp[i * n3 + j * 3 + t] = dlog_i * dmu_j[t];
                scratch.dlogp[j * n3 + i * 3 + t] = dlog_j * dmu_i[t];
                scratch.dlogp[j * n3 + j * 3 + t] += dlog_j * dmu_j[t];
            }

            let slot_i = (slot_of[i] != NOT_SELECTED).then_some(slot_of[i]);
            let slot_j = (slot_of[j] != NOT_SELECTED).then_some(slot_of[j]);
            let i3 = i * 3;
            let j3 = j * 3;
            for t in 0..3 {
                for s in 0..3 {
                    let oo_ii = dmu_i[t] * dmu_i[s];
                    let oo_ij = dmu_i[t] * dmu_j[s];
                    let oo_ji = dmu_j[t] * dmu_i[s];
                    let oo_jj = dmu_j[t] * dmu_j[s];
                    let blk_ii = coefdd * oo_ii + coef1 * d2mu_aa[t][s];
                    let blk_ij = coefdd * oo_ij + coef1 * d2mu_ab[t][s];
                    let blk_ji = coefdd * oo_ji + coef1 * d2mu_ab[s][t];
                    let blk_jj = coefdd * oo_jj + coef1 * d2mu_bb[t][s];

                    scratch.ds_pair[(t * n + i) * 3 + s] += blk_ii + blk_ji;
                    scratch.ds_pair[(t * n + j) * 3 + s] += blk_ij + blk_jj;

                    let pblk_ii = cdd_pg * oo_ii + c1_pg * d2mu_aa[t][s];
                    let pblk_ij = cdd_pg * oo_ij + c1_pg * d2mu_ab[t][s];
                    let pblk_ji = cdd_pg * oo_ji + c1_pg * d2mu_ab[s][t];
                    let pblk_jj = cdd_pg * oo_jj + c1_pg * d2mu_bb[t][s];
                    scratch.dt_pair[(t * n + i) * 3 + s] += pblk_ii + pblk_ji;
                    scratch.dt_pair[(t * n + j) * 3 + s] += pblk_ij + pblk_jj;

                    if let Some(slot_i) = slot_i {
                        let row = (slot_i * 3 + t) * n3;
                        scratch.ddz[row + i3 + s] += blk_ii;
                        scratch.ddz[row + j3 + s] = blk_ij;
                        scratch.ddp[row + i3 + s] += pblk_ii;
                        scratch.ddp[row + j3 + s] = pblk_ij;
                    }
                    if let Some(slot_j) = slot_j {
                        let row = (slot_j * 3 + t) * n3;
                        scratch.ddz[row + i3 + s] = blk_ji;
                        scratch.ddz[row + j3 + s] += blk_jj;
                        scratch.ddp[row + i3 + s] = pblk_ji;
                        scratch.ddp[row + j3 + s] += pblk_jj;
                    }
                }
            }
        }
    }

    // --- first-order completion --- //
    for m in 0..n {
        let row = &scratch.dlogp[m * n3..(m + 1) * n3];
        for t in 0..3 {
            let mut sum = simd::zero();
            let mut idx = t;
            while idx < n3 {
                sum += row[idx];
                idx += 3;
            }
            scratch.dlogp_sum[m * 3 + t] = sum;
        }
    }
    // Per-lane parent row/vector selections, now that dlogP is final.
    // The common homogeneous lane takes the direct row; only a mixed lane
    // pays for the mask selects.
    let uniform_parent = match attrib {
        LaneAttrib::ByAtom(parent) => Some(parent),
        LaneAttrib::ByGrid(_) => None,
    };
    let mut dlog_parent = vec![simd::zero(); n3];
    let mut dlogp_sum_parent = [simd::zero(); 3];
    if let Some(parent) = uniform_parent {
        dlog_parent.copy_from_slice(&scratch.dlogp[parent * n3..(parent + 1) * n3]);
        for (t, value) in dlogp_sum_parent.iter_mut().enumerate() {
            *value = scratch.dlogp_sum[parent * 3 + t];
        }
    } else {
        for m in 0..n {
            for (c, value) in dlog_parent.iter_mut().enumerate() {
                *value = attrib.select(m, scratch.dlogp[m * n3 + c], *value);
            }
            for (t, value) in dlogp_sum_parent.iter_mut().enumerate() {
                *value = attrib.select(m, scratch.dlogp_sum[m * 3 + t], *value);
            }
        }
    }

    let mut dz_sum = [simd::zero(); 3];
    let mut dp_sum = [simd::zero(); 3];
    for t in 0..3 {
        let mut s = simd::zero();
        let mut p = simd::zero();
        for b in 0..n {
            s += scratch.dz[b * 3 + t];
            p += scratch.dp[b * 3 + t];
        }
        dz_sum[t] = s;
        dp_sum[t] = p;
    }
    for c in 0..n3 {
        scratch.dq[c] = (scratch.dp[c] - q * scratch.dz[c]) * inv_z;
    }
    let dq_sum: [V; 3] = [
        (dp_sum[0] - q * dz_sum[0]) * inv_z,
        (dp_sum[1] - q * dz_sum[1]) * inv_z,
        (dp_sum[2] - q * dz_sum[2]) * inv_z,
    ];

    if let Some(grad) = grad {
        for (slot, &a) in sel.iter().enumerate() {
            for t in 0..3 {
                let mut v = wq * inv_z * (scratch.dp[a * 3 + t] - q * scratch.dz[a * 3 + t]);
                let sum = wq * inv_z * (dp_sum[t] - q * dz_sum[t]);
                v = (v - sum).mask_select(attrib.mask(a), v);
                v = simd::zero().mask_select(small, v);
                let row = slot * 3 + t;
                for l in 0..b {
                    grad[l * nsel * 3 + row] = v[l];
                }
            }
        }
    }

    // --- cross-completed dS / dT --- //
    scratch.ds_full.copy_from_slice(&scratch.ds_pair);
    scratch.dt_full.copy_from_slice(&scratch.dt_pair);
    for m in 0..n {
        let pm = scratch.p[m];
        let row_start = m * n3;
        for t in 0..3 {
            let x = pm * scratch.dlogp_sum[m * 3 + t];
            for b in 0..n {
                let dst = (t * n + b) * 3;
                let src = row_start + b * 3;
                for s in 0..3 {
                    scratch.ds_full[dst + s] += x * scratch.dlogp[src + s];
                }
            }
        }
    }
    for (t, &dlog_sum) in dlogp_sum_parent.iter().enumerate() {
        let x = pg * dlog_sum;
        for b in 0..n {
            let dst = (t * n + b) * 3;
            let src = b * 3;
            for s in 0..3 {
                scratch.dt_full[dst + s] += x * dlog_parent[src + s];
            }
        }
    }

    // --- cache-blocked rank-one cross terms --- //
    // Materialise C for every selected row before the quotient sweep; the `M`
    // loop runs innermost per 12x12 tile and the canonical `(min,max)`
    // orientation keeps row subsets bitwise identical to the full Hessian.
    if !scratch.c_full.is_empty() {
        if nsel == n {
            fill_cross_symmetric(&scratch.dlogp, &scratch.p, n, n3, &mut scratch.c_full);
        } else {
            fill_cross_tiled(&scratch.dlogp, &scratch.p, sel, n, n3, &mut scratch.c_full);
        }
    }

    // --- fullA from the summed quotient rule --- //
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
    let mut full_ab = [simd::zero(); 9];
    for b in 0..n {
        for t in 0..3 {
            for s in 0..3 {
                full_ab[t * 3 + s] += scratch.full_a[b * 9 + t * 3 + s];
            }
        }
    }
    let mut full_a_parent = [simd::zero(); 9];
    if let Some(parent) = uniform_parent {
        full_a_parent.copy_from_slice(&scratch.full_a[parent * 9..(parent + 1) * 9]);
    } else {
        for m in 0..n {
            for (c, value) in full_a_parent.iter_mut().enumerate() {
                *value = attrib.select(m, scratch.full_a[m * 9 + c], *value);
            }
        }
    }

    // --- selected-row finalisation and translation correction --- //
    for (slot, &a) in sel.iter().enumerate() {
        for t in 0..3 {
            let r = a * 3 + t;
            let out_row = (slot * 3 + t) * n3;
            let c_row = if scratch.c_full.is_empty() {
                scratch.c_row.fill(simd::zero());
                for m in 0..n {
                    let row = &scratch.dlogp[m * n3..(m + 1) * n3];
                    let pm = scratch.p[m];
                    // Canonical orientation: below the diagonal use
                    // (P dlogP[c]) dlogP[r], above it (P dlogP[r]) dlogP[c].
                    let x = pm * row[r];
                    for (c, dst) in scratch.c_row[r..n3].iter_mut().enumerate() {
                        *dst += x * row[r + c];
                    }
                    for c in 0..r {
                        scratch.c_row[c] += (pm * row[c]) * row[r];
                    }
                }
                &scratch.c_row[..]
            } else {
                &scratch.c_full[out_row..out_row + n3]
            };
            let dlog_parent_r = dlog_parent[r];
            let dq_r = scratch.dq[r];
            let dz_r = scratch.dz[r];
            let mut fs = [simd::zero(); 3];
            for b_idx in 0..n {
                for (s, fs_s) in fs.iter_mut().enumerate() {
                    let c = b_idx * 3 + s;
                    let ddz = scratch.ddz[out_row + c] + c_row[c];
                    let ddp = scratch.ddp[out_row + c] + pg * dlog_parent_r * dlog_parent[c];
                    let v = (ddp - scratch.dq[c] * dz_r - q * ddz) * inv_z
                        - dq_r * scratch.dz[c] * inv_z;
                    scratch.ddz[out_row + c] = v;
                    *fs_s += v;
                }
            }
            let is_self = attrib.mask(a);
            for b_idx in 0..n {
                let at_parent_col = attrib.mask(b_idx);
                for s in 0..3 {
                    let c = b_idx * 3 + s;
                    let idx = out_row + c;
                    let v = scratch.ddz[idx];
                    let full_a_b = scratch.full_a[b_idx * 9 + t * 3 + s];
                    let self_corr = (full_ab[t * 3 + s] - fs[s] - full_a_parent[t * 3 + s] + v)
                        .mask_select(at_parent_col, -full_a_b + v);
                    let other_corr = (v - fs[s]).mask_select(at_parent_col, v);
                    scratch.ddz[idx] = self_corr.mask_select(is_self, other_corr);
                }
            }
            for c in 0..n3 {
                let v = scratch.ddz[out_row + c] * wq;
                for l in 0..b {
                    ddw[l * nsel * 3 * n3 + out_row + c] = v[l];
                }
            }
        }
    }

    let w = simd::one().mask_select(small, q);
    simd::store(weights, w, b);
}

/// Fill `c_full[row * n3 + c] = sum_M P_M dlogP[M, min] dlogP[M, max]`
/// for every selected output row, with the `M` loop innermost in each
/// `CROSS_CB x CROSS_CB` tile.  Entries below the diagonal use the canonical
/// `(min,max)` orientation, so they can be compared bitwise with the full
/// symmetric fill.
fn fill_cross_tiled(dlogp: &[V], p: &[V], sel: &[usize], n: usize, n3: usize, c_full: &mut [V]) {
    let rows = 3 * sel.len();
    debug_assert_eq!(c_full.len(), rows * n3);
    for rb in (0..rows).step_by(CROSS_CB) {
        let rlen = CROSS_CB.min(rows - rb);
        let mut phys = [0usize; CROSS_CB];
        for (i, slot) in phys[..rlen].iter_mut().enumerate() {
            let local = rb + i;
            *slot = sel[local / 3] * 3 + local % 3;
        }
        let rmin = phys[..rlen].iter().copied().min().unwrap();
        let rmax = phys[..rlen].iter().copied().max().unwrap();
        for cb in (0..n3).step_by(CROSS_CB) {
            let clen = CROSS_CB.min(n3 - cb);
            let mut acc = [[V::zero(); CROSS_CB]; CROSS_CB];
            if cb >= rmax {
                // every column index is above every row index
                for m in 0..n {
                    let pm = p[m];
                    let row = &dlogp[m * n3..(m + 1) * n3];
                    for i in 0..rlen {
                        let x = pm * row[phys[i]];
                        let acc_i = &mut acc[i];
                        for (j, dst) in acc_i[..clen].iter_mut().enumerate() {
                            *dst += x * row[cb + j];
                        }
                    }
                }
            } else if cb + clen <= rmin {
                // every row index is above every column index: swap the roles
                for m in 0..n {
                    let pm = p[m];
                    let row = &dlogp[m * n3..(m + 1) * n3];
                    for j in 0..clen {
                        let x = pm * row[cb + j];
                        for i in 0..rlen {
                            acc[i][j] += x * row[phys[i]];
                        }
                    }
                }
            } else {
                // overlapping tile: choose the canonical orientation per entry
                for m in 0..n {
                    let pm = p[m];
                    let row = &dlogp[m * n3..(m + 1) * n3];
                    for i in 0..rlen {
                        let r = phys[i];
                        let xr = pm * row[r];
                        let acc_i = &mut acc[i];
                        for (j, dst) in acc_i[..clen].iter_mut().enumerate() {
                            let c = cb + j;
                            *dst += if c >= r {
                                xr * row[c]
                            } else {
                                (pm * row[c]) * row[r]
                            };
                        }
                    }
                }
            }
            for i in 0..rlen {
                c_full[(rb + i) * n3 + cb..(rb + i) * n3 + cb + clen]
                    .copy_from_slice(&acc[i][..clen]);
            }
        }
    }
}

/// Add one tile row of the rank-one cross term.
///
/// Keeping this out of line stops LLVM from scalarising the explicit `f64x8`
/// lanes and re-vectorising across tile rows (which lowers the varying-index
/// accesses to gather/scatter under AVX-512).
#[cfg_attr(target_feature = "avx512f", inline(never))]
#[cfg_attr(not(target_feature = "avx512f"), inline(always))]
fn cross_tile_row(acc: &mut [V; CROSS_CB], x: V, row: &[V], cb: usize, clen: usize) {
    for (j, dst) in acc[..clen].iter_mut().enumerate() {
        *dst += x * row[cb + j];
    }
}

/// Fill the full symmetric `n3 x n3` cross-term matrix, evaluating only the
/// upper-triangular tiles and mirroring them on store.
fn fill_cross_symmetric(dlogp: &[V], p: &[V], n: usize, n3: usize, c_full: &mut [V]) {
    debug_assert_eq!(c_full.len(), n3 * n3);
    for rb in (0..n3).step_by(CROSS_CB) {
        let rlen = CROSS_CB.min(n3 - rb);
        for cb in (rb..n3).step_by(CROSS_CB) {
            let clen = CROSS_CB.min(n3 - cb);
            let mut acc = [[V::zero(); CROSS_CB]; CROSS_CB];
            for m in 0..n {
                let pm = p[m];
                let row = &dlogp[m * n3..(m + 1) * n3];
                for i in 0..rlen {
                    let x = pm * row[rb + i];
                    cross_tile_row(&mut acc[i], x, row, cb, clen);
                }
            }
            for i in 0..rlen {
                // In a diagonal tile, only the upper half of the tile is
                // canonical; the lower half was already stored by the
                // transposed counterpart above the diagonal.
                let j0 = if rb == cb { i } else { 0 };
                for j in j0..clen {
                    let v = acc[i][j];
                    c_full[(rb + i) * n3 + cb + j] = v;
                    c_full[(cb + j) * n3 + rb + i] = v;
                }
            }
        }
    }
}

#[inline(always)]
fn neg3(v: &[V; 3]) -> [V; 3] {
    [-v[0], -v[1], -v[2]]
}

#[inline(always)]
fn neg33(m: &[[V; 3]; 3]) -> [[V; 3]; 3] {
    [
        [-m[0][0], -m[0][1], -m[0][2]],
        [-m[1][0], -m[1][1], -m[1][2]],
        [-m[2][0], -m[2][1], -m[2][2]],
    ]
}

/// Second derivative of a quotient `f / g` with respect to two role
/// directions, `X` and `Y`.
#[cfg(target_feature = "avx512f")]
#[allow(clippy::too_many_arguments)]
#[inline]
fn d2mu(
    fx: &[V; 3],
    fy: &[V; 3],
    fxy: &[[V; 3]; 3],
    gx: &[V; 3],
    gy: &[V; 3],
    gxy: &[[V; 3]; 3],
    f: V,
    g: f64,
    inv_g2: V,
    inv_g3: V,
) -> [[V; 3]; 3] {
    // Explicitly unrolled: under EVEX lowering LLVM otherwise re-vectorises
    // the t/s loops into ymm gathers/scatters.
    macro_rules! entry {
        ($t:expr, $s:expr) => {{
            let ofg = fx[$t] * gy[$s] + gx[$t] * fy[$s];
            let ogg = gx[$t] * gy[$s];
            (fxy[$t][$s] * simd::splat(g) - ofg - f * gxy[$t][$s]) * inv_g2
                + simd::splat(2.0) * f * ogg * inv_g3
        }};
    }
    [
        [entry!(0, 0), entry!(0, 1), entry!(0, 2)],
        [entry!(1, 0), entry!(1, 1), entry!(1, 2)],
        [entry!(2, 0), entry!(2, 1), entry!(2, 2)],
    ]
}

/// Second derivative of a quotient `f / g` with respect to two role
/// directions, `X` and `Y`.
#[cfg(not(target_feature = "avx512f"))]
#[allow(clippy::too_many_arguments)]
#[inline]
fn d2mu(
    fx: &[V; 3],
    fy: &[V; 3],
    fxy: &[[V; 3]; 3],
    gx: &[V; 3],
    gy: &[V; 3],
    gxy: &[[V; 3]; 3],
    f: V,
    g: f64,
    inv_g2: V,
    inv_g3: V,
) -> [[V; 3]; 3] {
    let mut out = [[simd::zero(); 3]; 3];
    for t in 0..3 {
        for s in 0..3 {
            let ofg = fx[t] * gy[s] + gx[t] * fy[s];
            let ogg = gx[t] * gy[s];
            out[t][s] = (fxy[t][s] * simd::splat(g) - ofg - f * gxy[t][s]) * inv_g2
                + simd::splat(2.0) * f * ogg * inv_g3;
        }
    }
    out
}
