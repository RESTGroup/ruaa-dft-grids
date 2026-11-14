//! Molecule-only tables for the LKO analytic derivatives.
//!
//! LKO replaces the interatomic distance `R_ij` of the classic Becke
//! geometry by the saturating function `sat(R_ij)`:
//!
//! ```text
//! fac_ij = 1 / sat(R_ij)           mu_ij = (d_i - d_j) fac_ij
//! k_ij   = sat'(R_ij) / sat(R_ij)
//! k'_ij  = sat''(R_ij) / sat(R_ij) - k_ij^2
//! ```
//!
//! so that, at a fixed grid point,
//!
//! ```text
//! d mu / dR_i = fac r_hat_i - mu k u_hat_ij
//! d mu / dR_j = -fac r_hat_j + mu k u_hat_ij
//! ```
//!
//! with `r_hat_i = (R_i - p)/d_i` and `u_hat_ij = (R_i - R_j)/R_ij`.  The
//! second derivative follows from the quotient rule with `f = d_i - d_j` and
//! `g = sat(R_ij)`; the Hessian kernel passes `(g, g', g'')` and the same
//! projection matrices as the classic Becke kernel (see
//! [`super::hessian`]).
//!
//! The clamp `g = clamp(mu, -1, 1)` makes both `d mu` and `d^2 mu` vanish
//! whenever `|mu| >= 1`: the cell factor is then locally constant (`0` or
//! `1`) and the pair contributes no derivative.  The kernels skip such
//! pairs explicitly because, unlike the first derivative, the second
//! derivative of the switch does **not** vanish at $nu = \pm1$.
//!
//! Storage is `O(natm^2)` (fac, k, k', `u`, `pu`), the same budget as the
//! classic derivative tables.

use crate::error::Result;
use crate::partition::{PartitionConfig, PartitionTables};

use super::lko_sat_d1_d2;

/// Molecule-only derivative tables of the LKO scheme.
#[derive(Debug, Clone)]
pub(crate) struct DerivTables {
    n: usize,
    centers: Vec<[f64; 3]>,
    a_ab: Vec<f64>,
    sat: Vec<f64>,
    fac: Vec<f64>,
    /// `k = sat'(R) / sat(R)`.
    k: Vec<f64>,
    /// `k' = sat''(R) / sat(R) - k^2`.
    kprime: Vec<f64>,
    /// `(R_i - R_j) / R_ij`.
    u_ab: Vec<[f64; 3]>,
    /// `(I - u u^T) / R_ij`.
    pu_ab: Vec<[f64; 9]>,
}

impl DerivTables {
    /// Build the LKO geometry tables for a molecule (Bohr).
    /// # Errors
    /// Propagates [`PartitionTables::new`]'s length/radii errors.
    pub(crate) fn new(
        cfg: &PartitionConfig,
        centers: &[[f64; 3]],
        charges: &[u32],
    ) -> Result<Self> {
        let base = PartitionTables::new(cfg, centers, charges)?;
        let n = base.len();
        let mut a_ab = vec![0.0; n * n];
        let mut sat = vec![0.0; n * n];
        let mut fac = vec![0.0; n * n];
        let mut k = vec![0.0; n * n];
        let mut kprime = vec![0.0; n * n];
        let mut u_ab = vec![[0.0; 3]; n * n];
        let mut pu_ab = vec![[0.0; 9]; n * n];
        for i in 1..n {
            for j in 0..i {
                let r = base.r_ab(i, j);
                let (s, sp, spp) = lko_sat_d1_d2(r);
                // `fac` is the weight-path value; derive the rest from the
                // same `sat` so the two paths agree to round-off.
                let f = base.fac(i, j);
                let kk = sp / s;
                let kp = spp / s - kk * kk;
                let inv = 1.0 / r;
                let mut u = [0.0; 3];
                for t in 0..3 {
                    u[t] = (centers[i][t] - centers[j][t]) * inv;
                }
                let mut pu = [0.0; 9];
                for t in 0..3 {
                    for q in 0..3 {
                        let delta = if t == q { 1.0 } else { 0.0 };
                        pu[t * 3 + q] = (delta - u[t] * u[q]) * inv;
                    }
                }
                for (idx, val) in [
                    (i * n + j, (base.a_ab(i, j), s, f, kk, kp, u, pu)),
                    (
                        j * n + i,
                        (base.a_ab(j, i), s, f, kk, kp, [-u[0], -u[1], -u[2]], pu),
                    ),
                ] {
                    let (a, s, f, kk, kp, u, pu) = val;
                    a_ab[idx] = a;
                    sat[idx] = s;
                    fac[idx] = f;
                    k[idx] = kk;
                    kprime[idx] = kp;
                    u_ab[idx] = u;
                    pu_ab[idx] = pu;
                }
            }
        }
        Ok(Self {
            n,
            centers: centers.to_vec(),
            a_ab,
            sat,
            fac,
            k,
            kprime,
            u_ab,
            pu_ab,
        })
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.n
    }

    #[inline]
    pub(crate) fn center(&self, i: usize) -> [f64; 3] {
        self.centers[i]
    }

    #[inline]
    pub(crate) fn a_ab(&self, i: usize, j: usize) -> f64 {
        self.a_ab[i * self.n + j]
    }

    /// `sat(R_ij)`.
    #[inline]
    pub(crate) fn sat(&self, i: usize, j: usize) -> f64 {
        self.sat[i * self.n + j]
    }

    /// `fac_ij = 1 / sat(R_ij)`.
    #[inline]
    pub(crate) fn fac(&self, i: usize, j: usize) -> f64 {
        self.fac[i * self.n + j]
    }

    /// `k = sat'(R) / sat(R)`.
    #[inline]
    pub(crate) fn k(&self, i: usize, j: usize) -> f64 {
        self.k[i * self.n + j]
    }

    /// `k' = sat''(R) / sat(R) - k^2`.
    #[inline]
    pub(crate) fn kprime(&self, i: usize, j: usize) -> f64 {
        self.kprime[i * self.n + j]
    }

    #[inline]
    pub(crate) fn u_ab(&self, i: usize, j: usize) -> [f64; 3] {
        self.u_ab[i * self.n + j]
    }

    #[inline]
    pub(crate) fn pu_ab(&self, i: usize, j: usize) -> [f64; 9] {
        self.pu_ab[i * self.n + j]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `sat'` and `sat''` against central differences of the value function.
    #[test]
    fn saturation_derivatives_match_finite_difference() {
        for k in 1..5000 {
            let r = 0.05 + 20.0 * k as f64 / 5000.0;
            let (sat, d1, d2) = lko_sat_d1_d2(r);
            assert_eq!(sat.to_bits(), super::super::lko_sat_func(r).to_bits());
            // A second difference of a nearly linear function loses ~eps/h^2
            // digits, so the second derivative uses a wider step.
            let h1 = 1e-6 * r.max(1.0);
            let (sp1, _, _) = lko_sat_d1_d2(r + h1);
            let (sm1, _, _) = lko_sat_d1_d2(r - h1);
            let fd1 = (sp1 - sm1) / (2.0 * h1);
            let h2 = 1e-3 * r.max(1.0);
            let (sp2, _, _) = lko_sat_d1_d2(r + h2);
            let (sm2, _, _) = lko_sat_d1_d2(r - h2);
            let fd2 = (sp2 - 2.0 * sat + sm2) / (h2 * h2);
            let scale1 = 1.0 + d1.abs();
            let scale2 = 1.0 + d2.abs();
            assert!(
                (d1 - fd1).abs() < 1e-6 * scale1,
                "r={r}: sat'={d1} fd={fd1}"
            );
            assert!(
                (d2 - fd2).abs() < 1e-4 * scale2,
                "r={r}: sat''={d2} fd={fd2}"
            );
        }
    }

    /// `k` and `k'` are finite and consistent with `sat`.
    #[test]
    fn curvature_coefficients_are_consistent() {
        for k in 1..2000 {
            let r = 0.5 + 20.0 * k as f64 / 2000.0;
            let (sat, sp, spp) = lko_sat_d1_d2(r);
            let kk = sp / sat;
            let kp = spp / sat - kk * kk;
            assert!(kk.is_finite() && kp.is_finite(), "r={r}");
            // d(k)/dr = k' by finite difference.
            let h = 1e-6 * r.max(1.0);
            let (s1, sp1, _) = lko_sat_d1_d2(r + h);
            let (s0, sp0, _) = lko_sat_d1_d2(r - h);
            let fdk = (sp1 / s1 - sp0 / s0) / (2.0 * h);
            assert!(
                (kp - fdk).abs() < 1e-3 * (1.0 + kp.abs()),
                "r={r}: k'={kp} fd={fdk}"
            );
        }
    }
}
