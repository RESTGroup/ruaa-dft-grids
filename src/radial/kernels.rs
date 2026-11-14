//! Radial node mappings: every construction, backend-agnostic.
//!
//! This module holds the actual `(n, xi) -> (r, ...)` kernels.  It is split
//! from the selector ([`crate::radial::Radial`]) and is independent of the
//! element-scale policy.
//!
//! # Weight outputs
//!
//! The three `radial_grid_*` functions return `(r, w)` with the `r^2`
//! measure already included ($\sum_i w_i f(r_i) \approx \int f(r) r^2 dr$). The
//! PySCF `*_xi` functions return `(r, dr)`, exactly like `pyscf.dft.radi`;
//! the caller forms either `4*pi*r^2*dr` or `r^2*dr`.
//!
//! # ORCA constructions
//!
//! The exact ORCA 6.1 "OptM3 with GC (2021)" construction is, for
//! $i = 1,\ldots,n_r$ and $\theta_i = i\pi/(n_r+1)$, with
//! $s = \sin\theta_i$ and $c = \cos\theta_i$:
//!
//! $$
//! t_i = \frac{2i}{n_r+1}
//! \quad - s c\left(0.4244131815783876\ s^2 + \frac{2}{\pi}\right),
//! $$
//!
//! $$
//! r_i = \frac{\xi}{\ln 2}\ln\frac{2}{t_i},
//! \qquad
//! w_i = r_i^2\ \frac{16\xi}{3}\
//! \frac{s^4}{t_i\ln 2\ (n_r+1)}.
//! $$
//!
//! ORCA stores the shells outside-in; we return ascending-r order.  The
//! published-reference [`radial_grid_m3`] (Treutler–Ahlrichs M3, JCP 102,
//! 346 (1995), eq. 19 with $\alpha = 0$) and the internal
//! `radial_grid_opt_m3` (Krack–Köster, JCP 108, 3226 (1998), eqs. 9–10) are
//! kept alongside it.
//!
//! # PySCF constructions
//!
//! The `*_xi` kernels are transcribed from PySCF 2.14.0
//! `pyscf/dft/radi.py`; they take the radial scale `xi` explicitly instead
//! of the element charge.  `treutler_ahlrichs_xi` returns PySCF's reversed
//! node order; the others return ascending order.  The element presets live
//! in [`crate::radial::Radial::xi_for_element`].
//!
//! # Euler–Maclaurin construction (Gaussian 16 default)
//!
//! [`euler_maclaurin_xi`] is the squared-ratio Euler–Maclaurin map used by
//! Gaussian 16:
//!
//! $$
//! r_i = \xi\left(\frac{i+1}{n_\mathrm{rad}-i}\right)^2,
//! \qquad
//! \Delta r_i = \frac{2\xi\ x_i}{(1-x_i)^3(n_\mathrm{rad}+1)},
//! \qquad
//! x_i = \frac{i+1}{n_\mathrm{rad}+1}.
//! $$
//!
//! The scale is supplied by the caller.

use crate::PI;
use crate::params::BRAGG_BOHR;

/// ln(2), cached.
const LN2: f64 = std::f64::consts::LN_2;

/// Constants of the ORCA GC correction term (GC_C2 = 2/ $\pi$).
const GC_C1: f64 = 0.42441318_15783876;
const GC_C2: f64 = core::f64::consts::FRAC_2_PI; // 0.6366197723675814

/// The exact ORCA 6.1 Gauss–Chebyshev / OptM3 radial grid
/// (`RadialGrid OptM3 with GC (2021)`).
///
/// See the module docs for the formula. Points are returned in ascending
/// `r` order (ORCA stores them outside-in; we reverse).
#[must_use]
pub fn radial_grid_orca(n: usize, xi: f64) -> (Vec<f64>, Vec<f64>) {
    let n_f = n as f64;
    let n1 = n_f + 1.0;
    let mut pts = Vec::with_capacity(n);
    let mut wts = Vec::with_capacity(n);
    for i in 1..=n {
        let theta = (i as f64) * PI / n1;
        let (s, c) = theta.sin_cos();
        let t = 2.0 * (i as f64) / n1 - s * c * (GC_C1 * s * s + GC_C2);
        let r = (xi / LN2) * (2.0 / t).ln();
        let w = r * r * (xi * 16.0 / 3.0) * s.powi(4) / (t * LN2 * n1);
        pts.push(r);
        wts.push(w);
    }
    pts.reverse();
    wts.reverse();
    (pts, wts)
}

/// Plain M3 Gauss–Chebyshev (second kind) radial grid with parameter `xi`.
///
/// Nodes `x_i = cos(i*pi/(n+1))`, i = 1..n (Chebyshev extrema), mapped through
/// the M3 transformation; weights include the Jacobian `dr/dx` and the `r^2`
/// measure (see module docs):
///
/// ```text
/// w_i = (pi/(n+1)) * sin(theta_i) * dr/dx |_{x_i} * r_i^2
/// ```
///
/// Points are returned in ascending `r` order.
#[must_use]
pub fn radial_grid_m3(n: usize, xi: f64) -> (Vec<f64>, Vec<f64>) {
    let n_f = n as f64;
    let step = PI / (n_f + 1.0);
    let mut pts = Vec::with_capacity(n);
    let mut wts = Vec::with_capacity(n);
    for i in 1..=n {
        let theta = (i as f64) * step;
        let (s, c) = theta.sin_cos();
        let x = c;
        let r = (xi / LN2) * (2.0 / (1.0 - x)).ln();
        // The second-kind Chebyshev rule integrates g(x)·sqrt(1-x^2); dividing
        // the mapped integrand f(r)·r^2·dr/dx by sqrt(1-x^2) = sin(theta) leaves
        // a single sin(theta) factor, w = (pi/(n+1))·sin(theta)·dr/dx·r^2
        // (OptM3's optimized nodes avoid this; plain M3 converges ~1/n).
        let w = (PI / (n_f + 1.0)) * s * (xi / (LN2 * (1.0 - x))) * r * r;
        pts.push(r);
        wts.push(w);
    }
    // i = 1..n gives x descending from cos(step) ~ 1 to cos(pi*n/(n+1)) ~ -1,
    // i.e. r descending; reverse for ascending r.
    pts.reverse();
    wts.reverse();
    (pts, wts)
}

/// Classic Krack–Köster "OptM3" radial grid (optimized Chebyshev nodes with
/// the M3 mapping), kept as the published-reference alternative.
///
/// Node/weight construction follows M. Krack, A. M. Köster, JCP 108, 3226
/// (1998), eqs. 9–10, with `xi` replacing their fixed `1/ln 2` scale:
///
/// ```text
/// x_i = (n+1-2i)/(n+1)
///       + (2/pi) [1 + (2/3) sin^2(theta_i)] cos(theta_i) sin(theta_i),
/// r_i = (xi/ln 2) ln(2/(1 - x_i)),
/// w_i = (16/(3(n+1))) sin^4(theta_i) * (xi/ln2)/(1-x_i) * r_i^2 .
/// ```
///
/// Points are returned in ascending `r` order.  This is the raw
/// Krack–Köster kernel; [`crate::radial::Radial`] has no variant for it.
#[allow(dead_code)] // reference kernel, exercised by the module tests
#[must_use]
pub(crate) fn radial_grid_opt_m3(n: usize, xi: f64) -> (Vec<f64>, Vec<f64>) {
    let n_f = n as f64;
    let mut pts = Vec::with_capacity(n);
    let mut wts = Vec::with_capacity(n);
    for i in 1..=n {
        let theta = (i as f64) * PI / (n_f + 1.0);
        let (s, c) = theta.sin_cos();
        let x = (n_f + 1.0 - 2.0 * (i as f64)) / (n_f + 1.0)
            + (2.0 / PI) * (1.0 + (2.0 / 3.0) * s * s) * c * s;
        let r = (xi / LN2) * (2.0 / (1.0 - x)).ln();
        let w = (16.0 / (3.0 * (n_f + 1.0))) * s.powi(4) * (xi / (LN2 * (1.0 - x))) * r * r;
        pts.push(r);
        wts.push(w);
    }
    pts.reverse();
    wts.reverse();
    (pts, wts)
}

/// Delley's fixed outer radius (`radi.delley`'s `r_outer`), in Bohr.
pub(crate) const DELLEY_R_OUTER: f64 = 12.0;

/// Unit scale reproducing PySCF `radi.gauss_chebyshev` exactly.  The PySCF
/// function has no free scale; a non-unit `xi` applies a similarity
/// transform `r -> xi*r`, which is the natural way to expose it as a
/// scale-parameterized kernel.
pub(crate) const GAUSS_CHEBYSHEV_SCALE: f64 = 1.0;

/// PySCF `radi.treutler_ahlrichs` (M4), returned in PySCF's (reversed) order.
pub(crate) fn treutler_ahlrichs_xi(n: usize, xi: f64) -> (Vec<f64>, Vec<f64>) {
    let step = PI / (n as f64 + 1.0);
    let ln2 = xi / std::f64::consts::LN_2;
    let mut r = vec![0.0; n];
    let mut dr = vec![0.0; n];
    for i in 0..n {
        let x = ((i + 1) as f64 * step).cos();
        let log_term = ((1.0 - x) / 2.0).ln();
        r[i] = -ln2 * (1.0 + x).powf(0.6) * log_term;
        dr[i] = step
            * ((i + 1) as f64 * step).sin()
            * ln2
            * (1.0 + x).powf(0.6)
            * (-0.6 / (1.0 + x) * log_term + 1.0 / (1.0 - x));
    }
    r.reverse();
    dr.reverse();
    (r, dr)
}

/// PySCF `radi.delley` (log2) with an explicit outer radius.
pub(crate) fn delley_xi(n: usize, r_outer: f64) -> (Vec<f64>, Vec<f64>) {
    let mut r = vec![0.0; n];
    let mut dr = vec![0.0; n];
    let step = 1.0 / (n as f64 + 1.0);
    let rfac = r_outer / (1.0 - (n as f64 * step).powi(2)).ln();
    for i in 1..=n {
        let i_f = i as f64;
        let u = 1.0 - (i_f * step).powi(2);
        r[i - 1] = rfac * u.ln();
        dr[i - 1] = rfac * (-2.0 * i_f * step.powi(2)) / u;
    }
    (r, dr)
}

/// PySCF `radi.mura_knowles` scale rule: 7 Bohr for Li, Be, Na, Mg, K, Ca;
/// 5.2 Bohr otherwise.
pub(crate) fn mura_knowles_far(z: u32) -> f64 {
    if matches!(z, 3 | 4 | 11 | 12 | 19 | 20) {
        7.0
    } else {
        5.2
    }
}

/// PySCF `radi.mura_knowles` (log3) with an explicit `far` scale.
pub(crate) fn mura_knowles_xi(n: usize, far: f64) -> (Vec<f64>, Vec<f64>) {
    let mut r = vec![0.0; n];
    let mut dr = vec![0.0; n];
    for i in 0..n {
        let x = (i as f64 + 0.5) / n as f64;
        let u = 1.0 - x.powi(3);
        r[i] = -far * u.ln();
        dr[i] = far * 3.0 * x * x / (u * n as f64);
    }
    (r, dr)
}

/// PySCF `radi.gauss_chebyshev` (M3, PySCF's own mapping — deliberately not
/// ORCA's `OptM3 with GC`) with a similarity scale.
///
/// With $x_i = i\pi/(n+1)$ and
/// $\xi_i = (n-1-2(i-1))/(n+1) + (1 + 2/3 \sin^2x_i)\cdot \sin(2x_i)/\pi$
/// (symmetrized as $(\xi_i - \xi_{n+1-i})/2$), the mapping is
/// $r_i = scale\cdot(1 - \log_2(1+\xi_i))$ and
/// $dr_i = scale\cdot(16/(3(n+1)))\cdot \sin^4x_i / ((1+\xi_i)\cdot \ln 2)$.
///
/// `scale = 1` reproduces PySCF exactly; other scales are a pure similarity
/// transform of the nodes and the derivative.
pub(crate) fn gauss_chebyshev_xi(n: usize, scale: f64) -> (Vec<f64>, Vec<f64>) {
    let ln2 = 1.0 / std::f64::consts::LN_2;
    let fac = 16.0 / 3.0 / (n as f64 + 1.0);
    let mut x1 = vec![0.0; n];
    let mut xi = vec![0.0; n];
    for i in 1..=n {
        let x = i as f64 * PI / (n as f64 + 1.0);
        x1[i - 1] = x;
        xi[i - 1] = (n as f64 - 1.0 - 2.0 * (i - 1) as f64) / (n as f64 + 1.0)
            + (1.0 + 2.0 / 3.0 * x.sin().powi(2)) * (2.0 * x).sin() / PI;
    }
    let rev: Vec<f64> = xi.iter().rev().copied().collect();
    for i in 0..n {
        xi[i] = (xi[i] - rev[i]) / 2.0;
    }
    let mut r = vec![0.0; n];
    let mut dr = vec![0.0; n];
    for i in 0..n {
        r[i] = scale * (1.0 - (1.0 + xi[i]).ln() * ln2);
        dr[i] = scale * (fac * x1[i].sin().powi(4) * ln2 / (1.0 + xi[i]));
    }
    (r, dr)
}

/// PySCF `radi.becke` radius scale: `BRAGG[z]`, halved for every element
/// except hydrogen. `None` when the PySCF BRAGG table does not cover `z`.
pub(crate) fn becke_rm(z: u32) -> Option<f64> {
    let r = pyscf_bragg(z)?;
    Some(if z == 1 { r } else { r * 0.5 })
}

/// PySCF `radi.becke` with an explicit `rm` scale.
pub(crate) fn becke_xi(n: usize, rm: f64) -> (Vec<f64>, Vec<f64>) {
    let mut r = vec![0.0; n];
    let mut dr = vec![0.0; n];
    for i in 1..=n {
        let theta = i as f64 * PI / (n as f64 + 1.0);
        let t = theta.cos();
        let w = PI / (n as f64 + 1.0) * theta.sin();
        r[i - 1] = (1.0 + t) / (1.0 - t) * rm;
        dr[i - 1] = w * 2.0 / (1.0 - t).powi(2) * rm;
    }
    (r, dr)
}

fn pyscf_bragg(z: u32) -> Option<f64> {
    BRAGG_BOHR.get(z as usize).copied()
}

/// Gaussian 16 squared-ratio Euler–Maclaurin radial grid with scale $\xi$:
///
/// $$
/// q_i = \frac{i+1}{n_\mathrm{rad}-i},
/// \qquad
/// r_i = \xi q_i^2,
/// $$
///
/// $$
/// x_i = \frac{i+1}{n_\mathrm{rad}+1},
/// \qquad
/// \Delta r_i = \frac{2\xi\ x_i}{(1-x_i)^3(n_\mathrm{rad}+1)}.
/// $$
///
/// Returns $(r, \Delta r)$ in ascending radius (the $4\pi r^2$ measure is
/// applied by [`crate::radial::Radial::grid_xi`]).
#[must_use]
pub fn euler_maclaurin_xi(n: usize, xi: f64) -> (Vec<f64>, Vec<f64>) {
    let mut r = Vec::with_capacity(n);
    let mut dr = Vec::with_capacity(n);
    let n1 = (n + 1) as f64;
    for i in 0..n {
        let q = (i + 1) as f64 / (n - i) as f64;
        let x = (i + 1) as f64 / n1;
        r.push(xi * q * q);
        dr.push(xi * 2.0 * x / (1.0 - x).powi(3) / n1);
    }
    (r, dr)
}

/// Exact dimensionless node ratios $((i+1)/(n-i))^2$ of the Euler–Maclaurin
/// map.
///
/// The five-batch ladder compares shells against fixed boundaries; deriving
/// the ratio as $r / \xi$ would let a rounding step move a tie shell across a
/// boundary (`17/34 = 0.5`-style ties), so the exact integer ratio is kept
/// separately.
#[must_use]
pub(crate) fn euler_maclaurin_ratios(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let q = (i + 1) as f64 / (n - i) as f64;
            q * q
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn total_charge(f: impl Fn(f64) -> f64, rs: &[f64], ws: &[f64]) -> f64 {
        rs.iter().zip(ws).map(|(&r, &w)| w * f(r)).sum()
    }

    /// The radial grid integrates `r^2 * exp(-r)` (atomic unit of the
    /// hydrogen-like density tail) to `2! = 2`.
    #[test]
    fn integrates_r2_exp_minus_r() {
        for &(n, xi) in &[(60usize, 1.0), (100, 1.0), (60, 0.8), (80, 1.4)] {
            let (rs, ws) = radial_grid_m3(n, xi);
            let val = total_charge(|r| (-r).exp(), &rs, &ws);
            let err = (val - 2.0).abs();
            // plain M3 converges algebraically (~1/n) in n
            assert!(err < 5e-2, "M3 n={n} xi={xi}: |{val}-2| = {err}");
        }
    }

    #[test]
    fn opt_m3_integrates_r2_exp_minus_r() {
        for &n in &[40usize, 70] {
            let (rs, ws) = radial_grid_opt_m3(n, 1.0);
            let val = total_charge(|r| (-r).exp(), &rs, &ws);
            assert!((val - 2.0).abs() < 5e-3, "OptM3 n={n}: {val}");
        }
    }

    /// The exact ORCA grid converges spectrally: `r^2 e^-r` -> 2 and
    /// `e^-r^2` -> sqrt(pi)/2 (radial part of the unit Gaussian).
    #[test]
    fn orca_grid_spectral_convergence() {
        for &(n, xi) in &[(30usize, 1.0), (50, 0.8), (50, 1.1), (35, 0.8)] {
            let (rs, ws) = radial_grid_orca(n, xi);
            let val = total_charge(|r| (-r).exp(), &rs, &ws);
            assert!(
                (val - 2.0).abs() < 2e-4,
                "ORCA GC n={n} xi={xi}: r2e^-r = {val}"
            );
            let g: f64 = rs.iter().zip(&ws).map(|(&r, &w)| w * (-r * r).exp()).sum();
            // radial part of a unit Gaussian: int_0^inf r^2 e^{-r^2} dr = sqrt(pi)/4
            let target = std::f64::consts::PI.sqrt() / 4.0;
            assert!((g - target).abs() < 1e-4, "ORCA GC n={n}: gaussian {g}");
        }
    }

    /// Shape of the GC-corrected grid used by the preset data (35 shells,
    /// `xi = 0.8`): all nodes positive, ascending r, the innermost shell near
    /// the nucleus, the outermost far out, weights finite and positive.
    #[test]
    fn gc_grid_h_shape() {
        let nr = 35;
        let (rs, ws) = radial_grid_orca(nr, 0.8);
        assert_eq!(rs.len(), 35);
        for (&r, &w) in rs.iter().zip(&ws) {
            assert!(r > 0.0 && w > 0.0 && w.is_finite(), "r={r} w={w}");
        }
        assert!(rs.windows(2).all(|w| w[0] < w[1]), "ascending r");
        // The GC correction (2/pi)·s·c cancels t = 2i/(nr+1) at BOTH ends
        // (t -> 0 at i = 1, t -> 2 at i = nr): the outermost shell lands
        // far out (~16 Bohr for H xi = 0.8, nr = 35) and the innermost
        // starts essentially at the nucleus (~1e-6 Bohr).
        assert!(rs[0] > 0.0 && rs[0] < 0.01, "innermost shell r = {}", rs[0]);
        assert!(
            (16.0..=16.5).contains(&rs[34]),
            "outermost shell r = {}",
            rs[34]
        );
    }

    /// Constant function: `sum w_i` should approach `integral r^2 dr -> inf`,
    /// i.e. grow with the largest radius; just check positivity and that
    /// weights are finite and positive.
    #[test]
    fn weights_positive() {
        for grid in [
            radial_grid_m3(50, 1.0),
            radial_grid_opt_m3(50, 1.1),
            radial_grid_orca(50, 1.1),
        ] {
            let (rs, ws) = grid;
            assert_eq!(rs.len(), ws.len());
            for (&r, &w) in rs.iter().zip(&ws) {
                assert!(r > 0.0 && w > 0.0 && w.is_finite());
            }
            assert!(rs.windows(2).all(|w| w[0] < w[1]), "ascending r");
        }
    }
}
