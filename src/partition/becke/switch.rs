//! Becke cell-function switch and its derivatives.
//!
//! The classic Becke scheme applies the polynomial
//!
//! ```text
//! p(x) = 3/2 x - 1/2 x^3
//! ```
//!
//! `hardness` times (function composition, Becke 1988 eq. 19/20) to the
//! size-adjusted coordinate
//!
//! ```text
//! nu(mu) = mu + a_AB (1 - mu^2).
//! ```
//!
//! This module owns the value and the first/second `nu`-derivatives used by
//! the analytic gradient/Hessian kernels.  The value path is shared with the
//! weight kernel, and the hardness-3 specialisation performs exactly the
//! same floating-point operations as the loop, so the ORCA/PySCF bitwise
//! regression outputs are unaffected.
//!
//! The chain rule for one composition layer is
//!
//! ```text
//! p(x)   = 3/2 x - 1/2 x^3
//! p'(x)  = 3/2 - 3/2 x^2
//! p''(x) = -3 x
//! f_{k+1}    = p(f_k)
//! f_{k+1}'   = p'(f_k) f_k'
//! f_{k+1}''  = p''(f_k) (f_k')^2 + p'(f_k) f_k''
//! ```
//!
//! and for `s = 1/2 (1 -+ f)` the derivatives with respect to `mu` are
//!
//! ```text
//! ds/dmu   = -+ 1/2 f' nu'
//! d2s/dmu2 = -+ 1/2 (f'' nu'^2 + f' nu'')
//! nu'  = 1 - 2 a mu,      nu'' = -2 a.
//! ```

/// ORCA's Becke size-adjustment factor `a_AB` clamped to `[-0.5, 0.5]`.
///
/// Let `x = sqrt(r_A / r_B)`, `u = (x - 1)/(x + 1)`; then
/// `a_AB = u/(u^2 - 1)`.  This is the ORCA-compatible size adjustment and,
/// algebraically, PySCF's `treutler_atomic_radii_adjust`
/// (`0.25 (r_B/r_A - r_A/r_B)` on the square-root radii); ORCA's evaluation
/// order is used so the ORCA preset stays bit-stable.
///
/// Only the radius ratio enters, so any common unit is fine.
#[inline]
#[must_use]
pub(crate) fn size_adjust_factor(r_a: f64, r_b: f64) -> f64 {
    if (r_a - r_b).abs() < 1e-14 {
        return 0.0;
    }
    let x = (r_a / r_b).sqrt();
    let u_ab = (x - 1.0) / (x + 1.0);
    (u_ab / (u_ab * u_ab - 1.0)).clamp(-0.5, 0.5)
}

/// One `p` composition layer, written exactly as the weight kernel does.
#[inline(always)]
fn p(x: f64) -> f64 {
    1.5 * x - 0.5 * x * x * x
}

/// `p'(x) = 3/2 (1 - x^2)`.
#[inline(always)]
fn dp(x: f64) -> f64 {
    1.5 - 1.5 * x * x
}

/// `p''(x) = -3 x`.
#[inline(always)]
fn ddp(x: f64) -> f64 {
    -3.0 * x
}

/// Step function `f` applied `hardness` times (eq. 20, hardness 3).
#[inline]
pub(crate) fn step_function(mut f: f64, hardness: usize) -> f64 {
    for _ in 0..hardness {
        f = p(f);
    }
    f
}

/// Hardness-3 step function with the loop unrolled.
///
/// The arithmetic is the same sequence as [`step_function`]; this is only a
/// constant-folded form that lets the weight kernel vectorise without a
/// trip-count loop in the innermost point loop.
#[inline(always)]
pub(crate) fn step3(nu: f64) -> f64 {
    p(p(p(nu)))
}

/// Value and first two `nu`-derivatives of the hardness-`h` switch.
#[inline]
pub(crate) fn value_d2(nu: f64, hardness: usize) -> (f64, f64, f64) {
    let (mut f, mut df, mut ddf) = (nu, 1.0, 0.0);
    for _ in 0..hardness {
        ddf = ddp(f) * df * df + dp(f) * ddf;
        df *= dp(f);
        f = p(f);
    }
    (f, df, ddf)
}

/// Value, first and second `nu`-derivative, dispatching hardness 3.
#[inline]
pub(crate) fn value_d2_any(nu: f64, hardness: usize) -> (f64, f64, f64) {
    if hardness == 3 {
        value_d2_h3(nu)
    } else {
        value_d2(nu, hardness)
    }
}

/// Unrolled hardness-3 value and first two derivatives.
#[inline(always)]
pub(crate) fn value_d2_h3(nu: f64) -> (f64, f64, f64) {
    // Same layer order as `value_d2`: second derivative first (it needs the
    // old first derivative), then the first derivative, then the value.
    let f0 = nu;
    let f1 = p(f0);
    let f2 = p(f1);
    let f3 = p(f2);
    // df3 = dp(f2) dp(f1) dp(f0)
    let df0 = 1.0;
    let df1 = dp(f0) * df0;
    let df2 = dp(f1) * df1;
    let df3 = dp(f2) * df2;
    // dd0 = 0, dd1 = ddp(f0) df0^2 = ddp(f0), then the recursion.
    let dd1 = ddp(f0) * df0 * df0;
    let dd2 = ddp(f1) * df1 * df1 + dp(f1) * dd1;
    let dd3 = ddp(f2) * df2 * df2 + dp(f2) * dd2;
    (f3, df3, dd3)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Finite-difference check of the switch derivatives, hardness 3 and a
    /// generic hardness.
    #[test]
    fn switch_derivatives_match_finite_difference() {
        for &h in &[1usize, 2, 3, 4, 7] {
            for &nu in &[-0.9, -0.4, 0.0, 0.3, 0.87] {
                let (f, df, ddf) = value_d2(nu, h);
                assert_eq!(f.to_bits(), step_function(nu, h).to_bits());
                let e = 1e-4;
                let fp = (step_function(nu + e, h) - step_function(nu - e, h)) / (2.0 * e);
                let fpp = (step_function(nu + e, h) - 2.0 * f + step_function(nu - e, h)) / (e * e);
                assert!(
                    (df - fp).abs() < 1e-5 * (1.0 + df.abs()),
                    "h={h} nu={nu}: df={df} fd={fp}"
                );
                assert!(
                    (ddf - fpp).abs() < 1e-4 * (1.0 + ddf.abs()),
                    "h={h} nu={nu}: ddf={ddf} fd={fpp}"
                );
            }
        }
    }

    /// The unrolled hardness-3 value must be bit-identical to the loop: the
    /// value path is guarded by the reference regression tests.
    #[test]
    fn step3_is_bitwise_identical() {
        for k in 0..200_000 {
            let nu = -2.0 + 4.0 * k as f64 / 200_000.0;
            assert_eq!(step3(nu).to_bits(), step_function(nu, 3).to_bits());
        }
    }

    /// The derivative form is checked to round-off only: LLVM may contract
    /// one form with FMA and not the other, which is irrelevant for a
    /// derivative kernel.
    #[test]
    fn hardness3_specialisation_matches_loop() {
        for k in 0..2000 {
            let nu = -1.2 + 2.4 * k as f64 / 2000.0;
            assert!((step3(nu) - step_function(nu, 3)).abs() < 1e-15);
            let (f, df, ddf) = value_d2_h3(nu);
            let (g, dg, ddg) = value_d2(nu, 3);
            assert!((f - g).abs() < 1e-14);
            assert!((df - dg).abs() < 1e-13);
            assert!((ddf - ddg).abs() < 1e-12);
        }
    }

    #[test]
    fn size_adjust_symmetry() {
        let a = size_adjust_factor(0.7, 0.6);
        assert!((a + size_adjust_factor(0.6, 0.7)).abs() < 1e-14);
        assert!((a + 0.03857583749052301).abs() < 1e-14);
        assert_eq!(size_adjust_factor(0.7, 0.05), -0.5);
        assert_eq!(size_adjust_factor(0.05, 0.7), 0.5);
        assert_eq!(size_adjust_factor(0.7, 0.7), 0.0);
    }
}
