//! Cell-function switches for the compact-support pair products.
//!
//! The SSF and Gaussian SS partitions share one pair-product kernel; they
//! differ only in the switch `f(nu)`.  Each switch is a zero-sized type
//! implementing [`SsSwitch`], so the kernel is monomorphised rather than
//! branching on a runtime scheme.

use crate::partition::simd::{self, V};

/// A compact-support switch `f(nu)` in `[-1, 1]` and its derivatives.
///
/// A cell factor is `s = 1/2 (1 - f)`.  Outside the support, `f` is exactly
/// `-1` on the negative side (so `s = 1`) and exactly `+1` on the positive
/// side (so `s = 0`).  The support bounds may be asymmetric.
pub(crate) trait SsSwitch: Copy + std::fmt::Debug + Send + Sync + 'static {
    /// Lower support bound: `f = -1` for `nu <= MU_MIN`.
    const MU_MIN: f64;
    /// Upper support bound: `f = +1` for `nu >= MU_MAX`.
    const MU_MAX: f64;
    /// Whether `f(-nu) = -f(nu)`, so the partner factor is `1/2 (1 + f)`.
    const COMPLEMENTARY_FACTOR: bool;

    /// Switch value.
    fn switch(g: f64) -> f64;

    /// Switch value and first derivative.
    fn switch_d1(g: f64) -> (f64, f64);

    /// Switch value and first two derivatives.
    fn switch_d2(g: f64) -> (f64, f64, f64);

    /// Switch value for eight lanes.
    fn switch_vec(g: V) -> V;
}

/// PySCF/ORCA SSF switch: `A = 0.64`, degree 7, symmetric support.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StratmannSwitch;

/// SSF cutoff `A` (PySCF `gen_grid.stratmann`, eq. 14).
pub(crate) const A: f64 = 0.64;
/// Exact `25/16 = 1 / 0.64`, used instead of a division in the hot switch.
/// `0.64` is *not* an exact binary fraction, so this differs from PySCF's
/// `g / a` by one rounding in `ma` (`~1e-17` relative); the switch value
/// differs by `~1e-16`, far below the `1e-10` PySCF asset tolerance.
pub(crate) const INV_A: f64 = 1.5625;

impl SsSwitch for StratmannSwitch {
    const MU_MIN: f64 = -A;
    const MU_MAX: f64 = A;
    const COMPLEMENTARY_FACTOR: bool = true;

    #[inline]
    fn switch(g: f64) -> f64 {
        if g <= -A {
            return -1.0;
        }
        if g >= A {
            return 1.0;
        }
        let ma = g * INV_A;
        let ma2 = ma * ma;
        (1.0 / 16.0) * (ma * (35.0 + ma2 * (-35.0 + ma2 * (21.0 - 5.0 * ma2))))
    }

    #[inline]
    fn switch_d1(g: f64) -> (f64, f64) {
        if g <= -A {
            return (-1.0, 0.0);
        }
        if g >= A {
            return (1.0, 0.0);
        }
        let ma = g * INV_A;
        let ma2 = ma * ma;
        let om = 1.0 - ma2;
        let f = (1.0 / 16.0) * (ma * (35.0 + ma2 * (-35.0 + ma2 * (21.0 - 5.0 * ma2))));
        let df = (35.0 / (16.0 * A)) * om * om * om;
        (f, df)
    }

    #[inline]
    fn switch_d2(g: f64) -> (f64, f64, f64) {
        if g <= -A {
            return (-1.0, 0.0, 0.0);
        }
        if g >= A {
            return (1.0, 0.0, 0.0);
        }
        let ma = g * INV_A;
        let ma2 = ma * ma;
        let om = 1.0 - ma2;
        let f = (1.0 / 16.0) * (ma * (35.0 + ma2 * (-35.0 + ma2 * (21.0 - 5.0 * ma2))));
        let df = (35.0 / (16.0 * A)) * om * om * om;
        let ddf = -(210.0 / (16.0 * A * A)) * ma * om * om;
        (f, df, ddf)
    }

    #[inline]
    fn switch_vec(g: V) -> V {
        let ma = g * simd::splat(INV_A);
        let ma2 = ma * ma;
        let poly = simd::splat(1.0 / 16.0)
            * (ma
                * (simd::splat(35.0)
                    + ma2
                        * (simd::splat(-35.0)
                            + ma2 * (simd::splat(21.0) - simd::splat(5.0) * ma2))));
        let neg = g.simd_le(simd::splat(-A));
        let pos = g.simd_ge(simd::splat(A));
        let neg_val = simd::splat(-1.0).mask_select(neg, poly);
        simd::splat(1.0).mask_select(pos, neg_val)
    }
}

/// Gaussian 16 `SSWeights` switch: `XScal = 0.86`, degree 13.
///
/// `P_A = 1/2 - F1(mu / 0.86)` with
///
/// ```text
/// F1(x) = (3003/2048) x (1 - 2x^2 + 3x^4 - (20/7)x^6 + (5/3)x^8
///                         - (6/11)x^10 + (1/13)x^12).
/// ```
///
/// Gaussian clips the switch at `mu <= -0.86` (`P = 1`) and at
/// `mu >= 0.852` (`P = 0`); the second bound is slightly inside the
/// polynomial's endpoint.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GaussianSsSwitch;

/// Gaussian `XScal` denominator.
const GAUSSIAN_X: f64 = 0.86;
/// `1 / XScal`; multiplication is cheaper than a division in the hot loop.
const GAUSSIAN_INV_X: f64 = 1.0 / GAUSSIAN_X;
/// Gaussian positive-side clip (`P = 0`).
pub(crate) const GAUSSIAN_MU_POS: f64 = 0.852;
/// `3003/2048`, the overall polynomial normalization.
const GAUSSIAN_C: f64 = 3003.0 / 2048.0;

/// `Q(y)` in `F1(x) = C x Q(x^2)`.
#[inline(always)]
fn gaussian_q(y: f64) -> f64 {
    1.0 + y
        * (-2.0
            + y * (3.0
                + y * (-20.0 / 7.0 + y * (5.0 / 3.0 + y * (-6.0 / 11.0 + y * (1.0 / 13.0))))))
}

/// `R(y) = Q(y) + 2y Q'(y)`, so `F1'(x) = C R(x^2)`.
#[inline(always)]
fn gaussian_r(y: f64) -> f64 {
    1.0 + y * (-6.0 + y * (15.0 + y * (-20.0 + y * (15.0 + y * (-6.0 + y)))))
}

/// `R'(y)`, so `F1''(x) = 2 C x R'(x^2)`.
#[inline(always)]
fn gaussian_s(y: f64) -> f64 {
    -6.0 + y * (30.0 + y * (-60.0 + y * (60.0 + y * (-30.0 + y * 6.0))))
}

impl SsSwitch for GaussianSsSwitch {
    const MU_MIN: f64 = -GAUSSIAN_X;
    const MU_MAX: f64 = GAUSSIAN_MU_POS;
    const COMPLEMENTARY_FACTOR: bool = false;

    #[inline]
    fn switch(g: f64) -> f64 {
        if g <= -GAUSSIAN_X {
            return -1.0;
        }
        if g >= GAUSSIAN_MU_POS {
            return 1.0;
        }
        let x = g * GAUSSIAN_INV_X;
        2.0 * GAUSSIAN_C * x * gaussian_q(x * x)
    }

    #[inline]
    fn switch_d1(g: f64) -> (f64, f64) {
        if g <= -GAUSSIAN_X {
            return (-1.0, 0.0);
        }
        if g >= GAUSSIAN_MU_POS {
            return (1.0, 0.0);
        }
        let x = g * GAUSSIAN_INV_X;
        let y = x * x;
        (
            2.0 * GAUSSIAN_C * x * gaussian_q(y),
            2.0 * GAUSSIAN_C * GAUSSIAN_INV_X * gaussian_r(y),
        )
    }

    #[inline]
    fn switch_d2(g: f64) -> (f64, f64, f64) {
        if g <= -GAUSSIAN_X {
            return (-1.0, 0.0, 0.0);
        }
        if g >= GAUSSIAN_MU_POS {
            return (1.0, 0.0, 0.0);
        }
        let x = g * GAUSSIAN_INV_X;
        let y = x * x;
        (
            2.0 * GAUSSIAN_C * x * gaussian_q(y),
            2.0 * GAUSSIAN_C * GAUSSIAN_INV_X * gaussian_r(y),
            4.0 * GAUSSIAN_C * GAUSSIAN_INV_X * GAUSSIAN_INV_X * x * gaussian_s(y),
        )
    }

    #[inline]
    fn switch_vec(g: V) -> V {
        let x = g * simd::splat(GAUSSIAN_INV_X);
        let y = x * x;
        let q = simd::splat(1.0)
            + y * (simd::splat(-2.0)
                + y * (simd::splat(3.0)
                    + y * (simd::splat(-20.0 / 7.0)
                        + y * (simd::splat(5.0 / 3.0)
                            + y * (simd::splat(-6.0 / 11.0) + y * simd::splat(1.0 / 13.0))))));
        let poly = simd::splat(2.0 * GAUSSIAN_C) * x * q;
        let neg = g.simd_le(simd::splat(-GAUSSIAN_X));
        let pos = g.simd_ge(simd::splat(GAUSSIAN_MU_POS));
        let neg_val = simd::splat(-1.0).mask_select(neg, poly);
        simd::splat(1.0).mask_select(pos, neg_val)
    }
}

/// Whether pair `nu` lies strictly inside the switch support.
#[inline(always)]
pub(crate) fn active<S: SsSwitch>(g: f64) -> bool {
    g > S::MU_MIN && g < S::MU_MAX
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stratmann_value_and_derivatives_match_finite_difference() {
        for k in 0..5000 {
            let x = -1.2 + 2.4 * k as f64 / 5000.0;
            let (f, df, ddf) = StratmannSwitch::switch_d2(x);
            assert_eq!(f.to_bits(), StratmannSwitch::switch(x).to_bits());
            let (f1, _df1) = StratmannSwitch::switch_d1(x);
            assert_eq!(f.to_bits(), f1.to_bits());
            let h = 1e-4;
            let fd = (StratmannSwitch::switch(x + h) - StratmannSwitch::switch(x - h)) / (2.0 * h);
            assert!((df - fd).abs() < 1e-6, "x={x} df={df} fd={fd}");
            let fd2 = (StratmannSwitch::switch(x + h) - 2.0 * f + StratmannSwitch::switch(x - h))
                / (h * h);
            assert!((ddf - fd2).abs() < 1e-4, "x={x} ddf={ddf} fd2={fd2}");
        }
    }

    #[test]
    fn gaussian_value_and_derivatives_match_finite_difference() {
        for k in 0..5000 {
            let x = -1.2 + 2.4 * k as f64 / 5000.0;
            let (f, df, ddf) = GaussianSsSwitch::switch_d2(x);
            assert_eq!(f.to_bits(), GaussianSsSwitch::switch(x).to_bits());
            let (f1, _df1) = GaussianSsSwitch::switch_d1(x);
            assert_eq!(f.to_bits(), f1.to_bits());
            let h = 1e-4;
            let fd =
                (GaussianSsSwitch::switch(x + h) - GaussianSsSwitch::switch(x - h)) / (2.0 * h);
            assert!((df - fd).abs() < 1e-6, "x={x} df={df} fd={fd}");
            let fd2 = (GaussianSsSwitch::switch(x + h) - 2.0 * f + GaussianSsSwitch::switch(x - h))
                / (h * h);
            assert!((ddf - fd2).abs() < 1e-4, "x={x} ddf={ddf} fd2={fd2}");
        }
    }

    #[test]
    fn gaussian_formula_matches_g16_polynomial() {
        // The G16 cell value P = 1/2 (1 - f), including the asymmetric clips.
        for mu in [-0.95, -0.86, -0.5, 0.0, 0.5, 0.84, 0.852, 0.9] {
            let p = 0.5 * (1.0 - GaussianSsSwitch::switch(mu));
            let want = if mu <= -0.86 {
                1.0
            } else if mu >= 0.852 {
                0.0
            } else {
                let x = mu / 0.86;
                let f1 = GAUSSIAN_C
                    * x
                    * (1.0 - 2.0 * x.powi(2) + 3.0 * x.powi(4) - (20.0 / 7.0) * x.powi(6)
                        + (5.0 / 3.0) * x.powi(8)
                        - (6.0 / 11.0) * x.powi(10)
                        + (1.0 / 13.0) * x.powi(12));
                0.5 - f1
            };
            assert!((p - want).abs() < 1e-15, "mu={mu}: {p} vs {want}");
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn gaussian_switch_matches_g16_reference_fractions() {
        // G16 SSWeights fractions for HeH at 2 Bohr; P = 1/2 (1 - f).
        let reference = [
            (-7.99984179819483554e-01, 9.99999910260342961e-01),
            (-5.99668895688807879e-01, 9.98660936653782305e-01),
            (-3.01649463753965730e-01, 9.08570226132438097e-01),
            (2.25955970031144204e-03, 4.96147481321425221e-01),
            (1.99447994564681652e-01, 1.93714911726396072e-01),
            (4.50106017493790755e-01, 1.87385979354260934e-02),
            (5.99672931420137334e-01, 1.33893677538121604e-03),
            (7.46706328245057893e-01, 6.47090637745373258e-06),
            (8.30147985874877481e-01, 7.42647277007790556e-10),
            (8.47463135143079249e-01, 0.0),
        ];
        for (mu, want) in reference {
            let got = 0.5 * (1.0 - GaussianSsSwitch::switch(mu));
            assert!(
                (got - want).abs() < 3e-10,
                "mu={mu}: crate {got:.12e} vs G16 {want:.12e}"
            );
        }
    }

    #[test]
    fn vector_values_match_scalar() {
        for k in 0..2000 {
            let x = -1.2 + 2.4 * k as f64 / 2000.0;
            let mut lane = [0.0; simd::LANES];
            for (l, v) in lane.iter_mut().enumerate() {
                *v = x + 1e-3 * l as f64;
            }
            let v = V::from(lane);
            let sv = StratmannSwitch::switch_vec(v);
            let gv = GaussianSsSwitch::switch_vec(v);
            for (l, x) in lane.iter().enumerate() {
                assert_eq!(sv[l].to_bits(), StratmannSwitch::switch(*x).to_bits());
                assert_eq!(gv[l].to_bits(), GaussianSsSwitch::switch(*x).to_bits());
            }
        }
    }
}
