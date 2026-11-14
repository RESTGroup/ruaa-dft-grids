//! Angular quadrature on the unit sphere.
//!
//! Weights use the canonical (PySCF) normalization: they **sum to one**.
//! The $4\pi$ factor lives entirely in the radial weight
//! ([`crate::radial::Radial::grid`], $4\pi r^2\ \Delta r$), so the product
//! $w_\mathrm{rad} w_\mathrm{ang}$ is the volume element and the placement of
//! the constant is invisible to the integrator.
//!
//! The internal Oh-orbit generator is an Apache-2.0 rstsr adaptation and
//! already sums to 1, but it has no 12-point rule.  [`angular_grid`] adds the
//! 12-vertex icosahedral rule (not an Oh orbit) with weight `1/12`.
//!
//! The module also holds the angular rules that are useful to any backend:
//! the 18-point octahedral rule ([`octahedral_18`]) and the Gauss–Legendre
//! $\times$ uniform-$\varphi$ product grid ([`gauss_legendre`],
//! [`product_angular_grid`]).
//!
//! The 12- and 18-point rules are fixed constructions.  Their orientation is
//! covered by moment, invariance and construction-pin tests.

use crate::PI;
use crate::error::{Error, Result};
use crate::lebedev;

/// One angular quadrature: unit-sphere directions `(x, y, z)` and weights
/// that sum to `1` (the $4\pi$ lives in the radial weight).
pub type AngularPoints = (Vec<(f64, f64, f64)>, Vec<f64>);

/// Angular quadrature of `num_points` directions: `(points, weights)` with the
/// weights summing to `1`, including the 12-point icosahedral and 18-point
/// octahedral rules.
///
/// The one-point rule is the trivial `MakeAngularGrid(1)` grid: the origin
/// with weight one.  PySCF's `gen_atomic_grids` selects that rule for angular
/// sizes 0 and 1 but then assumes a two-dimensional direction array; the
/// crate represents the rule directly.
///
/// ```
/// use ruaa_dft_grids::angular::angular_grid;
///
/// # fn main() -> ruaa_dft_grids::Result<()> {
/// let (dirs, weights) = angular_grid(50)?;
/// assert_eq!(dirs.len(), 50);
/// assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-14);
/// assert!(dirs.iter().all(|d| (d.0 * d.0 + d.1 * d.1 + d.2 * d.2 - 1.0).abs() < 1e-12));
/// # Ok(())
/// # }
/// ```
///
/// # Errors
/// [`Error::UnsupportedLebedevGrid`] for an unsupported point count (see
/// [`crate::params::LEBEDEV_NGRID`]).
pub fn angular_grid(num_points: usize) -> Result<AngularPoints> {
    if num_points == 0 {
        return Err(Error::UnsupportedLebedevGrid { points: num_points });
    }
    if num_points == 1 {
        return Ok((vec![(0.0, 0.0, 0.0)], vec![1.0]));
    }
    match num_points {
        12 => Ok(angular_grid_12()),
        18 => Ok(octahedral_18()),
        _ => lebedev::angular_grid(num_points)
            .ok_or(Error::UnsupportedLebedevGrid { points: num_points }),
    }
}

/// Whether [`angular_grid`] can build `num_points` directions.
#[must_use]
pub fn is_supported_size(num_points: usize) -> bool {
    num_points == 1
        || num_points == 12
        || num_points == 18
        || crate::params::LEBEDEV_NGRID.contains(&num_points)
}

/// The 12-point spherical quadrature: the 12 vertices of the icosahedron,
/// weight `1/12`.
///
/// Vertices are the cyclic permutations of
/// $(0, \pm1, \pm\varphi)$ with $\varphi = (1+\sqrt{5})/2$, normalized to
/// the unit sphere; the icosahedral symmetry annihilates every spherical
/// harmonic with $0 < l < 6$, so the rule is exact through algebraic
/// degree 5.
fn angular_grid_12() -> AngularPoints {
    let phi = (1.0 + 5.0f64.sqrt()) / 2.0;
    let norm = (1.0 + phi * phi).sqrt();
    let (a, b) = (1.0 / norm, phi / norm);
    let mut pts = Vec::with_capacity(12);
    // (0, ±a, ±b), (±b, 0, ±a), (±a, ±b, 0)
    for &s1 in &[-1.0, 1.0] {
        for &s2 in &[-1.0, 1.0] {
            pts.push([
                (0.0, s1 * a, s2 * b),
                (s2 * b, 0.0, s1 * a),
                (s1 * a, s2 * b, 0.0),
            ]);
        }
    }
    let points: Vec<(f64, f64, f64)> = pts.into_iter().flatten().collect();
    let weights = vec![1.0 / 12.0; 12];
    (points, weights)
}

/// The 18-point octahedral rule: the
/// six axis points and twelve edge midpoints of the octahedron, weights
/// `1/30` and `1/15` (edge/axis weight ratio 2).
#[must_use]
pub fn octahedral_18() -> AngularPoints {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let mut dirs = Vec::with_capacity(18);
    for v in [1.0, -1.0] {
        dirs.push((v, 0.0, 0.0));
        dirs.push((0.0, v, 0.0));
        dirs.push((0.0, 0.0, v));
    }
    for &(a, b) in &[(s, s), (s, -s), (-s, s), (-s, -s)] {
        dirs.push((0.0, a, b));
        dirs.push((a, 0.0, b));
        dirs.push((a, b, 0.0));
    }
    let weights = vec![1.0 / 30.0; 6]
        .into_iter()
        .chain(std::iter::repeat_n(1.0 / 15.0, 12))
        .collect::<Vec<_>>();
    (dirs, weights)
}

/// Legendre polynomial `P_n(x)` and its derivative (recurrence).
fn legendre(n: usize, x: f64) -> (f64, f64) {
    let (mut p0, mut p1) = (1.0, x);
    for k in 2..=n {
        let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
        p0 = p1;
        p1 = p2;
    }
    if n == 0 {
        return (1.0, 0.0);
    }
    let dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
    (p1, dp)
}

/// Gauss–Legendre nodes $x_i$ (descending) and weights on $(-1, 1)$ from
/// Newton iteration starting at
/// $x_0 = \cos((i\pi - \pi/4)/(n + 1/2))$, with
/// $w_i = 2(1-x_i^2)/((n+1)^2 P_{n+1}(x_i)^2)$.
///
/// ```
/// use ruaa_dft_grids::angular::gauss_legendre;
///
/// let (x, w) = gauss_legendre(8);
/// assert_eq!(x.len(), 8);
/// assert!((w.iter().sum::<f64>() - 2.0).abs() < 1e-14);
/// // Gauss–Legendre integrates x^3 exactly on [-1, 1].
/// let moment: f64 = x.iter().zip(&w).map(|(x, w)| w * x.powi(3)).sum();
/// assert!(moment.abs() < 1e-14);
/// ```
#[must_use]
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut xs = Vec::with_capacity(n);
    let mut ws = Vec::with_capacity(n);
    for i in 1..=n {
        let mut x = ((i as f64) * PI - core::f64::consts::FRAC_PI_4) / (n as f64 + 0.5);
        x = x.cos();
        for _ in 0..100 {
            let (pn, dpn) = legendre(n, x);
            let dx = pn / dpn;
            x -= dx;
            if dx.abs() < 1e-14 {
                break;
            }
        }
        let p_np1 = legendre(n + 1, x).0;
        xs.push(x);
        ws.push(2.0 * (1.0 - x * x) / ((n as f64 + 1.0).powi(2) * p_np1 * p_np1));
    }
    (xs, ws)
}

/// Product angular grid of a numeric `-mmmnnn` construction: $n_\theta$
/// Gauss–Legendre $\theta$ rings, each with $2n_\theta$ uniform $\varphi$
/// points.  Weights sum to one.
///
/// # Errors
/// [`Error::UnsupportedGrid`] for $n_\theta < 2$.
///
/// ```
/// use ruaa_dft_grids::angular::product_angular_grid;
///
/// # fn main() -> ruaa_dft_grids::Result<()> {
/// let (dirs, weights) = product_angular_grid(3)?;
/// assert_eq!(dirs.len(), 3 * 6);
/// assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-14);
/// # Ok(())
/// # }
/// ```
pub fn product_angular_grid(n_theta: usize) -> Result<AngularPoints> {
    if n_theta < 2 {
        return Err(Error::UnsupportedGrid {
            name: format!("product theta={n_theta}"),
            reason: "theta ring count must be at least 2".to_string(),
        });
    }
    let (xs, ws) = gauss_legendre(n_theta);
    let n_phi = 2 * n_theta;
    let mut dirs = Vec::with_capacity(n_theta * n_phi);
    let mut weights = Vec::with_capacity(n_theta * n_phi);
    for (&x, &w) in xs.iter().zip(&ws) {
        let theta = x.acos();
        let st = theta.sin();
        for l in 0..n_phi {
            let phi = 2.0 * PI * l as f64 / n_phi as f64;
            dirs.push((st * phi.cos(), st * phi.sin(), x));
            weights.push(w / (2.0 * n_phi as f64));
        }
    }
    Ok((dirs, weights))
}

/// Gauss–Legendre theta rings with an adaptive number of phi points per ring.
///
/// For each theta root, the ring carries the smallest `p >= 2` with
/// `sin(theta)^p <= 1e-8` (capped at `2 * n_theta`), and alternate rings are
/// staggered by half a phi step.  This is the variable-phi product rule used
/// by ORCA `AngularGrid 0`; it is exposed separately from the uniform
/// [`product_angular_grid`] because the two rules are not interchangeable.
///
/// # Errors
/// [`Error::UnsupportedGrid`] when `n_theta < 2`.
pub fn product_angular_grid_adaptive(n_theta: usize) -> Result<AngularPoints> {
    if n_theta < 2 {
        return Err(Error::UnsupportedGrid {
            name: format!("adaptive product theta={n_theta}"),
            reason: "theta ring count must be at least 2".to_string(),
        });
    }
    let (xs, ws) = gauss_legendre(n_theta);
    let mut dirs = Vec::new();
    let mut weights = Vec::new();
    for (i, (&x, &wgl)) in xs.iter().zip(&ws).enumerate() {
        let p = ring_phi_count(x, n_theta);
        let theta = x.acos();
        let st = theta.sin();
        let phi0 = if (n_theta - 1 - i).is_multiple_of(2) {
            PI / (p as f64)
        } else {
            0.0
        };
        for k in 0..p {
            let phi = 2.0 * PI * (k as f64) / (p as f64) + phi0;
            dirs.push((st * phi.cos(), st * phi.sin(), x));
            weights.push(wgl / (2.0 * p as f64));
        }
    }
    Ok((dirs, weights))
}

/// Number of phi points on the ring at `x = cos(theta)`: smallest `p >= 2`
/// with `sin(theta)^p <= 1e-8`, capped at `2 * n_theta`.
fn ring_phi_count(x: f64, n_theta: usize) -> usize {
    let s = (1.0 - x * x).max(0.0).sqrt();
    let mut p = 2usize;
    while p <= 2 * n_theta && s.powi(p as i32) > 1e-8 {
        p += 1;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::LEBEDEV_NGRID;

    /// Every rule sums to one, including the 12-point icosahedral rung and
    /// the plain Oh orbits (index 0 = 1 is the trivial unused grid).
    #[test]
    fn every_rule_sums_to_one() {
        for &n in &LEBEDEV_NGRID[1..] {
            let (_, w) = angular_grid(n).unwrap();
            let sum: f64 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-13, "n = {n}: weight sum {sum}");
        }
    }

    /// The 12-point rule: weights sum to 1, points lie on the unit sphere,
    /// and moments through degree 5 are exact.
    #[test]
    fn twelve_point_rule() {
        let (grid, weights) = angular_grid(12).unwrap();
        assert_eq!(grid.len(), 12);
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "sum {sum}");
        let mut x2 = 0.0;
        let mut x4 = 0.0;
        for (&(x, y, z), &w) in grid.iter().zip(&weights) {
            assert!((x * x + y * y + z * z - 1.0).abs() < 1e-13);
            x2 += w * x * x;
            x4 += w * x * x * x * x;
        }
        assert!((x2 - 1.0 / 3.0).abs() < 1e-12, "x^2 moment {x2}");
        assert!((x4 - 1.0 / 5.0).abs() < 1e-12, "x^4 moment {x4}");
        for i in 0..12 {
            for j in i + 1..12 {
                let (p, q) = (grid[i], grid[j]);
                assert!((p.0 - q.0).abs() + (p.1 - q.1).abs() + (p.2 - q.2).abs() > 1e-9);
            }
        }
    }

    /// The Oh-orbit point order is rstsr's (the $n=6$ orbit is the
    /// $\pm x, \pm y, \pm z$ sequence, not a re-sorted set).  This is the orientation check that
    /// matters for matching rstsr's `quads`.
    #[test]
    fn rstsr_orbit_order_is_preserved() {
        let (grid, _) = lebedev::angular_grid(6).unwrap();
        assert_eq!(
            grid,
            vec![
                (1.0, 0.0, 0.0),
                (-1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
            ]
        );
    }

    #[test]
    fn trivial_one_point_rule() {
        let (points, weights) = angular_grid(1).unwrap();
        assert_eq!(points, vec![(0.0, 0.0, 0.0)]);
        assert_eq!(weights, vec![1.0]);
        assert!(is_supported_size(1));
        assert!(angular_grid(0).is_err());
    }

    #[test]
    fn special_sizes_generate() {
        for n in [12usize, 18] {
            let (dirs, w) = angular_grid(n).unwrap();
            assert_eq!(dirs.len(), n, "size {n}");
            assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12, "size {n}");
            assert!(is_supported_size(n));
        }
        assert!(is_supported_size(974));
        assert!(!is_supported_size(72));
        assert!(angular_grid(72).is_err());
        assert!(angular_grid(1150).is_err());
    }

    #[test]
    fn octahedral_rule_is_exact_through_degree_3() {
        let (dirs, w) = octahedral_18();
        assert_eq!(dirs.len(), 18);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-15);
        let x2: f64 = dirs.iter().zip(&w).map(|(&(x, ..), &wi)| wi * x * x).sum();
        let x4: f64 = dirs
            .iter()
            .zip(&w)
            .map(|(&(x, ..), &wi)| wi * x * x * x * x)
            .sum();
        assert!((x2 - 1.0 / 3.0).abs() < 1e-15);
        assert!((x4 - 1.0 / 5.0).abs() < 1e-15);
    }

    #[test]
    #[allow(clippy::approx_constant)] // the literals are the pinned f64 values
    fn twelve_and_eighteen_points_are_pinned() {
        // These are construction pins, not an external direction reference:
        // they catch an accidental rotation, reordering or weight change.
        let (points, weights) = angular_grid(12).unwrap();
        assert_eq!(
            points,
            vec![
                (0.0, -0.5257311121191336, -0.85065080835204),
                (-0.85065080835204, 0.0, -0.5257311121191336),
                (-0.5257311121191336, -0.85065080835204, 0.0),
                (0.0, -0.5257311121191336, 0.85065080835204),
                (0.85065080835204, 0.0, -0.5257311121191336),
                (-0.5257311121191336, 0.85065080835204, 0.0),
                (0.0, 0.5257311121191336, -0.85065080835204),
                (-0.85065080835204, 0.0, 0.5257311121191336),
                (0.5257311121191336, -0.85065080835204, 0.0),
                (0.0, 0.5257311121191336, 0.85065080835204),
                (0.85065080835204, 0.0, 0.5257311121191336),
                (0.5257311121191336, 0.85065080835204, 0.0),
            ]
        );
        assert_eq!(weights, vec![1.0 / 12.0; 12]);

        let (points, weights) = octahedral_18();
        assert_eq!(
            points,
            vec![
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (-1.0, 0.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 0.0, -1.0),
                (0.0, 0.7071067811865476, 0.7071067811865476),
                (0.7071067811865476, 0.0, 0.7071067811865476),
                (0.7071067811865476, 0.7071067811865476, 0.0),
                (0.0, 0.7071067811865476, -0.7071067811865476),
                (0.7071067811865476, 0.0, -0.7071067811865476),
                (0.7071067811865476, -0.7071067811865476, 0.0),
                (0.0, -0.7071067811865476, 0.7071067811865476),
                (-0.7071067811865476, 0.0, 0.7071067811865476),
                (-0.7071067811865476, 0.7071067811865476, 0.0),
                (0.0, -0.7071067811865476, -0.7071067811865476),
                (-0.7071067811865476, 0.0, -0.7071067811865476),
                (-0.7071067811865476, -0.7071067811865476, 0.0),
            ]
        );
        assert_eq!(
            weights,
            vec![
                1.0 / 30.0,
                1.0 / 30.0,
                1.0 / 30.0,
                1.0 / 30.0,
                1.0 / 30.0,
                1.0 / 30.0,
                1.0 / 15.0,
                1.0 / 15.0,
                1.0 / 15.0,
                1.0 / 15.0,
                1.0 / 15.0,
                1.0 / 15.0,
                1.0 / 15.0,
                1.0 / 15.0,
                1.0 / 15.0,
                1.0 / 15.0,
                1.0 / 15.0,
                1.0 / 15.0,
            ]
        );
    }

    #[test]
    fn gauss_legendre_is_a_quadrature() {
        for n in 2..=20 {
            let (xs, ws) = gauss_legendre(n);
            assert_eq!(xs.len(), n);
            assert_eq!(ws.len(), n);
            assert!((ws.iter().sum::<f64>() - 2.0).abs() < 1e-12, "n = {n}");
            let m2: f64 = xs.iter().zip(&ws).map(|(&x, &w)| w * x * x).sum();
            assert!((m2 - 2.0 / 3.0).abs() < 1e-12, "n = {n}: x^2 {m2}");
        }
    }

    #[test]
    fn product_grid_totals() {
        let (dirs, w) = product_angular_grid(32).unwrap();
        assert_eq!(dirs.len(), 2 * 32 * 32);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        // S4 moment: sum w * (3 z^2 - 1) == 0 for the full sphere product rule
        let m: f64 = dirs
            .iter()
            .zip(&w)
            .map(|(d, wi)| wi * (3.0 * d.2 * d.2 - 1.0))
            .sum();
        assert!(m.abs() < 1e-12);
        assert!(product_angular_grid(1).is_err());
    }
}
