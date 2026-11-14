//! ORCA "mBecke (2022)" partition: classic Becke cell functions evaluated
//! with the pair products in nearest-atom-first order, plus mBecke's
//! early exit, clamp and nearest-atom shortcut.
//!
//! # Two evaluation paths
//!
//! * [`point_weights_all`] uses ORCA's nearest-atom-first order, the relative
//!   early exit, the clamp and the nearest-atom shortcut.  It is the reference
//!   path for the scalar `PartitionConfig::weight(s)` API and for a
//!   non-default clamp.
//! * The blocked production path for the default clamp (`mbecke_clamp ==
//!   1.0`, the only value [`PartitionConfig::new`] produces) uses the
//!   **classic pair product** in eight-point lanes
//!   ([`crate::partition::becke::block_no_early_exit`]).  The two are
//!   algebraically identical there: for the supported `p = 1.0` the clamp is
//!   inert because `|mu| <= 1` (triangle inequality) and `|a| <= 0.5` imply
//!   `|nu| <= 1`, `nu_ji = -nu_ij`
//!   and `a_BA = -a_AB`, so every factor
//!   is the same number; only the multiplication order and the last-bit
//!   asymmetry of the independently stored `a_AB`/`a_BA` differ, and the
//!   early exit truncates at the `1e-16` relative level.  The paths therefore
//!   agree to round-off.
//!
//! The analytic mBecke derivative uses the classic Becke backend, so value
//! and derivative are consistent.  A non-default `mbecke_clamp < 1.0`
//! activates the clamp/nearest-atom branches and keeps the nearest-first
//! kernel; it is not differentiable (see the [partition derivative
//! API](crate::doc::partition_derivatives)).

use super::becke::switch::{step_function, step3};
use super::simd::{self, V};
use super::{PartitionConfig, PartitionTables};
#[cfg(test)]
use super::{PartitionConfigBuilder, PartitionScheme};

/// ORCA "mBecke (2022)" weight-scheme clamp parameter.
///
/// ORCA exposes the clamp through the XC-grid keyword `%method GRIDA0 <real>`
/// and defaults to 0.99.  This crate keeps 1.0 for the differentiable
/// presets.  With the ORCA radii the clamp branches are inert at either
/// value (the size adjustment bounds `|nu|` below 0.99), so the weights are
/// numerically equivalent; the remaining 0.99-vs-1.0 difference is ORCA's
/// nearest-atom shortcut, whose threshold is `0.5 * (1 - p)`.  That shortcut
/// makes the literal 0.99 path non-differentiable for near-coincident
/// centres, which is why the analytic backend advertises 1.0 only.  A value
/// `< 1.0` is still honoured exactly by the nearest-first kernel.
///
/// Because the clamp is inert at 1.0, the blocked value path evaluates the
/// algebraically identical classic pair product (see the module docs); the
/// nearest-first path is kept by [`point_weights_all`].
pub const MBECKE_CLAMP: f64 = 1.0;

/// Relative early-exit threshold of the mBecke weight routine: skip the
/// remaining pair factors when the running sum times `1e-16` exceeds the
/// current factor.
pub const MBECKE_ACC_EPS: f64 = 1e-16;

/// "mBecke (2022)" cell functions of ALL centers at one grid point.
///
/// Matches the ORCA 6.1.1 reference grids.
///
/// Algorithm:
/// 1. distances `d_i`, sorted ascending (atoms processed nearest-first);
/// 2. `w_i = 1` initially; for each atom `i` in sorted order and each
///    `j` in sorted order: `mu = (d_i - d_j)/R_ij`,
///    `nu = mu + a_ij (1 - mu^2)`; `nu >= p` zeroes `w_i`, `nu <= -p`
///    skips the pair, else `w_i *= 0.5 (1 - f(nu))` with Becke's
///    hardness-3 polynomial (`f(mu) = 1.5 mu - 0.5 mu^3` iterated);
///    skip the whole `j` sweep when the running sum times `1e-16` exceeds
///    `w_i`;
/// 3. nearest-atom shortcut: if `d_A < 0.5 (1 - p) min_{B!=A} d_B` then
///    `w_A = 1`;
/// 4. normalize `w_A / sum(w)`.
///
/// Convenience wrapper that rebuilds the molecular tables on every call;
/// hot loops should build [`PartitionTables`] once and use
/// [`PartitionConfig::center_weights_par`].
#[cfg(test)]
#[must_use]
pub fn mbecke_partition_weights(
    hardness: usize,
    centers_bohr: &[(f64, f64, f64)],
    charges: &[u32],
    point_bohr: (f64, f64, f64),
) -> Vec<f64> {
    mbecke_partition_weights_clamped(hardness, centers_bohr, charges, point_bohr, MBECKE_CLAMP)
}

/// [`mbecke_partition_weights`] with an explicit clamp parameter `p`
/// (ORCA keyword `%method GRIDA0`; see [`MBECKE_CLAMP`]).
#[must_use]
#[cfg(test)]
pub fn mbecke_partition_weights_clamped(
    hardness: usize,
    centers_bohr: &[(f64, f64, f64)],
    charges: &[u32],
    point_bohr: (f64, f64, f64),
    clamp_p: f64,
) -> Vec<f64> {
    let centers: Vec<[f64; 3]> = centers_bohr.iter().map(|&(x, y, z)| [x, y, z]).collect();
    let cfg = PartitionConfigBuilder::default()
        .scheme(PartitionScheme::MBecke)
        .radii(crate::partition::test_support::custom_radii())
        .becke_early_exit(true)
        .hardness(hardness)
        .mbecke_clamp(clamp_p)
        .build()
        .unwrap();
    cfg.weights(
        [point_bohr.0, point_bohr.1, point_bohr.2],
        &centers,
        charges,
    )
    .unwrap()
}

/// Nearest-first mBecke weights of all centers at one point (Bohr).
///
/// `dist`, `order` and `prod` are per-center scratch of length `n`.
pub(crate) fn point_weights_all(
    cfg: &PartitionConfig,
    tables: &PartitionTables,
    point: [f64; 3],
    out: &mut [f64],
    dist: &mut [f64],
    order: &mut [usize],
    prod: &mut [f64],
) {
    let n = tables.len();
    if n == 0 {
        return;
    }
    for i in 0..n {
        dist[i] = super::dist3(point, tables.center(i));
        order[i] = i;
    }
    order.sort_by(|&i, &j| dist[i].total_cmp(&dist[j]));
    prod.fill(1.0);

    for pos in 0..n {
        let i = order[pos];
        let mut acc = 0.0;
        for &k in &order[..pos] {
            acc += prod[k];
        }
        for &j in order.iter() {
            if acc * MBECKE_ACC_EPS > prod[i] {
                break;
            }
            if j == i {
                continue;
            }
            let mu = (dist[i] - dist[j]) / tables.r_ab(i, j);
            let nu = mu + tables.a_ab(i, j) * (1.0 - mu * mu);
            if nu >= cfg.mbecke_clamp {
                prod[i] = 0.0;
                break;
            }
            if nu <= -cfg.mbecke_clamp {
                continue;
            }
            prod[i] *= 0.5 * (1.0 - step_function(nu, cfg.hardness));
        }
    }

    // nearest-atom shortcut: d_A < 0.5 (1-p) d_min.
    if cfg.mbecke_clamp < 1.0 {
        for i in 0..n {
            let dmin = (0..n)
                .filter(|&k| k != i)
                .map(|k| dist[k])
                .fold(f64::INFINITY, f64::min);
            if dist[i] < 0.5 * (1.0 - cfg.mbecke_clamp) * dmin {
                for (k, v) in prod.iter_mut().enumerate() {
                    *v = if k == i { 1.0 } else { 0.0 };
                }
            }
        }
    }

    let total: f64 = prod.iter().sum();
    if total.abs() > MBECKE_ACC_EPS {
        for v in prod.iter_mut() {
            *v /= total;
        }
    } else {
        prod.fill(0.0);
        prod[order[0]] = 1.0;
    }
    out.copy_from_slice(prod);
}

/// Per-worker scratch of the mBecke block kernel.
///
/// `lane` holds the eight-point distance lanes; `dist`/`order`/`prev`/`prod`
/// are the per-point working set of the nearest-first kernel.  `prev` is the
/// previous point's sorted order, used to seed an exact insertion sort (the
/// order changes little between adjacent points of one atom grid).
#[derive(Debug, Clone)]
pub(crate) struct Scratch {
    pub(crate) block: usize,
    pub(crate) lane: Vec<V>,
    pub(crate) dist: Vec<f64>,
    pub(crate) order: Vec<usize>,
    pub(crate) prev: Vec<usize>,
    pub(crate) prod: Vec<f64>,
    pub(crate) p: Vec<f64>,
    pub(crate) has_prev: bool,
}

impl Scratch {
    pub(crate) fn new(num_centers: usize, block: usize) -> Self {
        Self {
            block,
            lane: vec![V::zero(); num_centers],
            dist: vec![0.0; num_centers],
            order: (0..num_centers).collect(),
            prev: (0..num_centers).collect(),
            prod: vec![0.0; num_centers],
            p: vec![0.0; num_centers],
            has_prev: false,
        }
    }
}

/// Point-to-centre distance field of one lane of eight grid points.
///
/// Copied from the SSF sparse kernel; the arithmetic per lane is the same
/// expression as [`crate::partition::dist3`], so the result is bitwise equal
/// to the scalar reference.
#[inline]
fn fill_lane(tables: &PartitionTables, coords: [V; 3], lane: &mut [V]) {
    let n = tables.len();
    for (k, slot) in lane.iter_mut().take(n).enumerate() {
        let c = tables.center(k);
        let dx = coords[0] - simd::splat(c[0]);
        let dy = coords[1] - simd::splat(c[1]);
        let dz = coords[2] - simd::splat(c[2]);
        *slot = (dx * dx + dy * dy + dz * dz).sqrt();
    }
}

/// Transpose one lane of distances into `dist`.
#[inline]
fn scatter_lane(lane: &[V], l: usize, dist: &mut [f64]) {
    for (d, v) in dist.iter_mut().zip(lane) {
        *d = v[l];
    }
}

/// Whether the classic pair product is a valid evaluation of the mBecke
/// cell functions for this molecule.
///
/// The two schemes are algebraically identical only when ORCA's clamp cannot
/// fire away from the boundary:
///
/// * the clamp and the nearest-atom shortcut must be inert
///   (`mbecke_clamp >= 1.0`; `PartitionConfig::new` always uses `1.0` and the
///   shortcut branch is only entered below `1.0`);
/// * `nu(mu) = mu + a_AB (1 - mu^2)` must satisfy `|nu| <= 1` for every
///   `mu in [-1, 1]`.  For `|a| <= 1/2` the extremum is `1` at `|mu| = 1`;
///   for `a > 1/2` it is `a + 1/(4a) > 1` at `mu = 1/(2a)`, where ORCA zeroes
///   the product and the classic factor does not.  [`PartitionTables`]
///   contains `max|a_AB|`, and [`super::RadiiAdjust::factor`] clamps every
///   factor to `[-0.5, 0.5]`, so this is a guard rather than a restriction.
#[inline]
pub(crate) fn classic_product_is_valid(cfg: &PartitionConfig, tables: &PartitionTables) -> bool {
    cfg.mbecke_clamp >= 1.0 && tables.max_abs_a() <= 0.5
}

/// mBecke weights of `center` for one block of grid points.
///
/// # Default clamp: classic pair product
///
/// With ORCA's default clamp (`mbecke_clamp == 1.0`, the only value reachable
/// from [`PartitionConfig::new`]) the mBecke product is **mathematically
/// identical** to the classic Becke product with the ORCA radii: for every
/// pair, `s_ij = 1/2 (1 - f(nu_ij))` and `s_ji = 1/2 (1 + f(nu_ij)) = 1/2
/// (1 - f(-nu_ij))`.  ORCA evaluates each atom's row nearest-atom-first with
/// its early exit; the classic lane kernel evaluates each unordered pair once
/// and gives the partner the complementary factor.  The two differ only in
/// the floating-point order (and in `a_AB` vs `a_BA`, whose stored values
/// need not be exact negatives), at the `~1e-16` relative level.  The fast
/// path is therefore the classic eight-point lane product
/// ([`crate::partition::becke::block_no_early_exit`]), which is what the
/// analytic mBecke derivative is the derivative of (see the module docs).
///
/// # Non-default clamp: exact ORCA order
///
/// A clamp `< 1.0` activates ORCA's clamp branches and the nearest-atom
/// shortcut, which the classic product does not reproduce (and which is not
/// differentiable).  That case keeps the exact nearest-first kernel, which is
/// **bitwise identical** to [`point_weights_all`].  The kernel is evaluated
/// as fixed eight-point distance lanes; the sorted order is the same, the
/// running sum `acc` is accumulated incrementally instead of rebuilt from
/// scratch (the same left-to-right sum), the `j == i` skip is folded into two
/// loops around the known position, and hardness 3 is the unrolled
/// polynomial.
pub(crate) fn block_weights(
    cfg: &PartitionConfig,
    tables: &PartitionTables,
    center: usize,
    grid: &[[f64; 3]],
    out: &mut [f64],
    scratch: &mut Scratch,
) {
    if cfg.hardness == 3 {
        block_impl(cfg, tables, center, grid, out, scratch, step3);
    } else {
        let hardness = cfg.hardness;
        block_impl(cfg, tables, center, grid, out, scratch, |nu| {
            step_function(nu, hardness)
        });
    }
}

#[inline]
fn block_impl<F: Fn(f64) -> f64>(
    cfg: &PartitionConfig,
    tables: &PartitionTables,
    center: usize,
    grid: &[[f64; 3]],
    out: &mut [f64],
    scratch: &mut Scratch,
    step: F,
) {
    let n = tables.len();
    let nb = grid.len();
    let nlane = nb.div_ceil(simd::LANES);
    debug_assert!(nb <= scratch.block);
    let Scratch {
        lane,
        dist,
        order,
        prev,
        prod,
        p,
        has_prev,
        ..
    } = scratch;
    for l_lane in 0..nlane {
        let coords = simd::gather_coords(grid, l_lane);
        fill_lane(tables, coords, &mut lane[..n]);
        let len = simd::lane_len(nb, l_lane);
        for l in 0..len {
            scatter_lane(&lane[..n], l, &mut dist[..n]);
            sort_order(&dist[..n], &mut order[..n], &mut prev[..n], *has_prev);
            *has_prev = true;
            out[l_lane * simd::LANES + l] = point_weight(
                cfg,
                tables,
                center,
                &dist[..n],
                &order[..n],
                &mut prod[..n],
                &mut p[..n],
                &step,
            );
        }
    }
}

/// Sort `order` by (distance ascending, original index ascending).
///
/// The first point of a worker starts from the identity permutation and a
/// full unstable sort.  Later points start from the previous point's order
/// and use an exact insertion sort; if that needs more than `4 n + 16` moves
/// (a different part of the grid), the partial result is discarded and the
/// full sort is used instead.  Both paths produce the unique total order, so
/// they are bitwise interchangeable.
#[inline]
fn sort_order(dist: &[f64], order: &mut [usize], prev: &mut [usize], has_prev: bool) {
    let n = order.len();
    if !has_prev {
        order.copy_from_slice(prev);
        full_sort(dist, order);
        prev.copy_from_slice(order);
        return;
    }
    order.copy_from_slice(prev);
    let mut budget = 4 * n + 16;
    let mut ok = true;
    for i in 1..n {
        let x = order[i];
        let dx = dist[x];
        let mut j = i;
        while j > 0 {
            let y = order[j - 1];
            let dy = dist[y];
            let less = dx < dy || (dx == dy && x < y);
            if !less {
                break;
            }
            order[j] = y;
            j -= 1;
            budget -= 1;
            if budget == 0 {
                ok = false;
                break;
            }
        }
        if !ok {
            break;
        }
        order[j] = x;
    }
    if !ok {
        full_sort(dist, order);
    }
    prev.copy_from_slice(order);
}

#[inline]
fn full_sort(dist: &[f64], order: &mut [usize]) {
    order.sort_unstable_by(|&i, &j| dist[i].total_cmp(&dist[j]).then_with(|| i.cmp(&j)));
}

/// mBecke weight of `center` at one point, given the sorted `order`.
#[allow(clippy::too_many_arguments)]
#[inline]
fn point_weight<F: Fn(f64) -> f64>(
    cfg: &PartitionConfig,
    tables: &PartitionTables,
    center: usize,
    dist: &[f64],
    order: &[usize],
    prod: &mut [f64],
    p: &mut [f64],
    step: &F,
) -> f64 {
    let n = order.len();
    let clamp = cfg.mbecke_clamp;
    prod.fill(1.0);
    // Incremental `acc`: the reference rebuilds `sum_{k < pos} prod[order[k]]`
    // at every position; the products of earlier positions never change, so
    // the running left-to-right sum is the same value.
    let mut acc = 0.0;
    for pos in 0..n {
        let i = order[pos];
        let di = dist[i];
        let stop = acc * MBECKE_ACC_EPS;
        let mut pi = 1.0;
        let mut hit_zero = false;
        for &j in &order[..pos] {
            if stop > pi {
                break;
            }
            let mu = (di - dist[j]) / tables.r_ab(i, j);
            let nu = mu + tables.a_ab(i, j) * (1.0 - mu * mu);
            if nu >= clamp {
                hit_zero = true;
                break;
            }
            if nu <= -clamp {
                continue;
            }
            pi *= 0.5 * (1.0 - step(nu));
        }
        if !hit_zero && stop <= pi {
            for &j in &order[pos + 1..] {
                if stop > pi {
                    break;
                }
                let mu = (di - dist[j]) / tables.r_ab(i, j);
                let nu = mu + tables.a_ab(i, j) * (1.0 - mu * mu);
                if nu >= clamp {
                    hit_zero = true;
                    break;
                }
                if nu <= -clamp {
                    continue;
                }
                pi *= 0.5 * (1.0 - step(nu));
            }
        }
        let value = if hit_zero { 0.0 } else { pi };
        prod[i] = value;
        acc += value;
    }

    // nearest-atom shortcut: d_A < 0.5 (1-p) d_min (inert at the default
    // clamp; kept exactly as in the reference).
    if clamp < 1.0 {
        for i in 0..n {
            let dmin = (0..n)
                .filter(|&k| k != i)
                .map(|k| dist[k])
                .fold(f64::INFINITY, f64::min);
            if dist[i] < 0.5 * (1.0 - clamp) * dmin {
                for (k, v) in prod.iter_mut().enumerate() {
                    *v = if k == i { 1.0 } else { 0.0 };
                }
            }
        }
    }

    let total: f64 = prod.iter().sum();
    if total.abs() > MBECKE_ACC_EPS {
        p.copy_from_slice(prod);
        for v in p.iter_mut() {
            *v /= total;
        }
    } else {
        p.fill(0.0);
        p[order[0]] = 1.0;
    }
    p[center]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default-clamp fast path is the classic pair product; it must agree
    /// with ORCA's exact nearest-first kernel to round-off.  This measures and
    /// pins the deviation on a real (multi-shell) grid so a future change that
    /// silently swaps in a truncating kernel is caught.
    #[test]
    fn fast_path_matches_orca_order_to_roundoff() {
        let cfg = PartitionConfigBuilder::default()
            .scheme(PartitionScheme::MBecke)
            .radii(crate::partition::test_support::custom_radii())
            .becke_early_exit(true)
            .build()
            .unwrap();
        assert_eq!(cfg.mbecke_clamp, 1.0);
        // A ring of 12 atoms with mixed radii: enough shells for the clamp,
        // the shortcut and the early exit to be exercised.
        let mut centers = Vec::new();
        for k in 0..12 {
            let t = std::f64::consts::TAU * k as f64 / 12.0;
            centers.push([2.6 * t.cos(), 2.6 * t.sin(), 0.9 * (k as f64).sin()]);
        }
        let charges = [6u32, 6, 8, 8, 1, 1, 7, 1, 1, 6, 8, 1];
        let grid: Vec<[f64; 3]> = (0..2049)
            .map(|k| {
                let t = k as f64 * 0.011;
                [
                    6.5 * t.cos() + 0.5 * t.sin(),
                    5.5 * t.sin() - 0.3,
                    4.0 * t.cos() + 0.2 * t,
                ]
            })
            .collect();
        let tables = cfg.tables(&centers, &charges).unwrap();
        let mut max_abs = 0.0_f64;
        for center in 0..tables.len() {
            // The production blocked/parallel path (classic pair product for
            // the default clamp).
            let mut fast = vec![0.0; grid.len()];
            cfg.center_weights(&tables, center, &grid, &mut fast);
            for (p, &point) in grid.iter().enumerate() {
                let mut want = vec![0.0; tables.len()];
                let mut dist = vec![0.0; tables.len()];
                let mut order = vec![0usize; tables.len()];
                let mut prod = vec![0.0; tables.len()];
                point_weights_all(
                    &cfg, &tables, point, &mut want, &mut dist, &mut order, &mut prod,
                );
                max_abs = max_abs.max((fast[p] - want[center]).abs());
            }
        }
        assert!(
            max_abs < 1e-15,
            "mBecke fast path deviates from the ORCA order by {max_abs:e}"
        );
        eprintln!("mBecke fast-path max |dw| vs ORCA order: {max_abs:e}");
    }

    type Case = (Vec<(f64, f64, f64)>, Vec<u32>);

    #[test]
    fn mbecke_weights_sum_to_one() {
        let cases: Vec<Case> = vec![
            (vec![(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)], vec![1, 9]),
            (vec![(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)], vec![6, 8]),
            (
                vec![(0.0, 0.0, 0.0), (0.0, 0.0, 2.5), (1.5, 1.0, -1.0)],
                vec![1, 6, 17],
            ),
        ];
        for (centers, charges) in &cases {
            for &x in &[-2.0, -0.99, -0.5, 0.0, 0.35, 0.99, 2.5] {
                let pt = (x, 0.2 * (x + 1.0), -0.1);
                let w = mbecke_partition_weights(3, centers, charges, pt);
                let total: f64 = w.iter().sum();
                assert!(
                    (total - 1.0).abs() < 1e-10,
                    "mBecke sum = {total} at {pt:?}"
                );
                assert!(w.iter().all(|&v| v.is_finite() && v >= 0.0));
            }
        }
    }

    #[test]
    fn mbecke_homonuclear_midpoint_symmetry() {
        let centers = [(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)];
        let charges = [6u32, 6];
        for &x in &[-0.7, -0.2, 0.0, 0.3, 0.8] {
            let wa = mbecke_partition_weights(3, &centers, &charges, (x, 0.1, -0.05))[0];
            let wb = mbecke_partition_weights(3, &centers, &charges, (-x, 0.1, -0.05))[1];
            assert!((wa - wb).abs() < 1e-12, "asymmetry at x={x}: {wa} vs {wb}");
        }
        let w = mbecke_partition_weights(3, &centers, &charges, (0.0, 0.0, 0.0));
        assert!((w[0] - 0.5).abs() < 1e-12);
        assert!((w[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn mbecke_monotonicity() {
        let centers = [(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)];
        let charges = [1u32, 9];
        let mut prev = 1.0;
        for i in 1..40 {
            let x = -1.0 + 2.0 * (i as f64) / 40.0;
            let w = mbecke_partition_weights(3, &centers, &charges, (x, 0.0, 0.0))[0];
            assert!(w <= prev + 1e-12, "non-monotone at x={x}: {w} > {prev}");
            prev = w;
        }
    }

    #[test]
    fn mbecke_matches_becke_for_default_clamp() {
        let cases: Vec<Case> = vec![
            (vec![(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)], vec![1, 9]),
            (vec![(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)], vec![6, 8]),
            (
                vec![(0.0, 0.0, 0.0), (0.0, 0.0, 2.5), (1.5, 1.0, -1.0)],
                vec![1, 6, 17],
            ),
        ];
        for (centers, charges) in &cases {
            for &x in &[-2.0, -0.7, 0.0, 0.35, 1.0, 2.5] {
                let pt = (x, 0.2 * (x + 1.0), -0.1);
                let centers3: Vec<[f64; 3]> = centers.iter().map(|&(x, y, z)| [x, y, z]).collect();
                let becke = PartitionConfigBuilder::default()
                    .scheme(PartitionScheme::Becke)
                    .radii(crate::partition::test_support::custom_radii())
                    .becke_early_exit(true)
                    .build()
                    .unwrap();
                let wm = mbecke_partition_weights(3, centers, charges, pt);
                for (a, &w) in wm.iter().enumerate() {
                    let wb = becke
                        .weight(a, [pt.0, pt.1, pt.2], &centers3, charges)
                        .unwrap();
                    assert!(
                        (w - wb).abs() < 1e-12,
                        "mBecke {w} vs Becke {wb} at {pt:?} center {a}"
                    );
                }
            }
        }
    }

    #[test]
    fn mbecke_clamped_partition_sharpens() {
        let centers = [(-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)];
        let charges = [1u32, 9];
        for &x in &[-0.9, 0.9] {
            let pt = (x, 0.0, 0.0);
            let w = mbecke_partition_weights_clamped(3, &centers, &charges, pt, 0.5);
            let total: f64 = w.iter().sum();
            assert!((total - 1.0).abs() < 1e-10);
            let wref = mbecke_partition_weights_clamped(3, &centers, &charges, pt, 1.0);
            assert!(
                (w[0] - wref[0]).abs() > 1e-9 || (w[1] - wref[1]).abs() > 1e-9,
                "clamp p=0.5 had no effect at {pt:?}"
            );
        }
    }
}
