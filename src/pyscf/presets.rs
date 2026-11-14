//! PySCF `Grids.level` presets: per-level radial counts and Lebedev
//! orders (`pyscf.dft.gen_grid`).

use crate::error::{Error, Result};
use crate::params::period_index;

/// Resolve a PySCF angular-size argument to a Lebedev point count.
///
/// PySCF's `gen_atomic_grids` accepts either a supported Lebedev point count
/// (its `LEBEDEV_NGRID` table) or one of the `LEBEDEV_ORDER` keys, and maps an
/// angular size of 0 or 1 to the trivial one-point rule.  A supported point
/// count is returned unchanged.  The 12- and 18-point rules are crate
/// extensions; PySCF itself accepts only its Lebedev ladder here.
///
/// PySCF 2.15's `MakeAngularGrid(1)` returns a one-dimensional array, so its
/// atomic-grid path trips on angular sizes 0 and 1 before reaching this rule;
/// the crate represents the one-point rule directly and does not use that
/// path as a reference.
///
/// # Errors
/// [`Error::UnsupportedLebedevGrid`] for a value that is neither a supported
/// count nor a Lebedev order key.
///
/// ```
/// use ruaa_dft_grids::pyscf::angular_count;
///
/// # fn main() -> ruaa_dft_grids::Result<()> {
/// assert_eq!(angular_count(11)?, 50); // PySCF order 11 -> 50 points
/// assert_eq!(angular_count(50)?, 50); // already a point count
/// assert_eq!(angular_count(0)?, 1);   // PySCF's trivial rule
/// # Ok(())
/// # }
/// ```
pub fn angular_count(n_ang: usize) -> Result<usize> {
    if n_ang <= 1 {
        return Ok(1);
    }
    if crate::angular::is_supported_size(n_ang) {
        return Ok(n_ang);
    }
    let points = u8::try_from(n_ang)
        .ok()
        .and_then(|order| lebedev_points_for_order(order).ok());
    match points {
        Some(points) if crate::angular::is_supported_size(points) => Ok(points),
        _ => Err(Error::UnsupportedLebedevGrid { points: n_ang }),
    }
}

/// PySCF `LEBEDEV_ORDER` mapping (order -> number of points).
///
/// # Errors
/// [`Error::UnsupportedLebedevOrder`] for an order not in the PySCF table.
pub fn lebedev_points_for_order(order: u8) -> Result<usize> {
    Ok(match order {
        0 => 1,
        3 => 6,
        5 => 14,
        7 => 26,
        9 => 38,
        11 => 50,
        13 => 74,
        15 => 86,
        17 => 110,
        19 => 146,
        21 => 170,
        23 => 194,
        25 => 230,
        27 => 266,
        29 => 302,
        31 => 350,
        35 => 434,
        41 => 590,
        47 => 770,
        53 => 974,
        59 => 1202,
        65 => 1454,
        71 => 1730,
        77 => 2030,
        83 => 2354,
        89 => 2702,
        95 => 3074,
        101 => 3470,
        107 => 3890,
        113 => 4334,
        119 => 4802,
        125 => 5294,
        131 => 5810,
        _ => return Err(Error::UnsupportedLebedevOrder { order }),
    })
}

/// PySCF `(n_rad, n_ang)` for `z` at `level` (`_default_rad`/`_default_ang`).
///
/// # Errors
/// [`Error::UnsupportedLevel`] for `level > 9` and
/// [`Error::UnsupportedElement`] for Z outside `1..=118`.
///
/// ```
/// use ruaa_dft_grids::pyscf::level_defaults;
///
/// # fn main() -> ruaa_dft_grids::Result<()> {
/// assert_eq!(level_defaults(6, 3)?, (75, 302));  // carbon at level 3
/// assert_eq!(level_defaults(1, 0)?, (10, 50));
/// # Ok(())
/// # }
/// ```
pub fn level_defaults(z: u32, level: u8) -> Result<(usize, usize)> {
    let level_idx = usize::from(level);
    if level_idx >= 10 {
        return Err(Error::UnsupportedLevel { level });
    }
    let p = period_index(z).ok_or(Error::UnsupportedElement { z })?;
    let n_rad = PYSCF_RAD_GRIDS[level_idx][p];
    let n_ang = lebedev_points_for_order(PYSCF_ANG_ORDER[level_idx][p])?;
    Ok((n_rad, n_ang))
}

use crate::params::{PYSCF_ANG_ORDER, PYSCF_RAD_GRIDS};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_defaults_match_pyscf() {
        // _default_rad / _default_ang at level 3
        assert_eq!(level_defaults(1, 3), Ok((50, 302)));
        assert_eq!(level_defaults(6, 3), Ok((75, 302)));
        assert_eq!(level_defaults(18, 3), Ok((80, 434)));
        assert_eq!(level_defaults(26, 3), Ok((90, 434)));
        assert_eq!(level_defaults(118, 3), Ok((105, 434)));
        // level 0
        assert_eq!(level_defaults(1, 0), Ok((10, 50)));
        assert_eq!(
            level_defaults(6, 10),
            Err(Error::UnsupportedLevel { level: 10 })
        );
        assert_eq!(
            level_defaults(119, 3),
            Err(Error::UnsupportedElement { z: 119 })
        );
    }

    #[test]
    fn angular_count_follows_pyscf_aliases() {
        assert_eq!(angular_count(0), Ok(1));
        assert_eq!(angular_count(1), Ok(1));
        for (order, points) in [
            (3, 6),
            (5, 14),
            (9, 38),
            (11, 50),
            (15, 86),
            (29, 302),
            (65, 1454),
            (131, 5810),
        ] {
            assert_eq!(angular_count(order), Ok(points), "order {order}");
        }
        // Point counts pass through unchanged, including the 12/18 crate
        // extensions.
        for points in [6, 12, 18, 50, 302, 434, 974, 5810] {
            assert_eq!(angular_count(points), Ok(points), "count {points}");
        }
        for points in [2, 4, 72, 100, 200] {
            assert_eq!(
                angular_count(points),
                Err(Error::UnsupportedLebedevGrid { points })
            );
        }
    }
}
