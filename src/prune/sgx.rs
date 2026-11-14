//! PySCF `gen_grid.sgx_prune`: "ORCA-like" pruning for SGX grids.

use super::place_index;
use crate::error::{Error, Result};
use crate::radii::RadiiTable;

/// Lebedev point counts per SGX level and region (PySCF `SGX_ANG_MAPPING`
/// orders 5, 7, 11, 17, 23, 29, 35, 41 and 47 expanded to point counts).
const SGX_POINTS: [[usize; 5]; 8] = [
    [14, 26, 50, 50, 26],
    [14, 26, 50, 110, 50],
    [26, 50, 110, 194, 110],
    [26, 110, 194, 302, 194],
    [26, 194, 302, 434, 302],
    [50, 302, 434, 590, 434],
    [110, 434, 590, 770, 590],
    [770, 770, 770, 770, 770],
];

/// PySCF `gen_grid.sgx_prune` ("ORCA-like" pruning for SGX grids).
///
/// # Errors
/// [`Error::MissingElementData`] when the PySCF Bragg radii table has no
/// entry for `z`, and [`Error::UnsupportedLebedevGrid`] when `n_ang` exceeds
/// the largest SGX ladder level (770 points).
pub fn sgx_prune(z: u32, rads: &[f64], n_ang: usize) -> Result<Vec<usize>> {
    let lvls: [usize; 8] = SGX_POINTS.map(|row| row[3]);
    let level = lvls.iter().filter(|&&l| n_ang > l).count();
    if level >= SGX_POINTS.len() {
        return Err(Error::UnsupportedLebedevGrid { points: n_ang });
    }
    let leb_l = SGX_POINTS[level];
    let r_atom = RadiiTable::Bragg.bohr(z)?;
    let places = place_index(rads, r_atom, z);
    Ok(places.into_iter().map(|p| leb_l[p]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The SGX ladder stops at 770 points: larger base angular grids are a
    /// typed error, not an out-of-bounds index.
    #[test]
    fn n_ang_above_the_ladder_is_an_error() {
        let rads: Vec<f64> = (1..=20).map(|i| i as f64 * 0.1).collect();
        assert!(sgx_prune(6, &rads, 770).is_ok());
        for &n_ang in &[771, 974, 5810] {
            assert_eq!(
                sgx_prune(6, &rads, n_ang),
                Err(Error::UnsupportedLebedevGrid { points: n_ang })
            );
        }
    }
}
