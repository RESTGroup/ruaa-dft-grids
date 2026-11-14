//! PySCF `gen_grid.nwchem_prune`: region-based pruning against the PySCF
//! Bragg radius with the NWChem alpha boundaries.

use super::place_index;
use crate::error::{Error, Result};
use crate::params::PYSCF_LEBEDEV_NGRID;
use crate::radii::RadiiTable;

/// PySCF `gen_grid.nwchem_prune`.
///
/// # Errors
/// [`Error::UnsupportedLebedevGrid`] for an `n_ang` (>= 50) that is not in
/// PySCF's NWChem ladder, and [`Error::MissingElementData`] when the Bragg
/// radii table has no entry for `z`.
pub fn nwchem_prune(z: u32, rads: &[f64], n_ang: usize) -> Result<Vec<usize>> {
    if n_ang < 50 {
        return Ok(vec![n_ang; rads.len()]);
    }
    let leb_ngrid = &PYSCF_LEBEDEV_NGRID[4..];
    let leb_l: [usize; 5] = if n_ang == 50 {
        [1, 2, 2, 2, 1]
    } else {
        let idx = leb_ngrid
            .iter()
            .position(|&n| n == n_ang)
            .ok_or(Error::UnsupportedLebedevGrid { points: n_ang })?;
        [1, 3, idx - 1, idx, idx - 1]
    };
    let r_atom = RadiiTable::Bragg.bohr(z)?;
    let places = place_index(rads, r_atom, z);
    Ok(places.into_iter().map(|p| leb_ngrid[leb_l[p]]).collect())
}
