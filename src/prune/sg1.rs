//! PySCF `gen_grid.sg1_prune`: SG1 region pruning, H..Ar only.

use super::{SG1_LADDER, place_index};
use crate::error::{Error, Result};
use crate::params::SG1_RADII;

/// PySCF `gen_grid.sg1_prune` (H..Ar only).
///
/// # Errors
/// [`Error::MissingElementData`] for an element beyond Ar (the SG1 radii
/// table only covers H..Ar).
pub fn sg1_prune(z: u32, rads: &[f64]) -> Result<Vec<usize>> {
    let r_atom = SG1_RADII
        .get(z as usize)
        .copied()
        .ok_or(Error::MissingElementData {
            z,
            table: "PySCF SG1",
        })?;
    let places = place_index(rads, r_atom, z);
    Ok(places.into_iter().map(|p| SG1_LADDER[p]).collect())
}
