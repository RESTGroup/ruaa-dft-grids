//! Molecule value type shared by the grid, partition and derivative APIs.
//!
//! Coordinates are in Bohr. The type stores charges and positions in
//! structure-of-arrays form so the evaluators can borrow contiguous centers
//! and charges without building temporary vectors.

use std::borrow::Cow;

use crate::BOHR;
use crate::error::{Error, Result};

/// One atom: nuclear charge and position in Bohr.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Atom {
    /// Nuclear charge (atomic number), `1..=118`.
    pub z: u32,
    /// Cartesian position in Bohr.
    pub position: [f64; 3],
}

/// A molecule: charges and nuclear positions in Bohr.
///
/// Construct with [`Molecule::new`] or [`Molecule::from_angstrom`]; both
/// validate the charges and coordinates. The empty molecule is allowed so
/// generic code can build an empty grid, but grid constructors may reject it
/// through their own policies.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Molecule {
    centers: Vec<[f64; 3]>,
    charges: Vec<u32>,
}

impl Molecule {
    /// Build a molecule from `(Z, position in Bohr)` pairs.
    ///
    /// # Errors
    /// [`Error::UnsupportedElement`] when a charge is outside `1..=118`, and
    /// [`Error::NonFiniteCoordinate`] when a position component is not
    /// finite.
    pub fn new(atoms: impl IntoIterator<Item = (u32, [f64; 3])>) -> Result<Self> {
        let atoms: Vec<(u32, [f64; 3])> = atoms.into_iter().collect();
        let mut centers = Vec::with_capacity(atoms.len());
        let mut charges = Vec::with_capacity(atoms.len());
        for (ia, (z, position)) in atoms.into_iter().enumerate() {
            validate_atom(ia, z, position)?;
            charges.push(z);
            centers.push(position);
        }
        Ok(Self { centers, charges })
    }

    /// Build a molecule from `(Z, position in Ångström)` pairs.
    ///
    /// # Errors
    /// Same as [`Self::new`].
    pub fn from_angstrom(atoms: impl IntoIterator<Item = (u32, [f64; 3])>) -> Result<Self> {
        let scale = 1.0 / BOHR;
        Self::new(
            atoms
                .into_iter()
                .map(|(z, p)| (z, [p[0] * scale, p[1] * scale, p[2] * scale])),
        )
    }

    /// Build from a borrowed slice of `(Z, position in Bohr)` pairs.
    ///
    /// # Errors
    /// Same as [`Self::new`].
    pub fn from_slice(atoms: &[(u32, [f64; 3])]) -> Result<Self> {
        Self::new(atoms.iter().copied())
    }

    /// Nuclear charges, one per atom.
    #[must_use]
    pub fn charges(&self) -> &[u32] {
        &self.charges
    }

    /// Nuclear positions in Bohr, one per atom.
    #[must_use]
    pub fn centers(&self) -> &[[f64; 3]] {
        &self.centers
    }

    /// Atoms in order.
    pub fn atoms(&self) -> impl ExactSizeIterator<Item = Atom> + '_ {
        self.charges
            .iter()
            .copied()
            .zip(self.centers.iter().copied())
            .map(|(z, position)| Atom { z, position })
    }

    /// Number of atoms.
    #[must_use]
    pub fn len(&self) -> usize {
        self.charges.len()
    }

    /// Whether the molecule has no atoms.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.charges.is_empty()
    }

    /// Validate every charge and coordinate.
    ///
    /// This is cheap (`O(natom)`) and does not check for coincident centres;
    /// the partition evaluator rejects those because their pair distances are
    /// undefined.
    ///
    /// # Errors
    /// [`Error::UnsupportedElement`] or [`Error::NonFiniteCoordinate`].
    pub fn validate(&self) -> Result<()> {
        for (ia, (&z, &position)) in self.charges.iter().zip(&self.centers).enumerate() {
            validate_atom(ia, z, position)?;
        }
        Ok(())
    }
}

impl TryFrom<&[(u32, [f64; 3])]> for Molecule {
    type Error = Error;

    fn try_from(atoms: &[(u32, [f64; 3])]) -> Result<Self> {
        Self::from_slice(atoms)
    }
}

impl TryFrom<Vec<(u32, [f64; 3])>> for Molecule {
    type Error = Error;

    fn try_from(atoms: Vec<(u32, [f64; 3])>) -> Result<Self> {
        Self::new(atoms)
    }
}

fn validate_atom(atom: usize, z: u32, position: [f64; 3]) -> Result<()> {
    if !(1..=118).contains(&z) {
        return Err(Error::UnsupportedElement { z });
    }
    if position.iter().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteCoordinate { atom });
    }
    Ok(())
}

/// Argument accepted by the molecule-taking builders.
///
/// Implemented for [`Molecule`] references and for `(Z, position)` slices so
/// existing code can keep passing a geometry array while the main API uses
/// the validated value type.
pub trait AsMolecule<'a> {
    /// Borrow or build a validated molecule.
    ///
    /// # Errors
    /// The same validation errors as [`Molecule::new`].
    fn into_molecule(self) -> Result<Cow<'a, Molecule>>;
}

impl<'a> AsMolecule<'a> for &'a Molecule {
    fn into_molecule(self) -> Result<Cow<'a, Molecule>> {
        self.validate()?;
        Ok(Cow::Borrowed(self))
    }
}

impl<'a> AsMolecule<'a> for &'a [(u32, [f64; 3])] {
    fn into_molecule(self) -> Result<Cow<'a, Molecule>> {
        Ok(Cow::Owned(Molecule::from_slice(self)?))
    }
}

impl<'a, const N: usize> AsMolecule<'a> for &'a [(u32, [f64; 3]); N] {
    fn into_molecule(self) -> Result<Cow<'a, Molecule>> {
        Ok(Cow::Owned(Molecule::from_slice(self)?))
    }
}

impl<'a> AsMolecule<'a> for &'a Vec<(u32, [f64; 3])> {
    fn into_molecule(self) -> Result<Cow<'a, Molecule>> {
        Ok(Cow::Owned(Molecule::from_slice(self)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_and_iterates() {
        let molecule = Molecule::new([(8, [0.0, 0.0, 0.0]), (1, [0.0, 0.0, 1.8])]).unwrap();
        assert_eq!(molecule.len(), 2);
        assert_eq!(molecule.charges(), [8, 1]);
        assert_eq!(molecule.centers()[1], [0.0, 0.0, 1.8]);
        let atoms: Vec<_> = molecule.atoms().collect();
        assert_eq!(
            atoms,
            [
                Atom {
                    z: 8,
                    position: [0.0, 0.0, 0.0]
                },
                Atom {
                    z: 1,
                    position: [0.0, 0.0, 1.8]
                }
            ]
        );
    }

    #[test]
    fn rejects_bad_charge_and_nonfinite_coordinate() {
        assert_eq!(
            Molecule::new([(0, [0.0; 3])]).unwrap_err(),
            Error::UnsupportedElement { z: 0 }
        );
        assert_eq!(
            Molecule::new([(1, [f64::NAN, 0.0, 0.0])]).unwrap_err(),
            Error::NonFiniteCoordinate { atom: 0 }
        );
    }

    #[test]
    fn angstrom_conversion_round_trips() {
        let molecule = Molecule::from_angstrom([(1, [0.0, 0.0, 1.0])]).unwrap();
        assert!((molecule.centers()[0][2] - 1.0 / BOHR).abs() < 1e-15);
    }
}
