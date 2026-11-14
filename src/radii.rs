//! Atomic radii tables used by the Becke size adjustment and by the PySCF
//! pruning schemes.
//!
//! Tables are plain `[f64]` indexed by Z and served uniformly through
//! [`RadiiTable::bohr`]:
//!
//! * `Bragg`/`Covalent`/`Vdw`/`Sg1` — PySCF 2.14/2.15's `pyscf.data.radii`
//!   tables (in [`crate::params`]), in Bohr.
//! * `Custom` — a caller-provided table.  This is the path for a partition
//!   that needs a program-specific radius set: supply the values in Bohr.
//!
//! The PySCF tables are Apache-2.0; the crate `NOTICE` reproduces PySCF's
//! notice.

use std::sync::Arc;

use crate::error::{Error, Result};
use crate::params::{BRAGG_BOHR, COVALENT_BOHR, SG1_RADII, VDW_BOHR};

/// Atomic-radii source for the size adjustment / pruning.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum RadiiTable {
    /// PySCF 2.14/2.15 `dft.radi.BRAGG_RADII` (Bohr), index = Z.
    Bragg,
    /// PySCF 2.14/2.15 covalent radii (Bohr), index = Z.
    Covalent,
    /// PySCF 2.14/2.15 van der Waals radii (Bohr), index = Z.
    Vdw,
    /// PySCF 2.14/2.15 `dft.radi.SG1RADII` (Bohr, H..Ar only), index = Z.
    Sg1,
    /// Caller-provided table indexed by `Z`, in Bohr; index 0 is a dummy.
    ///
    /// [`RadiiConfig::validate`](crate::partition::RadiiConfig::validate)
    /// requires at least one real entry and finite positive values from index
    /// 1 onward.  A direct [`RadiiTable::get`] call treats an invalid entry as
    /// missing.
    Custom(Arc<[f64]>),
}

impl RadiiTable {
    /// Human-readable name of the table (used in error messages).
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Bragg => "PySCF BRAGG",
            Self::Covalent => "PySCF COVALENT",
            Self::Vdw => "PySCF VDW",
            Self::Sg1 => "PySCF SG1",
            Self::Custom(_) => "custom radii",
        }
    }

    /// Radius in Bohr for nuclear charge `z`.
    ///
    /// # Errors
    /// [`Error::MissingElementData`] if the selected table has no entry for
    /// `z`.
    pub fn bohr(&self, z: u32) -> Result<f64> {
        self.get(z).ok_or(Error::MissingElementData {
            z,
            table: self.name(),
        })
    }

    /// Radius in Bohr, or `None` if the table does not cover `z`.
    ///
    /// Index 0 is a dummy, and a zero, negative or non-finite table entry is
    /// treated as missing.  Custom tables are indexed by `Z`; see
    /// [`RadiiTable::Custom`].
    ///
    /// ```
    /// use ruaa_dft_grids::radii::RadiiTable;
    ///
    /// let r = RadiiTable::Bragg.get(8).expect("oxygen has a Bragg radius");
    /// assert!(r > 0.0);
    /// assert!(RadiiTable::Bragg.get(0).is_none());
    /// ```
    #[must_use]
    pub fn get(&self, z: u32) -> Option<f64> {
        if z == 0 {
            return None;
        }
        let idx = z as usize;
        let value = match self {
            Self::Bragg => BRAGG_BOHR.get(idx).copied(),
            Self::Covalent => COVALENT_BOHR.get(idx).copied(),
            Self::Vdw => VDW_BOHR.get(idx).copied(),
            Self::Sg1 => SG1_RADII.get(idx).copied(),
            Self::Custom(table) => table.get(idx).copied(),
        };
        value.filter(|radius| radius.is_finite() && *radius > 0.0)
    }

    /// Validated caller-provided radii table indexed by `Z`, in Bohr.
    ///
    /// Index 0 is a dummy; entries `1..` must be finite and positive, and the
    /// table must contain at least one real entry.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] for an empty, non-finite or non-positive
    /// table.
    pub fn custom(values: Vec<f64>) -> Result<Self> {
        if values.len() < 2
            || values[1..]
                .iter()
                .any(|radius| !radius.is_finite() || *radius <= 0.0)
        {
            return Err(Error::InvalidArgument {
                argument: "custom radii table",
            });
        }
        Ok(Self::Custom(Arc::from(values)))
    }
}

impl From<Vec<f64>> for RadiiTable {
    /// Unchecked convenience conversion; use [`RadiiTable::custom`] when the
    /// table must be validated.
    fn from(v: Vec<f64>) -> Self {
        Self::Custom(Arc::from(v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_are_distinct_and_indexed_by_z() {
        assert!((RadiiTable::Bragg.bohr(2).unwrap() - 2.645616574391086).abs() < 1e-15);
        assert!((RadiiTable::Bragg.bohr(18).unwrap() - 3.4015070242171115).abs() < 1e-15);
        assert!((RadiiTable::Covalent.bohr(6).unwrap() - 1.3795000709324952).abs() < 1e-15);
        assert_eq!(RadiiTable::Bragg.get(0), None);
        assert_eq!(RadiiTable::Bragg.get(2000), None);
    }

    #[test]
    fn dummy_and_invalid_entries_are_missing() {
        assert_eq!(RadiiTable::Bragg.get(0), None);
        let custom: RadiiTable = vec![0.0, -1.0, 1.5].into();
        assert_eq!(custom.get(0), None);
        assert_eq!(custom.get(1), None);
        assert_eq!(custom.get(2), Some(1.5));
    }

    #[test]
    fn custom_constructor_validates() {
        assert!(RadiiTable::custom(vec![0.0, 1.0, 2.0]).is_ok());
        for bad in [
            vec![],
            vec![0.0],
            vec![0.0, -1.0],
            vec![0.0, 0.0],
            vec![0.0, f64::NAN],
            vec![0.0, f64::INFINITY],
        ] {
            assert!(RadiiTable::custom(bad.clone()).is_err(), "accepted {bad:?}");
        }
    }

    /// The a_AB size adjustment is antisymmetric and bounded by 0.5.
    #[test]
    fn size_adjust_is_antisymmetric() {
        for &(a, b) in &[(1u32, 9u32), (6, 8), (26, 9), (54, 1)] {
            let aa = crate::partition::becke::size_adjust_factor(
                RadiiTable::Bragg.bohr(a).unwrap(),
                RadiiTable::Bragg.bohr(b).unwrap(),
            );
            let bb = crate::partition::becke::size_adjust_factor(
                RadiiTable::Bragg.bohr(b).unwrap(),
                RadiiTable::Bragg.bohr(a).unwrap(),
            );
            assert!(
                (aa + bb).abs() < 1e-15,
                "a({a},{b}) = {aa}, a({b},{a}) = {bb}"
            );
            assert!(aa.abs() <= 0.5);
        }
    }

    #[test]
    fn custom_table() {
        let t = RadiiTable::from(vec![0.0, 1.0, 2.0]);
        assert_eq!(t.bohr(1).unwrap(), 1.0);
        assert_eq!(t.get(3), None);
    }
}
