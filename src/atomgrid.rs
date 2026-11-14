//! Shared atomic-grid value type and assembly.
//!
//! The value type and the shell-loop assembly are shared so the
//! partition/evaluator layers do not depend on a particular construction.
//! Builders live in [`crate::pyscf`], [`crate::preset`] and the caller's own
//! code.

use crate::angular::AngularPoints;
use crate::error::{Error, Result};

/// One atom's integration grid (points in Bohr relative to the atom center,
/// i.e. the caller translates them; weights include the `r^2 * 4pi` measure).
///
/// `shell_angular` is the source of truth for the shell count; `nr` is
/// derived by [`Self::nr`]. The fields are public for custom constructions,
/// but call [`Self::validate`] before passing a hand-built value to
/// [`crate::grid::GridPlan`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AtomicGrid {
    /// Nuclear charge this grid was built for.
    pub z: u32,
    /// Grid point coordinates in Bohr (untranslated, center at origin).
    pub points: Vec<[f64; 3]>,
    /// Unpartitioned quadrature weights `w_radial * w_angular`.
    pub quad_weights: Vec<f64>,
    /// Angular size of every shell (for diagnostics/tests).
    pub shell_angular: Vec<usize>,
}

impl AtomicGrid {
    /// Build an atomic grid from its parts and validate it.
    ///
    /// # Errors
    /// [`Error::UnsupportedElement`] for an invalid charge, and
    /// [`Error::GridArrayLength`] when the shell sizes and point arrays do
    /// not agree.
    pub fn new(
        z: u32,
        points: Vec<[f64; 3]>,
        quad_weights: Vec<f64>,
        shell_angular: Vec<usize>,
    ) -> Result<Self> {
        let grid = Self {
            z,
            points,
            quad_weights,
            shell_angular,
        };
        grid.validate()?;
        Ok(grid)
    }

    /// Number of radial shells.
    #[must_use]
    pub fn nr(&self) -> usize {
        self.shell_angular.len()
    }

    /// Number of grid points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the grid has no points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Sum of the unpartitioned quadrature weights.
    #[must_use]
    pub fn total_quad_weight(&self) -> f64 {
        self.quad_weights.iter().sum()
    }

    /// Validate the charge, array lengths and non-finiteness.
    ///
    /// Negative quadrature weights are allowed because some Lebedev rules
    /// (notably 74, 230 and 266 points) have them.
    ///
    /// # Errors
    /// [`Error::UnsupportedElement`] for `z` outside `1..=118`, and
    /// [`Error::GridArrayLength`] or [`Error::InvalidArgument`] for a
    /// malformed array.
    pub fn validate(&self) -> Result<()> {
        if !(1..=118).contains(&self.z) {
            return Err(Error::UnsupportedElement { z: self.z });
        }
        if self.points.len() != self.quad_weights.len() {
            return Err(Error::GridArrayLength {
                array: "atomic grid quad_weights",
                expected: self.points.len(),
                got: self.quad_weights.len(),
            });
        }
        let expected: usize = self.shell_angular.iter().sum();
        if expected != self.points.len() {
            return Err(Error::GridArrayLength {
                array: "atomic grid points",
                expected,
                got: self.points.len(),
            });
        }
        if self.shell_angular.contains(&0) {
            return Err(Error::InvalidArgument {
                argument: "atomic grid shell angular size",
            });
        }
        if self
            .points
            .iter()
            .any(|p| p.iter().any(|value| !value.is_finite()))
            || self.quad_weights.iter().any(|value| !value.is_finite())
        {
            return Err(Error::InvalidArgument {
                argument: "atomic grid values",
            });
        }
        Ok(())
    }
}

/// Assemble an atomic grid from parallel radial nodes/weights and a
/// per-shell angular rule.
///
/// `angular(shell)` returns the unit-sphere directions and sum-to-1 weights
/// for shell index `shell`; the returned grid records the actual direction
/// count per shell in [`AtomicGrid::shell_angular`].
///
/// # Errors
/// [`Error::LengthMismatch`] if `rads` and `wrads` differ in length,
/// [`Error::InvalidArgument`] if an angular rule returns mismatched or
/// non-finite arrays, and any error returned by `angular`.
pub fn assemble_atomic_grid<F>(
    z: u32,
    rads: &[f64],
    wrads: &[f64],
    mut angular: F,
) -> Result<AtomicGrid>
where
    F: FnMut(usize) -> Result<AngularPoints>,
{
    if rads.len() != wrads.len() {
        return Err(Error::LengthMismatch {
            expected: rads.len(),
            got: wrads.len(),
        });
    }
    let mut points = Vec::new();
    let mut quad_weights = Vec::new();
    let mut shell_angular = Vec::with_capacity(rads.len());
    for (shell, (&r, &wr)) in rads.iter().zip(wrads).enumerate() {
        let (dirs, wang) = angular(shell)?;
        if dirs.len() != wang.len() {
            return Err(Error::LengthMismatch {
                expected: dirs.len(),
                got: wang.len(),
            });
        }
        if dirs
            .iter()
            .all(|d| d.0.is_finite() && d.1.is_finite() && d.2.is_finite())
            && wang.iter().all(|w| w.is_finite())
        {
            shell_angular.push(dirs.len());
            for (&(dx, dy, dz), &wa) in dirs.iter().zip(&wang) {
                points.push([r * dx, r * dy, r * dz]);
                quad_weights.push(wr * wa);
            }
        } else {
            return Err(Error::InvalidArgument {
                argument: "angular grid values",
            });
        }
    }
    AtomicGrid::new(z, points, quad_weights, shell_angular)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assembles_shells_in_radial_order() {
        let rads = [1.0, 2.0];
        let wrads = [3.0, 4.0];
        let grid = assemble_atomic_grid(1, &rads, &wrads, |shell| {
            let n = shell + 1;
            let dirs = vec![(1.0, 0.0, 0.0); n];
            let weights = vec![1.0 / n as f64; n];
            Ok((dirs, weights))
        })
        .unwrap();
        assert_eq!(grid.nr(), 2);
        assert_eq!(grid.shell_angular, vec![1, 2]);
        assert_eq!(grid.points[0], [1.0, 0.0, 0.0]);
        assert_eq!(grid.points[1], [2.0, 0.0, 0.0]);
        assert_eq!(grid.points[2], [2.0, 0.0, 0.0]);
        assert_eq!(grid.quad_weights, vec![3.0, 2.0, 2.0]);
    }

    #[test]
    fn rejects_mismatched_radial_arrays() {
        assert_eq!(
            assemble_atomic_grid(1, &[1.0, 2.0], &[1.0], |_| {
                Ok((vec![(1.0, 0.0, 0.0)], vec![1.0]))
            }),
            Err(Error::LengthMismatch {
                expected: 2,
                got: 1,
            })
        );
    }

    #[test]
    fn rejects_mismatched_angular_arrays() {
        assert!(matches!(
            assemble_atomic_grid(1, &[1.0], &[1.0], |_| {
                Ok((vec![(1.0, 0.0, 0.0)], vec![]))
            }),
            Err(Error::LengthMismatch { .. })
        ));
    }
}
