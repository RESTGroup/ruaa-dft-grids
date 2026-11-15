//! Angular-grid pruning: per-radial-shell angular point counts.
//!
//! Schemes:
//!
//! * [`Prune::Nwchem`], [`Prune::Sg1`], [`Prune::Treutler`], [`Prune::Sgx`] —
//!   the PySCF `gen_grid` schemes;
//! * [`Prune::Ladder`] — a caller-parameterized five-batch region ladder;
//! * [`Prune::None`] — no pruning.
//!
//! Every scheme implements the common [`Pruning`] trait and consumes a
//! [`PruneContext`], so a new policy can be added without editing the
//! selector.
//!
//! [`PYSCF_REGION_BOUNDS`] holds the published SG-1/NWChem alpha boundaries
//! used by the PySCF schemes.  [`SG1_LADDER`] is the published SG-1 ladder.
//! A five-batch [`LadderPrune`] carries its own element-row boundaries, so the
//! core crate holds no program-specific ladder data.  Every scheme answers
//! the same question: given the radial shells of one atom and the base
//! angular size, how many angular points does each shell carry.

mod nwchem;
mod sg1;
mod sgx;
mod treutler;

use nwchem::nwchem_prune;
use sg1::sg1_prune;
use sgx::sgx_prune;
use treutler::treutler_prune;

use crate::error::{Error, Result};

/// Angular-pruning scheme.
///
/// The PySCF variants are the schemes of `pyscf.dft.gen_grid`; [`Self::Ladder`]
/// is a caller-parameterized five-batch region ladder usable with any radial
/// construction (it compares $r / \xi$, where $\xi$ is the radial scale).
#[derive(Clone, Copy, Debug, PartialEq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Prune {
    /// No pruning.
    None,
    /// `gen_grid.nwchem_prune` (PySCF default).
    #[default]
    Nwchem,
    /// `gen_grid.sg1_prune` (H..Ar only).
    Sg1,
    /// `gen_grid.treutler_prune`.
    Treutler,
    /// `gen_grid.sgx_prune` ("ORCA-like" pruning for SGX grids).
    Sgx,
    /// A five-batch region ladder ([`LadderPrune`]).
    Ladder(LadderPrune),
}

/// Per-shell angular sizes plus any non-fatal notice.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct PrunedSizes {
    /// Angular size of every active radial shell.
    pub sizes: Vec<usize>,
    /// Non-fatal notice produced by the scheme, if any.
    pub warning: Option<PruneWarning>,
}

impl PrunedSizes {
    /// Sizes with no warning.
    #[must_use]
    pub fn new(sizes: Vec<usize>) -> Self {
        Self {
            sizes,
            warning: None,
        }
    }

    /// Attach a non-fatal notice.
    #[must_use]
    pub fn with_warning(mut self, warning: PruneWarning) -> Self {
        self.warning = Some(warning);
        self
    }

    fn plain(sizes: Vec<usize>) -> Self {
        Self::new(sizes)
    }
}

/// Inputs shared by every pruning policy.
///
/// A policy reads the fields it needs: the PySCF schemes use `z`, `rads_bohr`
/// and `n_ang`; the five-batch ladder also uses the `ratios`.
#[derive(Clone, Copy, Debug)]
pub struct PruneContext<'a> {
    /// Nuclear charge.
    pub z: u32,
    /// Active radial-shell radii in Bohr, ascending.
    pub rads_bohr: &'a [f64],
    /// Matching dimensionless `r / xi` values, or an empty slice.
    pub ratios: &'a [f64],
    /// Unpruned base angular size.
    pub n_ang: usize,
}

impl<'a> PruneContext<'a> {
    /// Context for a PySCF atomic grid.
    pub(crate) fn for_pyscf(z: u32, rads_bohr: &'a [f64], ratios: &'a [f64], n_ang: usize) -> Self {
        Self {
            z,
            rads_bohr,
            ratios,
            n_ang,
        }
    }

    /// Context for a data-driven preset.
    pub(crate) fn for_preset(
        z: u32,
        rads_bohr: &'a [f64],
        ratios: &'a [f64],
        n_ang: usize,
    ) -> Self {
        Self::for_pyscf(z, rads_bohr, ratios, n_ang)
    }
}

/// Common contract for an angular-pruning policy.
///
/// Implement [`Pruning`] to compute sizes directly or when assembling a
/// custom [`crate::atomgrid::AtomicGrid`]; the standard preset path accepts
/// the concrete [`Prune`] selector.
pub trait Pruning {
    /// Per-shell angular sizes for `context`.
    ///
    /// # Errors
    /// The policy's element/range errors.
    fn sizes(&self, context: &PruneContext<'_>) -> Result<PrunedSizes>;
}

/// Run a pruning policy and require one size per active shell.
pub(crate) fn sizes_checked<P: Pruning + ?Sized>(
    policy: &P,
    context: &PruneContext<'_>,
    expected: usize,
) -> Result<PrunedSizes> {
    let pruned = policy.sizes(context)?;
    if pruned.sizes.len() != expected {
        return Err(Error::LengthMismatch {
            expected,
            got: pruned.sizes.len(),
        });
    }
    Ok(pruned)
}

/// Non-fatal notices produced while sizing a pruned ladder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PruneWarning {
    /// The ladder policy switched to the unpruned base for an element beyond
    /// its pruned domain.
    UnprunedFallback {
        /// The nuclear charge that triggered the fallback.
        z: u32,
    },
    /// The ladder policy kept the five-batch ladder for an element beyond
    /// its pruned domain.
    DefaultLadderFallback {
        /// The nuclear charge that triggered the fallback.
        z: u32,
    },
}

/// Angular sizing of elements above [`LadderPrune::max_pruned_z`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum HeavyFallback {
    /// Use the unpruned base size for every shell (Coarse..UltraFine).
    #[default]
    Unpruned,
    /// Keep applying the five-batch ladder with the last region row
    /// (SuperFine).
    DefaultLadder,
}

/// Element-row boundaries in $r / \xi$ units for the PySCF pruning schemes.
///
/// Rows: `Z <= 2`, `Z <= 10`, `Z <= 18`.  These are the published
/// SG-1/NWChem alpha boundaries used by `place_index`.
pub const PYSCF_REGION_BOUNDS: [[f64; 4]; 3] = [
    [0.25, 0.5, 1.0, 4.5],
    [0.1667, 0.5, 0.9, 3.5],
    [0.1, 0.4, 0.8, 2.5],
];

/// The SG-1 angular ladder, shared by PySCF `sg1_prune` and a five-batch
/// preset.  The fifth batch steps back down to the third size.
pub const SG1_LADDER: [usize; 5] = [6, 38, 86, 194, 86];

/// Default heaviest element with a pruned five-batch ladder.
pub const MAX_PRUNED_Z: u32 = 36;

/// A five-batch region ladder: sizes `[a1, a2, a3, a4, a3]` inner to outer.
///
/// A shell at dimensionless radius $r / \xi$ carries `ladder[k]`, where `k` is
/// the number of [`Self::bounds`] entries of its element row that the ratio is
/// at least (inclusive comparison).  Elements above [`Self::max_pruned_z`] use
/// the [`Self::heavy_fallback`] sizing.
///
/// The ladder sizes and the four element-row boundaries are supplied by the
/// caller: the core crate carries no program-specific ladder data.  A preset
/// that uses a five-batch ladder fills the sizes, the unpruned fallback and
/// the boundaries; `max_pruned_z` and `heavy_fallback` default to
/// [`MAX_PRUNED_Z`] and [`HeavyFallback::Unpruned`].  A size that the angular
/// builder cannot construct is rejected when the grid is assembled.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LadderPrune {
    /// Five angular sizes, inner to outer.
    pub ladder: [usize; 5],
    /// Angular size used by the unpruned fallback.
    pub unpruned: usize,
    /// Element-row boundaries `Z <= 2`, `Z <= 10`, `Z <= 18`, heavier.
    pub bounds: [[f64; 4]; 4],
    /// Highest nuclear charge with a pruned ladder.
    #[serde(default = "default_max_pruned_z")]
    pub max_pruned_z: u32,
    /// Sizing for elements above `max_pruned_z`.
    #[serde(default)]
    pub heavy_fallback: HeavyFallback,
}

fn default_max_pruned_z() -> u32 {
    MAX_PRUNED_Z
}

impl LadderPrune {
    /// A ladder with the default [`MAX_PRUNED_Z`] domain and unpruned
    /// heavier-element fallback.
    #[must_use]
    pub const fn new(ladder: [usize; 5], unpruned: usize, bounds: [[f64; 4]; 4]) -> Self {
        Self {
            ladder,
            unpruned,
            bounds,
            max_pruned_z: MAX_PRUNED_Z,
            heavy_fallback: HeavyFallback::Unpruned,
        }
    }

    /// Use `fallback` for elements above [`Self::max_pruned_z`].
    #[must_use]
    pub const fn with_heavy_fallback(mut self, fallback: HeavyFallback) -> Self {
        self.heavy_fallback = fallback;
        self
    }

    /// Change the heaviest element with a pruned ladder.
    #[must_use]
    pub const fn with_max_pruned_z(mut self, max_pruned_z: u32) -> Self {
        self.max_pruned_z = max_pruned_z;
        self
    }

    /// Angular sizes for $ratios = r / \xi$.
    ///
    /// `ratios` must be parallel to the active radial shells.
    #[must_use]
    pub fn sizes(self, z: u32, ratios: &[f64]) -> (Vec<usize>, Option<PruneWarning>) {
        if z > self.max_pruned_z {
            let warning = match self.heavy_fallback {
                HeavyFallback::Unpruned => {
                    return (
                        vec![self.unpruned; ratios.len()],
                        Some(PruneWarning::UnprunedFallback { z }),
                    );
                }
                HeavyFallback::DefaultLadder => PruneWarning::DefaultLadderFallback { z },
            };
            let sizes = self.apply(prune_row(z), ratios);
            return (sizes, Some(warning));
        }
        (self.apply(prune_row(z), ratios), None)
    }

    fn apply(self, row: usize, ratios: &[f64]) -> Vec<usize> {
        let bounds = self.bounds[row];
        ratios
            .iter()
            .map(|&ratio| {
                let batch = bounds.iter().filter(|&&b| ratio >= b).count();
                self.ladder[batch.min(4)]
            })
            .collect()
    }
}

/// The ladder row of element `z`.
///
/// The periodic-table period ([`crate::params::period_index`]) capped at the
/// fourth row: `Z <= 2`, `Z <= 10`, `Z <= 18`, and everything heavier share
/// the last row.
#[must_use]
pub fn prune_row(z: u32) -> usize {
    crate::params::period_index(z).map_or(3, |period| period.min(3))
}

impl Prune {
    /// Per-radial-shell angular point counts.
    ///
    /// `rads` are the active shell radii in Bohr; `ratios` are the matching
    /// $r / \xi$ values and are only required by [`Self::Ladder`] (pass an
    /// empty slice for the other schemes); `n_ang` is the unpruned base
    /// angular size.
    ///
    /// # Errors
    /// [`Error::MissingElementData`] for an unsupported element in the
    /// SG1/SGX schemes, [`Error::UnsupportedLebedevGrid`] for an `n_ang` that
    /// is not in the scheme's ladder, and [`Error::LengthMismatch`] when a
    /// ladder scheme receives mismatched `rads`/`ratios`.
    pub fn sizes(self, z: u32, rads: &[f64], ratios: &[f64], n_ang: usize) -> Result<PrunedSizes> {
        Ok(match self {
            Self::None => PrunedSizes::plain(vec![n_ang; rads.len()]),
            Self::Nwchem => PrunedSizes::plain(nwchem_prune(z, rads, n_ang)?),
            Self::Sg1 => PrunedSizes::plain(sg1_prune(z, rads)?),
            Self::Treutler => PrunedSizes::plain(treutler_prune(rads, n_ang)),
            Self::Sgx => PrunedSizes::plain(sgx_prune(z, rads, n_ang)?),
            Self::Ladder(ladder) => {
                if ratios.len() != rads.len() {
                    return Err(Error::LengthMismatch {
                        expected: rads.len(),
                        got: ratios.len(),
                    });
                }
                let (sizes, warning) = ladder.sizes(z, ratios);
                PrunedSizes { sizes, warning }
            }
        })
    }

    /// Whether [`Self::sizes`] consumes the $r / \xi$ ratios.
    #[must_use]
    pub fn needs_ratios(self) -> bool {
        matches!(self, Self::Ladder(_))
    }
}

impl Pruning for Prune {
    fn sizes(&self, context: &PruneContext<'_>) -> Result<PrunedSizes> {
        (*self).sizes(context.z, context.rads_bohr, context.ratios, context.n_ang)
    }
}

/// PySCF's `(nuc <= 2 ? 0 : nuc <= 10 ? 1 : 2)` row selection plus the
/// `(rads/r_atom > alpha).sum()` region index used by NWChem/SG1/SGX.
pub(crate) fn place_index(rads: &[f64], r_atom: f64, z: u32) -> Vec<usize> {
    let row = if z <= 2 {
        0
    } else if z <= 10 {
        1
    } else {
        2
    };
    rads.iter()
        .map(|&r| {
            let ratio = r / (r_atom + 1e-200);
            PYSCF_REGION_BOUNDS[row]
                .iter()
                .filter(|&&a| ratio > a)
                .count()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::PYSCF_LEBEDEV_NGRID;

    const TEST_BOUNDS: [[f64; 4]; 4] = [
        [0.25, 0.5, 1.0, 4.5],
        [0.1667, 0.5, 0.9, 3.5],
        [0.1, 0.4, 0.8, 2.5],
        [0.02, 0.1, 0.2, 3.5],
    ];

    #[test]
    fn prune_counts_are_plausible() {
        let rads: Vec<f64> = (1..=50).map(|i| i as f64 * 0.1).collect();
        for scheme in [Prune::None, Prune::Nwchem, Prune::Treutler, Prune::Sgx] {
            let sizes = scheme.sizes(6, &rads, &[], 302).unwrap().sizes;
            assert_eq!(sizes.len(), rads.len());
            assert!(sizes.iter().all(|&n| PYSCF_LEBEDEV_NGRID.contains(&n)));
        }
        let sg1 = Prune::Sg1.sizes(6, &rads, &[], 302).unwrap().sizes;
        assert!(sg1.iter().all(|&n| SG1_LADDER.contains(&n)));
    }

    struct WrongSize;

    impl Pruning for WrongSize {
        fn sizes(&self, _context: &PruneContext<'_>) -> Result<PrunedSizes> {
            Ok(PrunedSizes::new(vec![12]))
        }
    }

    #[test]
    fn sizes_checked_rejects_a_wrong_shell_count() {
        let rads = [0.2, 0.5];
        let context = PruneContext {
            z: 6,
            rads_bohr: &rads,
            ratios: &[],
            n_ang: 50,
        };
        assert_eq!(
            sizes_checked(&WrongSize, &context, 2),
            Err(crate::Error::LengthMismatch {
                expected: 2,
                got: 1,
            })
        );
    }

    #[test]
    fn pruning_trait_covers_pyscf() {
        let rads = [0.2, 0.5, 1.0, 2.0, 4.0];
        let context = PruneContext {
            z: 6,
            rads_bohr: &rads,
            ratios: &[],
            n_ang: 302,
        };
        let pyscf = Prune::Nwchem;
        let sizes = Pruning::sizes(&pyscf, &context).unwrap().sizes;
        assert_eq!(sizes.len(), rads.len());
        assert!(sizes.iter().all(|&n| n > 0));
    }

    #[test]
    fn five_batch_ladder_uses_supplied_regions() {
        let ladder = LadderPrune::new([6, 18, 50, 110, 50], 110, TEST_BOUNDS);
        let ratios = [0.1, 0.3, 0.6, 0.95, 5.0];
        let (sizes, warning) = ladder.sizes(6, &ratios);
        assert_eq!(warning, None);
        assert_eq!(sizes, vec![6, 18, 50, 110, 50]);

        let (fallback, warning) = ladder.sizes(54, &ratios);
        assert_eq!(warning, Some(PruneWarning::UnprunedFallback { z: 54 }));
        assert!(fallback.iter().all(|&n| n == 110));
    }

    #[test]
    fn superfine_heavy_elements_keep_the_default_ladder() {
        let ladder = LadderPrune::new([194, 302, 590, 974, 590], 974, TEST_BOUNDS)
            .with_heavy_fallback(HeavyFallback::DefaultLadder);
        let ratios = [0.01, 0.05, 0.15, 1.0];
        let (sizes, warning) = ladder.sizes(54, &ratios);
        assert_eq!(warning, Some(PruneWarning::DefaultLadderFallback { z: 54 }));
        assert_eq!(sizes, vec![194, 302, 590, 974]);
    }

    #[test]
    fn ladder_requires_matching_ratios() {
        let ladder = Prune::Ladder(LadderPrune::new(SG1_LADDER, 194, TEST_BOUNDS));
        let rads = [0.1, 0.5];
        assert_eq!(
            ladder.sizes(6, &rads, &[0.1], 194),
            Err(crate::Error::LengthMismatch {
                expected: 2,
                got: 1,
            })
        );
        assert!(ladder.sizes(6, &rads, &[0.1, 0.5], 194).is_ok());
    }

    #[test]
    fn pyscf_region_table_stays_published() {
        assert_eq!(PYSCF_REGION_BOUNDS[0], [0.25, 0.5, 1.0, 4.5]);
        assert_eq!(PYSCF_REGION_BOUNDS[1], [0.1667, 0.5, 0.9, 3.5]);
        assert_eq!(PYSCF_REGION_BOUNDS[2], [0.1, 0.4, 0.8, 2.5]);
        assert_eq!(SG1_LADDER, [6, 38, 86, 194, 86]);
        assert_eq!(prune_row(2), 0);
        assert_eq!(prune_row(10), 1);
        assert_eq!(prune_row(18), 2);
        assert_eq!(prune_row(54), 3);
    }
}
