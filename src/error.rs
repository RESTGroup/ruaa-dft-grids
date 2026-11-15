//! The crate-wide error type.
//!
//! Every fallible operation in this crate returns [`Result<T>`], i.e.
//! `std::result::Result<T, Error>`. The `derive_builder`-generated builder
//! error types implement `From` into [`enum@Error`], so a caller that composes
//! construction, evaluation and ordering can propagate with one `?` and match
//! a single enum.
//!
//! # Error layers
//!
//! * [`crate::params`] and the other pure data tables answer "do we have an
//!   entry for this element / scheme / order?" with `Option<T>`; they never
//!   allocate and never panic on a missing entry.
//! * The constructors built on top of them — [`crate::grid::GridPlan`],
//!   the atomic-grid builders, [`crate::partition::PartitionEvaluator`],
//!   [`crate::moleculegrid::build_molecular_grid`] — turn a `None` from a
//!   table into one of the variants below, so the caller sees *why* a
//!   configuration was rejected.
//! * Buffer-length, range, selection and permutation checks are reported
//!   directly at the public API boundary.
//!
//! Panics are reserved for genuine internal invariants (a scratch buffer
//! of the wrong concrete type, an enum arm that a preceding `match` already
//! rules out, a builder whose generated default fields are valid by
//! construction). Bad user data — element charges, angular schemes, PySCF
//! levels, Lebedev orders, pruning domains, cutoffs, selections and buffer
//! lengths — is an [`enum@Error`].

use thiserror::Error;

use crate::partition::PartitionScheme;

/// `Result` alias used throughout the crate.
pub type Result<T> = core::result::Result<T, Error>;

/// Everything that can go wrong while building or evaluating a DFT grid.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum Error {
    /// An angular grid scheme is outside the supported `1..=7` range.
    #[error("angular grid scheme {scheme} is not supported (expected 1..=7)")]
    UnsupportedScheme {
        /// The offending scheme.
        scheme: u8,
    },

    /// A nuclear charge has no periodic-table row (Z = 0 or Z > 118).
    #[error("nuclear charge Z = {z} is not a supported element (expected 1..=118)")]
    UnsupportedElement {
        /// The offending nuclear charge.
        z: u32,
    },

    /// A reference data table has no entry for this element.
    ///
    /// `table` names the missing table so the caller can tell which data set
    /// is incomplete.
    #[error("no {table} data for Z = {z}")]
    MissingElementData {
        /// The nuclear charge that is not covered.
        z: u32,
        /// Human-readable name of the table that has no entry.
        table: &'static str,
    },

    /// A Lebedev angular quadrature with this point count is not supported.
    ///
    /// Used both for an exact unsupported count and for a requested minimum
    /// count larger than the largest ladder entry.
    #[error("no Lebedev angular grid with {points} points")]
    UnsupportedLebedevGrid {
        /// Requested number of angular points.
        points: usize,
    },

    /// A grid configuration by name or literal is not a supported
    /// construction.
    #[error("grid `{name}` is not supported: {reason}")]
    UnsupportedGrid {
        /// The offending configuration.
        name: String,
        /// Why it was rejected.
        reason: String,
    },

    /// A periodic-table row is outside the `1..=6` range covered by the
    /// pruning cutoff tables.
    #[error("periodic-table row {row} is outside 1..=6")]
    InvalidPeriodicRow {
        /// The offending row.
        row: u32,
    },

    /// PySCF has no Lebedev order with this index.
    #[error("PySCF has no Lebedev order {order}")]
    UnsupportedLebedevOrder {
        /// The offending Lebedev order.
        order: u8,
    },

    /// PySCF `Grids.level` is outside `0..=9`.
    #[error("PySCF level {level} is outside 0..=9")]
    UnsupportedLevel {
        /// The offending level.
        level: u8,
    },

    /// A `derive_builder` configuration was rejected before the grid was
    /// built (an unset required field or a custom validator).
    ///
    /// The generated partition/spec builders keep their own message while
    /// letting a caller use the crate-wide [`Result`] with `?`.
    #[error("{what} builder rejected the configuration: {message}")]
    InvalidBuilderConfig {
        /// Which builder rejected the configuration.
        what: &'static str,
        /// The builder's message.
        message: String,
    },

    /// A grid index is outside the supported range.
    #[error("grid index {index} is outside the supported range")]
    UnsupportedCosxGrid {
        /// The offending index.
        index: usize,
    },

    /// A requested range is reversed or extends beyond the plan.
    #[error("grid range {start}..{end} is outside 0..{len}")]
    RangeOutOfBounds {
        /// Requested start.
        start: usize,
        /// Requested end (exclusive).
        end: usize,
        /// Plan length.
        len: usize,
    },

    /// A caller-provided buffer has the wrong length.
    #[error("length mismatch: expected {expected}, got {got}")]
    LengthMismatch {
        /// Expected length.
        expected: usize,
        /// Actual length.
        got: usize,
    },

    /// A hand-built atomic or molecular grid has point arrays of different
    /// lengths.
    #[error("grid array {array} has length {got}, expected {expected}")]
    GridArrayLength {
        /// Name of the inconsistent array.
        array: &'static str,
        /// Length implied by the other arrays.
        expected: usize,
        /// Actual length of this array.
        got: usize,
    },

    /// A hand-built molecular grid names an atom outside `0..natm`.
    #[error("molecular grid atom index {atom} is outside 0..{natm}")]
    InvalidGridAtom {
        /// The offending source-atom index.
        atom: u32,
        /// Number of atoms in the grid.
        natm: usize,
    },

    /// A nuclear coordinate is not finite.
    #[error("nuclear coordinate of atom {atom} is not finite")]
    NonFiniteCoordinate {
        /// Index of the offending atom.
        atom: usize,
    },

    /// Two atoms occupy the same position, so their partition distance is
    /// undefined.
    #[error("atoms {a} and {b} occupy the same position")]
    CoincidentCenters {
        /// First atom index.
        a: usize,
        /// Second atom index.
        b: usize,
    },

    /// The charges stored in a grid or plan do not match the evaluator's.
    #[error("charge mismatch for atom {atom}: expected {expected}, got {got}")]
    ChargesMismatch {
        /// Index of the offending atom.
        atom: usize,
        /// Charge the evaluator was built with.
        expected: u32,
        /// Charge supplied by the plan or grid.
        got: u32,
    },

    /// A materialised grid has a different geometry from the derivative
    /// evaluator it is passed to.
    #[error("grid geometry does not match the derivative evaluator")]
    GeometryMismatch,

    /// Analytic derivatives are not implemented for the requested built-in
    /// scheme (or the ORCA classic early exit was requested).
    #[error(
        "analytic partition derivatives are not implemented for {scheme:?} (early_exit = {early_exit})"
    )]
    UnsupportedDerivative {
        /// Requested partition scheme.
        scheme: PartitionScheme,
        /// Whether the classic Becke early exit was requested.
        early_exit: bool,
    },

    /// An atom index (selection, centre, or differentiated row) is outside
    /// `0..natm`.
    #[error("atom index {atom} is outside 0..{natm}")]
    InvalidAtom {
        /// Offending atom index.
        atom: usize,
        /// Number of atoms.
        natm: usize,
    },

    /// A derivative backend's selection is not strictly increasing.
    #[error("derivative selection must be strictly increasing; got {atom} out of order")]
    InvalidSelectionOrder {
        /// Offending atom index.
        atom: usize,
    },

    /// A caller-provided argument must be finite and positive.
    #[error("{argument} must be a finite positive value")]
    InvalidArgument {
        /// Name of the offending argument.
        argument: &'static str,
    },

    /// A post-prune threshold is not a finite non-negative number.
    #[error("cutoff must be finite and non-negative; got {cutoff}")]
    InvalidCutoff {
        /// The rejected cutoff.
        cutoff: f64,
    },

    /// A `KeepIndices` prune list names a point outside `0..len`.
    #[error("prune index {index} is outside 0..{len}")]
    PruneIndexOutOfRange {
        /// The offending point index.
        index: usize,
        /// Number of points in the grid.
        len: usize,
    },

    /// A `KeepIndices` prune list names the same point more than once.
    #[error("prune index {index} is listed more than once")]
    PruneIndexDuplicate {
        /// The repeated point index.
        index: usize,
    },

    /// A grid has more points than the `u32` permutation space can address.
    #[error("grid of {points} points exceeds the {max}-point permutation limit")]
    TooManyPoints {
        /// Number of points requested.
        points: usize,
        /// Largest addressable point count.
        max: usize,
    },

    /// A caller-provided permutation holds a source index outside `0..len`.
    #[error("grid order slot {slot} holds source index {value} outside 0..{len}")]
    PermutationOutOfRange {
        /// Slot that holds the invalid source index.
        slot: usize,
        /// The invalid source index.
        value: u32,
        /// Number of points in the permutation.
        len: usize,
    },

    /// A caller-provided permutation holds a source index more than once.
    #[error("grid order lists source index {value} more than once")]
    PermutationDuplicate {
        /// The repeated source index.
        value: u32,
    },
    /// A preset catalog file or a referenced preset file could not be read
    /// or parsed.
    #[error("preset catalog: {message}")]
    Catalog {
        /// Human-readable failure detail, including the path and the parser
        /// message.
        message: String,
    },

    /// A preset name is not present in the catalog.
    #[error("preset `{name}` is not in program `{program}`")]
    CatalogEntryNotFound {
        /// Program key (for example `demo`).
        program: String,
        /// Preset name.
        name: String,
    },
}

impl From<crate::partition::PartitionConfigBuilderError> for Error {
    fn from(err: crate::partition::PartitionConfigBuilderError) -> Self {
        Self::InvalidBuilderConfig {
            what: "partition config",
            message: err.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every variant is `Display`able; this also pins the message shape for
    /// the most user-facing variants.
    #[test]
    fn messages_are_actionable() {
        assert_eq!(
            Error::UnsupportedScheme { scheme: 9 }.to_string(),
            "angular grid scheme 9 is not supported (expected 1..=7)"
        );
        assert_eq!(
            Error::MissingElementData {
                z: 60,
                table: "grid preset"
            }
            .to_string(),
            "no grid preset data for Z = 60"
        );
        assert_eq!(
            Error::PermutationDuplicate { value: 3 }.to_string(),
            "grid order lists source index 3 more than once"
        );
        assert_eq!(
            Error::InvalidCutoff { cutoff: -1.0 }.to_string(),
            "cutoff must be finite and non-negative; got -1"
        );
        assert_eq!(
            Error::PruneIndexOutOfRange { index: 3, len: 2 }.to_string(),
            "prune index 3 is outside 0..2"
        );
        assert_eq!(
            Error::PruneIndexDuplicate { index: 3 }.to_string(),
            "prune index 3 is listed more than once"
        );
    }

    /// The error is a standard library error source with no nested cause.
    #[test]
    fn implements_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(Error::UnsupportedElement { z: 0 });
        assert!(std::error::Error::source(&*err).is_none());
    }
}
