//! PySCF 2.14/2.15 grid build path: level presets, pruning schemes, and
//! atomic grids.
//!
//! # What is ported
//!
//! * `pyscf.dft.gen_grid.RAD_GRIDS` / `ANG_ORDER` level defaults
//!   ([`presets`]);
//! * the PySCF pruning schemes (see [`crate::prune`]);
//! * the PySCF radial methods in [`crate::radial::kernels`], exposed as
//!   `(n, xi)` kernels through [`crate::radial::Radial`];
//! * the PySCF partition presets in [`crate::partition::PartitionConfig`];
//! * the atomic-grid builder ([`builder`]).
//!
//! # Point order
//!
//! The Lebedev rules use PySCF's point set and point order at every shared
//! level.  PySCF's `gen_atomic_grids` groups radial shells by angular size and
//! then `Grids.build` sorts the molecular grid into spatial boxes and pads it
//! to `Grids.alignment`.  This module keeps the atomic grid in radial-shell
//! order so the shell diagnostics and the existing `AtomicGrid` contract stay
//! meaningful, and it adds no padding points.
//! Point *sets*, counts and weights are identical; only the serialization
//! order and the padding differ.  Use [`crate::order::GridOrder`] for a
//! consumer-chosen spatial order.
//!

pub mod builder;
pub mod presets;

pub use builder::build_pyscf_atom_grid;
pub use presets::{angular_count, lebedev_points_for_order, level_defaults};
