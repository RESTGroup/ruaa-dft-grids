//! # `ruaa_dft_grids`
//!
//! Molecular density-functional-theory (DFT) numerical-integration grids for
//! Rust: atom-centred radial and angular quadratures, angular pruning,
//! molecular-grid assembly, and Becke-type partition weights with analytic
//! gradients and Hessians.
//!
//! The crate is program-agnostic.  A grid is described by a
//! [`GridSpec`]: an atomic topology plus a partition
//! configuration.  The topology is either
//!
//! * a data-driven [`GridPreset`] — shell counts,
//!   radial scales, cutoffs and per-shell angular sizes supplied by the
//!   caller (or read from a TOML/JSON preset file), or
//! * the built-in [`PyscfGridSpec`], which
//!   implements the open-source PySCF `Grids.level` construction.
//!
//! Program-specific presets are data, not hidden tables: a caller who has the
//! relevant shell data supplies it as a [`GridPreset`]
//! and gets the same points, pruning and weights through the same API.
//!
//! This is a library, not a DFT program: it takes a geometry and a grid
//! specification and returns points, quadrature weights and partition
//! weights.  It does not know about basis sets, densities or functionals.
//!
//! ## Quick start
//!
//! ```rust
//! use ruaa_dft_grids::grid::PyscfGridSpec;
//! use ruaa_dft_grids::moleculegrid::build_molecular_grid;
//!
//! fn main() -> ruaa_dft_grids::Result<()> {
//!     // Water, Cartesian coordinates in Bohr.
//!     let water = [
//!         (8u32, [0.0, 0.0, 0.0]),
//!         (1, [0.0, 1.430, 1.110]),
//!         (1, [0.0, -1.430, 1.110]),
//!     ];
//!
//!     let spec = PyscfGridSpec::from_level(1)?;
//!     let grid = build_molecular_grid(&water, &spec)?;
//!
//!     assert_eq!(grid.coords().len(), grid.len());
//!     assert_eq!(grid.weights().len(), grid.len());
//!     println!("{} points, total weight {:.6}", grid.len(), grid.total_weight());
//!     Ok(())
//! }
//! ```
//!
//! ## Data-driven presets
//!
//! A [`GridPreset`] is a plain data structure.  It
//! can be built in Rust or deserialized from a file:
//!
//! ```rust
//! use ruaa_dft_grids::grid::GridSpec;
//! use ruaa_dft_grids::moleculegrid::build_molecular_grid;
//! use ruaa_dft_grids::preset::{ElementPreset, GridPreset};
//! use ruaa_dft_grids::radial::Radial;
//!
//! fn main() -> ruaa_dft_grids::Result<()> {
//!     let h_xi = Radial::TreutlerAhlrichs.xi_for_element(1)?;
//!     let o_xi = Radial::TreutlerAhlrichs.xi_for_element(8)?;
//!     let preset = GridPreset::new(Radial::TreutlerAhlrichs)
//!         .with_element(1, ElementPreset::fixed(35, h_xi, 50))
//!         .with_element(8, ElementPreset::fixed(35, o_xi, 302));
//!     let spec = GridSpec::from_preset(preset);
//!     let water = [(8u32, [0.0, 0.0, 0.0]), (1, [0.0, 1.43, 1.11])];
//!     let grid = build_molecular_grid(&water, &spec)?;
//!     assert!(!grid.is_empty());
//!     Ok(())
//! }
//! ```
//!
//! The preset representation uses one `[[element]]` entry per nuclear charge,
//! so it is readable in TOML:
//!
//! ```toml
//! radial = "treutler_ahlrichs"
//! molecular_cutoffs = false
//!
//! [[element]]
//! z = 8
//! n_rad = 75
//! radial_scale = 1.1
//! angular = { pruned = { base = 302, prune = "nwchem" } }
//! ```
//!
//! [`AngularLayout`] covers a fixed rule, a
//! pruning policy, an explicit per-shell size list and a
//! Gauss–Legendre x uniform-phi product rule.  A construction that does not
//! fit any of those can be assembled as an [`AtomicGrid`]
//! and materialised with
//! [`build_molecular_grid_from_atomic_grids`].
//!
//! ## What the crate provides
//!
//! * Lebedev angular rules with sum-to-one weights, the 12-point icosahedral
//!   and 18-point octahedral rules, and Gauss–Legendre product grids;
//! * eight radial node maps, including the Treutler–Ahlrichs, Delley,
//!   Mura–Knowles, Gauss–Chebyshev, Becke, Chebyshev-with-GC and
//!   Euler–Maclaurin constructions;
//! * PySCF pruning schemes, a five-batch ladder with caller-supplied
//!   boundaries, and explicit per-shell sizes;
//! * classic Becke, Stratmann–Scuseria–Frisch, SS, mBecke and LKO partitions,
//!   with **analytic gradients and Hessians** for the differentiable ones;
//! * a range-streaming evaluator that never allocates an
//!   $n_\mathrm{atom} \times n_\mathrm{grid}$ tensor;
//! * validated point-order permutations for consumer-defined sorting and
//!   blocking;
//! * one typed [`Error`] enum for construction, evaluation and ordering.
//!
//! ## Conventions
//!
//! * **Coordinates and lengths are in Bohr.**
//! * Angular weights sum to **one**; the $4\pi$ factor lives in the radial
//!   weight, so the quadrature weight is
//!   $q_g = 4\pi r^2\ \Delta r\ w_\mathrm{ang}$.
//! * The total grid weight is $w_g = q_g P_g$, where $P_g$ is the normalized
//!   partition fraction of the generating atom.  The unpartitioned
//!   quadrature weight and the fraction are stored separately, so response
//!   code can form $\partial w_g/\partial\mathbf R = q_g\ \partial P_g/\partial\mathbf R$.
//! * **Angular pruning** is part of the topology: `Pruned`/`Shells` choose the
//!   per-shell angular size.  **Post-build pruning** is a separate point
//!   selection: builders return every angular point; use
//!   [`moleculegrid::PostPrune`] to select a subset.
//!
//! ## Pipeline
//!
//! ```text
//! GridSpec -> GridPlan -> PartitionEvaluator -> BatchBuffer
//! ```
//!
//! [`grid::GridSpec`] is the full specification (topology plus partition).
//! [`grid::GridPlan`] stores the point topology; it is geometry-independent
//! unless the preset enables molecular radial cutoffs.  The evaluator fills a
//! caller-chosen point range into a reusable [`grid::BatchBuffer`].
//! [`moleculegrid::build_molecular_grid`] is the parallel convenience builder
//! that materialises the whole [`moleculegrid::MolecularGrid`].
//!
//! ## Documentation
//!
//! The long-form pages are part of the API documentation:
//!
//! | Page | What it covers |
//! |---|---|
//! | [User guide](doc::guide) | geometry -> preset -> grid -> derivatives -> ordering |
//! | [Presets](doc::presets) | the built-in PySCF level, preset files, the element domain, the preset catalog, the PySCF cross-check |
//! | [Numerical algorithms](doc::algorithms) | radial maps, angular rules, pruning and partition formulas |
//! | [Partition derivative API](doc::partition_derivatives) | tensor layouts, streaming, selection, sorted grids |
//! | [Becke derivatives](doc::becke_derivatives) | derivation of the classic Becke gradient and Hessian |
//! | [Scheme variants](doc::scheme_variants) | SSF, SS, LKO and mBecke: formula and cost differences |
//! | [Performance](doc::performance) | measured throughput and build-target guidance |
//!
//! API starting points:
//!
//! * [`grid::GridSpec`], [`grid::PyscfGridSpec`],
//!   [`preset::GridPreset`] — build specifications;
//! * [`moleculegrid::build_molecular_grid`],
//!   [`moleculegrid::MolecularGrid`] — the materialised grid;
//! * [`molecule::Molecule`] — validated charges and positions;
//! * [`partition::PartitionConfig`], [`partition::PartitionEvaluator`] —
//!   partition weights at arbitrary points and range evaluation;
//! * [`partition::PartitionDerivEvaluator`], [`partition::DerivPoints`] —
//!   analytic gradients and Hessians;
//! * [`grid::GridPlan::from_atomic_grids`], [`atomgrid::AtomicGrid`] — a
//!   caller-supplied atomic topology for a custom construction;
//! * [`radial::RadialSpec`], [`prune::Pruning`] — custom radial and pruning
//!   policies for a caller-assembled [`atomgrid::AtomicGrid`];
//! * [`radial::Radial`], [`angular`], [`prune::Prune`] — the building blocks;
//! * [`order::GridOrder`] — consumer point ordering.
//!
//! ## Module map
//!
//! | Module | Content |
//! |---|---|
//! | [`angular`] | Lebedev rules, 12/18-point rules, product grids |
//! | [`atomgrid`] | shared `AtomicGrid` value type and shell-loop assembly |
//! | [`error`] | crate-wide [`Error`] and [`Result`] |
//! | [`grid`] | specifications, `GridPlan`, `BatchBuffer`, range API |
//! | [`molecule`] | validated `Atom`/`Molecule` value type |
//! | [`moleculegrid`] | full-grid output, post-build pruning, grid-native derivatives |
//! | [`order`] | point permutations and spatial orders |
//! | [`partition`] | partition configurations, value and derivative evaluators |
//! | [`preset`] | data-driven grid presets |
//! | [`prune`] | pruning schemes and per-shell size policies |
//! | [`pyscf`] | built-in PySCF level presets and atomic-grid builder |
//! | [`radial`] | radial node mappings and cutoff-limited shells |
//! | [`radii`] | radii-table selection |
//! | [`doc`] | the long-form documentation pages above |
//!
//! ## Minimum supported Rust version
//!
//! `ruaa_dft_grids` uses the 2024 edition and requires Rust 1.87 or newer.
//!
//! ## References
//!
//! * A. D. Becke, *J. Chem. Phys.* **88**, 2547 (1988) (fuzzy cells).
//! * O. Treutler, R. Ahlrichs, *J. Chem. Phys.* **102**, 346 (1995) (M3/M4).
//! * M. Krack, A. M. Köster, *J. Chem. Phys.* **108**, 3226 (1998).
//! * H. Stratmann, G. E. Scuseria, M. J. Frisch, *Chem. Phys. Lett.* **257**,
//!   213 (1996).
//! * H. Laqua, J. Kussmann, C. Ochsenfeld, *J. Chem. Phys.* **149**, 204111
//!   (2018) (LKO).
//! * PySCF 2.x `pyscf.dft.radi` and `pyscf.dft.gen_grid`.
//! * V. I. Lebedev, D. N. Laikov, *Doklady Mathematics* **59**, 477 (1999).
//!
//! ## License
//!
//! Apache-2.0. See the `LICENSE` and `NOTICE` files in the repository.

#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]
// Multi-line LaTeX display math inside Markdown lists trips this lint even
// when the rendering is correct; the formula blocks are deliberate.
#![allow(clippy::doc_lazy_continuation, clippy::doc_overindented_list_items)]

/// Doctests for the README, which is the crates.io front page.
#[cfg(doctest)]
mod readme_doctests {
    #![doc = include_str!("../README.md")]
}

pub mod angular;
pub mod atomgrid;
pub mod catalog;
pub mod doc;
pub mod error;
pub mod grid;
mod lebedev;
pub mod molecule;
pub mod moleculegrid;
pub mod order;
#[doc(hidden)]
pub mod params;
pub mod partition;
pub mod preset;
pub mod prune;
pub mod pyscf;
pub mod radial;
pub mod radii;

pub use error::{Error, Result};
pub use molecule::{Atom, Molecule};

pub use atomgrid::AtomicGrid;
pub use grid::{BatchBuffer, GridPlan, GridSpec, PyscfGridSpec, PyscfGridSpecBuilder};
pub use moleculegrid::{
    GridPoint, GridSplit, MolecularGrid, PostPrune, build_molecular_grid,
    build_molecular_grid_from_atomic_grids, build_molecular_grid_from_atoms,
    build_molecular_grid_from_plan,
};
pub use order::GridOrder;
pub use partition::{
    DEFAULT_WORKER_MEMORY_BYTES, DerivContract, DerivOptions, DerivOutput, DerivPoints,
    DerivRequest, DerivScheme, DerivScratch, DerivWorkspace, GradientContract, HessianBackend,
    HessianContract, PartitionConfig, PartitionConfigBuilder, PartitionDerivEvaluator,
    PartitionEvaluator, PartitionScheme, RadiiAdjust, RadiiConfig, Workspace,
};
pub use preset::{
    AngularLayout, ElementEntry, ElementPreset, ElementRangeDefault, ElementRangeEntry, GridPreset,
};

/// The circle constant $\pi$.
pub const PI: f64 = std::f64::consts::PI;
/// The crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
/// Bohr radius in ångström (CODATA 2010), used for Ångström <-> Bohr
/// conversion of Bragg radii and the Becke size adjustment.
pub const BOHR: f64 = 0.52917721092;
