//! Coverage of the topology and partition building blocks.
//!
//! The crate should be able to construct every radial map, every angular
//! layout, every pruning scheme and every partition scheme through the public
//! API, without a program-specific branch.

use ruaa_dft_grids::atomgrid::AtomicGrid;
use ruaa_dft_grids::grid::{GridSpec, PyscfGridSpecBuilder};
use ruaa_dft_grids::molecule::Molecule;
use ruaa_dft_grids::moleculegrid::build_molecular_grid_from_atomic_grids;
use ruaa_dft_grids::partition::{PartitionConfig, PartitionScheme, RadiiConfig};
use ruaa_dft_grids::preset::{AngularLayout, ElementPreset, GridPreset};
use ruaa_dft_grids::prune::{LadderPrune, Prune};
use ruaa_dft_grids::radial::Radial;

const ALL_RADIAL: [Radial; 8] = [
    Radial::TreutlerAhlrichs,
    Radial::Delley,
    Radial::MuraKnowles,
    Radial::GaussChebyshev,
    Radial::Becke,
    Radial::OrcaOptM3,
    Radial::OrcaM3,
    Radial::EulerMaclaurin,
];

#[test]
fn every_radial_map_builds_a_preset_grid() {
    for radial in ALL_RADIAL {
        let preset = GridPreset::new(radial).with_element(6, ElementPreset::fixed(12, 1.0, 26));
        let spec = GridSpec::from_preset(preset);
        let grid = spec.atomic_grid(6).unwrap();
        assert_eq!(grid.nr(), 12, "{radial:?}");
        assert!(grid.total_quad_weight().is_finite());
    }
}

#[test]
fn every_angular_layout_builds() {
    let layouts = [
        AngularLayout::fixed(6),
        AngularLayout::fixed(12),
        AngularLayout::fixed(18),
        AngularLayout::fixed(26),
        AngularLayout::fixed(974),
        AngularLayout::pruned(110, Prune::Nwchem),
        AngularLayout::shells(vec![6, 26, 50]),
        AngularLayout::product(8),
    ];
    for layout in layouts {
        let element = ElementPreset {
            n_rad: 3,
            radial_scale: 1.0,
            cutoff: None,
            generate_all: true,
            angular: layout.clone(),
        };
        let preset = GridPreset::new(Radial::TreutlerAhlrichs).with_element(6, element);
        let spec = GridSpec::from_preset(preset);
        let grid = spec.atomic_grid(6).unwrap();
        assert!(!grid.is_empty(), "{layout:?}");
        assert_eq!(grid.nr(), 3, "{layout:?}");
    }
}

#[test]
fn every_pyscf_pruning_scheme_builds() {
    let bounds = [
        [0.25, 0.5, 1.0, 4.5],
        [0.1667, 0.5, 0.9, 3.5],
        [0.1, 0.4, 0.8, 2.5],
        [0.1, 0.4, 0.8, 2.5],
    ];
    let schemes = [
        Prune::None,
        Prune::Nwchem,
        Prune::Sg1,
        Prune::Treutler,
        Prune::Sgx,
        Prune::Ladder(LadderPrune::new([6, 26, 50, 110, 50], 110, bounds)),
    ];
    for scheme in schemes {
        let spec = PyscfGridSpecBuilder::default()
            .level(1)
            .prune(scheme)
            .build()
            .unwrap();
        let grid = GridSpec::from(spec).atomic_grid(6).unwrap();
        assert!(!grid.is_empty(), "{scheme:?}");
        assert!(grid.shell_angular.iter().all(|&n| n > 0), "{scheme:?}");
    }
}

#[test]
fn every_partition_scheme_evaluates() {
    let centers = [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]];
    let charges = [8u32, 1];
    let point = [0.3, 0.1, 1.0];
    for scheme in [
        PartitionScheme::Becke,
        PartitionScheme::Stratmann,
        PartitionScheme::GaussianSs,
        PartitionScheme::MBecke,
        PartitionScheme::Lko,
    ] {
        let config = PartitionConfig::new(scheme, RadiiConfig::pyscf());
        let weights = config.weights(point, &centers, &charges).unwrap();
        assert_eq!(weights.len(), 2);
        assert!(weights.iter().all(|w| w.is_finite()), "{scheme:?}");
        let total: f64 = weights.iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "{scheme:?}: {total}");
    }
}

#[test]
fn custom_atomic_grids_cover_special_angular_rules() {
    let molecule = Molecule::new([(1u32, [0.0, 0.0, 0.0]), (1, [2.0, 0.0, 0.0])]).unwrap();
    let grids = vec![
        AtomicGrid::new(1, vec![[0.1, 0.0, 0.0]], vec![1.0], vec![1]).unwrap(),
        AtomicGrid::new(1, vec![[1.9, 0.0, 0.0]], vec![1.0], vec![1]).unwrap(),
    ];
    let grid = build_molecular_grid_from_atomic_grids(
        &molecule,
        grids,
        PartitionConfig::new(PartitionScheme::Becke, RadiiConfig::pyscf()),
    )
    .unwrap();
    assert_eq!(grid.len(), 2);
    assert!(grid.spec().is_custom());
}
