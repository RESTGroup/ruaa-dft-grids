//! Regression coverage for the pre-publication review findings and for the
//! generic preset path.

use ruaa_dft_grids::atomgrid::AtomicGrid;
use ruaa_dft_grids::grid::{BatchBuffer, GridPlan, GridSpec, PyscfGridSpec};
use ruaa_dft_grids::molecule::Molecule;
use ruaa_dft_grids::moleculegrid::PostPrune;
use ruaa_dft_grids::moleculegrid::build_molecular_grid;
use ruaa_dft_grids::partition::{
    PartitionConfig, PartitionEvaluator, PartitionScheme, RadiiConfig, Workspace,
};
use ruaa_dft_grids::preset::{ElementPreset, GridPreset};
use ruaa_dft_grids::radial::Radial;

#[test]
fn total_weight_cut_keeps_negative_quadrature_weights() {
    let atoms = [(1u32, [-0.7, 0.0, 0.0]), (1, [0.7, 0.0, 0.0])];
    let grid = build_molecular_grid(&atoms, PyscfGridSpec::from_level(0).unwrap()).unwrap();
    assert!(grid.weights().iter().any(|&w| w < 0.0));
    let full = grid.total_weight();
    let split = grid.split(PostPrune::TotalWeightCut(0.0)).unwrap();
    assert!((split.retained.total_weight() - full).abs() < 1e-10);
    assert!(split.pruned.weights().iter().all(|&w| w == 0.0));
}

#[test]
fn ssf_degenerate_block_normalizes_uniformly() {
    let n = 60usize;
    let mut charges = Vec::new();
    let mut centers = Vec::new();
    for k in 0..n {
        let angle = std::f64::consts::TAU * k as f64 / n as f64;
        charges.push(6u32);
        centers.push([10.0 * angle.cos(), 10.0 * angle.sin(), 0.0]);
    }
    let config = PartitionConfig::new(PartitionScheme::Stratmann, RadiiConfig::pyscf());
    let scalar = config.weights([0.0, 0.0, 0.0], &centers, &charges).unwrap();
    let expected = 1.0 / n as f64;
    assert!((scalar[0] - expected).abs() < 1e-15);
    assert!(
        scalar[1..]
            .iter()
            .all(|&value| (value - expected).abs() < 1e-15)
    );

    // One grid point per generating atom, all at the global origin.
    let grids: Vec<AtomicGrid> = centers
        .iter()
        .map(|&center| {
            AtomicGrid::new(
                6,
                vec![[-center[0], -center[1], -center[2]]],
                vec![1.0],
                vec![1],
            )
            .unwrap()
        })
        .collect();
    let plan = GridPlan::from_atomic_grids(grids).unwrap();
    let evaluator = PartitionEvaluator::from_parts(&centers, &charges, config).unwrap();
    let mut ws = Workspace::new();
    let mut out = BatchBuffer::new();
    evaluator
        .fill_plan(&plan, 0..plan.len(), &mut ws, &mut out)
        .unwrap();
    assert_eq!(out.partition_weights(), scalar);
}

/// A symmetric cage whose unnormalized products are all tiny must normalize
/// to the uniform partition, not to a first-centre unit mass.
#[test]
fn symmetric_cage_normalizes_uniformly() {
    let n = 54usize;
    let mut charges = Vec::new();
    let mut centers = Vec::new();
    for k in 0..n {
        let angle = std::f64::consts::TAU * k as f64 / n as f64;
        charges.push(1u32);
        centers.push([10.0 * angle.cos(), 10.0 * angle.sin(), 0.0]);
    }
    let origin = [0.0, 0.0, 0.0];
    for scheme in [
        PartitionScheme::Becke,
        PartitionScheme::Stratmann,
        PartitionScheme::MBecke,
        PartitionScheme::Lko,
    ] {
        let config = PartitionConfig::new(scheme, RadiiConfig::pyscf());
        let w = config.weights(origin, &centers, &charges).unwrap();
        let expected = 1.0 / n as f64;
        for (atom, &value) in w.iter().enumerate() {
            assert!(
                (value - expected).abs() < 1e-12,
                "{scheme:?} atom {atom}: {value} != {expected}"
            );
        }
    }
}

#[test]
fn generic_preset_supports_molecular_cutoffs() {
    let element = ElementPreset::fixed(60, 1.0, 50).with_cutoff(4.0);
    let preset = GridPreset::new(Radial::EulerMaclaurin)
        .with_element(1, element)
        .with_molecular_cutoffs(true);
    let spec = GridSpec::from_preset(preset).with_partition(PartitionConfig::pyscf());
    let near = [(1u32, [-0.35, 0.0, 0.0]), (1, [0.35, 0.0, 0.0])];
    let far = [(1u32, [-1.5, 0.0, 0.0]), (1, [1.5, 0.0, 0.0])];
    assert!(spec.geometry_dependent());
    let near_plan = GridPlan::build(&near, &spec).unwrap();
    let far_plan = GridPlan::build(&far, &spec).unwrap();
    assert!(near_plan.geometry_dependent());
    assert!(far_plan.geometry_dependent());
    assert_ne!(near_plan.len(), far_plan.len());
    assert!(near_plan.len() <= far_plan.len());
}

#[test]
fn custom_atomic_grids_materialise_through_the_partition() {
    let atoms = [(1u32, [0.0, 0.0, 0.0]), (1, [1.0, 0.0, 0.0])];
    let molecule = Molecule::new(atoms).unwrap();
    let grids = vec![
        AtomicGrid::new(1, vec![[0.1, 0.0, 0.0]], vec![1.0], vec![1]).unwrap(),
        AtomicGrid::new(1, vec![[-0.1, 0.0, 0.0]], vec![1.0], vec![1]).unwrap(),
    ];
    let grid = ruaa_dft_grids::moleculegrid::build_molecular_grid_from_atomic_grids(
        &molecule,
        grids,
        PartitionConfig::new(PartitionScheme::Becke, RadiiConfig::pyscf()),
    )
    .unwrap();
    assert_eq!(grid.len(), 2);
    assert!(grid.spec().is_custom());
    assert!((1.99..=2.0).contains(&grid.total_weight()));
}
