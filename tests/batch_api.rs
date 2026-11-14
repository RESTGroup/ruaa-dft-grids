//! Batch/range API checks for the plan/evaluator pipeline.
//!
//! The full-build convenience output and the range API must agree; ranges may
//! cross atom boundaries; the spec can change partition without rebuilding the
//! topology; a plan/evaluator atom-count mismatch is rejected.

use ruaa_dft_grids::atomgrid::AtomicGrid;
use ruaa_dft_grids::grid::{BatchBuffer, GridPlan, GridSpec, PyscfGridSpecBuilder};
use ruaa_dft_grids::moleculegrid::build_molecular_grid;
use ruaa_dft_grids::partition::{PartitionConfig, PartitionScheme, RadiiConfig};
use ruaa_dft_grids::partition::{PartitionEvaluator, Workspace};

fn h2o() -> Vec<(u32, [f64; 3])> {
    vec![
        (8, [0.0, 0.0, 0.0]),
        (1, [0.0, 1.43, 1.11]),
        (1, [0.0, -1.43, 1.11]),
    ]
}

fn pyscf_spec_with_partition(partition: PartitionConfig) -> GridSpec {
    PyscfGridSpecBuilder::default()
        .level(1)
        .partition(partition)
        .build()
        .unwrap()
        .into()
}

fn pyscf_spec() -> GridSpec {
    pyscf_spec_with_partition(PartitionConfig::new(
        PartitionScheme::Becke,
        RadiiConfig::pyscf(),
    ))
}

fn alternate_spec() -> GridSpec {
    PyscfGridSpecBuilder::default()
        .level(2)
        .partition(PartitionConfig::pyscf())
        .build()
        .unwrap()
        .into()
}

fn evaluator(centers: &[[f64; 3]], charges: &[u32], spec: &GridSpec) -> PartitionEvaluator {
    PartitionEvaluator::from_parts(centers, charges, spec.partition().clone()).unwrap()
}

fn assert_close(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= tol, "index {i}: {x} vs {y}");
    }
}

#[test]
fn batch_ranges_match_full_build() {
    let atoms = h2o();
    let spec = pyscf_spec();
    let plan = GridPlan::build(&atoms, &spec).unwrap();
    let full = build_molecular_grid(&atoms, &spec).unwrap();

    assert_eq!(plan.len(), full.len());
    assert!(plan.len() > 1000);

    let centers: Vec<[f64; 3]> = atoms.iter().map(|a| a.1).collect();
    let charges: Vec<u32> = atoms.iter().map(|a| a.0).collect();
    let ev = evaluator(&centers, &charges, &spec);
    let mut ws = Workspace::new();
    let mut buf = BatchBuffer::new();

    ev.fill_plan(&plan, 0..plan.len(), &mut ws, &mut buf)
        .unwrap();
    assert_eq!(buf.len(), full.len());
    assert_close(buf.weights(), full.weights(), 1e-12);
    assert_close(buf.quad_weights(), full.quad_weights(), 0.0);
    assert_eq!(buf.atom(), full.atom_idx());
    for (a, b) in buf.coords().iter().zip(full.coords()) {
        for t in 0..3 {
            assert_eq!(a[t], b[t]);
        }
    }

    // Assemble the same weights from arbitrary ranges, including ranges that
    // cross atom boundaries.
    let ranges = [0..1, 1..7, 7..plan.len() / 2, plan.len() / 2..plan.len()];
    let mut weights = Vec::with_capacity(plan.len());
    let mut quads = Vec::with_capacity(plan.len());
    let mut atoms_idx = Vec::with_capacity(plan.len());
    for r in ranges {
        ev.fill_plan(&plan, r.clone(), &mut ws, &mut buf).unwrap();
        assert_eq!(buf.len(), r.len());
        weights.extend_from_slice(buf.weights());
        quads.extend_from_slice(buf.quad_weights());
        atoms_idx.extend_from_slice(buf.atom());
    }
    assert_close(&weights, full.weights(), 1e-12);
    assert_close(&quads, full.quad_weights(), 0.0);
    assert_eq!(atoms_idx, full.atom_idx());
}

#[test]
fn range_crossing_atom_boundary_matches_full() {
    let atoms = h2o();
    let spec = pyscf_spec();
    let plan = GridPlan::build(&atoms, &spec).unwrap();
    let centers: Vec<[f64; 3]> = atoms.iter().map(|a| a.1).collect();
    let charges: Vec<u32> = atoms.iter().map(|a| a.0).collect();
    let ev = evaluator(&centers, &charges, &spec);
    let mut ws = Workspace::new();
    let mut buf = BatchBuffer::new();

    let cut = plan.atom_range(0).end;
    let range = cut.saturating_sub(2)..(cut + 2).min(plan.len());
    ev.fill_plan(&plan, range.clone(), &mut ws, &mut buf)
        .unwrap();
    assert_eq!(buf.len(), range.len());
    assert!(buf.atom().windows(2).any(|w| w[0] != w[1]));

    let full = build_molecular_grid(&atoms, &spec).unwrap();
    assert_close(buf.weights(), &full.weights()[range.clone()], 1e-12);
    assert_close(buf.quad_weights(), &full.quad_weights()[range.clone()], 0.0);
    assert_eq!(buf.atom(), &full.atom_idx()[range]);
}

#[test]
fn batches_cover_the_plan() {
    let atoms = h2o();
    let plan = GridPlan::build(&atoms, pyscf_spec()).unwrap();
    let total: usize = plan.batches(1000).map(|r| r.len()).sum();
    assert_eq!(total, plan.len());
    assert!(plan.batches(1000).all(|r| r.len() <= 1000));
}

#[test]
fn partition_can_change_without_rebuilding_topology() {
    let atoms = h2o();
    let base = pyscf_spec();
    let lko = pyscf_spec_with_partition(PartitionConfig::new(
        PartitionScheme::Lko,
        RadiiConfig::pyscf(),
    ));
    assert_eq!(base.partition().scheme, PartitionScheme::Becke);
    assert_eq!(lko.partition().scheme, PartitionScheme::Lko);

    // One plan, two evaluators: the topology does not depend on the partition.
    let plan = GridPlan::build(&atoms, &base).unwrap();
    let centers: Vec<[f64; 3]> = atoms.iter().map(|a| a.1).collect();
    let charges: Vec<u32> = atoms.iter().map(|a| a.0).collect();
    let ev_b = evaluator(&centers, &charges, &base);
    let ev_l = evaluator(&centers, &charges, &lko);
    let mut ws = Workspace::new();
    let mut wb = BatchBuffer::new();
    let mut wl = BatchBuffer::new();
    ev_b.fill_plan(&plan, 0..plan.len(), &mut ws, &mut wb)
        .unwrap();
    ev_l.fill_plan(&plan, 0..plan.len(), &mut ws, &mut wl)
        .unwrap();
    let diff = wb
        .weights()
        .iter()
        .zip(wl.weights())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(diff > 1e-6, "LKO and Becke weights should differ");
}

#[test]
fn plan_atom_count_must_match_evaluator() {
    let atoms = h2o();
    let spec = pyscf_spec();
    let plan = GridPlan::build(&atoms, &spec).unwrap();
    let centers: Vec<[f64; 3]> = atoms[..2].iter().map(|a| a.1).collect();
    let charges: Vec<u32> = atoms[..2].iter().map(|a| a.0).collect();
    let ev = evaluator(&centers, &charges, &spec);
    let mut ws = Workspace::new();
    let mut buf = BatchBuffer::new();
    let err = ev
        .fill_plan(&plan, 0..plan.len(), &mut ws, &mut buf)
        .unwrap_err();
    assert_eq!(
        err,
        ruaa_dft_grids::Error::LengthMismatch {
            expected: 2,
            got: 3,
        }
    );
}

#[test]
fn molecular_grid_stores_the_spec() {
    let atoms = h2o();
    for spec in [pyscf_spec(), alternate_spec()] {
        let g = build_molecular_grid(&atoms, &spec).unwrap();
        assert_eq!(*g.spec(), spec);
    }
}

#[test]
fn custom_atomic_plans_feed_partition_evaluator() {
    let first = AtomicGrid::new(
        1,
        vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        vec![1.0, 2.0],
        vec![1, 1],
    )
    .unwrap();
    let second = AtomicGrid::new(1, vec![[0.0, 0.0, 0.0]], vec![3.0], vec![1]).unwrap();
    let plan = GridPlan::from_atomic_grids(vec![first, second]).unwrap();

    let centers = [[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]];
    let charges = [1u32, 1];
    let config = PartitionConfig::new(PartitionScheme::Becke, RadiiConfig::pyscf());
    let evaluator = PartitionEvaluator::from_parts(&centers, &charges, config).unwrap();
    let mut workspace = Workspace::new();
    let mut buffer = BatchBuffer::new();
    evaluator
        .fill_plan(&plan, 0..plan.len(), &mut workspace, &mut buffer)
        .unwrap();

    assert_eq!(buffer.len(), 3);
    assert_eq!(buffer.atom(), &[0u32, 0, 1]);
    assert_eq!(buffer.quad_weights(), &[1.0, 2.0, 3.0]);
    assert!(buffer.weights().iter().all(|weight| *weight >= 0.0));
}
