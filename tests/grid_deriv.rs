//! Grid-native partition derivatives.
//!
//! A [`MolecularGrid`] already stores global coordinates, generating atoms and
//! quadrature weights, so it can be differentiated without a [`GridPlan`].
//! These tests pin:
//!
//! * bitwise agreement with the plan-based path on an untouched grid;
//! * current-slot evaluation after [`MolecularGrid::reorder`];
//! * retained-only evaluation after a post-build prune;
//! * mixed generating atoms inside one block;
//! * thread-count invariance and the grid-shape errors.
use ruaa_dft_grids::partition::DerivPoints;

use ruaa_dft_grids::Error;
use ruaa_dft_grids::grid::{GridPlan, GridSpec, PyscfGridSpecBuilder};
use ruaa_dft_grids::moleculegrid::{MolecularGrid, PostPrune, build_molecular_grid};
use ruaa_dft_grids::order::GridOrder;
use ruaa_dft_grids::partition::{
    DerivContract, DerivOptions, DerivOutput, DerivRequest, DerivWorkspace, PartitionDerivEvaluator,
};

fn water() -> Vec<(u32, [f64; 3])> {
    vec![
        (8, [0.0, 0.0, 0.0]),
        (1, [0.0, 1.43, 1.11]),
        (1, [0.0, -1.43, 1.11]),
    ]
}

fn spec() -> GridSpec {
    PyscfGridSpecBuilder::default()
        .level(0)
        .build()
        .unwrap()
        .into()
}

fn bits_eq(got: &[f64], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{what}[{i}]: got {a:?}, want {b:?}"
        );
    }
}

fn plan_evaluator(
    centers: &[[f64; 3]],
    charges: &[u32],
    spec: &GridSpec,
) -> PartitionDerivEvaluator {
    PartitionDerivEvaluator::from_parts(
        centers,
        charges,
        spec.partition().clone(),
        DerivOptions::default(),
    )
    .unwrap()
}

#[test]
fn grid_path_matches_plan_path() {
    let atoms = water();
    let spec = spec();
    let plan = GridPlan::build(&atoms, &spec).unwrap();
    let grid = build_molecular_grid(&atoms, &spec).unwrap();
    let charges: Vec<u32> = atoms.iter().map(|a| a.0).collect();
    let centers: Vec<[f64; 3]> = atoms.iter().map(|a| a.1).collect();

    let ev_grid = grid
        .partition_deriv_evaluator(DerivOptions::default())
        .unwrap();
    let ev_plan = plan_evaluator(&centers, &charges, &spec);
    let req = DerivRequest {
        weights: true,
        dw: true,
        ddw: true,
        ..Default::default()
    };

    let mut out_grid = DerivOutput::new();
    ev_grid
        .fill(DerivPoints::all_grid(&grid), &req, &mut out_grid)
        .unwrap();

    let mut out_plan = DerivOutput::new();
    let mut ws = DerivWorkspace::new();
    ev_plan
        .fill_serial(
            DerivPoints::plan(&plan, 0..plan.len()),
            &mut ws,
            &req,
            &mut out_plan,
        )
        .unwrap();

    assert_eq!(out_grid.range(), 0..grid.len());
    bits_eq(
        out_grid.partition_weights(),
        out_plan.partition_weights(),
        "weights",
    );
    bits_eq(out_grid.dw(), out_plan.dw(), "dw");
    bits_eq(out_grid.ddw(), out_plan.ddw(), "ddw");
}

#[test]
fn reordered_grid_matches_per_point_evaluation() {
    let atoms = water();
    let spec = spec();
    let mut grid = build_molecular_grid(&atoms, &spec).unwrap();
    let order = GridOrder::boxes(grid.coords(), 1.2).unwrap();
    grid.reorder(&order).unwrap();

    let ev = grid
        .partition_deriv_evaluator(DerivOptions {
            block_points: 8,
            ..Default::default()
        })
        .unwrap();
    let req = DerivRequest {
        weights: true,
        dw: true,
        ..Default::default()
    };
    let mut out = DerivOutput::new();
    ev.fill(DerivPoints::all_grid(&grid), &req, &mut out)
        .unwrap();

    let g = grid.len();
    let step = (g / 11).max(1);
    for point in (0..g).step_by(step) {
        let parent = grid.atom_idx()[point] as usize;
        let quad = grid.quad_weights()[point];
        let (weight, dw) = ev.gradient_at(grid.coords()[point], parent);
        assert!((out.partition_weights()[point] - weight).abs() < 1e-12);
        assert!((out.partition_weights()[point] - grid.partition_fractions()[point]).abs() < 1e-12);
        for (row, &value) in dw.iter().enumerate() {
            let got = out.dw()[row * g + point];
            let want = quad * value;
            assert!(
                (got - want).abs() < 1e-12,
                "point {point} row {row}: got {got:?}, want {want:?}"
            );
        }
    }
}

/// A deterministic Fisher-Yates shuffle of `0..len`.
fn shuffled_order(len: usize, mut state: u64) -> GridOrder {
    let mut perm: Vec<u32> = (0..len as u32).collect();
    for i in (1..len).rev() {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let j = ((state >> 33) as usize) % (i + 1);
        perm.swap(i, j);
    }
    GridOrder::from_permutation(&perm).unwrap()
}

#[test]
fn randomly_reordered_grid_matches_sorted_grid() {
    let atoms = water();
    let spec = spec();
    let sorted = build_molecular_grid(&atoms, &spec).unwrap();
    let order = shuffled_order(sorted.len(), 0x5EED_1234);
    let mut permuted = sorted.clone();
    permuted.reorder(&order).unwrap();

    let g = sorted.len();
    let natm = atoms.len();
    let coefficients = vec![1.0; g];
    let req = DerivRequest {
        weights: true,
        dw: true,
        ddw: true,
        dc: Some(DerivContract::exact(&coefficients, 1)),
        ddc: Some(DerivContract::exact(&coefficients, 1)),
    };
    let ev = sorted
        .partition_deriv_evaluator(DerivOptions::default())
        .unwrap();
    let mut want = DerivOutput::new();
    let mut got = DerivOutput::new();
    ev.fill(DerivPoints::all_grid(&sorted), &req, &mut want)
        .unwrap();
    ev.fill(DerivPoints::all_grid(&permuted), &req, &mut got)
        .unwrap();

    // Per-point outputs are point-local: mapping the permuted slots back must
    // reproduce the sorted evaluation bit for bit.
    let perm = order.perm();
    for (slot, &source) in perm.iter().enumerate() {
        let source = source as usize;
        assert_eq!(
            got.partition_weights()[slot].to_bits(),
            want.partition_weights()[source].to_bits()
        );
    }
    for row in 0..natm * 3 {
        for (slot, &source) in perm.iter().enumerate() {
            assert_eq!(
                got.dw()[row * g + slot].to_bits(),
                want.dw()[row * g + source as usize].to_bits()
            );
        }
    }
    let cols = natm * 3 * natm * 3;
    for c in 0..cols {
        for (slot, &source) in perm.iter().enumerate() {
            assert_eq!(
                got.ddw()[c * g + slot].to_bits(),
                want.ddw()[c * g + source as usize].to_bits()
            );
        }
    }

    // Contracts sum over points in slot order, so a permutation changes the
    // rounding order but not the value.  The tolerance is relative to the
    // accumulated absolute per-point terms, which is the natural error scale
    // when a sum cancels.
    let ddc = want.ddc().unwrap().ddc();
    let ddc_perm = got.ddc().unwrap().ddc();
    for (c, (&y, &x)) in ddc.iter().zip(ddc_perm).enumerate() {
        let abs_sum: f64 = (0..g).map(|p| want.ddw()[c * g + p].abs()).sum();
        assert!(
            (x - y).abs() <= 1e-12 * abs_sum,
            "ddc[{c}]: {x} vs {y} (abs sum {abs_sum})"
        );
    }
    let dc = want.dc().unwrap().dc();
    let dc_perm = got.dc().unwrap().dc();
    for (c, (&y, &x)) in dc.iter().zip(dc_perm).enumerate() {
        let abs_sum: f64 = (0..g).map(|p| want.dw()[c * g + p].abs()).sum();
        assert!(
            (x - y).abs() <= 1e-12 * abs_sum,
            "dc[{c}]: {x} vs {y} (abs sum {abs_sum})"
        );
    }
}

#[test]
fn pruned_grid_evaluates_only_retained_points() {
    let atoms = water();
    let spec = spec();
    let plan = GridPlan::build(&atoms, &spec).unwrap();
    let full = build_molecular_grid(&atoms, &spec).unwrap();
    let keep: Vec<bool> = (0..full.len()).map(|i| i % 3 != 1).collect();
    let retained = full
        .clone()
        .split(PostPrune::KeepMask(keep.clone()))
        .unwrap()
        .retained;

    let charges: Vec<u32> = atoms.iter().map(|a| a.0).collect();
    let centers: Vec<[f64; 3]> = atoms.iter().map(|a| a.1).collect();
    let req = DerivRequest {
        weights: true,
        dw: true,
        ..Default::default()
    };

    let mut out = DerivOutput::new();
    retained
        .partition_deriv_evaluator(DerivOptions::default())
        .unwrap()
        .fill(DerivPoints::all_grid(&retained), &req, &mut out)
        .unwrap();

    let mut full_out = DerivOutput::new();
    let mut ws = DerivWorkspace::new();
    plan_evaluator(&centers, &charges, &spec)
        .fill_serial(
            DerivPoints::plan(&plan, 0..plan.len()),
            &mut ws,
            &req,
            &mut full_out,
        )
        .unwrap();

    let source: Vec<usize> = keep
        .iter()
        .enumerate()
        .filter_map(|(i, &k)| k.then_some(i))
        .collect();
    let m = retained.len();
    assert_eq!(source.len(), m);
    let rows = out.dw().len() / m;
    for (slot, &point) in source.iter().enumerate() {
        assert_eq!(
            out.partition_weights()[slot].to_bits(),
            full_out.partition_weights()[point].to_bits()
        );
        for row in 0..rows {
            assert_eq!(
                out.dw()[row * m + slot].to_bits(),
                full_out.dw()[row * plan.len() + point].to_bits()
            );
        }
    }
}

#[test]
fn mixed_generating_atoms_in_one_block() {
    let n = 9;
    let molecule =
        ruaa_dft_grids::molecule::Molecule::new([(1u32, [0.0, 0.0, 0.0]), (1, [0.0, 0.0, 3.0])])
            .unwrap();
    let grid = MolecularGrid::from_parts(
        spec(),
        molecule,
        vec![
            [0.1, 0.0, 0.0],
            [0.0, 0.0, 3.1],
            [0.2, 0.0, 0.0],
            [0.0, 0.0, 3.2],
            [0.3, 0.0, 0.0],
            [0.0, 0.0, 3.3],
            [0.4, 0.0, 0.0],
            [0.0, 0.0, 3.4],
            [0.5, 0.0, 0.0],
        ],
        vec![0.0; n],
        vec![1.0; n],
        vec![0.0; n],
        vec![0, 1, 0, 1, 0, 1, 0, 1, 0],
    )
    .unwrap();
    let ev = grid
        .partition_deriv_evaluator(DerivOptions {
            block_points: 8,
            ..Default::default()
        })
        .unwrap();

    let mut dw_out = DerivOutput::new();
    ev.fill(
        DerivPoints::all_grid(&grid),
        &DerivRequest {
            dw: true,
            ..Default::default()
        },
        &mut dw_out,
    )
    .unwrap();

    let mut ddw_out = DerivOutput::new();
    ev.fill(
        DerivPoints::all_grid(&grid),
        &DerivRequest {
            ddw: true,
            ..Default::default()
        },
        &mut ddw_out,
    )
    .unwrap();

    for point in 0..n {
        let parent = grid.atom_idx()[point] as usize;
        let (_, dw) = ev.gradient_at(grid.coords()[point], parent);
        let ddw = ev.hessian_at(grid.coords()[point], parent);
        for (row, &want) in dw.iter().enumerate() {
            assert_eq!(dw_out.dw()[row * n + point].to_bits(), want.to_bits());
        }
        for (k, &want) in ddw.iter().enumerate() {
            assert_eq!(ddw_out.ddw()[k * n + point].to_bits(), want.to_bits());
        }
    }
}

#[test]
fn grid_contraction_sums_the_retained_hessian() {
    let atoms = water();
    let spec = spec();
    let grid = build_molecular_grid(&atoms, &spec).unwrap();
    let ev = grid
        .partition_deriv_evaluator(DerivOptions::default())
        .unwrap();

    let coefficients = vec![1.0; grid.len()];
    let mut contracted = DerivOutput::new();
    ev.fill(
        DerivPoints::all_grid(&grid),
        &DerivRequest {
            ddc: Some(DerivContract::exact(&coefficients, 1)),
            ..Default::default()
        },
        &mut contracted,
    )
    .unwrap();
    let ddc = contracted.ddc().unwrap().ddc();

    let mut raw = DerivOutput::new();
    ev.fill(
        DerivPoints::all_grid(&grid),
        &DerivRequest {
            ddw: true,
            ..Default::default()
        },
        &mut raw,
    )
    .unwrap();

    let g = grid.len();
    for (component, &value) in ddc.iter().enumerate() {
        let sum: f64 = raw.ddw()[component * g..(component + 1) * g].iter().sum();
        assert!(
            (value - sum).abs() < 1e-10,
            "component {component}: contracted {value:?}, summed {sum:?}"
        );
    }
}

#[test]
fn grid_fill_is_thread_count_invariant() {
    let atoms = water();
    let spec = spec();
    let grid = build_molecular_grid(&atoms, &spec).unwrap();
    let ev = grid
        .partition_deriv_evaluator(DerivOptions::default())
        .unwrap();
    let req = DerivRequest {
        weights: true,
        dw: true,
        ddw: true,
        ..Default::default()
    };

    let mut many = DerivOutput::new();
    ev.fill(DerivPoints::all_grid(&grid), &req, &mut many)
        .unwrap();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let mut one = DerivOutput::new();
    pool.install(|| ev.fill(DerivPoints::all_grid(&grid), &req, &mut one))
        .unwrap();

    bits_eq(one.partition_weights(), many.partition_weights(), "weights");
    bits_eq(one.dw(), many.dw(), "dw");
    bits_eq(one.ddw(), many.ddw(), "ddw");
}

#[test]
fn grid_shape_errors_are_reported() {
    let atoms = water();
    let spec = spec();
    let grid = build_molecular_grid(&atoms, &spec).unwrap();
    let ev = grid
        .partition_deriv_evaluator(DerivOptions::default())
        .unwrap();
    let req = DerivRequest {
        dw: true,
        ..Default::default()
    };
    let mut out = DerivOutput::new();

    let mut shifted = atoms;
    shifted[0].1[0] += 0.1;
    let moved = build_molecular_grid(&shifted, &spec).unwrap();
    assert_eq!(
        ev.fill(DerivPoints::all_grid(&moved), &req, &mut out)
            .unwrap_err(),
        Error::GeometryMismatch
    );

    let empty = MolecularGrid::from_parts(
        spec.clone(),
        ruaa_dft_grids::molecule::Molecule::new([(1u32, [0.0, 0.0, 0.0])]).unwrap(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    )
    .unwrap();
    let ev_empty = empty
        .partition_deriv_evaluator(DerivOptions::default())
        .unwrap();
    let mut out_empty = DerivOutput::new();
    ev_empty
        .fill(DerivPoints::all_grid(&empty), &req, &mut out_empty)
        .unwrap();
    assert!(out_empty.dw().is_empty());
}
