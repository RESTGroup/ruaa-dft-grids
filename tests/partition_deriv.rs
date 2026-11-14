//! Numerical-difference validation of the analytic Becke partition
//! derivatives.
//!
//! The finite-difference reference perturbs the nuclear coordinates and
//! rebuilds the evaluator; the grid point is expressed in the **generating
//! atom's local frame**, so it translates rigidly with that atom.  The
//! resulting finite difference is therefore the true total derivative
//! (including the grid shift), which is exactly what the analytic kernels
//! return after their translation correction.
//!
//! These tests are the executable form of the workflow documented in the
//! `ruaa_dft_grids::doc::partition_derivatives` page.
use ruaa_dft_grids::partition::DerivPoints;
use ruaa_dft_grids::partition::deriv::MAX_BLOCK_POINTS;

use ruaa_dft_grids::grid::{GridPlan, GridSpec, PyscfGridSpecBuilder};
use ruaa_dft_grids::partition::{
    DerivContract, DerivOptions, DerivOutput, DerivRequest, DerivWorkspace, HessianBackend,
    PartitionConfig, PartitionDerivEvaluator, PartitionScheme, RadiiConfig,
};

fn water() -> (Vec<u32>, Vec<[f64; 3]>) {
    let atoms: Vec<(u32, [f64; 3])> = vec![
        (8, [0.0, 0.0, 0.0]),
        (1, [0.0, 1.43, 1.11]),
        (1, [0.0, -1.43, 1.11]),
    ];
    (
        atoms.iter().map(|a| a.0).collect(),
        atoms.iter().map(|a| a.1).collect(),
    )
}

fn benzene() -> (Vec<u32>, Vec<[f64; 3]>) {
    // D6h benzene, C-C 1.39 A, C-H 1.09 A, in Bohr.
    let cc = 1.39 / 0.52917721092;
    let ch = 1.09 / 0.52917721092;
    let mut charges = Vec::new();
    let mut centers = Vec::new();
    for k in 0..6 {
        let a = std::f64::consts::TAU * k as f64 / 6.0;
        charges.push(6u32);
        centers.push([cc * a.cos(), cc * a.sin(), 0.0]);
        charges.push(1u32);
        centers.push([(cc + ch) * a.cos(), (cc + ch) * a.sin(), 0.0]);
    }
    (charges, centers)
}

fn h_ring(n: usize, radius: f64) -> (Vec<u32>, Vec<[f64; 3]>) {
    let charges = vec![1u32; n];
    let centers = (0..n)
        .map(|k| {
            let a = std::f64::consts::TAU * k as f64 / n as f64;
            [radius * a.cos(), radius * a.sin(), 0.0]
        })
        .collect();
    (charges, centers)
}

fn plan_for(charges: &[u32], centers: &[[f64; 3]]) -> GridPlan {
    let atoms: Vec<(u32, [f64; 3])> = charges
        .iter()
        .copied()
        .zip(centers.iter().copied())
        .collect();
    let spec: GridSpec = PyscfGridSpecBuilder::default()
        .level(0)
        .build()
        .unwrap()
        .into();
    GridPlan::build(&atoms, &spec).unwrap()
}

fn config() -> PartitionConfig {
    PartitionConfig::new(PartitionScheme::Becke, RadiiConfig::pyscf())
}

fn stratmann_config() -> PartitionConfig {
    PartitionConfig::new(PartitionScheme::Stratmann, RadiiConfig::pyscf())
}

fn gaussian_ss_config() -> PartitionConfig {
    PartitionConfig::gaussian_ss()
}

fn lko_config() -> PartitionConfig {
    PartitionConfig::new(PartitionScheme::Lko, RadiiConfig::pyscf())
}

/// mBecke with ORCA radii and the default clamp: the derivative backend is
/// the classic Becke one (the value paths agree up to one ulp).
fn mbecke_config() -> PartitionConfig {
    PartitionConfig::new(PartitionScheme::MBecke, RadiiConfig::pyscf())
}

fn evaluator(
    centers: &[[f64; 3]],
    charges: &[u32],
    atoms: Option<Vec<usize>>,
) -> PartitionDerivEvaluator {
    evaluator_with(centers, charges, config(), atoms)
}

fn evaluator_with(
    centers: &[[f64; 3]],
    charges: &[u32],
    cfg: PartitionConfig,
    atoms: Option<Vec<usize>>,
) -> PartitionDerivEvaluator {
    PartitionDerivEvaluator::from_parts(
        centers,
        charges,
        cfg,
        DerivOptions {
            atoms,
            block_points: 0,
            worker_memory_bytes: 0,
            hessian_backend: HessianBackend::Auto,
        },
    )
    .unwrap()
}

/// Weight at a grid point whose local coordinate (relative to the
/// generating atom) is fixed while the centres move.
fn weight_at(centers: &[[f64; 3]], charges: &[u32], parent: usize, local: [f64; 3]) -> f64 {
    weight_at_scheme(centers, charges, parent, local, config())
}

fn weight_at_scheme(
    centers: &[[f64; 3]],
    charges: &[u32],
    parent: usize,
    local: [f64; 3],
    cfg: PartitionConfig,
) -> f64 {
    let ev = evaluator_with(centers, charges, cfg, None);
    let point = [
        centers[parent][0] + local[0],
        centers[parent][1] + local[1],
        centers[parent][2] + local[2],
    ];
    ev.gradient_at(point, parent).0
}

fn gradient_at(centers: &[[f64; 3]], charges: &[u32], parent: usize, local: [f64; 3]) -> Vec<f64> {
    gradient_at_scheme(centers, charges, parent, local, config())
}

fn gradient_at_scheme(
    centers: &[[f64; 3]],
    charges: &[u32],
    parent: usize,
    local: [f64; 3],
    cfg: PartitionConfig,
) -> Vec<f64> {
    let ev = evaluator_with(centers, charges, cfg, None);
    let point = [
        centers[parent][0] + local[0],
        centers[parent][1] + local[1],
        centers[parent][2] + local[2],
    ];
    ev.gradient_at(point, parent).1
}

fn perturbed(centers: &[[f64; 3]], atom: usize, comp: usize, h: f64) -> Vec<[f64; 3]> {
    let mut c = centers.to_vec();
    c[atom][comp] += h;
    c
}

/// Sample points: one point from each atom's grid, plus a point from the
/// middle of the first atom's grid.
fn sample_points(plan: &GridPlan, per_atom: usize) -> Vec<(usize, [f64; 3])> {
    let mut out = Vec::new();
    for ia in 0..plan.atom_count() {
        let range = plan.atom_range(ia);
        let count = per_atom.min(range.len());
        for k in 0..count {
            let g = range.start + (k * range.len()) / count.max(1);
            out.push((ia, plan.local_coord(g)));
        }
    }
    out
}

#[test]
fn gradient_matches_total_finite_difference() {
    for (charges, centers) in [water(), benzene()] {
        let plan = plan_for(&charges, &centers);
        let ev = evaluator(&centers, &charges, None);
        let h = 1e-5;
        let points = sample_points(&plan, 2);
        for (parent, local) in points {
            let (_, dw) = ev.gradient_at(
                [
                    centers[parent][0] + local[0],
                    centers[parent][1] + local[1],
                    centers[parent][2] + local[2],
                ],
                parent,
            );
            for atom in 0..centers.len() {
                for comp in 0..3 {
                    let cp = perturbed(&centers, atom, comp, h);
                    let cm = perturbed(&centers, atom, comp, -h);
                    let fd = (weight_at(&cp, &charges, parent, local)
                        - weight_at(&cm, &charges, parent, local))
                        / (2.0 * h);
                    let an = dw[atom * 3 + comp];
                    assert!(
                        (fd - an).abs() < 2e-6,
                        "gradient: n={} parent={parent} atom={atom} comp={comp} fd={fd} an={an}",
                        centers.len()
                    );
                }
            }
        }
    }
}

#[test]
fn gradient_sums_to_zero_over_atoms() {
    let (charges, centers) = benzene();
    let plan = plan_for(&charges, &centers);
    let ev = evaluator(&centers, &charges, None);
    for (parent, local) in sample_points(&plan, 2) {
        let point = [
            centers[parent][0] + local[0],
            centers[parent][1] + local[1],
            centers[parent][2] + local[2],
        ];
        let (_, dw) = ev.gradient_at(point, parent);
        let n = centers.len();
        for t in 0..3 {
            let sum: f64 = (0..n).map(|a| dw[a * 3 + t]).sum();
            assert!(sum.abs() < 1e-11, "translation sum {t} = {sum}");
        }
    }
}

#[test]
fn hessian_matches_gradient_finite_difference() {
    for (charges, centers) in [water(), benzene()] {
        let plan = plan_for(&charges, &centers);
        let ev = evaluator(&centers, &charges, None);
        let n = centers.len();
        let n3 = 3 * n;
        let h = 2e-5;
        for (parent, local) in sample_points(&plan, 1) {
            let point = [
                centers[parent][0] + local[0],
                centers[parent][1] + local[1],
                centers[parent][2] + local[2],
            ];
            let hess = ev.hessian_at(point, parent);
            for atom in 0..n {
                for comp in 0..3 {
                    let cp = perturbed(&centers, atom, comp, h);
                    let cm = perturbed(&centers, atom, comp, -h);
                    let gp = gradient_at(&cp, &charges, parent, local);
                    let gm = gradient_at(&cm, &charges, parent, local);
                    for b in 0..n {
                        for s in 0..3 {
                            let c = b * 3 + s;
                            let fd = (gp[c] - gm[c]) / (2.0 * h);
                            let an = hess[(atom * 3 + comp) * n3 + c];
                            assert!(
                                (fd - an).abs() < 2e-5,
                                "hessian: parent={parent} A={atom} t={comp} B={b} s={s} fd={fd} an={an}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn hessian_is_symmetric_and_translation_invariant() {
    let (charges, centers) = water();
    let plan = plan_for(&charges, &centers);
    let ev = evaluator(&centers, &charges, None);
    let n = centers.len();
    let n3 = 3 * n;
    for (parent, local) in sample_points(&plan, 1) {
        let point = [
            centers[parent][0] + local[0],
            centers[parent][1] + local[1],
            centers[parent][2] + local[2],
        ];
        let h = ev.hessian_at(point, parent);
        for a in 0..n {
            for t in 0..3 {
                for b in 0..n {
                    for s in 0..3 {
                        let ab = h[(a * 3 + t) * n3 + b * 3 + s];
                        let ba = h[(b * 3 + s) * n3 + a * 3 + t];
                        assert!(
                            (ab - ba).abs() < 1e-10,
                            "asymmetry A={a} t={t} B={b} s={s}: {ab} vs {ba}"
                        );
                    }
                }
            }
        }
        for t in 0..3 {
            for b in 0..n {
                for s in 0..3 {
                    let sum: f64 = (0..n).map(|a| h[(a * 3 + t) * n3 + b * 3 + s]).sum();
                    assert!(sum.abs() < 1e-10, "hessian translation sum = {sum}");
                }
            }
        }
    }
}

#[test]
fn selection_is_exactly_a_row_subset() {
    let (charges, centers) = benzene();
    let plan = plan_for(&charges, &centers);
    let n = centers.len();
    let all = evaluator(&centers, &charges, None);
    let selected = vec![0usize, 3, 7];
    let sub = evaluator(&centers, &charges, Some(selected.clone()));
    for (parent, local) in sample_points(&plan, 1) {
        let point = [
            centers[parent][0] + local[0],
            centers[parent][1] + local[1],
            centers[parent][2] + local[2],
        ];
        let (w_all, dw_all) = all.gradient_at(point, parent);
        let (w_sub, dw_sub) = sub.gradient_at(point, parent);
        assert_eq!(w_all.to_bits(), w_sub.to_bits());
        for (slot, &atom) in selected.iter().enumerate() {
            for t in 0..3 {
                assert_eq!(
                    dw_all[atom * 3 + t].to_bits(),
                    dw_sub[slot * 3 + t].to_bits(),
                    "gradient row mismatch atom={atom} t={t}"
                );
            }
        }
        let h_all = all.hessian_at(point, parent);
        let h_sub = sub.hessian_at(point, parent);
        let n3 = 3 * n;
        for (slot, &atom) in selected.iter().enumerate() {
            for t in 0..3 {
                for c in 0..n3 {
                    assert_eq!(
                        h_all[(atom * 3 + t) * n3 + c].to_bits(),
                        h_sub[(slot * 3 + t) * n3 + c].to_bits(),
                        "hessian row mismatch atom={atom} t={t} c={c}"
                    );
                }
            }
        }
    }
}

#[test]
fn small_selection_matches_full_hessian_rows() {
    let (charges, centers) = water();
    let plan = plan_for(&charges, &centers);
    let n3 = 3 * centers.len();
    let all = evaluator(&centers, &charges, None);
    let selected = vec![0usize, 2];
    let sub = evaluator(&centers, &charges, Some(selected.clone()));
    for (parent, local) in sample_points(&plan, 1) {
        let point = [
            centers[parent][0] + local[0],
            centers[parent][1] + local[1],
            centers[parent][2] + local[2],
        ];
        let h_all = all.hessian_at(point, parent);
        let h_sub = sub.hessian_at(point, parent);
        for (slot, &atom) in selected.iter().enumerate() {
            for t in 0..3 {
                for c in 0..n3 {
                    assert_eq!(
                        h_all[(atom * 3 + t) * n3 + c].to_bits(),
                        h_sub[(slot * 3 + t) * n3 + c].to_bits(),
                        "hessian row mismatch atom={atom} t={t} c={c}"
                    );
                }
            }
        }
    }
}

#[test]
fn hessian_backend_override_selects_kernel() {
    let (charges, centers) = water();
    let plan = plan_for(&charges, &centers);
    let make = |backend| {
        PartitionDerivEvaluator::from_parts(
            &centers,
            &charges,
            config(),
            DerivOptions {
                atoms: None,
                block_points: 0,
                worker_memory_bytes: 0,
                hessian_backend: backend,
            },
        )
        .unwrap()
    };
    let lane = make(HessianBackend::Lane);
    let pointwise = make(HessianBackend::Pointwise);
    assert!(!lane.scheme().hessian_pointwise());
    assert!(pointwise.scheme().hessian_pointwise());
    assert_ne!(
        lane.scheme().hessian_scratch_bytes(),
        pointwise.scheme().hessian_scratch_bytes()
    );
    for (parent, local) in sample_points(&plan, 1) {
        let point = [
            centers[parent][0] + local[0],
            centers[parent][1] + local[1],
            centers[parent][2] + local[2],
        ];
        let a = lane.hessian_at(point, parent);
        let b = pointwise.hessian_at(point, parent);
        assert_eq!(a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() <= 1e-12 * (1.0 + y.abs()),
                "hessian mismatch at {i}: {x} vs {y}"
            );
        }
    }
}

#[test]
fn streaming_contract_matches_full_blocks() {
    let (charges, centers) = water();
    let plan = plan_for(&charges, &centers);
    let ev = evaluator(&centers, &charges, None);
    let n = centers.len();
    let n3 = 3 * n;
    let range = 0..plan.len().min(40);
    let contract: Vec<f64> = (0..range.len()).map(|k| 0.3 + k as f64 * 0.01).collect();

    let mut out = DerivOutput::new();
    let mut ws = DerivWorkspace::new();
    let req = DerivRequest {
        ddc: Some(DerivContract::exact(&contract, 1)),
        ..Default::default()
    };
    ev.fill_serial(
        DerivPoints::plan(&plan, range.clone()),
        &mut ws,
        &req,
        &mut out,
    )
    .unwrap();
    let by_contract = out.ddc().unwrap();

    let mut by_blocks = vec![0.0; n3 * n3];
    let mut ws2 = DerivWorkspace::new();
    let mut raw = DerivOutput::new();
    ev.fill_serial_visit(
        DerivPoints::plan(&plan, range.clone()),
        &mut ws2,
        &DerivRequest::default(),
        &mut raw,
        |g, block| {
            let c = contract[g - range.start];
            for (dst, &src) in by_blocks.iter_mut().zip(block) {
                *dst += c * src;
            }
        },
    )
    .unwrap();

    for (i, (&a, &b)) in by_contract.ddc().iter().zip(&by_blocks).enumerate() {
        assert!(
            (a - b).abs() < 1e-13,
            "contract mismatch at {i}: {a} vs {b}"
        );
    }
}

#[test]
fn rejects_unsupported_schemes() {
    let (charges, centers) = water();
    // mBecke is supported for the default clamp only when the classic pair
    // product is valid; a clamp < 1.0 activates the non-differentiable
    // branches and must be rejected.
    let mut clamped = config();
    clamped.scheme = PartitionScheme::MBecke;
    clamped.mbecke_clamp = 0.5;
    let err =
        PartitionDerivEvaluator::from_parts(&centers, &charges, clamped, DerivOptions::default())
            .unwrap_err();
    assert!(matches!(
        err,
        ruaa_dft_grids::Error::UnsupportedDerivative { .. }
    ));
    let mut cfg = config();
    cfg.scheme = PartitionScheme::Becke;
    cfg.becke_early_exit = true;
    let err = PartitionDerivEvaluator::from_parts(&centers, &charges, cfg, DerivOptions::default())
        .unwrap_err();
    assert!(matches!(
        err,
        ruaa_dft_grids::Error::UnsupportedDerivative { .. }
    ));
}

#[test]
fn gradient_batch_matches_point_api() {
    let (charges, centers) = benzene();
    let plan = plan_for(&charges, &centers);
    let ev = evaluator(&centers, &charges, None);
    let range = 0..plan.len().min(64);
    let mut out = DerivOutput::new();
    let mut ws = DerivWorkspace::new();
    let req = DerivRequest {
        weights: true,
        dw: true,
        ..Default::default()
    };
    ev.fill_serial(
        DerivPoints::plan(&plan, range.clone()),
        &mut ws,
        &req,
        &mut out,
    )
    .unwrap();
    let range_len = range.len();
    for g in range {
        let parent = plan.parent_of(g) as usize;
        let local = plan.local_coord(g);
        let point = [
            centers[parent][0] + local[0],
            centers[parent][1] + local[1],
            centers[parent][2] + local[2],
        ];
        let q = plan.quad_weight(g);
        let (w, dw) = ev.gradient_at(point, parent);
        assert_eq!(out.partition_weights()[g].to_bits(), w.to_bits());
        for (i, &v) in dw.iter().enumerate() {
            let got = out.dw()[i * range_len + g] / q;
            assert!(
                (got - v).abs() <= 1e-13 * (1.0 + v.abs()),
                "point {g} row {i}: {got} vs {v}"
            );
        }
    }
}

#[test]
fn gradient_weights_match_weight_evaluator() {
    use ruaa_dft_grids::grid::BatchBuffer;
    use ruaa_dft_grids::partition::{PartitionEvaluator, Workspace};
    let (charges, centers) = benzene();
    let atoms: Vec<(u32, [f64; 3])> = charges
        .iter()
        .copied()
        .zip(centers.iter().copied())
        .collect();
    let spec: GridSpec = PyscfGridSpecBuilder::default()
        .level(0)
        .build()
        .unwrap()
        .into();
    let plan = GridPlan::build(&atoms, &spec).unwrap();
    let range = 0..plan.len().min(128);
    let ev = evaluator(&centers, &charges, None);
    let mut out = DerivOutput::new();
    let mut ws = DerivWorkspace::new();
    let req = DerivRequest {
        weights: true,
        ..Default::default()
    };
    ev.fill_serial(
        DerivPoints::plan(&plan, range.clone()),
        &mut ws,
        &req,
        &mut out,
    )
    .unwrap();

    let weight_ev = PartitionEvaluator::from_parts(&centers, &charges, config()).unwrap();
    let mut weight_ws = Workspace::new();
    let mut weight_batch = BatchBuffer::new();
    weight_ev
        .fill_plan(&plan, range.clone(), &mut weight_ws, &mut weight_batch)
        .unwrap();
    for (i, g) in range.enumerate() {
        assert_eq!(
            out.partition_weights()[i].to_bits(),
            weight_batch.partition_weights()[i].to_bits(),
            "point {g}"
        );
    }
}

#[test]
fn sorted_point_order_is_per_point_invariant() {
    use ruaa_dft_grids::moleculegrid::build_molecular_grid;
    use ruaa_dft_grids::order::GridOrder;

    let (charges, centers) = water();
    let atoms: Vec<(u32, [f64; 3])> = charges
        .iter()
        .copied()
        .zip(centers.iter().copied())
        .collect();
    let spec: GridSpec = PyscfGridSpecBuilder::default()
        .level(0)
        .build()
        .unwrap()
        .into();
    let grid = build_molecular_grid(&atoms, &spec).unwrap();
    let n = centers.len();
    let n3 = 3 * n;
    let ev = evaluator(&centers, &charges, None);
    let order = GridOrder::boxes(grid.coords(), 1.2).unwrap();
    let perm = order.perm();

    // Evaluate in source order and in sorted order; the per-point value is
    // independent of the permutation because sorting is not a coordinate
    // change.
    let eval = |g: usize| {
        let parent = grid.atom_idx()[g] as usize;
        let local = [
            grid.coords()[g][0] - centers[parent][0],
            grid.coords()[g][1] - centers[parent][1],
            grid.coords()[g][2] - centers[parent][2],
        ];
        (ev.gradient_at(grid.coords()[g], parent).1, {
            let _ = local;
            ev.hessian_at(grid.coords()[g], parent)
        })
    };
    let sample: Vec<usize> = (0..grid.len())
        .step_by((grid.len() / 17).max(1))
        .take(17)
        .collect();
    for &g in &sample {
        let (dw0, h0) = eval(g);
        let slot = order.inverse()[g] as usize;
        let g_sorted = perm[slot] as usize;
        assert_eq!(g_sorted, g);
        let (dw1, h1) = eval(g_sorted);
        for (a, b) in dw0.iter().zip(&dw1) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        for (a, b) in h0.iter().zip(&h1) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        for &v in h0.iter().take(n3) {
            assert!(v.is_finite(), "non-finite Hessian at point {g}");
        }
    }
}

#[test]
fn parallel_matches_serial() {
    let (charges, centers) = benzene();
    let plan = plan_for(&charges, &centers);
    let ev = evaluator(&centers, &charges, None);
    let range = 0..plan.len().min(200);

    let req = DerivRequest {
        weights: true,
        dw: true,
        ..Default::default()
    };
    let mut serial = DerivOutput::new();
    let mut ws = DerivWorkspace::new();
    ev.fill_serial(
        DerivPoints::plan(&plan, range.clone()),
        &mut ws,
        &req,
        &mut serial,
    )
    .unwrap();
    let mut parallel = DerivOutput::new();
    ev.fill(DerivPoints::plan(&plan, range.clone()), &req, &mut parallel)
        .unwrap();
    for (a, b) in serial.dw().iter().zip(parallel.dw()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    for (a, b) in serial
        .partition_weights()
        .iter()
        .zip(parallel.partition_weights())
    {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    let contract: Vec<f64> = (0..range.len()).map(|k| 0.2 + 0.01 * k as f64).collect();
    let req = DerivRequest {
        ddc: Some(DerivContract::exact(&contract, 1)),
        ..Default::default()
    };
    let mut h_serial = DerivOutput::new();
    let mut ws = DerivWorkspace::new();
    ev.fill_serial(
        DerivPoints::plan(&plan, range.clone()),
        &mut ws,
        &req,
        &mut h_serial,
    )
    .unwrap();
    let mut h_par = DerivOutput::new();
    ev.fill(DerivPoints::plan(&plan, range.clone()), &req, &mut h_par)
        .unwrap();
    for (a, b) in h_serial
        .ddc()
        .unwrap()
        .ddc()
        .iter()
        .zip(h_par.ddc().unwrap().ddc())
    {
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }
}

/// End-to-end check: the analytic Hessian against a second difference of the
/// weight itself, with the point translated by its generating atom.  This
/// closes the chain gradient-vs-weight FD -> Hessian-vs-gradient FD.
#[test]
fn hessian_matches_second_difference_of_weight() {
    let (charges, centers) = water();
    let plan = plan_for(&charges, &centers);
    let ev = evaluator(&centers, &charges, None);
    let n = centers.len();
    let n3 = 3 * n;
    let h = 1e-4;
    for (parent, local) in sample_points(&plan, 1) {
        let point = [
            centers[parent][0] + local[0],
            centers[parent][1] + local[1],
            centers[parent][2] + local[2],
        ];
        let hess = ev.hessian_at(point, parent);
        for a in 0..n {
            for t in 0..3 {
                for b in 0..n {
                    for s in 0..3 {
                        let mut cpp = centers.clone();
                        cpp[a][t] += h;
                        cpp[b][s] += h;
                        let mut cpm = centers.clone();
                        cpm[a][t] += h;
                        cpm[b][s] -= h;
                        let mut cmp = centers.clone();
                        cmp[a][t] -= h;
                        cmp[b][s] += h;
                        let mut cmm = centers.clone();
                        cmm[a][t] -= h;
                        cmm[b][s] -= h;
                        let wpp = weight_at(&cpp, &charges, parent, local);
                        let wpm = weight_at(&cpm, &charges, parent, local);
                        let wmp = weight_at(&cmp, &charges, parent, local);
                        let wmm = weight_at(&cmm, &charges, parent, local);
                        let fd = (wpp - wpm - wmp + wmm) / (4.0 * h * h);
                        let an = hess[(a * 3 + t) * n3 + b * 3 + s];
                        assert!(
                            (fd - an).abs() < 2e-4,
                            "second difference: parent={parent} A={a} t={t} B={b} s={s} fd={fd} an={an}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn empty_selection_is_valid() {
    let (charges, centers) = water();
    let plan = plan_for(&charges, &centers);
    let ev = evaluator(&centers, &charges, Some(Vec::new()));
    let range = 0..plan.len().min(16);
    let contract = vec![1.0; range.len()];
    let req = DerivRequest {
        weights: true,
        dw: true,
        dc: Some(DerivContract::exact(&contract, 1)),
        ddc: Some(DerivContract::exact(&contract, 1)),
        ..Default::default()
    };
    let mut out = DerivOutput::new();
    let mut ws = DerivWorkspace::new();
    ev.fill_serial(
        DerivPoints::plan(&plan, range.clone()),
        &mut ws,
        &req,
        &mut out,
    )
    .unwrap();
    assert!(out.dw().is_empty());
    assert_eq!(out.partition_weights().len(), range.len());
    assert!(out.dc().unwrap().dc().is_empty());
    assert!(out.ddc().unwrap().ddc().is_empty());
}

#[test]
fn single_atom_has_trivial_derivatives() {
    let charges = vec![8u32];
    let centers = vec![[0.0, 0.0, 0.0]];
    let plan = plan_for(&charges, &centers);
    let ev = evaluator(&centers, &charges, None);
    for (parent, local) in sample_points(&plan, 3) {
        let point = [local[0], local[1], local[2]];
        let (w, dw) = ev.gradient_at(point, parent);
        assert!((w - 1.0).abs() < 1e-15);
        assert!(dw.iter().all(|v| v.abs() < 1e-15));
        let hess = ev.hessian_at(point, parent);
        assert!(hess.iter().all(|v| v.abs() < 1e-12));
    }
}

/// A Hessian run also emits the gradient; it must be bit-identical to the
/// dedicated blocked gradient kernel (the combined path reads the same
/// `dP`/`dZ` and uses the same expression).
#[test]
fn combined_hessian_emits_bitwise_gradient() {
    let (charges, centers) = water();
    let plan = plan_for(&charges, &centers);
    let ev = evaluator(&centers, &charges, None);
    let range = 0..plan.len().min(32);
    let contract = vec![1.0; range.len()];

    let grad_req = DerivRequest {
        weights: true,
        dw: true,
        ..Default::default()
    };
    let mut grad = DerivOutput::new();
    let mut ws = DerivWorkspace::new();
    ev.fill_serial(
        DerivPoints::plan(&plan, range.clone()),
        &mut ws,
        &grad_req,
        &mut grad,
    )
    .unwrap();

    let both_req = DerivRequest {
        weights: true,
        dw: true,
        ddc: Some(DerivContract::exact(&contract, 1)),
        ..Default::default()
    };
    let mut both = DerivOutput::new();
    ev.fill_serial(
        DerivPoints::plan(&plan, range.clone()),
        &mut ws,
        &both_req,
        &mut both,
    )
    .unwrap();

    for (a, b) in grad
        .partition_weights()
        .iter()
        .zip(both.partition_weights())
    {
        assert_eq!(a.to_bits(), b.to_bits(), "weight mismatch");
    }
    for (i, (a, b)) in grad.dw().iter().zip(both.dw()).enumerate() {
        assert!(
            (a - b).abs() <= 1e-12 * (1.0 + a.abs()),
            "dw mismatch at {i}: {a} vs {b}"
        );
    }
}

/// `dc` and `ddc` may use different `nset`, and a contract tensor may have
/// more columns than the streamed range.  Both properties are checked by
/// comparing each set with an independent single-set run.
#[test]
fn contract_sets_and_windows_are_independent() {
    let (charges, centers) = water();
    let plan = plan_for(&charges, &centers);
    let ev = evaluator(&centers, &charges, None);
    let range = 0..plan.len().min(24);
    let b = range.len();
    let offset = 2;
    let cols = b + 5;

    let nset_dc = 2;
    let nset_ddc = 3;
    let c_dc: Vec<f64> = (0..nset_dc * cols)
        .map(|i| 0.05 + 0.013 * i as f64)
        .collect();
    let c_ddc: Vec<f64> = (0..nset_ddc * cols)
        .map(|i| 0.11 + 0.007 * i as f64)
        .collect();

    let req = DerivRequest {
        dc: Some(DerivContract::window(&c_dc, nset_dc, cols, offset)),
        ddc: Some(DerivContract::window(&c_ddc, nset_ddc, cols, offset)),
        ..Default::default()
    };
    let mut out = DerivOutput::new();
    let mut ws = DerivWorkspace::new();
    ev.fill_serial(
        DerivPoints::plan(&plan, range.clone()),
        &mut ws,
        &req,
        &mut out,
    )
    .unwrap();

    for set in 0..nset_dc {
        let coeff: Vec<f64> = (0..b).map(|p| c_dc[set * cols + offset + p]).collect();
        let one = DerivRequest {
            dc: Some(DerivContract::exact(&coeff, 1)),
            ..Default::default()
        };
        let mut reference = DerivOutput::new();
        ev.fill_serial(
            DerivPoints::plan(&plan, range.clone()),
            &mut ws,
            &one,
            &mut reference,
        )
        .unwrap();
        for (a, b) in out
            .dc()
            .unwrap()
            .dc_set(set)
            .iter()
            .zip(reference.dc().unwrap().dc_set(0))
        {
            assert!((a - b).abs() < 1e-13, "dc set {set}: {a} vs {b}");
        }
    }

    for set in 0..nset_ddc {
        let coeff: Vec<f64> = (0..b).map(|p| c_ddc[set * cols + offset + p]).collect();
        let one = DerivRequest {
            ddc: Some(DerivContract::exact(&coeff, 1)),
            ..Default::default()
        };
        let mut reference = DerivOutput::new();
        ev.fill_serial(
            DerivPoints::plan(&plan, range.clone()),
            &mut ws,
            &one,
            &mut reference,
        )
        .unwrap();
        for (a, b) in out
            .ddc()
            .unwrap()
            .ddc_set(set)
            .iter()
            .zip(reference.ddc().unwrap().ddc_set(0))
        {
            assert!((a - b).abs() < 1e-13, "ddc set {set}: {a} vs {b}");
        }
    }

    // The parallel point path must use the same absolute contract columns.
    let mut par = DerivOutput::new();
    ev.fill(DerivPoints::plan(&plan, range.clone()), &req, &mut par)
        .unwrap();
    for set in 0..nset_dc {
        for (a, b) in out
            .dc()
            .unwrap()
            .dc_set(set)
            .iter()
            .zip(par.dc().unwrap().dc_set(set))
        {
            assert!((a - b).abs() < 1e-12, "parallel dc set {set}: {a} vs {b}");
        }
    }
    for set in 0..nset_ddc {
        for (a, b) in out
            .ddc()
            .unwrap()
            .ddc_set(set)
            .iter()
            .zip(par.ddc().unwrap().ddc_set(set))
        {
            assert!((a - b).abs() < 1e-12, "parallel ddc set {set}: {a} vs {b}");
        }
    }
}

/// The optional raw `ddw` batch tensor is exactly the concatenation of the
/// per-point visitor blocks.
#[test]
fn raw_ddw_matches_visitor() {
    let (charges, centers) = water();
    let plan = plan_for(&charges, &centers);
    let ev = evaluator(&centers, &charges, None);
    let range = 0..plan.len().min(12);
    let b = range.len();
    let req = DerivRequest {
        ddw: true,
        ..Default::default()
    };
    let mut out = DerivOutput::new();
    let mut ws = DerivWorkspace::new();
    let mut blocks: Vec<Vec<f64>> = Vec::with_capacity(b);
    ev.fill_serial_visit(
        DerivPoints::plan(&plan, range.clone()),
        &mut ws,
        &req,
        &mut out,
        |_g, block| blocks.push(block.to_vec()),
    )
    .unwrap();
    assert_eq!(blocks.len(), b);
    for (idx, block) in blocks.iter().enumerate() {
        for (c, &value) in block.iter().enumerate() {
            assert_eq!(
                out.ddw()[c * b + idx].to_bits(),
                value.to_bits(),
                "ddw mismatch at point {idx} component {c}"
            );
        }
    }

    // The parallel path copies its chunk-local point-last tensor back to the
    // same global layout.
    let mut par = DerivOutput::new();
    ev.fill(DerivPoints::plan(&plan, range), &req, &mut par)
        .unwrap();
    for (a, b) in out.ddw().iter().zip(par.ddw()) {
        assert!((a - b).abs() < 1e-13, "parallel ddw: {a} vs {b}");
    }
}

/// A non-Becke [`DerivScheme`] can drive the same streaming/contraction
/// evaluator; this toy scheme returns constant weights and zero derivatives.
#[derive(Debug)]
struct ToyScratch {
    hess: Vec<f64>,
}

#[derive(Debug)]
struct ToyDeriv {
    n: usize,
    sel: Vec<usize>,
}

impl ruaa_dft_grids::partition::DerivScheme for ToyDeriv {
    fn atom_count(&self) -> usize {
        self.n
    }

    fn selection(&self) -> &[usize] {
        &self.sel
    }

    fn block_scratch_kind(&self) -> std::any::TypeId {
        std::any::TypeId::of::<()>()
    }

    fn point_scratch_kind(&self) -> std::any::TypeId {
        std::any::TypeId::of::<ToyScratch>()
    }

    fn new_block_scratch(&self, _block: usize) -> Box<dyn ruaa_dft_grids::partition::DerivScratch> {
        Box::new(())
    }

    fn new_point_scratch(&self) -> Box<dyn ruaa_dft_grids::partition::DerivScratch> {
        Box::new(ToyScratch {
            hess: vec![0.0; self.sel.len() * 3 * self.n * 3],
        })
    }

    fn gradient_block(
        &self,
        _coords: &[[f64; 3]],
        _parents: &[u32],
        _quad: &[f64],
        weights: &mut [f64],
        dw: &mut [f64],
        _scratch: &mut dyn ruaa_dft_grids::partition::DerivScratch,
    ) {
        weights.fill(1.0);
        dw.fill(0.0);
    }

    fn point_hessian<'s>(
        &self,
        _point: [f64; 3],
        _parent: usize,
        _quad: f64,
        grad: Option<&mut [f64]>,
        scratch: &'s mut dyn ruaa_dft_grids::partition::DerivScratch,
    ) -> (f64, &'s [f64]) {
        let scratch = scratch
            .as_any_mut()
            .downcast_mut::<ToyScratch>()
            .expect("toy scratch");
        if let Some(grad) = grad {
            grad.fill(0.0);
        }
        (1.0, &scratch.hess)
    }
}

#[test]
fn custom_scheme_drives_evaluator() {
    let (charges, centers) = water();
    let plan = plan_for(&charges, &centers);
    let n = centers.len();
    let ev = PartitionDerivEvaluator::with_scheme_from_parts(
        &centers,
        &charges,
        config(),
        Box::new(ToyDeriv {
            n,
            sel: (0..n).collect(),
        }),
        0,
    )
    .unwrap();
    let range = 0..plan.len().min(16);
    let contract = vec![1.0; range.len()];
    let req = DerivRequest {
        weights: true,
        dw: true,
        ddw: true,
        dc: Some(DerivContract::exact(&contract, 1)),
        ddc: Some(DerivContract::exact(&contract, 1)),
    };
    let mut out = DerivOutput::new();
    let mut ws = DerivWorkspace::new();
    ev.fill_serial(
        DerivPoints::plan(&plan, range.clone()),
        &mut ws,
        &req,
        &mut out,
    )
    .unwrap();
    assert!(out.partition_weights().iter().all(|&w| w == 1.0));
    assert!(out.dw().iter().all(|&v| v == 0.0));
    assert!(out.ddw().iter().all(|&v| v == 0.0));
    assert!(out.dc().unwrap().dc().iter().all(|&v| v == 0.0));
    assert!(out.ddc().unwrap().ddc().iter().all(|&v| v == 0.0));
}

#[test]
fn out_of_order_custom_selection_is_rejected() {
    let centers = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let err = PartitionDerivEvaluator::with_scheme_from_parts(
        &centers,
        &[1, 1],
        config(),
        Box::new(ToyDeriv {
            n: 2,
            sel: vec![1, 0],
        }),
        0,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        ruaa_dft_grids::Error::InvalidSelectionOrder { atom: 0 }
    ));
}

#[test]
fn mixed_parent_lane_matches_point_hessian() {
    let (charges, centers) = water();
    let plan = plan_for(&charges, &centers);
    let ev = evaluator(&centers, &charges, None);
    let first = plan.atom_range(0);
    let start = first.end.saturating_sub(2);
    let end = (start + 6).min(plan.len());
    assert!(end > start && start < first.end && end > first.end);

    let range = start..end;
    let range_len = range.len();
    let req = DerivRequest {
        weights: true,
        dw: true,
        ddw: true,
        ..Default::default()
    };
    let mut out = DerivOutput::new();
    let mut ws = DerivWorkspace::new();
    ev.fill_serial(
        DerivPoints::plan(&plan, range.clone()),
        &mut ws,
        &req,
        &mut out,
    )
    .unwrap();

    for (idx, g) in range.clone().enumerate() {
        let parent = plan.parent_of(g) as usize;
        let local = plan.local_coord(g);
        let point = [
            centers[parent][0] + local[0],
            centers[parent][1] + local[1],
            centers[parent][2] + local[2],
        ];
        let q = plan.quad_weight(g);
        let (weight, dw) = ev.gradient_at(point, parent);
        let hess = ev.hessian_at(point, parent);
        assert_eq!(
            out.partition_weights()[idx].to_bits(),
            weight.to_bits(),
            "weight g={g}"
        );
        for (row, &v) in dw.iter().enumerate() {
            let got = out.dw()[row * range_len + idx];
            let want = q * v;
            assert!(
                (got - want).abs() <= 1e-12 * (1.0 + want.abs()),
                "dw row={row} g={g}: {got} vs {want}"
            );
        }
        for (c, &v) in hess.iter().enumerate() {
            let got = out.ddw()[c * range_len + idx];
            let want = q * v;
            assert!(
                (got - want).abs() <= 1e-12 * (1.0 + want.abs()),
                "ddw c={c} g={g}: {got} vs {want}"
            );
        }
    }
}

#[test]
fn worker_memory_budget_is_configurable() {
    let (charges, centers) = water();
    let ev = PartitionDerivEvaluator::from_parts(
        &centers,
        &charges,
        config(),
        DerivOptions {
            atoms: None,
            block_points: 0,
            worker_memory_bytes: 64 << 20,
            hessian_backend: HessianBackend::Auto,
        },
    )
    .unwrap();
    assert_eq!(ev.worker_memory_bytes(), 64 << 20);
    assert_eq!(ev.gradient_block() % 8, 0);
}

// --- Stratmann–Scuseria–Frisch -------------------------------------------------

#[test]
fn stratmann_gradient_matches_total_finite_difference() {
    for (charges, centers) in [water(), benzene()] {
        let plan = plan_for(&charges, &centers);
        let ev = evaluator_with(&centers, &charges, stratmann_config(), None);
        let h = 1e-5;
        for (parent, local) in sample_points(&plan, 2) {
            let point = [
                centers[parent][0] + local[0],
                centers[parent][1] + local[1],
                centers[parent][2] + local[2],
            ];
            let (_, dw) = ev.gradient_at(point, parent);
            for atom in 0..centers.len() {
                for comp in 0..3 {
                    let cp = perturbed(&centers, atom, comp, h);
                    let cm = perturbed(&centers, atom, comp, -h);
                    let fd = (weight_at_scheme(&cp, &charges, parent, local, stratmann_config())
                        - weight_at_scheme(&cm, &charges, parent, local, stratmann_config()))
                        / (2.0 * h);
                    let an = dw[atom * 3 + comp];
                    assert!(
                        (fd - an).abs() < 2e-6,
                        "SSF gradient: n={} parent={parent} atom={atom} comp={comp} fd={fd} an={an}",
                        centers.len()
                    );
                }
            }
        }
    }
}

#[test]
fn stratmann_hessian_matches_gradient_finite_difference() {
    for (charges, centers) in [water(), benzene()] {
        let plan = plan_for(&charges, &centers);
        let ev = evaluator_with(&centers, &charges, stratmann_config(), None);
        let n = centers.len();
        let n3 = 3 * n;
        let h = 2e-5;
        for (parent, local) in sample_points(&plan, 1) {
            let point = [
                centers[parent][0] + local[0],
                centers[parent][1] + local[1],
                centers[parent][2] + local[2],
            ];
            let hess = ev.hessian_at(point, parent);
            for atom in 0..n {
                for comp in 0..3 {
                    let cp = perturbed(&centers, atom, comp, h);
                    let cm = perturbed(&centers, atom, comp, -h);
                    let gp = gradient_at_scheme(&cp, &charges, parent, local, stratmann_config());
                    let gm = gradient_at_scheme(&cm, &charges, parent, local, stratmann_config());
                    for b in 0..n {
                        for s in 0..3 {
                            let c = b * 3 + s;
                            let fd = (gp[c] - gm[c]) / (2.0 * h);
                            let an = hess[(atom * 3 + comp) * n3 + c];
                            assert!(
                                (fd - an).abs() < 2e-5,
                                "SSF hessian: parent={parent} A={atom} t={comp} B={b} s={s} fd={fd} an={an}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn stratmann_hessian_is_symmetric_and_translation_invariant() {
    let (charges, centers) = water();
    let plan = plan_for(&charges, &centers);
    let ev = evaluator_with(&centers, &charges, stratmann_config(), None);
    let n = centers.len();
    let n3 = 3 * n;
    for (parent, local) in sample_points(&plan, 1) {
        let point = [
            centers[parent][0] + local[0],
            centers[parent][1] + local[1],
            centers[parent][2] + local[2],
        ];
        let h = ev.hessian_at(point, parent);
        for a in 0..n {
            for t in 0..3 {
                for b in 0..n {
                    for s in 0..3 {
                        let ab = h[(a * 3 + t) * n3 + b * 3 + s];
                        let ba = h[(b * 3 + s) * n3 + a * 3 + t];
                        assert!(
                            (ab - ba).abs() < 1e-10,
                            "SSF asymmetry A={a} t={t} B={b} s={s}: {ab} vs {ba}"
                        );
                    }
                }
            }
        }
        for t in 0..3 {
            for b in 0..n {
                for s in 0..3 {
                    let sum: f64 = (0..n).map(|a| h[(a * 3 + t) * n3 + b * 3 + s]).sum();
                    assert!(sum.abs() < 1e-10, "SSF hessian translation sum = {sum}");
                }
            }
        }
    }
}

#[test]
fn stratmann_weights_match_value_evaluator() {
    use ruaa_dft_grids::grid::BatchBuffer;
    use ruaa_dft_grids::partition::{PartitionEvaluator, Workspace};
    let (charges, centers) = benzene();
    let atoms: Vec<(u32, [f64; 3])> = charges
        .iter()
        .copied()
        .zip(centers.iter().copied())
        .collect();
    let spec: GridSpec = PyscfGridSpecBuilder::default()
        .level(0)
        .build()
        .unwrap()
        .into();
    let plan = GridPlan::build(&atoms, &spec).unwrap();
    let range = 0..plan.len().min(256);
    let ev = evaluator_with(&centers, &charges, stratmann_config(), None);
    let mut out = DerivOutput::new();
    let mut ws = DerivWorkspace::new();
    let req = DerivRequest {
        weights: true,
        ..Default::default()
    };
    ev.fill_serial(
        DerivPoints::plan(&plan, range.clone()),
        &mut ws,
        &req,
        &mut out,
    )
    .unwrap();

    let weight_ev = PartitionEvaluator::from_parts(&centers, &charges, stratmann_config()).unwrap();
    let mut weight_ws = Workspace::new();
    let mut weight_batch = BatchBuffer::new();
    weight_ev
        .fill_plan(&plan, range.clone(), &mut weight_ws, &mut weight_batch)
        .unwrap();
    for (i, g) in range.enumerate() {
        assert_eq!(
            out.partition_weights()[i].to_bits(),
            weight_batch.partition_weights()[g].to_bits(),
            "SSF weight mismatch at point {g}"
        );
    }
}

#[test]
fn stratmann_selection_is_exactly_a_row_subset() {
    let (charges, centers) = water();
    let plan = plan_for(&charges, &centers);
    let all = evaluator_with(&centers, &charges, stratmann_config(), None);
    let sub = evaluator_with(&centers, &charges, stratmann_config(), Some(vec![1]));
    let n = centers.len();
    let n3 = 3 * n;
    for (parent, local) in sample_points(&plan, 1) {
        let point = [
            centers[parent][0] + local[0],
            centers[parent][1] + local[1],
            centers[parent][2] + local[2],
        ];
        let (w_all, dw_all) = all.gradient_at(point, parent);
        let (w_sub, dw_sub) = sub.gradient_at(point, parent);
        assert_eq!(w_all.to_bits(), w_sub.to_bits());
        for t in 0..3 {
            assert_eq!(
                dw_all[3 + t].to_bits(),
                dw_sub[t].to_bits(),
                "SSF selected gradient row mismatch"
            );
        }
        let h_all = all.hessian_at(point, parent);
        let h_sub = sub.hessian_at(point, parent);
        for t in 0..3 {
            for c in 0..n3 {
                assert_eq!(
                    h_all[(3 + t) * n3 + c].to_bits(),
                    h_sub[t * n3 + c].to_bits(),
                    "SSF selected Hessian row mismatch t={t} c={c}"
                );
            }
        }
    }
}

#[test]
fn stratmann_combined_hessian_emits_bitwise_gradient() {
    let (charges, centers) = water();
    let plan = plan_for(&charges, &centers);
    let ev = evaluator_with(&centers, &charges, stratmann_config(), None);
    let range = 0..plan.len().min(64);
    let contract = vec![1.0; range.len()];

    let grad_req = DerivRequest {
        weights: true,
        dw: true,
        ..Default::default()
    };
    let mut grad = DerivOutput::new();
    let mut ws = DerivWorkspace::new();
    ev.fill_serial(
        DerivPoints::plan(&plan, range.clone()),
        &mut ws,
        &grad_req,
        &mut grad,
    )
    .unwrap();

    let both_req = DerivRequest {
        weights: true,
        dw: true,
        ddc: Some(DerivContract::exact(&contract, 1)),
        ..Default::default()
    };
    let mut both = DerivOutput::new();
    ev.fill_serial(
        DerivPoints::plan(&plan, range.clone()),
        &mut ws,
        &both_req,
        &mut both,
    )
    .unwrap();

    assert_eq!(grad.partition_weights(), both.partition_weights());
    for (i, (&a, &b)) in grad.dw().iter().zip(both.dw()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "SSF combined gradient row {i}");
    }
}

// --- Gaussian 16 SS ---------------------------------------------------------

#[test]
fn gaussian_ss_gradient_matches_total_finite_difference() {
    let (charges, centers) = water();
    let plan = plan_for(&charges, &centers);
    let ev = evaluator_with(&centers, &charges, gaussian_ss_config(), None);
    let h = 1e-5;
    let mut max_err: f64 = 0.0;
    for (parent, local) in sample_points(&plan, 2) {
        let point = [
            centers[parent][0] + local[0],
            centers[parent][1] + local[1],
            centers[parent][2] + local[2],
        ];
        let (_, dw) = ev.gradient_at(point, parent);
        for atom in 0..centers.len() {
            for comp in 0..3 {
                let cp = perturbed(&centers, atom, comp, h);
                let cm = perturbed(&centers, atom, comp, -h);
                let fd = (weight_at_scheme(&cp, &charges, parent, local, gaussian_ss_config())
                    - weight_at_scheme(&cm, &charges, parent, local, gaussian_ss_config()))
                    / (2.0 * h);
                let an = dw[atom * 3 + comp];
                max_err = max_err.max((fd - an).abs());
                assert!(
                    (fd - an).abs() < 2e-5,
                    "Gaussian SS gradient: parent={parent} atom={atom} comp={comp} fd={fd} an={an}"
                );
            }
        }
    }
    assert!(max_err < 2e-5);
}

#[test]
fn gaussian_ss_adjusted_gradient_matches_total_finite_difference() {
    // The generic derivative backend also supports a non-zero size
    // adjustment; the kill cutoff then shifts with a_max.
    let cfg = PartitionConfig::new(PartitionScheme::GaussianSs, RadiiConfig::pyscf());
    let (charges, centers) = water();
    let plan = plan_for(&charges, &centers);
    let ev = evaluator_with(&centers, &charges, cfg.clone(), None);
    let h = 1e-5;
    for (parent, local) in sample_points(&plan, 1) {
        let point = [
            centers[parent][0] + local[0],
            centers[parent][1] + local[1],
            centers[parent][2] + local[2],
        ];
        let (_, dw) = ev.gradient_at(point, parent);
        for atom in 0..centers.len() {
            for comp in 0..3 {
                let cp = perturbed(&centers, atom, comp, h);
                let cm = perturbed(&centers, atom, comp, -h);
                let fd = (weight_at_scheme(&cp, &charges, parent, local, cfg.clone())
                    - weight_at_scheme(&cm, &charges, parent, local, cfg.clone()))
                    / (2.0 * h);
                let an = dw[atom * 3 + comp];
                assert!(
                    (fd - an).abs() < 3e-5,
                    "Gaussian SS adjusted gradient: parent={parent} atom={atom} comp={comp} fd={fd} an={an}"
                );
            }
        }
    }
}

#[test]
fn gaussian_ss_hessian_matches_gradient_finite_difference() {
    let (charges, centers) = water();
    let plan = plan_for(&charges, &centers);
    let ev = evaluator_with(&centers, &charges, gaussian_ss_config(), None);
    let n = centers.len();
    let n3 = 3 * n;
    let h = 2e-5;
    for (parent, local) in sample_points(&plan, 1) {
        let point = [
            centers[parent][0] + local[0],
            centers[parent][1] + local[1],
            centers[parent][2] + local[2],
        ];
        let hess = ev.hessian_at(point, parent);
        for atom in 0..n {
            for comp in 0..3 {
                let cp = perturbed(&centers, atom, comp, h);
                let cm = perturbed(&centers, atom, comp, -h);
                let gp = gradient_at_scheme(&cp, &charges, parent, local, gaussian_ss_config());
                let gm = gradient_at_scheme(&cm, &charges, parent, local, gaussian_ss_config());
                for b in 0..n {
                    for s in 0..3 {
                        let c = b * 3 + s;
                        let fd = (gp[c] - gm[c]) / (2.0 * h);
                        let an = hess[(atom * 3 + comp) * n3 + c];
                        assert!(
                            (fd - an).abs() < 3e-5,
                            "Gaussian SS hessian: parent={parent} A={atom} t={comp} B={b} s={s} fd={fd} an={an}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn gaussian_ss_hessian_is_symmetric_and_translation_invariant() {
    let (charges, centers) = water();
    let plan = plan_for(&charges, &centers);
    let ev = evaluator_with(&centers, &charges, gaussian_ss_config(), None);
    let n = centers.len();
    let n3 = 3 * n;
    for (parent, local) in sample_points(&plan, 1) {
        let point = [
            centers[parent][0] + local[0],
            centers[parent][1] + local[1],
            centers[parent][2] + local[2],
        ];
        let h = ev.hessian_at(point, parent);
        for a in 0..n {
            for t in 0..3 {
                for b in 0..n {
                    for s in 0..3 {
                        let ab = h[(a * 3 + t) * n3 + b * 3 + s];
                        let ba = h[(b * 3 + s) * n3 + a * 3 + t];
                        assert!(
                            (ab - ba).abs() < 1e-9,
                            "Gaussian SS asymmetry A={a} t={t} B={b} s={s}: {ab} vs {ba}"
                        );
                    }
                }
            }
        }
        for t in 0..3 {
            for b in 0..n {
                for s in 0..3 {
                    let sum: f64 = (0..n).map(|a| h[(a * 3 + t) * n3 + b * 3 + s]).sum();
                    assert!(
                        sum.abs() < 1e-9,
                        "Gaussian SS hessian translation sum = {sum}"
                    );
                }
            }
        }
    }
}

#[test]
fn gaussian_ss_weights_match_value_evaluator() {
    use ruaa_dft_grids::grid::BatchBuffer;
    use ruaa_dft_grids::partition::{PartitionEvaluator, Workspace};
    let (charges, centers) = benzene();
    let atoms: Vec<(u32, [f64; 3])> = charges
        .iter()
        .copied()
        .zip(centers.iter().copied())
        .collect();
    let spec: GridSpec = PyscfGridSpecBuilder::default()
        .level(0)
        .build()
        .unwrap()
        .into();
    let plan = GridPlan::build(&atoms, &spec).unwrap();
    let range = 0..plan.len().min(256);
    let ev = evaluator_with(&centers, &charges, gaussian_ss_config(), None);
    let mut out = DerivOutput::new();
    let mut ws = DerivWorkspace::new();
    let req = DerivRequest {
        weights: true,
        ..Default::default()
    };
    ev.fill_serial(
        DerivPoints::plan(&plan, range.clone()),
        &mut ws,
        &req,
        &mut out,
    )
    .unwrap();

    let weight_ev =
        PartitionEvaluator::from_parts(&centers, &charges, gaussian_ss_config()).unwrap();
    let mut weight_ws = Workspace::new();
    let mut weight_batch = BatchBuffer::new();
    weight_ev
        .fill_plan(&plan, range.clone(), &mut weight_ws, &mut weight_batch)
        .unwrap();
    for (i, g) in range.enumerate() {
        assert_eq!(
            out.partition_weights()[i].to_bits(),
            weight_batch.partition_weights()[g].to_bits(),
            "Gaussian SS weight mismatch at point {g}"
        );
    }
}

// --- Laqua–Kussmann–Ochsenfeld ------------------------------------------------

#[test]
fn lko_gradient_matches_total_finite_difference() {
    for (charges, centers) in [water(), benzene()] {
        let plan = plan_for(&charges, &centers);
        let ev = evaluator_with(&centers, &charges, lko_config(), None);
        let h = 1e-5;
        let mut max_err: f64 = 0.0;
        for (parent, local) in sample_points(&plan, 2) {
            let point = [
                centers[parent][0] + local[0],
                centers[parent][1] + local[1],
                centers[parent][2] + local[2],
            ];
            let (_, dw) = ev.gradient_at(point, parent);
            for atom in 0..centers.len() {
                for comp in 0..3 {
                    let cp = perturbed(&centers, atom, comp, h);
                    let cm = perturbed(&centers, atom, comp, -h);
                    let fd = (weight_at_scheme(&cp, &charges, parent, local, lko_config())
                        - weight_at_scheme(&cm, &charges, parent, local, lko_config()))
                        / (2.0 * h);
                    let an = dw[atom * 3 + comp];
                    max_err = max_err.max((fd - an).abs());
                    assert!(
                        (fd - an).abs() < 2e-6,
                        "LKO gradient: n={} parent={parent} atom={atom} comp={comp} fd={fd} an={an}",
                        centers.len()
                    );
                }
            }
        }
        eprintln!("LKO gradient max |FD - analytic| = {max_err:e}");
        assert!(max_err < 5e-7, "LKO gradient max error {max_err:e}");
    }
}

#[test]
fn lko_hessian_matches_gradient_finite_difference() {
    for (charges, centers) in [water(), benzene()] {
        let plan = plan_for(&charges, &centers);
        let ev = evaluator_with(&centers, &charges, lko_config(), None);
        let n = centers.len();
        let n3 = 3 * n;
        let h = 2e-5;
        let mut max_err: f64 = 0.0;
        for (parent, local) in sample_points(&plan, 1) {
            let point = [
                centers[parent][0] + local[0],
                centers[parent][1] + local[1],
                centers[parent][2] + local[2],
            ];
            let hess = ev.hessian_at(point, parent);
            for atom in 0..n {
                for comp in 0..3 {
                    let cp = perturbed(&centers, atom, comp, h);
                    let cm = perturbed(&centers, atom, comp, -h);
                    let gp = gradient_at_scheme(&cp, &charges, parent, local, lko_config());
                    let gm = gradient_at_scheme(&cm, &charges, parent, local, lko_config());
                    for b in 0..n {
                        for s in 0..3 {
                            let c = b * 3 + s;
                            let fd = (gp[c] - gm[c]) / (2.0 * h);
                            let an = hess[(atom * 3 + comp) * n3 + c];
                            max_err = max_err.max((fd - an).abs());
                            assert!(
                                (fd - an).abs() < 2e-5,
                                "LKO hessian: parent={parent} A={atom} t={comp} B={b} s={s} fd={fd} an={an}"
                            );
                        }
                    }
                }
            }
        }
        eprintln!("LKO hessian max |FD - analytic| = {max_err:e}");
        assert!(max_err < 5e-6, "LKO hessian max error {max_err:e}");
    }
}

#[test]
fn lko_hessian_is_symmetric_and_translation_invariant() {
    let (charges, centers) = water();
    let plan = plan_for(&charges, &centers);
    let ev = evaluator_with(&centers, &charges, lko_config(), None);
    let n = centers.len();
    let n3 = 3 * n;
    for (parent, local) in sample_points(&plan, 1) {
        let point = [
            centers[parent][0] + local[0],
            centers[parent][1] + local[1],
            centers[parent][2] + local[2],
        ];
        let h = ev.hessian_at(point, parent);
        for a in 0..n {
            for t in 0..3 {
                for b in 0..n {
                    for s in 0..3 {
                        let ab = h[(a * 3 + t) * n3 + b * 3 + s];
                        let ba = h[(b * 3 + s) * n3 + a * 3 + t];
                        assert!(
                            (ab - ba).abs() < 1e-10,
                            "LKO asymmetry A={a} t={t} B={b} s={s}: {ab} vs {ba}"
                        );
                    }
                }
            }
        }
        for t in 0..3 {
            for b in 0..n {
                for s in 0..3 {
                    let sum: f64 = (0..n).map(|a| h[(a * 3 + t) * n3 + b * 3 + s]).sum();
                    assert!(sum.abs() < 1e-10, "LKO hessian translation sum = {sum}");
                }
            }
        }
    }
}

#[test]
fn lko_weights_match_value_evaluator() {
    use ruaa_dft_grids::grid::BatchBuffer;
    use ruaa_dft_grids::partition::{PartitionEvaluator, Workspace};
    let (charges, centers) = benzene();
    let atoms: Vec<(u32, [f64; 3])> = charges
        .iter()
        .copied()
        .zip(centers.iter().copied())
        .collect();
    let spec: GridSpec = PyscfGridSpecBuilder::default()
        .level(0)
        .build()
        .unwrap()
        .into();
    let plan = GridPlan::build(&atoms, &spec).unwrap();
    let range = 0..plan.len().min(256);
    let ev = evaluator_with(&centers, &charges, lko_config(), None);
    let mut out = DerivOutput::new();
    let mut ws = DerivWorkspace::new();
    let req = DerivRequest {
        weights: true,
        ..Default::default()
    };
    ev.fill_serial(
        DerivPoints::plan(&plan, range.clone()),
        &mut ws,
        &req,
        &mut out,
    )
    .unwrap();

    let weight_ev = PartitionEvaluator::from_parts(&centers, &charges, lko_config()).unwrap();
    let mut weight_ws = Workspace::new();
    let mut batch = BatchBuffer::new();
    weight_ev
        .fill_plan(&plan, range.clone(), &mut weight_ws, &mut batch)
        .unwrap();
    for (i, (&deriv, &value)) in out
        .partition_weights()
        .iter()
        .zip(batch.partition_weights())
        .enumerate()
    {
        assert_eq!(
            deriv.to_bits(),
            value.to_bits(),
            "LKO weight {i} not bitwise"
        );
    }
}

#[test]
fn lko_selection_is_exactly_a_row_subset() {
    let (charges, centers) = benzene();
    let all = evaluator_with(&centers, &charges, lko_config(), None);
    let sub = evaluator_with(&centers, &charges, lko_config(), Some(vec![1]));
    let point = [
        centers[1][0] + 0.3,
        centers[1][1] - 0.2,
        centers[1][2] + 0.4,
    ];
    let (_, dw_all) = all.gradient_at(point, 1);
    let (_, dw_sub) = sub.gradient_at(point, 1);
    for t in 0..3 {
        assert_eq!(dw_all[3 + t].to_bits(), dw_sub[t].to_bits());
    }
}

#[test]
fn lko_combined_hessian_emits_bitwise_gradient() {
    let (charges, centers) = water();
    let plan = plan_for(&charges, &centers);
    let ev = evaluator_with(&centers, &charges, lko_config(), None);
    let range = 0..plan.len().min(64);
    let contract = vec![1.0; range.len()];

    let grad_req = DerivRequest {
        weights: true,
        dw: true,
        ..Default::default()
    };
    let mut grad = DerivOutput::new();
    let mut ws = DerivWorkspace::new();
    ev.fill_serial(
        DerivPoints::plan(&plan, range.clone()),
        &mut ws,
        &grad_req,
        &mut grad,
    )
    .unwrap();

    let both_req = DerivRequest {
        weights: true,
        dw: true,
        ddc: Some(DerivContract::exact(&contract, 1)),
        ..Default::default()
    };
    let mut both = DerivOutput::new();
    ev.fill_serial(
        DerivPoints::plan(&plan, range.clone()),
        &mut ws,
        &both_req,
        &mut both,
    )
    .unwrap();

    assert_eq!(grad.partition_weights(), both.partition_weights());
    // The Hessian kernel recomputes the first derivatives from its own
    // `dP`/`dZ` accumulators with a different summation order, so the two
    // paths agree to round-off rather than bit-for-bit.
    for (i, (&a, &b)) in grad.dw().iter().zip(both.dw()).enumerate() {
        assert!(
            (a - b).abs() <= 1e-12,
            "LKO combined gradient row {i}: {a} vs {b}"
        );
    }
}

// --- mBecke (classic derivative of the default-clamp product) ------------------

#[test]
fn mbecke_gradient_matches_finite_difference_of_the_value_path() {
    for (charges, centers) in [water(), benzene()] {
        let plan = plan_for(&charges, &centers);
        let ev = evaluator_with(&centers, &charges, mbecke_config(), None);
        let h = 1e-5;
        let mut max_err: f64 = 0.0;
        for (parent, local) in sample_points(&plan, 2) {
            let point = [
                centers[parent][0] + local[0],
                centers[parent][1] + local[1],
                centers[parent][2] + local[2],
            ];
            let (_, dw) = ev.gradient_at(point, parent);
            for atom in 0..centers.len() {
                for comp in 0..3 {
                    let cp = perturbed(&centers, atom, comp, h);
                    let cm = perturbed(&centers, atom, comp, -h);
                    let fd = (weight_at_scheme(&cp, &charges, parent, local, mbecke_config())
                        - weight_at_scheme(&cm, &charges, parent, local, mbecke_config()))
                        / (2.0 * h);
                    let an = dw[atom * 3 + comp];
                    max_err = max_err.max((fd - an).abs());
                    assert!(
                        (fd - an).abs() < 2e-6,
                        "mBecke gradient: parent={parent} atom={atom} comp={comp} fd={fd} an={an}"
                    );
                }
            }
        }
        eprintln!("mBecke gradient max |FD - analytic| = {max_err:e}");
    }
}

#[test]
fn mbecke_hessian_matches_gradient_finite_difference() {
    let (charges, centers) = water();
    let plan = plan_for(&charges, &centers);
    let ev = evaluator_with(&centers, &charges, mbecke_config(), None);
    let n = centers.len();
    let n3 = 3 * n;
    let h = 2e-5;
    let mut max_err: f64 = 0.0;
    for (parent, local) in sample_points(&plan, 1) {
        let point = [
            centers[parent][0] + local[0],
            centers[parent][1] + local[1],
            centers[parent][2] + local[2],
        ];
        let hess = ev.hessian_at(point, parent);
        for atom in 0..n {
            for comp in 0..3 {
                let cp = perturbed(&centers, atom, comp, h);
                let cm = perturbed(&centers, atom, comp, -h);
                let gp = gradient_at_scheme(&cp, &charges, parent, local, mbecke_config());
                let gm = gradient_at_scheme(&cm, &charges, parent, local, mbecke_config());
                for b in 0..n {
                    for s in 0..3 {
                        let c = b * 3 + s;
                        let fd = (gp[c] - gm[c]) / (2.0 * h);
                        let an = hess[(atom * 3 + comp) * n3 + c];
                        max_err = max_err.max((fd - an).abs());
                        assert!(
                            (fd - an).abs() < 2e-5,
                            "mBecke hessian: A={atom} t={comp} B={b} s={s} fd={fd} an={an}"
                        );
                    }
                }
            }
        }
    }
    eprintln!("mBecke hessian max |FD - analytic| = {max_err:e}");
}

#[test]
fn mbecke_hessian_is_symmetric_and_translation_invariant() {
    let (charges, centers) = water();
    let plan = plan_for(&charges, &centers);
    let ev = evaluator_with(&centers, &charges, mbecke_config(), None);
    let n = centers.len();
    let n3 = 3 * n;
    for (parent, local) in sample_points(&plan, 1) {
        let point = [
            centers[parent][0] + local[0],
            centers[parent][1] + local[1],
            centers[parent][2] + local[2],
        ];
        let h = ev.hessian_at(point, parent);
        for t in 0..3 {
            for b in 0..n {
                for s in 0..3 {
                    let sum: f64 = (0..n).map(|a| h[(a * 3 + t) * n3 + b * 3 + s]).sum();
                    assert!(sum.abs() < 1e-10, "mBecke hessian translation sum = {sum}");
                }
            }
        }
    }
}

#[test]
fn grid_range_matches_full_grid() {
    use ruaa_dft_grids::moleculegrid::build_molecular_grid;

    let (charges, centers) = water();
    let atoms: Vec<(u32, [f64; 3])> = charges
        .iter()
        .copied()
        .zip(centers.iter().copied())
        .collect();
    let spec: GridSpec = PyscfGridSpecBuilder::default()
        .level(0)
        .build()
        .unwrap()
        .into();
    let grid = build_molecular_grid(&atoms, &spec).unwrap();
    let ev = grid
        .partition_deriv_evaluator(DerivOptions::default())
        .unwrap();
    let full = grid.weight_derivatives().unwrap();

    let len = grid.len();
    let range = (len / 3)..(len / 3 + 97).min(len);
    assert!(!range.is_empty());

    let req = DerivRequest {
        weights: true,
        dw: true,
        ..DerivRequest::default()
    };
    let mut out = DerivOutput::new();
    ev.fill(DerivPoints::grid(&grid, range.clone()), &req, &mut out)
        .unwrap();
    assert_eq!(out.range(), range.clone());
    assert_eq!(out.partition_weights().len(), range.len());
    assert_eq!(out.dw().len(), grid.atom_count() * 3 * range.len());

    let block = range.len();
    for (i, point) in range.clone().enumerate() {
        assert_eq!(
            out.partition_weights()[i].to_bits(),
            full.partition_weights()[point].to_bits(),
            "weight at point {point}"
        );
    }
    for row in 0..grid.atom_count() * 3 {
        for (i, point) in range.clone().enumerate() {
            assert_eq!(
                out.dw()[row * block + i].to_bits(),
                full.dw()[row * len + point].to_bits(),
                "dw row {row} point {point}"
            );
        }
    }

    let err = ev
        .fill(
            DerivPoints::grid(&grid, 0..len + 1),
            &req,
            &mut DerivOutput::new(),
        )
        .unwrap_err();
    assert!(matches!(
        err,
        ruaa_dft_grids::Error::RangeOutOfBounds { .. }
    ));
}

#[test]
fn grid_range_serial_matches_parallel() {
    use ruaa_dft_grids::moleculegrid::build_molecular_grid;

    let (charges, centers) = water();
    let atoms: Vec<(u32, [f64; 3])> = charges
        .iter()
        .copied()
        .zip(centers.iter().copied())
        .collect();
    let spec: GridSpec = PyscfGridSpecBuilder::default()
        .level(0)
        .build()
        .unwrap()
        .into();
    let grid = build_molecular_grid(&atoms, &spec).unwrap();
    let ev = grid
        .partition_deriv_evaluator(DerivOptions::default())
        .unwrap();

    let range = 40..140;
    let req = DerivRequest {
        weights: true,
        dw: true,
        ddw: true,
        ..DerivRequest::default()
    };
    let mut parallel = DerivOutput::new();
    ev.fill(DerivPoints::grid(&grid, range.clone()), &req, &mut parallel)
        .unwrap();

    let mut serial = DerivOutput::new();
    let mut ws = DerivWorkspace::new();
    ev.fill_serial(
        DerivPoints::grid(&grid, range.clone()),
        &mut ws,
        &req,
        &mut serial,
    )
    .unwrap();

    assert_eq!(serial.range(), range);
    assert_eq!(
        serial.partition_weights().len(),
        parallel.partition_weights().len()
    );
    for (a, b) in serial
        .partition_weights()
        .iter()
        .zip(parallel.partition_weights())
    {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    for (a, b) in serial.dw().iter().zip(parallel.dw()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    for (a, b) in serial.ddw().iter().zip(parallel.ddw()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn grid_hessian_blocks_match_full_ddw() {
    use ruaa_dft_grids::moleculegrid::build_molecular_grid;

    let (charges, centers) = water();
    let atoms: Vec<(u32, [f64; 3])> = charges
        .iter()
        .copied()
        .zip(centers.iter().copied())
        .collect();
    let spec: GridSpec = PyscfGridSpecBuilder::default()
        .level(0)
        .build()
        .unwrap()
        .into();
    let grid = build_molecular_grid(&atoms, &spec).unwrap();
    let ev = grid
        .partition_deriv_evaluator(DerivOptions::default())
        .unwrap();

    let range = 30..130;
    let block_len = range.len();
    let nsel = grid.atom_count();
    let n3 = 3 * grid.atom_count();
    let row_len = nsel * 3;

    let mut full = DerivOutput::new();
    ev.fill(
        DerivPoints::grid(&grid, range.clone()),
        &DerivRequest {
            ddw: true,
            ..DerivRequest::default()
        },
        &mut full,
    )
    .unwrap();
    assert_eq!(full.ddw().len(), row_len * n3 * block_len);

    // The visitor is per point; store its blocks point-major for comparison
    // with the raw `[row, point]` layout of `DerivOutput::ddw`.
    let mut visited = vec![0.0f64; block_len * row_len * n3];
    let mut ws = DerivWorkspace::new();
    let mut raw = DerivOutput::new();
    ev.fill_serial_visit(
        DerivPoints::grid(&grid, range.clone()),
        &mut ws,
        &DerivRequest::default(),
        &mut raw,
        |point, block| {
            let offset = (point - range.start) * block.len();
            visited[offset..offset + block.len()].copy_from_slice(block);
        },
    )
    .unwrap();

    for row in 0..row_len * n3 {
        for i in 0..block_len {
            let want = full.ddw()[row * block_len + i];
            let got = visited[i * row_len * n3 + row];
            assert!(
                (got - want).abs() <= 1e-13 * (1.0 + want.abs()),
                "ddw row {row} point {}: {got} vs {want}",
                range.start + i
            );
        }
    }
}

#[test]
fn serial_output_reuse_without_the_old_contract_does_not_panic() {
    let (charges, centers) = water();
    let plan = plan_for(&charges, &centers);
    let ev = evaluator(&centers, &charges, None);
    let range = 0..plan.len().min(64);
    let coeff = vec![1.0; range.len()];
    let mut ws = DerivWorkspace::new();
    let mut out = DerivOutput::new();

    let with_dc = DerivRequest {
        weights: true,
        dc: Some(DerivContract::exact(&coeff, 1)),
        ..Default::default()
    };
    ev.fill_serial(
        DerivPoints::plan(&plan, range.clone()),
        &mut ws,
        &with_dc,
        &mut out,
    )
    .unwrap();
    assert!(out.dc().is_some());

    // The next call omits `dc`; the old serial filler passed the stale
    // contract to the kernel and panicked.
    let dw_only = DerivRequest {
        weights: true,
        dw: true,
        ..Default::default()
    };
    ev.fill_serial(
        DerivPoints::plan(&plan, range.clone()),
        &mut ws,
        &dw_only,
        &mut out,
    )
    .unwrap();
    assert!(out.dc().is_none());
    assert_eq!(out.dw().len(), 3 * out.atoms().len() * range.len());

    // Same for a Hessian output followed by `ddw`-only reuse.
    let mut out = DerivOutput::new();
    let hessian = DerivRequest {
        ddc: Some(DerivContract::exact(&coeff, 1)),
        ..Default::default()
    };
    ev.fill_serial(
        DerivPoints::plan(&plan, range.clone()),
        &mut ws,
        &hessian,
        &mut out,
    )
    .unwrap();
    let ddw_only = DerivRequest {
        ddw: true,
        ..Default::default()
    };
    ev.fill_serial(
        DerivPoints::plan(&plan, range.clone()),
        &mut ws,
        &ddw_only,
        &mut out,
    )
    .unwrap();
    assert!(out.ddc().is_none());
}

#[test]
fn derivative_workspace_is_keyed_by_atom_count() {
    let (charges3, centers3) = water();
    let charges2 = [1u32, 1];
    let centers2 = [[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]];
    let plan3 = plan_for(&charges3, &centers3);
    let plan2 = plan_for(&charges2, &centers2);

    let ev3 = PartitionDerivEvaluator::from_parts(
        &centers3,
        &charges3,
        config(),
        DerivOptions {
            atoms: Some(vec![0, 1]),
            block_points: 8,
            worker_memory_bytes: 0,
            hessian_backend: HessianBackend::Auto,
        },
    )
    .unwrap();
    let ev2 = PartitionDerivEvaluator::from_parts(
        &centers2,
        &charges2,
        config(),
        DerivOptions {
            atoms: Some(vec![0, 1]),
            block_points: 8,
            worker_memory_bytes: 0,
            hessian_backend: HessianBackend::Auto,
        },
    )
    .unwrap();
    let req = DerivRequest {
        weights: true,
        dw: true,
        ..Default::default()
    };

    for (first, first_plan, second, second_plan) in
        [(&ev3, &plan3, &ev2, &plan2), (&ev2, &plan2, &ev3, &plan3)]
    {
        let mut shared = DerivWorkspace::new();
        let mut out = DerivOutput::new();
        first
            .fill_serial(
                DerivPoints::plan(first_plan, 0..first_plan.len().min(128)),
                &mut shared,
                &req,
                &mut out,
            )
            .unwrap();
        second
            .fill_serial(
                DerivPoints::plan(second_plan, 0..second_plan.len().min(128)),
                &mut shared,
                &req,
                &mut out,
            )
            .unwrap();

        let mut fresh = DerivWorkspace::new();
        let mut want = DerivOutput::new();
        second
            .fill_serial(
                DerivPoints::plan(second_plan, 0..second_plan.len().min(128)),
                &mut fresh,
                &req,
                &mut want,
            )
            .unwrap();
        assert_eq!(out.dw(), want.dw());
    }
}

/// F7: a global contract uses `range.start` as its first column.
#[test]
fn contract_over_uses_the_global_range_start() {
    let (charges, centers) = water();
    let plan = plan_for(&charges, &centers);
    let ev = evaluator(&centers, &charges, None);
    let start = 100;
    let end = (start + 20).min(plan.len());
    assert!(end > start + 8, "plan is too short: {}", plan.len());
    let full: Vec<f64> = (0..plan.len()).map(|i| 0.25 + 0.001 * i as f64).collect();
    let req = DerivRequest {
        dc: Some(DerivContract::over(&full, 1, plan.len(), start..end)),
        ..Default::default()
    };
    let mut out = DerivOutput::new();
    let mut ws = DerivWorkspace::new();
    ev.fill_serial(
        DerivPoints::plan(&plan, start..end),
        &mut ws,
        &req,
        &mut out,
    )
    .unwrap();

    let coeff: Vec<f64> = full[start..end].to_vec();
    let one = DerivRequest {
        dc: Some(DerivContract::exact(&coeff, 1)),
        ..Default::default()
    };
    let mut reference = DerivOutput::new();
    ev.fill_serial(
        DerivPoints::plan(&plan, start..end),
        &mut ws,
        &one,
        &mut reference,
    )
    .unwrap();
    assert_eq!(out.dc().unwrap().dc_set(0).len(), 3 * ev.atoms().len());
    for (a, b) in out
        .dc()
        .unwrap()
        .dc_set(0)
        .iter()
        .zip(reference.dc().unwrap().dc_set(0))
    {
        assert!((a - b).abs() < 1e-13, "{a} vs {b}");
    }
}

/// F3/F15: raw-family accessors report omission instead of panicking.
#[test]
fn unrequested_raw_families_return_none() {
    let (charges, centers) = water();
    let plan = plan_for(&charges, &centers);
    let ev = evaluator(&centers, &charges, None);
    let req = DerivRequest {
        weights: true,
        ..Default::default()
    };
    let mut out = DerivOutput::new();
    let mut ws = DerivWorkspace::new();
    ev.fill_serial(
        DerivPoints::plan(&plan, 0..plan.len().min(32)),
        &mut ws,
        &req,
        &mut out,
    )
    .unwrap();
    assert!(!out.partition_weights().is_empty());
    assert!(out.dw_atom(0).is_none());
    assert!(out.ddw_atom(0).is_none());
}

#[test]
fn two_atom_fill_matches_the_point_kernels() {
    let charges = [1u32, 1];
    let centers = [[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]];
    let plan = plan_for(&charges, &centers);
    let evaluator = evaluator(&centers, &charges, None);
    let range = 0..plan.len().min(48);
    let request = DerivRequest {
        weights: true,
        dw: true,
        ddw: true,
        ..DerivRequest::default()
    };
    let mut workspace = DerivWorkspace::new();
    let mut output = DerivOutput::new();
    evaluator
        .fill_serial(
            DerivPoints::plan(&plan, range.clone()),
            &mut workspace,
            &request,
            &mut output,
        )
        .unwrap();

    let len = range.len();
    for point in range {
        let parent = plan.parent_of(point) as usize;
        let local = plan.local_coord(point);
        let world = [
            centers[parent][0] + local[0],
            centers[parent][1] + local[1],
            centers[parent][2] + local[2],
        ];
        let (weight, gradient) = evaluator.gradient_at(world, parent);
        assert!(
            (output.partition_weights()[point] - weight).abs() <= 1e-12 * (1.0 + weight.abs()),
            "point {point}: weight"
        );
        let dw = output.dw_atom(parent).unwrap();
        for t in 0..3 {
            let got = dw[t * len + point];
            assert!(
                (got - gradient[t]).abs() <= 1e-11 * (1.0 + gradient[t].abs()),
                "point {point} t {t}: dw {got} vs {}",
                gradient[t]
            );
        }
        let hessian = evaluator.hessian_at(world, parent);
        let ddw = output.ddw_atom(parent).unwrap();
        for t in 0..3 {
            for atom in 0..centers.len() {
                for s in 0..3 {
                    let index = ((t * centers.len() + atom) * 3 + s) * len + point;
                    assert!(
                        (ddw[index] - hessian[(t * centers.len() + atom) * 3 + s]).abs()
                            <= 1e-10 * (1.0 + hessian[(t * centers.len() + atom) * 3 + s].abs()),
                        "point {point} hessian ({t},{atom},{s})"
                    );
                }
            }
        }
    }
}

#[test]
fn evaluator_rejects_bad_selection_and_block_size() {
    let charges = [1u32, 1];
    let centers = [[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]];
    let bad_atom = PartitionDerivEvaluator::from_parts(
        &centers,
        &charges,
        config(),
        DerivOptions {
            atoms: Some(vec![2]),
            ..DerivOptions::default()
        },
    )
    .unwrap_err();
    assert!(matches!(
        bad_atom,
        ruaa_dft_grids::Error::InvalidAtom { atom: 2, natm: 2 }
    ));

    let bad_block = PartitionDerivEvaluator::from_parts(
        &centers,
        &charges,
        config(),
        DerivOptions {
            block_points: MAX_BLOCK_POINTS + 1,
            ..DerivOptions::default()
        },
    )
    .unwrap_err();
    assert!(matches!(
        bad_block,
        ruaa_dft_grids::Error::InvalidArgument {
            argument: "block_points"
        }
    ));

    // A non-multiple is rounded up, not rejected.
    let rounded = PartitionDerivEvaluator::from_parts(
        &centers,
        &charges,
        config(),
        DerivOptions {
            block_points: 3,
            ..DerivOptions::default()
        },
    );
    assert!(rounded.is_ok());
}

#[test]
fn lko_sparse_gradient_dispatch_runs_at_32_atoms() {
    let (charges, centers) = h_ring(32, 5.0);
    let plan = plan_for(&charges, &centers);
    let evaluator = evaluator_with(&centers, &charges, lko_config(), None);
    let range = 0..plan.len().min(32);
    let request = DerivRequest {
        weights: true,
        dw: true,
        ..DerivRequest::default()
    };
    let mut workspace = DerivWorkspace::new();
    let mut output = DerivOutput::new();
    evaluator
        .fill_serial(
            DerivPoints::plan(&plan, range.clone()),
            &mut workspace,
            &request,
            &mut output,
        )
        .unwrap();
    assert_eq!(output.range(), range);
    assert!(output.dw().iter().all(|value| value.is_finite()));
    let parent = plan.parent_of(0) as usize;
    let local = plan.local_coord(0);
    let world = [
        centers[parent][0] + local[0],
        centers[parent][1] + local[1],
        centers[parent][2] + local[2],
    ];
    let (_, gradient) = evaluator.gradient_at(world, parent);
    let dw = output.dw_atom(parent).unwrap();
    let len = range.len();
    for t in 0..3 {
        assert!((dw[t * len] - gradient[t]).abs() <= 1e-11 * (1.0 + gradient[t].abs()));
    }
}

#[test]
fn lko_sparse_hessian_dispatch_runs_at_36_atoms() {
    let (charges, centers) = h_ring(36, 6.0);
    let plan = plan_for(&charges, &centers);
    let evaluator = evaluator_with(&centers, &charges, lko_config(), None);
    let range = 0..plan.len().min(8);
    let request = DerivRequest {
        dw: true,
        ddw: true,
        ..DerivRequest::default()
    };
    let mut workspace = DerivWorkspace::new();
    let mut output = DerivOutput::new();
    evaluator
        .fill_serial(
            DerivPoints::plan(&plan, range.clone()),
            &mut workspace,
            &request,
            &mut output,
        )
        .unwrap();
    assert!(output.ddw().iter().all(|value| value.is_finite()));
    let parent = plan.parent_of(0) as usize;
    let local = plan.local_coord(0);
    let world = [
        centers[parent][0] + local[0],
        centers[parent][1] + local[1],
        centers[parent][2] + local[2],
    ];
    let hessian = evaluator.hessian_at(world, parent);
    let ddw = output.ddw_atom(parent).unwrap();
    let len = range.len();
    for t in 0..3 {
        for atom in 0..centers.len() {
            for s in 0..3 {
                let index = ((t * centers.len() + atom) * 3 + s) * len;
                assert!((ddw[index] - hessian[(t * centers.len() + atom) * 3 + s]).abs() <= 1e-10);
            }
        }
    }
}

#[test]
fn stratmann_one_and_two_atom_hessians_are_finite() {
    for (charges, centers) in [
        (vec![2u32], vec![[0.0, 0.0, 0.0]]),
        (vec![1u32, 1], vec![[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]]),
    ] {
        let plan = plan_for(&charges, &centers);
        let evaluator = evaluator_with(&centers, &charges, stratmann_config(), None);
        let range = 0..plan.len().min(16);
        let request = DerivRequest {
            weights: true,
            dw: true,
            ddw: true,
            ..DerivRequest::default()
        };
        let mut workspace = DerivWorkspace::new();
        let mut output = DerivOutput::new();
        evaluator
            .fill_serial(
                DerivPoints::plan(&plan, range.clone()),
                &mut workspace,
                &request,
                &mut output,
            )
            .unwrap();
        assert!(
            output
                .partition_weights()
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(output.dw().iter().all(|value| value.is_finite()));
        assert!(output.ddw().iter().all(|value| value.is_finite()));
    }
}
