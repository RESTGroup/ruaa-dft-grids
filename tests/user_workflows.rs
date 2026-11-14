//! User-facing workflow checks.
//!
//! These tests mirror the `user_*` examples and the user guide:
//!
//! * geometry + a PySCF preset or a data-driven preset -> `MolecularGrid`;
//! * `MolecularGrid` -> analytic weight derivatives;
//! * sorting a grid and deriving in the sorted slot order;
//! * pruning first, then sorting and deriving.
//!
//! They intentionally assert observable consumer behavior (counts, shapes,
//! invariants) rather than internal kernels.

use ruaa_dft_grids::Error;
use ruaa_dft_grids::grid::{GridSpec, PyscfGridSpec, PyscfGridSpecBuilder};
use ruaa_dft_grids::moleculegrid::{MolecularGrid, PostPrune, build_molecular_grid};
use ruaa_dft_grids::order::GridOrder;
use ruaa_dft_grids::partition::{
    DerivContract, DerivOptions, DerivOutput, DerivPoints, DerivRequest, PartitionConfig,
    PartitionConfigBuilder, PartitionScheme, RadiiConfig,
};
use ruaa_dft_grids::preset::{ElementPreset, GridPreset};
use ruaa_dft_grids::radial::Radial;

const BOHR_PER_ANGSTROM: f64 = 1.889_726_125_457_828_1;

/// Water in a common input orientation (O, H, H).
fn water() -> [(u32, [f64; 3]); 3] {
    let h = [0.757 * BOHR_PER_ANGSTROM, 0.587 * BOHR_PER_ANGSTROM];
    [
        (8, [0.0, 0.0, 0.0]),
        (1, [0.0, h[0], h[1]]),
        (1, [0.0, -h[0], h[1]]),
    ]
}

fn assert_close(got: f64, want: f64, tolerance: f64, what: &str) {
    assert!(
        (got - want).abs() <= tolerance,
        "{what}: got {got:?}, want {want:?}"
    );
}

/// A small data-driven preset for H and O (20 shells, 50-point rule).
fn small_preset() -> GridPreset {
    let h_xi = Radial::TreutlerAhlrichs.xi_for_element(1).unwrap();
    let o_xi = Radial::TreutlerAhlrichs.xi_for_element(8).unwrap();
    GridPreset::new(Radial::TreutlerAhlrichs)
        .with_name("small")
        .with_element(1, ElementPreset::fixed(20, h_xi, 50))
        .with_element(8, ElementPreset::fixed(20, o_xi, 50))
}

/// The metadata contract every consumer may rely on.
fn assert_grid_metadata(grid: &MolecularGrid) {
    assert_eq!(grid.coords().len(), grid.weights().len());
    assert_eq!(grid.coords().len(), grid.quad_weights().len());
    assert_eq!(grid.coords().len(), grid.partition_fractions().len());
    assert_eq!(grid.coords().len(), grid.atom_idx().len());
    assert_eq!(grid.atom_coords().len(), grid.atom_charges().len());

    for point in grid.points() {
        assert!((point.atom as usize) < grid.atom_count());
        assert_close(
            point.weight,
            point.quad_weight * point.partition_fraction,
            1e-12 * point.weight.abs().max(1.0),
            "weight == q * P",
        );
    }

    let total: f64 = grid.weights().iter().sum();
    assert_close(
        grid.total_weight(),
        total,
        1e-12 * total.abs().max(1.0),
        "total",
    );
}

#[test]
fn geometry_to_molecular_grid_with_pyscf_and_preset() {
    let atoms = water();

    let pyscf = PyscfGridSpec::from_level(0).unwrap();
    let grid = build_molecular_grid(&atoms, &pyscf).unwrap();
    assert_eq!(grid.spec().pyscf_spec(), Some(&pyscf));
    assert!(grid.len() > 1000);
    assert_grid_metadata(&grid);

    let preset = small_preset();
    let spec = GridSpec::from_preset(preset.clone());
    let grid = build_molecular_grid(&atoms, &spec).unwrap();
    assert_eq!(grid.spec().preset(), Some(&preset));
    assert_eq!(grid.len(), 3 * 20 * 50);
    assert_grid_metadata(&grid);
}

#[test]
fn presets_compose_topology_and_partition() {
    // The PySCF level constructor equals the builder form.
    assert_eq!(
        PyscfGridSpec::from_level(3).unwrap(),
        PyscfGridSpecBuilder::default().level(3).build().unwrap()
    );

    // A data-driven topology can be combined with any partition without
    // rebuilding the shell data.
    let preset = small_preset();
    let spec = GridSpec::from_preset(preset.clone()).with_partition(PartitionConfig::new(
        PartitionScheme::MBecke,
        RadiiConfig::pyscf(),
    ));
    assert_eq!(spec.preset(), Some(&preset));
    assert_eq!(spec.partition().scheme, PartitionScheme::MBecke);

    // The topology is geometry-independent: the same spec can build a moved
    // molecule.
    let grid = build_molecular_grid(&water(), &spec).unwrap();
    assert_eq!(grid.len(), 3 * 20 * 50);
    assert!(!spec.geometry_dependent());
}

#[test]
fn molecular_grid_to_weight_derivatives() {
    let atoms = water();
    let grid = build_molecular_grid(&atoms, PyscfGridSpec::from_level(0).unwrap()).unwrap();
    let g = grid.len();
    let natm = grid.atom_count();

    let deriv = grid.weight_derivatives().unwrap();
    assert_eq!(deriv.range(), 0..g);
    assert_eq!(deriv.atoms(), &[0, 1, 2]);
    assert_eq!(deriv.partition_weights().len(), g);
    assert_eq!(deriv.dw().len(), natm * 3 * g);
    for (got, want) in deriv
        .partition_weights()
        .iter()
        .zip(grid.partition_fractions())
    {
        assert_close(*got, *want, 1e-12, "partition weight");
    }

    // Translation invariance: for every point and Cartesian component, the
    // sum over atoms of dw is zero.
    let mut max_translation = 0.0_f64;
    for point in 0..g {
        for t in 0..3 {
            let sum: f64 = (0..natm)
                .map(|atom| deriv.dw()[(atom * 3 + t) * g + point])
                .sum();
            max_translation = max_translation.max(sum.abs());
        }
    }
    assert!(
        max_translation < 1e-10,
        "translation sum {max_translation:e}"
    );

    // One component against a central finite difference of the built grid.
    let (atom, component, eps) = (0usize, 2usize, 1e-5);
    let mut plus_atoms = atoms;
    plus_atoms[atom].1[component] += eps;
    let mut minus_atoms = atoms;
    minus_atoms[atom].1[component] -= eps;
    let plus = build_molecular_grid(&plus_atoms, PyscfGridSpec::from_level(0).unwrap()).unwrap();
    let minus = build_molecular_grid(&minus_atoms, PyscfGridSpec::from_level(0).unwrap()).unwrap();
    let row = deriv.dw_atom(atom).unwrap();
    let max_error = (0..g)
        .map(|point| {
            let fd = (plus.weights()[point] - minus.weights()[point]) / (2.0 * eps);
            (row[component * g + point] - fd).abs()
        })
        .fold(0.0_f64, f64::max);
    assert!(max_error < 1e-6, "finite-difference error {max_error:e}");

    // A selection writes only the requested rows.
    let selected = grid
        .partition_deriv_evaluator(DerivOptions {
            atoms: Some(vec![0]),
            ..DerivOptions::default()
        })
        .unwrap();
    let mut out = DerivOutput::new();
    selected
        .fill(
            DerivPoints::all_grid(&grid),
            &DerivRequest {
                dw: true,
                ..DerivRequest::default()
            },
            &mut out,
        )
        .unwrap();
    assert_eq!(out.atoms(), &[0]);
    assert_eq!(out.dw().len(), 3 * g);
    for t in 0..3 {
        for point in 0..g {
            assert_close(
                out.dw()[t * g + point],
                deriv.dw()[t * g + point],
                0.0,
                "selected row",
            );
        }
    }

    // A response-style contraction: sum_g c_g dw, one coefficient set.
    let coefficients = vec![1.0; g];
    let evaluator = grid
        .partition_deriv_evaluator(DerivOptions::default())
        .unwrap();
    let mut contracted = DerivOutput::new();
    evaluator
        .fill(
            DerivPoints::all_grid(&grid),
            &DerivRequest {
                dc: Some(DerivContract::exact(&coefficients, 1)),
                ..DerivRequest::default()
            },
            &mut contracted,
        )
        .unwrap();
    let dc = contracted.dc().expect("dc was requested").dc_set(0);
    for atom in 0..natm {
        for t in 0..3 {
            let want: f64 = (0..g)
                .map(|point| deriv.dw()[(atom * 3 + t) * g + point])
                .sum();
            assert_close(dc[atom * 3 + t], want, 1e-10, "contracted gradient");
        }
    }
}

#[test]
fn mbecke_partition_is_differentiable() {
    let partition = PartitionConfig::new(PartitionScheme::MBecke, RadiiConfig::pyscf());
    let spec = GridSpec::from(PyscfGridSpec::from_level(0).unwrap()).with_partition(partition);
    let grid = build_molecular_grid(&water(), &spec).unwrap();
    let deriv = grid.weight_derivatives().unwrap();
    assert_eq!(deriv.dw().len(), 3 * 3 * grid.len());
}

#[test]
fn classic_becke_early_exit_refuses_derivatives() {
    let partition = PartitionConfigBuilder::default()
        .scheme(PartitionScheme::Becke)
        .radii(RadiiConfig::pyscf())
        .becke_early_exit(true)
        .build()
        .unwrap();
    let spec = GridSpec::from(PyscfGridSpec::from_level(0).unwrap()).with_partition(partition);
    let grid = build_molecular_grid(&water(), &spec).unwrap();
    let error = grid.weight_derivatives().unwrap_err();
    assert!(matches!(
        error,
        Error::UnsupportedDerivative {
            scheme: PartitionScheme::Becke,
            early_exit: true
        }
    ));
}

#[test]
fn sorting_preserves_totals_and_derivatives() {
    let atoms = water();
    let original = build_molecular_grid(&atoms, PyscfGridSpec::from_level(0).unwrap()).unwrap();
    let total = original.total_weight();
    let total_quad = original.total_quad_weight();
    let by_atom = original.quad_weight_by_atom();
    let deriv_before = original.weight_derivatives().unwrap();

    let mut grid = original.clone();
    let order = grid.reorder_boxes(1.2).unwrap();
    assert_eq!(order.len(), grid.len());
    assert_eq!(grid.len(), original.len());

    // Every per-point array moved together; totals are invariant.
    for (slot, &source) in order.perm().iter().enumerate() {
        let source = source as usize;
        assert_eq!(grid.coords()[slot], original.coords()[source]);
        assert_eq!(grid.weights()[slot], original.weights()[source]);
        assert_eq!(grid.atom_idx()[slot], original.atom_idx()[source]);
    }
    assert_close(
        grid.total_weight(),
        total,
        1e-10 * total.abs().max(1.0),
        "total weight",
    );
    assert_close(
        grid.total_quad_weight(),
        total_quad,
        1e-10 * total_quad.abs().max(1.0),
        "total quadrature weight",
    );
    for (before, after) in by_atom.iter().zip(grid.quad_weight_by_atom()) {
        assert_close(
            after,
            *before,
            1e-10 * before.abs().max(1.0),
            "per-atom quadrature weight",
        );
    }

    // The first occupied box is contiguous in the sorted order.
    let cell = |point: [f64; 3]| {
        (
            (point[0] / 1.2).floor() as i64,
            (point[1] / 1.2).floor() as i64,
            (point[2] / 1.2).floor() as i64,
        )
    };
    let first = cell(grid.coords()[0]);
    let count = grid
        .coords()
        .iter()
        .take_while(|point| cell(**point) == first)
        .count();
    assert!(count >= 1);
    assert!(
        count == grid.len() || cell(grid.coords()[count]) != first,
        "the first box must be contiguous"
    );

    // Deriving in the sorted order is a pure remap of the unsorted result.
    let deriv_after = grid.weight_derivatives().unwrap();
    let g = grid.len();
    for (slot, &source) in order.perm().iter().enumerate() {
        let source = source as usize;
        for atom in 0..3 {
            for t in 0..3 {
                let got = deriv_after.dw()[(atom * 3 + t) * g + slot];
                let want = deriv_before.dw()[(atom * 3 + t) * g + source];
                assert_close(got, want, 1e-12 * want.abs().max(1.0), "sorted dw");
            }
        }
    }

    // Re-sorting a sorted grid is the identity.
    assert_eq!(
        GridOrder::boxes(grid.coords(), 1.2).unwrap(),
        GridOrder::identity(grid.len()).unwrap()
    );

    // Morton is another geometry-only order with the same invariants.
    let mut morton = original.clone();
    let morton_order = morton.reorder_morton(1.2).unwrap();
    assert_eq!(morton_order.len(), morton.len());
    assert_close(
        morton.total_weight(),
        total,
        1e-10 * total.abs().max(1.0),
        "Morton total",
    );

    // Caller-supplied keys are accepted verbatim (here: group by generator).
    let keys = original.atom_idx().iter().map(|&atom| u64::from(atom));
    let mut by_generator = original.clone();
    by_generator
        .reorder(&GridOrder::from_keys(keys).unwrap())
        .unwrap();
    assert!(
        by_generator
            .atom_idx()
            .windows(2)
            .all(|window| window[0] <= window[1])
    );
}

#[test]
fn prune_then_sort_then_derive() {
    let atoms = water();
    let partition = PartitionConfig::new(PartitionScheme::MBecke, RadiiConfig::pyscf());
    let spec = GridSpec::from(PyscfGridSpec::from_level(0).unwrap()).with_partition(partition);
    let full = build_molecular_grid(&atoms, &spec).unwrap();
    let full_len = full.len();

    let split = full.split(PostPrune::PartitionCut(1e-14)).unwrap();
    let mut grid = split.retained;
    assert!(grid.len() < full_len);
    assert_eq!(split.pruned.len() + grid.len(), full_len);
    let total = grid.total_weight();

    let order = grid.reorder_boxes(1.2).unwrap();
    assert_eq!(order.len(), grid.len());
    assert_close(
        grid.total_weight(),
        total,
        1e-10 * total.abs().max(1.0),
        "post-prune total",
    );

    let deriv = grid.weight_derivatives().unwrap();
    let g = grid.len();
    assert_eq!(deriv.dw().len(), 3 * 3 * g);
    let mut max_translation = 0.0_f64;
    for point in 0..g {
        for t in 0..3 {
            let sum: f64 = (0..3)
                .map(|atom| deriv.dw()[(atom * 3 + t) * g + point])
                .sum();
            max_translation = max_translation.max(sum.abs());
        }
    }
    assert!(
        max_translation < 1e-10,
        "translation sum {max_translation:e}"
    );
}
