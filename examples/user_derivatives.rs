//! `MolecularGrid` -> analytic partition-weight derivatives.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example user_derivatives
//! ```
//!
//! The quadrature weight `q` is fixed in each atom's local frame, so only the
//! normalized partition fraction `P` depends on the nuclear coordinates.  The
//! derivative API returns the derivative of the **total** grid weight
//! `w = q * P`; divide by `q` to obtain `dP/dR`.
//!
//! The grids used here use differentiable partitions.  ORCA's program default
//! early-exit sweep is deliberately rejected (see the end of the example).
use ruaa_dft_grids::partition::DerivPoints;

use ruaa_dft_grids::grid::{GridSpec, PyscfGridSpec};
use ruaa_dft_grids::moleculegrid::build_molecular_grid;
use ruaa_dft_grids::partition::{
    DerivContract, DerivOptions, DerivOutput, DerivRequest, PartitionConfig,
    PartitionConfigBuilder, PartitionScheme, RadiiConfig,
};

const BOHR_PER_ANGSTROM: f64 = 1.889_726_125_457_828_1;

fn water() -> [(u32, [f64; 3]); 3] {
    let h = [0.757 * BOHR_PER_ANGSTROM, 0.587 * BOHR_PER_ANGSTROM];
    [
        (8, [0.0, 0.0, 0.0]),
        (1, [0.0, h[0], h[1]]),
        (1, [0.0, -h[0], h[1]]),
    ]
}

fn main() -> ruaa_dft_grids::Result<()> {
    let atoms = water();
    let grid = build_molecular_grid(&atoms, PyscfGridSpec::from_level(0)?)?;
    let points = grid.len();
    let atoms_n = grid.atom_count();

    // One call: all atoms selected, partition fractions and raw first
    // derivatives, evaluated in parallel on the current point order.
    let deriv = grid.weight_derivatives()?;
    assert_eq!(deriv.range(), 0..points);
    assert_eq!(deriv.atoms(), &[0, 1, 2]);
    println!("grid: {points} points, {atoms_n} atoms");
    println!(
        "output: partition_weights [{}], dw [{}] = [natm, 3, G]",
        deriv.partition_weights().len(),
        deriv.dw().len()
    );

    // The bare partition fractions reproduce the value path.
    let max_weight_error = deriv
        .partition_weights()
        .iter()
        .zip(grid.partition_fractions())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("max |P_deriv - P_grid|      {max_weight_error:.1e}");

    // The grid weights sum to q per point, independent of a common rigid
    // translation: for each point, sum_A dw[A, t] must vanish.
    let g = points;
    let mut max_translation = 0.0_f64;
    for point in 0..g {
        for t in 0..3 {
            let sum: f64 = (0..atoms_n)
                .map(|a| deriv.dw()[(a * 3 + t) * g + point])
                .sum();
            max_translation = max_translation.max(sum.abs());
        }
    }
    println!("max |sum_A dw[A, t]|        {max_translation:.1e}");

    // Integral gradients: sum over points.  These are the derivatives of the
    // grid's total weight, one row per atom.
    for atom in 0..atoms_n {
        let row = deriv.dw_atom(atom).unwrap();
        let gx: f64 = row[0..g].iter().sum();
        let gy: f64 = row[g..2 * g].iter().sum();
        let gz: f64 = row[2 * g..3 * g].iter().sum();
        println!("d(total weight)/d atom {atom} = [{gx:+.6}, {gy:+.6}, {gz:+.6}]");
    }

    // Finite-difference check of one component: move one atom by +-eps.
    let (atom, component, eps) = (0usize, 2usize, 1e-5);
    let mut plus_atoms = atoms;
    plus_atoms[atom].1[component] += eps;
    let mut minus_atoms = atoms;
    minus_atoms[atom].1[component] -= eps;
    let plus = build_molecular_grid(&plus_atoms, PyscfGridSpec::from_level(0)?)?;
    let minus = build_molecular_grid(&minus_atoms, PyscfGridSpec::from_level(0)?)?;
    let row = deriv.dw_atom(atom).unwrap();
    let max_fd_error = (0..g)
        .map(|point| {
            let finite_difference = (plus.weights()[point] - minus.weights()[point]) / (2.0 * eps);
            (row[component * g + point] - finite_difference).abs()
        })
        .fold(0.0_f64, f64::max);
    println!("max |analytic - central FD| {max_fd_error:.1e} (atom {atom}, component {component})");

    // A contraction is the streaming response-style use: sum_g c_g dw.
    // One coefficient set, all coefficients 1.
    let coefficients = vec![1.0; g];
    let evaluator = grid.partition_deriv_evaluator(DerivOptions::default())?;
    let request = DerivRequest {
        dc: Some(DerivContract::exact(&coefficients, 1)),
        ..DerivRequest::default()
    };
    let mut contracted = DerivOutput::new();
    evaluator.fill(DerivPoints::all_grid(&grid), &request, &mut contracted)?;
    let dc = contracted.dc().expect("dc was requested").dc_set(0);
    println!(
        "contracted dc[atom 0]        [{:+.6}, {:+.6}, {:+.6}]",
        dc[0], dc[1], dc[2]
    );

    // A selected-row evaluator differentiates only the requested atoms.
    let ev_one = grid.partition_deriv_evaluator(DerivOptions {
        atoms: Some(vec![0]),
        ..DerivOptions::default()
    })?;
    let mut one = DerivOutput::new();
    ev_one.fill(
        DerivPoints::all_grid(&grid),
        &DerivRequest {
            dw: true,
            ..DerivRequest::default()
        },
        &mut one,
    )?;
    println!("selected atom 0: dw [{}] = [1, 3, G]", one.dw().len());

    // mBecke at the default clamp is differentiable; the early-exit flag
    // belongs to the classic Becke kernel and is inert for mBecke.
    let mbecke = GridSpec::from(PyscfGridSpec::from_level(0)?).with_partition(
        PartitionConfig::new(PartitionScheme::MBecke, RadiiConfig::pyscf()),
    );
    let mbecke_grid = build_molecular_grid(&atoms, &mbecke)?;
    let mbecke_deriv = mbecke_grid.weight_derivatives()?;
    println!(
        "mBecke partition: {} points, dw [{}]",
        mbecke_grid.len(),
        mbecke_deriv.dw().len()
    );

    // Classic Becke with an early-exit pair sweep truncates the product.
    // The derivative API refuses it instead of silently returning a wrong
    // value; use the same scheme without the early exit when derivatives are
    // needed.
    let classic_early = GridSpec::from(PyscfGridSpec::from_level(0)?).with_partition(
        PartitionConfigBuilder::default()
            .scheme(PartitionScheme::Becke)
            .radii(RadiiConfig::pyscf())
            .becke_early_exit(true)
            .build()
            .expect("classic Becke partition is valid"),
    );
    let early_grid = build_molecular_grid(&atoms, &classic_early)?;
    match early_grid.weight_derivatives() {
        Ok(_) => println!("unexpected: early-exit derivatives were accepted"),
        Err(error) => println!("classic Becke + early exit: {error}"),
    }
    Ok(())
}
