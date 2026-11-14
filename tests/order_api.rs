//! Integration checks for the order/permutation utilities.
//!
//! The order is a pure permutation: the point set, the per-point arrays and
//! the integrated quantities must be invariant, and the inverse must map a
//! sorted slot back to its original index.

use ruaa_dft_grids::grid::GridSpec;
use ruaa_dft_grids::moleculegrid::{MolecularGrid, build_molecular_grid};
use ruaa_dft_grids::order::GridOrder;

fn synthetic() -> MolecularGrid {
    let molecule =
        ruaa_dft_grids::molecule::Molecule::new([(1u32, [0.0, 0.0, 0.0]), (1, [5.0, 5.0, 5.0])])
            .unwrap();
    MolecularGrid::from_parts(
        GridSpec::default(),
        molecule,
        vec![
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [5.1, 0.0, 0.0],
            [0.1, 5.1, 0.0],
            [5.1, 5.1, 5.1],
            [0.05, 0.0, 0.0],
        ],
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        vec![0, 0, 1, 1, 0, 0],
    )
    .unwrap()
}

#[test]
fn reorder_permutes_every_point_array() {
    let mut grid = synthetic();
    let before = grid.clone();
    let order = GridOrder::boxes(grid.coords(), 1.0).unwrap();

    // Cells: (0,0,0){0,1,5}, (0,5,0){3}, (5,0,0){2}, (5,5,5){4}.
    assert_eq!(order.perm(), [0, 1, 5, 3, 2, 4]);
    assert_eq!(order.inverse(), [0, 1, 4, 3, 5, 2]);

    grid.reorder(&order).unwrap();
    assert_eq!(grid.len(), before.len());
    for (slot, &source) in order.perm().iter().enumerate() {
        let source = source as usize;
        assert_eq!(grid.coords()[slot], before.coords()[source]);
        assert_eq!(grid.weights()[slot], before.weights()[source]);
        assert_eq!(grid.quad_weights()[slot], before.quad_weights()[source]);
        assert_eq!(
            grid.partition_fractions()[slot],
            before.partition_fractions()[source]
        );
        assert_eq!(grid.atom_idx()[slot], before.atom_idx()[source]);
    }
    let total_before: f64 = before.weights().iter().sum();
    let total_after: f64 = grid.weights().iter().sum();
    assert!((total_after - total_before).abs() < 1e-12);
    assert_eq!(grid.spec(), before.spec());
    assert_eq!(grid.atom_coords(), before.atom_coords());
    assert_eq!(grid.atom_charges(), before.atom_charges());
}

#[test]
fn reorder_preserves_a_real_grid() {
    let atoms = [(1u32, [-0.7, 0.0, 0.0]), (1, [0.7, 0.0, 0.0])];
    let spec = GridSpec::default();
    let mut grid = build_molecular_grid(&atoms, &spec).unwrap();

    let total = grid.total_weight();
    let by_atom = grid.quad_weight_by_atom();
    let order = GridOrder::boxes(grid.coords(), 1.2).unwrap();
    grid.reorder(&order).unwrap();

    assert_eq!(grid.len(), order.len());
    assert!((grid.total_weight() - total).abs() <= 1e-12 * total.abs().max(1.0));
    let after = grid.quad_weight_by_atom();
    for (a, b) in by_atom.iter().zip(&after) {
        assert!((a - b).abs() <= 1e-12 * a.abs().max(1.0));
    }

    // Re-sorting an already sorted grid is the identity.
    assert_eq!(
        GridOrder::boxes(grid.coords(), 1.2).unwrap(),
        GridOrder::identity(grid.len()).unwrap()
    );
}

#[test]
fn reorder_with_keys_is_a_permutation() {
    let mut grid = synthetic();
    let before = grid.clone();
    // A caller-supplied screening-style key: bucket points by their source
    // atom.
    let keys = before.atom_idx().iter().map(|&a| u64::from(a));
    let order = GridOrder::from_keys(keys).unwrap();
    grid.reorder(&order).unwrap();

    for (slot, &source) in order.perm().iter().enumerate() {
        assert_eq!(grid.coords()[slot], before.coords()[source as usize]);
    }
    let atom_order: Vec<u32> = grid.atom_idx().to_vec();
    assert!(atom_order.windows(2).all(|w| w[0] <= w[1]));
}

#[test]
fn reorder_rejects_wrong_length() {
    let mut grid = synthetic();
    let err = grid.reorder(&GridOrder::identity(3).unwrap()).unwrap_err();
    assert_eq!(
        err,
        ruaa_dft_grids::Error::LengthMismatch {
            expected: grid.len(),
            got: 3
        }
    );
}
