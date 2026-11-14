//! Public post-build pruning contract.
//!
//! Pruning is applied to a materialised [`MolecularGrid`], never during the
//! build. These tests pin the four [`PostPrune`] policies, their strict
//! comparison semantics, error cases, and the retained /
//! pruned split.

use ruaa_dft_grids::Error;
use ruaa_dft_grids::grid::GridSpec;
use ruaa_dft_grids::moleculegrid::{MolecularGrid, PostPrune, build_molecular_grid};

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
            [5.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.4, 0.0, 0.0],
        ],
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        vec![1.0, 4.0, 1.0, 2.0, 1.0, 6.0],
        vec![0.1, 0.05, 0.3, 0.2, 0.5, 0.1],
        vec![0, 0, 1, 1, 0, 0],
    )
    .unwrap()
}

#[test]
fn total_weight_cut_uses_the_product_not_the_fraction() {
    let src = synthetic();
    let split = src.clone().split(PostPrune::TotalWeightCut(0.25)).unwrap();
    assert_eq!(
        split.retained.coords(),
        vec![
            [5.1, 0.0, 0.0],
            [5.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.4, 0.0, 0.0],
        ]
    );
    assert_eq!(
        split.pruned.coords(),
        vec![[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]
    );
    assert_eq!(split.retained.len() + split.pruned.len(), src.len());
    assert_eq!(
        split.retained.total_weight().to_bits(),
        (src.weights()[2] + src.weights()[3] + src.weights()[4] + src.weights()[5]).to_bits()
    );

    let by_fraction = src.clone().split(PostPrune::PartitionCut(0.25)).unwrap();
    assert_eq!(
        by_fraction.retained.coords(),
        vec![[5.1, 0.0, 0.0], [0.3, 0.0, 0.0]]
    );
    assert_eq!(
        by_fraction.pruned.coords(),
        vec![
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [5.2, 0.0, 0.0],
            [0.4, 0.0, 0.0]
        ]
    );
}

#[test]
fn partition_cut_is_strict() {
    let split = synthetic().split(PostPrune::PartitionCut(0.1)).unwrap();
    // P == 0.1 at points 0, 1 and 5 does not pass.
    assert_eq!(
        split.retained.coords(),
        vec![[5.1, 0.0, 0.0], [5.2, 0.0, 0.0], [0.3, 0.0, 0.0]]
    );
    assert_eq!(split.pruned.atom_idx(), vec![0, 0, 0]);
}

#[test]
fn keep_indices_preserves_original_order() {
    let split = synthetic()
        .split(PostPrune::KeepIndices(vec![4, 0]))
        .unwrap();
    assert_eq!(
        split.retained.coords(),
        vec![[0.1, 0.0, 0.0], [0.3, 0.0, 0.0]]
    );
    assert_eq!(split.pruned.len(), 4);
    assert_eq!(
        split.retained.total_weight().to_bits(),
        (0.1f64 + 0.5).to_bits()
    );
}

#[test]
fn explicit_selection_selects_listed_points() {
    let split = synthetic()
        .split(PostPrune::KeepMask(vec![
            false, false, false, false, false, true,
        ]))
        .unwrap();
    assert_eq!(split.retained.len(), 1);
    let point = split.retained.points().next().unwrap();
    assert_eq!(point.atom, 0);
    assert_eq!(point.weight.to_bits(), 0.6f64.to_bits());

    let by_index = synthetic().split(PostPrune::KeepIndices(vec![5])).unwrap();
    assert_eq!(by_index.retained, split.retained);
}

#[test]
fn split_off_pruned_matches_split() {
    let src = synthetic();
    for policy in [
        PostPrune::TotalWeightCut(0.25),
        PostPrune::PartitionCut(0.1),
        PostPrune::KeepIndices(vec![5, 2, 0]),
        PostPrune::KeepMask(vec![true, false, true, false, true, false]),
    ] {
        let split = src.clone().split(policy.clone()).unwrap();
        let mut in_place = src.clone();
        let pruned = in_place.split_off_pruned(policy).unwrap();
        assert_eq!(in_place, split.retained);
        assert_eq!(pruned, split.pruned);
    }
}

#[test]
fn invalid_policies_are_rejected() {
    let src = synthetic();
    for cutoff in [f64::NAN, f64::NEG_INFINITY, -1.0, f64::INFINITY] {
        assert!(matches!(
            src.keep_mask(&PostPrune::TotalWeightCut(cutoff))
                .unwrap_err(),
            Error::InvalidCutoff { .. }
        ));
        assert!(matches!(
            src.keep_mask(&PostPrune::PartitionCut(cutoff)).unwrap_err(),
            Error::InvalidCutoff { .. }
        ));
    }
    assert_eq!(
        src.keep_mask(&PostPrune::KeepIndices(vec![6])).unwrap_err(),
        Error::PruneIndexOutOfRange {
            index: 6,
            len: src.len()
        }
    );
    assert_eq!(
        src.keep_mask(&PostPrune::KeepIndices(vec![3, 3]))
            .unwrap_err(),
        Error::PruneIndexDuplicate { index: 3 }
    );
    assert_eq!(
        src.keep_mask(&PostPrune::KeepMask(vec![true; 5]))
            .unwrap_err(),
        Error::LengthMismatch {
            expected: src.len(),
            got: 5
        }
    );
}

#[test]
fn inconsistent_hand_built_grid_is_rejected() {
    let molecule =
        ruaa_dft_grids::molecule::Molecule::new([(1u32, [0.0, 0.0, 0.0]), (1, [5.0, 5.0, 5.0])])
            .unwrap();
    let spec = GridSpec::default();

    let bad_length = MolecularGrid::from_parts(
        spec.clone(),
        molecule.clone(),
        vec![[0.0, 0.0, 0.0]],
        vec![1.0],
        vec![1.0],
        vec![1.0],
        vec![0, 0],
    );
    assert!(matches!(
        bad_length,
        Err(Error::GridArrayLength {
            array: "atom_idx",
            ..
        })
    ));

    let bad_atom = MolecularGrid::from_parts(
        spec.clone(),
        molecule.clone(),
        vec![[0.0, 0.0, 0.0]],
        vec![1.0],
        vec![1.0],
        vec![1.0],
        vec![2],
    );
    assert_eq!(
        bad_atom.unwrap_err(),
        Error::InvalidGridAtom { atom: 2, natm: 2 }
    );

    let bad_value = MolecularGrid::from_parts(
        spec,
        molecule,
        vec![[0.0, 0.0, 0.0]],
        vec![f64::NAN],
        vec![1.0],
        vec![1.0],
        vec![0],
    );
    assert!(matches!(bad_value, Err(Error::InvalidArgument { .. })));
}

#[test]
fn empty_and_full_splits_are_valid() {
    let src = synthetic();
    let none = src.clone().split(PostPrune::PartitionCut(1.0)).unwrap();
    assert!(none.retained.is_empty());
    assert_eq!(none.pruned, src);

    let all = src
        .clone()
        .split(PostPrune::KeepMask(vec![true; 6]))
        .unwrap();
    assert!(all.pruned.is_empty());
    assert_eq!(all.retained, src);
}

#[test]
fn built_grid_partition_cut_leaves_a_consistent_grid() {
    let atoms = [(1u32, [-0.7, 0.0, 0.0]), (1, [0.7, 0.0, 0.0])];
    let spec = GridSpec::default();
    let full = build_molecular_grid(&atoms, &spec).unwrap();
    let split = full.clone().split(PostPrune::PartitionCut(1e-14)).unwrap();
    assert!(split.retained.len() < full.len());
    assert_eq!(split.retained.len() + split.pruned.len(), full.len());
    for point in split.retained.points() {
        assert!(point.partition_fraction > 1e-14);
        assert_eq!(
            point.weight.to_bits(),
            (point.quad_weight * point.partition_fraction).to_bits()
        );
    }
    for point in split.pruned.points() {
        assert!(point.partition_fraction <= 1e-14);
    }
}
