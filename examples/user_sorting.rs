//! Sort a `MolecularGrid` without changing what it integrates.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example user_sorting
//! ```
//!
//! An order is a pure permutation: `reorder` moves every per-point array, so
//! totals, per-atom quadrature weights, and analytic derivatives are
//! unchanged.  The order object also exposes the inverse map and fixed-size
//! blocks for consumers that want cache-local point batches.

use ruaa_dft_grids::grid::PyscfGridSpec;
use ruaa_dft_grids::moleculegrid::build_molecular_grid;
use ruaa_dft_grids::order::GridOrder;

const BOHR_PER_ANGSTROM: f64 = 1.889_726_125_457_828_1;
const BOX: f64 = 1.2; // PySCF's GROUP_BOX_SIZE

fn water() -> [(u32, [f64; 3]); 3] {
    let h = [0.757 * BOHR_PER_ANGSTROM, 0.587 * BOHR_PER_ANGSTROM];
    [
        (8, [0.0, 0.0, 0.0]),
        (1, [0.0, h[0], h[1]]),
        (1, [0.0, -h[0], h[1]]),
    ]
}

fn box_of(point: [f64; 3]) -> (i64, i64, i64) {
    (
        (point[0] / BOX).floor() as i64,
        (point[1] / BOX).floor() as i64,
        (point[2] / BOX).floor() as i64,
    )
}

fn main() -> ruaa_dft_grids::Result<()> {
    let atoms = water();
    let mut grid = build_molecular_grid(&atoms, PyscfGridSpec::from_level(1)?)?;
    let original = grid.clone();
    let total = grid.total_weight();
    let total_quad = grid.total_quad_weight();
    let by_atom = grid.quad_weight_by_atom();
    println!(
        "grid: {} points, total weight {:.10}, total quadrature {:.10}",
        grid.len(),
        total,
        total_quad
    );

    // Spatial order: 1.2 Bohr boxes, the PySCF-style grouping.  It needs only
    // the coordinates, not a basis set or a density.
    let box_order = grid.reorder_boxes(BOX)?;
    let first_cell = box_of(grid.coords()[0]);
    let mut first_cell_len = 0;
    while first_cell_len < grid.len() && box_of(grid.coords()[first_cell_len]) == first_cell {
        first_cell_len += 1;
    }
    assert!(
        first_cell_len == grid.len() || box_of(grid.coords()[first_cell_len]) != first_cell,
        "the first box must be contiguous"
    );
    println!(
        "box sort: first box {:?} holds {} points in slots 0..{}",
        first_cell, first_cell_len, first_cell_len
    );
    let mut occupancy = std::collections::BTreeMap::new();
    for point in grid.coords() {
        *occupancy.entry(box_of(*point)).or_insert(0_usize) += 1;
    }
    let (densest_cell, densest_len) = occupancy
        .iter()
        .max_by_key(|&(_, &count)| count)
        .map(|(cell, count)| (*cell, *count))
        .unwrap();
    let densest_start = grid
        .coords()
        .iter()
        .position(|point| box_of(*point) == densest_cell)
        .unwrap();
    assert!(
        (densest_start..densest_start + densest_len)
            .all(|slot| box_of(grid.coords()[slot]) == densest_cell),
        "a box must be contiguous after sorting"
    );
    println!(
        "  densest box {:?} holds {} points in slots {}..{}",
        densest_cell,
        densest_len,
        densest_start,
        densest_start + densest_len
    );
    println!(
        "  original point 0 is now slot {}; blocks of 64: {}",
        box_order.inverse()[0],
        box_order.blocks(64).count()
    );

    // The order is a permutation of every per-point array.
    assert_eq!(box_order.len(), grid.len());
    assert!((grid.total_weight() - total).abs() <= 1e-10 * total.abs().max(1.0));
    assert!((grid.total_quad_weight() - total_quad).abs() <= 1e-10 * total_quad.abs().max(1.0));
    for (before, after) in by_atom.iter().zip(grid.quad_weight_by_atom()) {
        assert!((before - after).abs() <= 1e-10 * before.abs().max(1.0));
    }
    println!("  totals and per-atom quadrature weights are unchanged");

    // Re-sorting an already sorted grid is the identity.
    let again = GridOrder::boxes(grid.coords(), BOX)?;
    assert_eq!(again, GridOrder::identity(grid.len())?);
    println!("  re-sorting the sorted grid is the identity");

    // Morton (Z-order) is the space-filling-curve variant.
    let mut morton = original.clone();
    let morton_order = morton.reorder_morton(BOX)?;
    assert_eq!(morton_order.len(), morton.len());
    assert!((morton.total_weight() - total).abs() <= 1e-10 * total.abs().max(1.0));
    println!(
        "morton sort: {} points, first 2 points {:?}",
        morton.len(),
        &morton.coords()[..2]
    );

    // A consumer-side key is accepted verbatim; equal keys keep their
    // original relative order.
    let keys = original.atom_idx().iter().map(|&atom| u64::from(atom));
    let mut by_generator = original.clone();
    by_generator.reorder(&GridOrder::from_keys(keys)?)?;
    assert!(
        by_generator
            .atom_idx()
            .windows(2)
            .all(|window| window[0] <= window[1]),
        "generating atoms must be nondecreasing"
    );
    println!("key sort: points grouped by generating atom");

    // Derivatives read the generating atom per point, so they work in the
    // sorted slot order without a sort-back.
    let deriv = grid.weight_derivatives()?;
    let g = grid.len();
    let mut max_translation = 0.0_f64;
    for point in 0..g {
        for t in 0..3 {
            let sum: f64 = (0..grid.atom_count())
                .map(|atom| deriv.dw()[(atom * 3 + t) * g + point])
                .sum();
            max_translation = max_translation.max(sum.abs());
        }
    }
    println!(
        "derivatives on the box-sorted grid: dw [{}], max translation sum {:.1e}",
        deriv.dw().len(),
        max_translation
    );
    Ok(())
}
