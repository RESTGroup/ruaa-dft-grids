//! Geometry + preset -> `MolecularGrid`, using the parallel builder.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example user_build
//! ```
//!
//! The one-call `build_molecular_grid` is the parallel builder: the partition
//! weights are evaluated on Rayon's thread pool.  Use `RAYON_NUM_THREADS` or a
//! caller-installed Rayon pool to control the worker count.
//!
//! The example shows the same water molecule with the built-in PySCF preset
//! and with a small data-driven preset.  All coordinates are in Bohr.

use ruaa_dft_grids::atomgrid::AtomicGrid;
use ruaa_dft_grids::grid::{GridSpec, PyscfGridSpec};
use ruaa_dft_grids::moleculegrid::{MolecularGrid, PostPrune, build_molecular_grid};
use ruaa_dft_grids::partition::PartitionConfig;
use ruaa_dft_grids::preset::{ElementPreset, GridPreset};
use ruaa_dft_grids::radial::Radial;

/// CODATA 2010 Bohr radius used by the crate's radii tables.
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

fn point_counts_per_atom(grid: &MolecularGrid) -> Vec<usize> {
    let mut counts = vec![0; grid.atom_count()];
    for &atom in grid.atom_idx() {
        counts[atom as usize] += 1;
    }
    counts
}

fn summary(label: &str, grid: &MolecularGrid) {
    let max_identity_error = grid
        .points()
        .map(|p| (p.weight - p.quad_weight * p.partition_fraction).abs())
        .fold(0.0_f64, f64::max);
    println!("{label}");
    println!("  atoms                 {}", grid.atom_count());
    println!("  points                {}", grid.len());
    println!("  points per atom       {:?}", point_counts_per_atom(grid));
    println!("  total weight          {:.10}", grid.total_weight());
    println!("  max |w - q*P|         {:.1e}", max_identity_error);
}

/// A caller-supplied topology: 35 radial shells, a 302-point rule on each
/// shell.  The radial scale comes from the PySCF Treutler table, but any
/// explicit scale can be passed instead.
fn small_preset() -> GridPreset {
    let mut preset = GridPreset::new(Radial::TreutlerAhlrichs).with_name("small");
    for z in [1u32, 8] {
        let scale = Radial::TreutlerAhlrichs.xi_for_element(z).unwrap();
        preset = preset.with_element(z, ElementPreset::fixed(35, scale, 302));
    }
    preset
}

fn main() -> ruaa_dft_grids::Result<()> {
    println!(
        "ruaa_dft_grids user build example ({} Rayon threads)",
        rayon::current_num_threads()
    );
    let atoms = water();

    // PySCF: one `Grids.level` number selects the (n_rad, n_ang) presets.
    let pyscf = PyscfGridSpec::from_level(0)?;
    let grid = build_molecular_grid(&atoms, &pyscf)?;
    summary("PySCF, level 0", &grid);

    // A data-driven preset: the shell counts, radial scales and angular sizes
    // are explicit data, not a built-in program table.
    let preset = small_preset();
    let spec = GridSpec::from_preset(preset);
    let full = build_molecular_grid(&atoms, &spec)?;
    let split = full.split(PostPrune::PartitionCut(1e-14))?;
    summary("data-driven preset, partition cut 1e-14", &split.retained);
    println!("  points removed        {}", split.pruned.len());

    // Any partition can be combined with any topology.
    let with_mbecke = spec.clone().with_partition(PartitionConfig::new(
        ruaa_dft_grids::partition::PartitionScheme::MBecke,
        ruaa_dft_grids::partition::RadiiConfig::pyscf(),
    ));
    let grid = build_molecular_grid(&atoms, &with_mbecke)?;
    println!(
        "same topology, mBecke partition: {} points, spec preserved = {}",
        grid.len(),
        grid.spec().preset().is_some()
    );

    // A caller-built atomic grid uses the same materialisation and partition
    // path; this is the extension point for a rule the preset format does not
    // express.
    let molecule = ruaa_dft_grids::Molecule::new(atoms)?;
    let custom = vec![
        AtomicGrid::new(8, vec![[0.1, 0.0, 0.0]], vec![1.0], vec![1])?,
        AtomicGrid::new(1, vec![[0.0, 1.4, 1.1]], vec![1.0], vec![1])?,
        AtomicGrid::new(1, vec![[0.0, -1.4, 1.1]], vec![1.0], vec![1])?,
    ];
    let grid = ruaa_dft_grids::moleculegrid::build_molecular_grid_from_atomic_grids(
        &molecule,
        custom,
        PartitionConfig::pyscf(),
    )?;
    println!(
        "custom atomic grids: {} points, custom spec = {}",
        grid.len(),
        grid.spec().is_custom()
    );
    Ok(())
}
