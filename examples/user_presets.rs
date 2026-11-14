//! Presets: the built-in PySCF level, the full data preset, and the optional
//! cross-check against PySCF.
//!
//! ```text
//! cargo run --release --example user_presets
//! cargo run --release --example user_presets -- --dump --elements 1,6,8,10
//! cargo run --release --example user_presets -- --dump --elements all
//! ```
//!
//! The demo opens PySCF's default `Grids.level = 3` preset both ways: as the
//! built-in [`PyscfGridSpec`] and as the full [`GridPreset`] data file
//! `examples/presets/pyscf-level-3.toml`.  The two atomic grids are identical,
//! so a caller can start from PySCF's tables, take the data form, and change
//! individual elements.
//!
//! `--dump` writes the file preset's atomic grids to standard output for
//! `examples/presets/check_against_pyscf.py`, which compares them with
//! PySCF's own construction.  The cross-check is optional: it needs PySCF and
//! is not part of `cargo test`.
//!
//! All coordinates are in Bohr.

use std::io::Write;

use ruaa_dft_grids::grid::{GridSpec, PyscfGridSpec};
use ruaa_dft_grids::moleculegrid::{MolecularGrid, build_molecular_grid};
use ruaa_dft_grids::preset::{ElementPreset, GridPreset};
use ruaa_dft_grids::prune::Prune;
use ruaa_dft_grids::pyscf::level_defaults;
use ruaa_dft_grids::radial::Radial;
use ruaa_dft_grids::{Error, Result};

/// CODATA 2010 Bohr radius used by the crate's radii tables.
const BOHR_PER_ANGSTROM: f64 = 1.889_726_125_457_828_1;

/// PySCF periodic rows: the `(Z_min, Z_max)` of each column of
/// `gen_grid.RAD_GRIDS` and `gen_grid.ANG_ORDER`.
const PERIODS: [(u32, u32); 7] = [
    (1, 2),
    (3, 10),
    (11, 18),
    (19, 36),
    (37, 54),
    (55, 86),
    (87, 118),
];

/// The PySCF Treutler scale table ends at Z = 103.
const MAX_TREUTLER_Z: u32 = 103;

/// The complete level-3 preset file used by the demo and the dump.
const PRESET_TOML: &str = include_str!("presets/pyscf-level-3.toml");

const USAGE: &str = "\
Usage:
  cargo run --release --example user_presets
  cargo run --release --example user_presets -- --dump [--elements 1,6,8,10|all]

--dump prints the preset file's atomic grids for
examples/presets/check_against_pyscf.py.  The default element list is 1,6,8,10.
";

/// Water in a common input orientation (O, H, H).
fn water() -> [(u32, [f64; 3]); 3] {
    let h = [0.757 * BOHR_PER_ANGSTROM, 0.587 * BOHR_PER_ANGSTROM];
    [
        (8, [0.0, 0.0, 0.0]),
        (1, [0.0, h[0], h[1]]),
        (1, [0.0, -h[0], h[1]]),
    ]
}

/// PySCF `Grids(level = level)` as a data-driven preset.
///
/// The level tables give one `(n_rad, n_ang)` per periodic row, which is
/// exactly what a [`GridPreset`] range expresses; the Treutler scale is per
/// element, so each element entry overrides only `radial_scale`.
///
/// # Errors
/// Propagates the level and element-table errors.
fn pyscf_level_preset(level: u8) -> Result<GridPreset> {
    let radial = Radial::TreutlerAhlrichs;
    let mut preset = GridPreset::new(radial).with_name(format!("pyscf-level-{level}"));
    for &(z_min, z_max) in &PERIODS {
        let z_max = z_max.min(MAX_TREUTLER_Z);
        if z_min > z_max {
            continue;
        }
        let (n_rad, n_ang) = level_defaults(z_min, level)?;
        // The range scale is a placeholder: every element sets its own.
        let base = ElementPreset::pruned(n_rad, 1.0, n_ang, Prune::Nwchem);
        preset = preset.with_default_range(z_min, z_max, base.clone());
        for z in z_min..=z_max {
            preset = preset.with_element(
                z,
                ElementPreset {
                    radial_scale: radial.xi_for_element(z)?,
                    ..base.clone()
                },
            );
        }
    }
    Ok(preset)
}

/// Parse the complete preset file.
fn file_preset() -> GridPreset {
    toml::from_str(PRESET_TOML).expect("examples/presets/pyscf-level-3.toml is a valid preset")
}

fn point_counts_per_atom(grid: &MolecularGrid) -> Vec<usize> {
    let mut counts = vec![0; grid.atom_count()];
    for &atom in grid.atom_idx() {
        counts[atom as usize] += 1;
    }
    counts
}

fn summary(label: &str, grid: &MolecularGrid) {
    println!("{label}");
    println!("  atoms                 {}", grid.atom_count());
    println!("  points                {}", grid.len());
    println!("  points per atom       {:?}", point_counts_per_atom(grid));
    println!("  total weight          {:.10}", grid.total_weight());
}

/// Dump the preset file's atomic grids in the format read by
/// `examples/presets/check_against_pyscf.py`.
fn dump_atomic_grids(elements: &[u32]) -> Result<()> {
    let preset = file_preset();
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    writeln!(out, "# ruaa_dft_grids atomic-grid dump").expect("stdout");
    writeln!(out, "# preset: examples/presets/pyscf-level-3.toml").expect("stdout");
    writeln!(
        out,
        "# format: element <Z> | shell <index> <n_ang> | point <x> <y> <z> <w>"
    )
    .expect("stdout");
    for &z in elements {
        let grid = preset.atomic_grid(z)?;
        writeln!(out, "element {z}").expect("stdout");
        let mut start = 0;
        for (shell, &n_ang) in grid.shell_angular.iter().enumerate() {
            writeln!(out, "shell {shell} {n_ang}").expect("stdout");
            for index in start..start + n_ang {
                let point = grid.points[index];
                writeln!(
                    out,
                    "point {:.17e} {:.17e} {:.17e} {:.17e}",
                    point[0], point[1], point[2], grid.quad_weights[index]
                )
                .expect("stdout");
            }
            start += n_ang;
        }
        assert_eq!(start, grid.points.len(), "shell sizes cover the grid");
    }
    Ok(())
}

fn parse_elements(text: &str) -> Result<Vec<u32>> {
    if text == "all" {
        return Ok((1..=MAX_TREUTLER_Z).collect());
    }
    let mut elements = Vec::new();
    for part in text.split(',') {
        let z: u32 = part.trim().parse().map_err(|_| Error::InvalidArgument {
            argument: "elements",
        })?;
        if !(1..=MAX_TREUTLER_Z).contains(&z) {
            return Err(Error::UnsupportedElement { z });
        }
        elements.push(z);
    }
    Ok(elements)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{USAGE}");
        return Ok(());
    }
    if args.iter().any(|arg| arg == "--dump") {
        let elements = match args.iter().position(|arg| arg == "--elements") {
            Some(index) => parse_elements(args.get(index + 1).map_or("", String::as_str))?,
            None => vec![1, 6, 8, 10],
        };
        return dump_atomic_grids(&elements);
    }

    println!("ruaa_dft_grids user presets example");

    // 1. The built-in PySCF preset: one number selects the radial and
    //    angular sizes from PySCF's published level tables.
    let pyscf = PyscfGridSpec::from_level(3)?;
    let atoms = water();
    let built_in = build_molecular_grid(&atoms, &pyscf)?;
    summary("PySCF level 3, built-in preset", &built_in);

    // 2. The same level as data: one range per periodic row plus one
    //    per-element radial scale.  The complete file is
    //    `examples/presets/pyscf-level-3.toml`; the loop below builds the
    //    same topology from the public PySCF tables.
    let preset = pyscf_level_preset(3)?;
    assert_eq!(file_preset(), preset, "the file matches the level tables");
    for z in [1, 6, 8, 10] {
        assert_eq!(preset.atomic_grid(z)?, pyscf.atomic_grid(z)?, "Z = {z}");
    }
    println!("  data preset equals the built-in atomic grid for H, C, O, Ne: true");
    println!(
        "  examples/presets/pyscf-level-3.toml: {} ranges, {} elements",
        preset.defaults.len(),
        preset.elements.len()
    );

    // 3. Use the data preset exactly like any other topology.
    let spec = GridSpec::from_preset(preset.clone());
    let data_grid = build_molecular_grid(&atoms, &spec)?;
    assert_eq!(data_grid.len(), built_in.len());
    assert_eq!(data_grid.total_weight(), built_in.total_weight());
    summary("PySCF level 3, data preset", &data_grid);

    // 4. Change one element: promote oxygen to the level-4 sizes while the
    //    rest of the table keeps level 3.
    let (o_rad, o_ang) = level_defaults(8, 4)?;
    let o_scale = Radial::TreutlerAhlrichs.xi_for_element(8)?;
    let modified = preset.clone().with_element(
        8,
        ElementPreset::pruned(o_rad, o_scale, o_ang, Prune::Nwchem),
    );
    let modified_grid = build_molecular_grid(&atoms, GridSpec::from_preset(modified.clone()))?;
    summary("PySCF level 3 with oxygen at level 4", &modified_grid);
    assert_eq!(modified.element(8)?.n_rad, o_rad);
    assert_eq!(modified.element(1)?.n_rad, level_defaults(1, 3)?.0);

    // 5. The file form: serialize a preset, read it back, and confirm that
    //    the round trip preserves the topology.
    let text = toml::to_string(&preset).expect("the preset is a TOML table");
    let loaded: GridPreset = toml::from_str(&text).expect("the written preset parses");
    assert_eq!(loaded, preset);
    println!(
        "  TOML round trip: {} bytes, {} element entries, {} range blocks",
        text.len(),
        loaded.elements.len(),
        loaded.defaults.len()
    );
    println!();
    println!("Cross-check against PySCF (optional, needs PySCF):");
    println!("  python examples/presets/check_against_pyscf.py");
    Ok(())
}
