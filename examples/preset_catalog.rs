//! Load a caller-supplied preset catalog and build one atomic grid.
//!
//! The crate ships no preset data.  Pass a catalog manifest and one of its
//! entries:
//!
//! ```text
//! cargo run --example preset_catalog -- /path/to/data/manifest.json demo demo-grid 8
//! ```
//!
//! The manifest's `file` paths are resolved relative to the manifest's
//! directory.

use std::path::Path;

use ruaa_dft_grids::catalog::PresetCatalog;

fn main() -> ruaa_dft_grids::Result<()> {
    let mut args = std::env::args().skip(1);
    let manifest = args
        .next()
        .expect("usage: preset_catalog <manifest.json> <program> <name> <Z>");
    let program = args.next().expect("missing program key");
    let name = args.next().expect("missing preset name");
    let z: u32 = args
        .next()
        .expect("missing nuclear charge")
        .parse()
        .expect("nuclear charge must be an integer");

    let manifest = Path::new(&manifest);
    let base_dir = manifest.parent().unwrap_or_else(|| Path::new("."));
    let catalog = PresetCatalog::from_json_path(manifest)?;
    let preset = catalog.load_preset(&program, &name, base_dir)?;
    let grid = preset.atomic_grid(z)?;
    println!(
        "{program}/{name}: Z={z}, nr={}, npoints={}",
        grid.nr(),
        grid.points.len()
    );
    Ok(())
}
