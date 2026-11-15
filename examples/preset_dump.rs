//! Dump the canonical `serde` JSON of a preset file.
//!
//! ```text
//! cargo run --example preset_dump -- examples/presets/pyscf-level-3.toml
//! ```
//!
//! The output is the mapping the Python preset tooling in `python/` must
//! reproduce; `tests/preset_fixture.rs` and `python/tests/` share the
//! checked-in canonical fixture.

use std::path::Path;

use ruaa_dft_grids::preset::GridPreset;

fn main() -> ruaa_dft_grids::Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: preset_dump <preset.toml|preset.json>");
    let text = std::fs::read_to_string(&path).expect("read the preset file");
    let preset: GridPreset = match Path::new(&path).extension().and_then(|ext| ext.to_str()) {
        Some("toml") => toml::from_str(&text).expect("valid TOML preset"),
        Some("json") => serde_json::from_str(&text).expect("valid JSON preset"),
        _ => panic!("expected a .toml or .json preset file"),
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&preset).expect("serialize the preset")
    );
    Ok(())
}
