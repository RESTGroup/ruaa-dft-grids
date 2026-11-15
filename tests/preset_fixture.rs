//! Canonical fixture shared with the Python preset tooling.
//!
//! `python/tests/test_preset_file.py` checks that the Python serializer
//! reproduces the same mapping; this test checks the Rust side of the same
//! fixture, so the two implementations cannot drift apart silently.

use ruaa_dft_grids::preset::GridPreset;

#[test]
fn python_fixture_matches_the_rust_serializer() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("data/pyscf-level-3-canonical.json")).unwrap();
    let preset: GridPreset =
        toml::from_str(include_str!("../examples/presets/pyscf-level-3.toml")).unwrap();
    assert_eq!(serde_json::to_value(&preset).unwrap(), fixture);
}
