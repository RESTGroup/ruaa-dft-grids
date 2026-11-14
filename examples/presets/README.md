# PySCF level-3 preset example

`pyscf-level-3.toml` is PySCF's `Grids(level = 3)` construction as a
`ruaa_dft_grids::preset::GridPreset` data file: seven `[[range]]` blocks (one
per PySCF periodic row) and one `[[element]]` entry per nuclear charge, Z =
1–103.  The element entries carry the Treutler radial scale and inherit the
level sizes and the NWChem pruning layout from the range; the range
`radial_scale` is a placeholder that every element entry replaces.

The file is loaded by `../user_presets.rs` and parsed by
`../../tests/preset_api.rs`, which requires it to equal the preset built from
`pyscf::level_defaults` and `Radial::xi_for_element`.

## Cross-check against PySCF

`check_against_pyscf.py` runs the example in dump mode and compares the
preset's atomic grids with PySCF's own `gen_atomic_grids` at level 3:

```bash
python3 examples/presets/check_against_pyscf.py
python3 examples/presets/check_against_pyscf.py --elements all
```

It needs PySCF, NumPy and a Rust toolchain.  The check is optional and is not
part of `cargo test`.  The [presets page](../../docs/presets.md) describes the
preset model and both files in detail.
