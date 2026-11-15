# Python preset tooling

Dependency-free reader and validator for `ruaa_dft_grids` preset files and
preset catalogs.  It mirrors the Rust file form:

* the `format = 1` version field;
* `element` > `range` > `default` inheritance (partial element entries);
* strict unknown-key rejection;
* the element domain, angular sizes, radial scale/cutoff and moving-cutoff
  shell-list invariants;
* the JSON catalog index (`from_json_path` validates names, paths and element
  ranges; `Catalog.load` refuses a wider range or a name mismatch).

It validates the *file form* only.  Construction-dependent checks — whether a
`Shells` list covers the active radial shells, for example — need the radial
code and stay in the Rust crate.

## Usage

```python
import ruaa_dft_grids_presets as presets

preset = presets.load("examples/presets/pyscf-level-3.toml")
print(preset.element(8).radial_scale)

catalog = presets.load_catalog("data/manifest.json")
entry = catalog.load("orca", "orca-defgrid2", "data")
```

`load` accepts `.toml` and `.json`; `dumps` writes canonical JSON (the TOML
writer is not implemented; TOML is the authoring form and needs `tomli-w` or
a small schema-specific writer if it is ever required).

## Contract with the crate

The canonical mapping is the Rust `serde` form.  The checked-in
`tests/data/pyscf-level-3-canonical.json` is produced by

```bash
cargo run --example preset_dump -- examples/presets/pyscf-level-3.toml \
  > tests/data/pyscf-level-3-canonical.json
```

and is read by both `tests/preset_fixture.rs` (Rust side) and
`python/tests/test_preset_file.py` (Python side), so a change to either
serializer breaks the other check.

## Tests

```bash
python -m unittest discover -s python/tests -v
```

The tests need no third-party package and no Rust toolchain; the canonical
fixture is checked in.
