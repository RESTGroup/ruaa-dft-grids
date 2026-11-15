"""Tests for the dependency-free Python preset tooling.

Run from the repository root::

    python -m unittest discover -s python/tests -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import ruaa_dft_grids_presets as presets  # noqa: E402

PRESET_TOML = ROOT / "examples" / "presets" / "pyscf-level-3.toml"
FIXTURE = ROOT / "tests" / "data" / "pyscf-level-3-canonical.json"

TINY_PRESET = {
    "format": 1,
    "name": "tiny",
    "z_min": 1,
    "z_max": 2,
    "radial": "treutler_ahlrichs",
    "default": {
        "n_rad": 20,
        "radial_scale": 0.8,
        "angular": {"fixed": {"n_ang": 50}},
    },
}


def tiny_manifest(entry: dict, name: str = "tiny") -> dict:
    return {
        "format": 1,
        "programs": {
            "demo": {
                "software": "Demo 1.0",
                "presets": [
                    {
                        "name": name,
                        "category": "xc",
                        "file": "tiny.json",
                        "elements": [1, 2],
                        **entry,
                    }
                ],
            }
        },
    }


class PresetFileTests(unittest.TestCase):
    def test_shipped_preset_matches_the_canonical_fixture(self):
        preset = presets.load(PRESET_TOML)
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(preset.to_json(), fixture)
        self.assertEqual(len(preset.elements), 103)
        self.assertEqual(len(preset.ranges), 7)
        self.assertEqual(preset.element(8).radial_scale, 0.9)

    def test_json_round_trip(self):
        preset = presets.load(PRESET_TOML)
        reloaded = presets.loads(presets.dumps(preset), format="json")
        self.assertEqual(reloaded, preset)

    def test_unknown_keys_are_rejected(self):
        with self.assertRaisesRegex(presets.PresetError, "molecular_cutoff"):
            presets.loads(
                'radial = "delley"\nmolecular_cutoff = true\n', format="toml"
            )
        with self.assertRaisesRegex(presets.PresetError, "cuttoff"):
            presets.loads(
                'radial = "delley"\n[default]\nn_rad = 20\nradial_scale = 0.9\n'
                'cuttoff = 1.0\nangular = { fixed = { n_ang = 50 } }\n',
                format="toml",
            )
        with self.assertRaisesRegex(presets.PresetError, "elements"):
            presets.loads(
                'radial = "delley"\n[elements.8]\nn_rad = 20\n', format="toml"
            )

    def test_format_version(self):
        with self.assertRaisesRegex(presets.PresetError, "unsupported preset format 2"):
            presets.loads('format = 2\nradial = "delley"\n', format="toml")
        preset = presets.loads(
            'radial = "delley"\n[default]\nn_rad = 20\nradial_scale = 0.9\n'
            'angular = { fixed = { n_ang = 50 } }\n',
            format="toml",
        )
        self.assertEqual(preset.to_json()["format"], 1)

    def test_range_outside_the_domain_is_rejected(self):
        text = (
            'z_min = 1\nz_max = 2\nradial = "delley"\n[[range]]\nz_min = 3\n'
            'z_max = 10\nn_rad = 20\nradial_scale = 0.9\n'
            'angular = { fixed = { n_ang = 50 } }\n'
        )
        with self.assertRaisesRegex(presets.PresetError, "outside the element domain"):
            presets.loads(text, format="toml")

    def test_toml_output_is_not_implemented(self):
        preset = presets.load(PRESET_TOML)
        with self.assertRaisesRegex(presets.PresetError, "JSON output"):
            presets.dumps(preset, format="toml")


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "tiny.json").write_text(
            json.dumps(TINY_PRESET), encoding="utf-8"
        )

    def tearDown(self):
        self._tmp.cleanup()

    def write_manifest(self, manifest: dict) -> Path:
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_load_applies_the_catalog_range(self):
        catalog = presets.load_catalog(self.write_manifest(tiny_manifest({})))
        preset = catalog.load("demo", "tiny", self.root)
        self.assertEqual((preset.z_min, preset.z_max), (1, 2))
        self.assertEqual(preset.element(1).n_rad, 20)

    def test_duplicate_names_are_rejected(self):
        manifest = tiny_manifest({})
        manifest["programs"]["demo"]["presets"].append(
            {
                "name": "tiny",
                "category": "xc",
                "file": "other.json",
                "elements": [1, 2],
            }
        )
        with self.assertRaisesRegex(presets.PresetError, "duplicate preset name"):
            presets.load_catalog(self.write_manifest(manifest))

    def test_catalog_range_cannot_widen_the_preset_domain(self):
        catalog = presets.load_catalog(
            self.write_manifest(tiny_manifest({"elements": [1, 4]}))
        )
        with self.assertRaisesRegex(presets.PresetError, "exceeds the preset domain"):
            catalog.load("demo", "tiny", self.root)

    def test_preset_name_must_match_the_entry_name(self):
        catalog = presets.load_catalog(
            self.write_manifest(tiny_manifest({}, name="other"))
        )
        with self.assertRaisesRegex(presets.PresetError, "does not match catalog entry"):
            catalog.load("demo", "other", self.root)

    def test_catalog_preset_files_must_be_json(self):
        manifest = tiny_manifest({})
        manifest["programs"]["demo"]["presets"][0]["file"] = "tiny.toml"
        catalog = presets.load_catalog(self.write_manifest(manifest))
        with self.assertRaisesRegex(presets.PresetError, "must be JSON"):
            catalog.load("demo", "tiny", self.root)

    def test_unknown_catalog_keys_are_rejected(self):
        manifest = tiny_manifest({})
        manifest["programs"]["demo"]["presets"][0]["elemnts"] = [1, 2]
        with self.assertRaisesRegex(presets.PresetError, "elemnts"):
            presets.load_catalog(self.write_manifest(manifest))

    def test_unsafe_paths_are_rejected(self):
        manifest = tiny_manifest({})
        manifest["programs"]["demo"]["presets"][0]["file"] = "../escape.json"
        with self.assertRaisesRegex(presets.PresetError, "unsafe relative path"):
            presets.load_catalog(self.write_manifest(manifest))


if __name__ == "__main__":
    unittest.main()

    def test_scalar_types_are_strict(self):
        with self.assertRaisesRegex(presets.PresetError, "radial shell count"):
            presets.loads(
                'radial = "delley"\n[default]\nn_rad = -5\nradial_scale = 1.0\n'
                'angular = { fixed = { n_ang = 50 } }\n',
                format="toml",
            )
        with self.assertRaisesRegex(presets.PresetError, "boolean"):
            presets.loads(
                'radial = "delley"\nmolecular_cutoffs = "no"\n[default]\n'
                'n_rad = 20\nradial_scale = 1.0\n'
                'angular = { fixed = { n_ang = 50 } }\n',
                format="toml",
            )
        with self.assertRaisesRegex(presets.PresetError, "boolean"):
            presets.loads(
                'radial = "delley"\n[default]\nn_rad = 20\nradial_scale = 1.0\n'
                'generate_all = "false"\nangular = { fixed = { n_ang = 50 } }\n',
                format="toml",
            )
        with self.assertRaisesRegex(presets.PresetError, "string"):
            presets.loads(
                'name = 123\nradial = "delley"\n[default]\nn_rad = 20\n'
                'radial_scale = 1.0\nangular = { fixed = { n_ang = 50 } }\n',
                format="toml",
            )

    def test_missing_angular_field_is_a_preset_error(self):
        with self.assertRaisesRegex(presets.PresetError, "missing `n_ang`"):
            presets.loads(
                'radial = "delley"\n[default]\nn_rad = 20\nradial_scale = 1.0\n'
                'angular = { fixed = {} }\n',
                format="toml",
            )


class CatalogStrictnessTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "tiny.json").write_text(
            json.dumps(TINY_PRESET), encoding="utf-8"
        )

    def tearDown(self):
        self._tmp.cleanup()

    def write_manifest(self, manifest: dict) -> Path:
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_catalog_requires_a_format(self):
        with self.assertRaisesRegex(presets.PresetError, "missing `format`"):
            presets.Catalog.from_json_str('{"programs": {}}')

    def test_unknown_kind_is_rejected(self):
        manifest = tiny_manifest({"kind": "bogus"})
        with self.assertRaisesRegex(presets.PresetError, "unknown kind"):
            presets.load_catalog(self.write_manifest(manifest))

    def test_ladder_defaults_match_rust(self):
        preset = presets.loads(
            'radial = "delley"\n[default]\nn_rad = 20\nradial_scale = 1.0\n'
            'angular = { pruned = { base = 194, prune = { ladder = { '
            'ladder = [50, 110, 194, 302, 434], unpruned = 434, '
            'bounds = [[0.25,0.5,1.0,4.5],[0.1667,0.5,0.9,3.5],'
            '[0.1,0.4,0.8,2.5],[0.02,0.1,0.2,3.5]] } } } }\n',
            format="toml",
        )
        ladder = preset.default.angular["pruned"]["prune"]["ladder"]
        self.assertEqual(ladder["max_pruned_z"], 36)
        self.assertEqual(ladder["heavy_fallback"], "unpruned")

    def test_narrowing_clips_file_data(self):
        narrow = {
            "format": 1,
            "name": "narrow",
            "z_min": 1,
            "z_max": 4,
            "radial": "delley",
            "default": {
                "n_rad": 20,
                "radial_scale": 1.0,
                "angular": {"fixed": {"n_ang": 50}},
            },
            "range": [
                {
                    "z_min": 1,
                    "z_max": 4,
                    "n_rad": 30,
                    "radial_scale": 1.0,
                    "angular": {"fixed": {"n_ang": 50}},
                }
            ],
            "element": [
                {"z": 1, "radial_scale": 1.1},
                {"z": 4, "radial_scale": 1.4},
            ],
        }
        (self.root / "narrow.json").write_text(json.dumps(narrow), encoding="utf-8")
        manifest = tiny_manifest(
            {"file": "narrow.json", "name": "narrow", "elements": [2, 3]}
        )
        catalog = presets.load_catalog(self.write_manifest(manifest))
        preset = catalog.load("demo", "narrow", self.root)
        self.assertEqual((preset.z_min, preset.z_max), (2, 3))
        self.assertEqual([(lo, hi) for lo, hi, _ in preset.ranges], [(2, 3)])
        self.assertEqual(sorted(preset.elements), [])
        self.assertEqual(preset.element(2).n_rad, 30)

    @unittest.skipUnless(hasattr(__import__("os"), "symlink"), "symlink unsupported")
    def test_symlink_escape_is_rejected(self):
        import os

        outside = self.root.parent / "outside.json"
        outside.write_text(json.dumps(TINY_PRESET), encoding="utf-8")
        os.symlink(outside, self.root / "link.json")
        manifest = tiny_manifest({"file": "link.json"})
        catalog = presets.load_catalog(self.write_manifest(manifest))
        with self.assertRaisesRegex(presets.PresetError, "escapes the data directory"):
            catalog.load("demo", "tiny", self.root)
