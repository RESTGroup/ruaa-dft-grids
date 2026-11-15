"""Reader and validator for ``ruaa_dft_grids`` preset files.

The module mirrors the Rust file form (``GridPreset`` serde), including the
``format`` version, the ``element`` > ``range`` > ``default`` inheritance and
the catalog loader.  It is deliberately dependency-free: TOML uses the
standard-library ``tomllib`` (Python 3.11+) and JSON uses ``json``.

It validates the *file form* only.  Construction-dependent checks (for
example whether a ``Shells`` list covers the active radial shells) need the
radial code and stay in the Rust crate.

Typical use::

    from ruaa_dft_grids_presets import load, load_catalog

    preset = load("examples/presets/pyscf-level-3.toml")
    preset.element(8).radial_scale

    catalog = load_catalog("data/manifest.json")
    entry = catalog.load("orca", "orca-defgrid2")
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - older Python
    tomllib = None  # type: ignore[assignment]

FORMAT = 1

RADIAL = {
    "treutler_ahlrichs",
    "delley",
    "mura_knowles",
    "gauss_chebyshev",
    "becke",
    "orca_opt_m3",
    "orca_m3",
    "euler_maclaurin",
}

LEBEDEV = {
    1, 6, 12, 14, 18, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302,
    350, 434, 590, 770, 974, 1202, 1454, 1730, 2030, 2354, 2702, 3074, 3470,
    3890, 4334, 4802, 5294, 5810,
}

PRUNE = {"none", "nwchem", "sg1", "treutler", "sgx"}
HEAVY_FALLBACK = {"unpruned", "default_ladder"}

PRESET_KEYS = {
    "format", "name", "z_min", "z_max", "radial", "molecular_cutoffs",
    "default", "range", "element",
}
ELEMENT_KEYS = {"n_rad", "radial_scale", "cutoff", "generate_all", "angular"}
RANGE_KEYS = {"z_min", "z_max"} | ELEMENT_KEYS
ENTRY_KEYS = {"z"} | ELEMENT_KEYS


class PresetError(ValueError):
    """A preset or catalog file failed to load or validate."""


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise PresetError(f"{where}: unknown key(s) {unknown}")


def _int(value: Any, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise PresetError(f"{where}: expected an integer, got {value!r}")
    return value


def _float(value: Any, where: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise PresetError(f"{where}: expected a number, got {value!r}")
    return float(value)


def _parse_prune(value: Any, where: str) -> Any:
    if isinstance(value, str):
        if value not in PRUNE:
            raise PresetError(f"{where}: unknown prune scheme {value!r}")
        return value
    if not isinstance(value, dict):
        raise PresetError(f"{where}: prune must be a string or a ladder object")
    _reject_unknown(value, {"ladder"}, where)
    ladder = value["ladder"]
    _reject_unknown(
        ladder,
        {"ladder", "unpruned", "bounds", "max_pruned_z", "heavy_fallback"},
        f"{where}.ladder",
    )
    sizes = ladder["ladder"]
    if not isinstance(sizes, list) or len(sizes) != 5:
        raise PresetError(f"{where}.ladder.ladder: expected five sizes")
    bounds = ladder["bounds"]
    if not isinstance(bounds, list) or len(bounds) != 4:
        raise PresetError(f"{where}.ladder.bounds: expected four rows")
    for row in bounds:
        if not isinstance(row, list) or len(row) != 4:
            raise PresetError(f"{where}.ladder.bounds: expected four columns")
    fallback = ladder.get("heavy_fallback", "unpruned")
    if fallback not in HEAVY_FALLBACK:
        raise PresetError(f"{where}.ladder.heavy_fallback: unknown {fallback!r}")
    return {
        "ladder": {
            "ladder": [_int(v, f"{where}.ladder.ladder") for v in sizes],
            "unpruned": _int(ladder["unpruned"], f"{where}.ladder.unpruned"),
            "bounds": [[_float(v, f"{where}.ladder.bounds") for v in row] for row in bounds],
            "max_pruned_z": _int(
                ladder.get("max_pruned_z", 36), f"{where}.ladder.max_pruned_z"
            ),
            "heavy_fallback": fallback,
        }
    }


def _parse_angular(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or len(value) != 1:
        raise PresetError(f"{where}: angular must be a one-key object")
    (kind, body), = value.items()
    if kind == "fixed":
        _reject_unknown(body, {"n_ang"}, f"{where}.fixed")
        return {"fixed": {"n_ang": _int(body["n_ang"], f"{where}.fixed.n_ang")}}
    if kind == "pruned":
        _reject_unknown(body, {"base", "prune"}, f"{where}.pruned")
        return {
            "pruned": {
                "base": _int(body["base"], f"{where}.pruned.base"),
                "prune": _parse_prune(body["prune"], f"{where}.pruned.prune"),
            }
        }
    if kind == "shells":
        if not isinstance(body, list):
            raise PresetError(f"{where}: shells must be an array")
        return {"shells": [_int(v, f"{where}.shells") for v in body]}
    if kind == "product":
        _reject_unknown(body, {"theta"}, f"{where}.product")
        return {"product": {"theta": _int(body["theta"], f"{where}.product.theta")}}
    if kind == "product_shells":
        if not isinstance(body, list):
            raise PresetError(f"{where}: product_shells must be an array")
        return {"product_shells": [_int(v, f"{where}.product_shells") for v in body]}
    raise PresetError(f"{where}: unknown angular layout {kind!r}")


def _validate_angular(angular: dict[str, Any], where: str) -> None:
    (kind, body), = angular.items()
    if kind == "fixed":
        if body["n_ang"] not in LEBEDEV:
            raise PresetError(f"{where}: unsupported angular size {body['n_ang']}")
    elif kind == "pruned":
        if body["base"] not in LEBEDEV:
            raise PresetError(f"{where}: unsupported angular size {body['base']}")
    elif kind == "shells":
        if not body:
            raise PresetError(f"{where}: empty shell list")
        for size in body:
            if size not in LEBEDEV:
                raise PresetError(f"{where}: unsupported angular size {size}")
    elif kind == "product":
        if not 2 <= body["theta"] <= 999:
            raise PresetError(f"{where}: product theta out of range")
    else:
        if not body:
            raise PresetError(f"{where}: empty product shell list")
        for theta in body:
            if not 2 <= theta <= 999:
                raise PresetError(f"{where}: product theta out of range")


def _uses_shell_list(element: "Element") -> bool:
    return "shells" in element.angular or "product_shells" in element.angular


@dataclass
class Element:
    """Radial and angular parameters of one element."""

    n_rad: int
    radial_scale: float
    cutoff: float | None
    generate_all: bool
    angular: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n_rad": self.n_rad,
            "radial_scale": self.radial_scale,
        }
        if self.cutoff is not None:
            out["cutoff"] = self.cutoff
        if not self.generate_all:
            out["generate_all"] = False
        out["angular"] = self.angular
        return out

    def validate(self, where: str) -> None:
        if self.n_rad == 0:
            raise PresetError(f"{where}: radial shell count must be positive")
        if not math.isfinite(self.radial_scale) or self.radial_scale <= 0:
            raise PresetError(f"{where}: radial scale must be finite and positive")
        if self.cutoff is not None and (
            not math.isfinite(self.cutoff) or self.cutoff <= 0
        ):
            raise PresetError(f"{where}: radial cutoff must be finite and positive")
        _validate_angular(self.angular, where)


@dataclass
class Preset:
    """A program-independent grid topology."""

    name: str = ""
    z_min: int = 1
    z_max: int = 118
    radial: str = "treutler_ahlrichs"
    molecular_cutoffs: bool = False
    default: Element | None = None
    ranges: list[tuple[int, int, Element]] = field(default_factory=list)
    elements: dict[int, Element] = field(default_factory=dict)

    def element(self, z: int) -> Element:
        if not 1 <= z <= 118:
            raise PresetError(f"unsupported element Z = {z}")
        if not self.z_min <= z <= self.z_max:
            raise PresetError(f"no grid preset data for Z = {z}")
        if z in self.elements:
            return self.elements[z]
        for z_min, z_max, element in self.ranges:
            if z_min <= z <= z_max:
                return element
        if self.default is not None:
            return self.default
        raise PresetError(f"no grid preset data for Z = {z}")

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"format": FORMAT}
        if self.name:
            out["name"] = self.name
        if self.z_min != 1:
            out["z_min"] = self.z_min
        if self.z_max != 118:
            out["z_max"] = self.z_max
        out["radial"] = self.radial
        if self.molecular_cutoffs:
            out["molecular_cutoffs"] = True
        if self.default is not None:
            out["default"] = self.default.to_json()
        if self.ranges:
            out["range"] = [
                {"z_min": lo, "z_max": hi, **element.to_json()}
                for lo, hi, element in self.ranges
            ]
        if self.elements:
            out["element"] = [
                {"z": z, **self.elements[z].to_json()} for z in sorted(self.elements)
            ]
        return out

    def validate(self) -> None:
        if self.z_min == 0 or self.z_min > self.z_max or self.z_max > 118:
            raise PresetError("preset element range")
        for z in self.elements:
            if not 1 <= z <= 118:
                raise PresetError(f"unsupported element Z = {z}")
            if not self.z_min <= z <= self.z_max:
                raise PresetError("preset element range")
        for index, (lo, hi, element) in enumerate(self.ranges):
            if not (1 <= lo <= 118) or not (1 <= hi <= 118):
                raise PresetError(f"range {lo}..={hi}: unsupported element range")
            if lo > hi:
                raise PresetError(f"range {lo}..={hi}: reversed element range")
            if lo < self.z_min or hi > self.z_max:
                raise PresetError(
                    f"range {lo}..={hi}: preset default range outside the element domain"
                )
            element.validate(f"range {lo}..={hi}")
            if self.molecular_cutoffs and element.cutoff is not None and _uses_shell_list(element):
                raise PresetError(
                    f"range {lo}..={hi}: an explicit per-shell list cannot follow "
                    "a moving molecular cutoff"
                )
            for olo, ohi, _ in self.ranges[:index]:
                if lo <= ohi and olo <= hi:
                    raise PresetError("overlapping preset default ranges")
        for z, element in self.elements.items():
            element.validate(f"element Z={z}")
            if self.molecular_cutoffs and element.cutoff is not None and _uses_shell_list(element):
                raise PresetError(
                    f"element Z={z}: an explicit per-shell list cannot follow "
                    "a moving molecular cutoff"
                )
        if self.default is not None:
            self.default.validate("default")
            if (
                self.molecular_cutoffs
                and self.default.cutoff is not None
                and _uses_shell_list(self.default)
            ):
                raise PresetError(
                    "default: an explicit per-shell list cannot follow a moving "
                    "molecular cutoff"
                )


def _parse_element(body: dict[str, Any], where: str, base: Element | None) -> Element:
    _reject_unknown(body, ELEMENT_KEYS, where)
    n_rad = body.get("n_rad")
    radial_scale = body.get("radial_scale")
    angular = body.get("angular")
    if base is None:
        for key, value in (("n_rad", n_rad), ("radial_scale", radial_scale), ("angular", angular)):
            if value is None:
                raise PresetError(
                    f"{where} is missing `{key}` and the preset has no default"
                )
        return Element(
            n_rad=_int(n_rad, f"{where}.n_rad"),
            radial_scale=_float(radial_scale, f"{where}.radial_scale"),
            cutoff=(
                _float(body["cutoff"], f"{where}.cutoff")
                if body.get("cutoff") is not None
                else None
            ),
            generate_all=bool(body.get("generate_all", True)),
            angular=_parse_angular(angular, f"{where}.angular"),
        )
    return Element(
        n_rad=_int(n_rad, f"{where}.n_rad") if n_rad is not None else base.n_rad,
        radial_scale=(
            _float(radial_scale, f"{where}.radial_scale")
            if radial_scale is not None
            else base.radial_scale
        ),
        cutoff=(
            _float(body["cutoff"], f"{where}.cutoff")
            if body.get("cutoff") is not None
            else base.cutoff
        ),
        generate_all=(
            bool(body["generate_all"])
            if body.get("generate_all") is not None
            else base.generate_all
        ),
        angular=(
            _parse_angular(angular, f"{where}.angular")
            if angular is not None
            else base.angular
        ),
    )


def from_mapping(data: dict[str, Any]) -> Preset:
    """Build and validate a :class:`Preset` from a parsed mapping."""
    _reject_unknown(data, PRESET_KEYS, "preset")
    version = _int(data.get("format", FORMAT), "preset.format")
    if version != FORMAT:
        raise PresetError(f"unsupported preset format {version} (expected {FORMAT})")
    if "radial" not in data:
        raise PresetError("preset is missing `radial`")
    radial = data["radial"]
    if radial not in RADIAL:
        raise PresetError(f"preset: unknown radial map {radial!r}")
    default = (
        _parse_element(data["default"], "default", None)
        if data.get("default") is not None
        else None
    )
    ranges: list[tuple[int, int, Element]] = []
    for index, entry in enumerate(data.get("range") or []):
        _reject_unknown(entry, RANGE_KEYS, f"range[{index}]")
        element = _parse_element(
            {k: v for k, v in entry.items() if k not in {"z_min", "z_max"}},
            f"range[{index}]",
            None,
        )
        ranges.append(
            (
                _int(entry["z_min"], f"range[{index}].z_min"),
                _int(entry["z_max"], f"range[{index}].z_max"),
                element,
            )
        )
    elements: dict[int, Element] = {}
    for index, entry in enumerate(data.get("element") or []):
        _reject_unknown(entry, ENTRY_KEYS, f"element[{index}]")
        z = _int(entry["z"], f"element[{index}].z")
        base = None
        for lo, hi, element in ranges:
            if lo <= z <= hi:
                base = element
                break
        if base is None:
            base = default
        element = _parse_element(
            {k: v for k, v in entry.items() if k != "z"},
            f"element[{index}] (Z = {z})",
            base,
        )
        if z in elements:
            raise PresetError(f"duplicate element Z = {z} in preset")
        elements[z] = element
    preset = Preset(
        name=data.get("name", ""),
        z_min=_int(data.get("z_min", 1), "preset.z_min"),
        z_max=_int(data.get("z_max", 118), "preset.z_max"),
        radial=radial,
        molecular_cutoffs=bool(data.get("molecular_cutoffs", False)),
        default=default,
        ranges=ranges,
        elements=elements,
    )
    preset.validate()
    return preset


def loads(text: str, *, format: str) -> Preset:
    """Parse a preset from TOML or JSON text (``format`` is explicit)."""
    if format == "toml":
        if tomllib is None:
            raise PresetError("TOML input requires Python 3.11+")
        data = tomllib.loads(text)
    elif format == "json":
        data = json.loads(text)
    else:
        raise PresetError(f"unknown preset format {format!r}")
    return from_mapping(data)


def load(path: str | Path) -> Preset:
    """Read a preset file; ``.toml`` and ``.json`` suffixes are accepted."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in (".toml", ".json"):
        raise PresetError(f"{path}: expected a .toml or .json preset file")
    return loads(path.read_text(encoding="utf-8"), format=suffix[1:])


def dumps(preset: Preset, *, format: str = "json") -> str:
    """Serialize a preset in the canonical mapping (JSON only for now)."""
    if format != "json":
        raise PresetError("only canonical JSON output is implemented")
    return json.dumps(preset.to_json(), indent=2) + "\n"


@dataclass
class CatalogEntry:
    """One entry of a preset catalog."""

    name: str
    category: str
    file: str
    kind: str = "grid"
    elements: tuple[int, int] | None = None
    grid: str = ""
    scheme: int | None = None


@dataclass
class Catalog:
    """A parsed and validated preset catalog."""

    programs: dict[str, dict[str, Any]]

    @classmethod
    def from_json_str(cls, text: str) -> "Catalog":
        data = json.loads(text)
        _reject_unknown(data, {"format", "description", "programs"}, "catalog")
        if _int(data.get("format", FORMAT), "catalog.format") != FORMAT:
            raise PresetError(f"unsupported catalog format (expected {FORMAT})")
        programs = data.get("programs")
        if not isinstance(programs, dict):
            raise PresetError("catalog has no programs object")
        catalog = cls(programs=programs)
        catalog.validate()
        return catalog

    @classmethod
    def from_json_path(cls, path: str | Path) -> "Catalog":
        return cls.from_json_str(Path(path).read_text(encoding="utf-8"))

    def validate(self) -> None:
        for program, block in self.programs.items():
            if not isinstance(block, dict):
                raise PresetError(f"program `{program}`: expected an object")
            _reject_unknown(
                block,
                {"software", "reference_basis", "parameters", "presets"},
                f"program `{program}`",
            )
            entries = block.get("presets", [])
            if not isinstance(entries, list):
                raise PresetError(f"program `{program}`: presets is not a list")
            names: set[str] = set()
            files: set[str] = set()
            for index, entry in enumerate(entries):
                where = f"program `{program}` entry[{index}]"
                if not isinstance(entry, dict):
                    raise PresetError(f"{where}: expected an object")
                _reject_unknown(
                    entry,
                    {
                        "name", "category", "file", "kind", "grid", "scheme",
                        "elements", "reference_basis", "basis", "basis_profile",
                        "notes",
                    },
                    where,
                )
                name = entry.get("name")
                if not isinstance(name, str) or not name:
                    raise PresetError(f"{where}: missing name")
                if name in names:
                    raise PresetError(f"program `{program}`: duplicate preset name `{name}`")
                names.add(name)
                rel = entry.get("file")
                if not isinstance(rel, str) or not rel:
                    raise PresetError(f"{where}: missing file")
                if Path(rel).is_absolute() or ".." in Path(rel).parts:
                    raise PresetError(f"{where}: unsafe relative path {rel!r}")
                if rel in files:
                    raise PresetError(f"program `{program}`: preset `{name}` reuses file `{rel}`")
                files.add(rel)
                if entry.get("kind", "grid") == "grid":
                    elements = entry.get("elements")
                    if (
                        not isinstance(elements, list)
                        or len(elements) != 2
                        or not all(isinstance(z, int) and not isinstance(z, bool) for z in elements)
                        or elements[0] == 0
                        or elements[0] > elements[1]
                        or elements[1] > 118
                    ):
                        raise PresetError(
                            f"program `{program}`: grid entry `{name}` has no valid element range"
                        )

    def entry(self, program: str, name: str) -> CatalogEntry:
        block = self.programs.get(program)
        if block is None:
            raise PresetError(f"program `{program}` is not in the catalog")
        entries = block.get("presets", [])
        for entry in entries:
            if entry.get("name") == name:
                elements = entry.get("elements")
                return CatalogEntry(
                    name=name,
                    category=entry.get("category", ""),
                    file=entry["file"],
                    kind=entry.get("kind", "grid"),
                    elements=tuple(elements) if elements else None,
                    grid=entry.get("grid", ""),
                    scheme=entry.get("scheme"),
                )
        raise PresetError(f"preset `{name}` is not in program `{program}`")

    def load(self, program: str, name: str, base_dir: str | Path) -> Preset:
        entry = self.entry(program, name)
        if entry.kind != "grid":
            raise PresetError(f"program `{program}` preset `{name}` is not a grid")
        base_dir = Path(base_dir)
        path = base_dir / entry.file
        if path.suffix.lower() != ".json":
            raise PresetError(f"{path}: catalog preset files must be JSON")
        if entry.elements is None:
            raise PresetError(f"{path}: grid entry `{name}` has no element range")
        z_min, z_max = entry.elements
        preset = load(path)
        if preset.name and preset.name != name:
            raise PresetError(
                f"{path}: preset name `{preset.name}` does not match catalog entry `{name}`"
            )
        if z_min < preset.z_min or z_max > preset.z_max:
            raise PresetError(
                f"{path}: catalog range [{z_min}, {z_max}] exceeds the preset "
                f"domain {preset.z_min}..={preset.z_max}"
            )
        preset.z_min = z_min
        preset.z_max = z_max
        try:
            preset.validate()
        except PresetError as error:
            raise PresetError(f"{path}: preset `{name}`: {error}") from error
        return preset


def load_catalog(path: str | Path) -> Catalog:
    """Read and validate a catalog file."""
    return Catalog.from_json_path(path)
