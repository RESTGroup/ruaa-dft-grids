#!/usr/bin/env python3
"""Compare the crate's preset grids with PySCF's own construction.

The script runs the ``user_presets`` example in dump mode and compares each
selected element's atomic grid with
``pyscf.dft.gen_grid.gen_atomic_grids`` at the same level (the shipped preset
file is PySCF ``Grids.level = 3``).  For every element it checks

* the number of radial shells and the point count;
* the per-shell angular size, shell radius and radial volume element;
* the total quadrature weight;
* the full point set: every coordinate and every quadrature weight.

Requirements: PySCF, NumPy and a Rust toolchain.  The check is optional and
is not part of ``cargo test``.

Usage::

    python3 examples/presets/check_against_pyscf.py
    python3 examples/presets/check_against_pyscf.py --elements 1,8,26,54,86
    python3 examples/presets/check_against_pyscf.py --elements all
    python3 examples/presets/check_against_pyscf.py --tolerance 1e-11

The element list accepts a comma-separated list of nuclear charges or
``all`` (Z = 1-103, the domain of PySCF's Treutler scale table).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

import pyscf
from pyscf import gto
from pyscf.data.elements import ELEMENTS
from pyscf.dft import gen_grid, radi

# The committed preset file is PySCF Grids.level = 3.
LEVEL = 3

# Periodic-row boundaries: 0 = H/He, ..., 6 = Fr-Og.
PERIOD_CUTS = (2, 10, 18, 36, 54, 86, 118)

# The Treutler scale table ends at Z = 103.
MAX_Z = 103

REPO = Path(__file__).resolve().parents[2]


def period_index(z: int) -> int:
    return sum(z > cut for cut in PERIOD_CUTS)


def parse_dump(text: str) -> dict[int, list[dict]]:
    """Parse the example's ``--dump`` output into shell records."""
    grids: dict[int, list[dict]] = {}
    current_z = None
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        kind = fields[0]
        if kind == "element":
            current_z = int(fields[1])
            if current_z in grids:
                raise ValueError(f"duplicate element Z = {current_z} in dump")
            grids[current_z] = []
        elif kind == "shell":
            if current_z is None:
                raise ValueError("shell record before any element record")
            shells = grids[current_z]
            index, n_ang = int(fields[1]), int(fields[2])
            if index != len(shells):
                raise ValueError(f"shell {index} out of order for Z = {current_z}")
            shells.append({"n_ang": n_ang, "coords": [], "weights": []})
        elif kind == "point":
            if current_z is None or not grids[current_z]:
                raise ValueError("point record before any shell record")
            shell = grids[current_z][-1]
            if len(fields) != 5:
                raise ValueError(f"malformed point record: {line}")
            shell["coords"].append([float(value) for value in fields[1:4]])
            shell["weights"].append(float(fields[4]))
        else:
            raise ValueError(f"unknown dump record: {line}")
    for z, shells in grids.items():
        for shell in shells:
            if len(shell["coords"]) != shell["n_ang"]:
                raise ValueError(
                    f"Z = {z}: shell has {len(shell['coords'])} points, "
                    f"expected {shell['n_ang']}"
                )
    return grids


def run_example(cargo: str, elements: str, debug: bool) -> str:
    command = [cargo, "run", "--quiet"]
    if not debug:
        command.append("--release")
    command += ["--example", "user_presets", "--", "--dump", "--elements", elements]
    completed = subprocess.run(
        command, cwd=REPO, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr)
        raise SystemExit(
            f"cargo run --example user_presets failed "
            f"(exit code {completed.returncode})"
        )
    return completed.stdout


def pyscf_atomic_grid(z: int) -> tuple[np.ndarray, np.ndarray]:
    """PySCF's atomic grid for one element at ``LEVEL``.

    The molecule carries a one-function synthetic basis: the atomic grid does
    not depend on the basis, and this keeps every element constructible.
    """
    symbol = ELEMENTS[z]
    molecule = gto.M(
        atom=f"{symbol} 0 0 0",
        unit="Bohr",
        verbose=0,
        spin=z % 2,
        basis={symbol: [[0, [1.0, 1.0]]]},
    )
    table = gen_grid.gen_atomic_grids(
        molecule,
        level=LEVEL,
        radi_method=radi.treutler,
        prune=gen_grid.nwchem_prune,
    )
    return table[symbol]


def expected_shells(z: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PySCF's radial nodes, per-shell angular sizes and radial weights."""
    period = period_index(z)
    n_rad = int(gen_grid.RAD_GRIDS[LEVEL][period])
    n_ang = int(gen_grid.LEBEDEV_ORDER[gen_grid.ANG_ORDER[LEVEL][period]])
    radii, increments = radi.treutler(n_rad, z)
    angular = np.asarray(gen_grid.nwchem_prune(z, radii, n_ang), dtype=int)
    radial_weights = 4.0 * np.pi * radii**2 * increments
    return radii, angular, radial_weights


def compare(z: int, shells: list[dict], tolerance: float) -> tuple[list[str], dict]:
    """Compare one dumped element with PySCF; return errors and statistics."""
    errors: list[str] = []
    symbol = ELEMENTS[z]
    coords_ref, weights_ref = pyscf_atomic_grid(z)
    radii_ref, angular_ref, radial_ref = expected_shells(z)

    if len(shells) != len(radii_ref):
        errors.append(
            f"{len(shells)} shells in the dump, PySCF has {len(radii_ref)}"
        )

    stats = {"max_dr": 0.0, "max_dw_rad": 0.0, "max_dxyz": 0.0, "max_dw": 0.0}
    rust_coords = []
    rust_weights = []
    for index, shell in enumerate(shells):
        coords = np.asarray(shell["coords"], dtype=float)
        weights = np.asarray(shell["weights"], dtype=float)
        rust_coords.append(coords)
        rust_weights.append(weights)
        if index >= len(radii_ref):
            break
        if shell["n_ang"] != angular_ref[index]:
            errors.append(
                f"shell {index}: {shell['n_ang']} angular points, "
                f"PySCF prunes to {angular_ref[index]}"
            )
        radius = np.linalg.norm(coords, axis=1) if len(coords) else np.empty(0)
        if len(radius):
            stats["max_dr"] = max(
                stats["max_dr"], float(np.max(np.abs(radius - radii_ref[index])))
            )
            stats["max_dw_rad"] = max(
                stats["max_dw_rad"],
                float(abs(weights.sum() - radial_ref[index])),
            )

    if not rust_coords:
        return errors, stats
    coords = np.vstack(rust_coords)
    weights = np.concatenate(rust_weights)
    if len(coords) != len(coords_ref):
        errors.append(f"{len(coords)} points in the dump, PySCF has {len(coords_ref)}")
        return errors, stats

    # Same points in a different serialization order: sort both by coordinate.
    order_dump = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
    order_pyscf = np.lexsort(
        (coords_ref[:, 2], coords_ref[:, 1], coords_ref[:, 0])
    )
    stats["max_dxyz"] = float(
        np.max(np.abs(coords[order_dump] - coords_ref[order_pyscf]))
    )
    stats["max_dw"] = float(
        np.max(np.abs(weights[order_dump] - weights_ref[order_pyscf]))
    )
    stats["total_dw"] = float(abs(weights.sum() - weights_ref.sum()))

    if stats["max_dr"] > tolerance:
        errors.append(f"max |dr| = {stats['max_dr']:.3e} > {tolerance:.1e}")
    if stats["max_dw_rad"] > tolerance:
        errors.append(f"max |dw_rad| = {stats['max_dw_rad']:.3e} > {tolerance:.1e}")
    if stats["max_dxyz"] > tolerance:
        errors.append(f"max |dxyz| = {stats['max_dxyz']:.3e} > {tolerance:.1e}")
    if stats["max_dw"] > tolerance:
        errors.append(f"max |dw| = {stats['max_dw']:.3e} > {tolerance:.1e}")
    if stats["total_dw"] > tolerance:
        errors.append(f"total weight differs by {stats['total_dw']:.3e}")

    return errors, stats


def parse_elements(text: str) -> list[int]:
    if text == "all":
        return list(range(1, MAX_Z + 1))
    elements = [int(part) for part in text.split(",") if part.strip()]
    if not elements:
        raise SystemExit("--elements is empty")
    for z in elements:
        if not 1 <= z <= MAX_Z:
            raise SystemExit(f"Z = {z} is outside 1-{MAX_Z}")
    return elements


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare the shipped preset with PySCF's atomic grids."
    )
    parser.add_argument(
        "--elements",
        default="1,6,8,10,26,54,86",
        help="comma-separated nuclear charges, or 'all' (default: 1,6,8,10,26,54,86)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="absolute tolerance for coordinates and weights (default: 1e-9)",
    )
    parser.add_argument("--cargo", default="cargo", help="cargo executable")
    parser.add_argument(
        "--debug", action="store_true", help="run the example in the debug profile"
    )
    args = parser.parse_args()

    elements = parse_elements(args.elements)
    if max(elements) > MAX_Z:
        raise SystemExit(f"the preset file covers Z = 1-{MAX_Z}")
    dumped = parse_dump(run_example(args.cargo, args.elements, args.debug))

    noun = "element" if len(elements) == 1 else "elements"
    print(
        f"ruaa_dft_grids preset vs PySCF {pyscf.__version__} "
        f"(level {LEVEL}, {len(elements)} {noun})"
    )
    failures = 0
    for z in elements:
        if z not in dumped:
            print(f"Z = {z:3d} {ELEMENTS[z]:3s}: FAIL - element missing from the dump")
            failures += 1
            continue
        errors, stats = compare(z, dumped[z], args.tolerance)
        status = "PASS" if not errors else "FAIL"
        print(
            f"Z = {z:3d} {ELEMENTS[z]:3s}: {len(dumped[z]):3d} shells, "
            f"{sum(s['n_ang'] for s in dumped[z]):6d} points, "
            f"max|dxyz|={stats['max_dxyz']:.2e} max|dw|={stats['max_dw']:.2e}  {status}"
        )
        for error in errors:
            print(f"          {error}")
        failures += len(errors)

    if failures:
        print(f"\n{failures} check(s) failed")
        return 1
    print(f"\nall checks passed (tolerance {args.tolerance:.1e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
