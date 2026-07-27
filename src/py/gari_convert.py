#!/usr/bin/env python3

"""Converts Stim circuits into GARI DEM and detector-layout files.

The ``.dem`` file stores the GARI transformed check and logical matrices using
Stim syntax. It is not a physical detector error model and must not be sampled.
The layout JSON maps source detector samples into the GARI detector space.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import stim

from _tesseract_py_util.gari import (
    build_gari_dem,
    circuit_to_gari_source_dem,
    dem_to_matrices,
    detector_partition_from_fourth_coordinate,
    gari_transform,
    paper_prior_probabilities,
    tesseract_lp_maximin_prior_probabilities,
    tesseract_xor_prior_probabilities,
)


_LAYOUT_SCHEMA = "tesseract.gari_layout.v1"
_BASIS_CONVENTION = "color-code-style-fourth-coordinate"
_PRIOR_FUNCTIONS = {
    "paper": paper_prior_probabilities,
    "xor": tesseract_xor_prior_probabilities,
    "lp-maximin": tesseract_lp_maximin_prior_probabilities,
}


def _workspace_path(value: str | Path) -> Path:
    path = Path(value)
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if workspace and not path.is_absolute():
        return Path(workspace) / path
    return path


def _output_paths(
    circuit_path: Path,
    prior_policy: str,
    output_prefix: Path | None,
) -> tuple[Path, Path]:
    prefix = output_prefix or (
        circuit_path.parent
        / "gari"
        / f"{circuit_path.stem}-gari-{prior_policy}"
    )
    return Path(f"{prefix}.dem"), Path(f"{prefix}-layout.json")


def _circuit_paths(circuit_directory: Path) -> list[Path]:
    if circuit_directory.is_symlink() or not circuit_directory.is_dir():
        raise ValueError(
            f"Circuit directory is not a directory: {circuit_directory}"
        )
    paths = sorted(
        (
            path
            for path in circuit_directory.rglob("*.stim", recurse_symlinks=False)
            if path.is_file()
        ),
        key=lambda path: path.relative_to(circuit_directory).as_posix(),
    )
    if not paths:
        raise ValueError(
            f"No .stim circuits found under {circuit_directory}."
        )
    return paths


def _layout_dict(transform, prior_policy: str) -> dict[str, object]:
    blocks = {
        "physical_x": transform.physical_x_rows,
        "physical_z": transform.physical_z_rows,
        "virtual_z": transform.virtual_z_rows,
        "virtual_x": transform.virtual_x_rows,
    }
    return {
        "schema": _LAYOUT_SCHEMA,
        "source_detector_count": len(transform.source_to_gari_detectors),
        "gari_detector_count": transform.checks.shape[0],
        "source_to_gari": [
            int(value) for value in transform.source_to_gari_detectors
        ],
        "row_blocks": {
            name: [int(rows.start), int(rows.stop)]
            for name, rows in blocks.items()
        },
        "detector_order": "physical_then_virtual",
        "logical_placement": "physical",
        "prior_policy": prior_policy,
    }


def _write_gari_outputs(
    gari_dem_path: Path,
    gari_dem_text: str,
    layout_path: Path,
    layout_text: str,
    *,
    force: bool,
) -> None:
    outputs = [
        (gari_dem_path, gari_dem_text),
        (layout_path, layout_text),
    ]
    paths = [path for path, _ in outputs]
    if any(path.is_dir() for path in paths):
        raise IsADirectoryError("A GARI output path is a directory.")
    existing = [path for path in paths if os.path.lexists(path)]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Output already exists: {names}. Use --force to replace both "
            "GARI output files."
        )

    gari_dem_path.parent.mkdir(parents=True, exist_ok=True)
    scratch_paths: list[Path] = []
    backups: dict[Path, Path] = {}
    published: list[Path] = []

    def scratch_file(contents: str) -> Path:
        descriptor, name = tempfile.mkstemp(
            dir=gari_dem_path.parent, prefix=".gari-convert-"
        )
        path = Path(name)
        scratch_paths.append(path)
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(contents)
        return path

    try:
        staged = [scratch_file(contents) for _, contents in outputs]
        for path in existing:
            backup = scratch_file("")
            os.replace(path, backup)
            backups[path] = backup
        for temporary, final in zip(staged, paths):
            os.replace(temporary, final)
            published.append(final)
    except BaseException:
        for path in published:
            path.unlink(missing_ok=True)
        for path, backup in backups.items():
            if os.path.lexists(backup):
                os.replace(backup, path)
        raise
    finally:
        for path in scratch_paths:
            path.unlink(missing_ok=True)


def _convert_circuit(
    circuit_path: Path,
    *,
    prior_policy: str,
    output_prefix: Path | None,
    force: bool,
):
    if prior_policy not in _PRIOR_FUNCTIONS:
        raise ValueError(f"Unknown GARI prior policy {prior_policy!r}.")

    gari_dem_path, layout_path = _output_paths(
        circuit_path, prior_policy, output_prefix
    )
    for path in [gari_dem_path, layout_path]:
        aliases_input = circuit_path.resolve(strict=False) == path.resolve(
            strict=False
        )
        if not aliases_input and os.path.lexists(path):
            try:
                aliases_input = os.path.samefile(circuit_path, path)
            except OSError:
                pass
        if aliases_input:
            raise ValueError(
                f"Output path {path} aliases source circuit {circuit_path}; "
                "refusing to overwrite the input."
            )

    circuit = stim.Circuit.from_file(str(circuit_path))
    source_error_model = circuit_to_gari_source_dem(circuit)
    checks, logicals, probabilities = dem_to_matrices(source_error_model)
    x_detectors, z_detectors = detector_partition_from_fourth_coordinate(
        source_error_model
    )
    transform = gari_transform(
        checks,
        logicals,
        x_detectors=x_detectors,
        z_detectors=z_detectors,
    )
    gari_dem = build_gari_dem(
        transform,
        probabilities,
        prior_function=_PRIOR_FUNCTIONS[prior_policy],
    )

    gari_dem_text = str(gari_dem)
    if not gari_dem_text.endswith("\n"):
        gari_dem_text += "\n"
    layout_text = json.dumps(
        _layout_dict(transform, prior_policy), indent=2, sort_keys=True
    ) + "\n"
    _write_gari_outputs(
        gari_dem_path,
        gari_dem_text,
        layout_path,
        layout_text,
        force=force,
    )
    return source_error_model, transform, gari_dem_path, layout_path


def _convert_directory(
    circuit_directory: Path,
    *,
    prior_policy: str,
    force: bool,
) -> int:
    circuit_paths = _circuit_paths(circuit_directory)
    failures = 0
    for circuit_path in circuit_paths:
        relative_path = circuit_path.relative_to(circuit_directory)
        try:
            _convert_circuit(
                circuit_path,
                prior_policy=prior_policy,
                output_prefix=None,
                force=force,
            )
        except (OSError, RuntimeError, ValueError) as ex:
            failures += 1
            print(f"ERROR {relative_path}: {ex}", file=sys.stderr)
        else:
            print(f"OK    {relative_path}")

    print(
        f"\nRepository scan: {circuit_directory}\n"
        f"Circuits found:  {len(circuit_paths)}\n"
        f"Converted:       {len(circuit_paths) - failures}\n"
        f"Failed:          {failures}\n"
        "GARI DEM files store matrices only; do not sample them."
    )
    return int(failures != 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert correlated CSS Stim circuits into GARI matrix .dem "
            "storage files and detector-layout JSON files."
        )
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--circuit", type=_workspace_path)
    inputs.add_argument(
        "--circuit-directory",
        type=_workspace_path,
        help="Recursively convert .stim circuits in deterministic order.",
    )
    parser.add_argument(
        "--prior-policy", required=True, choices=list(_PRIOR_FUNCTIONS)
    )
    parser.add_argument(
        "--basis-convention",
        choices=[_BASIS_CONVENTION],
        default=_BASIS_CONVENTION,
        help=(
            "Use the repository testdata's color-code-style fourth "
            "coordinate: values <= 2 are X and values >= 3 are Z."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=_workspace_path,
        help="Custom output prefix for --circuit only.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing GARI output files.",
    )
    args = parser.parse_args(argv)

    if args.circuit_directory is not None:
        if args.output_prefix is not None:
            parser.error("--output-prefix can only be used with --circuit.")
    try:
        if args.circuit_directory is not None:
            return _convert_directory(
                args.circuit_directory,
                prior_policy=args.prior_policy,
                force=args.force,
            )
        assert args.circuit is not None
        source_error_model, transform, gari_dem_path, layout_path = (
            _convert_circuit(
                args.circuit,
                prior_policy=args.prior_policy,
                output_prefix=args.output_prefix,
                force=args.force,
            )
        )
    except (OSError, RuntimeError, ValueError) as ex:
        print(f"gari_convert: {ex}", file=sys.stderr)
        return 1

    row_counts = [
        ("Physical X rows", transform.physical_x_rows),
        ("Physical Z rows", transform.physical_z_rows),
        ("Virtual Z rows", transform.virtual_z_rows),
        ("Virtual X rows", transform.virtual_x_rows),
    ]
    print("GARI transformed-matrix outputs created\n")
    print(f"Source circuit:        {args.circuit}")
    print(f"Source detectors:      {source_error_model.num_detectors}")
    print(f"GARI detectors:        {transform.checks.shape[0]}")
    for label, rows in row_counts:
        print(f"{label + ':':23}{rows.stop - rows.start}")
    print(f"Prior policy:          {args.prior_policy}")
    print("Logical placement:     physical")
    print("Detector order:        physical_then_virtual\n")
    print("GARI DEM (.dem matrix storage only; do not sample):")
    print(f"  {gari_dem_path}\n")
    print("Detector layout:")
    print(f"  {layout_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
