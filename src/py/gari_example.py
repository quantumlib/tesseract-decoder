#!/usr/bin/env python3
"""Decode source-circuit samples using saved GARI matrix artifacts.

The ``.dem`` stores transformed matrices, not a physical error model; only the
source circuit is sampled, and virtual detector entries remain zero.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import stim
import tesseract_decoder


_LAYOUT_SCHEMA = "tesseract.gari_layout.v1"
_DETECTOR_ORDER = "physical_then_virtual"
_LOGICAL_PLACEMENT = "physical"
_PRIOR_POLICIES = {"paper", "xor", "lp-maximin"}
_ROW_BLOCKS = ("physical_x", "physical_z", "virtual_z", "virtual_x")


def _workspace_path(path: Path) -> Path:
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    return (
        Path(workspace) / path
        if workspace and not path.is_absolute()
        else path
    )


def _required_count(layout: dict[str, object], name: str) -> int:
    value = layout.get(name)
    if type(value) is not int or value < 0:
        raise ValueError(f"Layout field {name!r} must be a nonnegative integer.")
    return value


def _required_text(layout: dict[str, object], name: str) -> str:
    value = layout.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Layout field {name!r} must be a nonempty string.")
    return value


def _validate_row_blocks(
    layout: dict[str, object], source_count: int, gari_count: int
) -> None:
    blocks = layout.get("row_blocks")
    if not isinstance(blocks, dict):
        raise ValueError("Layout field 'row_blocks' must be an object.")
    expected_start = 0
    for name in _ROW_BLOCKS:
        interval = blocks.get(name)
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or any(type(endpoint) is not int for endpoint in interval)
        ):
            raise ValueError(
                f"row_blocks[{name!r}] must contain two integer endpoints."
            )
        start, stop = interval
        if start != expected_start:
            raise ValueError(
                f"row_blocks[{name!r}] must start at {expected_start}; "
                f"found {start}."
            )
        if stop < start:
            raise ValueError(f"row_blocks[{name!r}] is not a valid interval.")
        expected_start = stop

    if expected_start != gari_count:
        raise ValueError(
            f"Row blocks must end at GARI detector count {gari_count}; "
            f"found {expected_start}."
        )
    physical_stop = blocks["physical_z"][1]
    if physical_stop != source_count:
        raise ValueError(
            "Physical row blocks must contain exactly one row per source "
            f"detector; found {physical_stop} for {source_count}."
        )


def _load_layout(path: Path) -> tuple[int, int, tuple[int, ...], str, str, str]:
    with path.open(encoding="utf-8") as file:
        layout = json.load(file)
    if not isinstance(layout, dict):
        raise ValueError("GARI layout must be a JSON object.")
    if layout.get("schema") != _LAYOUT_SCHEMA:
        raise ValueError(
            f"GARI layout schema must be {_LAYOUT_SCHEMA!r}; "
            f"found {layout.get('schema')!r}."
        )

    source_count = _required_count(layout, "source_detector_count")
    gari_count = _required_count(layout, "gari_detector_count")
    if gari_count < source_count:
        raise ValueError(
            "GARI detector count must not be smaller than the source count."
        )
    mapping = layout.get("source_to_gari")
    if not isinstance(mapping, list):
        raise ValueError("Layout field 'source_to_gari' must be a list.")
    if len(mapping) != source_count:
        raise ValueError(
            "Layout field 'source_to_gari' must contain one target per "
            f"source detector; found {len(mapping)} for {source_count}."
        )
    for source, target in enumerate(mapping):
        if type(target) is not int:
            raise ValueError(
                f"source_to_gari[{source}] must be an integer; found "
                f"{target!r}."
            )
        if target < 0 or target >= gari_count:
            raise ValueError(
                f"source_to_gari[{source}]={target} is outside [0, "
                f"{gari_count})."
            )
        if target >= source_count:
            raise ValueError(
                f"source_to_gari[{source}]={target} refers to a virtual row."
            )
    if len(set(mapping)) != len(mapping):
        raise ValueError("Layout field 'source_to_gari' must be injective.")

    prior_policy = _required_text(layout, "prior_policy")
    if prior_policy not in _PRIOR_POLICIES:
        raise ValueError(f"Unknown GARI prior policy {prior_policy!r}.")
    metadata = {
        "logical_placement": _LOGICAL_PLACEMENT,
        "detector_order": _DETECTOR_ORDER,
    }
    for name, expected in metadata.items():
        value = _required_text(layout, name)
        if value != expected:
            raise ValueError(
                f"Layout field {name!r} must be {expected!r}; found {value!r}."
            )
    _validate_row_blocks(layout, source_count, gari_count)
    return (
        source_count, gari_count, tuple(mapping),
        prior_policy, _LOGICAL_PLACEMENT, _DETECTOR_ORDER,
    )


def _run(
    circuit_path: Path,
    dem_path: Path,
    layout_path: Path,
    *,
    shots: int,
    seed: int,
) -> None:
    if shots <= 0:
        raise ValueError("shots must be positive.")
    if seed < 0 or seed >= 2**64:
        raise ValueError("seed must be in [0, 2**64).")

    circuit = stim.Circuit.from_file(str(circuit_path))
    gari_dem = stim.DetectorErrorModel.from_file(str(dem_path))
    layout_values = _load_layout(layout_path)
    source_count, gari_count, source_to_gari = layout_values[:3]
    prior_policy, logical_placement, detector_order = layout_values[3:]

    count_checks = (
        ("Circuit detectors", circuit.num_detectors, source_count),
        ("GARI DEM detectors", gari_dem.num_detectors, gari_count),
        (
            "Circuit and GARI DEM observables",
            circuit.num_observables,
            gari_dem.num_observables,
        ),
    )
    for name, actual, expected in count_checks:
        if actual != expected:
            raise ValueError(
                f"{name} differ: found {actual}, expected {expected}."
            )

    source_samples, actual_observables = circuit.compile_detector_sampler(
        seed=seed
    ).sample(shots=shots, separate_observables=True)
    gari_samples = np.zeros((shots, gari_count), dtype=np.bool_)
    gari_samples[:, np.asarray(source_to_gari, dtype=np.int64)] = source_samples

    config = tesseract_decoder.tesseract.TesseractConfig(
        dem=gari_dem, det_orders=[list(range(gari_count))]
    )
    predictions = config.compile_decoder().decode_batch(gari_samples)
    if predictions.shape != actual_observables.shape:
        raise RuntimeError(
            f"Decoder returned observable shape {predictions.shape}; "
            f"expected {actual_observables.shape}."
        )
    failures = np.any(predictions != actual_observables, axis=1)
    logical_failures = int(np.count_nonzero(failures))

    print("GARI saved-artifact decoding completed")
    print(f"Source circuit:        {circuit_path}")
    print(f"GARI DEM file:         {dem_path} (.dem matrix storage; not sampled)")
    print(f"Detector layout:       {layout_path}")
    print(f"Source detectors:      {source_count}")
    print(f"GARI detectors:        {gari_count}")
    print(f"Prior policy:          {prior_policy}")
    print(f"Logical placement:     {logical_placement}")
    print(f"Detector order:        {detector_order}")
    print(f"Shots:                 {shots}")
    print(f"Logical failures:      {logical_failures}/{shots}")
    print("This small run is a functional smoke check, not a benchmark or proof.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sample a circuit and decode it using a saved GARI DEM."
    )
    parser.add_argument("--circuit", required=True, type=Path)
    parser.add_argument(
        "--dem", required=True, type=Path,
        help="Storage-only GARI matrix file; this file is never sampled.",
    )
    parser.add_argument("--gari-layout", required=True, type=Path)
    parser.add_argument("--shots", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args(argv)

    try:
        _run(
            _workspace_path(args.circuit),
            _workspace_path(args.dem),
            _workspace_path(args.gari_layout),
            shots=args.shots,
            seed=args.seed,
        )
    except (OSError, RuntimeError, ValueError) as ex:
        print(f"gari_example: {ex}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
