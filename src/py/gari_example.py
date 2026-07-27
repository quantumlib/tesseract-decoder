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
_DECODER_PRESET = "tesseract-short-beam"


def _workspace_path(path: Path) -> Path:
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    return (
        Path(workspace) / path
        if workspace and not path.is_absolute()
        else path
    )


def _load_layout(path: Path) -> tuple[int, int, tuple[int, ...], str]:
    with path.open(encoding="utf-8") as file:
        layout = json.load(file)
    if not isinstance(layout, dict) or layout.get("schema") != _LAYOUT_SCHEMA:
        raise ValueError(
            f"GARI layout must use schema {_LAYOUT_SCHEMA!r}."
        )

    source_count = layout.get("source_detector_count")
    gari_count = layout.get("gari_detector_count")
    if (
        type(source_count) is not int
        or type(gari_count) is not int
        or source_count < 0
        or gari_count < source_count
    ):
        raise ValueError(
            "Layout detector counts must be nonnegative integers with "
            "gari_detector_count >= source_detector_count."
        )

    mapping = layout.get("source_to_gari")
    if not isinstance(mapping, list) or len(mapping) != source_count:
        raise ValueError(
            "Layout source_to_gari must contain one entry per source detector."
        )
    if any(
        type(target) is not int or not 0 <= target < gari_count
        for target in mapping
    ):
        raise ValueError("Layout source_to_gari contains an invalid target.")
    if len(set(mapping)) != source_count:
        raise ValueError("Layout source_to_gari must be injective.")

    prior_policy = layout.get("prior_policy")
    if not isinstance(prior_policy, str) or not prior_policy:
        raise ValueError("Layout prior_policy must be a nonempty string.")
    if layout.get("logical_placement") != _LOGICAL_PLACEMENT:
        raise ValueError("Layout logical_placement must be 'physical'.")
    if layout.get("detector_order") != _DETECTOR_ORDER:
        raise ValueError(
            "Layout detector_order must be 'physical_then_virtual'."
        )
    return source_count, gari_count, tuple(mapping), prior_policy


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
    source_count, gari_count, source_to_gari, prior_policy = _load_layout(
        layout_path
    )

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

    decoder = tesseract_decoder.make_tesseract_sinter_decoders_dict()[
        _DECODER_PRESET
    ]
    decoder.num_det_orders = 1
    compiled_decoder = decoder.compile_decoder_for_dem(dem=gari_dem)
    predictions = compiled_decoder.decoder.decode_batch(gari_samples)
    decoder_order_count = len(compiled_decoder.decoder.config.det_orders)
    if predictions.shape != actual_observables.shape:
        raise RuntimeError(
            f"Decoder returned observable shape {predictions.shape}; "
            f"expected {actual_observables.shape}."
        )
    logical_failures = int(
        np.count_nonzero(np.any(predictions != actual_observables, axis=1))
    )

    print("GARI saved-artifact decoding completed")
    print(f"Source circuit:        {circuit_path}")
    print(f"GARI DEM file:         {dem_path} (.dem matrix storage; not sampled)")
    print(f"Detector layout:       {layout_path}")
    print(f"Source detectors:      {source_count}")
    print(f"GARI detectors:        {gari_count}")
    print(f"Prior policy:          {prior_policy}")
    print(f"Decoder preset:        {_DECODER_PRESET}")
    print(f"Decoder order count:   {decoder_order_count}")
    print(f"Logical placement:     {_LOGICAL_PLACEMENT}")
    print(f"GARI row order:        {_DETECTOR_ORDER}")
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
