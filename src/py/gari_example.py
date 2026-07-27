#!/usr/bin/env python3
"""Decode source-circuit samples using saved GARI matrix artifacts.

The ``.dem`` stores transformed matrices, not a physical error model; only the
source circuit is sampled, and virtual detector entries remain zero.
The layout must be the unchanged companion file written by ``gari_convert``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import stim
import tesseract_decoder


_DECODER_PRESET = "tesseract-short-beam"


def _workspace_path(path: Path) -> Path:
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    return (
        Path(workspace) / path
        if workspace and not path.is_absolute()
        else path
    )


def _run(
    circuit_path: Path,
    dem_path: Path,
    layout_path: Path,
    *,
    shots: int,
    seed: int,
) -> None:
    circuit = stim.Circuit.from_file(str(circuit_path))
    gari_dem = stim.DetectorErrorModel.from_file(str(dem_path))
    with layout_path.open(encoding="utf-8") as file:
        layout = json.load(file)

    source_samples, actual_observables = circuit.compile_detector_sampler(
        seed=seed
    ).sample(shots=shots, separate_observables=True)
    gari_samples = np.zeros(
        (source_samples.shape[0], gari_dem.num_detectors), dtype=np.bool_
    )
    source_to_gari = np.asarray(layout["source_to_gari"])
    gari_samples[:, source_to_gari] = source_samples

    decoder = tesseract_decoder.make_tesseract_sinter_decoders_dict()[
        _DECODER_PRESET
    ]
    decoder.num_det_orders = 1
    compiled_decoder = decoder.compile_decoder_for_dem(dem=gari_dem)
    predictions = compiled_decoder.decoder.decode_batch(gari_samples)
    decoder_order_count = len(compiled_decoder.decoder.config.det_orders)
    if predictions.shape != actual_observables.shape:
        raise ValueError("Circuit and GARI observable counts differ.")
    logical_failures = int(
        np.count_nonzero(np.any(predictions != actual_observables, axis=1))
    )

    print("GARI saved-artifact decoding completed")
    print(f"Source circuit:        {circuit_path}")
    print(f"GARI DEM file:         {dem_path} (.dem matrix storage; not sampled)")
    print(f"Detector layout:       {layout_path}")
    print(f"Source detectors:      {circuit.num_detectors}")
    print(f"GARI detectors:        {gari_dem.num_detectors}")
    print(f"Prior policy:          {layout['prior_policy']}")
    print(f"Decoder preset:        {_DECODER_PRESET}")
    print(f"Decoder order count:   {decoder_order_count}")
    print(f"Logical placement:     {layout['logical_placement']}")
    print(f"GARI row order:        {layout['detector_order']}")
    print(f"Shots:                 {shots}")
    print(f"Logical failures:      {logical_failures}/{shots}")
    print("This small run is a functional smoke check, not a benchmark or proof.")


def main(argv: list[str] | None = None) -> None:
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

    _run(
        _workspace_path(args.circuit),
        _workspace_path(args.dem),
        _workspace_path(args.gari_layout),
        shots=args.shots,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
