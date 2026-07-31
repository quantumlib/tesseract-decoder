# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from pathlib import Path
from typing import List

import numpy as np
import stim


def _command_path(path: str) -> str:
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if workspace and path != "-" and not Path(path).is_absolute():
        return str(Path(workspace, path))
    return path


def get_dets_logicals(error: stim.DemInstruction):
    dets = set()
    logicals = set()
    for t in error.targets_copy():
        if t.is_logical_observable_id():
            logicals = logicals.symmetric_difference({t.val})
        elif t.is_relative_detector_id():
            dets = dets.symmetric_difference({t.val})
    return dets, logicals


def spatial_key(
    detector_coords: dict, min_t_coord: float, max_t_coord: float, dets, logicals
):
    d_coords = sorted([tuple(detector_coords[d]) for d in dets])
    min_d_coord = d_coords[0]
    relative_d_coords = [tuple(np.array(c) - np.array(min_d_coord)) for c in d_coords]
    min_xy = (min_d_coord[0], min_d_coord[1])
    min_t_error = min(c[2] for c in d_coords)
    max_t_error = min(c[2] for c in d_coords)
    is_begin = bool(min_t_error == min_t_coord)
    is_end = bool(max_t_error == max_t_coord)
    rel_coords = tuple(sorted(relative_d_coords))
    return (min_xy, rel_coords, tuple(logicals), is_begin, is_end)
    # return (min_xy, rel_coords, tuple(logicals))


def get_detector_coords(dem: stim.DetectorErrorModel):
    detector_coords = {}
    for inst in dem.flattened():
        if inst.type != "detector":
            continue
        coords = np.array(inst.args_copy())
        dets = inst.targets_copy()
        D = dets[0].val
        detector_coords[D] = coords[:3]
    min_t_coord = min(c[2] for c in detector_coords.values())
    max_t_coord = max(c[2] for c in detector_coords.values())
    return detector_coords, min_t_coord, max_t_coord


# Analyze the errors to make the flip tables
def merged_errors(dem):
    errors_by_symptom = {}
    for error in dem.flattened():
        if error.type != "error":
            continue
        probability = error.args_copy()[0]
        assert 0 <= probability and probability <= 1, error
        detectors, observables = get_dets_logicals(error)
        key = (tuple(sorted(detectors)), tuple(sorted(observables)))
        if key in errors_by_symptom:
            p0 = errors_by_symptom[key]["probability"]
            probability = p0 * (1 - probability) + (1 - p0) * probability
        error = {
            "probability": probability,
            "likelihood_cost": -np.log(probability / (1 - probability)),
            "detectors": list(detectors),
            "observables": list(observables),
        }
        errors_by_symptom[key] = error

    return list(errors_by_symptom.values())


def get_key_to_probabilities(spatial_data, template, verbose=False):
    key_to_probabilities = {}
    for error in merged_errors(template):
        probability = error["probability"]
        key = spatial_key(*spatial_data, error["detectors"], error["observables"])
        if key not in key_to_probabilities:
            key_to_probabilities[key] = []
        key_to_probabilities[key].append(probability)
    if verbose:
        print(
            f"identified {len(key_to_probabilities)} distinct errors out of {template.num_errors}"
        )
    return key_to_probabilities


def merge_concat(dictionaries: List[dict]):
    merged = {}
    for d in dictionaries:
        for k in d:
            if k not in merged:
                merged[k] = []
            merged[k] = np.concatenate([merged[k], d[k]])
    return merged


def generalize(
    templates: List[stim.DetectorErrorModel],
    scaffold: stim.DetectorErrorModel,
    verbose: bool = False,
) -> stim.DetectorErrorModel:
    # Get detector coords for all detectors
    spatial_data_scaffold = get_detector_coords(scaffold)
    # Build a lookup table from unique key to probabilities
    all_key_to_probabilities = [
        get_key_to_probabilities(
            get_detector_coords(template), template, verbose=verbose
        )
        for template in templates
    ]
    key_to_probabilities = merge_concat(all_key_to_probabilities)
    key_to_probability = {
        key: float(np.mean(probabilities))
        for key, probabilities in key_to_probabilities.items()
    }
    output_dem = stim.DetectorErrorModel()
    for instruction in scaffold.flattened():
        if instruction.type != "error":
            output_dem.append(instruction)
    for error in merged_errors(scaffold):
        # update the probability
        key = spatial_key(
            *spatial_data_scaffold, error["detectors"], error["observables"]
        )
        inst = stim.DemInstruction(
            type="error",
            args=[key_to_probability[key]],
            targets=[stim.target_relative_detector_id(D) for D in error["detectors"]]
            + [stim.target_logical_observable_id(L) for L in error["observables"]],
        )
        output_dem.append(inst)

    return output_dem


def call_generalize(
    template_fnames: List[str],
    scaffold_fname: str,
    output_fname: str,
    verbose: bool = False,
):
    template_dems = [
        stim.DetectorErrorModel.from_file(template_fname)
        for template_fname in template_fnames
    ]
    scaffold_dem = stim.DetectorErrorModel.from_file(scaffold_fname)
    output_dem = generalize(template_dems, scaffold_dem, verbose)
    if output_fname == "-":
        print(output_dem)
    else:
        output_dem.to_file(output_fname)


def call_gari(circuit_fname: str, prior_name: str, output_prefix: str):
    from _tesseract_py_util import gari

    prior_function = {
        "paper": gari.paper_prior_probabilities,
        "xor": gari.tesseract_xor_prior_probabilities,
        "lp-max-barred-cost": gari.tesseract_lp_max_barred_cost_prior_probabilities,
    }[prior_name]
    gari_dem, layout = gari.circuit_to_gari(
        stim.Circuit.from_file(circuit_fname), prior_function=prior_function
    )
    output_name = f"{output_prefix}-gari-{prior_name}"
    gari_dem.to_file(f"{output_name}.dem")
    Path(f"{output_name}-layout.json").write_text(
        json.dumps(layout, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main():
    import argparse
    import sys

    argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Generalize detector error models or create GARI files."
    )
    if argv[:1] == ["gari"]:
        parser.add_argument("--circuit", required=True)
        parser.add_argument(
            "--prior",
            choices=("paper", "xor", "lp-max-barred-cost"),
            required=True,
        )
        parser.add_argument("--out-prefix", required=True)
        args = parser.parse_args(argv[1:])
        call_gari(
            _command_path(args.circuit), args.prior, _command_path(args.out_prefix)
        )
        return

    parser.add_argument(
        "--template",
        required=True,
        action="append",
        help="Template file names (at least one required)",
    )
    parser.add_argument("--scaffold", required=True, help="Scaffold file name")
    parser.add_argument(
        "--out", required=True, help="Output file name (use '-' for stdout)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    args = parser.parse_args(argv)
    call_generalize(
        [_command_path(path) for path in args.template],
        _command_path(args.scaffold),
        _command_path(args.out),
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
