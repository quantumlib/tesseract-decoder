#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import stim


STIM_RESULT_FORMATS = ("01", "b8", "r8", "ptb64", "hits", "dets")
STIM_RESULT_FORMATS_HELP = "/".join(STIM_RESULT_FORMATS)
BEAM_PRUNE_MODES = ("fixed", "delta", "mass")
BEAM_PRUNE_MODES_HELP = "/".join(BEAM_PRUNE_MODES)


@dataclass(frozen=True)
class Fault:
    q: float
    p: float
    delta_scale: float
    det_mask: int
    likelihood_cost: float


@dataclass(frozen=True)
class DecoderModel:
    faults: tuple[Fault, ...]
    retiring_masks: tuple[int, ...]
    live_masks_after: tuple[int, ...]
    future_detcost: tuple[tuple[float, ...], ...]
    all_possible_dets_mask: int
    max_width: int


@dataclass(frozen=True)
class BeamPruningConfig:
    mode: str
    beam_width: int
    score_delta: float | None
    mass_epsilon: float | None
    hard_cap: int | None


@dataclass(frozen=True)
class BeamDecodeResult:
    predicted_logical: bool | None
    certified: bool
    margin: float
    discarded_mass: float
    kept_state_counts: tuple[int, ...]
    max_width: int
    elapsed_seconds: float


@dataclass(frozen=True)
class DecodingShot:
    det_mask: int
    actual_logical: bool | None


@dataclass(frozen=True)
class IntegerSeriesSummary:
    count: int
    minimum: int | None
    median: float | None
    mean: float | None
    maximum: int | None


@dataclass
class IntegerHistogramAccumulator:
    count: int = 0
    total: int = 0
    minimum: int | None = None
    maximum: int | None = None
    histogram: dict[int, int] = field(default_factory=dict)

    def add(self, value: int) -> None:
        self.count += 1
        self.total += value
        if self.minimum is None or value < self.minimum:
            self.minimum = value
        if self.maximum is None or value > self.maximum:
            self.maximum = value
        self.histogram[value] = self.histogram.get(value, 0) + 1

    def add_many(self, values: tuple[int, ...] | list[int]) -> None:
        for value in values:
            self.add(value)

    def summary(self) -> IntegerSeriesSummary:
        if self.count == 0:
            return IntegerSeriesSummary(
                count=0,
                minimum=None,
                median=None,
                mean=None,
                maximum=None,
            )

        lower_target = (self.count - 1) // 2
        upper_target = self.count // 2
        seen = 0
        lower_value: int | None = None
        upper_value: int | None = None
        for value in sorted(self.histogram):
            seen += self.histogram[value]
            if lower_value is None and seen > lower_target:
                lower_value = value
            if upper_value is None and seen > upper_target:
                upper_value = value
                break

        assert lower_value is not None and upper_value is not None
        return IntegerSeriesSummary(
            count=self.count,
            minimum=self.minimum,
            median=(lower_value + upper_value) / 2.0,
            mean=self.total / self.count,
            maximum=self.maximum,
        )


@dataclass(frozen=True)
class ExperimentSummary:
    predictions: list[bool | None]
    num_certified: int
    num_low_confidence: int
    num_errors: int
    num_truth_shots: int
    num_scored_shots: int
    total_elapsed: float
    total_triggered: int
    max_width_seen: int
    kept_state_summary: IntegerSeriesSummary


def _likelihood_cost(probability: float) -> float:
    if probability <= 0.0:
        return math.inf
    if probability >= 1.0:
        return 0.0
    return -math.log(probability / (1.0 - probability))


def _detectors_from_mask(mask: int) -> list[int]:
    detectors: list[int] = []
    while mask:
        low_bit = mask & -mask
        detectors.append(low_bit.bit_length() - 1)
        mask ^= low_bit
    return detectors


def _mask_from_bool_row(row: np.ndarray) -> int:
    mask = 0
    for index in np.flatnonzero(row):
        mask |= 1 << int(index)
    return mask


def _future_detcost_by_detector(faults: tuple[Fault, ...], num_detectors: int) -> tuple[tuple[float, ...], ...]:
    future_detcost: list[list[float]] = [[math.inf] * num_detectors for _ in range(len(faults) + 1)]
    next_row = future_detcost[-1]
    for fault_index in range(len(faults) - 1, -1, -1):
        row = next_row.copy()
        fault = faults[fault_index]
        det_count = fault.det_mask.bit_count()
        if det_count:
            ecost = fault.likelihood_cost / det_count
            for det_id in _detectors_from_mask(fault.det_mask):
                if ecost < row[det_id]:
                    row[det_id] = ecost
        future_detcost[fault_index] = row
        next_row = row
    return tuple(tuple(row) for row in future_detcost)


def _build_decoder_model(circuit: stim.Circuit) -> DecoderModel:
    dem = circuit.detector_error_model(decompose_errors=False).flattened()

    faults: list[Fault] = []
    all_possible_dets_mask = 0
    last_seen_index: dict[int, int] = {}

    for inst in dem:
        if inst.type != "error":
            continue

        p = float(inst.args_copy()[0])
        det_mask = 0
        flip_l0 = 0
        for target in inst.targets_copy():
            if target.is_separator():
                continue
            if target.is_relative_detector_id():
                det_mask ^= 1 << target.val
            elif target.is_logical_observable_id() and target.val == 0:
                flip_l0 ^= 1

        faults.append(
            Fault(
                q=1.0 - p,
                p=p,
                delta_scale=(-p if flip_l0 else p),
                det_mask=det_mask,
                likelihood_cost=_likelihood_cost(p),
            )
        )
        all_possible_dets_mask |= det_mask

        for det_id in _detectors_from_mask(det_mask):
            last_seen_index[det_id] = len(faults) - 1

    retiring_masks = [0] * len(faults)
    for det_id, index in last_seen_index.items():
        retiring_masks[index] |= 1 << det_id

    live_masks_after = [0] * (len(faults) + 1)
    active_mask = 0
    max_width = 0
    for i, fault in enumerate(faults):
        active_mask |= fault.det_mask
        max_width = max(max_width, active_mask.bit_count())
        active_mask &= ~retiring_masks[i]
        live_masks_after[i + 1] = active_mask

    frozen_faults = tuple(faults)
    return DecoderModel(
        faults=frozen_faults,
        retiring_masks=tuple(retiring_masks),
        live_masks_after=tuple(live_masks_after),
        future_detcost=_future_detcost_by_detector(frozen_faults, circuit.num_detectors),
        all_possible_dets_mask=all_possible_dets_mask,
        max_width=max_width,
    )


def _detcost_penalty(mismatch_mask: int, future_detcost: tuple[float, ...]) -> float:
    total = 0.0
    pending = mismatch_mask

    while pending:
        low_bit = pending & -pending
        detector = low_bit.bit_length() - 1
        pending ^= low_bit

        best = future_detcost[detector]
        if best == math.inf:
            return math.inf
        total += best

    return total


def _accumulate_collapsed_state(
    collapsed_probs: dict[int, list[float]],
    *,
    state: int,
    total: float,
    delta: float,
) -> float:
    if total <= 0.0:
        return 0.0

    entry = collapsed_probs.get(state)
    if entry is None:
        collapsed_probs[state] = [total, delta]
    else:
        entry[0] += total
        entry[1] += delta
    return total


def _prune_ranked_states(
    ranked_states: list[tuple[float, float, int, float]],
    *,
    total_mass: float,
    pruning: BeamPruningConfig,
) -> tuple[list[tuple[int, float, float]], float]:
    if not ranked_states:
        return [], total_mass

    ranked_states.sort(reverse=True)

    if pruning.mode == "fixed":
        kept_ranked = ranked_states[:pruning.beam_width]
    elif pruning.mode == "delta":
        assert pruning.score_delta is not None
        best_score = ranked_states[0][0]
        if best_score == -math.inf:
            kept_ranked = ranked_states
        else:
            cutoff = best_score - pruning.score_delta
            kept_ranked = [entry for entry in ranked_states if entry[0] >= cutoff]
        if not kept_ranked:
            kept_ranked = ranked_states[:1]
        if pruning.hard_cap is not None and len(kept_ranked) > pruning.hard_cap:
            kept_ranked = kept_ranked[:pruning.hard_cap]
    elif pruning.mode == "mass":
        assert pruning.mass_epsilon is not None
        retained_target_mass = (1.0 - pruning.mass_epsilon) * total_mass
        retained_mass = 0.0
        kept_ranked = []
        for entry in ranked_states:
            kept_ranked.append(entry)
            retained_mass += entry[1]
            if retained_mass >= retained_target_mass:
                break
        if not kept_ranked:
            kept_ranked = ranked_states[:1]
        if pruning.hard_cap is not None and len(kept_ranked) > pruning.hard_cap:
            kept_ranked = kept_ranked[:pruning.hard_cap]
    else:
        raise ValueError(f"Unsupported pruning mode: {pruning.mode!r}")

    kept_mass = sum(total for _, total, _, _ in kept_ranked)
    dropped_mass = total_mass - kept_mass
    kept_beam = [(state, total, delta) for _, total, state, delta in kept_ranked]
    return kept_beam, dropped_mass


def _summarize_int_values(values: tuple[int, ...] | list[int]) -> IntegerSeriesSummary:
    if not values:
        return IntegerSeriesSummary(
            count=0,
            minimum=None,
            median=None,
            mean=None,
            maximum=None,
        )

    sorted_values = sorted(values)
    count = len(sorted_values)
    lower = sorted_values[(count - 1) // 2]
    upper = sorted_values[count // 2]
    return IntegerSeriesSummary(
        count=count,
        minimum=sorted_values[0],
        median=(lower + upper) / 2.0,
        mean=sum(sorted_values) / count,
        maximum=sorted_values[-1],
    )


def _format_optional_int(value: int | None) -> str:
    return "none" if value is None else str(value)


def _format_pruning_value(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}"


def _format_summary_int(value: int | None) -> str:
    return "n/a" if value is None else str(value)


def _format_summary_float(value: float | None, *, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _print_pruning_configuration(*, pruning: BeamPruningConfig, log_stream) -> None:
    print(f"Beam Prune Mode:     {pruning.mode}", file=log_stream)
    if pruning.mode == "fixed":
        print(f"Beam Width:          {pruning.beam_width}", file=log_stream)
    elif pruning.mode == "delta":
        print(f"Beam Score Delta:    {_format_pruning_value(pruning.score_delta)}", file=log_stream)
        print(f"Beam Hard Cap:       {_format_optional_int(pruning.hard_cap)}", file=log_stream)
    elif pruning.mode == "mass":
        assert pruning.mass_epsilon is not None
        print(f"Beam Mass Epsilon:   {_format_pruning_value(pruning.mass_epsilon)}", file=log_stream)
        print(f"Beam Retained Mass:  {_format_pruning_value(1.0 - pruning.mass_epsilon)}", file=log_stream)
        print(f"Beam Hard Cap:       {_format_optional_int(pruning.hard_cap)}", file=log_stream)
    else:
        raise ValueError(f"Unsupported pruning mode: {pruning.mode!r}")


def _beam_pruning_config_from_args(args: argparse.Namespace) -> BeamPruningConfig:
    return BeamPruningConfig(
        mode=args.beam_prune_mode,
        beam_width=args.beam,
        score_delta=args.beam_score_delta,
        mass_epsilon=args.beam_mass_epsilon,
        hard_cap=args.beam_hard_cap,
    )


def _as_bool_2d(data: np.ndarray, *, expected_cols: int, description: str) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError(f"Expected {description} to be a 2D array but got shape {arr.shape!r}.")
    if arr.shape[1] != expected_cols:
        raise ValueError(
            f"Expected {description} to have {expected_cols} columns but got {arr.shape[1]}."
        )
    if arr.dtype != np.bool_:
        arr = arr.astype(np.bool_, copy=False)
    return arr


def _sample_shot_arrays(
    circuit: stim.Circuit,
    *,
    shots: int,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    sampler = circuit.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots=shots, separate_observables=True)
    return (
        _as_bool_2d(dets, expected_cols=circuit.num_detectors, description="sampled detector data"),
        _as_bool_2d(obs, expected_cols=circuit.num_observables, description="sampled observable data"),
    )


def _read_detector_shot_arrays(
    *,
    path: str,
    fmt: str,
    num_detectors: int,
    num_observables: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    common_kwargs = dict(
        path=path,
        format=fmt,
        bit_packed=False,
        num_measurements=0,
        num_detectors=num_detectors,
        num_observables=num_observables,
    )

    if num_observables:
        try:
            dets, obs = stim.read_shot_data_file(**common_kwargs, separate_observables=True)
            return (
                _as_bool_2d(dets, expected_cols=num_detectors, description="input detector data"),
                _as_bool_2d(obs, expected_cols=num_observables, description="appended observable data"),
            )
        except TypeError:
            flat = stim.read_shot_data_file(**common_kwargs)
            flat = _as_bool_2d(
                flat,
                expected_cols=num_detectors + num_observables,
                description="combined detector/observable input data",
            )
            return flat[:, :num_detectors], flat[:, num_detectors:]

    flat = stim.read_shot_data_file(**common_kwargs)
    return _as_bool_2d(flat, expected_cols=num_detectors, description="input detector data"), None


def _read_observable_shot_array(*, path: str, fmt: str, num_observables: int) -> np.ndarray:
    obs = stim.read_shot_data_file(
        path=path,
        format=fmt,
        bit_packed=False,
        num_measurements=0,
        num_detectors=0,
        num_observables=num_observables,
    )
    return _as_bool_2d(obs, expected_cols=num_observables, description="observable input data")


def _apply_shot_range(
    dets: np.ndarray,
    obs: np.ndarray | None,
    *,
    shot_range_begin: int,
    shot_range_end: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if not (shot_range_begin or shot_range_end):
        return dets, obs

    if shot_range_end < shot_range_begin:
        raise ValueError("Provided shot range must satisfy --shot-range-end >= --shot-range-begin.")
    if shot_range_end > len(dets):
        raise ValueError(
            f"Shot range end {shot_range_end} is past the end of the shot data (size {len(dets)})."
        )

    dets = dets[shot_range_begin:shot_range_end]
    if obs is not None:
        obs = obs[shot_range_begin:shot_range_end]
    return dets, obs


def _shots_from_arrays(dets: np.ndarray, obs: np.ndarray | None) -> list[DecodingShot]:
    shots: list[DecodingShot] = []
    for shot_index in range(dets.shape[0]):
        actual_logical = None if obs is None else bool(obs[shot_index, 0])
        shots.append(
            DecodingShot(
                det_mask=_mask_from_bool_row(dets[shot_index]),
                actual_logical=actual_logical,
            )
        )
    return shots


def _resolve_stdin_path_if_needed(path: str, *, temp_dir: str, stem: str) -> str:
    if path != "-":
        return path
    temp_path = str(Path(temp_dir) / f"{stem}.bin")
    with open(temp_path, "wb") as f:
        f.write(sys.stdin.buffer.read())
    return temp_path


def _resolve_stdout_path_if_needed(path: str, *, temp_dir: str, stem: str) -> tuple[str, bool]:
    if path != "-":
        return path, False
    return str(Path(temp_dir) / f"{stem}.bin"), True


def _copy_file_to_stdout(path: str) -> None:
    sys.stdout.flush()
    with open(path, "rb") as f:
        shutil.copyfileobj(f, sys.stdout.buffer)
    sys.stdout.buffer.flush()


def _load_shots(
    circuit: stim.Circuit,
    args: argparse.Namespace,
    *,
    temp_dir: str,
) -> list[DecodingShot]:
    if args.in_file:
        in_path = _resolve_stdin_path_if_needed(args.in_file, temp_dir=temp_dir, stem="shots_in")
        appended_obs_count = circuit.num_observables if args.in_includes_appended_observables else 0
        dets, obs = _read_detector_shot_arrays(
            path=in_path,
            fmt=args.in_format,
            num_detectors=circuit.num_detectors,
            num_observables=appended_obs_count,
        )

        if args.obs_in_file:
            obs_in_path = _resolve_stdin_path_if_needed(args.obs_in_file, temp_dir=temp_dir, stem="obs_in")
            obs = _read_observable_shot_array(
                path=obs_in_path,
                fmt=args.obs_in_format,
                num_observables=circuit.num_observables,
            )
            if len(obs) != len(dets):
                raise ValueError("Observable input ended before, or after, the detector shot data.")
    else:
        dets, obs = _sample_shot_arrays(circuit, shots=args.sample_num_shots, seed=args.sample_seed)

    dets, obs = _apply_shot_range(
        dets,
        obs,
        shot_range_begin=args.shot_range_begin,
        shot_range_end=args.shot_range_end,
    )
    return _shots_from_arrays(dets, obs)


def decode_beam_search_detcost_ranked(
    model: DecoderModel,
    actual_dets_mask: int,
    pruning: BeamPruningConfig,
) -> BeamDecodeResult:
    start_time = time.perf_counter()
    retained_state_counts: list[int] = []

    if (actual_dets_mask & ~model.all_possible_dets_mask) != 0:
        return BeamDecodeResult(
            predicted_logical=None,
            certified=False,
            margin=0.0,
            discarded_mass=0.0,
            kept_state_counts=(0,) * len(model.faults),
            max_width=model.max_width,
            elapsed_seconds=time.perf_counter() - start_time,
        )

    beam = [(0, 1.0, 1.0)]
    discarded_mass = 0.0

    for i, fault in enumerate(model.faults):
        collapsed_probs: dict[int, list[float]] = {}
        total_mass = 0.0
        retiring_mask = model.retiring_masks[i]

        if retiring_mask == 0:
            for state, total, delta in beam:
                total_mass += _accumulate_collapsed_state(
                    collapsed_probs,
                    state=state,
                    total=total * fault.q,
                    delta=delta * fault.q,
                )

                total_mass += _accumulate_collapsed_state(
                    collapsed_probs,
                    state=state ^ fault.det_mask,
                    total=total * fault.p,
                    delta=delta * fault.delta_scale,
                )
        else:
            expected_bits = actual_dets_mask & retiring_mask
            keep_mask = ~retiring_mask
            for state, total, delta in beam:
                absent_total = total * fault.q
                if absent_total > 0.0 and (state & retiring_mask) == expected_bits:
                    total_mass += _accumulate_collapsed_state(
                        collapsed_probs,
                        state=state & keep_mask,
                        total=absent_total,
                        delta=delta * fault.q,
                    )

                toggled = state ^ fault.det_mask
                present_total = total * fault.p
                if present_total > 0.0 and (toggled & retiring_mask) == expected_bits:
                    total_mass += _accumulate_collapsed_state(
                        collapsed_probs,
                        state=toggled & keep_mask,
                        total=present_total,
                        delta=delta * fault.delta_scale,
                    )

        if total_mass == 0.0:
            retained_state_counts.append(0)
            retained_state_counts.extend([0] * (len(model.faults) - i - 1))
            return BeamDecodeResult(
                predicted_logical=None,
                certified=False,
                margin=0.0,
                discarded_mass=discarded_mass,
                kept_state_counts=tuple(retained_state_counts),
                max_width=model.max_width,
                elapsed_seconds=time.perf_counter() - start_time,
            )

        ranked_states: list[tuple[float, float, int, float]] = []
        live_target_mask = actual_dets_mask & model.live_masks_after[i + 1]
        next_future_detcost = model.future_detcost[i + 1]
        for state, (total, delta) in collapsed_probs.items():
            if total <= 0.0:
                continue
            mismatch_mask = state ^ live_target_mask
            penalty = _detcost_penalty(mismatch_mask=mismatch_mask, future_detcost=next_future_detcost)
            if penalty == math.inf:
                rank_score = -math.inf
            else:
                rank_score = math.log(total) - penalty
            ranked_states.append((rank_score, total, state, delta))

        beam, dropped_mass = _prune_ranked_states(
            ranked_states,
            total_mass=total_mass,
            pruning=pruning,
        )
        retained_state_counts.append(len(beam))

        inv_total_mass = 1.0 / total_mass
        discarded_mass = (discarded_mass + dropped_mass) * inv_total_mass
        beam = [
            (state, total * inv_total_mass, delta * inv_total_mass)
            for state, total, delta in beam
        ]

    _, _, final_delta = next((entry for entry in beam if entry[0] == 0), (0, 0.0, 0.0))
    margin = abs(final_delta)
    certified = margin > discarded_mass

    if final_delta == 0.0:
        return BeamDecodeResult(
            predicted_logical=None,
            certified=False,
            margin=margin,
            discarded_mass=discarded_mass,
            kept_state_counts=tuple(retained_state_counts),
            max_width=model.max_width,
            elapsed_seconds=time.perf_counter() - start_time,
        )
    return BeamDecodeResult(
        predicted_logical=final_delta < 0.0,
        certified=certified,
        margin=margin,
        discarded_mass=discarded_mass,
        kept_state_counts=tuple(retained_state_counts),
        max_width=model.max_width,
        elapsed_seconds=time.perf_counter() - start_time,
    )


def _print_run_header(
    *,
    circuit: stim.Circuit,
    args: argparse.Namespace,
    pruning: BeamPruningConfig,
    num_faults: int,
    num_shots: int,
    log_stream,
) -> None:
    print(f"Running on circuit {args.circuit}", file=log_stream)
    print(f"Total Detectors:      {circuit.num_detectors}", file=log_stream)
    print(f"Total Observables:    {circuit.num_observables}", file=log_stream)
    print(f"Total Faults:         {num_faults}", file=log_stream)
    _print_pruning_configuration(pruning=pruning, log_stream=log_stream)
    if args.in_file:
        print(f"Shot Input:           {args.in_file}", file=log_stream)
        print(f"Shot Input Format:    {args.in_format}", file=log_stream)
        if args.in_includes_appended_observables:
            print("Observable Input:     appended to --in", file=log_stream)
        elif args.obs_in_file:
            print(f"Observable Input:     {args.obs_in_file}", file=log_stream)
            print(f"Observable Format:    {args.obs_in_format}", file=log_stream)
        else:
            print("Observable Input:     none", file=log_stream)
    else:
        print(f"Sample Seed:          {args.sample_seed}", file=log_stream)
        print(f"Requested Shots:      {args.sample_num_shots}", file=log_stream)
    if args.shot_range_begin or args.shot_range_end:
        print(
            f"Shot Range:           [{args.shot_range_begin}, {args.shot_range_end})",
            file=log_stream,
        )
    print(f"Num Shots:            {num_shots}", file=log_stream)


def run_experiment(args: argparse.Namespace) -> ExperimentSummary:
    circuit = stim.Circuit.from_file(args.circuit)
    if circuit.num_observables != 1:
        raise ValueError(
            "This decoder currently supports exactly one logical observable, because it only tracks L0. "
            f"The circuit has {circuit.num_observables} observables."
        )

    model = _build_decoder_model(circuit)
    pruning = _beam_pruning_config_from_args(args)
    log_stream = sys.stderr if args.out_file == "-" else sys.stdout

    with tempfile.TemporaryDirectory() as temp_dir:
        shots = _load_shots(circuit, args, temp_dir=temp_dir)
        _print_run_header(
            circuit=circuit,
            args=args,
            pruning=pruning,
            num_faults=len(model.faults),
            num_shots=len(shots),
            log_stream=log_stream,
        )

        num_errors = 0
        num_low_confidence = 0
        num_certified = 0
        num_truth_shots = 0
        num_scored_shots = 0
        total_elapsed = 0.0
        total_triggered = 0
        max_width_seen = 0
        predictions: list[bool | None] = []
        kept_state_accumulator = IntegerHistogramAccumulator()

        for shot_index, shot in enumerate(shots):
            result = decode_beam_search_detcost_ranked(model, shot.det_mask, pruning)
            predictions.append(result.predicted_logical)
            kept_state_accumulator.add_many(result.kept_state_counts)
            kept_state_summary = _summarize_int_values(result.kept_state_counts)

            success: bool | None
            if shot.actual_logical is None or result.predicted_logical is None:
                success = None
            else:
                success = result.predicted_logical == shot.actual_logical

            if result.predicted_logical is None:
                num_low_confidence += 1
            if shot.actual_logical is not None:
                num_truth_shots += 1
                if success is not None:
                    num_scored_shots += 1
                    if not success:
                        num_errors += 1
            if result.certified:
                num_certified += 1

            total_elapsed += result.elapsed_seconds
            triggered_dets = shot.det_mask.bit_count()
            total_triggered += triggered_dets
            max_width_seen = max(max_width_seen, result.max_width)

            shots_done = shot_index + 1
            error_rate_so_far = num_errors / num_scored_shots if num_scored_shots else 0.0
            print(
                f"progress shots_done={shots_done}/{len(shots)} errors_so_far={num_errors} "
                f"low_conf_so_far={num_low_confidence} scored_shots_so_far={num_scored_shots} "
                f"error_rate_so_far={error_rate_so_far:.6f} elapsed_total_seconds={total_elapsed:.6f} "
                f"kept_states_min={_format_summary_int(kept_state_summary.minimum)} "
                f"kept_states_median={_format_summary_float(kept_state_summary.median)} "
                f"kept_states_mean={_format_summary_float(kept_state_summary.mean)} "
                f"kept_states_max={_format_summary_int(kept_state_summary.maximum)}",
                file=log_stream,
            )

            if args.print_per_shot:
                print(
                    f"shot={shot_index} triggered_detectors={triggered_dets} "
                    f"predicted_logical={result.predicted_logical} actual_logical={shot.actual_logical} "
                    f"success={success} certified={result.certified} "
                    f"margin={result.margin:.6e} discarded_mass={result.discarded_mass:.6e} "
                    f"kept_states_min={_format_summary_int(kept_state_summary.minimum)} "
                    f"kept_states_median={_format_summary_float(kept_state_summary.median)} "
                    f"kept_states_mean={_format_summary_float(kept_state_summary.mean)} "
                    f"kept_states_max={_format_summary_int(kept_state_summary.maximum)} "
                    f"elapsed_seconds={result.elapsed_seconds:.6f}",
                    file=log_stream,
                )

        if args.out_file:
            output_path, copy_to_stdout = _resolve_stdout_path_if_needed(
                args.out_file,
                temp_dir=temp_dir,
                stem="predictions_out",
            )
            prediction_data = np.zeros((len(predictions), circuit.num_observables), dtype=np.bool_)
            for shot_index, predicted_logical in enumerate(predictions):
                prediction_data[shot_index, 0] = bool(predicted_logical) if predicted_logical is not None else False

            if args.out_format == "ptb64" and len(prediction_data) % 64 != 0:
                raise ValueError("The ptb64 format requires the number of shots to be a multiple of 64.")

            stim.write_shot_data_file(
                data=prediction_data,
                path=output_path,
                format=args.out_format,
                num_measurements=0,
                num_detectors=0,
                num_observables=circuit.num_observables,
            )
            if copy_to_stdout:
                _copy_file_to_stdout(output_path)
            if num_low_confidence:
                print(
                    f"warning: wrote {num_low_confidence} low-confidence predictions as L0=0 because Stim result "
                    "files can only store bits, not unknown values.",
                    file=log_stream,
                )

    kept_state_summary = kept_state_accumulator.summary()

    print(f"Mean Triggered Dets:  {total_triggered / max(1, len(shots)):.2f}", file=log_stream)
    print(f"Max Width:            {max_width_seen}", file=log_stream)
    print(f"{'Kept States/Fault Min:':<26}{_format_summary_int(kept_state_summary.minimum)}", file=log_stream)
    print(f"{'Kept States/Fault Median:':<26}{_format_summary_float(kept_state_summary.median)}", file=log_stream)
    print(f"{'Kept States/Fault Mean:':<26}{_format_summary_float(kept_state_summary.mean)}", file=log_stream)
    print(f"{'Kept States/Fault Max:':<26}{_format_summary_int(kept_state_summary.maximum)}", file=log_stream)
    print(f"Certified Shots:      {num_certified}", file=log_stream)
    print(f"Low Confidence:       {num_low_confidence}", file=log_stream)
    print(f"Truth-Labeled Shots:  {num_truth_shots}", file=log_stream)
    print(f"Scored Shots:         {num_scored_shots}", file=log_stream)
    if num_truth_shots:
        print(f"Logical Errors:       {num_errors}", file=log_stream)
    else:
        print("Logical Errors:       n/a", file=log_stream)
    print(f"Total Seconds:        {total_elapsed:.6f}", file=log_stream)
    print(f"Mean Seconds/Shot:    {total_elapsed / max(1, len(shots)):.6f}", file=log_stream)

    return ExperimentSummary(
        predictions=predictions,
        num_certified=num_certified,
        num_low_confidence=num_low_confidence,
        num_errors=num_errors,
        num_truth_shots=num_truth_shots,
        num_scored_shots=num_scored_shots,
        total_elapsed=total_elapsed,
        total_triggered=total_triggered,
        max_width_seen=max_width_seen,
        kept_state_summary=kept_state_summary,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run trellis beam decoding ranked by mass minus a detcost-style future penalty, "
            "with optional adaptive threshold pruning and Stim-compatible shot-data I/O options."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--circuit", required=True, help="Path to the .stim circuit file.")
    parser.add_argument(
        "--beam",
        type=int,
        default=1000,
        help="Beam width cutoff used when --beam-prune-mode=fixed.",
    )
    parser.add_argument(
        "--beam-prune-mode",
        choices=BEAM_PRUNE_MODES,
        default="fixed",
        help=(
            "Beam pruning rule: fixed keeps the top --beam states, delta keeps all states within "
            "--beam-score-delta of the best rank score, and mass keeps a rank-sorted prefix whose "
            "retained normalized mass reaches 1-epsilon."
        ),
    )
    parser.add_argument(
        "--beam-score-delta",
        type=float,
        default=None,
        help=(
            "For --beam-prune-mode=delta, keep every state whose rank score is within this additive "
            "gap of the best state's rank score."
        ),
    )
    parser.add_argument(
        "--beam-mass-epsilon",
        type=float,
        default=None,
        help=(
            "For --beam-prune-mode=mass, keep the smallest rank-sorted prefix whose retained "
            "normalized mass is at least 1 - epsilon."
        ),
    )
    parser.add_argument(
        "--beam-retained-mass",
        type=float,
        default=None,
        help=(
            "For --beam-prune-mode=mass, equivalent to setting --beam-mass-epsilon to "
            "1 - retained_mass."
        ),
    )
    parser.add_argument(
        "--beam-hard-cap",
        type=int,
        default=None,
        help=(
            "Optional hard cap on the number of states retained after delta or mass thresholding. "
            "Ignored in fixed mode."
        ),
    )
    parser.add_argument(
        "--sample-num-shots",
        type=int,
        default=None,
        help="Number of sampled shots. Defaults to 1 unless --in is provided.",
    )
    parser.add_argument("--sample-seed", type=int, default=None, help="Stim sampler seed.")
    parser.add_argument(
        "--shot-range-begin",
        type=int,
        default=0,
        help=(
            "If both --shot-range-begin and --shot-range-end are 0, decode all available shots. "
            "Otherwise only decode shots in [begin, end)."
        ),
    )
    parser.add_argument(
        "--shot-range-end",
        type=int,
        default=0,
        help=(
            "If both --shot-range-begin and --shot-range-end are 0, decode all available shots. "
            "Otherwise only decode shots in [begin, end)."
        ),
    )
    parser.add_argument(
        "--in",
        dest="in_file",
        default="",
        help="File to read detection events from (use - for stdin).",
    )
    parser.add_argument(
        "--in-format",
        "--in_format",
        dest="in_format",
        choices=STIM_RESULT_FORMATS,
        default="01",
        help=f"Format of the file read by --in ({STIM_RESULT_FORMATS_HELP}).",
    )
    parser.add_argument(
        "--in-includes-appended-observables",
        "--in_includes_appended_observables",
        dest="in_includes_appended_observables",
        action="store_true",
        help="Assume the observable flips are appended to each shot in --in.",
    )
    parser.add_argument(
        "--obs-in",
        "--obs_in",
        dest="obs_in_file",
        default="",
        help="File to read observable flips from (use - for stdin).",
    )
    parser.add_argument(
        "--obs-in-format",
        "--obs_in_format",
        dest="obs_in_format",
        choices=STIM_RESULT_FORMATS,
        default="01",
        help=f"Format of the file read by --obs-in ({STIM_RESULT_FORMATS_HELP}).",
    )
    parser.add_argument(
        "--out",
        dest="out_file",
        default="",
        help="File to write predicted observable flips to (use - for stdout).",
    )
    parser.add_argument(
        "--out-format",
        "--out_format",
        dest="out_format",
        choices=STIM_RESULT_FORMATS,
        default="01",
        help=f"Format of the file written by --out ({STIM_RESULT_FORMATS_HELP}).",
    )
    parser.add_argument(
        "--print-per-shot",
        action="store_true",
        help="Print a detailed line per decoded shot.",
    )
    args = parser.parse_args()

    if args.sample_num_shots is None:
        # Preserve the original script's one-shot default while still allowing
        # file input without requiring --sample-num-shots 0.
        args.sample_num_shots = 0 if args.in_file else 1

    if args.beam <= 0:
        raise ValueError("--beam must be positive.")
    if args.beam_hard_cap is not None and args.beam_hard_cap <= 0:
        raise ValueError("--beam-hard-cap must be positive when provided.")
    if args.beam_score_delta is not None:
        if math.isnan(args.beam_score_delta) or args.beam_score_delta < 0.0:
            raise ValueError("--beam-score-delta must be a non-negative number.")
    if args.beam_mass_epsilon is not None:
        if math.isnan(args.beam_mass_epsilon) or not (0.0 <= args.beam_mass_epsilon < 1.0):
            raise ValueError("--beam-mass-epsilon must satisfy 0 <= epsilon < 1.")
    if args.beam_retained_mass is not None:
        if math.isnan(args.beam_retained_mass) or not (0.0 <= args.beam_retained_mass <= 1.0):
            raise ValueError("--beam-retained-mass must satisfy 0 <= retained_mass <= 1.")
        if args.beam_mass_epsilon is not None:
            raise ValueError("Choose at most one of --beam-mass-epsilon and --beam-retained-mass.")
        args.beam_mass_epsilon = 1.0 - args.beam_retained_mass
    if args.sample_num_shots < 0:
        raise ValueError("--sample-num-shots must be non-negative.")
    if args.sample_seed is not None and args.sample_seed < 0:
        raise ValueError("--sample-seed must be non-negative.")
    if args.shot_range_begin < 0 or args.shot_range_end < 0:
        raise ValueError("--shot-range-begin and --shot-range-end must be non-negative.")
    if args.shot_range_end < args.shot_range_begin:
        raise ValueError("Provided shot range must satisfy --shot-range-end >= --shot-range-begin.")
    if args.in_includes_appended_observables and args.obs_in_file:
        raise ValueError(
            "Choose either --in-includes-appended-observables or --obs-in, not both."
        )
    if args.obs_in_file and not args.in_file:
        raise ValueError("Cannot load observable flips from --obs-in without also providing --in.")
    if args.in_file == "-" and args.obs_in_file == "-":
        raise ValueError("At most one of --in and --obs-in may read from stdin.")

    if args.beam_prune_mode == "fixed":
        if args.beam_score_delta is not None:
            raise ValueError("--beam-score-delta is only valid with --beam-prune-mode=delta.")
        if args.beam_mass_epsilon is not None:
            raise ValueError(
                "--beam-mass-epsilon/--beam-retained-mass are only valid with --beam-prune-mode=mass."
            )
        if args.beam_hard_cap is not None:
            raise ValueError("--beam-hard-cap is only meaningful with adaptive pruning modes.")
    elif args.beam_prune_mode == "delta":
        if args.beam_score_delta is None:
            raise ValueError("--beam-prune-mode=delta requires --beam-score-delta.")
        if args.beam_mass_epsilon is not None:
            raise ValueError(
                "--beam-mass-epsilon/--beam-retained-mass are not valid with --beam-prune-mode=delta."
            )
    elif args.beam_prune_mode == "mass":
        if args.beam_mass_epsilon is None:
            raise ValueError(
                "--beam-prune-mode=mass requires --beam-mass-epsilon or --beam-retained-mass."
            )
        if args.beam_score_delta is not None:
            raise ValueError("--beam-score-delta is not valid with --beam-prune-mode=mass.")
    else:
        raise ValueError(f"Unsupported --beam-prune-mode {args.beam_prune_mode!r}.")

    num_shot_sources = int(args.sample_num_shots > 0) + int(bool(args.in_file))
    if num_shot_sources != 1:
        raise ValueError("Requires exactly one source of shots: either --sample-num-shots > 0 or --in.")

    return args


if __name__ == "__main__":
    run_experiment(_parse_args())
