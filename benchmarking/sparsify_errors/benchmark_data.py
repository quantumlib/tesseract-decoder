# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Strict loading, enrichment, and aggregation for sparsification benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import stim
from tesseract_decoder import common as tesseract_common


CODE_FAMILY_BY_DIRECTORY = {
    "surfacecodes": "surfacecodes",
    "colorcodes": "colorcodes",
    "bivariatebicyclecodes": "bivariatebicyclecodes",
}

AGGREGATE_TYPES: dict[str, type | tuple[type, ...]] = {
    "circuit_path": str,
    "dem_path": str,
    "det_beam": int,
    "det_order_seed": int,
    "det_penalty": (int, float),
    "no_revisit_dets": bool,
    "num_det_orders": int,
    "pqlimit": int,
    "sparsify_base_degree": int,
    "sparsify_errors": bool,
    "sparsify_max_degree": int,
    "sparsify_reactivate_limit": int,
    "total_time_seconds": (int, float),
    "num_shots": int,
    "num_low_confidence": int,
    "num_errors": int,
}

METADATA_TYPES: dict[str, type | tuple[type, ...]] = {
    "basis": str,
    "code_family": str,
    "distance": int,
    "num_compiled_errors": int,
    "num_detectors": int,
    "num_qubits": int,
    "num_raw_dem_errors": int,
    "physical_error_rate": (int, float),
    "rounds": int,
    "circuit_sha256": str,
    "det_order_method": str,
    "merge_errors": bool,
}

RAW_ONLY_TYPES: dict[str, type | tuple[type, ...]] = {
    "beam_climbing": bool,
    "det_order_method": str,
    "max_errors": int,
    "merge_errors": bool,
    "num_threads": int,
    "sample_num_shots": int,
    "sample_seed": int,
}

RAW_MODEL_TYPES: dict[str, type | tuple[type, ...]] = {
    "num_compiled_errors": int,
    "num_detectors": int,
    "num_mandatory_errors": (int, type(None)),
    "num_optional_errors": (int, type(None)),
    "num_raw_dem_errors": int,
}

GROUP_FIELDS = tuple(AGGREGATE_TYPES) + (
    "beam_climbing",
    "det_order_method",
    "max_errors",
    "merge_errors",
    "num_threads",
    "sample_num_shots",
)

FIXED_SWEEP_FIELDS = (
    "beam_climbing",
    "det_beam",
    "det_order_seed",
    "det_order_method",
    "det_penalty",
    "max_errors",
    "merge_errors",
    "no_revisit_dets",
    "num_det_orders",
    "num_threads",
    "pqlimit",
    "sample_num_shots",
)

RUN_MANIFEST_TYPES: dict[str, type | tuple[type, ...]] = {
    "schema_version": int,
    "run_id": str,
    "created_at_utc": str,
    "tesseract_commit": str,
    "stim_revision": str,
    "hardware_description": str,
    "tesseract_binary_sha256": str,
    "git_dirty": bool,
    "det_order_method": str,
    "merge_errors": bool,
    "circuit_sha256": dict,
    "expected_job_count": int,
    "sample_seed_namespace": int,
    "sample_seed_scheme": str,
    "sample_seed_stride": int,
    "sweep": dict,
}

RUN_SWEEP_TYPES: dict[str, type | tuple[type, ...]] = {
    "include_baseline": bool,
    "repetitions_per_configuration": int,
    "sparsify_base_degree_by_directory": dict,
    "sparsify_max_degree": int,
    "sparsify_reactivate_limits": list,
}

SUPPORTED_DET_ORDER_METHODS = {"bfs", "coordinate", "index"}
SUPPORTED_SAMPLE_SEED_SCHEME = "run_namespace_times_stride_plus_job_index"

OUTCOME_FIELDS = (
    "num_errors",
    "num_low_confidence",
    "num_shots",
    "total_time_seconds",
)


class BenchmarkDataError(ValueError):
    """Raised when benchmark input is incomplete, inconsistent, or invalid."""


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"non-standard numeric constant {value!r}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def _parse_json(text: str, source: str) -> Any:
    try:
        return json.loads(
            text,
            parse_constant=_reject_nonstandard_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as ex:
        message = ex.msg if isinstance(ex, json.JSONDecodeError) else str(ex)
        raise BenchmarkDataError(f"{source}: invalid JSON: {message}") from ex


def _context(source: str, row: Mapping[str, Any]) -> str:
    circuit = row.get("circuit_path", "<missing circuit_path>")
    return f"{source} [{circuit}]"


def _matches_type(value: Any, expected: type | tuple[type, ...]) -> bool:
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(expected, tuple) and int in expected:
        return isinstance(value, expected) and not isinstance(value, bool)
    return isinstance(value, expected)


def _require_fields(
    row: Mapping[str, Any],
    field_types: Mapping[str, type | tuple[type, ...]],
    source: str,
) -> None:
    missing = sorted(set(field_types) - set(row))
    if missing:
        raise BenchmarkDataError(
            f"{_context(source, row)}: missing required fields: {', '.join(missing)}"
        )
    for field, expected in field_types.items():
        if not _matches_type(row[field], expected):
            expected_names = (
                "/".join(t.__name__ for t in expected)
                if isinstance(expected, tuple)
                else expected.__name__
            )
            raise BenchmarkDataError(
                f"{_context(source, row)}: field {field!r} must be {expected_names}, "
                f"got {type(row[field]).__name__}"
            )


def validate_aggregate_row(
    row: Mapping[str, Any], source: str, *, require_metadata: bool = True
) -> None:
    """Validates one already-aggregated benchmark row."""

    _require_fields(row, AGGREGATE_TYPES, source)
    if require_metadata:
        _require_fields(row, METADATA_TYPES, source)

    if not row["circuit_path"]:
        raise BenchmarkDataError(
            f"{_context(source, row)}: circuit_path must not be empty"
        )
    for field in ("num_errors", "num_low_confidence"):
        if row[field] < 0:
            raise BenchmarkDataError(
                f"{_context(source, row)}: {field} must be non-negative"
            )
    if row["num_shots"] <= 0:
        raise BenchmarkDataError(f"{_context(source, row)}: num_shots must be positive")
    if row["num_errors"] + row["num_low_confidence"] > row["num_shots"]:
        raise BenchmarkDataError(
            f"{_context(source, row)}: errors plus low-confidence shots exceed num_shots"
        )
    if row["total_time_seconds"] < 0:
        raise BenchmarkDataError(
            f"{_context(source, row)}: total_time_seconds must be non-negative"
        )
    for field in ("det_penalty", "total_time_seconds"):
        if not math.isfinite(row[field]):
            raise BenchmarkDataError(f"{_context(source, row)}: {field} must be finite")
    if row["det_beam"] < 0:
        raise BenchmarkDataError(
            f"{_context(source, row)}: det_beam must be non-negative"
        )
    for field in ("num_det_orders", "pqlimit"):
        if row[field] <= 0:
            raise BenchmarkDataError(
                f"{_context(source, row)}: {field} must be positive"
            )

    if row["sparsify_errors"]:
        if row["sparsify_base_degree"] <= 0:
            raise BenchmarkDataError(
                f"{_context(source, row)}: sparsify_base_degree must be positive"
            )
        max_degree = row["sparsify_max_degree"]
        if max_degree != -1 and max_degree < row["sparsify_base_degree"]:
            raise BenchmarkDataError(
                f"{_context(source, row)}: sparsify_max_degree must be -1 or "
                "at least sparsify_base_degree"
            )
        if row["sparsify_reactivate_limit"] < -1:
            raise BenchmarkDataError(
                f"{_context(source, row)}: sparsify_reactivate_limit must be -1 or non-negative"
            )
    elif any(
        row[field] != -1
        for field in (
            "sparsify_base_degree",
            "sparsify_max_degree",
            "sparsify_reactivate_limit",
        )
    ):
        raise BenchmarkDataError(
            f"{_context(source, row)}: non-sparsified rows must use -1 for all "
            "sparsification parameters"
        )

    if not require_metadata:
        return

    for field in (
        "distance",
        "num_compiled_errors",
        "num_detectors",
        "num_qubits",
        "num_raw_dem_errors",
        "rounds",
    ):
        if row[field] <= 0:
            raise BenchmarkDataError(
                f"{_context(source, row)}: {field} must be positive"
            )
    if row["basis"] not in {"X", "Z"}:
        raise BenchmarkDataError(f"{_context(source, row)}: basis must be X or Z")
    if row["code_family"] not in CODE_FAMILY_BY_DIRECTORY.values():
        raise BenchmarkDataError(
            f"{_context(source, row)}: unsupported code_family {row['code_family']!r}"
        )
    if not math.isfinite(row["physical_error_rate"]):
        raise BenchmarkDataError(
            f"{_context(source, row)}: physical_error_rate must be finite"
        )
    if not 0 < row["physical_error_rate"] < 1:
        raise BenchmarkDataError(
            f"{_context(source, row)}: physical_error_rate must be between 0 and 1"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", row["circuit_sha256"]):
        raise BenchmarkDataError(
            f"{_context(source, row)}: circuit_sha256 must be a lowercase SHA-256 digest"
        )
    if row["det_order_method"] not in SUPPORTED_DET_ORDER_METHODS:
        raise BenchmarkDataError(
            f"{_context(source, row)}: unsupported det_order_method "
            f"{row['det_order_method']!r}"
        )

    for field in ("num_mandatory_errors", "num_optional_errors"):
        if field not in row:
            raise BenchmarkDataError(
                f"{_context(source, row)}: missing required field {field}"
            )
    if row["sparsify_errors"]:
        for field in ("num_mandatory_errors", "num_optional_errors"):
            if not _matches_type(row[field], int) or row[field] < 0:
                raise BenchmarkDataError(
                    f"{_context(source, row)}: {field} must be a non-negative integer"
                )
        if (
            row["num_mandatory_errors"] + row["num_optional_errors"]
            > row["num_compiled_errors"]
        ):
            raise BenchmarkDataError(
                f"{_context(source, row)}: mandatory and optional errors exceed compiled errors"
            )
        if (
            row["sparsify_max_degree"] == -1
            and row["num_mandatory_errors"] + row["num_optional_errors"]
            != row["num_compiled_errors"]
        ):
            raise BenchmarkDataError(
                f"{_context(source, row)}: unlimited max degree must partition all "
                "compiled errors into mandatory and optional sets"
            )
    elif (
        row["num_mandatory_errors"] is not None
        or row["num_optional_errors"] is not None
    ):
        raise BenchmarkDataError(
            f"{_context(source, row)}: baseline mandatory/optional counts must be null"
        )


def read_jsonl(path: Path, *, require_metadata: bool = True) -> list[dict[str, Any]]:
    """Reads a JSONL file with strict, line-addressed validation."""

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as infile:
        for line_number, text in enumerate(infile, start=1):
            if not text.strip():
                continue
            source = f"{path}:{line_number}"
            row = _parse_json(text, source)
            if not isinstance(row, dict):
                raise BenchmarkDataError(
                    f"{source}: each JSONL value must be an object"
                )
            validate_aggregate_row(row, source, require_metadata=require_metadata)
            rows.append(row)
    if not rows:
        raise BenchmarkDataError(f"{path}: no benchmark rows found")
    return rows


def _parse_descriptor(circuit_path: str) -> dict[str, Any]:
    family_matches = [
        family
        for directory, family in CODE_FAMILY_BY_DIRECTORY.items()
        if directory in circuit_path
    ]
    if len(family_matches) != 1:
        raise BenchmarkDataError(
            f"cannot determine one code family from circuit path {circuit_path!r}"
        )

    def require(pattern: str, name: str) -> str:
        match = re.search(pattern, circuit_path)
        if match is None:
            raise BenchmarkDataError(
                f"cannot determine {name} from circuit path {circuit_path!r}"
            )
        return match.group(1)

    return {
        "basis": require(r"_([XZ])(?:,|\.stim$)", "basis"),
        "code_family": family_matches[0],
        "distance": int(require(r"(?:^|,)d=(\d+)", "distance")),
        "num_qubits": int(require(r"(?:^|,)q=(\d+)", "num_qubits")),
        "physical_error_rate": float(
            require(r"(?:^|,)p=([0-9]+(?:\.[0-9]+)?)", "physical error rate")
        ),
        "rounds": int(require(r"(?:^|/)r=(\d+)", "round count")),
    }


def _resolve_circuit(repo_root: Path, circuit_path: str) -> Path:
    root = repo_root.resolve()
    path = (root / circuit_path).resolve()
    if root != path and root not in path.parents:
        raise BenchmarkDataError(
            f"circuit path escapes repository root: {circuit_path!r}"
        )
    if not path.is_file():
        raise BenchmarkDataError(f"circuit file not found: {path}")
    return path


def _load_circuit_metadata(
    repo_root: Path, circuit_path: str
) -> tuple[dict[str, Any], list[int]]:
    path = _resolve_circuit(repo_root, circuit_path)
    circuit_bytes = path.read_bytes()
    circuit = stim.Circuit.from_file(path)
    dem = circuit.detector_error_model(
        decompose_errors=False,
        flatten_loops=False,
        allow_gauge_detectors=True,
        approximate_disjoint_errors=1,
        ignore_decomposition_failures=False,
        block_decomposition_from_introducing_remnant_edges=False,
    )
    compiled_dem = tesseract_common.remove_zero_probability_errors(
        tesseract_common.merge_indistinguishable_errors(dem)
    )
    degrees = [
        len(tesseract_common.Error(instruction).symptom.detectors)
        for instruction in compiled_dem.flattened()
        if instruction.type == "error"
    ]
    metadata = {
        **_parse_descriptor(circuit_path),
        "circuit_sha256": hashlib.sha256(circuit_bytes).hexdigest(),
        "num_compiled_errors": compiled_dem.num_errors,
        "num_detectors": compiled_dem.num_detectors,
        "num_raw_dem_errors": dem.num_errors,
    }
    if len(degrees) != compiled_dem.num_errors:
        raise BenchmarkDataError(
            f"compiled DEM count mismatch for {circuit_path}: "
            f"{len(degrees)} instructions versus {compiled_dem.num_errors} errors"
        )
    return metadata, degrees


def enrich_rows(
    rows: Sequence[Mapping[str, Any]],
    repo_root: Path,
    *,
    legacy_merge_errors: bool | None = None,
    legacy_det_order_method: str | None = None,
    expected_circuit_sha256: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Attaches exact circuit/DEM metadata to aggregate rows."""

    if (
        legacy_det_order_method is not None
        and legacy_det_order_method not in SUPPORTED_DET_ORDER_METHODS
    ):
        raise BenchmarkDataError(
            f"unsupported legacy detector-order method {legacy_det_order_method!r}"
        )

    cache: dict[str, tuple[dict[str, Any], list[int]]] = {}
    enriched: list[dict[str, Any]] = []
    for index, original in enumerate(rows, start=1):
        row = dict(original)
        validate_aggregate_row(row, f"input row {index}", require_metadata=False)
        circuit_path = row["circuit_path"]
        if row["dem_path"]:
            raise BenchmarkDataError(
                f"input row {index} [{circuit_path}]: nonempty dem_path is unsupported; "
                "metadata must be tied to the sampled circuit"
            )
        if circuit_path not in cache:
            cache[circuit_path] = _load_circuit_metadata(repo_root, circuit_path)
        metadata, degrees = cache[circuit_path]
        row.update(metadata)
        if "merge_errors" not in row:
            if legacy_merge_errors is None:
                raise BenchmarkDataError(
                    f"input row {index} [{circuit_path}]: merge_errors was not recorded; "
                    "pass an explicit legacy assumption"
                )
            row["merge_errors"] = legacy_merge_errors
        if "det_order_method" not in row:
            if legacy_det_order_method is None:
                raise BenchmarkDataError(
                    f"input row {index} [{circuit_path}]: det_order_method was not "
                    "recorded; pass an explicit legacy assumption"
                )
            row["det_order_method"] = legacy_det_order_method
        if not row["merge_errors"]:
            raise BenchmarkDataError(
                f"input row {index} [{circuit_path}]: only merge_errors=true is supported "
                "for reconstructed compiled counts"
            )
        if expected_circuit_sha256 is not None:
            expected_digest = expected_circuit_sha256.get(circuit_path)
            if expected_digest is None:
                raise BenchmarkDataError(
                    f"input row {index} [{circuit_path}]: circuit is absent from the "
                    "submission-time manifest"
                )
            if row["circuit_sha256"] != expected_digest:
                raise BenchmarkDataError(
                    f"input row {index} [{circuit_path}]: current circuit SHA-256 "
                    "does not match the submission-time manifest"
                )
        if row["sparsify_errors"]:
            base_degree = row["sparsify_base_degree"]
            max_degree = row["sparsify_max_degree"]
            row["num_mandatory_errors"] = sum(
                degree <= base_degree for degree in degrees
            )
            row["num_optional_errors"] = sum(
                degree > base_degree and (max_degree == -1 or degree <= max_degree)
                for degree in degrees
            )
        else:
            row["num_mandatory_errors"] = None
            row["num_optional_errors"] = None
        validate_aggregate_row(row, f"enriched row {index}")
        enriched.append(row)
    return enriched


def _stable_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["code_family"],
        row["distance"],
        row["num_qubits"],
        row["physical_error_rate"],
        row["basis"],
        row["circuit_path"],
        not row["sparsify_errors"],
        row["sparsify_base_degree"],
        row["sparsify_max_degree"],
        row["sparsify_reactivate_limit"],
    )


def canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    """Returns deterministically ordered, compact JSONL."""

    ordered = sorted(rows, key=_stable_sort_key)
    return "".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n"
        for row in ordered
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    write_text_atomic(path, canonical_jsonl(rows))


def write_text_atomic(path: Path, content: str) -> None:
    """Atomically replaces a UTF-8 text file in its destination directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as outfile:
            outfile.write(content)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_raw_row(row: Mapping[str, Any], source: str) -> None:
    known_fields = set(AGGREGATE_TYPES) | set(RAW_ONLY_TYPES) | set(RAW_MODEL_TYPES)
    unknown = sorted(set(row) - known_fields)
    if unknown:
        raise BenchmarkDataError(
            f"{_context(source, row)}: unknown raw fields: {', '.join(unknown)}"
        )
    validate_aggregate_row(row, source, require_metadata=False)
    _require_fields(row, RAW_ONLY_TYPES, source)
    _require_fields(row, RAW_MODEL_TYPES, source)
    if row["dem_path"]:
        raise BenchmarkDataError(
            f"{_context(source, row)}: nonempty dem_path is unsupported"
        )
    if row["det_order_method"] not in SUPPORTED_DET_ORDER_METHODS:
        raise BenchmarkDataError(
            f"{_context(source, row)}: unsupported det_order_method "
            f"{row['det_order_method']!r}"
        )
    for field in ("max_errors", "num_threads", "sample_num_shots"):
        if row[field] <= 0:
            raise BenchmarkDataError(
                f"{_context(source, row)}: {field} must be positive"
            )
    if row["sample_seed"] < 0:
        raise BenchmarkDataError(
            f"{_context(source, row)}: sample_seed must be non-negative"
        )
    for field in ("num_compiled_errors", "num_detectors", "num_raw_dem_errors"):
        if row[field] <= 0:
            raise BenchmarkDataError(
                f"{_context(source, row)}: {field} must be positive"
            )
    if row["sparsify_errors"]:
        for field in ("num_mandatory_errors", "num_optional_errors"):
            if not _matches_type(row[field], int) or row[field] < 0:
                raise BenchmarkDataError(
                    f"{_context(source, row)}: {field} must be a non-negative integer"
                )
        classified = row["num_mandatory_errors"] + row["num_optional_errors"]
        if classified > row["num_compiled_errors"] or (
            row["sparsify_max_degree"] == -1
            and classified != row["num_compiled_errors"]
        ):
            raise BenchmarkDataError(
                f"{_context(source, row)}: runtime sparsification counts are "
                "inconsistent with num_compiled_errors"
            )
    elif any(
        row[field] is not None
        for field in ("num_mandatory_errors", "num_optional_errors")
    ):
        raise BenchmarkDataError(
            f"{_context(source, row)}: baseline mandatory/optional counts must be null"
        )


def _attach_snapshot_metadata(
    rows: Sequence[tuple[str, dict[str, Any]]],
    *,
    manifest: Mapping[str, Any],
    snapshot_root: Path,
) -> None:
    """Adds fields absent from native stats using the run's immutable snapshot."""

    cache: dict[str, tuple[dict[str, Any], list[int]]] = {}
    for source, row in rows:
        validate_aggregate_row(row, source, require_metadata=False)
        circuit_path = row["circuit_path"]
        if circuit_path not in cache:
            cache[circuit_path] = _load_circuit_metadata(snapshot_root, circuit_path)
        metadata, degrees = cache[circuit_path]
        expected_digest = manifest["circuit_sha256"].get(circuit_path)
        if metadata["circuit_sha256"] != expected_digest:
            raise BenchmarkDataError(
                f"{_context(source, row)}: reconstructed circuit SHA-256 does not "
                "match the run manifest"
            )

        reconstructed = {
            "det_order_method": manifest["det_order_method"],
            "merge_errors": manifest["merge_errors"],
            "num_compiled_errors": metadata["num_compiled_errors"],
            "num_detectors": metadata["num_detectors"],
            "num_raw_dem_errors": metadata["num_raw_dem_errors"],
        }
        if row["sparsify_errors"]:
            base_degree = row["sparsify_base_degree"]
            max_degree = row["sparsify_max_degree"]
            reconstructed["num_mandatory_errors"] = sum(
                degree <= base_degree for degree in degrees
            )
            reconstructed["num_optional_errors"] = sum(
                degree > base_degree and (max_degree == -1 or degree <= max_degree)
                for degree in degrees
            )
        else:
            reconstructed["num_mandatory_errors"] = None
            reconstructed["num_optional_errors"] = None

        for field, expected in reconstructed.items():
            if field in row and row[field] != expected:
                raise BenchmarkDataError(
                    f"{_context(source, row)}: runtime field {field!r} disagrees "
                    "with the run manifest or snapshotted circuit"
                )
            row[field] = expected


def numerical_content_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hashes the original numerical/configuration fields independent of line order."""

    projected = [{field: row[field] for field in AGGREGATE_TYPES} for row in rows]
    projected.sort(key=lambda row: tuple(row[field] for field in AGGREGATE_TYPES))
    content = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in projected
    )
    return hashlib.sha256(content.encode()).hexdigest()


def aggregate_raw_rows(
    rows: Iterable[tuple[str, Mapping[str, Any], str]],
    *,
    run_provenance: Mapping[str, Any],
    expected_circuit_sha256: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Aggregates per-job stats without mixing circuit identities or configurations."""

    provenance_fields = {
        "tesseract_commit": str,
        "stim_revision": str,
        "hardware_description": str,
        "tesseract_binary_sha256": str,
        "run_ids": list,
        "sample_seed_namespace_by_run": dict,
        "sample_seed_scheme": str,
        "sample_seed_stride": int,
    }
    _require_fields(run_provenance, provenance_fields, "run provenance")
    for field in ("tesseract_commit", "stim_revision"):
        if re.fullmatch(r"[0-9a-f]{40}", run_provenance[field]) is None:
            raise BenchmarkDataError(
                f"run provenance: {field} must be a full lowercase Git revision"
            )
    if re.fullmatch(r"[0-9a-f]{64}", run_provenance["tesseract_binary_sha256"]) is None:
        raise BenchmarkDataError(
            "run provenance: tesseract_binary_sha256 must be a lowercase SHA-256 digest"
        )
    if not run_provenance["hardware_description"].strip():
        raise BenchmarkDataError(
            "run provenance: hardware_description must not be empty"
        )
    run_ids = run_provenance["run_ids"]
    if not run_ids or any(
        not isinstance(run_id, str) or not run_id for run_id in run_ids
    ):
        raise BenchmarkDataError("run provenance: run_ids must be nonempty strings")
    if run_provenance["sample_seed_scheme"] != SUPPORTED_SAMPLE_SEED_SCHEME:
        raise BenchmarkDataError("run provenance: unsupported sample_seed_scheme")
    if run_provenance["sample_seed_stride"] <= 0:
        raise BenchmarkDataError("run provenance: sample_seed_stride must be positive")
    seed_namespaces = run_provenance["sample_seed_namespace_by_run"]
    if set(seed_namespaces) != set(run_ids) or any(
        not _matches_type(namespace, int) or namespace < 0
        for namespace in seed_namespaces.values()
    ):
        raise BenchmarkDataError(
            "run provenance: sample seed namespaces must cover all run IDs"
        )
    if len(set(seed_namespaces.values())) != len(seed_namespaces):
        raise BenchmarkDataError(
            "run provenance: sample seed namespaces must be unique"
        )

    known_run_ids = set(run_ids)
    materialized: list[tuple[str, dict[str, Any], str]] = []
    for source, original, run_id in rows:
        if run_id not in known_run_ids:
            raise BenchmarkDataError(
                f"{source}: row names unknown benchmark run {run_id!r}"
            )
        row = dict(original)
        _validate_raw_row(row, source)
        materialized.append((source, row, run_id))
    if not materialized:
        raise BenchmarkDataError("no raw benchmark rows found")

    for field in FIXED_SWEEP_FIELDS:
        values = {row[field] for _, row, _ in materialized}
        if len(values) != 1:
            raise BenchmarkDataError(
                f"raw benchmark invariant {field!r} has multiple values: {sorted(values)!r}"
            )

    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    seeds: dict[tuple[Any, ...], set[int]] = {}
    contributing_run_ids: dict[tuple[Any, ...], set[str]] = {}
    for source, row, run_id in materialized:
        key = tuple(row[field] for field in GROUP_FIELDS if field not in OUTCOME_FIELDS)
        if key not in groups:
            groups[key] = {
                field: row[field]
                for field in GROUP_FIELDS
                if field not in OUTCOME_FIELDS and field != "sample_seed"
            }
            groups[key].update({field: row[field] for field in RAW_MODEL_TYPES})
            groups[key].update({field: 0 for field in OUTCOME_FIELDS})
            groups[key]["num_jobs"] = 0
            seeds[key] = set()
            contributing_run_ids[key] = set()
        aggregate = groups[key]
        for field in RAW_MODEL_TYPES:
            if aggregate[field] != row[field]:
                raise BenchmarkDataError(
                    f"{_context(source, row)}: runtime model field {field!r} "
                    "changed within one aggregate configuration"
                )
        if row["sample_seed"] in seeds[key]:
            raise BenchmarkDataError(
                f"{_context(source, row)}: duplicate sample_seed "
                f"{row['sample_seed']} within one aggregate configuration"
            )
        for field in OUTCOME_FIELDS:
            aggregate[field] += row[field]
        aggregate["num_jobs"] += 1
        seeds[key].add(row["sample_seed"])
        contributing_run_ids[key].add(run_id)

    for key, aggregate in groups.items():
        digest_input = "\n".join(str(seed) for seed in sorted(seeds[key])).encode()
        aggregate["sample_seed_sha256"] = hashlib.sha256(digest_input).hexdigest()
        aggregate["benchmark_tesseract_commit"] = run_provenance["tesseract_commit"]
        aggregate["benchmark_stim_revision"] = run_provenance["stim_revision"]
        aggregate["benchmark_hardware"] = run_provenance["hardware_description"]
        aggregate["benchmark_tesseract_binary_sha256"] = run_provenance[
            "tesseract_binary_sha256"
        ]
        aggregate["benchmark_run_ids"] = sorted(contributing_run_ids[key])
        aggregate["benchmark_run_count"] = len(contributing_run_ids[key])
        aggregate["benchmark_sample_seed_scheme"] = run_provenance["sample_seed_scheme"]
        aggregate["benchmark_sample_seed_stride"] = run_provenance["sample_seed_stride"]
        aggregate["benchmark_sample_seed_namespace_by_run"] = {
            run_id: seed_namespaces[run_id]
            for run_id in sorted(contributing_run_ids[key])
        }
        aggregate["model_metadata_source"] = "snapshot_reconstruction"

        circuit_path = aggregate["circuit_path"]
        circuit_digest = expected_circuit_sha256.get(circuit_path)
        if circuit_digest is None:
            raise BenchmarkDataError(
                f"aggregate row [{circuit_path}]: circuit is absent from the run manifest"
            )
        aggregate.update(_parse_descriptor(circuit_path))
        aggregate["circuit_sha256"] = circuit_digest
        validate_aggregate_row(aggregate, f"aggregate row [{circuit_path}]")
    return list(groups.values())


def _validate_run_manifest(manifest: Mapping[str, Any], source: str) -> None:
    _require_fields(manifest, RUN_MANIFEST_TYPES, source)
    unknown = sorted(set(manifest) - set(RUN_MANIFEST_TYPES))
    if unknown:
        raise BenchmarkDataError(
            f"{source}: unknown manifest fields: {', '.join(unknown)}"
        )
    if manifest["schema_version"] != 1:
        raise BenchmarkDataError(f"{source}: unsupported manifest schema_version")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", manifest["run_id"]):
        raise BenchmarkDataError(f"{source}: invalid run_id")
    if not manifest["created_at_utc"]:
        raise BenchmarkDataError(f"{source}: created_at_utc must not be empty")
    for field in ("tesseract_commit", "stim_revision"):
        if re.fullmatch(r"[0-9a-f]{40}", manifest[field]) is None:
            raise BenchmarkDataError(
                f"{source}: {field} must be a full lowercase Git revision"
            )
    if re.fullmatch(r"[0-9a-f]{64}", manifest["tesseract_binary_sha256"]) is None:
        raise BenchmarkDataError(
            f"{source}: tesseract_binary_sha256 must be a lowercase SHA-256 digest"
        )
    if manifest["git_dirty"]:
        raise BenchmarkDataError(
            f"{source}: benchmark submission used a dirty checkout"
        )
    if manifest["det_order_method"] not in SUPPORTED_DET_ORDER_METHODS:
        raise BenchmarkDataError(f"{source}: unsupported det_order_method")
    if not manifest["merge_errors"]:
        raise BenchmarkDataError(
            f"{source}: snapshot metadata reconstruction requires merge_errors=true"
        )
    if not manifest["hardware_description"].strip():
        raise BenchmarkDataError(f"{source}: hardware_description must not be empty")
    if not manifest["circuit_sha256"]:
        raise BenchmarkDataError(f"{source}: circuit_sha256 must not be empty")
    for circuit_path, digest in manifest["circuit_sha256"].items():
        if not isinstance(circuit_path, str) or not circuit_path:
            raise BenchmarkDataError(
                f"{source}: circuit paths must be nonempty strings"
            )
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise BenchmarkDataError(
                f"{source}: invalid SHA-256 for circuit {circuit_path!r}"
            )

    sweep = manifest["sweep"]
    _require_fields(sweep, RUN_SWEEP_TYPES, f"{source}: sweep")
    unknown_sweep_fields = sorted(set(sweep) - set(RUN_SWEEP_TYPES))
    if unknown_sweep_fields:
        raise BenchmarkDataError(
            f"{source}: unknown sweep fields: {', '.join(unknown_sweep_fields)}"
        )
    repetitions = sweep["repetitions_per_configuration"]
    if repetitions <= 0:
        raise BenchmarkDataError(
            f"{source}: repetitions_per_configuration must be positive"
        )
    reactivate_limits = sweep["sparsify_reactivate_limits"]
    if (
        not reactivate_limits
        or any(
            not _matches_type(limit, int) or limit < -1 for limit in reactivate_limits
        )
        or reactivate_limits != sorted(set(reactivate_limits))
    ):
        raise BenchmarkDataError(
            f"{source}: sparsify_reactivate_limits must be unique, sorted, "
            "and contain only -1 or non-negative integers"
        )
    base_degrees = sweep["sparsify_base_degree_by_directory"]
    if not base_degrees:
        raise BenchmarkDataError(
            f"{source}: sparsify_base_degree_by_directory must not be empty"
        )
    for directory, degree in base_degrees.items():
        if (
            not isinstance(directory, str)
            or directory not in CODE_FAMILY_BY_DIRECTORY
            or not _matches_type(degree, int)
            or degree <= 0
        ):
            raise BenchmarkDataError(
                f"{source}: invalid sparsify base degree entry {directory!r}: "
                f"{degree!r}"
            )
    max_degree = sweep["sparsify_max_degree"]
    if max_degree != -1 and max_degree < max(base_degrees.values()):
        raise BenchmarkDataError(
            f"{source}: sparsify_max_degree must be -1 or at least every base degree"
        )
    for circuit_path in manifest["circuit_sha256"]:
        _sweep_base_degree(circuit_path, base_degrees, source)

    configurations_per_circuit = len(reactivate_limits) + int(sweep["include_baseline"])
    expected_job_count = (
        len(manifest["circuit_sha256"]) * configurations_per_circuit * repetitions
    )
    if manifest["expected_job_count"] != expected_job_count:
        raise BenchmarkDataError(
            f"{source}: expected_job_count is {manifest['expected_job_count']}, "
            f"but sweep shape requires {expected_job_count}"
        )
    if manifest["sample_seed_scheme"] != SUPPORTED_SAMPLE_SEED_SCHEME:
        raise BenchmarkDataError(f"{source}: unsupported sample_seed_scheme")
    seed_namespace = manifest["sample_seed_namespace"]
    seed_stride = manifest["sample_seed_stride"]
    if seed_namespace < 0:
        raise BenchmarkDataError(
            f"{source}: sample_seed_namespace must be non-negative"
        )
    if seed_stride < expected_job_count:
        raise BenchmarkDataError(
            f"{source}: sample_seed_stride must be at least expected_job_count"
        )
    maximum_seed = seed_namespace * seed_stride + expected_job_count - 1
    if maximum_seed > (1 << 64) - 1:
        raise BenchmarkDataError(f"{source}: sample seed range exceeds uint64")


def _sweep_base_degree(
    circuit_path: str, base_degrees: Mapping[str, Any], source: str
) -> int:
    directories = [
        directory
        for directory in CODE_FAMILY_BY_DIRECTORY
        if f"/{directory}/" in f"/{circuit_path}"
    ]
    if len(directories) != 1 or directories[0] not in base_degrees:
        raise BenchmarkDataError(
            f"{source}: circuit has no unique configured code directory: "
            f"{circuit_path!r}"
        )
    return base_degrees[directories[0]]


def _validate_run_coverage(
    manifest: Mapping[str, Any],
    rows: Sequence[tuple[str, Mapping[str, Any]]],
    source: str,
) -> None:
    """Requires a complete repetition grid before aggregation can proceed."""

    if len(rows) != manifest["expected_job_count"]:
        raise BenchmarkDataError(
            f"{source}: found {len(rows)} completed jobs; expected "
            f"{manifest['expected_job_count']}"
        )

    actual: Counter[tuple[Any, ...]] = Counter()
    circuit_hashes = manifest["circuit_sha256"]
    seed_base = manifest["sample_seed_namespace"] * manifest["sample_seed_stride"]
    for job_index, (row_source, row) in enumerate(rows):
        _validate_raw_row(row, row_source)
        expected_seed = seed_base + job_index
        if row["sample_seed"] != expected_seed:
            raise BenchmarkDataError(
                f"{_context(row_source, row)}: sample_seed is {row['sample_seed']}; "
                f"expected {expected_seed} from the manifest seed scheme"
            )
        circuit_path = row["circuit_path"]
        if circuit_path not in circuit_hashes:
            raise BenchmarkDataError(
                f"{_context(row_source, row)}: circuit is absent from the run manifest"
            )
        actual[
            (
                circuit_path,
                row["sparsify_errors"],
                row["sparsify_base_degree"],
                row["sparsify_max_degree"],
                row["sparsify_reactivate_limit"],
            )
        ] += 1

    sweep = manifest["sweep"]
    repetitions = sweep["repetitions_per_configuration"]
    expected: Counter[tuple[Any, ...]] = Counter()
    for circuit_path in circuit_hashes:
        if sweep["include_baseline"]:
            expected[(circuit_path, False, -1, -1, -1)] = repetitions
        base_degree = _sweep_base_degree(
            circuit_path, sweep["sparsify_base_degree_by_directory"], source
        )
        for reactivate_limit in sweep["sparsify_reactivate_limits"]:
            expected[
                (
                    circuit_path,
                    True,
                    base_degree,
                    sweep["sparsify_max_degree"],
                    reactivate_limit,
                )
            ] = repetitions

    if actual != expected:
        deficits = expected - actual
        excesses = actual - expected
        details = []
        if deficits:
            details.append(f"missing {sum(deficits.values())} jobs")
        if excesses:
            details.append(f"has {sum(excesses.values())} unexpected jobs")
        raise BenchmarkDataError(
            f"{source}: incomplete sweep grid: {', '.join(details)}"
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_run_artifacts(
    directory: Path, manifest: Mapping[str, Any], source: str
) -> None:
    artifact_directory = directory / "artifacts"
    binary_path = artifact_directory / "tesseract"
    if not binary_path.is_file() or not os.access(binary_path, os.X_OK):
        raise BenchmarkDataError(f"{source}: missing executable artifact {binary_path}")
    if binary_path.stat().st_mode & 0o222:
        raise BenchmarkDataError(f"{source}: binary artifact must be read-only")
    if _file_sha256(binary_path) != manifest["tesseract_binary_sha256"]:
        raise BenchmarkDataError(f"{source}: binary artifact SHA-256 mismatch")

    snapshot_root = (artifact_directory / "repo").resolve()
    if not snapshot_root.is_dir():
        raise BenchmarkDataError(f"{source}: missing circuit snapshot directory")
    for circuit_path, expected_digest in manifest["circuit_sha256"].items():
        snapshot_path = (snapshot_root / circuit_path).resolve()
        if (
            snapshot_root != snapshot_path
            and snapshot_root not in snapshot_path.parents
        ):
            raise BenchmarkDataError(
                f"{source}: snapshot circuit escapes artifact directory: {circuit_path!r}"
            )
        if not snapshot_path.is_file():
            raise BenchmarkDataError(
                f"{source}: missing snapshot circuit {circuit_path!r}"
            )
        if snapshot_path.stat().st_mode & 0o222:
            raise BenchmarkDataError(
                f"{source}: snapshot circuit must be read-only: {circuit_path!r}"
            )
        if _file_sha256(snapshot_path) != expected_digest:
            raise BenchmarkDataError(
                f"{source}: snapshot circuit SHA-256 mismatch: {circuit_path!r}"
            )


def read_run_directories(
    run_directories: Sequence[Path],
) -> tuple[list[tuple[str, dict[str, Any], str]], dict[str, Any], dict[str, str]]:
    """Reads manifest-scoped job directories and verifies compatible provenance."""

    if not run_directories:
        raise BenchmarkDataError("no benchmark run directories provided")
    manifests: list[dict[str, Any]] = []
    rows: list[tuple[str, dict[str, Any], str]] = []
    seen_directories: set[Path] = set()
    for original_directory in sorted(run_directories):
        directory = original_directory.resolve()
        if directory in seen_directories:
            raise BenchmarkDataError(f"duplicate run directory: {directory}")
        seen_directories.add(directory)
        manifest_path = directory / "manifest.json"
        manifest = _parse_json(
            manifest_path.read_text(encoding="utf-8"), str(manifest_path)
        )
        if not isinstance(manifest, dict):
            raise BenchmarkDataError(f"{manifest_path}: manifest must be an object")
        _validate_run_manifest(manifest, str(manifest_path))
        if manifest["run_id"] != directory.name:
            raise BenchmarkDataError(
                f"{manifest_path}: run_id must match its directory name"
            )
        _validate_run_artifacts(directory, manifest, str(manifest_path))
        job_paths = sorted((directory / "jobs").glob("*.json"))
        if not job_paths:
            raise BenchmarkDataError(f"{directory}: no jobs/*.json files found")
        if any(
            not path.stem.isdigit() or str(int(path.stem)) != path.stem
            for path in job_paths
        ):
            raise BenchmarkDataError(
                f"{directory}: job files must use canonical numeric names such as 0.json"
            )
        job_paths.sort(key=lambda path: int(path.stem))
        if len(job_paths) != manifest["expected_job_count"] or any(
            int(path.stem) != index for index, path in enumerate(job_paths)
        ):
            raise BenchmarkDataError(
                f"{directory}: jobs must be the complete contiguous range "
                f"0.json through {manifest['expected_job_count'] - 1}.json"
            )
        run_rows: list[tuple[str, dict[str, Any]]] = []
        for job_path in job_paths:
            job_rows = _read_raw_file(job_path)
            if len(job_rows) != 1:
                raise BenchmarkDataError(
                    f"{job_path}: each job file must contain exactly one JSON object"
                )
            run_rows.extend(job_rows)
        _attach_snapshot_metadata(
            run_rows,
            manifest=manifest,
            snapshot_root=directory / "artifacts" / "repo",
        )
        _validate_run_coverage(manifest, run_rows, str(directory))
        manifests.append(manifest)
        rows.extend((source, row, manifest["run_id"]) for source, row in run_rows)

    compatibility_fields = (
        "tesseract_commit",
        "stim_revision",
        "hardware_description",
        "tesseract_binary_sha256",
        "det_order_method",
        "merge_errors",
        "circuit_sha256",
        "expected_job_count",
        "sample_seed_scheme",
        "sample_seed_stride",
        "sweep",
    )
    first = manifests[0]
    for manifest in manifests[1:]:
        for field in compatibility_fields:
            if manifest[field] != first[field]:
                raise BenchmarkDataError(
                    f"run manifests disagree on provenance field {field!r}"
                )
    run_ids = sorted(manifest["run_id"] for manifest in manifests)
    if len(set(run_ids)) != len(run_ids):
        raise BenchmarkDataError("duplicate run_id across manifests")
    seed_namespaces = [manifest["sample_seed_namespace"] for manifest in manifests]
    if len(set(seed_namespaces)) != len(seed_namespaces):
        raise BenchmarkDataError("duplicate sample_seed_namespace across manifests")
    provenance = {
        field: first[field]
        for field in compatibility_fields
        if field != "circuit_sha256"
    }
    provenance["run_ids"] = run_ids
    provenance["sample_seed_namespace_by_run"] = {
        manifest["run_id"]: manifest["sample_seed_namespace"] for manifest in manifests
    }
    return rows, provenance, dict(first["circuit_sha256"])


def read_raw_files(paths: Sequence[Path]) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(paths):
        rows.extend(_read_raw_file(path))
    return rows


def _read_raw_file(path: Path) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    with path.open(encoding="utf-8") as infile:
        for line_number, text in enumerate(infile, start=1):
            if not text.strip():
                continue
            source = f"{path}:{line_number}"
            row = _parse_json(text, source)
            if not isinstance(row, dict):
                raise BenchmarkDataError(f"{source}: each JSON value must be an object")
            rows.append((source, row))
    return rows
