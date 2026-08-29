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

import numpy as np
import pytest
import sinter
import stim
import tesseract_decoder
from sinter._decoding._stim_then_decode_sampler import StimThenDecodeSampler

from multi_pass_sinter_decoders import MultiPassSinterDecoder, get_sinter_decoders


def _two_basis_dem() -> stim.DetectorErrorModel:
    return stim.DetectorErrorModel(
        r"""
        error(0.1) D0 L0
        error(0.2) D1 L0
        detector[{"measure_basis":"X"}] D0
        detector[{"md":{"basis":"Z"}}] D1
        """
    )


def test_zero_configuration_and_stable_registry_workflows():
    assert not hasattr(tesseract_decoder, "MultiPassSinterDecoder")
    assert hasattr(tesseract_decoder, "_compile_multi_pass_decoder_for_dem")
    decoder = MultiPassSinterDecoder()
    assert decoder.num_passes == 2
    assert decoder.strategy == tesseract_decoder.SchedulingStrategy.Causal
    assert decoder.compile_decoder_for_dem(dem=_two_basis_dem()).num_components == 2

    decoders = get_sinter_decoders()
    assert set(decoders) == {
        "tesseract-long-beam-mono",
        "tesseract-long-beam-multipass-1pass",
        "tesseract-long-beam-multipass-2pass",
    }
    assert decoders["tesseract-long-beam-multipass-1pass"].num_passes == 1
    assert decoders["tesseract-long-beam-multipass-2pass"].num_passes == 2
    for name, configured_decoder in decoders.items():
        if "multipass" in name:
            assert configured_decoder.det_beam == 20
            assert configured_decoder.det_order_method == (
                tesseract_decoder.utils.DetOrder.DetIndex
            )


def test_zero_configuration_uses_chromobius_fourth_coordinate_adapter():
    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0 L0
        error(0.2) D1 L0
        detector(0, 0, 0, 0) D0
        detector(0, 0, 0, 3) D1
        """
    )
    MultiPassSinterDecoder().compile_decoder_for_dem(dem=dem)


@pytest.mark.parametrize(
    ("detector_zero", "message"),
    [
        (
            'detector[{"measure_basis":0,"basis":"X"}](0,0,0,0) D0',
            "measure_basis",
        ),
        ("detector(0,0,0,0.5) D0", "nonintegral fourth coordinate"),
    ],
)
def test_zero_configuration_keeps_automatic_classifier_strict(detector_zero, message):
    dem = stim.DetectorErrorModel(
        "\n".join(
            [
                "error(0.1) D0 L0",
                "error(0.2) D1 L0",
                detector_zero,
                "detector(0,0,0,3) D1",
            ]
        )
    )
    with pytest.raises(ValueError, match=message):
        MultiPassSinterDecoder().compile_decoder_for_dem(dem=dem)


def test_custom_detector_basis_classifier_is_resolved_once_in_python():
    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0 L0
        error(0.2) D1 L0
        detector D0
        detector D1
        """
    )
    calls = []

    def classifier(index, coordinates, tag):
        calls.append((index, tuple(coordinates), tag))
        return "X" if index == 0 else "Z"

    decoder = MultiPassSinterDecoder(detector_basis_classifier=classifier)
    decoder.compile_decoder_for_dem(dem=dem)
    assert [call[0] for call in calls] == [0, 1]


def test_detector_classifier_compatibility_alias_and_conflict():
    def classifier(index, _coordinates, _tag):
        return 4 if index == 0 else 9

    decoder = MultiPassSinterDecoder(detector_classifier=classifier)
    assert decoder.detector_classifier is classifier
    decoder.compile_decoder_for_dem(dem=_two_basis_dem())

    with pytest.raises(ValueError, match="at most one"):
        MultiPassSinterDecoder(
            detector_basis_classifier=classifier,
            detector_classifier=classifier,
        )


def test_all_standard_configuration_keywords_and_unknown_keyword_validation():
    decoder = MultiPassSinterDecoder(
        det_beam=17,
        beam_climbing=True,
        no_revisit_dets=False,
        verbose=True,
        merge_errors=False,
        pqlimit=1234,
        det_penalty=0.25,
        create_visualization=False,
        sparsify_errors=True,
        sparsify_base_degree=2,
        sparsify_max_degree=4,
        sparsify_reactivate_limit=7,
        det_orders=[[1, 0]],
        num_det_orders=3,
        det_order_method=tesseract_decoder.utils.DetOrder.DetCoordinate,
        seed=123,
    )
    assert decoder.det_beam == 17
    assert decoder.merge_errors is False
    assert decoder.sparsify_reactivate_limit == 7
    assert decoder.det_orders == [[1, 0]]
    assert decoder.num_det_orders == 3
    decoder.compile_decoder_for_dem(dem=_two_basis_dem())

    with pytest.raises(TypeError, match="unexpected keyword"):
        MultiPassSinterDecoder(typoed_option=True)


@pytest.mark.parametrize("num_passes", [0, 3])
def test_invalid_pass_count_is_rejected(num_passes):
    with pytest.raises(ValueError, match="1 or 2"):
        MultiPassSinterDecoder(num_passes=num_passes)


def test_invalid_strategy_is_rejected():
    with pytest.raises(ValueError, match="strategy"):
        MultiPassSinterDecoder(strategy="causal")


def _two_byte_dem() -> stim.DetectorErrorModel:
    lines = ["error(0.1) D0 L0", "error(0.1) D8 L1"]
    for detector in range(9):
        basis = "Z" if detector == 8 else "X"
        lines.append(f'detector[{{"basis":"{basis}"}}] D{detector}')
    return stim.DetectorErrorModel("\n".join(lines))


@pytest.mark.parametrize("layout", ["row_stride", "column_stride", "fortran"])
def test_decode_shots_bit_packed_honors_both_numpy_strides(layout):
    packed = np.array(
        [
            [0b00000001, 0],
            [0, 0b00000001],
            [0b00000001, 0b00000001],
        ],
        dtype=np.uint8,
    )
    if layout == "row_stride":
        storage = np.zeros((6, 2), dtype=np.uint8)
        detections = storage[::2]
        detections[:] = packed
    elif layout == "column_stride":
        storage = np.zeros((3, 4), dtype=np.uint8)
        detections = storage[:, ::2]
        detections[:] = packed
    else:
        detections = np.asfortranarray(packed)
    assert not detections.flags.c_contiguous

    compiled = MultiPassSinterDecoder().compile_decoder_for_dem(dem=_two_byte_dem())
    predictions = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=detections
    )

    assert np.array_equal(
        predictions,
        np.array([[0b01, 0], [0b10, 0], [0b11, 0]], dtype=np.uint8),
    )


def _low_confidence_dem() -> stim.DetectorErrorModel:
    return stim.DetectorErrorModel(
        r"""
        error(0.1) D0 L0
        error(0.1) D1 L0
        detector[{"basis":"X"}] D0
        detector[{"basis":"Z"}] D1
        detector[{"basis":"Z"}] D2
        """
    )


def test_decode_shots_bit_packed_appends_sinter_discard_byte():
    compiled = MultiPassSinterDecoder().compile_decoder_for_dem(
        dem=_low_confidence_dem()
    )
    predictions = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=np.array([[0b001], [0b100]], dtype=np.uint8)
    )
    assert np.array_equal(
        predictions,
        np.array([[0b1, 0], [0, 1]], dtype=np.uint8),
    )


def test_real_sinter_sampler_counts_low_confidence_shots_as_discards():
    circuit = stim.Circuit(
        """
        R 0 1 2
        X_ERROR(1) 2
        M 0 1 2
        DETECTOR rec[-3]
        DETECTOR rec[-2]
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-2]
        """
    )
    sampler = StimThenDecodeSampler(
        decoder=MultiPassSinterDecoder(),
        count_observable_error_combos=False,
        count_detection_events=False,
        tmp_dir=None,
    ).compiled_sampler_for_task(
        sinter.Task(circuit=circuit, detector_error_model=_low_confidence_dem())
    )

    result = sampler.sample(5)
    assert result.shots == 5
    assert result.discards == 5
    assert result.errors == 0
