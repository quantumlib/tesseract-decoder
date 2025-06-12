import math
import pytest
import stim

from src import tesseract_decoder


_DETECTOR_ERROR_MODEL = stim.DetectorErrorModel(
    """
error(0.125) D0
error(0.375) D0 D1
error(0.25) D1
"""
)


def test_module_has_global_constants():
    assert tesseract_decoder.utils.EPSILON <= 1e-7
    assert not math.isfinite(tesseract_decoder.utils.INF)


def test_get_detector_coords():
    assert tesseract_decoder.utils.get_detector_coords(_DETECTOR_ERROR_MODEL) == []


def test_build_detector_graph():
    assert tesseract_decoder.utils.build_detector_graph(_DETECTOR_ERROR_MODEL) == [
        [1],
        [0],
    ]


def test_get_errors_from_dem():
    expected = "Error{cost=1.945910, symptom=Symptom{D0 }}, Error{cost=0.510826, symptom=Symptom{D0 D1 }}, Error{cost=1.098612, symptom=Symptom{D1 }}"
    assert (
        ", ".join(
            map(str, tesseract_decoder.utils.get_errors_from_dem(_DETECTOR_ERROR_MODEL))
        )
        == expected
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
