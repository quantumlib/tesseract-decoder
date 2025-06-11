import pytest
import stim

# import tesseract_decoder
from src import tesseract_decoder


def test_as_dem_instruction_targets():
    s = tesseract_decoder.common.Symptom([1, 2], 4324)
    dits = s.as_dem_instruction_targets()
    assert dits == [
        stim.DemTarget("D1"),
        stim.DemTarget("D2"),
        stim.DemTarget("L2"),
        stim.DemTarget("L5"),
        stim.DemTarget("L6"),
        stim.DemTarget("L7"),
        stim.DemTarget("L12"),
    ]


def test_error_from_dem_instruction():
    di = stim.DemInstruction("error", [0.125], [stim.target_logical_observable_id(3)])
    error = tesseract_decoder.common.Error(di)

    assert str(error) == "Error{cost=1.945910, symptom=Symptom{}}"


def test_merge_identical_errors():
    dem = stim.DetectorErrorModel()
    assert isinstance(
        tesseract_decoder.common.merge_identical_errors(dem), stim.DetectorErrorModel
    )


def test_remove_zero_probability_errors():
    dem = stim.DetectorErrorModel()
    assert isinstance(
        tesseract_decoder.common.remove_zero_probability_errors(dem),
        stim.DetectorErrorModel,
    )


def test_dem_from_counts():
    dem = stim.DetectorErrorModel()
    assert isinstance(
        tesseract_decoder.common.dem_from_counts(dem, [], 3), stim.DetectorErrorModel
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
