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

import pytest
import stim

import tesseract_decoder

def get_set_bits(n):
    """
    Converts an observable bitmask (integer) into a list of observable indices.

    Args:
        n (int): The integer representing the observable bitmask.

    Returns:
        list[int]: A list containing the indices of the set bits (observable IDs)
    """
    bits = []
    i = 0

    while n > 0:
        if n & 1:
            bits.append(i)
        n >>= 1
        i += 1
    return bits


def test_as_dem_instruction_targets():
    s = tesseract_decoder.common.Symptom([1, 2], get_set_bits(4324))
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
