#Copyright 2025 Google LLC
#
#Licensed under the Apache License, Version 2.0(the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http : #www.apache.org / licenses / LICENSE - 2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import math import pytest import stim

    from src import tesseract_decoder

    _DETECTOR_ERROR_MODEL = stim.DetectorErrorModel(""
                                                    "
                                                    error(0.125) D0 error(0.375) D0 D1 error(0.25) D1 ""
                                                                                                      "
                                                    )

        def test_create_simplex_config() :sc = tesseract_decoder.simplex.SimplexConfig(_DETECTOR_ERROR_MODEL, window_length = 5) assert sc.dem == _DETECTOR_ERROR_MODEL assert sc.window_length == 5 assert(str(sc) == "SimplexConfig(dem=DetectorErrorModel_Object, window_length=5, window_slide_length=0, verbose=0)")

                                                                                                                  def test_create_simplex_decoder() :decoder = tesseract_decoder.simplex.SimplexDecoder(tesseract_decoder.simplex.SimplexConfig(_DETECTOR_ERROR_MODEL, window_length = 5)) decoder.init_ilp() decoder.decode_to_errors([1]) assert decoder.get_observables_from_errors([1]) ==[] assert decoder.cost_from_errors([2]) == pytest.approx(1.0986123) assert decoder.decode([1]) ==[]

                                                                                                                                                                                                                                                                                                                                                                                                                                                       def test_simplex_decoder_predicts_various_observable_flips() : ""
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      "
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  Tests that the Simplex decoder correctly predicts a logical observable flip when a specific detector is triggered by an error that explicitly flips that logical observable.

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  This test iterates through various observable IDs to ensure the backend logic correctly handles different positions.""
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      "
#Iterate through observable IDs from 0 to 63(inclusive)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  for observable_id in range(64) :
#Create a simple DetectorErrorModel where an error on D0 also flips L{observable_id }
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  dem_string = f''' error(0.01) D0 L{observable_id }
        ''' dem = stim.DetectorErrorModel(dem_string)

#Initialize SimplexConfig and SimplexDecoder with the generated DEM
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      config = tesseract_decoder.simplex.SimplexConfig(dem, window_length = 1) #window_length must be set decoder = tesseract_decoder.simplex.SimplexDecoder(config) decoder.init_ilp() #Initialize the ILP solver

#Simulate a detection event on D0.
#The decoder should identify the most likely error causing D0,
#which in this DEM is the error that also flips L{observable_id }.
#The decode method is expected to return an array where array[i] is True if observable i is flipped.
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      predicted_logical_flips_array = decoder.decode(detections =[0])

#Convert the boolean array / list to a list of flipped observable IDs
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          actual_flipped_observables =[idx for idx, is_flipped in enumerate(predicted_logical_flips_array) if is_flipped]

#Assert that the list of actual flipped observables matches the single expected observable_id.
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       assert actual_flipped_observables ==[observable_id], (f "For observable L{observable_id}: " f "Expected predicted logical flips: [{observable_id}], " f "but got: {actual_flipped_observables} (from raw: {predicted_logical_flips_array})")

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                if __name__ == "__main__" :raise SystemExit(pytest.main([__file__]))
