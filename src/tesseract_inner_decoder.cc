// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "common.h"
#include "stim.h"
#include "tesseract_inner_decoder.h"

// Helper function that builds a default TesseractConfig from a DEM.
TesseractConfig tesseract_config_from_dem(const stim::DetectorErrorModel& dem) {
  TesseractConfig config;
  config.dem = common::merge_identical_errors(dem);
  return config;
}

TesseractInnerDecoder::TesseractInnerDecoder(
    const stim::DetectorErrorModel& dem) :
    decoder(tesseract_config_from_dem(dem)) {}

double TesseractInnerDecoder::decode_to_weight(const std::vector<uint64_t>& detections) {
  decoder.decode_to_errors(detections);
  return decoder.cost_from_errors(decoder.predicted_errors_buffer);
}


