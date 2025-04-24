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

#ifndef SRC_TESSERACT_INNER_DECODER_H_
#define SRC_TESSERACT_INNER_DECODER_H_

#include "tesseract.h"
#include "stim.h"
#include "src/stim/io/sparse_shot.h"
#include "src/stim/dem/detector_error_model.h"

// Wrapper class for InnerDecoder template.
struct TesseractInnerDecoder {

  TesseractDecoder decoder;

  TesseractInnerDecoder(const stim::DetectorErrorModel& dem);

  double decode_to_weight(const std::vector<uint64_t>& detections);

};

#endif //SRC_TESSERACT_INNER_DECODER_H_
