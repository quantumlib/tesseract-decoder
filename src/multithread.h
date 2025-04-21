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

#ifndef SRC_MULTITHREAD_H_
#define SRC_MULTITHREAD_H_

#include <atomic>
#include <thread>

#include "stim.h"
#include "tesseract.h"

namespace multithread {

// Decodes using tesseract with multiple threads parallelizing over shots.
void decode_multithreaded(const size_t& num_threads,
                          TesseractConfig& config,
                          std::vector<stim::SparseShot>& shots,
                          std::atomic<size_t>& next_unclaimed_shot,
                          std::vector<std::atomic<bool>>& finished,
                          std::vector<common::ObservablesMask>& obs_predicted,
                          std::vector<double>& cost_predicted,
                          std::vector<double>& decoding_time_seconds,
                          std::vector<std::atomic<bool>>& low_confidence,
                          std::vector<std::thread>& decoder_threads,
                          std::vector<std::atomic<size_t>>& error_use_totals,
                          const bool& has_obs,
                          std::atomic<bool>& worker_threads_please_terminate,
                          std::atomic<size_t>& num_worker_threads_active);

} // namespace multithread

#endif //SRC_MULTITHREAD_H_
