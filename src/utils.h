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

#ifndef __TESSERACT_UTILS_H__
#define __TESSERACT_UTILS_H__

#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <random>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "common.h"
#include "stim.h"

constexpr const double EPSILON = 1e-7;

std::vector<std::vector<double>> get_detector_coords(const stim::DetectorErrorModel& dem);

// Builds an adjacency list graph where two detectors share an edge iff an error
// in the model activates them both.
std::vector<std::vector<size_t>> build_detector_graph(const stim::DetectorErrorModel& dem);

enum class DetOrder { DetBFS, DetIndex, DetCoordinate };

std::vector<std::vector<size_t>> build_det_orders(const stim::DetectorErrorModel& dem,
                                                  size_t num_det_orders,
                                                  DetOrder method = DetOrder::DetBFS,
                                                  uint64_t seed = 0);

const double INF = std::numeric_limits<double>::infinity();

bool sampling_from_dem(uint64_t seed, size_t num_shots, stim::DetectorErrorModel dem,
                       std::vector<stim::SparseShot>& shots);

void sample_shots(uint64_t sample_seed, stim::Circuit& circuit, size_t sample_num_shots,
                  std::vector<stim::SparseShot>& shots);

std::vector<common::Error> get_errors_from_dem(const stim::DetectorErrorModel& dem);

std::vector<std::string> get_files_recursive(const std::string& directory_path);

uint64_t vector_to_u64_mask(const std::vector<int>& v);

// Applies a shot-wise worker function in parallel while consuming completed
// shots in increasing order.
//
// process_shot(thread_index, shot_index):
//   - Runs on worker threads.
//   - thread_index is stable for each worker and lies in [0, num_threads).
//
// consume_shot(shot_index):
//   - Runs on the caller thread in increasing shot order.
//
// If consume_shot returns false, workers stop claiming new shots but always
// finish any shot they already started.
template <typename ProcessShot, typename ConsumeShot>
size_t parallel_for_shots_in_order(size_t num_shots, size_t num_threads, ProcessShot&& process_shot,
                                   ConsumeShot&& consume_shot) {
  std::atomic<size_t> next_unclaimed_shot = 0;
  std::vector<std::atomic<bool>> finished(num_shots);
  std::atomic<bool> worker_threads_please_terminate = false;
  std::atomic<size_t> num_worker_threads_active = 0;
  std::vector<std::thread> workers;
  workers.reserve(num_threads);

  for (size_t t = 0; t < num_threads; ++t) {
    ++num_worker_threads_active;
    workers.emplace_back([&, t]() {
      for (size_t shot;
           !worker_threads_please_terminate && ((shot = next_unclaimed_shot++) < num_shots);) {
        process_shot(t, shot);
        finished[shot] = true;
      }
      --num_worker_threads_active;
    });
  }

  size_t shot = 0;
  for (; shot < num_shots; ++shot) {
    while (num_worker_threads_active && !finished[shot]) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!finished[shot]) {
      assert(num_worker_threads_active == 0);
      break;
    }
    if (!consume_shot(shot)) {
      worker_threads_please_terminate = true;
    }
  }

  for (auto& worker : workers) {
    worker.join();
  }
  return shot;
}

#endif  // __TESSERACT_UTILS_H__
