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

#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

#include "stim.h"
#include "tesseract.h"
#include "utils.h"

int main(int argc, const char** argv) {
  TesseractArgs args(argc, argv);

  TesseractConfig config;
  std::vector<stim::SparseShot> shots;
  std::unique_ptr<stim::MeasureRecordWriter> writer;
  args.extract(config, shots, writer);
  stim::DetectorErrorModel original_dem = config.dem.flattened();

  std::atomic<size_t> next_unclaimed_shot;
  std::vector<std::atomic<bool>> finished(shots.size());
  std::vector<uint64_t> obs_predicted(shots.size());
  std::vector<double> cost_predicted(shots.size());
  std::vector<double> decoding_time_seconds(shots.size());
  std::vector<std::atomic<bool>> low_confidence(shots.size());
  std::vector<std::thread> decoder_threads;
  std::vector<std::atomic<size_t>> error_use_totals(original_dem.count_errors());
  bool has_obs = args.has_observables();
  std::atomic<bool> worker_threads_please_terminate = false;
  std::atomic<size_t> num_worker_threads_active;
  for (size_t t = 0; t < args.num_threads; ++t) {
    // After this value returns to 0, we know that no further shots will
    // transition to finished.
    ++num_worker_threads_active;
    decoder_threads.push_back(std::thread([&config, &next_unclaimed_shot, &shots, &obs_predicted,
                                           &cost_predicted, &decoding_time_seconds, &low_confidence,
                                           &finished, &error_use_totals, &has_obs,
                                           &worker_threads_please_terminate,
                                           &num_worker_threads_active]() {
      TesseractDecoder decoder(config);
      std::vector<size_t> error_use(decoder.original_dem.count_errors());
      for (size_t shot;
           !worker_threads_please_terminate and ((shot = next_unclaimed_shot++) < shots.size());) {
        auto start_time = std::chrono::high_resolution_clock::now();
        std::vector<int> flipped_obs = decoder.decode(shots[shot].hits);
        auto stop_time = std::chrono::high_resolution_clock::now();
        decoding_time_seconds[shot] =
            std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time).count() /
            1e6;
        obs_predicted[shot] = vector_to_u64_mask(flipped_obs);
        low_confidence[shot] = decoder.low_confidence_flag;
        cost_predicted[shot] = 0;
        if (!has_obs or shots[shot].obs_mask_as_u64() == obs_predicted[shot]) {
          for (size_t ei : decoder.predicted_errors_buffer) {
            ++error_use[decoder.get_original_error_index(ei)];
          }
        }
        finished[shot] = true;
      }
      // Add the error counts to the total
      for (size_t ei = 0; ei < error_use_totals.size(); ++ei) {
        error_use_totals[ei] += error_use[ei];
      }
      --num_worker_threads_active;
    }));
  }

  size_t num_errors = 0;
  size_t num_low_confidence = 0;
  size_t shot = 0;
  auto start_time = std::chrono::high_resolution_clock::now();
  for (; shot < shots.size(); ++shot) {
    while (!finished[shot]) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (has_obs) {
      if (shots[shot].obs_mask_as_u64() != obs_predicted[shot]) {
        num_errors++;
      }
    }
    if (low_confidence[shot]) {
      num_low_confidence++;
    }
    if (writer != nullptr) {
      writer->begin_shot();
      for (size_t i = 0; i < original_dem.count_observables(); ++i) {
        writer->write_bit((obs_predicted[shot] >> i) & 1);
      }
      writer->end_shot();
    }
  }
  auto stop_time = std::chrono::high_resolution_clock::now();
  double total_time_seconds =
      std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time).count() / 1e6;

  worker_threads_please_terminate = true;
  for (std::thread& t : decoder_threads) {
    t.join();
  }

  if (!args.dem_out_fname.empty()) {
    std::vector<size_t> counts(error_use_totals.begin(), error_use_totals.end());
    size_t num_usage_dem_shots = shot;
    if (has_obs) {
      // When we know the obs, we only count non-error shots.
      num_usage_dem_shots -= num_errors;
    }
    stim::DetectorErrorModel est_dem =
        common::dem_from_counts(original_dem, counts, num_usage_dem_shots);
    std::ofstream out(args.dem_out_fname, std::ofstream::out);
    if (!out.is_open()) {
      throw std::invalid_argument("Failed to open " + args.dem_out_fname);
    }
    out << est_dem << '\n';
  }

  if (args.print_final_stats) {
    std::cout << "num_shots = " << shot;
    std::cout << " num_low_confidence = " << num_low_confidence;
    if (has_obs) {
      std::cout << " num_errors = " << num_errors;
    }
    std::cout << " total_time_seconds = " << total_time_seconds;
    std::cout << std::endl;
  }
}
