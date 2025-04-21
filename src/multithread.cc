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

#include "multithread.h"

void multithread::decode_multithreaded(const size_t& num_threads,
                          const bool& print_stats,
                          const size_t& max_errors,
                          TesseractConfig& config,
                          std::vector<stim::SparseShot>& shots,
                          std::unique_ptr<stim::MeasureRecordWriter>& writer,
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
                          std::atomic<size_t>& num_worker_threads_active,
                          size_t& num_errors,
                          size_t& num_low_confidence,
                          double& total_time_seconds) {
  for (size_t t = 0; t < num_threads; ++t) {
    // After this value returns to 0, we know that no further shots will
    // transition to finished.
    ++num_worker_threads_active;
    decoder_threads.push_back(std::thread(
        [&config, &next_unclaimed_shot, &shots, &obs_predicted, &cost_predicted,
         &decoding_time_seconds, &low_confidence, &finished, &error_use_totals,
         &has_obs, &worker_threads_please_terminate,
         &num_worker_threads_active]() {
          TesseractDecoder decoder(config);
          std::vector<size_t> error_use(config.dem.count_errors());
          for (size_t shot; !worker_threads_please_terminate and
                            ((shot = next_unclaimed_shot++) < shots.size());) {
            auto start_time = std::chrono::high_resolution_clock::now();
            decoder.decode_to_errors(shots[shot].hits);
            auto stop_time = std::chrono::high_resolution_clock::now();
            decoding_time_seconds[shot] =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    stop_time - start_time)
                    .count() /
                1e6;
            obs_predicted[shot] =
                decoder.mask_from_errors(decoder.predicted_errors_buffer);
            low_confidence[shot] = decoder.low_confidence_flag;
            cost_predicted[shot] =
                decoder.cost_from_errors(decoder.predicted_errors_buffer);
            if (!has_obs or
                shots[shot].obs_mask_as_u64() == obs_predicted[shot]) {
              // Only count the error uses for shots that did not have a logical
              // error, if we know the obs flips.
              for (size_t ei : decoder.predicted_errors_buffer) {
                ++error_use[ei];
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

  size_t num_observables = config.dem.count_observables();
  size_t shot = 0;
  for (; shot < shots.size(); ++shot) {
    while (num_worker_threads_active and !finished[shot]) {
      // We break once the number of active worker threads is 0, at which point
      // there will be no further changes to finished[shot].
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    // There can be no further changes to finished[shot]. If it is true, we
    // process it and go to the next shot. If it is false, we break now as it
    // will never be decoded and no subsequent shots will be decoded.
    if (!finished[shot]) {
      assert(num_worker_threads_active == 0);
      // This and subsequent shots will never become decoded.
      break;
    }

    if (writer) {
      writer->write_bits((uint8_t*)&obs_predicted[shot], num_observables);
      writer->write_end();
    }

    if (low_confidence[shot]) {
      ++num_low_confidence;
    } else if (obs_predicted[shot] != shots[shot].obs_mask_as_u64()) {
      ++num_errors;
    }

    total_time_seconds += decoding_time_seconds[shot];

    if (print_stats) {
      std::cout << "num_shots = " << (shot + 1)
                << " num_low_confidence = " << num_low_confidence
                << " num_errors = " << num_errors
                << " total_time_seconds = " << total_time_seconds << std::endl;
      std::cout << "cost = " << cost_predicted[shot] << std::endl;
      std::cout.flush();
    }

    if (num_errors >= max_errors) {
      worker_threads_please_terminate = true;
    }
  }
  for (size_t t = 0; t < num_threads; ++t) {
    decoder_threads[t].join();
  }
}