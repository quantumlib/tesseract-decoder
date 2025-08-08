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

#include "simplex.h"

#include <cassert>

#include "Highs.h"
#include "io/HMPSIO.h"

constexpr size_t T_COORD = 2;

std::string SimplexConfig::str() {
  auto& self = *this;
  std::stringstream ss;
  ss << "SimplexConfig(";
  ss << "dem=" << "DetectorErrorModel_Object" << ", ";
  ss << "window_length=" << self.window_length << ", ";
  ss << "window_slide_length=" << self.window_slide_length << ", ";
  ss << "verbose=" << self.verbose << ")";
  return ss.str();
}

SimplexDecoder::SimplexDecoder(SimplexConfig _config) : config(_config) {
  config.dem = common::remove_zero_probability_errors(config.dem);
  std::vector<double> detector_t_coords(config.dem.count_detectors());
  for (const stim::DemInstruction& instruction : config.dem.flattened().instructions) {
    switch (instruction.type) {
      case stim::DemInstructionType::DEM_SHIFT_DETECTORS:
        assert(false && "unreachable");
        break;
      case stim::DemInstructionType::DEM_ERROR: {
        assert(instruction.arg_data[0] > 0);
        errors.emplace_back(instruction);
        break;
      }
      case stim::DemInstructionType::DEM_DETECTOR:
        detector_t_coords[instruction.target_data[0].val()] = instruction.arg_data[T_COORD];
        break;
      default:
        assert(false && "unreachable");
    }
  }
  std::map<double, std::vector<size_t>> start_time_to_errors_map, end_time_to_errors_map;
  std::set<double> times;
  for (size_t ei = 0; ei < errors.size(); ++ei) {
    double min_error_time = std::numeric_limits<double>::max();
    double max_error_time = -std::numeric_limits<double>::max();
    for (int d : errors[ei].symptom.detectors) {
      double time = detector_t_coords[d];
      min_error_time = std::min(min_error_time, time);
      max_error_time = std::max(max_error_time, time);
      times.insert(time);
    }
    start_time_to_errors_map[min_error_time].push_back(ei);
    end_time_to_errors_map[max_error_time].push_back(ei);
  }
  start_time_to_errors.resize(times.size());
  end_time_to_errors.resize(times.size());
  size_t t = 0;
  for (const double& time : times) {
    start_time_to_errors[t] = start_time_to_errors_map[time];
    end_time_to_errors[t] = end_time_to_errors_map[time];
    ++t;
  }
  num_detectors = config.dem.count_detectors();
  num_observables = config.dem.count_observables();
  init_ilp();
}

void SimplexDecoder::init_ilp() {
  model = std::make_unique<HighsModel>();

  // There is one variable for each error and one slack variable for each
  // detector
  size_t num_vars = errors.size() + num_detectors;

  // Set up objective function: minimize total likelihood cost
  model->lp_.num_col_ = num_vars;
  model->lp_.sense_ = ObjSense::kMinimize;
  model->lp_.col_cost_.resize(num_vars, 0.0);

  for (size_t ei = 0; ei < errors.size(); ++ei) {
    model->lp_.col_cost_[ei] = errors[ei].likelihood_cost;
  }

  // Set up variable bounds
  model->lp_.col_lower_.resize(num_vars, 0.0);
  model->lp_.col_upper_.resize(num_vars, 1.0);  // Error variables are binary
  // Slack variables are arbitrary integers. But for numerical stability, we
  // constrain them to be in the range -num_errors, num_errors, which is a safe
  // upper bound on how big they need to get.
  for (size_t d = errors.size(); d < num_vars; ++d) {
    model->lp_.col_lower_[d] = -double(errors.size());
    model->lp_.col_upper_[d] = double(errors.size());
  }

  // There is one parity constraint for each detector
  model->lp_.num_row_ = num_detectors;

  // Sparse constraint matrix
  model->lp_.a_matrix_.format_ = MatrixFormat::kColwise;
  model->lp_.a_matrix_.start_.resize(num_vars + 1);
  std::vector<int> index;
  std::vector<double> value;

  for (size_t ei = 0; ei < errors.size(); ++ei) {
    for (size_t detector : errors[ei].symptom.detectors) {
      index.push_back(detector);
      value.push_back(1.0);
    }
    model->lp_.a_matrix_.start_[ei + 1] = index.size();
  }

  for (size_t d = 0; d < num_detectors; ++d) {
    index.push_back(d);
    value.push_back(2.0);
    model->lp_.a_matrix_.start_[errors.size() + d + 1] = index.size();
  }

  model->lp_.a_matrix_.index_ = std::move(index);
  model->lp_.a_matrix_.value_ = std::move(value);

  // Set integrality of error variables
  model->lp_.integrality_.resize(num_vars, HighsVarType::kInteger);

  // Constraint bounds
  model->lp_.row_lower_.resize(num_detectors, 0);
  model->lp_.row_upper_.resize(num_detectors, 0);

  // Set HiGHS options
  highs = std::make_unique<Highs>();
  return_status = std::make_unique<HighsStatus>();
  if (config.parallelize) {
    highs->setOptionValue("parallel", "choose");
    highs->setOptionValue("threads", 0);
  } else {
    highs->setOptionValue("parallel", "off");
    highs->setOptionValue("threads", 1);
  }
  // Disabled presolve entirely after encountering bugs similar to this one:
  // https://github.com/ERGO-Code/HiGHS/issues/1273
  highs->setOptionValue("presolve", "off");
  highs->setOptionValue("output_flag", config.verbose);
}

void SimplexDecoder::decode_to_errors(const std::vector<uint64_t>& detections) {
  predicted_errors_buffer.clear();
  // Adjust the constraints for the detection events
  for (size_t d : detections) {
    assert(d < num_detectors && "invalid detector");
    model->lp_.row_lower_[d] = 1;
    model->lp_.row_upper_[d] = 1;
  }

  if (config.windowing_enabled()) {
    std::set<size_t> set_detections(detections.begin(), detections.end());

    // Set all errors to have zero cost
    for (size_t ei = 0; ei < errors.size(); ++ei) {
      model->lp_.col_cost_[ei] = 0;
    }

    auto add_costs_for_time = [&](size_t t) -> void {
      // Update the cost of the errors
      for (size_t ei : start_time_to_errors.at(t)) {
        model->lp_.col_cost_[ei] = errors[ei].likelihood_cost;
      }
    };
    // Set the errors in the first window_length time slices to have their true
    // cost, and constrain the detectors to match the shot detections.
    size_t t1 = 0;
    for (t1 = 0; (t1 + config.window_slide_length < config.window_length) &&
                 (t1 < start_time_to_errors.size());
         ++t1) {
      add_costs_for_time(t1);
    }
    // All error slices strictly below t0 have been frozen
    size_t t0 = 0;
    HighsSolution solution;
    bool solution_empty = true;

    while (t1 < start_time_to_errors.size() or solution_empty) {
      for (size_t step = 0; step < config.window_slide_length && t1 < start_time_to_errors.size();
           ++step) {
        add_costs_for_time(t1);
        ++t1;
      }
      if (config.verbose) {
        std::cout << "t0 = " << t0 << " t1 = " << t1 << std::endl;
      }

      // Pass the model to HiGHS
      *return_status = highs->passModel(*model);
      if (*return_status != HighsStatus::kOk) {
        std::cerr << "Error: passModel failed with status: " << highsStatusToString(*return_status)
                  << std::endl;
      }
      assert(*return_status == HighsStatus::kOk);

      // Set the feasible solution, if one is known
      if (!solution_empty) {
        *return_status = highs->setSolution(solution);
        if (*return_status != HighsStatus::kOk) {
          std::cerr << "Error: setSolution failed with status: "
                    << highsStatusToString(*return_status) << std::endl;
        }
        assert(*return_status == HighsStatus::kOk);
      }

      // Solve the model
      *return_status = highs->run();
      if (*return_status != HighsStatus::kOk) {
        std::cerr << "Error: run failed with status: " << highsStatusToString(*return_status)
                  << std::endl;
        // Write out the model in mps format for debugging
        HighsStatus write_return_status =
            writeModelAsMps(highs->getOptions(), "bad_shot.mps", *model,
                            /*free_format=*/true);
        std::cerr << "Write return had status: " << highsStatusToString(write_return_status)
                  << std::endl;
        assert(write_return_status == HighsStatus::kOk or
               write_return_status == HighsStatus::kWarning);
      }
      assert(*return_status == HighsStatus::kOk);

      if (config.verbose) {
        // Get the solution information
        const HighsInfo& info = highs->getInfo();
        std::cout << "Simplex iteration count: " << info.simplex_iteration_count << std::endl;
        std::cout << "Objective function value: " << info.objective_function_value << std::endl;
        std::cout << "Primal  solution status: "
                  << highs->solutionStatusToString(info.primal_solution_status) << std::endl;
        std::cout << "Dual    solution status: "
                  << highs->solutionStatusToString(info.dual_solution_status) << std::endl;
        std::cout << "Basis: " << highs->basisValidityToString(info.basis_validity) << std::endl;
      }

      // Get the model status
      const HighsModelStatus& model_status = highs->getModelStatus();
      if (model_status != HighsModelStatus::kOptimal) {
        std::cerr << "Error: Model did not reach an optimal solution. Status: "
                  << highs->modelStatusToString(model_status) << std::endl;
      }
      assert(model_status == HighsModelStatus::kOptimal);

      // Extract the used errors
      solution = highs->getSolution();
      assert(!solution.hasUndefined());
      solution_empty = false;

      for (size_t step = 0; step < config.window_slide_length && t0 < end_time_to_errors.size();
           ++step) {
        // Freeze all errors at time slice t0 to their current values, and
        // increment t0
        for (size_t ei : end_time_to_errors.at(t0++)) {
          model->lp_.col_lower_[ei] = solution.col_value.at(ei);
          model->lp_.col_upper_[ei] = solution.col_value.at(ei);
        }
      }
    }

    // Reset bounds and cost for all error variables
    for (size_t ei = 0; ei < errors.size(); ++ei) {
      model->lp_.col_lower_[ei] = 0.0;
      model->lp_.col_upper_[ei] = 1.0;
      model->lp_.col_cost_[ei] = errors[ei].likelihood_cost;
    }
  } else {
    // Pass the model to HiGHS
    *return_status = highs->passModel(*model);
    assert(*return_status == HighsStatus::kOk);

    // Solve the model
    *return_status = highs->run();
    assert(*return_status == HighsStatus::kOk);

    if (config.verbose) {
      // Get the solution information
      const HighsInfo& info = highs->getInfo();
      std::cout << "Simplex iteration count: " << info.simplex_iteration_count << std::endl;
      std::cout << "Objective function value: " << info.objective_function_value << std::endl;
      std::cout << "Primal  solution status: "
                << highs->solutionStatusToString(info.primal_solution_status) << std::endl;
      std::cout << "Dual    solution status: "
                << highs->solutionStatusToString(info.dual_solution_status) << std::endl;
      std::cout << "Basis: " << highs->basisValidityToString(info.basis_validity) << std::endl;
    }

    // Get the model status
    [[maybe_unused]] const HighsModelStatus& model_status = highs->getModelStatus();
    assert(model_status == HighsModelStatus::kOptimal);
  }

  // Extract the used errors
  const HighsSolution& solution = highs->getSolution();
  for (size_t ei = 0; ei < errors.size(); ++ei) {
    if (std::round(solution.col_value[ei]) == 1) {
      predicted_errors_buffer.push_back(ei);
    }
  }
  // Reset the constraints for the detection events
  for (size_t d : detections) {
    model->lp_.row_lower_[d] = 0;
    model->lp_.row_upper_[d] = 0;
  }
}

double SimplexDecoder::cost_from_errors(const std::vector<size_t>& predicted_errors) {
  double total_cost = 0;
  // Iterate over all errors and add to the mask
  for (size_t ei : predicted_errors_buffer) {
    total_cost += errors[ei].likelihood_cost;
  }
  return total_cost;
}

std::vector<int> SimplexDecoder::get_flipped_observables(
    const std::vector<size_t>& predicted_errors) {
  std::unordered_set<int> flipped_observables_set;

  // Iterate over all predicted errors
  for (size_t ei : predicted_errors) {
    // Iterate over the observables associated with each error
    for (int obs_index : errors[ei].symptom.observables) {
      // Perform an XOR-like sum using a set.
      // If the observable is already in the set, it means we've seen it an
      // even number of times, so we remove it.
      // If it's not, we add it, which means we've seen it an odd number of times.
      if (flipped_observables_set.count(obs_index)) {
        flipped_observables_set.erase(obs_index);
      } else {
        flipped_observables_set.insert(obs_index);
      }
    }
  }

  // Convert the set to a vector and return it.
  std::vector<int> flipped_observables(flipped_observables_set.begin(),
                                       flipped_observables_set.end());
  // Sort observables
  std::sort(flipped_observables.begin(), flipped_observables.end());
  return flipped_observables;
}

std::vector<int> SimplexDecoder::decode(const std::vector<uint64_t>& detections) {
  decode_to_errors(detections);
  return get_flipped_observables(predicted_errors_buffer);
}

void SimplexDecoder::decode_shots(std::vector<stim::SparseShot>& shots,
                                  std::vector<std::vector<int>>& obs_predicted) {
  obs_predicted.resize(shots.size());
  for (size_t i = 0; i < shots.size(); ++i) {
    obs_predicted[i] = decode(shots[i].hits);
  }
}

SimplexDecoder::~SimplexDecoder() {}
