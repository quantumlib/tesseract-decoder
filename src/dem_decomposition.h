#ifndef DEM_DECOMPOSITION_H
#define DEM_DECOMPOSITION_H

#include <algorithm>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <vector>
#include <string>
#include <tuple>

#include "stim.h"

namespace tesseract {

// Calculates the symmetric difference of a multiset of items.
// Returns items that appear an odd number of times in the input.
std::vector<int> reduce_symmetric_difference(const std::vector<int>& items);

// Calculates the symmetric difference of a multiset of items given as a vector of sets.
std::vector<int> reduce_set_symmetric_difference(const std::vector<std::vector<int>>& sets);

// Extracts detector and observable indices from a Stim error instruction,
// handling decomposed errors by taking the symmetric difference.
std::pair<std::vector<int>, std::vector<int>> undecomposed_error_detectors_and_observables(
    const stim::DemInstruction& instruction);

/**
 * Given possible observables for each component and the error's observables,
 * finds a consistent assignment of observables to components.
 * 
 * @param obs_options_by_component A list of sets, where each set contains the possible
 *                                 observable flip combinations for a component.
 * @param error_obs The total logical observables flipped by the undecomposed error.
 * @param num_missing_components Number of components that were not found in the DEM.
 * @param allow_remnant_errors If true, allow components missing from the DEM to be assigned 
 *                             residual observables.
 */
std::vector<std::vector<int>> get_component_obs_matching_undecomposed_obs(
    const std::vector<std::set<std::vector<int>>>& obs_options_by_component,
    const std::vector<int>& error_obs,
    int num_missing_components = 0,
    bool allow_remnant_errors = false);

/**
 * Decomposes errors in a DetectorErrorModel based on detector assignments to components.
 * 
 * @param dem The input DetectorErrorModel.
 * @param detector_component_func A function that maps a detector ID to a component ID (int).
 * @param allow_remnant_errors If true, allow the decomposition to succeed even if some
 *                             components are missing from the DEM, by inferring their observables.
 */
stim::DetectorErrorModel decompose_errors_using_detector_assignment(
    const stim::DetectorErrorModel& dem,
    const std::function<int(int)>& detector_component_func,
    bool allow_remnant_errors = false);

/**
 * A generic classifier that receives full metadata for a detector.
 */
using DetectorClassifier = std::function<int(int index, const std::vector<double>& coords, const std::string& tag)>;

/**
 * Decomposes errors using a generic classifier that can look at index, coordinates, and tags.
 */
stim::DetectorErrorModel decompose_errors_using_generic_classifier(
    const stim::DetectorErrorModel& dem,
    const DetectorClassifier& classifier,
    bool allow_remnant_errors = false);

/**
 * Splits a decomposed DEM into separate DEMs, one for each component ID.
 */
std::map<int, stim::DetectorErrorModel> split_dem_by_component(
    const stim::DetectorErrorModel& dem,
    const std::function<int(int)>& detector_component_func);

// Returns a detector error model with any error decompositions removed.
stim::DetectorErrorModel undecompose_errors(
    const stim::DetectorErrorModel& dem);

// Merges error instructions in a DEM that have the same symptom.
stim::DetectorErrorModel merge_indistinguishable_errors(
    const stim::DetectorErrorModel& dem);


} // namespace tesseract

#endif // DEM_DECOMPOSITION_H
