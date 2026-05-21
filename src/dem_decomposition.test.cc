#include "gtest/gtest.h"
#include "dem_decomposition.h"
#include <vector>
#include <set>
#include <string>

using namespace tesseract;

TEST(DemDecompositionTest, ReduceSymmetricDifference) {
    ASSERT_EQ(reduce_symmetric_difference({1, 2, 3}), std::vector<int>({1, 2, 3}));
    ASSERT_EQ(reduce_symmetric_difference({1, 1}), std::vector<int>({}));
    ASSERT_EQ(reduce_symmetric_difference({3, 0, 1, 4, 1, 2, 4}), std::vector<int>({0, 2, 3}));
}

TEST(DemDecompositionTest, ReduceSetSymmetricDifference) {
    ASSERT_EQ(reduce_set_symmetric_difference({{1, 2, 3}, {2, 4, 0}}), std::vector<int>({0, 1, 3, 4}));
    ASSERT_EQ(reduce_set_symmetric_difference({{}, {}}), std::vector<int>({}));
}

TEST(DemDecompositionTest, GetComponentObsMatchingUndecomposedObs) {
    std::vector<std::set<std::vector<int>>> component_obs = {{{0, 1}, {2, 1}}, {{3, 4}, {10, 0}}};
    std::vector<int> error_obs = {1, 10};
    std::vector<std::vector<int>> expected_output = {{0, 1}, {10, 0}};
    ASSERT_EQ(get_component_obs_matching_undecomposed_obs(component_obs, error_obs, 0, false), expected_output);

    component_obs = {{{}}, {{}}};
    error_obs = {};
    expected_output = {{}, {}};
    ASSERT_EQ(get_component_obs_matching_undecomposed_obs(component_obs, error_obs, 0, false), expected_output);

    component_obs = {{{}}, {{}}};
    error_obs = {0};
    expected_output = {};
    ASSERT_EQ(get_component_obs_matching_undecomposed_obs(component_obs, error_obs, 0, false), expected_output);
}

TEST(DemDecompositionTest, RemnantErrorsSingleMissingComponent) {
    std::vector<std::set<std::vector<int>>> component_obs = {{{1}}};
    std::vector<int> error_obs = {1, 2};
    std::vector<std::vector<int>> expected_output = {{1}, {2}};
    ASSERT_EQ(get_component_obs_matching_undecomposed_obs(component_obs, error_obs, 1, true), expected_output);
}

TEST(DemDecompositionTest, RemnantErrorsNoKnownComponents) {
    std::vector<std::set<std::vector<int>>> component_obs = {};
    std::vector<int> error_obs = {1, 2};
    std::vector<std::vector<int>> expected_output = {{1, 2}};
    ASSERT_EQ(get_component_obs_matching_undecomposed_obs(component_obs, error_obs, 1, true), expected_output);
}

TEST(DemDecompositionTest, RemnantErrorsBestEffortForcedFirst) {
    // Known components provide {1}. Error needs {2}. Residual is {1, 2}.
    // Forced first takes {1} XOR {1, 2} = {2}.
    std::vector<std::set<std::vector<int>>> component_obs = {{{1}}};
    std::vector<int> error_obs = {2};
    std::vector<std::vector<int>> expected_output = {{2}};
    ASSERT_EQ(get_component_obs_matching_undecomposed_obs(component_obs, error_obs, 0, true), expected_output);
}

TEST(DemDecompositionTest, DecomposeErrorsUsingGenericClassifier) {
    stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D0 ^ D1 L1
        error(0.01) D0 D3 D3 D1 L5 L4 L4
        error(0.3) D0 D1 D3 D3 D2 D3 L0 L5
        error(0.2) D3 D2 D0 D0 L0
        detector(0) D0
        detector(0) D1
        detector(1) D2
        detector(1) D3
    )DEM");
    
    // Classifier based on coordinate
    auto classifier = [](int index, const std::vector<double>& coords, const std::string& tag) -> int {
        if (coords.empty()) return 0;
        return (int)coords.back();
    };

    stim::DetectorErrorModel expected_decomposed_dem(R"DEM(
        error(0.1) D0 D1 L1
        error(0.01) D0 D1 L5
        error(0.3) D0 D1 L5 ^ D2 D3 L0
        error(0.2) D2 D3 L0
        detector(0) D0
        detector(0) D1
        detector(1) D2
        detector(1) D3
    )DEM");
    ASSERT_EQ(decompose_errors_using_generic_classifier(dem, classifier).str(), expected_decomposed_dem.str());
}

TEST(DemDecompositionTest, DecomposeErrorsUsingGenericClassifierTagBased) {
    stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D0 D1
        error(0.2) D2 D3
        error(0.3) D0 D2
        error(0.01) D0
        error(0.01) D2
        detector[{"basis": "X"}] D0
        detector[{"basis": "X"}] D1
        detector[{"basis": "Z"}] D2
        detector[{"basis": "Z"}] D3
    )DEM");
    
    // Classifier based on finding "X" or "Z" in the tag
    auto classifier = [](int index, const std::vector<double>& coords, const std::string& tag) -> int {
        if (tag.find("\"X\"") != std::string::npos) return 0;
        if (tag.find("\"Z\"") != std::string::npos) return 1;
        return 2;
    };

    stim::DetectorErrorModel decomposed = decompose_errors_using_generic_classifier(dem, classifier);
    
    bool found_d0d2_decomposed = false;
    for (const auto& inst : decomposed.flattened().instructions) {
        if (inst.type == stim::DemInstructionType::DEM_ERROR && inst.arg_data[0] == 0.3) {
            bool has_separator = false;
            for (const auto& target : inst.target_data) {
                if (target.is_separator()) {
                    has_separator = true;
                    break;
                }
            }
            if (has_separator) {
                found_d0d2_decomposed = true;
            }
        }
    }
    ASSERT_TRUE(found_d0d2_decomposed);
}

TEST(DemDecompositionTest, SplitDemByComponent) {
    stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D0 D1
        error(0.2) D2 D3
        error(0.3) D0 D2 L0
        error(0.01) D0
        error(0.01) D2 L0
        detector D0
        detector D1
        detector D2
        detector D3
        logical_observable L0
    )DEM");
    
    auto classifier = [](int index, const std::vector<double>& coords, const std::string& tag) -> int {
        return (index < 2) ? 0 : 1; // 0,1 -> comp 0; 2,3 -> comp 1
    };

    stim::DetectorErrorModel decomposed = decompose_errors_using_generic_classifier(dem, classifier);
    
    auto comp_func = [](int id) { return (id < 2) ? 0 : 1; };
    auto dems = split_dem_by_component(decomposed, comp_func);

    ASSERT_EQ(dems.size(), 2);
    ASSERT_EQ(dems[0].count_errors(), 3);
    ASSERT_EQ(dems[1].count_errors(), 3);
}

TEST(DemDecompositionTest, UndecomposeErrorsWithRepeatBlock) {
    stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D2 D5 ^ D10 L1
        repeat 10 {
            error(0.4) D1 L2 L3 ^ D2 ^ D2 L2
            repeat 3 {
                error(0.3) D10 D11 ^ D12
            }
        }
        error(0.5) D0 D100
    )DEM");
    stim::DetectorErrorModel expected_undecomposed_dem(R"DEM(
        error(0.1) D2 D5 D10 L1
        repeat 10 {
            error(0.4) D1 L3
            repeat 3 {
                error(0.3) D10 D11 D12
            }
        }
        error(0.5) D0 D100
    )DEM");
    ASSERT_EQ(undecompose_errors(dem).str(), expected_undecomposed_dem.str());
}

TEST(DemDecompositionTest, MergeIndistinguishableErrors) {
    stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D0 D1
        error(0.2) D0 D1
        error(0.05) D2
        error(0.05) D2
        detector D0
        detector D1
        detector D2
    )DEM");
    stim::DetectorErrorModel merged = merge_indistinguishable_errors(dem);
    ASSERT_EQ(merged.count_errors(), 2);
}
