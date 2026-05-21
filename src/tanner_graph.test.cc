#include "gtest/gtest.h"
#include "tanner_graph.h"
#include <algorithm>

using namespace tesseract;

TEST(TannerGraphTest, SingleComponent) {
    stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D0 D1
        error(0.1) D1 L0
        detector D0
        detector D1
        logical_observable L0
    )DEM");
    auto components = TannerGraph::find_components(dem);
    ASSERT_EQ(components.size(), 1);
    ASSERT_EQ(components[0].detectors.size(), 2);
    ASSERT_EQ(components[0].observables.size(), 1);
    ASSERT_TRUE(components[0].affects_observable);
}

TEST(TannerGraphTest, TwoDisjointComponents) {
    stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D0 D1
        error(0.1) D2 L0
        detector D0
        detector D1
        detector D2
        logical_observable L0
    )DEM");
    auto components = TannerGraph::find_components(dem);
    ASSERT_EQ(components.size(), 2);
    
    int obs_comp_idx = components[0].affects_observable ? 0 : 1;
    int other_comp_idx = 1 - obs_comp_idx;
    
    ASSERT_EQ(components[obs_comp_idx].detectors.size(), 1); // D2
    ASSERT_EQ(components[obs_comp_idx].observables.size(), 1); // L0
    
    ASSERT_EQ(components[other_comp_idx].detectors.size(), 2); // D0, D1
    ASSERT_EQ(components[other_comp_idx].observables.size(), 0);
    ASSERT_FALSE(components[other_comp_idx].affects_observable);
}

TEST(TannerGraphTest, DecomposedErrorDoesNotUnion) {
    stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D0 D1 ^ D2 D3
        detector D0
        detector D1
        detector D2
        detector D3
    )DEM");
    auto components = TannerGraph::find_components(dem);
    // Should be two components: {D0, D1} and {D2, D3}
    ASSERT_EQ(components.size(), 2);
}

TEST(TannerGraphTest, UndecomposedBridgeUnions) {
    stim::DetectorErrorModel dem(R"DEM(
        error(0.1) D0 D1 D2 D3
        detector D0
        detector D1
        detector D2
        detector D3
    )DEM");
    auto components = TannerGraph::find_components(dem);
    // Should be one component: {D0, D1, D2, D3}
    ASSERT_EQ(components.size(), 1);
}

TEST(TannerGraphTest, PureLogicalErrorComponent) {
    stim::DetectorErrorModel dem(R"DEM(
        error(0.1) L0
        logical_observable L0
    )DEM");
    auto components = TannerGraph::find_components(dem);
    ASSERT_EQ(components.size(), 1);
    ASSERT_EQ(components[0].detectors.size(), 0);
    ASSERT_EQ(components[0].observables.size(), 1);
    ASSERT_TRUE(components[0].affects_observable);
}
