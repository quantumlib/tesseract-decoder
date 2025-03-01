#include "common.h"

#include "gtest/gtest.h"

#include "stim.h"

TEST(common, ErrorsStructFromDemInstruction) {
  // Test a pathological DEM error instruction
  stim::DetectorErrorModel dem("error(0.1) D0 ^  D0 D1  L0 L1 L1");
  stim::DemInstruction instruction = dem.instructions.at(0);
  common::Error ES(instruction);
  EXPECT_EQ(ES.symptom.detectors, std::vector<int>{1});
  EXPECT_EQ(ES.symptom.observables, 0b01);
}
