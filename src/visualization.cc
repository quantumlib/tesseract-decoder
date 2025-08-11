
#include "visualization.h"

void Visualizer::add_errors(const std::vector<common::Error>& errors) {
  for (auto& error : errors) {
    lines.push_back(error.str());
  }
}
void Visualizer::add_detector_coords(const std::vector<std::vector<double>>& detector_coords) {
  for (size_t d = 0; d < detector_coords.size(); ++d) {
    std::stringstream ss;
    ss << "Detector D" << d << " coordinate (";
    size_t e = std::min(3ul, detector_coords[d].size());
    for (size_t i = 0; i < e; ++i) {
      ss << detector_coords[d][i];
      if (i + 1 < e) ss << ", ";
    }
    ss << ")";
    lines.push_back(ss.str());
  }
}

void Visualizer::add_activated_errors(const std::vector<size_t>& activated_errors) {
  std::stringstream ss;
  ss << "activated_errors = ";
  for (size_t oei : activated_errors) {
    ss << oei << ", ";
  }
  lines.push_back(ss.str());
}

void Visualizer::add_activated_detectors(const boost::dynamic_bitset<>& detectors,
                                         size_t num_detectors) {
  std::stringstream ss;
  ss << "activated_detectors = ";
  for (size_t d = 0; d < num_detectors; ++d) {
    if (detectors[d]) {
      ss << d << ", ";
    }
  }
  lines.push_back(ss.str());
}

void Visualizer::write(const char* fpath) {
  FILE* fout = fopen(fpath, "w");

  for (std::string& line : lines) {
    fprintf(fout, line.c_str());
    fputs("\n", fout);
  }

  fclose(fout);
}
