#ifndef _VISUALIZATION_H
#define _VISUALIZATION_H

#include <boost/dynamic_bitset.hpp>
#include <list>
#include <vector>

#include "common.h"

struct Visualizer {
  void add_detector_coords(const std::vector<std::vector<double>>&);
  void add_errors(const std::vector<common::Error>&);
  void add_activated_errors(int64_t node_idx, const std::vector<common::ErrorChainNode>& arena);
  void add_activated_detectors(const boost::dynamic_bitset<>&, size_t);

  void write(const char* fpath);

 private:
  std::list<std::string> lines;
};

#endif
