import argparse

def parse_d2e_and_ecosts_file(filepath):
    d2e = {}
    ecosts = {}
    min_costs = {}

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("d2e["):
                d = int(line.split("[")[1].split("]")[0])
                values = line.split("=")[1].split(",")
                d2e[d] = [int(x) for x in values if x]
            elif line.startswith("ecosts["):
                d = int(line.split("[")[1].split("]")[0])
                values = line.split("=")[1].split(",")
                ecosts[d] = [float(x) for x in values if x]
            elif line.startswith("error_min_costs["):
                d = int(line.split("[")[1].split("]")[0])
                values = line.split("=")[1].split(",")
                min_costs[d] = [float(x) for x in values if x]

    result = {}
    for d in d2e:
        if d not in ecosts or d not in min_costs:
            raise ValueError(f"Missing data for detector {d}")
        if not (len(d2e[d]) == len(ecosts[d]) == len(min_costs[d])):
            raise ValueError(f"Mismatch in lengths for detector {d}")
        result[d] = list(zip(d2e[d], ecosts[d], min_costs[d]))

    return result

def emit_cpp_switch(d2e_and_costs):
    lines = []
    lines.append("#ifndef TREE_H")
    lines.append("#define TREE_H\n")
    lines.append("#include <vector>")
    lines.append("#include <limits>")
    lines.append("#include \"tesseract.h\"\n")
    lines.append("double decision_tree_predict(size_t d, const std::vector<DetectorCostTuple>& detector_cost_tuples) {")
    lines.append("  switch (d) {")

    for d, triples in sorted(d2e_and_costs.items()):
        lines.append(f"    case {d}: {{")
        lines.append("      double min_cost = std::numeric_limits<double>::infinity();")
        count = 0
        for ei, ecost, next_min in triples:
            lines.append(f"      if (!detector_cost_tuples[{ei}].error_blocked) {{")
            lines.append(f"        double cost = {ecost:.10f} / detector_cost_tuples[{ei}].detectors_count;")
            lines.append("        if (cost < min_cost) min_cost = cost;")
            lines.append("      }")
            if count > 10 and count % 20 == 0:
              lines.append(f"      if (min_cost <= {next_min:.10f}) return min_cost;")
            if count > 10:
              break
            count += 1
        lines.append("      return min_cost;")
        lines.append("    }")

    lines.append("    default:")
    lines.append("      return -1.0;")
    lines.append("  }")
    lines.append("}")
    lines.append("\n#endif // TREE_H")
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d2e-file", required=True, help="Path to file containing d2e[...] and ecosts[...] and error_min_costs[...]")
    parser.add_argument("--cpp-impl-out", required=True, help="Output .cpp file path")
    args = parser.parse_args()

    d2e_data = parse_d2e_and_ecosts_file(args.d2e_file)
    cpp_code = emit_cpp_switch(d2e_data)

    with open(args.cpp_impl_out, "w") as f:
        f.write(cpp_code)
    print(f"Wrote C++ implementation to: {args.cpp_impl_out}")

if __name__ == "__main__":
    main()
