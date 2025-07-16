#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.tree import DecisionTreeRegressor, plot_tree, _tree
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def parse_log_file(filepath):
    features = []
    labels = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                error_blocked_str = line.split("error_blocked=")[1].split(";")[0]
                detectors_str = line.split("detectors_count=")[1].split(";")[0]
                result_str = line.split("result=")[1]

                error_blocked = np.array([int(b) for b in error_blocked_str], dtype=np.uint8)
                detectors = np.array([int(x) for x in detectors_str.split(",") if x], dtype=np.uint8)
                label = float(result_str)

                feature_vec = (1 - error_blocked) * detectors
                features.append(feature_vec)
                labels.append(label)

            except Exception as e:
                print(f"Failed to parse line: {e}")
    return np.array(features), np.array(labels)


def evaluate_model(name, model, X_train, y_train, X_test, y_test, fit_time):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    abs_errors = np.abs(y_test - y_test_pred)

    print(f"\n=== {name} ===")
    print(f"Train time:                 {fit_time:.3f} s")
    print(f"Train R^2 score:            {r2_score(y_train, y_train_pred):.4f}")
    print(f"Test R^2 score:             {r2_score(y_test, y_test_pred):.4f}")
    print(f"Max absolute deviation:     {np.max(abs_errors):.6f}")
    print(f"Median absolute deviation:  {np.median(abs_errors):.6f}")


def visualize_tree(model, feature_dim, filename):
    plt.figure(figsize=(min(32, feature_dim // 2), 20))
    plot_tree(model, filled=True, max_depth=None, feature_names=[f"x{i}" for i in range(feature_dim)], fontsize=6)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Decision tree plot saved to '{filename}'")


def tree_to_cpp_switch(trees_by_d: dict) -> str:
    def emit_tree(tree, d):
        tree_ = tree.tree_

        def recurse(node, depth):
            indent = "    " * depth
            if tree_.feature[node] == _tree.TREE_UNDEFINED:
                return f"{indent}return {tree_.value[node][0][0]:.10f};\n"
            i = tree_.feature[node]
            threshold = tree_.threshold[node]
            feature_expr = (
                f"(1 - int(detector_cost_tuples[d2e[d][{i}]].error_blocked)) * "
                f"detector_cost_tuples[d2e[d][{i}]].detectors_count"
            )
            left = recurse(tree_.children_left[node], depth + 1)
            right = recurse(tree_.children_right[node], depth + 1)
            return (
                f"{indent}if ({feature_expr} <= {threshold:.10f}) {{\n"
                f"{left}"
                f"{indent}}} else {{\n"
                f"{right}"
                f"{indent}}}\n"
            )

        return f"    case {d}:\n{recurse(0, 2)}    break;\n"

    all_switch_cases = "".join(emit_tree(tree, d) for d, tree in sorted(trees_by_d.items()))

    return (
        "#ifndef TREE_H\n"
        "#define TREE_H\n\n"
        "#include <vector>\n"
        "#include \"tesseract.h\"\n\n"
        "double decision_tree_predict(size_t d, const std::vector<DetectorCostTuple>& detector_cost_tuples, const std::vector<std::vector<int>>& d2e) {\n"
        "  switch (d) {\n"
        f"{all_switch_cases}"
        "    default:\n"
        "      return -1.0; // Unknown detector\n"
        "  }\n"
        "}\n\n"
        "#endif // TREE_H\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile", help="Path to logfile.txt")
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--max-leaf-nodes", type=int, default=None)
    parser.add_argument("--cpp-impl-out", type=str, default=None, help="Write C++ function to file")
    args = parser.parse_args()

    from collections import defaultdict
    detectors = defaultdict(list)  # d → list of (X_row, y)

    with open(args.logfile) as f:
        for line in f:
            if not line.startswith("Detcost("):
                continue
            try:
                prefix, rest = line.strip().split("):error_blocked=")
                d = int(prefix[len("Detcost("):])
                eb_str, rest = rest.split(";detectors_count=")
                dc_str, result_str = rest.split(";result=")

                eb = np.array([int(b) for b in eb_str.strip()], dtype=np.uint8)
                dc = np.array([int(x) for x in dc_str.strip().split(",") if x], dtype=np.uint8)
                y = float(result_str.strip())
                x = (1 - eb) * dc
                detectors[d].append((x, y))
            except Exception as e:
                print(f"Skipping line due to parse error: {e}\n{line.strip()}")

    trees_by_d = {}
    print("\n=== Per-detector Decision Tree Training Stats ===")
    for d, examples in sorted(detectors.items()):
        if not examples:
            continue
        X = np.array([e[0] for e in examples])
        y = np.array([e[1] for e in examples])
        tree = DecisionTreeRegressor(
            max_depth=args.max_depth,
            max_leaf_nodes=args.max_leaf_nodes,
            min_samples_leaf=10,
            min_samples_split=10,
            random_state=42
        )
        start = time.time()
        tree.fit(X, y)
        end = time.time()
        y_pred = tree.predict(X)
        abs_errors = np.abs(y - y_pred)
        r2 = r2_score(y, y_pred)
        print(
            f"Detector {d:3d}: "
            f"R^2 = {r2:6.4f}, "
            f"max error = {np.max(abs_errors):.6f}, "
            f"median error = {np.median(abs_errors):.6f}, "
            f"n = {len(y):4d}, "
            f"time = {end - start:.3f}s"
        )
        trees_by_d[d] = tree


    if args.cpp_impl_out:
        cpp_code = tree_to_cpp_switch(trees_by_d)
        with open(args.cpp_impl_out, "w") as f:
            f.write(cpp_code)
        print(f"C++ implementation written to {args.cpp_impl_out}")


if __name__ == "__main__":
    main()
