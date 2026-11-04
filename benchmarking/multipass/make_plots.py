#!/usr/bin/env python3
"""Generate unified tradeoff, Hyperbolic, and Stage Breakdown plots via benchmarking/viz/export_static.py."""

import pathlib
import sys

VIZ_DIR = pathlib.Path(__file__).resolve().parent.parent / "viz"
if str(VIZ_DIR) not in sys.path:
    sys.path.insert(0, str(VIZ_DIR))

import export_static


def main() -> None:
    export_static.generate_all()


if __name__ == "__main__":
    main()
