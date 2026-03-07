#!/usr/bin/env python3
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate Python type stub (.pyi) files for the tesseract_decoder module.

This script uses pybind11-stubgen to produce .pyi stub files for the tesseract decoder
module's C++ API. The .pyi stub files provide information for IDEs.

Built from //src/py:generated_stubs which uses a genrule to call this script.
"""

import argparse
import os
import sys


def _ensure_module_importable():
    """Ensure tesseract_decoder is importable, adjusting sys.path if needed."""

    try:
        import tesseract_decoder
        return tesseract_decoder
    except Exception as e:
        import traceback
        print("====== DEBUG INFO ======", file=sys.stderr)
        print(f"Exception: {type(e).__name__}: {e}", file=sys.stderr)
        print(f"sys.path: {sys.path}", file=sys.stderr)
        print(f"__file__: {__file__}", file=sys.stderr)
        print(f"cwd: {os.getcwd()}", file=sys.stderr)
        print("========================", file=sys.stderr)
        print(
            f"ERROR: Cannot import tesseract_decoder.\n"
            "Ensure the compiled module is on sys.path or PYTHONPATH.",
            file=sys.stderr,
        )
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate .pyi stubs for tesseract_decoder"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to place the generated .pyi file. "
        "Defaults to the directory containing the tesseract_decoder module.",
    )
    args = parser.parse_args()

    module = _ensure_module_importable()
    module_file = os.path.abspath(module.__file__)
    module_dir = os.path.dirname(module_file)

    output_dir = os.path.abspath(args.output_dir) if args.output_dir else module_dir

    print(f"Generating stubs for tesseract_decoder...")
    print(f"  Module location: {module_file}")
    print(f"  Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Use pybind11_stubgen programmatically.
    try:
        from pybind11_stubgen import main as stubgen_main
    except ImportError as e:
        print(f"ImportError: {e}", file=sys.stderr)
        print(f"sys.path: {sys.path}", file=sys.stderr)
        print(
            "ERROR: pybind11-stubgen is not installed. "
            "Install with: pip install pybind11-stubgen",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build argv for pybind11-stubgen CLI.
    # --enum-class-locations maps enum names to their fully-qualified module path
    # so pybind11-stubgen can resolve default values like <DetOrder.DetBFS: 0>.
    stubgen_argv = [
        "pybind11-stubgen",
        "tesseract_decoder",
        "--output-dir",
        output_dir,
        "--enum-class-locations",
        "DetOrder:tesseract_decoder.utils",
    ]

    # Save and restore sys.argv since pybind11-stubgen uses argparse.
    old_argv = sys.argv
    sys.argv = stubgen_argv
    try:
        stubgen_main()
    except SystemExit as e:
        if e.code != 0:
            print(f"ERROR: pybind11-stubgen exited with code {e.code}", file=sys.stderr)
            sys.exit(1)
    finally:
        sys.argv = old_argv

    # Verify the output exists, and append -stubs to the directory name.
    original_stub_pkg_dir = os.path.join(output_dir, "tesseract_decoder")
    stub_pkg_dir = os.path.join(output_dir, "tesseract_decoder-stubs")
    if os.path.exists(original_stub_pkg_dir):
        import shutil
        if os.path.exists(stub_pkg_dir):
            shutil.rmtree(stub_pkg_dir)
        os.rename(original_stub_pkg_dir, stub_pkg_dir)

    stub_init = os.path.join(stub_pkg_dir, "__init__.pyi")

    if os.path.isfile(stub_init):
        print(f"Stubs generated successfully at: {stub_pkg_dir}/")
        for root, dirs, files in os.walk(stub_pkg_dir):
            for f in sorted(files):
                if f.endswith(".pyi"):
                    rel = os.path.relpath(os.path.join(root, f), output_dir)
                    print(f"  {rel}")
    else:
        flat_stub = os.path.join(output_dir, "tesseract_decoder.pyi")
        if os.path.isfile(flat_stub):
            print(f"Stubs generated successfully: {flat_stub}")
        else:
            print("WARNING: Could not verify stub output location.", file=sys.stderr)

    print("Done.")


if __name__ == "__main__":
    main()