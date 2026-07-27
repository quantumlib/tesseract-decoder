# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Synchronizes the paired Jupytext tutorial notebook."""

import argparse
import difflib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from jupytext.cli import jupytext

FORMATS = "ipynb,py:percent"
NOTEBOOK_PATH = Path("docs/tutorial.ipynb")
SOURCE_PATH = Path("docs/tutorial.py")


def _run_jupytext(args: list[str], cwd: Path) -> None:
    old_cwd = Path.cwd()
    try:
        os.chdir(cwd)
        exit_code = jupytext(args)
    except SystemExit as ex:
        exit_code = ex.code if isinstance(ex.code, int) else 1
    finally:
        os.chdir(old_cwd)

    if exit_code:
        raise RuntimeError(f"jupytext failed with exit code {exit_code}: {' '.join(args)}")


def _write_notebook_from_source(root: Path) -> None:
    if not (root / SOURCE_PATH).exists():
        _run_jupytext(
            [
                str(NOTEBOOK_PATH),
                "--set-formats",
                FORMATS,
                "--quiet",
            ],
            root,
        )
        return

    _run_jupytext(
        [
            str(SOURCE_PATH),
            "--to",
            "ipynb",
            "--update",
            "--output",
            str(NOTEBOOK_PATH),
            "--quiet",
        ],
        root,
    )


def _canonical_json(path: Path) -> list[str]:
    return json.dumps(
        json.loads(path.read_text()),
        indent=2,
        sort_keys=True,
    ).splitlines(keepends=True)


def _check_pair() -> int:
    docs_dir = Path(__file__).resolve().parent
    source = docs_dir / SOURCE_PATH.name
    notebook = docs_dir / NOTEBOOK_PATH.name

    if not source.exists():
        sys.stderr.write(f"{SOURCE_PATH} is missing; run `bazel run //docs:sync_tutorial_notebook -- --write`.\n")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        tmp_docs = tmp_root / "docs"
        tmp_docs.mkdir()
        shutil.copy2(source, tmp_docs / SOURCE_PATH.name)
        shutil.copy2(notebook, tmp_docs / NOTEBOOK_PATH.name)

        _write_notebook_from_source(tmp_root)

        expected = _canonical_json(notebook)
        actual = _canonical_json(tmp_docs / NOTEBOOK_PATH.name)
        if expected != actual:
            sys.stderr.write(
                f"{NOTEBOOK_PATH} is not synchronized with {SOURCE_PATH}. "
                "Run `bazel run //docs:sync_tutorial_notebook -- --write`.\n"
            )
            sys.stderr.writelines(
                difflib.unified_diff(
                    expected,
                    actual,
                    fromfile=str(NOTEBOOK_PATH),
                    tofile="generated tutorial.ipynb",
                )
            )
            return 1

    return 0


def _write_pair() -> int:
    workspace = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if not workspace:
        sys.stderr.write("--write must be run with `bazel run //docs:sync_tutorial_notebook -- --write`.\n")
        return 1

    _write_notebook_from_source(Path(workspace))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Check that the paired tutorial files are synchronized.")
    mode.add_argument("--write", action="store_true", help="Synchronize docs/tutorial.ipynb from docs/tutorial.py.")
    args = parser.parse_args()

    if args.write:
        return _write_pair()
    return _check_pair()


if __name__ == "__main__":
    sys.exit(main())
