# Agent Instructions

- Use the **CMake** build system when interacting with this repository. Humans use Bazel.
- A bug in some LLM coding environments makes Bazel difficult to use, so agents should rely on CMake.
- Keep both the CMake and Bazel builds working at all times.

## Building the Python Wheel

To build the Python wheel for `tesseract_decoder` locally, you will need to use the `bazel build` command and provide the version and Python target version as command-line arguments. This is because the `py_wheel` rule in the `BUILD` file uses "Make" variable substitution, which expects these values to be defined at build time.

Use the following command:

```bash
bazel build //:tesseract_decoder_wheel --define=VERSION=0.1.1 --define=TARGET_VERSION=py313
```

- `--define=VERSION=0.1.1`: Sets the version of the wheel. You should replace `0.1.1` with the current version from the `_version.py` file.
- `--define=TARGET_VERSION=py313`: Sets the Python version tag for the wheel. Replace `py313` with the appropriate tag for the Python version you are targeting (e.g., `py312` for Python 3.12).

The resulting wheel file will be located in the `bazel-bin/` directory.

## Updating Dependencies

If you change the Python version in `MODULE.bazel`, you will need to regenerate the `requirements_lock.txt` file to ensure all dependencies are compatible. To do this, run the following command:

```bash
bazel run //src/py:requirements.update
```

This will update the `src/py/requirements_lock.txt` file with the correct dependency versions for the new Python environment.
