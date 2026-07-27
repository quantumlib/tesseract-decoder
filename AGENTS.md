# Agent Instructions

- Use the Bazel build system when interacting with this repository.
- The CMake build system is available to help users who need it.
- Keep both the CMake and Bazel builds working at all times.
- Use at most one CPU core for all builds, tests, and other expensive commands.
  For Bazel, pass `--jobs=1`. For CMake builds, pass `--parallel 1`. For
  `make`, pass `-j1`.

## Building with Bazel

To build all code with bazel:
```bash
bazel build --jobs=1 src:all
```
To build the Tesseract and Simplex main binaries:
```bash
bazel build --jobs=1 src:tesseract src:simplex
```

## Running Tests with Bazel

```bash
bazel test --jobs=1 src/...
```

## Building with CMake

In case you need to build with CMake, keep the build single-core.

- When using `cmake --build`, add `--parallel 1`.
- When using `make`, add `-j1`.

## Running Tests with CMake

To run the tests, execute the following commands from the root of the repository:

```bash
mkdir -p build
cd build
cmake ..
cmake --build . --parallel 1
ctest
```



## Building the Python Wheel

To build the Python wheel for `tesseract_decoder` locally, you will need to use the `bazel build` command and provide the version and Python target version as command-line arguments. This is because the `py_wheel` rule in the `BUILD` file uses "Make" variable substitution, which expects these values to be defined at build time.

Use the following command:

```bash
bazel build --jobs=1 //:tesseract_decoder_wheel --define=VERSION=0.1.1 --define=TARGET_VERSION=py313
```

- `--define=VERSION=0.1.1`: Sets the version of the wheel. You should replace `0.1.1` with the current version from the `_version.py` file.
- `--define=TARGET_VERSION=py313`: Sets the Python version tag for the wheel. Replace `py313` with the appropriate tag for the Python version you are targeting (e.g., `py312` for Python 3.12).

The resulting wheel file will be located in the `bazel-bin/` directory.

## Updating Dependencies

If you change the Python version in `MODULE.bazel`, you will need to regenerate the `requirements_lock.txt` file to ensure all dependencies are compatible. To do this, run the following command:

```bash
bazel run --jobs=1 //src/py:requirements.update
```

This will update the `src/py/requirements_lock.txt` file with the correct dependency versions for the new Python environment.
