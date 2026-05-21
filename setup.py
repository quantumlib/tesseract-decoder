from setuptools import setup, find_packages
import subprocess
import os
import sys

def build_with_bazel():
    print("Building C++ extension with Bazel...")
    try:
        subprocess.check_call(["bazel", "build", "//src/py:tesseract_decoder"])
        # Copy the output .so file to the package directory
        src = "bazel-bin/src/py/tesseract_decoder/_core.so"
        dst = "src/py/tesseract_decoder/_core.so"
        print(f"Copying {src} to {dst}...")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        subprocess.check_call(["cp", src, dst])
    except Exception as e:
        print(f"Warning: Failed to build C++ extension with Bazel: {e}")
        print("You may need to build it manually using 'bazel build //src/py:tesseract_decoder'")

# Always attempt to build with bazel. 
# Bazel's own incremental build logic will ensure this is fast if no changes occurred.
build_with_bazel()

setup(
    name="tesseract_decoder",
    version="0.1.1",
    package_dir={"": "src/py"},
    packages=find_packages(where="src/py"),
    install_requires=[
        "stim",
        "sinter",
        "numpy",
    ],
    package_data={
        "tesseract_decoder": ["_core.so"],
    },
    include_package_data=True,
    zip_safe=False,
)
