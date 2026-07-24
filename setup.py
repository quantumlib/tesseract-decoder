import os
import shutil
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class BazelExtension(Extension):
    def __init__(self, name, bazel_target):
        super().__init__(name, sources=[])
        self.bazel_target = bazel_target

class bazel_build_ext(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_bazel(ext)
        # Removed super().run() to prevent setuptools from overwriting our file.

    def build_bazel(self, ext):
        # 1. Run bazel build
        print(f"Building {ext.bazel_target} with Bazel...")
        subprocess.check_call(["bazel", "build", ext.bazel_target])
        
        # 2. Locate the output .so file
        src_path = os.path.join("bazel-bin", "src", "tesseract_decoder.so")
        
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Could not find built extension at {src_path}")
            
        # 3. Get the destination path for the extension
        ext_path = self.get_ext_fullpath(ext.name)
        ext_dir = os.path.dirname(ext_path)
        os.makedirs(ext_dir, exist_ok=True)
        
        # 4. Copy the file to the destination
        print(f"Copying {src_path} to {ext_path}")
        shutil.copyfile(src_path, ext_path)

setup(
    name="tesseract_decoder",
    version="0.1.1",
    ext_modules=[BazelExtension("tesseract_decoder", "//src:tesseract_decoder")],
    cmdclass={"build_ext": bazel_build_ext},
    package_dir={"": "src/py"},
    packages=["_tesseract_py_util"],
    zip_safe=False,
    install_requires=[
        "numpy",
        "stim",
        "sinter",
    ],
)
