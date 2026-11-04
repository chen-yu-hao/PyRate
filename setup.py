from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import Extension, setup

ROOT = Path(__file__).parent

if sys.platform == "win32":
    compile_args = ["/std:c++17", "/O2", "/EHsc"]
else:
    compile_args = ["-std=c++17", "-O3", "-Wall", "-Wextra", "-Wpedantic"]
    if os.environ.get("PYRATE_NATIVE_ARCH") == "1":
        compile_args.append("-march=native")

extensions = [
    Extension(
        "pyrate._kernels",
        sources=["cpp/kernels.cpp"],
        language="c++",
        extra_compile_args=compile_args,
    ),
    Extension(
        "pyrate._wkb_kernels",
        sources=["cpp/wkb_kernels.cpp"],
        language="c++",
        extra_compile_args=compile_args,
    ),
]

setup(ext_modules=extensions)
