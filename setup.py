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

# One translation unit per numerical area; module.cpp registers their tables.
sources = sorted(path.relative_to(ROOT).as_posix() for path in (ROOT / "cpp").glob("*.cpp"))
headers = sorted(path.relative_to(ROOT).as_posix() for path in (ROOT / "cpp").glob("*.hpp"))

extensions = [
    Extension(
        "pyrate._kernels",
        sources=sources,
        depends=headers,
        include_dirs=["cpp"],
        language="c++",
        extra_compile_args=compile_args,
    )
]

setup(ext_modules=extensions)
