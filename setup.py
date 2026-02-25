"""Build script for Cython accelerator extension (optional)."""

import numpy as np
from setuptools import setup, Extension

try:
    from Cython.Build import cythonize
    extensions = cythonize([
        Extension("pykarambola._accel", ["pykarambola/_accel.pyx"],
                  include_dirs=[np.get_include()])
    ], language_level=3)
except ImportError:
    extensions = []

setup(ext_modules=extensions)
