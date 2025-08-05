# setup.py

from setuptools import setup
from Cython.Build import cythonize
import numpy
import os

setup(
    ext_modules=cythonize("convolution22.pyx", compiler_directives={"language_level": "3"}),
    include_dirs=[numpy.get_include()]
)
