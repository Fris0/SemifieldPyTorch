"""
setup.py;

Builds the C++ and CUDA files for the semifield package.
"""

from setuptools import setup
from torch.utils import cpp_extension

setup (
    name="semifield",  # Sets the PyPi package name. Best to set equal to package folder.
    ext_modules=[cpp_extension.CUDAExtension(
        name='semifield.dilation',  # Sets the module name for current file in package. semifield.dilation in Python.
        sources=['semifield/csrc/dilation.cpp', 'semifield/csrc/cuda/dilation_kernel.cu'],  # Source for this extension.
        include_dirs=cpp_extension.include_paths(),
        extra_compile_args=['-O3'],
        language='c++')],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    packages=['semifield'])