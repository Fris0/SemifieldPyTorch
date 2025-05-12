import os, glob, torch

from setuptools import setup


from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)

library_name = "extension_cpp"

setup (
    name = library_name,
    version = "0.0.1",
    description= "Extension for Semifield Convolutions in PyTorch.",
    packages=None,
    ext_modules=[
        CppExtension(
            name=f"{library_name}._C",
            sources=["extension_cpp/csrc/muladd.cpp"],
            include_dirs=["./env/include/python3.12"],  # Only needed if Python.h is not found automatically
        )
    ],
    install_requires=["torch"],
    include_dirs=["./env/include/python3.12"],
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}}
)
