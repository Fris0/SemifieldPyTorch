import os, glob, torch

from setuptools import setup, find_packages, Extension

setup(
    name="dilation",
    version="0.0.1",
    packages=find_packages()
    ext_modules=[
        Extension
        ]
)
