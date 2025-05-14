from setuptools import setup
from torch.utils import cpp_extension

setup (
    name="semifield",  # Sets the PyPi package name. Best to set equal to package folder.
    ext_modules=[cpp_extension.CppExtension(
        name='semifield.dilation',  # Sets the module name for current file in package. semifield.dilation in Python.
        sources=['semifield/csrc/dilation.cpp'],  # Source for this extension.
        include_dirs=cpp_extension.include_paths(),
        extra_compile_args=['-O3'],
        language='c++')],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    packages=['semifield'])