from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension("my_cpp_module",  # import this-name
                      ["my_module.cpp"],  # source files
                      define_macros = [('FLOAT_COMPATIBLE', True)]),  # define macros when compile
]

setup(
    name="my_cpp_module",  # package name install in the os
    version="0.0.0", 
    author="author name", 
    author_email="author email", 
    url="home page",  # 
    description="summary",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},  # search for the highest supported C++ standard
    zip_safe=False,  # whether the project can be safely installed and run from a zip file.
    python_requires=">=3.7",
    install_requires=[],
)
