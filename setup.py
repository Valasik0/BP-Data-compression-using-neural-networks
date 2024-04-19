# setup.py
import setuptools
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys

__version__ = '0.0.1'

class get_pybind_include(object):
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        return "C:\Python311\Lib\site-packages\pybind11\include"

ext_modules = [
    Extension(
        'entropy',
        ['pybind_interface.cpp'],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++'
    ),
]

setup(
    name='entropy',
    version=__version__,
    author='Your Name',
    author_email='your.email@example.com',
    description='A Python wrapper for the KthEntropyCalculator class',
    long_description='',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.5.0'],
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)