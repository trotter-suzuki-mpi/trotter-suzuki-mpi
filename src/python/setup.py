#!/usr/bin/env python2

"""
setup.py file for SWIG example
"""

from setuptools import setup, Extension
from setuptools.command.install import install
from subprocess import call
import numpy
import os
import sys

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

if sys.platform.startswith('win'):
    extra_compile_args = ['-openmp', '-DWIN32']
    extra_link_args = []
    sources_files=[	'common.cpp',
                    'cpublock.cpp',
                    'trotter.cpp',
                    'trotter_wrap.cxx']

else:
    extra_compile_args = ['-fopenmp']
    extra_link_args = [
        '-lgomp'
    ]        
    sources_files=[	'common.cpp',
                    'cpublock.cpp',
                    'cpublocksse.cpp',
                    'trotter.cpp',
                    'trotter_wrap.cxx']

trottersuzuki_module = Extension('_trottersuzuki',
                                 sources=sources_files,						   
                                 include_dirs=[numpy_include],
                                 extra_compile_args=extra_compile_args,
                                 extra_link_args=extra_link_args)


setup(name='trottersuzuki',
      version='1.3',
      license='GPL3',
      author="Peter Wittek, Luca Calderaro",
      author_email='peterwittek@users.noreply.github.com',
      url="http://peterwittek.github.io/trotter-suzuki-mpi/",
      platforms=["unix", "windows"],
      description="A massively parallel implementation of the Trotter-Suzuki decomposition",
      ext_modules=[trottersuzuki_module],
      py_modules=["trottersuzuki"],
      install_requires=['numpy']
      )
