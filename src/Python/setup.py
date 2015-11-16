"""
setup.py file
"""

from setuptools import setup, Extension
import numpy
import sys

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

if sys.platform.startswith('win'):
    extra_compile_args = ['-openmp', '-DWIN32']
    extra_link_args = []
    sources_files = ['trottersuzuki/common.cpp',
                     'trottersuzuki/cpublock.cpp',
                     'trottersuzuki/trotter.cpp',
                     'trottersuzuki/solver.cpp',
                     'trottersuzuki/trotter_wrap.cxx']
elif sys.platform.startswith('darwin'):
    extra_compile_args = ['-fopenmp']
    extra_link_args = [
        '-lgomp'
    ]
    sources_files = ['trottersuzuki/common.cpp',
                     'trottersuzuki/cpublock.cpp',
                     'trottersuzuki/trotter.cpp',
                     'trottersuzuki/solver.cpp',
                     'trottersuzuki/trotter_wrap.cxx']
else:
    extra_compile_args = ['-fopenmp']
    extra_link_args = [
        '-lgomp'
    ]
    sources_files = ['trottersuzuki/common.cpp',
                     'trottersuzuki/cpublock.cpp',
                     'trottersuzuki/cpublocksse.cpp',
                     'trottersuzuki/trotter.cpp',
                     'trottersuzuki/solver.cpp',
                     'trottersuzuki/trotter_wrap.cxx']

trottersuzuki_module = Extension('_trottersuzuki',
                                 sources=sources_files,
                                 include_dirs=[numpy_include],
                                 extra_compile_args=extra_compile_args,
                                 extra_link_args=extra_link_args)


setup(name='trottersuzuki',
      version='1.4',
      license='GPL3',
      author="Peter Wittek, Luca Calderaro",
      author_email='peterwittek@users.noreply.github.com',
      url="http://trotter-suzuki-mpi.github.io/",
      platforms=["unix", "windows"],
      description="A massively parallel implementation of the Trotter-Suzuki decomposition",
      ext_modules=[trottersuzuki_module],
      py_modules=["trottersuzuki"],
      packages=["trottersuzuki"],
      install_requires=['numpy']
      )
