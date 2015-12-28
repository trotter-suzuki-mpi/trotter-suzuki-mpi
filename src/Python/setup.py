"""
setup.py file
"""
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy
import os
import sys
import platform

sse = False

def find_cuda():
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = os.path.join(home, 'bin', 'nvcc')
    else:
        nvcc = None
        for dir in os.environ['PATH'].split(os.pathsep):
            binpath = os.path.join(dir, 'nvcc')
            if os.path.exists(binpath):
                nvcc = os.path.abspath(binpath)
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be located in '
                                   'your $PATH. Either add it to your path, or'
                                   'set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))
    libdir = "lib"
    arch = int(platform.architecture()[0][0:2])
    if sys.platform.startswith('win'):
        os.path.join(libdir, "x"+str(arch))
    elif arch == 64:
        libdir += "64"
    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': os.path.join(home, 'include'),
                  'lib': os.path.join(home, libdir)}
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in '
                                   '%s' % (k, v))

    return cudaconfig

try:
    CUDA = find_cuda()
except EnvironmentError:
    CUDA = None
    print("Proceeding without CUDA")

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


arch = int(platform.architecture()[0][0:2])
cmdclass = {}

def customize_compiler_for_nvcc(self):
    '''This is a verbatim copy of the NVCC compiler extension from
    https://github.com/rmcgibbo/npcuda-example
    '''
    self.src_extensions.append('.cu')
    default_compiler_so = self.compiler_so
    super = self._compile

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            self.set_executable('compiler_so', CUDA['nvcc'])
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['cc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        self.compiler_so = default_compiler_so
    self._compile = _compile


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


if sys.platform.startswith('win'):
    extra_compile_args = ['-openmp', '-DWIN32']
    openmp = ''
else:
    extra_compile_args = ['-fopenmp']
    if 'CC' in os.environ and 'clang-omp' in os.environ['CC']:
        openmp = 'iomp5'
    else:
        openmp = 'gomp'
sources_files = ['trottersuzuki/src/common.cpp',
                 'trottersuzuki/src/cpublock.cpp',
                 'trottersuzuki/src/trotter.cpp',
                 'trottersuzuki/src/solver.cpp',
                 'trottersuzuki/trotter_wrap.cxx']

if sse:
    sources_files.append('trottersuzuki/src/cpublocksse.cpp')
ts_module = Extension('_trottersuzuki', sources=sources_files,
                      include_dirs=[numpy_include, 'src'],
                      extra_compile_args={'cc': extra_compile_args},
                      libraries=[openmp],
                      )
if CUDA is not None:
    ts_module.sources.append('trottersuzuki/src/cc2kernel.cu')
    ts_module.define_macros = [('CUDA', None)]
    ts_module.include_dirs.append(CUDA['include'])
    ts_module.library_dirs = [CUDA['lib']]
    ts_module.libraries += ['cudart', 'cublas']
    ts_module.runtime_library_dirs = [CUDA['lib']]
    ts_module.extra_compile_args['nvcc']=['-use_fast_math', 
                                          '--ptxas-options=-v', '-c',
                                          '--compiler-options','-fPIC ' +
                                          extra_compile_args[0]]


setup(name='trottersuzuki',
      version='1.4',
      license='GPL3',
      author="Peter Wittek, Luca Calderaro",
      author_email='peterwittek@users.noreply.github.com',
      url="http://trotter-suzuki-mpi.github.io/",
      platforms=["unix", "windows"],
      description="A massively parallel implementation of the Trotter-Suzuki decomposition",
      ext_modules=[ts_module],
      packages=["trottersuzuki"],
      install_requires=['numpy'],
      cmdclass={'build_ext': custom_build_ext}
      )
