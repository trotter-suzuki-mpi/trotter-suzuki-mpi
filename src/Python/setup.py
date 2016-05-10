"""
setup.py file
"""
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy
import os
import sys
import platform
win_cuda_dir = ""


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
    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': os.path.join(home, 'include')}
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in '
                                   '%s' % (k, v))
    libdir = os.path.join(home, 'lib')
    arch = int(platform.architecture()[0][0:2])
    if sys.platform.startswith('win'):
        os.path.join(libdir, "x"+str(arch))
    if os.path.exists(os.path.join(home, libdir + "64")):
        cudaconfig['lib'] = libdir + "64"
    elif os.path.exists(os.path.join(home, libdir)):
        cudaconfig['lib'] = libdir
    else:
        raise EnvironmentError('The CUDA libraries could not be located')
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
    '''This is an almost  verbatim copy of the NVCC compiler extension from
    https://github.com/rmcgibbo/npcuda-example
    '''
    if not sys.platform.startswith('win'):
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

cmdclass = {}
if sys.platform.startswith('win') and os.path.exists(win_cuda_dir):
    arch = int(platform.architecture()[0][0:2])
    ts_module = Extension('_trottersuzuki_wrap',
                          sources=['trottersuzuki/trottersuzuki_wrap.cxx'],
                          extra_objects=['trottersuzuki/src/common.obj',
                                         'trottersuzuki/src/cpublock.obj',
                                         'trottersuzuki/src/model.obj',
                                         'trottersuzuki/src/solver.obj',
                                         'trottersuzuki/src/hybrid.cu.obj',
                                         'trottersuzuki/src/cc2kernel.cu.obj'],
                          define_macros=[('CUDA', None)],
                          library_dirs=[win_cuda_dir+"/lib/x"+str(arch)],
                          libraries=['cudart', 'cublas'],
                          include_dirs=[numpy_include])
else:
    if sys.platform.startswith('win'):
        extra_compile_args = ['-openmp', '-DWIN32']
        libraries = None
    elif sys.platform.startswith('darwin') and 'CC' not in os.environ:
        extra_compile_args = {'cc': []}
        libraries = None
    else:
        extra_compile_args = {'cc': ['-fopenmp']}
        if 'CC' in os.environ and 'clang-omp' in os.environ['CC']:
            libraries = ['iomp5']
        else:
            libraries = ['gomp']
    sources_files = ['trottersuzuki/src/common.cpp',
                     'trottersuzuki/src/cpublock.cpp',
                     'trottersuzuki/src/model.cpp',
                     'trottersuzuki/src/solver.cpp',
                     'trottersuzuki/trottersuzuki_wrap.cxx']

    ts_module = Extension('_trottersuzuki', sources=sources_files,
                          include_dirs=[numpy_include, 'src'],
                          extra_compile_args=extra_compile_args,
                          libraries=libraries,
                          )
    if CUDA is not None:
        ts_module.sources += ['trottersuzuki/src/cc2kernel.cu',
                              'trottersuzuki/src/hybrid.cu']
        ts_module.define_macros = [('CUDA', None)]
        ts_module.include_dirs.append(CUDA['include'])
        ts_module.library_dirs = [CUDA['lib']]
        ts_module.libraries += ['cudart', 'cublas']
        ts_module.runtime_library_dirs = [CUDA['lib']]
        if len(ts_module.extra_compile_args['cc']) > 0:
            extra_args = ts_module.extra_compile_args['cc'][0]
        else:
            extra_args = ""
        ts_module.extra_compile_args['nvcc'] = ['-use_fast_math',
                                                '--ptxas-options=-v', '-c',
                                                '--compiler-options',
                                                '-fPIC ' + extra_args]
    cmdclass = {'build_ext': custom_build_ext}


setup(name='trottersuzuki',
      version='1.5.4',
      license='GPL3',
      author="Peter Wittek, Luca Calderaro",
      author_email='peterwittek@users.noreply.github.com',
      url="http://trotter-suzuki-mpi.github.io/",
      platforms=["unix", "windows"],
      description="Massively Parallel Trotter-Suzuki Solver",
      ext_modules=[ts_module],
      packages=["trottersuzuki"],
      install_requires=['numpy'],
      classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Development Status :: 5 - Production/Stable',
        'Topic :: Scientific/Engineering :: Physics',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: C++'
      ],
      cmdclass=cmdclass
      )
