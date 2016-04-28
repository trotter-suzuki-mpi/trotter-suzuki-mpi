Download
========
Download the latest stable release [here](https://github.com/trotter_suzuki-mpi/trotter-suzuki-mpi/releases/latest). The development version is on [GitHub](https://github.com/trotter-suzuki-mpi/trotter-suzuki-mpi). If you are interested in the Python version, refer to [Read the Docs](https://trotter-suzuki-mpi.readthedocs.io/). If you encounter problems or a bug, please open an [issue on GitHub](https://github.com/trotter-suzuki-mpi/trotter-suzuki-mpi/issues).

Compilation & Installation
--------------------------
The code was tested with the GNU Compiler Chain (GCC), Clang, Intel compilers, and with Visual Studio. For the distributed variant, you need an MPI installation; OpenMPI and MPICH were tested. The unit testing framework requires [CppUnit](http://sourceforge.net/projects/cppunit/) to compile. To use the GPU-accelerated version, CUDA and a GPU with at least Compute Cabapility 2.0 are necessary.

If you clone the git repository, first run

    $ ./autogen.sh

Then follow the standard procedure:

    $ ./configure [options]
    $ make
    $ make install

To compile and run the unit tests, enter

    $ make test
    $ test/unittest

If you prefer the Intel compilers you have to set the following variables, so mpic++ will invoke icpc instead of the default compiler:

    $ export CC=/path/of/intel/compiler/icc
    $ export CXX=/path/of/intel/compiler/icpc
    $ export OMPI_CC=/path/of/intel/compiler/icc
    $ export OMPI_CXX=/path/of/intel/compiler/icpc

If you compile it on OS X and your Clang version does not support OpenMP, then install GCC first with brew, and then set the environment variables to point to this compiler chain:

    $ export CC=/usr/local/bin/gcc
    $ export CXX=/usr/local/bin/g++
    $ export CPP=/usr/local/bin/cpp
    $ export LD=/usr/local/bin/gcc
    $ alias c++=/usr/local/bin/c++
    $ alias g++=/usr/local/bin/g++
    $ alias gcc=/usr/local/bin/gcc
    $ alias cpp=/usr/local/bin/cpp
    $ alias ld=/usr/local/bin/gcc
    $ alias cc=/usr/local/bin/gcc

Note that GCC is no longer supported by NVCC on OS X, therefore you have to choose between multicore execution and GPU support at compile time if your Clang version does not support OpenMP.

Options for configure

    --prefix=PATH           Set directory prefix for installation

By default, the executable is installed into /usr/local. If you prefer a
different location, use this option to select an installation
directory.

    --with-mpi=MPIROOT      Use MPI root directory.
    --with-mpi-compilers=DIR or --with-mpi-compilers=yes
                              use MPI compiler (mpicxx) found in directory DIR, or
                              in your PATH if =yes
    --with-mpi-libs=LIBS  MPI libraries [default "-lmpi"]
    --with-mpi-incdir=DIR   MPI include directory [default MPIROOT/include]
    --with-mpi-libdir=DIR   MPI library directory [default MPIROOT/lib]

The above flags allow the identification of the correct MPI library the user wishes to use. The flags are especially useful if MPI is installed in a non-standard location, or when multiple MPI libraries are available.

    --with-cuda=/path/to/cuda           Set path for CUDA

The configure script looks for CUDA in /usr/local/cuda. If your installation is elsewhere, then specify the path with this parameter. If you do not want CUDA enabled, set the parameter to ```--without-cuda```.
