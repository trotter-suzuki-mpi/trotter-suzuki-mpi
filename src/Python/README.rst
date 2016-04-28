Trotter-Suzuki-MPI - Python Interface
=====================================

The module is a massively parallel implementation of the Trotter-Suzuki approximation to simulate the evolution of quantum systems classically. It relies on interfacing with C++ code with OpenMP for multicore execution, and it can be accelerated by CUDA.

Key features of the Python interface:

* Simulation of 2D quantum systems.
* Fast execution by parallelization: OpenMP and CUDA are supported.
* Many-body simulations with non-interacting particles.
* Solving the Gross-Pitaevskii equation  (e.g., `dark soltions <https://github.com/Lucacalderaro/Master-Thesis/blob/master/Soliton%20generation%20on%20Bose-Einstein%20Condensate.ipynb>`_, `vortex dynamics in Bose-Einstein Condensates <http://nbviewer.jupyter.org/github/trotter-suzuki-mpi/notebooks/blob/master/Vortex%20Dynamics.ipynb>`_).
* Imaginary time evolution to approximate the ground state.
* Stationary and time-dependent external potential.
* NumPy arrays are supported for efficient data exchange.
* Multi-platform: Linux, OS X, and Windows are supported.

Documentation is available on `Read the Docs <https://trotter-suzuki-mpi.readthedocs.io/>`_.

Installation
------------
The code is available on PyPI, hence it can be installed by

::

    $ sudo pip install trottersuzuki

If you want the latest git version, follow the standard procedure for installing Python modules:

::

    $ sudo python setup.py install

Build on Mac OS X
-----------------
Before installing using pip, gcc should be installed first. As of OS X 10.9, gcc is just symlink to clang. To build trottersuzuki and this extension correctly, it is recommended to install gcc using something like:
::

    $ brew install gcc48

and set environment using:
::

    export CC=/usr/local/bin/gcc
    export CXX=/usr/local/bin/g++
    export CPP=/usr/local/bin/cpp
    export LD=/usr/local/bin/gcc
    alias c++=/usr/local/bin/c++
    alias g++=/usr/local/bin/g++
    alias gcc=/usr/local/bin/gcc
    alias cpp=/usr/local/bin/cpp
    alias ld=/usr/local/bin/gcc
    alias cc=/usr/local/bin/gcc

Then you can issue
::

    $ sudo pip install trottersuzuki

Build with CUDA support on Linux and OS X:
------------------------------------------
If your CUDA is installed elsewhere than /usr/local/cuda, you cannot directly install the module from PyPI. Please download the `source distribution <https://pypi.python.org/pypi/trottersuzuki/>`_ from PyPI. Open the setup.py file in an editor and modify the path to your CUDA installation directory:

::

   cuda_dir = /path/to/cuda

Then run the install command

::

    $ sudo python setup.py install

Build with CUDA support on Windows:
--------------------------------------
You should first follow the instructions to `build the Windows binary <http://trotter-suzuki-mpi.github.io/>`_ with MPI disabled with the same version Visual Studio as your Python is built with.(Since currently Python is built by VS2008 by default and CUDA v6.5 removed VS2008 support, you may use CUDA 6.0 with VS2008 or find a Python prebuilt with VS2010. And remember to install VS2010 or Windows SDK7.1 to get the option in Platform Toolset if you use VS2013.) Then you should copy the .obj files generated in the release build path to the Python/src folder.

Then modify the win_cuda_dir in setup.py to your CUDA path and run the install command

::

    $ sudo python setup.py install

Then it should be able to build and install the module.

Citations
---------

1. Bederián, C. and Dente, A. (2011). Boosting quantum evolutions using Trotter-Suzuki algorithms on GPUs. *Proceedings of HPCLatAm-11, 4th High-Performance Computing Symposium*.

2. Wittek, P. and Cucchietti, F.M. (2013). `A Second-Order Distributed Trotter-Suzuki Solver with a Hybrid CPU-GPU Kernel <http://dx.doi.org/10.1016/j.cpc.2012.12.008>`_. *Computer Physics Communications*, 184, pp. 1165-1171.

3. Wittek, P. and Calderaro, L. (2015). `Extended computational kernels in a massively parallel implementation of the Trotter-Suzuki approximation <http://dx.doi.org/10.1016/j.cpc.2015.07.017>`_. *Computer Physics Communications*, 197, pp. 339-340.
