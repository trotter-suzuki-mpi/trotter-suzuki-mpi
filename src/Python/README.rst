Trotter-Suzuki-MPI - Python Interface
=====================================

The module is a massively parallel implementation of the Trotter-Suzuki approximation to simulate the evolution of quantum systems classically. It relies on interfacing with C++ code with OpenMP for multicore execution, and it can be accelerated by CUDA.

Key features of the Python interface:

* Fast execution by parallelization: OpenMP and CUDA are supported.
* NumPy arrays are supported for efficient data exchange.
* Multi-platform: Linux, OS X, and Windows are supported.

Usage
------
Documentation is available on [Read the Docs](http://trotter-suzuki-mpi.readthedocs.org). The following code block gives a simple example of initializing a state and calculating the expectation values of the Hamiltonian and kinetic operators and the norm of the state after the evolution.

.. code-block:: python
		
    from __future__ import print_function
    import numpy as np
    import trottersuzuki as ts

    # lattice parameters
    dim = 200
    delta_x = 1.
    delta_y = 1.
    periods = [1, 1]

    # Hamiltonian parameter
    particle_mass = 1
    external_potential = np.zeros((dim, dim))

    # initial state
    p_real = np.ones((dim, dim))
    p_imag = np.zeros((dim, dim))
    for y in range(0, dim):
        for x in range(0, dim):
            p_real[y, x] = np.sin(2*np.pi*x / dim) * np.sin(2*np.pi*y / dim)

    # evolution parameters
    delta_t = 0.001
    iterations = 200

    # launch evolution
    ts.evolve(p_real, p_imag, particle_mass, external_potential, delta_x, delta_y,
              delta_t, iterations, periods=periods)

    # expectation values
    Energy = ts.calculate_total_energy(p_real, p_imag, particle_mass,
                                       external_potential, delta_x, delta_y)
    print(Energy)

    Kinetic_Energy = ts.calculate_kinetic_energy(p_real, p_imag, particle_mass,
                                                 delta_x, delta_y)
    print(Kinetic_Energy)

    Norm2 = ts.calculate_norm2(p_real, p_imag, delta_x, delta_y)
    print(Norm2)


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
