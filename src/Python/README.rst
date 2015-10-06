Trotter-Suzuki-MPI - Python Interface
=====================================

The module is a massively parallel implementation of the Trotter-Suzuki approximation to simulate the evolution of quantum systems classically. It relies on interfacing with C++ code with OpenMP for multicore execution.

Key features of the Python interface:

* Fast execution by parallelization: OpenMP.
* NumPy arrays are supported for efficient data exchange.
* Multi-platform: Linux, OS X, and Windows are supported.

Usage
------
The following code block gives a simple example of initializing a state and calculating the expectation values of the Hamiltonian and kinetic operators and the norm of the state after the evolution.

.. code-block:: python
		
    import numpy as np
    import trottersuzuki as ts
    from matplotlib import pyplot as plt

    # lattice parameters
    dim = 200
    delta_x = 1.
    delta_y = 1.
    periods = [1, 1]

    # Hamiltonian parameter
    particle_mass = 1
    coupling_const = 0.
    external_potential = np.zeros((dim, dim))

    # initial state
    p_real = np.ones((dim,dim))
    p_imag = np.zeros((dim,dim))
    for y in range(0, dim):
        for x in range(0, dim):
            p_real[y, x] = np.sin(2 * np.pi * x / dim) * np.sin(2 * np.pi * y / dim)

    # evolution parameters
    imag_time = False
    delta_t = 0.001
    iterations = 200
    kernel_type = 0

    # launch evolution
    ts.solver(p_real, p_imag, particle_mass, coupling_const, external_potential, delta_x, delta_y, delta_t, iterations, kernel_type, periods, imag_time)

    # expectation values
    Energy = ts.H(p_real, p_imag, particle_mass, coupling_const, external_potential, delta_x, delta_y)
    print Energy

    Kinetic_Energy = ts.K(p_real, p_imag, particle_mass, delta_x, delta_y)
    print Kinetic_Energy

    Norm2 = ts.Norm2(p_real, p_imag, delta_x, delta_y)
    print Norm2


Another code block that finds the ground state of a single particle in a well potential.

.. code-block:: python

    import numpy as np
    import trottersuzuki as ts
    from matplotlib import pyplot as plt

    # lattice parameters
    dim = 200
    delta_x = 1.
    delta_y = 1.
    periods = [0, 0]

    # Hamiltonian parameter
    particle_mass = 1
    coupling_const = 0.
    external_potential = np.zeros((dim, dim))

    # initial state
    p_real = np.ones((dim,dim))
    p_imag = np.zeros((dim,dim))	

    # evolution parameters
    imag_time = True
    delta_t = 0.1
    iterations = 2000
    kernel_type = 1

    for i in range(0, 15):
        # launch evolution
        ts.solver(p_real, p_imag, particle_mass, coupling_const, external_potential, delta_x, delta_y, delta_t, iterations, kernel_type, periods, imag_time)
        print ts.H(p_real, p_imag, particle_mass, coupling_const, external_potential, delta_x, delta_y)
        print ts.Norm2(p_real, p_imag, delta_x, delta_y)

    heatmap = plt.pcolor(p_real)
    plt.show()


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
