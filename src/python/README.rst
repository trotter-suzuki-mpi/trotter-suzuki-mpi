Trotter-Suzuki-MPI - Python Interface
=====================================

The module is a massively parallel implementation of the Trotter-Suzuki approximation to simulate the evolution of quantum systems classically. It relies on interfacing with C++ code with OpenMP for multicore execution.

Key features of the Python interface:

* Fast execution by parallelization: OpenMP.
* NumPy arrays are supported for efficient data exchange.
* Multi-platform: Linux, OS X, and Windows are supported.

Usage
------
The following code block gives a simple example of initializing a state and plotting the result of the evolution.

.. code-block:: python
		
    import numpy as np
    import trottersuzuki as ts
    from matplotlib import pyplot as plt

    imag_time = False
    order_approx = 2
    dim = 640
    iterations = 200
    kernel_type = 0
    periods = [0, 0]

    particle_mass = 1
    time_single_it = 0.08 * particle_mass / 2

    h_a = np.cos(time_single_it / (2. * particle_mass))
    h_b = np.sin(time_single_it / (2. * particle_mass))

    p_real = np.ones((dim,dim))
    p_imag = np.zeros((dim,dim))
    pot_r = np.zeros((dim, dim))
    pot_i = np.zeros((dim, dim))

    CONST = -1. * time_single_it * order_approx
    for y in range(0, dim):
        for x in range(0, dim):
            p_real[y, x] = np.sin(2 * np.pi * x / dim) * np.sin(2 * np.pi * y / dim)
            pot_r[y, x] = np.cos(CONST * pot_r[y, x])
      pot_i[y, x] = np.sin(CONST * pot_i[y, x])

    ts.trotter(h_a, h_b, pot_r, pot_i, p_real, p_imag, iterations, kernel_type, periods, imag_time)

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

Build on Windows
----------------
The pip install command might fail on Windows. If this happens, compile the source with Visual Studio and run the setupWin.py script.
