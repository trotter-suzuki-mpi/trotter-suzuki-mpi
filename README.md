Massively Parallel Trotter-Suzuki Solver
========================================

The Trotter-Suzuki approximation leads to an efficient algorithm for solving the time-dependent Schödinger equation. This library provides a scalable, high-precision implementation that uses parallel and distributed computational resources. The implementation built on [single-node parallel kernels](https://bitbucket.org/zzzoom/trottersuzuki) [1], extending them to use distributed resources [2], and generalizing the kernels to be able to tackle a wider range of problems in quantum physics. The key features are as follows:

  - Arbitrary single-body initial state with closed and periodic boundary conditions.
  - Many-body simulations with non-interacting particles.
  - Imaginary time evolution to calculate the ground state.
  - Stationary external potential.
  - Command-line interface (CLI) and application programming interface (API) for flexible use.
  - Cache optimized multi-core, SSE, GPU, and hybrid kernels.
  - Near-linear scaling across multiple nodes with computations overlapping communication.


Usage
-----

**Command-line Interface**

Usage: trotter [OPTIONS] -n filename 

The file specified contains the complex matrix describing the initial state in the position picture.

Arguments:

    -a NUMBER     Parameter h_a of kinetic evolution operator (cosine part)
    -b NUMBER     Parameter h_b of kinetic evolution operator (sine part)
    -d NUMBER     Matrix dimension (default: 640)
    -g            Imaginary time evolution to evolve towards the ground state
    -i NUMBER     Number of iterations (default: 1000)
    -k NUMBER     Kernel type (default: 0): 
                    0: CPU, cache-optimized
                    1: CPU, SSE and cache-optimized
                    2: GPU
                    3: Hybrid CPU-GPU (experimental)                    
    -s NUMBER     Snapshots are taken at every NUMBER of iterations.
                    Zero means no snapshots. Default: 0.
    -n FILENAME   The initial state.
    -N NUMBER     Number of particles of the system.
    -p FILENAME   Name of file that stores the potential operator 
                  (in coordinate representation)

Examples:

For a single-threaded run for a hundred iterations with a GPU, starting on some initial state of size 640x640 in psi0.dat and taking snapshots at every ten iterations, enter:

    trotter -k 2 -i 100 -d 640 -s 10 -n psi0.dat

For using eight cores with the CPU kernel, type:

    mpirun -np 4 trotter -i 100 -d 640 -s 10 -n psi0.dat
    
Naturally, if the system is distributed, MPI must be told of a host file. 

In case of the SSE kernel, the chunk of the matrix assigned to a node, that is, a tile, must have a width that is divisible by two. This puts a constraint on the possible matrix sizes. For instance, running twelve MPI threads in a 4x3 configuration, the dimensions must be divisible by six and eight.

The hybrid kernel is experimental. It splits the work between the GPU and the CPU. It uses one MPI thread per GPU, and uses OpenMP to use parallelism on the CPU. It can be faster than the GPU kernel alone, especially if the GPU is consumer-grade. The kernel is especially efficient if the matrix does not fit the GPU memory. For instance, given twelve physical cores in a single node with two Tesla C2050 GPUs, a 14,000x14,000 would not fit the GPU memory. The following command would calculate the part that does not fit the device memory on the CPU:

    OMP_NUM_THREADS=6 mpirun -np 2 build/trotter -k 3 -i 100 -d 14000 -n psi0.dat

**Application Programming Interface**

If the command-line interface is not flexible enough, the function that performs the evolution is exposed as an API:

    void trotter(double h_a, double h_b, 
                 double * external_pot_real, double * external_pot_imag, 
                 double * p_real, double * p_imag, 
                 const int matrix_width, const int matrix_height, 
                 const int iterations, const int snapshots, const int kernel_type, 
                 int *periods, const char *output_folder, 
                 bool verbose = false, bool imag_time = false, int particle_tag = 1);

where the parameters are as follows:

    h_a               Kinetic term of the Hamiltonian (cosine part)
    h_b               Kinetic term of the Hamiltonian (sine part)
    external_pot_real External potential, real part
    external_pot_imag External potential, imaginary part
    p_real            Initial state, real part
    p_imag            Initial state, imaginary part
    matrix_width      The width of the initial state
    matrix_height     The height of the initial state
    iterations        Number of iterations to be calculated
    snapshots         Number of iterations between taking snapshots 
                             (0 means no snapshots)
    kernel_type       The kernel type:
                              0: CPU block kernel
                              1: CPU SSE block kernel
                              2: GPU kernel
                              3: Hybrid kernel
    periods            Whether the grid is periodic in any of the directions
    output_folder      The folder to write the snapshots in
    verbose            Optional verbosity parameter
    imag_time          Optional parameter to calculate imaginary time evolution
    particle_tag       Optional parameter to tag a particle in the snapshots
  
MPI must be initialized before the function is called. Examples of using the API are included in the source tree. The respective files are in the examples folder:

  - `exponential_initial_state.cpp`: Time evolution of a particle in a box with an exponential initial state with periodic boundary conditions.
  - `gaussian-like_initial_state.cpp`: Time evolution of a particle in a box with a Gaussian-like initial state with closed boundary conditions.
  - `imag_evolution.cpp`: Imaginary time evolution of an exponential initial state with periodic boundary conditions.
  - `sinusoid_initial_state.cpp`: Time evolution of a particle in a box with a sinusoid initial state with periodic boundary conditions.
  - `two_particles_exponential_initial_state.cpp`: Time evolution of two free particles in a box with periodic boundary conditions.


Compilation & Installation
--------------------------
The code was tested with the GNU Compiler Chain (GCC) and with Intel compilers. An MPI implementation is required for compiling. The unit testing framework is separate and it requires [CppUnit](http://sourceforge.net/projects/cppunit/) to compile. To use the GPU-accelerated version, CUDA and a GPU with at least Compute Cabapility 2.0 are necessary.

If you clone the git repository, first run

    $ ./autogen.sh

Then follow the standard POSIX procedure:

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

Acknowledgement
---------------
The [original high-performance kernels](https://bitbucket.org/zzzoom/trottersuzuki) were developed by Carlos Bederián. The distributed extension was carried out while [Peter Wittek](http://peterwittek.com/) was visiting the [Department of Computer Applications in Science \& Engineering](http://www.bsc.es/computer-applications) at the [Barcelona Supercomputing Center](http://www.bsc.es/), funded by the "Access to BSC Facilities" project of the [HPC-Europe2](http://www.hpc-europa.org/) programme (contract no. 228398). Generalizing the capabilities of kernels was carried out by Luca Calderaro while visiting the [Quantum Information Theory Group](https://www.icfo.eu/research/group_details.php?id=19) at [ICFO-The Institute of Photonic Sciences](https://www.icfo.eu/), sponsored by the [Erasmus+](http://ec.europa.eu/programmes/erasmus-plus/index_en.htm) programme.

References
----------
  
  1. Bederián, C. & Dente, A. [Boosting quantum evolutions using Trotter-Suzuki algorithms on GPUs](http://www.famaf.unc.edu.ar/grupos/GPGPU/boosting_trotter-suzuki.pdf). Proceedings of HPCLatAm-11, 4th High-Performance Computing Symposium, 2011.
  
  2. Wittek, P. and Cucchietti, F.M. (2013). [A Second-Order Distributed Trotter-Suzuki Solver with a Hybrid CPU-GPU Kernel](http://dx.doi.org/10.1016/j.cpc.2012.12.008). Computer Physics Communications, 184, pp. 1165-1171. [PDF](http://arxiv.org/pdf/1208.2407)
