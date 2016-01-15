Tutorial
========

This tutorial is about using the command-line interface and the C++ API. If you are interested in the Python or the MATLAB version, refer to [Read the Docs](https://trotter-suzuki-mpi.readthedocs.org) and [FileExchange](https://www.mathworks.com/matlabcentral/fileexchange/51975-mextrotter), respectively.

Command-line Interface
----------------------

Usage: `trottersuzuki [OPTIONS] -n filename`

The file specified contains the complex matrix describing the initial state in the position picture.

Arguments:

    -m NUMBER     Particle mass
    -c NUMBER     Coupling constant of the self-interacting term (default: 0)
    -d NUMBER     Matrix dimension (default: 640)
    -l NUMBER     Physical dimension of the square lattice's edge (default: 640)
    -t NUMBER     Single time step (default: 0.01)
    -i NUMBER     Number of iterations (default: 1000)
    -g            Imaginary time evolution to evolve towards the ground state
    -k NUMBER     Kernel type (default: 0): 
                    0: CPU, cache-optimized
                    2: GPU
                    3: Hybrid CPU-GPU (experimental)                    
    -s NUMBER     Snapshots are taken at every NUMBER of iterations.
                    Zero means no snapshots. Default: 0.
    -n FILENAME   The initial state.
    -p FILENAME   Name of file that stores the potential operator 
                  (in coordinate representation)

Examples:

For using all cores of the CPU kernel with OpenMP parallelization starting on some initial state of size 640x640 in psi0.dat and taking snapshots at every ten iterations, type:

    trottersuzuki -i 100 -d 640 -s 10 -n psi0.dat


For a hundred iterations with a GPU, enter:

    trotter -k 2 -i 100 -d 640 -s 10 -n psi0.dat

To run it on a cluster, you must compile the code with MPI. In this case, OpenMP parallelization is disabled in the CPU kernel. Hence the total number of MPI processes must match your number of cores per node multiplied by the total number of nodes. Say, with an eight core CPU in four nodes, you would type

    mpirun -np 32 trotter -i 100 -d 640 -s 10 -n psi0.dat

   
Naturally, if the system is distributed, MPI must be told of a host file. 

The hybrid kernel is experimental. It splits the work between the GPU and the CPU. It uses one MPI thread per GPU, and uses OpenMP to use parallelism on the CPU. It can be faster than the GPU kernel alone, especially if the GPU is consumer-grade. The kernel is especially efficient if the matrix does not fit the GPU memory. For instance, given twelve physical cores in a single node with two Tesla C2050 GPUs, a 14,000x14,000 would not fit the GPU memory. The following command would calculate the part that does not fit the device memory on the CPU:

    OMP_NUM_THREADS=6 mpirun -np 2 build/trotter -k 3 -i 100 -d 14000 -n psi0.dat

Application Programming Interface
---------------------------------
If the command-line interface is not flexible enough, the function that performs the evolution is exposed as an API:

    void trotter(double h_a, double h_b, double coupling_const,
                 double * external_pot_real, double * external_pot_imag,
                 double * p_real, double * p_imag, double delta_x, double delta_y,
                 const int matrix_width, const int matrix_height,
                 const int iterations, const int kernel_type,
                 int *periods, double norm, bool imag_time);

where the parameters are as follows:

    h_a               Kinetic term of the Hamiltonian (cosine part)
    h_b               Kinetic term of the Hamiltonian (sine part)
    coupling_const    Coupling constant of the self-interacting term
    external_pot_real External potential, real part
    external_pot_imag External potential, imaginary part
    p_real            Initial state, real part
    p_imag            Initial state, imaginary part
    delta_x           Physical distance between two neighbour points of the lattice along the x axis
    delta_y           Physical distance between two neighbour points of the lattice along the y axis
    matrix_width      The width of the initial state
    matrix_height     The height of the initial state
    iterations        Number of iterations to be calculated
    kernel_type       The kernel type:
                              0: CPU block kernel
                              2: GPU kernel
                              3: Hybrid kernel
    periods           Whether the grid is periodic in any of the directions
    norm              Norm of the final state (only for imaginary time evolution)
    imag_time         Optional parameter to calculate imaginary time evolution
  
MPI must be initialized before the function is called. 
