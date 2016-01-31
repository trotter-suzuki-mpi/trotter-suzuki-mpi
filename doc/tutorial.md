Tutorial
========

This tutorial is about using the command-line interface and the C++ API. If you are interested in the Python or the MATLAB version, refer to [Read the Docs](https://trotter-suzuki-mpi.readthedocs.org) and [FileExchange](https://www.mathworks.com/matlabcentral/fileexchange/51975-mextrotter), respectively.

Command-line Interface
----------------------
The command-line interface is severely limited in functionality. For instance, only static external potential and single-component Hamiltonians are supported. Most functionality related to the solution of various flavours of the Gross-Pitaevskii equation is also inaccessible. It is primarily useful to study the evolution of states with the linear Schrödinger equation. For anything more complicated, please use the C++ API or the Python interface.

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
The command-line interface is restricted in what it can do and the full capabalities of the library are unleashed through its C++ API. Here we give an introduction to the features.

**Multicore computations**

This example uses all cores on a single computer to calculate the total energy after evolving a sinusoid initial state. First we set the physical and simulation parameters of the model. We set the mass equal to one, we discretize the space in 500 lattice points in either direction, and we set the physical length to the same value. We would like to have a hundred iterations with 0.01 second between each:

~~~~~~~~~~~~~~~{.cpp}
double particle_mass = 1.;
int dimension = 500.;
double length_x = double(dimension), length_y = double(dimension);
double delta_t = 0.01;
int iterations  = 100;
~~~~~~~~~~~~~~~

The next step is the define the lattice, the state, and the Hamiltonian:

~~~~~~~~~~~~~~~{.cpp}
Lattice *grid = new Lattice(dimension, length_x, length_y);
State *state = new SinusoidState(grid, 1, 1);
Hamiltonian *hamiltonian = new Hamiltonian(grid, NULL, particle_mass);
~~~~~~~~~~~~~~~

With these objects representing the physics of the problem, we can initialize the solver:

~~~~~~~~~~~~~~~{.cpp}
Solver *solver = new Solver(grid, state, hamiltonian, delta_t);
~~~~~~~~~~~~~~~

Then we can evolve the state for the hundred iterations:

~~~~~~~~~~~~~~~{.cpp}
solver->evolve(iterations);
~~~~~~~~~~~~~~~

If we would like to have imaginary time evolution to approximate the ground state of the system, a second boolean parameter can be passed to the `evolve` method. Flipping it to true will yield imaginary time evolution.

We can write the evolved state to a file:

~~~~~~~~~~~~~~~{.cpp}
state->write_to_file("evolved_state");
~~~~~~~~~~~~~~~

If we need a series of snapshots of the evolution, say, every hundred iterations, we can loop these two steps, adjusting the prefix of the file to be written to reflect the number of evolution steps.

Finally, we can calculate the expectation value of the energies:

~~~~~~~~~~~~~~~{.cpp}
std::cout << "Squared norm: " << solver->get_squared_norm();
std::cout << " Kinetic energy: " << solver->get_kinetic_energy();
std::cout << " Total energy: " << solver->get_total_energy() << std::endl;
~~~~~~~~~~~~~~~

The following file, `simple_example.cpp`, summarizes the above:

~~~~~~~~~~~~~~~{.cpp}
#include <iostream>
#include "trottersuzuki.h"

int main(int argc, char** argv) {
    double particle_mass = 1.;
    int dimension = 500.;
    double length_x = double(dimension), length_y = double(dimension);
    double delta_t = 0.01;
    int iterations  = 100;

    Lattice *grid = new Lattice(dimension, length_x, length_y);
    State *state = new SinusoidState(grid, 1, 1);
    Hamiltonian *hamiltonian = new Hamiltonian(grid, NULL, particle_mass);
    Solver *solver = new Solver(grid, state, hamiltonian, delta_t);

    solver->evolve(iterations);
    state->write_to_file("evolved_state");

    std::cout << "Squared norm: " << solver->get_squared_norm();
    std::cout << " Kinetic energy: " << solver->get_kinetic_energy();
    std::cout << " Total energy: " << solver->get_total_energy() << std::endl;
    delete solver;
    delete hamiltonian;
    delete state;
    delete grid;
    return 0;
}
~~~~~~~~~~~~~~~

Compile it with

~~~~~~~~~~~~~~~{.cpp}
g++ -I/PATH/TO/TROTTERSUZUKI/HEADER -L/PATH/TO/TROTTERSUZUKI/LIBRARY simple_example.cpp -o simple_example -ltrottersuzuki
~~~~~~~~~~~~~~~

**GPU version**

If the library was compiled with CUDA support, it is enough to change a single line of code, requesting the GPU kernel when instantiating the solver class:

~~~~~~~~~~~~~~~{.cpp}
     Solver *solver = new Solver(grid, state, hamiltonian, delta_t, "gpu");
~~~~~~~~~~~~~~~

The compilation is the same as above.

**Distributed version**

There is very little modification required in the code to make it work with MPI. It is sufficient to initialize MPI and finalize it before returning from `main`. It is worth noting that the `Lattice` class keeps track of the MPI-related topology, and it also knows the MPI rank of the current process. The code for `simple_example_mpi.cpp` is as follows:


~~~~~~~~~~~~~~~{.cpp}
#include <iostream>
#include <mpi.h>
#include "trottersuzuki.h"

int main(int argc, char** argv) {
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif  
    double particle_mass = 1.;
    int dimension = 500.;
    double length_x = double(dimension), length_y = double(dimension);
    double delta_t = 0.01;
    int iterations  = 100;

    Lattice *grid = new Lattice(dimension, length_x, length_y);
    State *state = new SinusoidState(grid, 1, 1);
    Hamiltonian *hamiltonian = new Hamiltonian(grid, NULL, particle_mass);
    Solver *solver = new Solver(grid, state, hamiltonian, delta_t);

    solver->evolve(iterations, false);

    if(grid->mpi_rank == 0){
        state->write_to_file("evolved_state");
        std::cout << "Squared norm: " << solver->get_squared_norm();
        std::cout << " Kinetic energy: " << solver->get_kinetic_energy();
        std::cout << " Total energy: " << solver->get_total_energy() << std::endl;
    }
    delete solver;
    delete hamiltonian;
    delete state;
    delete grid;
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return 0;
}
~~~~~~~~~~~~~~~

Compile it with

~~~~~~~~~~~~~~~{.cpp}
mpic++ -I/PATH/TO/TROTTERSUZUKI/HEADER -L/PATH/TO/TROTTERSUZUKI/LIBRARY simple_example_mpi.cpp -o simple_example_mpi -ltrottersuzuki
~~~~~~~~~~~~~~~

Keep in mind that the library itself has to be compiled with MPI to make it work.

The same caveats apply for execution as for the command-line interface. The MPI compilation disables OpenMP multicore execution in the CPU kernel, therefore you must launch a process for each CPU core you want use.
