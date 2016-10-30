Tutorial
========

This tutorial is about using the C++ API. If you are interested in the Python version, refer to [Read the Docs](https://trotter-suzuki-mpi.readthedocs.io/).

**Multicore computations**

This example uses all cores on a single computer to calculate the total energy after evolving a sinusoid initial state. First we set the physical and simulation parameters of the model. We set the mass equal to one, we discretize the space in 500 lattice points in either direction, and we set the physical length to the same value. We would like to have a hundred iterations with 0.01 second between each:

~~~~~~~~~~~~~~~{.cpp}
double particle_mass = 1.;
int dimension = 500.;
double length = double(dimension);
double delta_t = 0.01;
int iterations  = 100;
~~~~~~~~~~~~~~~

The next step is the define the lattice, the state, and the Hamiltonian:

~~~~~~~~~~~~~~~{.cpp}
Lattice2D *grid = new Lattice2D(dimension, length);
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
    double length = double(dimension);
    double delta_t = 0.01;
    int iterations  = 100;

    Lattice2D *grid = new Lattice2D(dimension, length);
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

The compilation is the same as above. For using multiple GPUs, compile the code with MPI and launch one process for each GPU.

**Distributed version**

There is very little modification required in the code to make it work with MPI. It is sufficient to initialize MPI and finalize it before returning from `main`. It is worth noting that the `Lattice` class keeps track of the MPI-related topology, and it also knows the MPI rank of the current process. The code for `simple_example_mpi.cpp` is as follows:


~~~~~~~~~~~~~~~{.cpp}
#include <iostream>
#include <mpi.h>
#include "trottersuzuki.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    double particle_mass = 1.;
    int dimension = 500.;
    double length = double(dimension);
    double delta_t = 0.01;
    int iterations  = 100;

    Lattice2D *grid = new Lattice2D(dimension, length);
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
    MPI_Finalize();
    return 0;
}
~~~~~~~~~~~~~~~

Compile it with

~~~~~~~~~~~~~~~{.cpp}
mpic++ -I/PATH/TO/TROTTERSUZUKI/HEADER -L/PATH/TO/TROTTERSUZUKI/LIBRARY simple_example_mpi.cpp -o simple_example_mpi -ltrottersuzuki
~~~~~~~~~~~~~~~

Keep in mind that the library itself has to be compiled with MPI to make it work.

The MPI compilation disables OpenMP multicore execution in the CPU kernel, therefore you must launch a process for each CPU core you want use.
