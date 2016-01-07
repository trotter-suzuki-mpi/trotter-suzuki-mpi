/**
 * Massively Parallel Trotter-Suzuki Solver
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

/**
 * This source provides an example of the trotter-suzuki program.
 * It calculates the time-evolution of a particle in a box, where the initial
 * state is the following:
 * 		exp(-( (x - 180)² + (y - 300)² )/ 2s²)* exp(0.4j * (x + y - 480.));
 */
#include <iostream>
#include <sys/stat.h>
#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include "trottersuzuki.h"

#define ITERATIONS 1000
#define DIM 640
#define KERNEL_TYPE "cpu"
#define SNAPSHOTS 20

complex<double> gauss_state(double x, double y) {
    return complex<double>(exp(-(pow(x - 180, 2.0) + pow(y - 300, 2.0)) / (2.0 * 64 * 64.)), 0.0)
           * exp(complex<double>(0.0, 0.4 * (x + y - 480.0)));  
}

int main(int argc, char** argv) {
    double length_x = double(DIM), length_y = double(DIM);
    double delta_t = 0.1;
    bool verbose = true;
    bool imag_time = false;
    double particle_mass = 1.;

#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif
    //set lattice
    Lattice *grid = new Lattice(DIM, length_x, length_y);
    //set initial state
    State *state = new State(grid);
    state->init_state(gauss_state);
    //set hamiltonian
    Hamiltonian *hamiltonian = new Hamiltonian(grid, particle_mass);
    hamiltonian->initialize_potential(const_potential);
    //set evolution
    Solver *solver = new Solver(grid, state, hamiltonian, delta_t, KERNEL_TYPE);

    if(grid->mpi_rank == 0) {
        cout << "\n* This source provides an example of the trotter-suzuki program.\n";
        cout << "* It calculates the time-evolution of a particle in a box, where the initial\n";
        cout << "* state is the following:\n";
        cout << "* \texp(-( (x - 180)² + (y - 300)² )/ 2s²)* exp(0.4j * (x + y - 480.))\n\n";
    }

    //set file output directory
    stringstream dirname, fileprefix;
    dirname << "D" << DIM << "_I" << ITERATIONS << "_S" << SNAPSHOTS << "";
    mkdir(dirname.str().c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    
    for(int count_snap = 0; count_snap < SNAPSHOTS; count_snap++) {
        solver->evolve(ITERATIONS, imag_time);
        fileprefix.str("");
        fileprefix << dirname << "/" << 1 << "-" << ITERATIONS * count_snap;
        state->write_to_file(fileprefix.str());
    }
    if (grid->mpi_coords[0] == 0 && grid->mpi_coords[1] == 0 && verbose == true) {
        cout << "TROTTER " << DIM << "x" << DIM << " kernel:" << KERNEL_TYPE << " np:" << grid->mpi_procs << endl;
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
