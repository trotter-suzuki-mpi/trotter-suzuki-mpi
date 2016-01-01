/**
- * Massively Parallel Trotter-Suzuki Solver
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
 * 		exp(i2M_PI / L (x + y))
 */
#include <iostream>
#include <sys/stat.h>
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "trottersuzuki.h"

#define DIM 640
#define ITERATIONS 100
#define KERNEL_TYPE "cpu"
#define SNAPSHOTS 10

complex<double> super_position_two_exp_state(int x, int y, Lattice *grid) {
    double L_x = grid->global_dim_x - grid->periods[1] * 2 * grid->halo_x;

    return exp(complex<double>(0. , 2.*3.14159/L_x*(x-grid->periods[1]*grid->halo_x))) +
           exp(complex<double>(0. , 10.*2.*3.14159/L_x*(x-grid->periods[1]*grid->halo_x)));
}

int main(int argc, char** argv) {
    int periods[2] = {1, 1};
    bool verbose = true;
    double coupling_const = 0;
    double delta_x = 1, delta_y = 1;
    double delta_t = 0.08;
    const double particle_mass = 1.;
    int rot_coord_x = 320, rot_coord_y = 320;
    double omega = 0;
    double norm = 1;
    bool imag_time = true;

#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif
    Lattice *grid = new Lattice(DIM, delta_x, delta_y, periods, omega);

    //set initial state
    State *state = new State(grid);
    state->init_state(super_position_two_exp_state);
    Hamiltonian *hamiltonian = new Hamiltonian(grid, particle_mass, coupling_const, 0, 0, rot_coord_x, rot_coord_y, omega);
    hamiltonian->initialize_potential(const_potential);
    Solver *solver = new Solver(grid, state, hamiltonian, delta_t, KERNEL_TYPE);


    if(grid->mpi_rank == 0) {
        cout << "\n* This source provides an example of the trotter-suzuki program.\n";
        cout << "* It calculates the imaginary time-evolution of a free particle in a box\n";
        cout << "* with periodic boundary conditions, where the initial\n";
        cout << "* state is the following:\n";
        cout << "* \texp(i2M_PI / L * x) + exp(i20M_PI / L * x)\n\n";
        cout << "* The state will reach the eigenfunction of the Hamiltonian with the lowest\n";
        cout << "* eigenvalue:   exp(i2M_PI / L * x)\n\n";
    }

    //set file output directory
    stringstream dirname;
    string dirnames;
    if (SNAPSHOTS) {
        int status;
        dirname.str("");
        dirname << "D" << DIM << "_I" << ITERATIONS << "_S" << SNAPSHOTS << "";
        dirnames = dirname.str();
        status = mkdir(dirnames.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if(status != 0 && status != -1)
            dirnames = ".";
    } else {
        dirnames = ".";
    }
    for(int count_snap = 0; count_snap < SNAPSHOTS; count_snap++) {
        solver->evolve(ITERATIONS, imag_time);
        stamp(grid, state, 0, ITERATIONS, count_snap, dirnames.c_str());
    }
    if (grid->mpi_rank == 0 && verbose == true) {
        cout << "TROTTER " << DIM << "x" << DIM << " kernel:" << KERNEL_TYPE << " np:" << grid->mpi_procs << endl;
    }
    delete solver;
    delete hamiltonian;
    delete state;
    delete grid;
    return 0;
}
