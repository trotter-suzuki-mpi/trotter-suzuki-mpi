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
 * 		exp(i2M_PI / L (x + y))
 */
#include <iostream>
#include <sys/stat.h>
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "trottersuzuki.h"

#define LENGTH 50
#define DIM 640
#define ITERATIONS 2000
#define PARTICLES_NUM 1700000
#define KERNEL_TYPE "cpu"
#define SNAPSHOTS 10
#define SCATTER_LENGTH_2D 5.662739242e-5

complex<double> gauss_ini_state(double x, double y) {
	double x_c = x - double(LENGTH)*0.5, y_c = y - double(LENGTH)*0.5;
    double w = 0.01;
    return complex<double>(sqrt(w * double(PARTICLES_NUM) / M_PI) * exp(-(x_c * x_c + y_c * y_c) * 0.5 * w), 0.0);
}

double parabolic_potential(double x, double y) {
    double x_c = x - double(LENGTH)*0.5, y_c = y - double(LENGTH)*0.5;
    double w_x = 1, w_y = 1. / sqrt(2); 
    return 0.5 * (w_x * w_x * x_c * x_c + w_y * w_y * y_c * y_c);
}

int main(int argc, char** argv) {
    int periods[2] = {0, 0};
    bool verbose = true;
    int rot_coord_x = 320, rot_coord_y = 320;
    double omega = 0;
    double norm = 1;
    bool imag_time = true;
    double delta_t = 1.e-4;
    const double particle_mass = 1.;
    double coupling_const = 4. * M_PI * double(SCATTER_LENGTH_2D);    
    
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif
    Lattice *grid = new Lattice(DIM, (double)LENGTH, (double)LENGTH, periods, omega);

    //set initial state

    State *state = new State(grid);
    state->init_state(gauss_ini_state);
    Hamiltonian *hamiltonian = new Hamiltonian(grid, particle_mass, coupling_const, 0, 0, rot_coord_x, rot_coord_y, omega);
    hamiltonian->initialize_potential(parabolic_potential);
    Solver *solver = new Solver(grid, state, hamiltonian, delta_t, KERNEL_TYPE);

    if(grid->mpi_rank == 0) {
        cout << "\n* This source provides an example of the trotter-suzuki program.\n";
        cout << "* It calculates the time-evolution of a particle in a box\n";
        cout << "* with periodic boundary conditions, where the initial\n";
        cout << "* state is the following:\n";
        cout << "* \texp(i2M_PI / L (x + y))\n\n";
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
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return 0;
}
