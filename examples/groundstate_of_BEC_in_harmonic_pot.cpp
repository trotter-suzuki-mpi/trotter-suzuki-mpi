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

complex<double> gauss_ini_state(int m, int n, Lattice *grid) {
    double x = (m - grid->global_dim_x / 2.) * grid->delta_x;
    double y = (n - grid->global_dim_y / 2.) * grid->delta_x;
    double w = 0.01;
    return complex<double>(sqrt(w * double(PARTICLES_NUM) / M_PI) * exp(-(x * x + y * y) * 0.5 * w), 0.0);
}

double parabolic_potential(int m, int n, Lattice *grid) {
    double x = (m - grid->global_dim_x / 2.) * grid->delta_x;
    double y = (n - grid->global_dim_x / 2.) * grid->delta_x;
    double w_x = 1, w_y = 1. / sqrt(2); 
    return 0.5 * (w_x * w_x * x * x + w_y * w_y * y * y);
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
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif
    Lattice *grid = new Lattice(DIM, (double)LENGTH/DIM, (double)LENGTH/DIM, periods, omega);

    double *external_pot_real = new double[grid->dim_x * grid->dim_y];
    double *external_pot_imag = new double[grid->dim_x * grid->dim_y];

    //set and calculate evolution operator variables from hamiltonian
    double coupling_const = 4. * M_PI * double(SCATTER_LENGTH_2D);
    double delta_x = double(LENGTH)/double(DIM), delta_y = double(LENGTH)/double(DIM);
    double time_single_it = delta_t / 2.;	//second approx trotter-suzuki: time/2
    double h_a = cosh(time_single_it / (2. * particle_mass * delta_x * delta_y));
    double h_b = sinh(time_single_it / (2. * particle_mass * delta_x * delta_y));

    initialize_exp_potential(grid, external_pot_real, external_pot_imag, 
                             parabolic_potential, time_single_it, 
                             particle_mass, imag_time);
    //set initial state

    State *state = new State(grid);
    state->init_state(gauss_ini_state);
    Hamiltonian *hamiltonian = new Hamiltonian(grid, particle_mass, coupling_const, 0, 0, rot_coord_x, rot_coord_y, omega);

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
        trotter(grid, state, hamiltonian, h_a, h_b, 
                external_pot_real, external_pot_imag, delta_t, ITERATIONS, 
                KERNEL_TYPE, norm, imag_time);

        stamp(grid, state, 0, ITERATIONS, count_snap, dirnames.c_str());
    }
    if (grid->mpi_rank == 0 && verbose == true) {
        cout << "TROTTER " << DIM << "x" << DIM << " kernel:" << KERNEL_TYPE << " np:" << grid->mpi_procs << endl;
    }
    delete hamiltonian;
    delete state;
    delete grid;
    return 0;
}
