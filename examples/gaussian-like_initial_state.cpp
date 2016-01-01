/**
 * Distributed Trotter-Suzuki solver
 * Copyright (C) 2015 Luca Calderaro, 2012-2015 Peter Wittek,
 * 2010-2012 Carlos Bederián
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
#include <sys/stat.h>
#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include "trottersuzuki.h"

#define ITERATIONS 10
#define DIM 640
#define KERNEL_TYPE "cpu"
#define SNAPSHOTS 10

std::complex<double> gauss_state(int x, int y, Lattice *grid) {
    double s = 64.0;
    return std::complex<double>(exp(-(pow(x - 180.0, 2.0) + pow(y - 300.0, 2.0)) / (2.0 * pow(s, 2.0))), 0.0)
           * exp(std::complex<double>(0.0, 0.4 * (x + y - 480.0)));
}

int main(int argc, char** argv) {
    int periods[2] = {0, 0};
    double coupling_const = 0;
    double delta_x = 1, delta_y = 1;
    double delta_t = 0.08;
    int rot_coord_x = 320, rot_coord_y = 320;
    double omega = 0;
    bool verbose = true;
    double norm;
    bool imag_time = false;
    const double particle_mass = 1.;

#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif
    Lattice *grid = new Lattice(DIM, delta_x, delta_y, periods, omega);

    //set and calculate evolution operator variables from hamiltonian
    double *external_pot_real = new double[grid->dim_x * grid->dim_y];
    double *external_pot_imag = new double[grid->dim_x * grid->dim_y];
    double time_single_it = delta_t * particle_mass / 2.;	//second approx trotter-suzuki: time/2
    double h_a = cos(time_single_it / (2. * particle_mass));
    double h_b = sin(time_single_it / (2. * particle_mass));
    initialize_exp_potential(grid, external_pot_real, external_pot_imag, 
                             const_potential, time_single_it, 
                             particle_mass, imag_time);

    //set initial state
    State *state = new State(grid);
    state->init_state(gauss_state);
    Hamiltonian *hamiltonian = new Hamiltonian(grid, particle_mass, coupling_const, 0, 0, rot_coord_x, rot_coord_y, omega);
    if(grid->mpi_rank == 0) {
        std::cout << "\n* This source provides an example of the trotter-suzuki program.\n";
        std::cout << "* It calculates the time-evolution of a particle in a box, where the initial\n";
        std::cout << "* state is the following:\n";
        std::cout << "* \texp(-( (x - 180)² + (y - 300)² )/ 2s²)* exp(0.4j * (x + y - 480.))\n\n";
    }

    //set file output directory
    std::stringstream dirname;
    std::string dirnames;
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
        trotter(grid, state, hamiltonian, h_a, h_b, external_pot_real, external_pot_imag, delta_t, ITERATIONS, KERNEL_TYPE, norm, imag_time);
        stamp(grid, state, 0, ITERATIONS, count_snap, dirnames.c_str());
    }
    if (grid->mpi_coords[0] == 0 && grid->mpi_coords[1] == 0 && verbose == true) {
        std::cout << "TROTTER " << DIM << "x" << DIM << " kernel:" << KERNEL_TYPE << " np:" << grid->mpi_procs << std::endl;
    }
    delete hamiltonian;
    delete state;
    delete grid;
    return 0;
}
