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
 * 		exp(i2M_PI / L (x + y))
 */

#include <sys/types.h>
#include <sys/stat.h>
#include <string>
#include <sstream>

#include <fstream>
#include <unistd.h>
#include <stdlib.h>
#include <iostream>
#include <complex>
#include "mpi.h"
#include "common.h"
#include "trotter.h"

#define DIM 640
#define ITERATIONS 1000
#define KERNEL_TYPE 0
#define SNAPSHOTS 100

//set external potential operator in coordinate representation
void potential_op_coord_representation(double *hamilt_pot, int dimx, int dimy, int halo_x, int halo_y, int *periods) {
    double constant = 0.;
    for(int i = 0; i < dimy; i++) {
        for(int j = 0; j < dimx; j++) {
            hamilt_pot[i * dimx + j] = constant;
        }
    }
}

//set initial state
void init_state(double *p_real, double *p_imag, int dimx, int dimy, int halo_x, int halo_y, int *periods) {
    double L_x = dimx - periods[1] * 2 * halo_x;

    for (int y = 1; y <= dimy; y++) {
        for (int x = 1; x <= dimx; x++) {
            std::complex<double> tmp = exp(std::complex<double>(0. , 2. * 3.14159 / L_x * (x - periods[1] * halo_x))) +
                                       exp(std::complex<double>(0. , 10. * 2. * 3.14159 / L_x * (x - periods[1] * halo_x)));

            p_real[y * dimx + x] = real(tmp);
            p_imag[y * dimx + x] = imag(tmp);
        }
    }
}

//calculate potential part of evolution operator
void init_pot_evolution_op(double * hamilt_pot, double * external_pot_real, double * external_pot_imag, int dimx, int dimy, double particle_mass, double time_single_it) {
    double CONST_1 = -1. * time_single_it;
    double CONST_2 = 2. * time_single_it / particle_mass;		//CONST_2: discretization of momentum operator and the only effect is to produce a scalar operator, so it could be omitted

    std::complex<double> tmp;
    for(int i = 0; i < dimy; i++) {
        for(int j = 0; j < dimx; j++) {
            tmp = exp(std::complex<double> (CONST_1 * hamilt_pot[i * dimx + j] + CONST_2, 0.));
            external_pot_real[i * dimx + j] = real(tmp);
            external_pot_imag[i * dimx + j] = imag(tmp);
        }
    }
}

int main(int argc, char** argv) {
    int dim = DIM, iterations = ITERATIONS, snapshots = SNAPSHOTS, kernel_type = KERNEL_TYPE;
    int periods[2] = {1, 1};
    bool verbose = true;
    int halo_x = (kernel_type == 2 ? 3 : 4);
    int halo_y = 4;
    int matrix_width = dim + periods[1] * 2 * halo_x;
    int matrix_height = dim + periods[0] * 2 * halo_y;

    //set hamiltonian variables
    const double particle_mass = 1.;
    double *hamilt_pot = new double[matrix_width * matrix_height];

    potential_op_coord_representation(hamilt_pot, matrix_width, matrix_height, halo_x, halo_y, periods);	//set potential operator

    //set and calculate evolution operator variables from hamiltonian
    const double time_single_it = 8 * particle_mass / 2.;	//second approx trotter-suzuki: time/2
    double *external_pot_real = new double[matrix_width * matrix_height];
    double *external_pot_imag = new double[matrix_width * matrix_height];
    init_pot_evolution_op(hamilt_pot, external_pot_real, external_pot_imag, matrix_width, matrix_height, particle_mass, time_single_it);	//calculate potential part of evolution operator
    double constant = 6.;
    double h_a = cosh(time_single_it / (2. * particle_mass)) / constant;
    double h_b = sinh(time_single_it / (2. * particle_mass)) / constant;

    //set initial state
    double *p_real = new double[matrix_width * matrix_height];
    double *p_imag = new double[matrix_width * matrix_height];
    init_state(p_real, p_imag, matrix_width, matrix_height, halo_x, halo_y, periods);

    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0) {
        std::cout << "\n* This source provides an example of the trotter-suzuki program.\n";
        std::cout << "* It calculates the imaginary time-evolution of a free particle in a box\n";
        std::cout << "* with periodic boundary conditions, where the initial\n";
        std::cout << "* state is the following:\n";
        std::cout << "* \texp(i2M_PI / L * x) + exp(i20M_PI / L * x)\n\n";
        std::cout << "* The state will reach the eigenfunction of the Hamiltonian with the lowest\n";
        std::cout << "* eigenvalue:   exp(i2M_PI / L * x)\n\n";
    }

    //set file output directory
    std::stringstream dirname;
    std::string dirnames;
    if(snapshots) {
        int status;

        dirname.str("");
        dirname << "IMAG_EVO_D" << dim << "_I" << iterations << "_S" << snapshots << "";
        dirnames = dirname.str();

        status = mkdir(dirnames.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

        if(status != 0 && status != -1)
            dirnames = ".";
    }
    else
        dirnames = ".";

    trotter(h_a, h_b, external_pot_real, external_pot_imag, p_real, p_imag, matrix_width, matrix_height, iterations, snapshots, kernel_type, periods, dirnames.c_str(), verbose, true);

    MPI_Finalize();
    return 0;
}
