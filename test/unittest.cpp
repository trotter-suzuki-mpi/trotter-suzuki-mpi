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

#include <cppunit/CompilerOutputter.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <complex>
#include <sys/stat.h>

#include "trotter.h"
#include "kernel.h"
#include "common.h"
#ifdef HAVE_MPI
#include <mpi.h>
#endif

#define DIM 640
#define ITERATIONS 100
#define KERNEL_TYPE 0
#define SNAPSHOTS 10

struct MAGIC_NUMBER {
    double threshold_E, threshold_P;
    double expected_E;
    double expected_Px;
    double expected_Py;
    MAGIC_NUMBER();
};

MAGIC_NUMBER::MAGIC_NUMBER() : threshold_E(3), threshold_P(3),
    expected_E((2. * M_PI / DIM) * (2. * M_PI / DIM)), expected_Px(0), expected_Py(0) {}

std::complex<double> sinus_state(int x, int y, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y) {
    double L_x = matrix_width - periods[1] * 2 * halo_x;
    double L_y = matrix_height - periods[0] * 2 * halo_y;

    return std::complex<double> (sin(2 * 3.14159 / L_x * (x - periods[1] * halo_x)) * sin(2 * 3.14159 / L_y * (y - periods[0] * halo_y)), 0.0);
}

int main(int argc, char** argv) {
    int dim = DIM, iterations = ITERATIONS, snapshots = SNAPSHOTS, kernel_type = KERNEL_TYPE;
    int periods[2] = {1, 1};
    char file_name[1] = "";
    char pot_name[1] = "";
    const double particle_mass = 1.;
    bool show_time_sim = false;
    bool imag_time = false;
    double h_a = 0.;
    double h_b = 0.;
    int time, tot_time = 0;
    
    // Get the top level suite from the registry
    CppUnit::Test *suite = CppUnit::TestFactoryRegistry::getRegistry().makeTest();
    // Adds the test to the list of test to run
    CppUnit::TextUi::TestRunner runner;
    runner.addTest( suite );
    // Change the default outputter to a compiler error format outputter
    runner.setOutputter( new CppUnit::CompilerOutputter( &runner.result(), std::cerr ) );
    // Run the tests.
    bool wasSucessful = runner.run();
    // Return error code 1 if the one of test failed.
    if(!wasSucessful)
        return 1;

    int halo_x = (kernel_type == 2 ? 3 : 4);
    int halo_y = 4;
    int matrix_width = dim + periods[1] * 2 * halo_x;
    int matrix_height = dim + periods[0] * 2 * halo_y;

#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif
    //define the topology
    int coords[2], dims[2] = {0, 0};
    int rank;
    int nProcs;
#ifdef HAVE_MPI
    MPI_Comm cartcomm;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    MPI_Dims_create(nProcs, 2, dims);  //partition all the processes (the size of MPI_COMM_WORLD's group) into an 2-dimensional topology
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cartcomm);
    MPI_Comm_rank(cartcomm, &rank);
    MPI_Cart_coords(cartcomm, rank, 2, coords);
#else
    nProcs = 1;
    rank = 0;
    dims[0] = dims[1] = 1;
    coords[0] = coords[1] = 0;
#endif

    //set dimension of tiles and offsets
    int start_x, end_x, inner_start_x, inner_end_x,
        start_y, end_y, inner_start_y, inner_end_y;
    calculate_borders(coords[1], dims[1], &start_x, &end_x, &inner_start_x, &inner_end_x, matrix_width - 2 * periods[1]*halo_x, halo_x, periods[1]);
    calculate_borders(coords[0], dims[0], &start_y, &end_y, &inner_start_y, &inner_end_y, matrix_height - 2 * periods[0]*halo_y, halo_y, periods[0]);
    int tile_width = end_x - start_x;
    int tile_height = end_y - start_y;

    //set and calculate evolution operator variables from hamiltonian
    double time_single_it;
    double *external_pot_real = new double[tile_width * tile_height];
    double *external_pot_imag = new double[tile_width * tile_height];
    double (*hamiltonian_pot)(int x, int y, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y);
    hamiltonian_pot = const_potential;

    if(imag_time) {
        double constant = 6.;
        time_single_it = 8 * particle_mass / 2.;	//second approx trotter-suzuki: time/2
        if(h_a == 0. && h_b == 0.) {
            h_a = cosh(time_single_it / (2. * particle_mass)) / constant;
            h_b = sinh(time_single_it / (2. * particle_mass)) / constant;
        }
    }
    else {
        time_single_it = 0.08 * particle_mass / 2.;	//second approx trotter-suzuki: time/2
        if(h_a == 0. && h_b == 0.) {
            h_a = cos(time_single_it / (2. * particle_mass));
            h_b = sin(time_single_it / (2. * particle_mass));
        }
    }
    initialize_exp_potential(external_pot_real, external_pot_imag, pot_name, hamiltonian_pot, tile_width, tile_height, matrix_width, matrix_height,
                             start_x, start_y, periods, coords, dims, halo_x, halo_y, time_single_it, particle_mass, imag_time);

    //set initial state
    double *p_real = new double[tile_width * tile_height];
    double *p_imag = new double[tile_width * tile_height];
    std::complex<double> (*ini_state)(int x, int y, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y);
    ini_state = sinus_state;
    initialize_state(p_real, p_imag, file_name, ini_state, tile_width, tile_height, matrix_width, matrix_height, start_x, start_y,
                     periods, coords, dims, halo_x, halo_y);

    //set file output directory
    std::stringstream filename;
    std::string filenames;
    if(snapshots) {
        int status = 0;

        filename.str("");
        filename << "D" << dim << "_I" << iterations << "_S" << snapshots << "";
        filenames = filename.str();

        status = mkdir(filenames.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

        if(status != 0 && status != -1)
            filenames = ".";
    }
    else
        filenames = ".";

    for(int count_snap = 0; count_snap <= snapshots; count_snap++) {
        stamp(p_real, p_imag, matrix_width, matrix_height, halo_x, halo_y, start_x, inner_start_x, inner_end_x,
              start_y, inner_start_y, inner_end_y, dims, coords, periods, 
              0, iterations, count_snap, filenames.c_str()
#ifdef HAVE_MPI
              , cartcomm
#endif
              );                  
              
        if(count_snap != snapshots) {
            trotter(h_a, h_b, external_pot_real, external_pot_imag, p_real, p_imag, matrix_width, matrix_height, iterations, kernel_type, periods, imag_time, &time);
            tot_time += time;
        }
    }
        
    if(rank == 0) {
        MAGIC_NUMBER th_values;
        energy_momentum_statistics sample;
        double hamilt_pot[dim * dim];

        initialize_potential(hamilt_pot, hamiltonian_pot, dim, dim, periods, halo_x, halo_y);
        expect_values(dim, iterations, snapshots, hamilt_pot, particle_mass, filenames.c_str(), periods, halo_x, halo_y, &sample);

        if(std::abs(sample.mean_E - th_values.expected_E) / sample.var_E < th_values.threshold_E)
            std::cout << "Energy -> OK\tsigma: " << std::abs(sample.mean_E - th_values.expected_E) / sample.var_E << std::endl;
        else
            std::cout << "Energy value is not the one theoretically expected: sigma " << std::abs(sample.mean_E - th_values.expected_E) / sample.var_E << std::endl;
        if(std::abs(sample.mean_Px - th_values.expected_Px) / sample.var_Px < th_values.threshold_P)
            std::cout << "Momentum Px -> OK\tsigma: " << std::abs(sample.mean_Px - th_values.expected_Px) / sample.var_Px << std::endl;
        else
            std::cout << "Momentum Px value is not the one theoretically expected: sigma " << std::abs(sample.mean_Px - th_values.expected_Px) / sample.var_Px << std::endl;
        if(std::abs(sample.mean_Py - th_values.expected_Py) / sample.var_Py < th_values.threshold_P)
            std::cout << "Momentum Py -> OK\tsigma: " << std::abs(sample.mean_Py - th_values.expected_Py) / sample.var_Py << std::endl;
        else
            std::cout << "Momentum Py value is not the one theoretically expected: sigma " << std::abs(sample.mean_Py - th_values.expected_Py) / sample.var_Py << std::endl;
    }

#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return 0;
}
