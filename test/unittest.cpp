/**
 * Distributed Trotter-Suzuki solver
 * Copyright (C) 2012 Peter Wittek, 2010-2012 Carlos Bederián, 2015 Luca Calderaro
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

#include "trotter.h"
#include "common.h"

#define DIM 640
#define ITERATIONS 1000
#define KERNEL_TYPE 0
#define SNAPSHOTS 100

struct MAGIC_NUMBER {
    double threshold_E, threshold_P;
    double expected_E;
    double expected_Px;
    double expected_Py;
    MAGIC_NUMBER();
};

MAGIC_NUMBER::MAGIC_NUMBER() : threshold_E(3), threshold_P(2),
    expected_E((2. * M_PI / DIM) * (2. * M_PI / DIM)), expected_Px(0), expected_Py(0) {}

//external potential operator in coordinate representation
void potential_op_coord_representation(double *hamilt_pot, int dimx, int dimy, int halo_x, int halo_y, int *periods) {
    double constant = 0.;
    for(int i = 0; i < dimy; i++) {
        for(int j = 0; j < dimx; j++) {
            hamilt_pot[i * dimx + j] = constant;
        }
    }
}

void init_state(double *p_real, double *p_imag, int dimx, int dimy, int halo_x, int halo_y, int *periods) {
    double s = 64.0; // FIXME: y esto?
    double L_x = dimx - periods[1] * 2 * halo_x;
    double L_y = dimy - periods[0] * 2 * halo_y;
    double n_x = 1., n_y = 1.;

    for (int y = 1; y <= dimy; y++) {
        for (int x = 1; x <= dimx; x++) {
            //std::complex<double> tmp = std::complex<double>(exp(-(pow(x - 180.0, 2.0) + pow(y - 300.0, 2.0)) / (2.0 * pow(s, 2.0))), 0.0)
            //                      * exp(std::complex<double>(0.0, 0.4 * (x + y - 480.0)));

            //std::complex<double>  tmp = std::complex<double> (sin(2 * 3.14159 / L_x * (x - periods[1] * halo_x)) * sin(2 * 3.14159 / L_y * (y - periods[0] * halo_y)), 0.0);

            std::complex<double> tmp = exp(std::complex<double>(0. , 2 * 3.14159 / L_x * (x - periods[1]*halo_x))) + exp(std::complex<double>(0., 20 * 3.14159 / L_x * (x - periods[1]*halo_x) ));

            p_real[y * dimx + x] = real(tmp);
            p_imag[y * dimx + x] = imag(tmp);
        }
    }
}

void read_initial_state(double *p_real, double *p_imag, int dimx, int dimy, char *file_name, int halo_x, int halo_y, int *periods) {
    std::ifstream input(file_name);

    int in_width = dimx - 2 * periods[1] * halo_x;
    int in_height = dimy - 2 * periods[0] * halo_y;
    std::complex<double> tmp;
    for(int i = 0, idy = periods[0] * halo_y ; i < in_height; i++, idy++) {
        for(int j = 0, idx = periods[1] * halo_x ; j < in_width; j++, idx++) {
            input >> tmp;
            p_real[idy * dimx + idx] = real(tmp);
            p_imag[idy * dimx + idx] = imag(tmp);

            //Down band
            if(i < halo_y && periods[0] != 0) {
                p_real[(idy + in_height) * dimx + idx] = real(tmp);
                p_imag[(idy + in_height) * dimx + idx] = imag(tmp);
                //Down right corner
                if(j < halo_x && periods[1] != 0) {
                    p_real[(idy + in_height) * dimx + idx + in_width] = real(tmp);
                    p_imag[(idy + in_height) * dimx + idx + in_width] = imag(tmp);
                }
                //Down left corner
                if(j >= in_width - halo_x && periods[1] != 0) {
                    p_real[(idy + in_height) * dimx + idx - in_width] = real(tmp);
                    p_imag[(idy + in_height) * dimx + idx - in_width] = imag(tmp);
                }
            }

            //Upper band
            if(i >= in_height - halo_y && periods[0] != 0) {
                p_real[(idy - in_height) * dimx + idx] = real(tmp);
                p_imag[(idy - in_height) * dimx + idx] = imag(tmp);
                //Up right corner
                if(j < halo_x && periods[1] != 0) {
                    p_real[(idy - in_height) * dimx + idx + in_width] = real(tmp);
                    p_imag[(idy - in_height) * dimx + idx + in_width] = imag(tmp);
                }
                //Up left corner
                if(j >= in_width - halo_x && periods[1] != 0) {
                    p_real[(idy - in_height) * dimx + idx - in_width] = real(tmp);
                    p_imag[(idy - in_height) * dimx + idx - in_width] = imag(tmp);
                }
            }
            //Right band
            if(j < halo_x && periods[1] != 0) {
                p_real[idy * dimx + idx + in_width] = real(tmp);
                p_imag[idy * dimx + idx + in_width] = imag(tmp);
            }
            //Left band
            if(j >= in_width - halo_x && periods[1] != 0) {
                p_real[idy * dimx + idx - in_width] = real(tmp);
                p_imag[idy * dimx + idx - in_width] = imag(tmp);
            }
        }
    }
    input.close();
}

//calculate potential part of evolution operator
void init_pot_evolution_op(double * hamilt_pot, double * external_pot_real, double * external_pot_imag, int dimx, int dimy, double particle_mass, double time_single_it, bool imag_time) {
    double order_approx = 2.;
    double CONST_1 = -1. * time_single_it * order_approx;
    double CONST_2 = 2. * time_single_it / particle_mass * order_approx;		//CONST_2: discretization of momentum operator and the only effect is to produce a scalar operator, so it could be omitted

    std::complex<double> tmp;
    for(int i = 0; i < dimy; i++) {
        for(int j = 0; j < dimx; j++) {
            if(imag_time)
                tmp = exp(std::complex<double> (CONST_1 * hamilt_pot[i * dimx + j] , CONST_2));
            else
                tmp = exp(std::complex<double> (0., CONST_1 * hamilt_pot[i * dimx + j] + CONST_2));
            external_pot_real[i * dimx + j] = real(tmp);
            external_pot_imag[i * dimx + j] = imag(tmp);
        }
    }
}

int main(int argc, char** argv) {
    int dim = DIM, iterations = ITERATIONS, snapshots = SNAPSHOTS, kernel_type = KERNEL_TYPE;
    int periods[2] = {1, 1};
    char file_name[100];
    file_name[0] = '\0';
    bool show_time_sim = false;
    bool imag_time = true;
    double h_a = 0.;
    double h_b = 0.;

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

    //set hamiltonian variables
    const double particle_mass = 1.;
    double *hamilt_pot = new double[matrix_width * matrix_height];
    potential_op_coord_representation(hamilt_pot, matrix_width, matrix_height, halo_x, halo_y, periods);	//set potential operator

    //set and calculate evolution operator variables from hamiltonian
    //const double time_single_it = 0.08 * particle_mass / 2.;	//second approx trotter-suzuki: time/2
    double *external_pot_real = new double[matrix_width * matrix_height];
    double *external_pot_imag = new double[matrix_width * matrix_height];
    //init_pot_evolution_op(hamilt_pot, external_pot_real, external_pot_imag, matrix_width, matrix_height, particle_mass, time_single_it);	//calculate potential part of evolution operator
    //static const double h_a = cos(time_single_it / (2. * particle_mass));
    //static const double h_b = sin(time_single_it / (2. * particle_mass));
    
    if(imag_time) {
        const double time_single_it = 8 * particle_mass / 2.;	//second approx trotter-suzuki: time/2
        init_pot_evolution_op(hamilt_pot, external_pot_real, external_pot_imag, matrix_width, matrix_height, particle_mass, time_single_it, true);	//calculate potential part of evolution operator
        if(h_a == 0. && h_b == 0.) {
            h_a = cosh(time_single_it / (2. * particle_mass));
            h_b = sinh(time_single_it / (2. * particle_mass));
        }
    }
    else {
        const double time_single_it = 0.08 * particle_mass / 2.;	//second approx trotter-suzuki: time/2
        init_pot_evolution_op(hamilt_pot, external_pot_real, external_pot_imag, matrix_width, matrix_height, particle_mass, time_single_it, false);	//calculate potential part of evolution operator
        if(h_a == 0. && h_b == 0.) {
            h_a = cos(time_single_it / (2. * particle_mass));
            h_b = sin(time_single_it / (2. * particle_mass));
        }
    }
    
    //set initial state
    double *p_real = new double[matrix_width * matrix_height];
    double *p_imag = new double[matrix_width * matrix_height];
    if(file_name[0] == '\0')
        init_state(p_real, p_imag, matrix_width, matrix_height, halo_x, halo_y, periods);
    else
        read_initial_state(p_real, p_imag, matrix_width, matrix_height, file_name, halo_x, halo_y, periods);

    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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
            filenames = "./";
    }
    else
        filenames = "./";

    trotter(h_a, h_b, external_pot_real, external_pot_imag, p_real, p_imag, matrix_width, matrix_height, iterations, snapshots, kernel_type, periods, argc, argv, filenames.c_str(), show_time_sim, imag_time, 1);

    if(rank == 0) {
        MAGIC_NUMBER th_values;
        energy_momentum_statistics sample;
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

    delete[] hamilt_pot;

    MPI_Finalize();
    return 0;
}
