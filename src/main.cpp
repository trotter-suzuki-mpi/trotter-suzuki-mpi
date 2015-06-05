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

#include <string>
#include <sstream>
#include <fstream>
#include <unistd.h>
#include <stdlib.h>
#include <iostream>
#include <complex>
#include <stdio.h>
#include "mpi.h"
#include "common.h"
#include "trotter.h"

#define DIM 640
#define ITERATIONS 1000
#define KERNEL_TYPE 0
#define SNAPSHOTS 0
#define PARTICLES_NUMBER 1

//external potential operator in coordinate representation
void potential_op_coord_representation(double *hamilt_pot, int dimx, int dimy, int halo_x, int halo_y, int *periods) {
    double constant = 0.;
    for(int i = 0; i < dimy; i++) {
        for(int j = 0; j < dimx; j++) {
            hamilt_pot[i * dimx + j] = constant;
        }
    }
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

//read potential form a file
void read_pot(double *hamilt_pot, int dimx, int dimy, char *file_name, int halo_x, int halo_y, int *periods) {

    std::ifstream input(file_name);

    int in_width = dimx - 2 * periods[1] * halo_x;
    int in_height = dimy - 2 * periods[0] * halo_y;
    double tmp;
    for(int i = 0, idy = periods[0] * halo_y ; i < in_height; i++, idy++) {
        for(int j = 0, idx = periods[1] * halo_x ; j < in_width; j++, idx++) {
            input >> tmp;
            hamilt_pot[idy * dimx + idx] = tmp;

            //Down band
            if(i < halo_y && periods[0] != 0) {
                hamilt_pot[(idy + in_height) * dimx + idx] = tmp;

                //Down right corner
                if(j < halo_x && periods[1] != 0)
                    hamilt_pot[(idy + in_height) * dimx + idx + in_width] = tmp;

                //Down left corner
                if(j >= in_width - halo_x && periods[1] != 0)
                    hamilt_pot[(idy + in_height) * dimx + idx - in_width] = tmp;
            }

            //Upper band
            if(i >= in_height - halo_y && periods[0] != 0) {
                hamilt_pot[(idy - in_height) * dimx + idx] = tmp;

                //Up right corner
                if(j < halo_x && periods[1] != 0)
                    hamilt_pot[(idy - in_height) * dimx + idx + in_width] = tmp;

                //Up left corner
                if(j >= in_width - halo_x && periods[1] != 0)
                    hamilt_pot[(idy - in_height) * dimx + idx - in_width] = tmp;
            }
            //Right band
            if(j < halo_x && periods[1] != 0)
                hamilt_pot[idy * dimx + idx + in_width] = tmp;

            //Left band
            if(j >= in_width - halo_x && periods[1] != 0)
                hamilt_pot[idy * dimx + idx - in_width] = tmp;
        }
    }
    input.close();
}

void read_initial_state(double *p_real, double *p_imag, int dimx, int dimy, char *file_name, int halo_x, int halo_y, int *periods, int Particles_number) {
    std::ifstream input(file_name);

    int in_width = dimx - 2 * periods[1] * halo_x;
    int in_height = dimy - 2 * periods[0] * halo_y;
    std::complex<double> tmp;
    for(int offset = 0; offset < Particles_number * dimx * dimy; offset += dimx * dimy) {
        for(int i = 0, idy = periods[0] * halo_y ; i < in_height; i++, idy++) {
            for(int j = 0, idx = periods[1] * halo_x ; j < in_width; j++, idx++) {
                input >> tmp;
                p_real[idy * dimx + idx + offset] = real(tmp);
                p_imag[idy * dimx + idx + offset] = imag(tmp);

                //Down band
                if(i < halo_y && periods[0] != 0) {
                    p_real[(idy + in_height) * dimx + idx + offset] = real(tmp);
                    p_imag[(idy + in_height) * dimx + idx + offset] = imag(tmp);
                    //Down right corner
                    if(j < halo_x && periods[1] != 0) {
                        p_real[(idy + in_height) * dimx + idx + in_width + offset] = real(tmp);
                        p_imag[(idy + in_height) * dimx + idx + in_width + offset] = imag(tmp);
                    }
                    //Down left corner
                    if(j >= in_width - halo_x && periods[1] != 0) {
                        p_real[(idy + in_height) * dimx + idx - in_width + offset] = real(tmp);
                        p_imag[(idy + in_height) * dimx + idx - in_width + offset] = imag(tmp);
                    }
                }

                //Upper band
                if(i >= in_height - halo_y && periods[0] != 0) {
                    p_real[(idy - in_height) * dimx + idx + offset] = real(tmp);
                    p_imag[(idy - in_height) * dimx + idx + offset] = imag(tmp);
                    //Up right corner
                    if(j < halo_x && periods[1] != 0) {
                        p_real[(idy - in_height) * dimx + idx + in_width + offset] = real(tmp);
                        p_imag[(idy - in_height) * dimx + idx + in_width + offset] = imag(tmp);
                    }
                    //Up left corner
                    if(j >= in_width - halo_x && periods[1] != 0) {
                        p_real[(idy - in_height) * dimx + idx - in_width + offset] = real(tmp);
                        p_imag[(idy - in_height) * dimx + idx - in_width + offset] = imag(tmp);
                    }
                }
                //Right band
                if(j < halo_x && periods[1] != 0) {
                    p_real[idy * dimx + idx + in_width + offset] = real(tmp);
                    p_imag[idy * dimx + idx + in_width + offset] = imag(tmp);
                }
                //Left band
                if(j >= in_width - halo_x && periods[1] != 0) {
                    p_real[idy * dimx + idx - in_width + offset] = real(tmp);
                    p_imag[idy * dimx + idx - in_width + offset] = imag(tmp);
                }
            }
        }
    }
    input.close();
}

void print_usage() {
    std::cout << "Usage:\n" \
              "     trotter [OPTION] -n file_name\n" \
              "Arguments:\n" \
              "     -g            Imaginary time evolution to evolve towards the ground state\n" \
              "     -d NUMBER     Matrix dimension (default: " << DIM << ")\n" \
              "     -i NUMBER     Number of iterations (default: " << ITERATIONS << ")\n" \
              "     -k NUMBER     Kernel type (default: " << KERNEL_TYPE << "): \n" \
              "                      0: CPU, cache-optimized\n" \
              "                      1: CPU, SSE and cache-optimized\n" \
              "                      2: GPU\n" \
              "                      3: Hybrid (experimental) \n" \
              "     -s NUMBER     Snapshots are taken at every NUMBER of iterations.\n" \
              "                   Zero means no snapshots. Default: " << SNAPSHOTS << ".\n"\
              "     -n STRING     Name of file that defines the initial state.\n"\
              "     -N NUMBER     Number of particles of the system.\n"\
              "     -a NUMBER     Parameter h_a of kinetic evolution operator\n"\
              "     -b NUMBER     Parameter h_b of kinetic evolution operator\n"\
              "     -p STRING     Name of file that stores the potential operator (in coordinate representation)\n";
}

void process_command_line(int argc, char** argv, int *dim, int *iterations, int *snapshots, int *kernel_type, char *file_name, double *h_a, double *h_b, char * pot_name, bool *imag_time, int *Particles_number) {
    // Setting default values
    *dim = DIM;
    *iterations = ITERATIONS;
    *snapshots = SNAPSHOTS;
    *kernel_type = KERNEL_TYPE;
    *Particles_number = PARTICLES_NUMBER;

    int c;
    bool file_supplied = false;
    int kinetic_par = 0;
    while ((c = getopt (argc, argv, "gd:hi:k:s:n:a:b:p:N:")) != -1) {
        switch (c) {
        case 'g':
            *imag_time = true;
            break;
        case 'd':
            *dim = atoi(optarg);
            if (*dim <= 0) {
                fprintf (stderr, "The argument of option -d should be a positive integer.\n");
                abort ();
            }
            break;
        case 'i':
            *iterations = atoi(optarg);
            if (*iterations <= 0) {
                fprintf (stderr, "The argument of option -i should be a positive integer.\n");
                abort ();
            }
            break;
        case 'h':
            print_usage();
            abort ();
            break;
        case 'k':
            *kernel_type = atoi(optarg);
            if (*kernel_type < 0 || *kernel_type > 3) {
                fprintf (stderr, "The argument of option -k should be a valid kernel.\n");
                abort ();
            }
            break;
        case 's':
            *snapshots = atoi(optarg);
            if (*snapshots <= 0) {
                fprintf (stderr, "The argument of option -s should be a positive integer.\n");
                abort ();
            }
            break;
        case 'n':
            for(size_t i = 0; i < strlen(optarg); i++)
                file_name[i] = optarg[i];
            file_supplied = true;
            break;
        case 'a':
            *h_a = atoi(optarg);
            kinetic_par++;
            break;
        case 'b':
            *h_b = atoi(optarg);
            kinetic_par++;
            break;
        case 'p':
            for(size_t i = 0; i < strlen(optarg); i++)
                pot_name[i] = optarg[i];
            break;
        case 'N':
            *Particles_number = atoi(optarg);
            if (*Particles_number <= 0) {
                fprintf (stderr, "The argument of option -N should be a positive integer.\n");
                abort ();
            }
            break;
        case '?':
            if (optopt == 'd' || optopt == 'i' || optopt == 'k' || optopt == 's') {
                fprintf (stderr, "Option -%c requires an argument.\n", optopt);
                print_usage();
                abort ();
            }
            else if (isprint (optopt)) {
                fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                print_usage();
                abort ();
            }
            else {
                fprintf (stderr, "Unknown option character `\\x%x'.\n",  optopt);
                print_usage();
                abort ();
            }
        default:
            abort ();
        }
    }
    if(!file_supplied) {
        fprintf (stderr, "Initial state file has not been supplied\n");
        print_usage();
        abort();
    }
    if(kinetic_par == 1) {
        std::cout << "Both the kinetic parameters should be provided.\n";
        abort ();
    }
}

int main(int argc, char** argv) {
    int dim = 0, iterations = 0, snapshots = 0, kernel_type = 0, Particles_number = 0;
    int periods[2] = {1, 1};
    char file_name[100];
    bool show_time_sim = true;
    bool imag_time = false;
    double h_a = 0.;
    double h_b = 0.;
    for(int i = 0; i < 100; i++)
        file_name[i] = '\0';
    char pot_name[100];
    for(int i = 0; i < 100; i++)
        pot_name[i] = '\0';

    MPI_Init(&argc, &argv);
    process_command_line(argc, argv, &dim, &iterations, &snapshots, &kernel_type, file_name, &h_a, &h_b, pot_name, &imag_time, &Particles_number);

    int halo_x = (kernel_type == 2 ? 3 : 4);
    int halo_y = 4;
    int matrix_width = dim + periods[1] * 2 * halo_x;
    int matrix_height = dim + periods[0] * 2 * halo_y;

    //set hamiltonian variables
    const double particle_mass = 1.;
    double *hamilt_pot = new double[matrix_width * matrix_height];
    if(pot_name[0] == '\0')
        potential_op_coord_representation(hamilt_pot, matrix_width, matrix_height, halo_x, halo_y, periods);	//set potential operator
    else
        read_pot(hamilt_pot, matrix_width, matrix_height, pot_name, halo_x, halo_y, periods);	//set potential operator from file

    //set and calculate evolution operator variables from hamiltonian
    double *external_pot_real = new double[matrix_width * matrix_height];
    double *external_pot_imag = new double[matrix_width * matrix_height];
    if(imag_time) {
        double constant = 6.;
        const double time_single_it = 8 * particle_mass / 2.;	//second approx trotter-suzuki: time/2
        init_pot_evolution_op(hamilt_pot, external_pot_real, external_pot_imag, matrix_width, matrix_height, particle_mass, time_single_it, true);	//calculate potential part of evolution operator
        if(h_a == 0. && h_b == 0.) {
            h_a = cosh(time_single_it / (2. * particle_mass)) / constant;
            h_b = sinh(time_single_it / (2. * particle_mass)) / constant;
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
    double *p_real = new double[Particles_number * matrix_width * matrix_height];
    double *p_imag = new double[Particles_number * matrix_width * matrix_height];
    read_initial_state(p_real, p_imag, matrix_width, matrix_height, file_name, halo_x, halo_y, periods, Particles_number);

    for(int i = 0; i < Particles_number; i++)
        trotter(h_a, h_b, external_pot_real, external_pot_imag, &p_real[i * matrix_width * matrix_height], &p_imag[i * matrix_width * matrix_height], matrix_width, matrix_height, iterations, snapshots, kernel_type, periods, argc, argv, ".", show_time_sim, imag_time, i + 1);

    MPI_Finalize();
    return 0;
}
