/**
 * Distributed Trotter-Suzuki solver
 * Copyright (C) 2012 Peter Wittek, 2010-2012 Carlos Bederián
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
#define SNAPSHOTS 0

//external potential operator in coordinate representation
void potential_op_coord_representation(float *hamilt_pot, int dimx, int dimy, int halo_x, int halo_y, int *periods) {
    float constant = 0.;
    for(int i = 0; i < dimy; i++) {
        for(int j = 0; j < dimx; j++) {
            hamilt_pot[i * dimx + j] = constant;
        }
    }
}

void init_state(float *p_real, float *p_imag, int dimx, int dimy, int halo_x, int halo_y, int *periods) {
    double s = 64.0; // FIXME: y esto?
    double L_x = dimx - periods[1] * 2 * halo_x;
    double L_y = dimy - periods[0] * 2 * halo_y;
    double n_x = 1., n_y = 1.;

    for (int y = 1; y <= dimy; y++) {
        for (int x = 1; x <= dimx; x++) {
            std::complex<float> tmp = exp(std::complex<float>(0. , 2 * 3.14159 / L_x * (x - periods[1]*halo_x) + 2 * 3.14159 / L_y * (y - periods[0]*halo_y) ));

            p_real[y * dimx + x] = real(tmp);
            p_imag[y * dimx + x] = imag(tmp);
        }
    }
}

//calculate potential part of evolution operator
void init_pot_evolution_op(float * hamilt_pot, float * external_pot_real, float * external_pot_imag, int dimx, int dimy, double particle_mass, double time_single_it ) {
    float CONST_1 = -1. * time_single_it;
    float CONST_2 = 2. * time_single_it / particle_mass;		//CONST_2: discretization of momentum operator and the only effect is to produce a scalar operator, so it could be omitted

    std::complex<float> tmp;
    for(int i = 0; i < dimy; i++) {
        for(int j = 0; j < dimx; j++) {
            tmp = exp(std::complex<float> (0., CONST_1 * hamilt_pot[i * dimx + j] + CONST_2));
            external_pot_real[i * dimx + j] = real(tmp);
            external_pot_imag[i * dimx + j] = imag(tmp);
        }
    }
}

//read potential form a file
void read_pot(float *hamilt_pot, int dimx, int dimy, char *file_name, int halo_x, int halo_y, int *periods) {
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

void print_usage() {
    std::cout << "\nSimulate the evolution of a quantum particle in a box with periodic boundary conditions.\n"\
    	      "Initial wave function:\n"\
              "  sin(2M_PI / d * x) * sin(2M_PI / d * y) \n"\
              "Usage:\n" \
              "     trotter [OPTION]\n" \
              "Arguments:\n" \
              "     -d NUMBER     Matrix dimension (default: " << DIM << ")\n" \
              "     -i NUMBER     Number of iterations (default: " << ITERATIONS << ")\n" \
              "     -k NUMBER     Kernel type (default: " << KERNEL_TYPE << "): \n" \
              "                      0: CPU, cache-optimized\n" \
              "                      1: CPU, SSE and cache-optimized\n" \
              "                      2: GPU\n" \
              "                      3: Hybrid (experimental) \n" \
              "     -s NUMBER     Snapshots are taken at every NUMBER of iterations.\n" \
              "                   Zero means no snapshots. Default: " << SNAPSHOTS << ".\n"\
              "     -a NUMBER     Parameter h_a of kinetic evolution operator\n"\
              "     -b NUMBER     Parameter h_b of kinetic evolution operator\n"\
              "     -p STRING     Name of file that stores the potential operator (in coordinate representation)\n";
}

void process_command_line(int argc, char** argv, int *dim, int *iterations, int *snapshots, int *kernel_type, double *h_a, double *h_b, char * pot_name) {
    // Setting default values
    *dim = DIM;
    *iterations = ITERATIONS;
    *snapshots = SNAPSHOTS;
    *kernel_type = KERNEL_TYPE;

    int c;
    int kinetic_par = 0;
    while ((c = getopt (argc, argv, "d:hi:k:s:a:b:p:")) != -1) {
        switch (c) {
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
        case 'a':
            *h_a = atoi(optarg);
            kinetic_par++;
            break;
        case 'b':
            *h_b = atoi(optarg);
            kinetic_par++;
            break;
        case 'p':
            for(int i = 0; i < 100; i++)
                pot_name[i] = optarg[i];
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
    
    if(kinetic_par == 1) {
		std::cout << "Both the kinetic parameters should be provided.\n";
		abort ();
	}
}

int main(int argc, char** argv) {
    int dim = 0, iterations = 0, snapshots = 0, kernel_type = 0;
    int periods[2] = {1, 1};
    bool test = false;
    double h_a = 0.;
    double h_b = 0.;
    char pot_name[100];
    for(int i = 0; i < 100; i++)
		pot_name[i] = '\0';

    process_command_line(argc, argv, &dim, &iterations, &snapshots, &kernel_type, &h_a, &h_b, pot_name);

    int halo_x = (kernel_type == 2 ? 3 : 4);
    int halo_y = 4;
    int matrix_width = dim + periods[1] * 2 * halo_x;
    int matrix_height = dim + periods[0] * 2 * halo_y;

    //set hamiltonian variables
    const double particle_mass = 1.;
    float *hamilt_pot = new float[matrix_width * matrix_height];
    if(pot_name[0] == '\0')
		potential_op_coord_representation(hamilt_pot, matrix_width, matrix_height, halo_x, halo_y, periods);	//set potential operator
	else
		read_pot(hamilt_pot, matrix_width, matrix_height, pot_name, halo_x, halo_y, periods);	//set potential operator from file

    //set and calculate evolution operator variables from hamiltonian
    const double time_single_it = 0.08 * particle_mass / 2.;	//second approx trotter-suzuki: time/2
    float *external_pot_real = new float[matrix_width * matrix_height];
    float *external_pot_imag = new float[matrix_width * matrix_height];
    init_pot_evolution_op(hamilt_pot, external_pot_real, external_pot_imag, matrix_width, matrix_height, particle_mass, time_single_it);	//calculate potential part of evolution operator
    if(h_a == 0. && h_b == 0.) { 
		h_a = cos(time_single_it / (2. * particle_mass));
		h_b = sin(time_single_it / (2. * particle_mass));
	}

    //set initial state
    float *p_real = new float[matrix_width * matrix_height];
    float *p_imag = new float[matrix_width * matrix_height];
    init_state(p_real, p_imag, matrix_width, matrix_height, halo_x, halo_y, periods);

    //set file output directory
    std::stringstream dirname;
    std::string dirnames;
    if(snapshots) {
        int status;

        dirname.str("");
        dirname << "D" << dim << "_I" << iterations << "_S" << snapshots << "";
        dirnames = dirname.str();

        status = mkdir(dirnames.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

        if(status != 0 && status != -1)
            dirnames = "./";
    }
    else
        dirnames = "./";

    procs_topology var = trotter(h_a, h_b, external_pot_real, external_pot_imag, p_real, p_imag, matrix_width, matrix_height, iterations, snapshots, kernel_type, periods, argc, argv, dirnames.c_str(), test);
	
	if(var.rank == 0 && snapshots != 0) {
		int N_files = (int)ceil(double(iterations) / double(snapshots));
		std::complex<float> psi[dim*dim];
		int N_name[N_files];
		N_name[0] = 0;
		for(int i = 1; i < N_files; i++) {
			N_name[i] = N_name[i - 1] + snapshots;
		}
		
		std::stringstream filename;
		std::string filenames;
		for(int i = 0; i < N_files; i++) {
			stick_files(N_files, N_name[i], psi, dirnames.c_str(), var, dim, periods, halo_x, halo_y);
			
			for(int idy = 0; idy < var.dimsy; idy++) {
				for(int idx = 0; idx < var.dimsx; idx++) {
					filename.str("");
					filename << dirnames << "/" << N_name[i] << "-iter-" << idx << "-" << idy << "-comp.dat";
					filenames = filename.str();
					remove(filenames.c_str());
					filename.str("");
					filename << dirnames << "/" << N_name[i] << "-iter-" << idx << "-" << idy << "-real.dat";
					filenames = filename.str();
					remove(filenames.c_str());
				}
			}
		}
	}
	
    return 0;
}
