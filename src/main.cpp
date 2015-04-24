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


#include <fstream>
#include <unistd.h>
#include <stdlib.h>
#include <iostream>
#include <complex>
#include "trotter.h"

#define DIM 640
#define ITERATIONS 1000
#define KERNEL_TYPE 0
#define SNAPSHOTS 0

void print_usage() {
    std::cout << "Usage:\n" \
              "     trotter [OPTION] -n file_name\n" \
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
              "     -n STRING     Name of file that defines the initial state.\n";
}

void read_initial_state(float *p_real, float *p_imag, int dim, char *file_name){
	std::ifstream input(file_name);
	
	std::complex<float> tmp;
	for(int i=0; i<dim; i++){
		for(int j=0; j<dim; j++){
			input >> tmp;
			p_real[i*dim+j] = real(tmp);
			p_imag[i*dim+j] = imag(tmp);
		}
	}
	input.close();
}

void process_command_line(int argc, char** argv, int *dim, int *iterations, int *snapshots, int *kernel_type, char *file_name) {
    // Setting default values
    *dim = DIM;
    *iterations = ITERATIONS;
    *snapshots = SNAPSHOTS;
    *kernel_type = KERNEL_TYPE;

    int c;
    bool file_supplied=false;
    while ((c = getopt (argc, argv, "d:hi:k:s:n:")) != -1) {
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
        case 'n':
			for(int i=0; i<40; i++)
				file_name[i] = optarg[i];
			file_supplied=true;
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
    if(!file_supplied){
		fprintf (stderr, "Initial state file has not been supplied\n");
		print_usage();
		abort();
	}
}

int main(int argc, char** argv) {
    int dim = 0, iterations = 0, snapshots = 0, kernel_type = 0;
    char file_name[40];
	float * p_real;
	float * p_imag;
   
    process_command_line(argc, argv, &dim, &iterations, &snapshots, &kernel_type, file_name);
    
    p_real = new float[dim*dim];
	p_imag = new float[dim*dim];
	
	read_initial_state(p_real, p_imag, dim, file_name);
    trotter(p_real, p_imag, dim, dim, iterations, snapshots, kernel_type, argc, argv);

    return 0;
}
