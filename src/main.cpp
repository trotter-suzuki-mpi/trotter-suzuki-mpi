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

#include <string.h>
#include <fstream>
#include <unistd.h>
#include <stdlib.h>
#include <iostream>
#include <complex>

#if HAVE_CONFIG_H
#include <config.h>
#endif
#include "common.h"
#include "trotter.h"
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#define DIM 640
#define ITERATIONS 1000
#define KERNEL_TYPE 0
#define SNAPSHOTS 0
#define N_PARTICLES 1
#define FILENAME_LENGTH 255

void print_usage() {
    std::cout << "Usage:\n" \
              "     trotter [OPTION] -n filename\n" \
              "Arguments:\n" \
              "     -a NUMBER     Parameter h_a of kinetic evolution operator (cosine part)\n"\
              "     -b NUMBER     Parameter h_b of kinetic evolution operator (sine part)\n"\
              "     -d NUMBER     Matrix dimension (default: " << DIM << ")\n" \
              "     -g            Imaginary time evolution to evolve towards the ground state\n" \
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
              "     -p STRING     Name of file that stores the potential operator (in coordinate representation)\n";
}

void process_command_line(int argc, char** argv, int *dim, int *iterations, int *snapshots, int *kernel_type, char *filename, double *h_a, double *h_b, char * pot_name, bool *imag_time, int *n_particles) {
    // Setting default values
    *dim = DIM;
    *iterations = ITERATIONS;
    *snapshots = SNAPSHOTS;
    *kernel_type = KERNEL_TYPE;
    *n_particles = N_PARTICLES;

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
                filename[i] = optarg[i];
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
            *n_particles = atoi(optarg);
            if (*n_particles <= 0) {
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
    int dim = 0, iterations = 0, snapshots = 0, kernel_type = 0, n_particles = 0;
    int periods[2] = {1, 1};
    const double particle_mass = 1.;
    char filename[FILENAME_LENGTH] = "";
    char pot_name[FILENAME_LENGTH] = "";
    bool verbose = true, imag_time = false;
    double h_a = .0, h_b = .0;

#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif
    process_command_line(argc, argv, &dim, &iterations, &snapshots, &kernel_type, filename, &h_a, &h_b, pot_name, &imag_time, &n_particles);

    int halo_x = (kernel_type == 2 ? 3 : 4);
    int halo_y = 4;
    int matrix_width = dim + periods[1] * 2 * halo_x;
    int matrix_height = dim + periods[0] * 2 * halo_y;
    
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
    
    for(int i = 0; i < n_particles; i++) {
        int read_offset = i * dim * dim;
        
        //set and calculate evolution operator variables from hamiltonian
        double time_single_it;
        double *external_pot_real = new double[tile_width*tile_height];
        double *external_pot_imag = new double[tile_width*tile_height];
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
        double *p_real = new double[tile_width*tile_height];
        double *p_imag = new double[tile_width*tile_height];
        std::complex<double> (*ini_state)(int x, int y, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y);
        ini_state = NULL;
        initialize_state(p_real, p_imag, filename, ini_state, tile_width, tile_height, matrix_width, matrix_height, start_x, start_y,
                         periods, coords, dims, halo_x, halo_y, read_offset);

        trotter(h_a, h_b, external_pot_real, external_pot_imag, p_real, p_imag, matrix_width, matrix_height, iterations, snapshots, kernel_type, periods, ".", verbose, imag_time, i + 1);
    }
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return 0;
}
