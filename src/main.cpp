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

#include <stdlib.h>
#include <iostream>
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#ifdef WIN32
#include "unistd.h"
#include <windows.h>
#else
#include <sys/time.h>
#include <unistd.h>
#endif
#if HAVE_CONFIG_H
#include "config.h"
#endif
#include "trottersuzuki.h"

#define DIM 640
#define EDGE_LENGHT 640
#define SINGLE_TIME_STEP 0.01
#define ITERATIONS 1000
#define KERNEL_TYPE "cpu"
#define SNAPSHOTS 1
#define PARTICLE_MASS 1
#define COUPLING_CONST 0
#define FILENAME_LENGTH 255

int rot_coord_x = 320, rot_coord_y = 320;
double omega = 0;

void print_usage() {
    std::cout << "Usage:\n" \
              "     trotter [OPTION] -n filename\n" \
              "Arguments:\n" \
              "     -m NUMBER     Particle mass (default: " << PARTICLE_MASS << ")\n"\
              "     -c NUMBER     Coupling constant of the self-interacting term (default: " << COUPLING_CONST << ")\n"\
              "     -d NUMBER     Matrix dimension (default: " << DIM << ")\n" \
              "     -l NUMBER     Physical dimension of the square lattice's edge (default: " << EDGE_LENGHT << ")\n" \
              "     -t NUMBER     Single time step (default: " << SINGLE_TIME_STEP << ")\n" \
              "     -i NUMBER     Number of iterations before a snapshot (default: " << ITERATIONS << ")\n" \
              "     -g            Imaginary time evolution to evolve towards the ground state\n" \
              "     -k STRING     Kernel type (cpu, gpu, or hybrid; default: " << KERNEL_TYPE << "): \n" \
              "     -s NUMBER     Snapshots are taken at every NUMBER of iterations.\n" \
              "                   Zero means no snapshots. Default: " << SNAPSHOTS << ".\n"\
              "     -n STRING     Name of file that defines the initial state.\n"\
              "     -p STRING     Name of file that stores the potential operator (in coordinate representation)\n";
}

void process_command_line(int argc, char** argv, int *dim, double *delta_x, double *delta_y, int *iterations, int *snapshots, string *kernel_type, char *filename, double *delta_t, double *coupling_const, double *particle_mass, char *pot_name, bool *imag_time) {
    // Setting default values
    *dim = DIM;
    *iterations = ITERATIONS;
    *snapshots = SNAPSHOTS;
    *kernel_type = KERNEL_TYPE;
    *delta_t = double(SINGLE_TIME_STEP);
    *coupling_const = double(COUPLING_CONST);
    *particle_mass = double(PARTICLE_MASS);

	double lenght = double(EDGE_LENGHT);
    int c;
    bool file_supplied = false;
    while ((c = getopt (argc, argv, "gd:hi:k:s:n:t:l:p:c:m:")) != -1) {
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
            *kernel_type = optarg;
            if (*kernel_type != "cpu" && *kernel_type != "gpu" && *kernel_type != "hybrid") {
                fprintf (stderr, "The argument of option -t should be cpu, gpu, or hybrid.");
                abort();
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
        case 'c':
            *coupling_const = atoi(optarg);
            break;
        case 'm':
            *particle_mass = atoi(optarg);
            if (delta_t <= 0) {
                fprintf (stderr, "The argument of option -m should be a positive real number.\n");
                abort ();
            }
            break;
        case 't':
            *delta_t = atoi(optarg);
            if (delta_t <= 0) {
                fprintf (stderr, "The argument of option -t should be a positive real number.\n");
                abort ();
            }
            break;
        case 'p':
            for(size_t i = 0; i < strlen(optarg); i++)
                pot_name[i] = optarg[i];
            break;
        case 'l':
            lenght = atoi(optarg);
            if (lenght <= 0) {
                fprintf (stderr, "The argument of option -l should be a positive real number.\n");
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
    *delta_x = lenght / double(*dim);
    *delta_y = lenght / double(*dim);
}

int main(int argc, char** argv) {
    int dim = 0, iterations = 0, snapshots = 0;
    string kernel_type = KERNEL_TYPE;
    int periods[2] = {0, 0};
    double particle_mass = 1.;
    char filename[FILENAME_LENGTH] = "";
    char pot_name[FILENAME_LENGTH] = "";
    bool verbose = true, imag_time = false;
    double h_a = .0, h_b = .0;
    double norm = 1;
    int time, tot_time = 0;
    char output_folder[2] = {'.', '\0'};
    double delta_t = 0;
    double coupling_const = 0;
    double delta_x = 1, delta_y = 1;

#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif
    process_command_line(argc, argv, &dim, &delta_x, &delta_y, &iterations, &snapshots, &kernel_type, filename, &delta_t, &coupling_const, &particle_mass, pot_name, &imag_time);
	
    Lattice *grid = new Lattice(dim, delta_x, delta_y, periods, omega);
    
    int read_offset = 0;

    //set and calculate evolution operator variables from hamiltonian
    double time_single_it;
    double *external_pot_real = new double[grid->dim_x * grid->dim_y];
    double *external_pot_imag = new double[grid->dim_x * grid->dim_y];

    if(imag_time) {
      double constant = 6.;
      time_single_it = delta_t / 2.;	//second approx trotter-suzuki: time/2
      if(h_a == 0. && h_b == 0.) {
        h_a = cosh(time_single_it / (2. * particle_mass)) / constant;
        h_b = sinh(time_single_it / (2. * particle_mass)) / constant;
      }
    }
    else {
      time_single_it = delta_t / 2.;	//second approx trotter-suzuki: time/2
      if(h_a == 0. && h_b == 0.) {
        h_a = cos(time_single_it / (2. * particle_mass));
        h_b = sin(time_single_it / (2. * particle_mass));
      }
    }
    initialize_exp_potential(grid, external_pot_real, external_pot_imag, 
                             const_potential, time_single_it, 
                             particle_mass, imag_time);
    Hamiltonian *hamiltonian = new Hamiltonian(grid, particle_mass, coupling_const, 0, 0, rot_coord_x, rot_coord_y, omega);

    //set initial state
    State *state = new State(grid);
    state->read_state(filename, read_offset);

    for(int count_snap = 0; count_snap <= snapshots; count_snap++) {
      stamp(grid, state, 0, iterations, count_snap, output_folder);

      if(count_snap != snapshots) {
#ifdef WIN32
        SYSTEMTIME start;
        GetSystemTime(&start);
#else
        struct timeval start, end;
        gettimeofday(&start, NULL);
#endif
        trotter(grid, state, hamiltonian, h_a, h_b, external_pot_real, external_pot_imag, delta_t, iterations, kernel_type, norm, imag_time);
#ifdef WIN32
        SYSTEMTIME end;
        GetSystemTime(&end);
        time = (end.wMinute - start.wMinute) * 60000 + (end.wSecond - start.wSecond) * 1000 + (end.wMilliseconds - start.wMilliseconds);
#else
        gettimeofday(&end, NULL);
        time = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
#endif
        tot_time += time;
      }
    }

    if (grid->mpi_coords[0] == 0 && grid->mpi_coords[1] == 0 && verbose == true) {
        std::cout << "TROTTER " << dim << "x" << dim << " kernel:" << kernel_type << " np:" << grid->mpi_procs << " " << tot_time << std::endl;
    }
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return 0;
}
