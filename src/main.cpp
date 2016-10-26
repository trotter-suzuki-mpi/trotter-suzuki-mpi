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

#include <iostream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include "trottersuzuki.h"
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
#include "common.h"

#define DIM 640
#define EDGE_LENGTH 640
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
    cout << "Usage:\n" \
         "     trotter [OPTION] -n filename\n" \
         "Arguments:\n" \
         "     -m NUMBER     Particle mass (default: " << PARTICLE_MASS << ")\n"\
         "     -c NUMBER     Coupling constant of the self-interacting term (default: " << COUPLING_CONST << ")\n"\
         "     -d NUMBER     Matrix dimension (default: " << DIM << ")\n" \
         "     -l NUMBER     Physical dimension of the square lattice's edge (default: " << EDGE_LENGTH << ")\n" \
         "     -t NUMBER     Single time step (default: " << SINGLE_TIME_STEP << ")\n" \
         "     -i NUMBER     Number of iterations before a snapshot (default: " << ITERATIONS << ")\n" \
         "     -g            Imaginary time evolution to evolve towards the ground state\n" \
         "     -k STRING     Kernel type (cpu, gpu, or hybrid; default: " << KERNEL_TYPE << "): \n" \
         "     -s NUMBER     Snapshots are taken at every NUMBER of iterations.\n" \
         "                   Zero means no snapshots. Default: " << SNAPSHOTS << ".\n"\
         "     -n STRING     Name of file that defines the initial state.\n"\
         "     -p STRING     Name of file that stores the potential operator (in coordinate representation)\n";
}

void process_command_line(int argc, char** argv, int *dim, double *length_x, double *length_y, int *iterations, int *snapshots, string *kernel_type, char *filename, double *delta_t, double *coupling_const, double *particle_mass, char *pot_name, bool *imag_time) {
    // Setting default values
    *dim = DIM;
    *iterations = ITERATIONS;
    *snapshots = SNAPSHOTS;
    *kernel_type = KERNEL_TYPE;
    *delta_t = double(SINGLE_TIME_STEP);
    *coupling_const = double(COUPLING_CONST);
    *particle_mass = double(PARTICLE_MASS);

    double length = double(EDGE_LENGTH);
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
                my_abort("The argument of option -d should be a positive integer.\n");
            }
            break;
        case 'i':
            *iterations = atoi(optarg);
            if (*iterations <= 0) {
                my_abort("The argument of option -i should be a positive integer.\n");
            }
            break;
        case 'h':
            print_usage();
#ifdef HAVE_MPI
            MPI_Finalize();
#endif
            my_abort("");
            break;
        case 'k':
            *kernel_type = optarg;
            if (*kernel_type != "cpu" && *kernel_type != "gpu" && *kernel_type != "hybrid") {
                my_abort("The argument of option -t should be cpu, gpu, or hybrid.");
            }
            break;
        case 's':
            *snapshots = atoi(optarg);
            if (*snapshots <= 0) {
                my_abort("The argument of option -s should be a positive integer.\n");
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
                my_abort("The argument of option -m should be a positive real number.\n");
            }
            break;
        case 't':
            *delta_t = atoi(optarg);
            if (delta_t <= 0) {
                my_abort("The argument of option -t should be a positive real number.\n");
            }
            break;
        case 'p':
            for(size_t i = 0; i < strlen(optarg); i++)
                pot_name[i] = optarg[i];
            break;
        case 'l':
            length = atoi(optarg);
            if (length <= 0) {
                my_abort("The argument of option -l should be a positive real number.\n");
            }
            break;
        case '?':
            if (optopt == 'd' || optopt == 'i' || optopt == 'k' || optopt == 's') {
                stringstream sstm;
                sstm << "Option -" <<  optopt << " requires an argument.";
                my_abort(sstm.str());
            }
            else if (isprint (optopt)) {
                stringstream sstm;
                sstm << "Unknown option -" << optopt;
                my_abort(sstm.str());
            }
            else {
                stringstream sstm;
                sstm << "Unknown option -" << optopt;
                my_abort(sstm.str());
            }
        default:
            stringstream sstm;
            sstm << "Unknown option -" << optopt;
            my_abort(sstm.str());
        }
    }
    if(!file_supplied) {
        my_abort("Initial state file has not been supplied\n");
    }
    *length_x = length;
    *length_y = length;
}

int main(int argc, char** argv) {
    int dim = 0, iterations = 0, snapshots = 0;
    string kernel_type = KERNEL_TYPE;
    double particle_mass = 1.;
    char filename[FILENAME_LENGTH] = "";
    char pot_name[FILENAME_LENGTH] = "";
    bool verbose = true, imag_time = false;
    int time, tot_time = 0;
    double delta_t = 0;
    double coupling_a = 0;
    double length_x = 10, length_y = 10;

#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif
    try {
        process_command_line(argc, argv, &dim, &length_x, &length_y, &iterations, &snapshots, &kernel_type, filename, &delta_t, &coupling_a, &particle_mass, pot_name, &imag_time);
    }
    catch (runtime_error& e) {
        cerr << e.what() << endl;
        return 1; // exit is okay here because an MPI runtime would have aborted in my_abort
    }
    //set lattice
    Lattice2D *grid = new Lattice2D(dim, length_x, length_y);

    //set hamiltonian
    Hamiltonian *hamiltonian = new Hamiltonian(grid, NULL, particle_mass, coupling_a);

    //set initial state
    State *state = new State(grid);
    state->loadtxt(filename);

    //set evolution
    Solver *solver = new Solver(grid, state, hamiltonian, delta_t, kernel_type);

    //evolve the state
    for(int count_snap = 0; count_snap <= snapshots; count_snap++) {

        if(count_snap != snapshots) {
#ifdef WIN32
            SYSTEMTIME start;
            GetSystemTime(&start);
#else
            struct timeval start, end;
            gettimeofday(&start, NULL);
#endif
            solver->evolve(iterations, imag_time);
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
        cout << "TROTTER " << dim << "x" << dim << " kernel:" << kernel_type << " np:" << grid->mpi_procs << " " << tot_time << endl;
    }
    delete solver;
    delete hamiltonian;
    delete state;
    delete grid;
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return 0;
}
