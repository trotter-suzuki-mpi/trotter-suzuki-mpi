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
#include <fstream>
#include <sys/stat.h>
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "trottersuzuki.h"

#define LENGTH 25
#define DIM 640
#define ITERATIONS 3000
#define PARTICLES_NUM 1.e+6
#define KERNEL_TYPE "cpu"
#define SNAPSHOTS 20
#define SNAP_PER_STAMP 1
#define COUPLING_CONST_2D 7.116007999594e-4

complex<double> gauss_ini_state(double x, double y) {
    double x_c = x - double(LENGTH)*0.5, y_c = y - double(LENGTH)*0.5;
    double w = 0.2;
    return complex<double>(sqrt(0.5 * w * double(PARTICLES_NUM) / M_PI) * exp(-(x_c * x_c + y_c * y_c) * 0.5 * w), sqrt(0.5 * w * double(PARTICLES_NUM) / M_PI) * exp(-(x_c * x_c + y_c * y_c) * 0.5 * w));
}

double parabolic_potential(double x, double y) {
	double x_c = x - double(LENGTH)*0.5, y_c = y - double(LENGTH)*0.5;
    double w_x = 1., w_y = 1.; 
    return 0.5 * (w_x * w_x * x_c * x_c + w_y * w_y * y_c * y_c);
}

int main(int argc, char** argv) {
    int periods[2] = {0, 0};
    int rot_coord_x = 320, rot_coord_y = 320;
    double angular_velocity = 0.9;
    const double particle_mass = 1.;
    bool imag_time = true;
    double delta_t = 2.e-4;
    double length_x = double(LENGTH), length_y = double(LENGTH);
    double coupling_const = double(COUPLING_CONST_2D);
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif

    //set lattice
    Lattice *grid = new Lattice(DIM, length_x, length_y, periods, angular_velocity);
    //set initial state
    State *state = new State(grid);
    state->init_state(gauss_ini_state);
    //set hamiltonian
    Hamiltonian *hamiltonian = new Hamiltonian(grid, particle_mass, coupling_const, 
                                               angular_velocity, rot_coord_x, rot_coord_y);
    hamiltonian->initialize_potential(parabolic_potential);
    //set evolution
    Solver *solver = new Solver(grid, state, hamiltonian, delta_t, KERNEL_TYPE);
    
    //set file output directory
    stringstream dirname, file_info;
    string dirnames, file_infos;
    if (SNAPSHOTS) {
        int status = 0;
        dirname.str("");
        dirname << "vortexesdir";
        dirnames = dirname.str();
        status = mkdir(dirnames.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if(status != 0 && status != -1)
            dirnames = ".";
    } else {
        dirnames = ".";
    }
    file_info.str("");
    file_info << dirnames << "/file_info.txt";
    file_infos = file_info.str();
    ofstream out(file_infos.c_str());
    
    double *_matrix = new double[grid->dim_x*grid->dim_y];
    double _norm2, sum;
    double norm2 = state->calculate_squared_norm();
    double _rot_energy = calculate_rotational_energy(grid, state, hamiltonian, norm2);
    double _tot_energy = calculate_total_energy(grid, state, hamiltonian, parabolic_potential, NULL, norm2);
    double _kin_energy = calculate_kinetic_energy(grid, state, hamiltonian, norm2);

    if(grid->mpi_rank == 0){
      out << "iterations \t rotation energy \t kin energy \t total energy \t norm2\n";
      out << "0\t" << _rot_energy << "\t" << _kin_energy << "\t" << _tot_energy << "\t" << norm2 << endl;
    }
    
    for(int count_snap = 0; count_snap < SNAPSHOTS; count_snap++) {
        solver->evolve(ITERATIONS, imag_time);

        _norm2 = state->calculate_squared_norm();
        _rot_energy = calculate_rotational_energy(grid, state, hamiltonian, _norm2);
        _tot_energy = calculate_total_energy(grid, state, hamiltonian, parabolic_potential, NULL, _norm2);
        _kin_energy = calculate_kinetic_energy(grid, state, hamiltonian, _norm2);
        if(grid->mpi_rank == 0){
            out << (count_snap + 1) * ITERATIONS << "\t" << _rot_energy << "\t" << _kin_energy << "\t" << _tot_energy << "\t" << _norm2 << endl;
        }
    
        //stamp phase and particles density
        if(count_snap % SNAP_PER_STAMP == 0.) {
            //get and stamp phase
            state->get_phase(_matrix);
            stamp_real(grid, _matrix, ITERATIONS * (count_snap + 1), dirnames.c_str(), "phase");
            //get and stamp particles density
            state->get_particle_density(_matrix);
            stamp_real(grid, _matrix, ITERATIONS * (count_snap + 1), dirnames.c_str(), "density");
        }
    }
    out.close();
    stamp(grid, state, 0, ITERATIONS, SNAPSHOTS, dirnames.c_str());
    cout << "\n";
    delete solver;
    delete hamiltonian;
    delete state;
    delete grid;
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return 0;
}
