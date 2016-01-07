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
    State *state = new GaussianState(grid, 0.2, 0., 0., PARTICLES_NUM);
    //set hamiltonian
    Hamiltonian *hamiltonian = new Hamiltonian(grid, particle_mass, coupling_const, 
                                               angular_velocity, rot_coord_x, rot_coord_y);
    hamiltonian->initialize_potential(parabolic_potential);
    //set evolution
    Solver *solver = new Solver(grid, state, hamiltonian, delta_t, KERNEL_TYPE);
    
    //set file output directory
    stringstream fileprefix;
    string dirname = "vortexesdir";
    mkdir(dirname.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    
    fileprefix << dirname << "/file_info.txt";
    ofstream out(fileprefix.str().c_str());
    
    double norm2 = state->calculate_squared_norm();
    double rot_energy = solver->calculate_rotational_energy(norm2);
    double tot_energy = solver->calculate_total_energy(norm2);
    double kin_energy = solver->calculate_kinetic_energy(norm2);

    if(grid->mpi_rank == 0){
      out << "iterations \t rotation energy \t kin energy \t total energy \t norm2\n";
      out << "0\t" << rot_energy << "\t" << kin_energy << "\t" << tot_energy << "\t" << norm2 << endl;
    }
    
    for(int count_snap = 0; count_snap < SNAPSHOTS; count_snap++) {
        solver->evolve(ITERATIONS, imag_time);

        norm2 = state->calculate_squared_norm();
        rot_energy = solver->calculate_rotational_energy(norm2);
        tot_energy = solver->calculate_total_energy(norm2);
        kin_energy = solver->calculate_kinetic_energy(norm2);
        if (grid->mpi_rank == 0){
            out << (count_snap + 1) * ITERATIONS << "\t" << rot_energy << "\t" << kin_energy << "\t" << tot_energy << "\t" << norm2 << endl;
        }
    
        //stamp phase and particles density
        if(count_snap % SNAP_PER_STAMP == 0.) {
            //get and stamp phase
            fileprefix.str("");
            fileprefix << dirname << "/" << ITERATIONS * (count_snap + 1);
            state->write_phase(fileprefix.str());
            //get and stamp particles density
            state->write_particle_density(fileprefix.str());
        }
    }
    out.close();
    fileprefix.str("");
    fileprefix << dirname << "/" << 1 << "-" << ITERATIONS * SNAPSHOTS;
    state->write_to_file(fileprefix.str());
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
