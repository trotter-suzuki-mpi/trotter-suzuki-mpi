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

#include "trottersuzuki.h"
#ifdef HAVE_MPI
#include <mpi.h>
#endif

#define LENGTH 10
#define DIM 100
#define ITERATIONS 1000
#define PARTICLES_NUM 1700000
#define KERNEL_TYPE "cpu"
#define SNAPSHOTS 20
#define SNAP_PER_STAMP 5

double parabolic_potential(double x, double y) {
	double x_c = x - double(LENGTH)*0.5, y_c = y - double(LENGTH)*0.5;
    double w_x = 1., w_y = 1.;
    return 0.5 * (w_x * w_x * x_c * x_c + w_y * w_y * y_c * y_c);
}

int main(int argc, char** argv) {
    char file_name[] = "";
    char pot_name[1] = "";
    const double particle_mass_a = 1., particle_mass_b = 1.;
    bool imag_time = false;
    double delta_t = 1.e-3;
    double length_x = double(LENGTH), length_y = double(LENGTH);
    double coupling_a = 0;//7.116007999594e-4;
    double coupling_b = 0;//7.116007999594e-4;
    double coupling_ab = 0;
    double omega_i = 2.*M_PI/20.;
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif
    //set lattice
    Lattice *grid = new Lattice(DIM, length_x, length_y);
    //set initial state
    State *state1 = new GaussianState(grid, 1, 0., 0., PARTICLES_NUM);
    State *state2 = new GaussianState(grid, 1, 0., 0., PARTICLES_NUM);
    
    //set hamiltonian
    Hamiltonian2Component *hamiltonian = new Hamiltonian2Component(grid, particle_mass_a, particle_mass_b, coupling_a, coupling_ab, coupling_b, 0., omega_i);
    hamiltonian->initialize_potential(parabolic_potential, 0);
    hamiltonian->initialize_potential(parabolic_potential, 1);
    
    //set evolution
    Solver *solver = new Solver(grid, state1, state2, hamiltonian, delta_t, KERNEL_TYPE);
    
    //set file output directory
    stringstream dirname, file_info;
    string dirnames, file_infos;
    if (SNAPSHOTS) {
        int status = 0;
        dirname.str("");
        dirname << "coupledGPE";
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

    double *matrix = new double[grid->dim_x*grid->dim_y];

    double norm2[2];
    norm2[0] = state1->calculate_squared_norm();
    norm2[1] = state2->calculate_squared_norm();
    double tot_energy = solver->calculate_total_energy(norm2[0]+norm2[1]);

    if(grid->mpi_rank == 0){
        out << "time \t total energy \ttot norm2\tnorm2_a\tnorm2_b\n";
        out << "0\t" << "\t" << tot_energy << "\t" << norm2[0] + norm2[1] << "\t" << norm2[0] << "\t" << norm2[1] << endl;
    }
    
    //get and stamp phase
    state1->get_phase(matrix);
    stamp_real(grid, matrix, 0, dirnames.c_str(), "phase_a");
    state2->get_phase(matrix);
    stamp_real(grid, matrix, 0, dirnames.c_str(), "phase_b");
    //get and stamp particles density
    state1->get_particle_density(matrix);
    stamp_real(grid, matrix, 0, dirnames.c_str(), "density_a");
    state2->get_particle_density(matrix);
    stamp_real(grid, matrix, 0, dirnames.c_str(), "density_b");
    for (int count_snap = 0; count_snap < SNAPSHOTS; count_snap++) {
        solver->evolve(ITERATIONS, imag_time);
        //norm calculation
        norm2[0] = state1->calculate_squared_norm();
        norm2[1] = state2->calculate_squared_norm();
        tot_energy = solver->calculate_total_energy(norm2[0]+norm2[1]);

        if (grid->mpi_rank == 0){
            out << (count_snap + 1) * ITERATIONS * delta_t << "\t" << tot_energy << "\t" << norm2[0]+norm2[1] << "\t" << norm2[0] << "\t" << norm2[1] << endl;
        }

        //stamp phase and particles density
        if(count_snap % SNAP_PER_STAMP == 0.) {
            //get and stamp phase
            state1->get_phase(matrix);
            stamp_real(grid, matrix, ITERATIONS * (count_snap + 1), dirnames.c_str(), "phase_a");
            state2->get_phase(matrix);
            stamp_real(grid, matrix, ITERATIONS * (count_snap + 1), dirnames.c_str(), "phase_b");
            //get and stamp particles density
            state1->get_particle_density(matrix);
            stamp_real(grid, matrix, ITERATIONS * (count_snap + 1), dirnames.c_str(), "density_a");
            state2->get_particle_density(matrix);
            stamp_real(grid, matrix, ITERATIONS * (count_snap + 1), dirnames.c_str(), "density_b");
        }
    }

    out.close();
    stamp(grid, state1, 0, ITERATIONS, SNAPSHOTS, dirnames.c_str());
    //delete solver;
    //delete hamiltonian;
    delete state1;
    delete state2;
    delete grid;
    delete matrix;
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return 0;
}
