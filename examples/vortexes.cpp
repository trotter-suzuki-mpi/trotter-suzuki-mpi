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

#define LENGHT 50
#define DIM 640
#define ITERATIONS 2
#define PARTICLES_NUM 8.e+6
#define KERNEL_TYPE "cpu"
#define SNAPSHOTS 2
#define SNAP_PER_STAMP 20
#define COUPLING_CONST_2D 7.116007999594e-4

complex<double> gauss_ini_state(int m, int n, Lattice *grid) {
    double x = (m - grid->global_dim_x / 2.) * grid->delta_x;
    double y = (n - grid->global_dim_y / 2.) * grid->delta_x;
    double w = 0.002/(grid->delta_x * grid->delta_x);
    return complex<double>(sqrt(0.5 * w * double(PARTICLES_NUM) / M_PI) * exp(-(x * x + y * y) * 0.5 * w), sqrt(0.5 * w * double(PARTICLES_NUM) / M_PI) * exp(-(x * x + y * y) * 0.5 * w));
}

double parabolic_potential(int m, int n, Lattice *grid) {
    double x = (m - grid->global_dim_x / 2.) * grid->delta_x;
    double y = (n - grid->global_dim_x / 2.) * grid->delta_x;
    double w_x = 1., w_y = 1.; 
    return 0.5 * (w_x * w_x * x * x + w_y * w_y * y * y);
}

int main(int argc, char** argv) {
    int periods[2] = {0, 0};
    int rot_coord_x = 320, rot_coord_y = 320;
    double omega = 0.9;
    const double particle_mass = 1.;
    bool imag_time = true;
    double h_a, h_b;
    double delta_t = 2.e-4;
    double delta_x = double(LENGHT)/double(DIM), delta_y = double(LENGHT)/double(DIM);
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif
    Lattice *grid = new Lattice(DIM, delta_x, delta_y, periods, omega);

    double *external_pot_real = new double[grid->dim_x * grid->dim_y];
    double *external_pot_imag = new double[grid->dim_x * grid->dim_y];
    
    //set and calculate evolution operator variables from hamiltonian
    double time_single_it = delta_t / 2.; //second approx trotter-suzuki: time/2
    double coupling_const = double(COUPLING_CONST_2D);

    if (imag_time) {
        h_a = cosh(time_single_it / (2. * particle_mass * delta_x * delta_y));
        h_b = sinh(time_single_it / (2. * particle_mass * delta_x * delta_y));
    }
    else {
        h_a = cos(time_single_it / (2. * particle_mass * delta_x * delta_y));
        h_b = sin(time_single_it / (2. * particle_mass * delta_x * delta_y));
    }
    initialize_exp_potential(grid, external_pot_real, external_pot_imag, 
                             parabolic_potential, time_single_it, 
                             particle_mass, imag_time);

    //set initial state
    State *state = new State(grid);
    state->init_state(gauss_ini_state);
    Hamiltonian *hamiltonian = new Hamiltonian(grid, particle_mass, 
                                               coupling_const, 0, 0, 
                                               rot_coord_x, rot_coord_y, omega);
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
        trotter(grid, state, hamiltonian, h_a, h_b, 
                external_pot_real, external_pot_imag, delta_t, ITERATIONS, 
                KERNEL_TYPE, norm2, imag_time);

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
    delete hamiltonian;
    delete state;
    delete grid;
    return 0;
}
