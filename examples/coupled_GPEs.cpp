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

#define LENGHT 20
#define DIM 400
#define ITERATIONS 4
#define PARTICLES_NUM 1700000
#define KERNEL_TYPE "cpu"
#define SNAPSHOTS 2
#define SNAP_PER_STAMP 1

int rot_coord_x = 320, rot_coord_y = 320;
double omega = 0.;

complex<double> gauss_ini_state(int m, int n, Lattice *grid) {
  double delta_x = double(LENGHT)/double(DIM);
    double x = (m - grid->global_dim_x / 2.) * delta_x, y = (n - grid->global_dim_y / 2.) * delta_x;
    double w = 1.;
    return complex<double>(sqrt(w * double(PARTICLES_NUM) / M_PI) * exp(-(x * x + y * y) * 0.5 * w), 0.);
}


double parabolic_potential(int m, int n, Lattice *grid) {
    double x = (m - grid->global_dim_x / 2.) * grid->delta_x,
           y = (n - grid->global_dim_x / 2.) * grid->delta_y;
    double w_x = 1., w_y = 1.;
    return 0.5 * (w_x * w_x * x * x + w_y * w_y * y * y);
}

int main(int argc, char** argv) {
    int periods[2] = {0, 0};
    char file_name[] = "";
    char pot_name[1] = "";
    const double particle_mass_a = 1., particle_mass_b = 1.;
    bool imag_time = true;
    double h_a[2];
    double h_b[2];

    double delta_t = 5.e-5;
    double delta_x = double(LENGHT)/double(DIM), delta_y = double(LENGHT)/double(DIM);

#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif
    Lattice *grid = new Lattice(DIM, delta_x, delta_y, periods, omega);
    
    //set and calculate evolution operator variables from hamiltonian
    double time_single_it;
    double coupling_const[5] = {7.116007999594e-4, 7.116007999594e-4, 0., 0., 0.};
    double *external_pot_real[2];
    double *external_pot_imag[2];
    external_pot_real[0] = new double[grid->dim_x * grid->dim_y];
    external_pot_imag[0] = new double[grid->dim_x * grid->dim_y];
    external_pot_real[1] = new double[grid->dim_x * grid->dim_y];
    external_pot_imag[1] = new double[grid->dim_x * grid->dim_y];

    if(imag_time) {
        time_single_it = delta_t / 2.;  //second approx trotter-suzuki: time/2
    h_a[0] = cosh(time_single_it / (2. * particle_mass_a * delta_x * delta_y));
    h_b[0] = sinh(time_single_it / (2. * particle_mass_a * delta_x * delta_y));
    h_a[1] = cosh(time_single_it / (2. * particle_mass_b * delta_x * delta_y));
    h_b[1] = sinh(time_single_it / (2. * particle_mass_b * delta_x * delta_y));
    }
    else {
        time_single_it = delta_t / 2.;  //second approx trotter-suzuki: time/2
    h_a[0] = cos(time_single_it / (2. * particle_mass_a * delta_x * delta_y));
    h_b[0] = sin(time_single_it / (2. * particle_mass_a * delta_x * delta_y));
    h_a[1] = cos(time_single_it / (2. * particle_mass_b * delta_x * delta_y));
    h_b[1] = sin(time_single_it / (2. * particle_mass_b * delta_x * delta_y));
    }
    initialize_exp_potential(grid, external_pot_real[0], external_pot_imag[0], 
                             parabolic_potential, time_single_it, 
                             particle_mass_a, imag_time);
    initialize_exp_potential(grid, external_pot_real[1], external_pot_imag[1], 
                             parabolic_potential, time_single_it, 
                             particle_mass_b, imag_time);
    //set initial state
    State *state1 = new State(grid);
    state1->init_state(gauss_ini_state);
    State *state2 = new State(grid);
    state2->init_state(gauss_ini_state);
    double *p_real[2];
    double *p_imag[2];
    p_real[0] = state1->p_real;
    p_imag[0] = state1->p_imag;
    p_real[1] = state2->p_real;
    p_imag[1] = state2->p_imag;
    Hamiltonian2Component *hamiltonian = new Hamiltonian2Component(grid, particle_mass_a, particle_mass_b, coupling_const[0], coupling_const[2], coupling_const[1], rot_coord_x, rot_coord_y, omega, coupling_const[3], coupling_const[4]);

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

    double *_matrix = new double[grid->dim_x*grid->dim_y];
    double _norm2;

    double norm2[2];
    norm2[0] = state1->calculate_squared_norm();
    norm2[1] = state2->calculate_squared_norm();
    double _tot_energy = calculate_total_energy(grid, state1, state2, hamiltonian, parabolic_potential, parabolic_potential, NULL, norm2[0]+norm2[1]);

    if(grid->mpi_rank == 0){
        out << "iterations \t total energy \t norm2\n";
        out << "0\t" << "\t" << _tot_energy << "\t" << norm2[0] + norm2[1] << endl;
    }

    for (int count_snap = 0; count_snap < SNAPSHOTS; count_snap++) {
        trotter(grid, state1, state2, hamiltonian, h_a, h_b, external_pot_real, external_pot_imag, delta_t,
                ITERATIONS, KERNEL_TYPE, norm2, imag_time);
        //norm calculation
        _norm2 = state1->calculate_squared_norm() + state2->calculate_squared_norm();
        _tot_energy = calculate_total_energy(grid, state1, state2, hamiltonian, parabolic_potential, parabolic_potential, NULL, _norm2);

        if (grid->mpi_rank == 0){
            out << (count_snap + 1) * ITERATIONS << "\t" << _tot_energy << "\t" << _norm2 << endl;
        }

        //stamp phase and particles density
        if(count_snap % SNAP_PER_STAMP == 0.) {
            //get and stamp phase
            state1->get_phase(_matrix);
            stamp_real(grid, _matrix, ITERATIONS * (count_snap + 1), dirnames.c_str(), "phase_a");
            state2->get_phase(_matrix);
            stamp_real(grid, _matrix, ITERATIONS * (count_snap + 1), dirnames.c_str(), "phase_b");
            //get and stamp particles density
            state1->get_particle_density(_matrix);
            stamp_real(grid, _matrix, ITERATIONS * (count_snap + 1), dirnames.c_str(), "density_a");
            state2->get_particle_density(_matrix);
            stamp_real(grid, _matrix, ITERATIONS * (count_snap + 1), dirnames.c_str(), "density_b");
        }
    }

    out.close();
    stamp(grid, state1, 0, ITERATIONS, SNAPSHOTS, dirnames.c_str());
    delete hamiltonian;
    delete state1;
    delete state2;
    delete grid;
    return 0;
}
