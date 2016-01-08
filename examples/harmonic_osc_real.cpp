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
#include <sys/time.h>
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "trottersuzuki.h"

#define EDGE_LENGTH 14.14     //Physical length of the grid's edge
#define DIM 256         //Number of dots of the grid's edge
#define DELTA_T 1.e-4     //Time step evolution
#define ITERATIONS 1000     //Number of iterations before calculating expected values
#define KERNEL_TYPE "cpu"
#define SNAPSHOTS 40      //Number of times the expected values are calculated
#define SNAP_PER_STAMP 5    //The particles density and phase of the wave function are stamped every "SNAP_PER_STAMP" expected values calculations
#define PARTICLES_NUM 1     //Particle numbers (nonlinear Schrodinger equation)

complex<double> gauss_ini_state(double x, double y) {
	double x_c = x - double(EDGE_LENGTH)*0.5, y_c = y - double(EDGE_LENGTH)*0.5;
    double w = 1.;
    return complex<double>(sqrt(0.5 * w / M_PI) * exp(-(x_c * x_c + y_c * y_c) * 0.5 * w) * (1. + sqrt(2. * w) * x_c), 0.);
}
int main(int argc, char** argv) {
    char file_name[] = "";
    char pot_name[1] = "";
    const double particle_mass = 1.;
    bool imag_time = false;
    int time, tot_time = 0;
    double delta_t = double(DELTA_T);
    double length_x = double(EDGE_LENGTH), length_y = double(EDGE_LENGTH);
    
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif

    //set lattice
    Lattice *grid = new Lattice(DIM, length_x, length_y);
    //set initial state
    State *state = new State(grid);
    state->init_state(gauss_ini_state);
    //set Hamiltonian
    Potential *potential = new ParabolicPotential(grid, 1., 1.);
    Hamiltonian *hamiltonian = new Hamiltonian(grid, potential, particle_mass);
    //set evolution
    Solver *solver = new Solver(grid, state, hamiltonian, delta_t, KERNEL_TYPE);
 
    //set file output directory
    stringstream file_info, fileprefix;
    string dirname = "Harmonic_osc_RE";
    mkdir(dirname.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    file_info  << dirname << "/file_info.txt";
    ofstream out(file_info.str().c_str());
    
    double mean_positions[4], mean_momenta[4], results[4];
    
    //norm calculation
    double norm2 = state->calculate_squared_norm();
    double tot_energy = solver->calculate_total_energy(norm2);
    double kin_energy = solver->calculate_kinetic_energy(norm2);

    //Position expected values
    state->calculate_mean_position(grid->dim_x / 2, grid->dim_y / 2, mean_positions, norm2);

    //Momenta expected values
    state->calculate_mean_momentum(mean_momenta, norm2);
                   
    //get and stamp phase
    fileprefix.str("");
    fileprefix << dirname << "/" << 0;
    state->write_phase(fileprefix.str());

    //get and stamp particles density
    state->write_particle_density(fileprefix.str());    

    if (grid->mpi_rank == 0){
      out << "iterations\tsquared norm\ttotal_energy\tkinetic_energy\t<X>\t<(X-<X>)^2>\t<Y>\t<(Y-<Y>)^2>\t<Px>\t<(Px-<Px>)^2>\t<Py>\t<(Py-<Py>)^2>\n";
      out << "0\t\t" << norm2 << "\t\t"<< tot_energy << "\t" << kin_energy << "\t" << mean_positions[0] << "\t" << mean_positions[1] << "\t" << mean_positions[2] << "\t" << mean_positions[3] << "\t" << mean_momenta[0] << "\t" << mean_momenta[1] << "\t" << mean_momenta[2] << "\t" << mean_momenta[3] << endl;
    }
  
    struct timeval start, end;
    for (int count_snap = 0; count_snap < SNAPSHOTS; count_snap++) {
        
        gettimeofday(&start, NULL);
        solver->evolve(ITERATIONS, imag_time);
        gettimeofday(&end, NULL);
        time = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
        tot_time += time;
        
        //norm calculation
        norm2 = state->calculate_squared_norm();
        tot_energy = solver->calculate_total_energy(norm2);
        kin_energy = solver->calculate_kinetic_energy(norm2);
              
        //Position expected values
        state->calculate_mean_position(grid->dim_x / 2, grid->dim_y / 2, mean_positions, norm2);

        //Momenta expected values
        state->calculate_mean_momentum(mean_momenta, norm2);

        if(grid->mpi_rank == 0){
            out << (count_snap + 1) * ITERATIONS << "\t\t" << norm2 << "\t\t"<< tot_energy << "\t" << kin_energy << "\t" << mean_positions[0] << "\t" << mean_positions[1] << "\t" << mean_positions[2] << "\t" << mean_positions[3] << "\t" << mean_momenta[0] << "\t" << mean_momenta[1] << "\t" << mean_momenta[2] << "\t" << mean_momenta[3] << endl;
        }
    
        //stamp phase and particles density
        if((count_snap + 1) % SNAP_PER_STAMP == 0.) {
            //get and stamp phase
            fileprefix.str("");
            fileprefix << dirname << "/" << ITERATIONS * (count_snap + 1);
            state->write_phase(fileprefix.str());
            
            //get and stamp particles density
            state->write_particle_density(fileprefix.str());
        }
    }
    out.close();
    
    if (grid->mpi_rank == 0) {
        cout << "TROTTER " << DIM << "x" << DIM << " kernel:" << KERNEL_TYPE << " np:" << grid->mpi_procs << " time:" << tot_time << " usec" << endl;
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
