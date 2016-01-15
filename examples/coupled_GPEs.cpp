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
#include <iomanip>
#include <sys/stat.h>

#include "trottersuzuki.h"
#ifdef HAVE_MPI
#include <mpi.h>
#endif

#define LENGTH 10
#define DIM 200
#define ITERATIONS 1000
#define PARTICLES_NUM 1700000
#define KERNEL_TYPE "cpu"
#define SNAPSHOTS 20
#define SNAP_PER_STAMP 5

int main(int argc, char** argv) {
    double particle_mass_a = 1., particle_mass_b = 1.;
    bool imag_time = false;
    double delta_t = 1.e-3;
    double length_x = double(LENGTH), length_y = double(LENGTH);
    double coupling_a = 0;//7.116007999594e-4;
    double coupling_b = 0;//7.116007999594e-4;
    double coupling_ab = 0;
    double omega_i = 0.;
    double omega_r = 2.*M_PI/20.;
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif
    //set lattice
    Lattice *grid = new Lattice(DIM, length_x, length_y);
    //set initial state
    State *state1 = new GaussianState(grid, 1, 0., 0., PARTICLES_NUM);
    State *state2 = new GaussianState(grid, 1, 0., 0., PARTICLES_NUM, M_PI/2.);
    
    //set hamiltonian
    Potential *potential = new ParabolicPotential(grid, 1., 1.);
    Hamiltonian2Component *hamiltonian = new Hamiltonian2Component(grid, potential, potential, particle_mass_a, particle_mass_b, coupling_a, coupling_ab, coupling_b, omega_r, omega_i);
    
    //set evolution
    Solver *solver = new Solver(grid, state1, state2, hamiltonian, delta_t, KERNEL_TYPE);
    
    //set file output directory
    stringstream fileprefix;
    string dirname = "coupledGPE";
    mkdir(dirname.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    
    fileprefix << dirname << "/file_info.txt";
    ofstream out(fileprefix.str().c_str());

    double norm2[2];
    norm2[0] = state1->get_squared_norm();
    norm2[1] = state2->get_squared_norm();
    //get expected values
    double tot_energy = solver->get_total_energy();
    double kin_energy = solver->get_kinetic_energy();
    double rabi_energy = solver->get_rabi_energy();
    double pot_energy = solver->get_potential_energy();
    double intra_energy = solver->get_intra_species_energy();
    double inter_energy = solver->get_inter_species_energy();

    
    if (grid->mpi_rank == 0){
      out << std::setw(11) << "time" << std::setw(14) << "squared norm" << std::setw(14) << "sq norm1" << std::setw(14) << "sq norm2" << std::setw(14) << "tot energy" << std::setw(14) 
          << "kin energy" << std::setw(14) << "pot energy" << std::setw(14) << "intra energy" << std::setw(14) << "inter energy"  << std::setw(14) << "rabi energy\n";        
      out << std::setw(11) << "0" << std::setw(14) << norm2[0] + norm2[1] << std::setw(14) << norm2[0] << std::setw(14) << norm2[1] << std::setw(14) << tot_energy << std::setw(14) 
          << kin_energy << std::setw(14) << pot_energy << std::setw(14) << intra_energy << std::setw(14) << inter_energy << std::setw(14) << rabi_energy << endl;
    }
    
    //write phase and density
    fileprefix.str("");
    fileprefix << dirname << "/1-" << 0;
    state1->write_phase(fileprefix.str());
    state1->write_particle_density(fileprefix.str());    
    fileprefix.str("");
    fileprefix << dirname << "/2-" << 0;
    state2->write_phase(fileprefix.str());
    state2->write_particle_density(fileprefix.str());    

    for (int count_snap = 0; count_snap < SNAPSHOTS; count_snap++) {
        solver->evolve(ITERATIONS, imag_time);
        //norm calculation
        norm2[0] = state1->get_squared_norm();
        norm2[1] = state2->get_squared_norm();
        //get expected values
        tot_energy = solver->get_total_energy();
        kin_energy = solver->get_kinetic_energy();
        rabi_energy = solver->get_rabi_energy();
        pot_energy = solver->get_potential_energy();
        intra_energy = solver->get_intra_species_energy();
        inter_energy = solver->get_inter_species_energy();
        
        if(grid->mpi_rank == 0){
            out << std::setw(11) << (count_snap + 1) * ITERATIONS * delta_t << std::setw(14) << norm2[0] + norm2[1] << std::setw(14) << norm2[0] << std::setw(14) << norm2[1] << std::setw(14) << tot_energy << std::setw(14) 
                << kin_energy << std::setw(14) << pot_energy << std::setw(14) << intra_energy << std::setw(14) << inter_energy << std::setw(14) << rabi_energy << endl;
        }

        //stamp phase and particles density
        if(count_snap % SNAP_PER_STAMP == 0.) {
            //write phase and density
            fileprefix.str("");
            fileprefix << dirname << "/1-" << ITERATIONS * (count_snap + 1);
            state1->write_phase(fileprefix.str());
            state1->write_particle_density(fileprefix.str());    
            fileprefix.str("");
            fileprefix << dirname << "/2-" << ITERATIONS * (count_snap + 1);
            state2->write_phase(fileprefix.str());
            state2->write_particle_density(fileprefix.str());    
        }
    }

    out.close();
    fileprefix.str("");
    fileprefix << dirname << "/" << 1 << "-" << ITERATIONS * SNAPSHOTS;
    state1->write_to_file(fileprefix.str());
    //delete solver;
    //delete hamiltonian;
    delete state1;
    delete state2;
    delete grid;
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return 0;
}
