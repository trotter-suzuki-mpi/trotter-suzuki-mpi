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

/**
 * This source provides an example of the trotter-suzuki program.
 * It calculates the time-evolution of a particle in a box, where the initial
 * state is the following:
 * 		exp(i2M_PI / L (x + y))
 */
#include <iostream>
#include <sys/stat.h>
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "trottersuzuki.h"

#define LENGTH 30
#define DIM 640
#define ITERATIONS 1000
#define PARTICLES_NUM 1700000
#define KERNEL_TYPE "cpu"
#define SNAPSHOTS 10
#define SCATTER_LENGTH_2D 5.662739242e-5

int main(int argc, char** argv) {
    bool verbose = true;
    bool imag_time = true;
    double delta_t = 1.e-5;
    const double particle_mass = 1.;
    double coupling_a = 4. * M_PI * double(SCATTER_LENGTH_2D);

#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif

    //set lattice
    Lattice2D *grid = new Lattice2D(DIM, (double)LENGTH);
    //set initial state
    State *state = new GaussianState(grid, 1., 1., 0., 0., PARTICLES_NUM);
    //set hamiltonian
    Potential *potential = new HarmonicPotential(grid, 1., sqrt(2));
    Hamiltonian *hamiltonian = new Hamiltonian(grid, potential, particle_mass, coupling_a);
    //set solver
    Solver *solver = new Solver(grid, state, hamiltonian, delta_t, KERNEL_TYPE);

    if(grid->mpi_rank == 0) {
        cout << "\n* This source provides an example of the trotter-suzuki program.\n";
        cout << "* It calculates the ground state of a BEC trapped in a harmonic potential.\n";
    }

    //set file output directory
    stringstream dirname, fileprefix;
    dirname << "D" << DIM << "_I" << ITERATIONS << "_S" << SNAPSHOTS << "";
    mkdir(dirname.str().c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    fileprefix.str("");
    fileprefix << dirname.str() << "/" << 0;
    state->write_to_file(fileprefix.str());
    for(int count_snap = 0; count_snap < SNAPSHOTS; count_snap++) {
        solver->evolve(ITERATIONS, imag_time);
        fileprefix.str("");
        fileprefix << dirname.str() << "/" << ITERATIONS * (count_snap + 1);
        state->write_to_file(fileprefix.str());
    }
    if (grid->mpi_rank == 0 && verbose == true) {
        cout << "TROTTER " << DIM << "x" << DIM << " kernel:" << KERNEL_TYPE << " np:" << grid->mpi_procs << endl;
    }
    delete solver;
    delete hamiltonian;
    delete potential;
    delete state;
    delete grid;
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return 0;
}
