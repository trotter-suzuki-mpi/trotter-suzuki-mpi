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
#include <fstream>
#include <iomanip>
#include <sys/stat.h>
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "trottersuzuki.h"

#define DIM 640
#define ITERATIONS 1000
#define KERNEL_TYPE "cpu"
#define SNAPSHOTS 10

int main(int argc, char** argv) {
    double length = double(DIM);
    const double particle_mass = 1.;
    bool imag_time = false;
    double delta_t = 5.e-4;

#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif
    //set lattice
    Lattice2D *grid = new Lattice2D(DIM, length, true, true);
    //set initial state
    State *state = new ExponentialState(grid, 1, 0);
    //set hamiltonian
    Hamiltonian *hamiltonian = new Hamiltonian(grid, NULL, particle_mass);
    //set evolution
    Solver *solver = new Solver(grid, state, hamiltonian, delta_t, KERNEL_TYPE);

    if(grid->mpi_rank == 0) {
        cout << "\n* This source provides an example of the trotter-suzuki program.\n";
        cout << "* It calculates the time-evolution of a particle in a box\n";
        cout << "* with periodic boundary conditions, where the initial\n";
        cout << "* state is the following:\n";
        cout << "* \texp(i2M_PI / L (x + y))\n\n";
    }

    //set file output directory
    stringstream dirname, fileprefix, file_info;
    dirname << "D" << DIM << "_I" << ITERATIONS << "_S" << SNAPSHOTS << "";
    mkdir(dirname.str().c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    file_info  << dirname.str() << "/file_info.txt";
    ofstream out(file_info.str().c_str());

    double mean_X, var_X;
    double mean_Y, var_Y;
    double mean_Px, var_Px;
    double mean_Py, var_Py;

    //get expected values
    double norm2 = solver->get_squared_norm();
    double tot_energy = solver->get_total_energy();
    double kin_energy = solver->get_kinetic_energy();

    mean_X = state->get_mean_x();
    var_X = state->get_mean_xx() - state->get_mean_x() * state->get_mean_x();
    mean_Y = state->get_mean_y();
    var_Y = state->get_mean_yy() - state->get_mean_y() * state->get_mean_y();
    mean_Px = state->get_mean_px();
    var_Px = state->get_mean_pxpx() - state->get_mean_px() * state->get_mean_px();
    mean_Py = state->get_mean_py();
    var_Py = state->get_mean_pypy() - state->get_mean_py() * state->get_mean_py();

    if (grid->mpi_rank == 0) {
        out << std::setw(11) << "time" << std::setw(14) << "squared norm" << std::setw(14) << "tot energy" << std::setw(14) << "kin energy" << std::setw(14) << "pot energy"  << std::setw(14) << "rot energy"
            << std::setw(14) << "<X>" << std::setw(14) << "<(X-<X>)^2>" << std::setw(14) << "<Y>" << std::setw(14) << "<(Y-<Y>)^2>"
            << std::setw(14) << "<Px>" << std::setw(14) << "<(Px-<Px>)^2>" << std::setw(14) << "<Py>" << std::setw(14) << "<(Py-<Py>)^2>\n";
        out << std::setw(11) << "0" << std::setw(14) << norm2 << std::setw(14) << std::setw(14) << tot_energy << std::setw(14) << kin_energy << std::setw(14) << solver->get_potential_energy() << std::setw(14) << solver->get_rotational_energy() << std::setw(14)
            << mean_X << std::setw(14) << var_X << std::setw(14) << mean_Y << std::setw(14) << var_Y << std::setw(14)
            << mean_Px << std::setw(14) << var_Px << std::setw(14) << mean_Py << std::setw(14) << var_Py << endl;
    }

    //evolve and stamp the state
    for(int count_snap = 0; count_snap < SNAPSHOTS; count_snap++) {
        solver->evolve(ITERATIONS, imag_time);
        fileprefix.str("");
        fileprefix << dirname.str() << "/" << ITERATIONS * count_snap;
        state->write_to_file(fileprefix.str());

        //get expected values
        norm2 = solver->get_squared_norm();
        tot_energy = solver->get_total_energy();
        kin_energy = solver->get_kinetic_energy();
        mean_X = state->get_mean_x();
        var_X = state->get_mean_xx() - state->get_mean_x() * state->get_mean_x();
        mean_Y = state->get_mean_y();
        var_Y = state->get_mean_yy() - state->get_mean_y() * state->get_mean_y();
        mean_Px = state->get_mean_px();
        var_Px = state->get_mean_pxpx() - state->get_mean_px() * state->get_mean_px();
        mean_Py = state->get_mean_py();
        var_Py = state->get_mean_pypy() - state->get_mean_py() * state->get_mean_py();

        if(grid->mpi_rank == 0) {
            out << std::setw(11) << (count_snap + 1) * ITERATIONS * delta_t << std::setw(14) << norm2 << std::setw(14) << tot_energy << std::setw(14) << kin_energy << std::setw(14) << solver->get_potential_energy() << std::setw(14) << solver->get_rotational_energy() << std::setw(14) <<
                mean_X << std::setw(14) << var_X << std::setw(14) << mean_Y << std::setw(14) << var_Y << std::setw(14) <<
                mean_Px << std::setw(14) << var_Px << std::setw(14) << mean_Py << std::setw(14) << var_Py << endl;
        }
    }
    out.close();

    if (grid->mpi_rank == 0) {
        cout << "TROTTER " << DIM << "x" << DIM << " kernel:" << KERNEL_TYPE << " np:" << grid->mpi_procs << endl;
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
