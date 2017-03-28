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
#include <iomanip>
#include <fstream>
#include <sys/stat.h>
#include <sys/time.h>
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "trottersuzuki.h"

#define IMAG_TIME false    //If true performs imaginary time evolution
#define EDGE_LENGTH 50     //Physical length of the grid's edge
#define DIM 50         //Number of dots of the grid's edge
#define DELTA_T 1.e-2     //Time step evolution
#define ITERATIONS 10000     //Number of iterations before calculating expected values
#define KERNEL_TYPE "cpu"
#define SNAPSHOTS 10      //Number of times the expected values are calculated
#define SNAP_PER_STAMP 5    //The particles density and phase of the wave function are stamped every "SNAP_PER_STAMP" expected values calculations
#define ANGULAR_MOMENTUM 0


complex<double> const_ini_state(double r, double z) {
    return 1 / double(EDGE_LENGTH);
}

int main(int argc, char** argv) {

    int time, tot_time = 0;
    double delta_t = double(DELTA_T);
    double length = double(EDGE_LENGTH);
    int angular_momentum = int(ANGULAR_MOMENTUM);

#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif

    //set lattice
    Lattice2D *grid = new Lattice2D(DIM, length, 10, 50, false, true, 0., "cylindrical");
    //Lattice1D *grid = new Lattice1D(DIM, length, false, "cylindrical");
    //set initial state
    State *state;
    if (IMAG_TIME) {
        state = new State(grid, angular_momentum);
        state->init_state(const_ini_state);
    }
    else {
        state = new BesselState(grid, angular_momentum);
    }
    //set Hamiltonian
    Potential *potential = new Potential(grid, const_potential);
    Hamiltonian *hamiltonian = new Hamiltonian(grid, potential);
    //set evolution
    Solver *solver = new Solver(grid, state, hamiltonian, delta_t, KERNEL_TYPE);

    //set file output directory
    stringstream file_info, fileprefix;
    string dirname = "cyl_coordinate";
    mkdir(dirname.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    file_info  << dirname << "/file_info.txt";
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

    //get and stamp phase
    fileprefix.str("");
    fileprefix << dirname << "/" << 0 << "-" << grid->mpi_rank;
    state->write_phase(fileprefix.str());

    //get and stamp particles density
    state->write_particle_density(fileprefix.str());

    if (grid->mpi_rank == 0) {
        out << std::setw(11) << "time" << std::setw(14) << "squared norm" << std::setw(14) << "tot energy" << std::setw(14) << "kin energy" << std::setw(14) << "pot energy"
            << std::setw(14) << "<X>" << std::setw(14) << "<(X-<X>)^2>" << std::setw(14) << "<Y>" << std::setw(14) << "<(Y-<Y>)^2>"
            << std::setw(14) << "<Px>" << std::setw(14) << "<(Px-<Px>)^2>" << std::setw(14) << "<Py>" << std::setw(14) << "<(Py-<Py>)^2>\n";
        out << std::setw(11) << "0" << std::setw(14) << norm2 << std::setw(14) << std::setw(14) << tot_energy << std::setw(14) << kin_energy << std::setw(14) << solver->get_potential_energy() << std::setw(14)
            << mean_X << std::setw(14) << var_X << std::setw(14) << mean_Y << std::setw(14) << var_Y << std::setw(14)
            << mean_Px << std::setw(14) << var_Px << std::setw(14) << mean_Py << std::setw(14) << var_Py << endl;
    }

    struct timeval start, end;
    for (int count_snap = 0; count_snap < SNAPSHOTS; count_snap++) {

        gettimeofday(&start, NULL);
        solver->evolve(ITERATIONS, IMAG_TIME);
        gettimeofday(&end, NULL);
        time = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
        tot_time += time;

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
            out << std::setw(11) << (count_snap + 1) * ITERATIONS * delta_t << std::setw(14) << norm2 << std::setw(14) << tot_energy << std::setw(14) << kin_energy << std::setw(14) << solver->get_potential_energy() << std::setw(14) <<
                mean_X << std::setw(14) << var_X << std::setw(14) << mean_Y << std::setw(14) << var_Y << std::setw(14) <<
                mean_Px << std::setw(14) << var_Px << std::setw(14) << mean_Py << std::setw(14) << var_Py << endl;
        }

        //stamp phase and particles density
        if((count_snap + 1) % SNAP_PER_STAMP == 0.) {
            //get and stamp phase
            fileprefix.str("");
            fileprefix << dirname << "/" << ITERATIONS * (count_snap + 1) << "-" << grid->mpi_rank;
            state->write_phase(fileprefix.str());

            //get and stamp particles density
            state->write_particle_density(fileprefix.str());
        }
    }
    out.close();

    if (grid->mpi_rank == 0) {
        cout << "TROTTER " << DIM << "x" << 10 << " kernel:" << KERNEL_TYPE << " np:" << grid->mpi_procs << " time:" << tot_time << " usec" << endl;
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
