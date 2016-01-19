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
#include "trottersuzuki.h"
#ifdef HAVE_MPI
#include <mpi.h>
#endif

int main(int argc, char** argv) {
    int dim = 640, iterations = 10*1000;
    string kernel_type = "cpu";
    const double particle_mass = 1., length=50;
    bool imag_time = false;
    double norm = 1.5;
	  double delta_t = 5.e-5;

#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif

    Lattice *grid = new Lattice(dim, length, length);
    State *state = new SinusoidState(grid, 1, 1);
    Hamiltonian *hamiltonian = new Hamiltonian(grid, NULL, particle_mass);
    Solver *solver = new Solver(grid, state, hamiltonian, delta_t, kernel_type);

    solver->evolve(iterations, imag_time);

    if (grid->mpi_rank == 0) {
        double threshold_E = 0.1;
        double threshold_P = 3.;
        double expected_Px = 0.;
        double expected_Py = 0.;
        double expected_E = (2. * M_PI / dim) * (2. * M_PI / dim);
        //get expected values
        double tot_energy = solver->get_total_energy();

        double mean_Px = state->get_mean_px();
        double var_Px = state->get_mean_pxpx() - state->get_mean_px() * state->get_mean_px();
        double mean_Py = state->get_mean_py();
        double var_Py = state->get_mean_pypy() - state->get_mean_py() * state->get_mean_py();

        if(std::abs(tot_energy - expected_E) < threshold_E)
            std::cout << "Energy -> OK\tsigma: " << std::abs(tot_energy - expected_E)  << std::endl;
        else
            std::cout << "Energy value is not the one theoretically expected: sigma " << std::abs(tot_energy - expected_E)  << std::endl;
        if(std::abs(mean_Px - expected_Px) / var_Px < threshold_P)
            std::cout << "Momentum Px -> OK\tsigma: " << std::abs(mean_Px - expected_Px) / var_Px << std::endl;
        else
            std::cout << "Momentum Px value is not the one theoretically expected: sigma " << std::abs(mean_Px - expected_Px) / var_Px << std::endl;
        if(std::abs(mean_Py - expected_Py) / var_Py < threshold_P)
            std::cout << "Momentum Py -> OK\tsigma: " << std::abs(mean_Py - expected_Py) / var_Py << std::endl;
        else
            std::cout << "Momentum Py value is not the one theoretically expected: sigma " << std::abs(mean_Py - expected_Py) / var_Py << std::endl;
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
