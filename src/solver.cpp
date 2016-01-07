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
#include "trottersuzuki.h"
#include "common.h"
#include "kernel.h"
#include <iostream>

Solver::Solver(Lattice *_grid, State *_state, Hamiltonian *_hamiltonian,
               double _delta_t, string _kernel_type):
    grid(_grid), state(_state), hamiltonian(_hamiltonian), delta_t(_delta_t),
    kernel_type(_kernel_type) {
    external_pot_real = new double* [2];
    external_pot_imag = new double* [2];
    external_pot_real[0] = new double[grid->dim_x * grid->dim_y];
    external_pot_imag[0] = new double[grid->dim_x * grid->dim_y];
    external_pot_real[1] = NULL;
    external_pot_imag[1] = NULL;
    norm2[0] = -1.;
    norm2[1] = 0;
    state_b = NULL;
    single_component = true;
}

Solver::Solver(Lattice *_grid, State *state1, State *state2, 
               Hamiltonian2Component *_hamiltonian,
               double _delta_t, string _kernel_type):
    grid(_grid), state(state1), state_b(state2), hamiltonian(_hamiltonian), delta_t(_delta_t),
    kernel_type(_kernel_type) {
    external_pot_real = new double* [2];
    external_pot_imag = new double* [2];
    external_pot_real[0] = new double[grid->dim_x * grid->dim_y];
    external_pot_imag[0] = new double[grid->dim_x * grid->dim_y];
    external_pot_real[1] = new double[grid->dim_x * grid->dim_y];
    external_pot_imag[1] = new double[grid->dim_x * grid->dim_y];
    norm2[0] = -1.;
    norm2[1] = -1.;
    kernel = NULL;
    current_evolution_time = 0;
    single_component = false;
}

Solver::~Solver() {
    delete [] external_pot_real[0];
    delete [] external_pot_imag[0];
    delete [] external_pot_real[1];
    delete [] external_pot_imag[1];
    delete [] external_pot_real;
    delete [] external_pot_imag;
    if (kernel != NULL) {
        delete kernel;
    }
}

void Solver::initialize_exp_potential(double delta_t, int which) {
      double particle_mass;
      if (which == 0) {
          particle_mass = hamiltonian->mass;
      } else {
          particle_mass = static_cast<Hamiltonian2Component*>(hamiltonian)->mass_b;
      }
      complex<double> tmp;
      for (int y = 0, idy = grid->start_y; y < grid->dim_y; y++, idy++) {
          for (int x = 0, idx = grid->start_x; x < grid->dim_x; x++, idx++) {
              if(imag_time)
                  tmp = exp(complex<double> (-delta_t*hamiltonian->external_pot[y * grid->dim_x + x], 0.));
              else
                  tmp = exp(complex<double> (0., -delta_t*hamiltonian->external_pot[y * grid->dim_x + x]));
              external_pot_real[which][y * grid->dim_x + x] = real(tmp);
              external_pot_imag[which][y * grid->dim_x + x] = imag(tmp);
          }
      }
}

void Solver::init_kernel() {
    if (kernel != NULL) {
        delete kernel;
    }
    if (kernel_type == "cpu") {
      if (single_component) {
          kernel = new CPUBlock(grid, state, hamiltonian, external_pot_real[0], external_pot_imag[0], h_a[0], h_b[0], delta_t, norm2[0], imag_time);
      } else {
          kernel = new CPUBlock(grid, state, state_b, static_cast<Hamiltonian2Component*>(hamiltonian), external_pot_real, external_pot_imag, h_a, h_b, delta_t, norm2, imag_time);
      }
    } else if (!single_component) {
        my_abort("Two-component Hamiltonians only work with the CPU kernel!");      
    } else if (kernel_type == "gpu") {
#ifdef CUDA
        kernel = new CC2Kernel(grid, state, hamiltonian, external_pot_real[0], external_pot_imag[0], h_a[0], h_b[0], delta_t, norm2[0], imag_time);
#else
        my_abort("Compiled without CUDA\n");
#endif
    } else if (kernel_type == "hybrid") {
#ifdef CUDA
        kernel = new HybridKernel(grid, state, hamiltonian, external_pot_real[0], external_pot_imag[0], h_a[0], h_b[0], delta_t, norm2[0], imag_time);
#else
        my_abort("Compiled without CUDA\n");
#endif
    } else {
        my_abort("Unknown kernel\n");
    }

}

void Solver::evolve(int iterations, bool _imag_time) {
    if (_imag_time != imag_time||kernel == NULL) {
        imag_time = _imag_time;
        if(imag_time) {
            h_a[0] = cosh(delta_t / (4. * hamiltonian->mass * grid->delta_x * grid->delta_y));
            h_b[0] = sinh(delta_t / (4. * hamiltonian->mass * grid->delta_x * grid->delta_y));
            if (hamiltonian->evolve_potential != 0) {
                 hamiltonian->update_potential(delta_t, 0);
            }
            initialize_exp_potential(delta_t, 0);
            norm2[0] = state->calculate_squared_norm();
            if (!single_component) {
                h_a[1] = cosh(delta_t / (4. * static_cast<Hamiltonian2Component*>(hamiltonian)->mass_b * grid->delta_x * grid->delta_y));
                h_b[1] = sinh(delta_t / (4. * static_cast<Hamiltonian2Component*>(hamiltonian)->mass_b * grid->delta_x * grid->delta_y));
                initialize_exp_potential(delta_t, 1);
                norm2[1] = state_b->calculate_squared_norm();
            }
        }
        else {
            h_a[0] = cos(delta_t / (4. * hamiltonian->mass * grid->delta_x * grid->delta_y));
            h_b[0] = sin(delta_t / (4. * hamiltonian->mass * grid->delta_x * grid->delta_y));
            if (hamiltonian->evolve_potential != 0) {
                 hamiltonian->update_potential(delta_t, 0);
            }
            initialize_exp_potential(delta_t, 0);
            if (!single_component) {
                h_a[1] = cos(delta_t / (4. * static_cast<Hamiltonian2Component*>(hamiltonian)->mass_b * grid->delta_x * grid->delta_y));
                h_b[1] = sin(delta_t / (4. * static_cast<Hamiltonian2Component*>(hamiltonian)->mass_b * grid->delta_x * grid->delta_y));
                initialize_exp_potential(delta_t, 1);
            }
        }
        init_kernel();
    }
    // Main loop
    double var = 0.5;
    if (!single_component) {
        kernel->rabi_coupling(var, delta_t);
    }
    var = 1.;
    // Main loop
    for (int i = 0; i < iterations; i++) {
        if (hamiltonian->evolve_potential != NULL && i > 0) {
             hamiltonian->update_potential(delta_t, i);
             initialize_exp_potential(delta_t, 0);
             kernel->update_potential(external_pot_real[0], external_pot_imag[0]);
        }
        //first wave function
        kernel->run_kernel_on_halo();
        if (i != iterations - 1) {
            kernel->start_halo_exchange();
        }
        kernel->run_kernel();
        if (i != iterations - 1) {
            kernel->finish_halo_exchange();
        }
        kernel->wait_for_completion();
        norm2[0] = -.1;
        if (!single_component) {
            //second wave function
            kernel->run_kernel_on_halo();
            if (i != iterations - 1) {
              kernel->start_halo_exchange();
            }
            kernel->run_kernel();
            if (i != iterations - 1) {
              kernel->finish_halo_exchange();
            }
            kernel->wait_for_completion();
            if (i == iterations - 1) {
                var = 0.5;
            }
            kernel->rabi_coupling(var, delta_t);
            kernel->normalization();
            norm2[1] = -.1;
        }
        current_evolution_time += delta_t;
    }
    if (single_component) {
        kernel->get_sample(grid->dim_x, 0, 0, grid->dim_x, grid->dim_y, state->p_real, state->p_imag);      
    } else {
        kernel->get_sample(grid->dim_x, 0, 0, grid->dim_x, grid->dim_y, state->p_real, state->p_imag, state_b->p_real, state_b->p_imag);
    }
}

double Solver::calculate_kinetic_energy(int which, double _norm2) {

    int ini_halo_x = grid->inner_start_x - grid->start_x;
    int ini_halo_y = grid->inner_start_y - grid->start_y;
    int end_halo_x = grid->end_x - grid->inner_end_x;
    int end_halo_y = grid->end_y - grid->inner_end_y;
    int tile_width = grid->end_x - grid->start_x;
    State *current_state;
    if (which == 0) {
        current_state = state;
    } else {
        current_state = state_b;
    }
    if (_norm2 == 0 && norm2[which] == -1) {
        norm2[which] = current_state->calculate_squared_norm();
    } else if (_norm2 != 0) {
        norm2[which] = _norm2;
    }
    
    complex<double> sum = 0;
    complex<double> cost_E = -1. / (2. * hamiltonian->mass);
    complex<double> psi_up, psi_down, psi_center, psi_left, psi_right;
    for (int i = grid->inner_start_y - grid->start_y + (ini_halo_y == 0);
         i < grid->inner_end_y - grid->start_y - (end_halo_y == 0); i++) {
        for (int j = grid->inner_start_x - grid->start_x + (ini_halo_x == 0);
             j < grid->inner_end_x - grid->start_x - (end_halo_x == 0); j++) {
            psi_center = complex<double> (current_state->p_real[i * tile_width + j],
                                             current_state->p_imag[i * tile_width + j]);
            psi_up = complex<double> (current_state->p_real[(i - 1) * tile_width + j],
                                         current_state->p_imag[(i - 1) * tile_width + j]);
            psi_down = complex<double> (current_state->p_real[(i + 1) * tile_width + j],
                                           current_state->p_imag[(i + 1) * tile_width + j]);
            psi_right = complex<double> (current_state->p_real[i * tile_width + j + 1],
                                            current_state->p_imag[i * tile_width + j + 1]);
            psi_left = complex<double> (current_state->p_real[i * tile_width + j - 1],
                                           current_state->p_imag[i * tile_width + j - 1]);
            sum += conj(psi_center) * (cost_E *
            (complex<double>(1./(grid->delta_x * grid->delta_x), 0.) *
               (psi_right + psi_left - psi_center * complex<double>(2., 0.)) +
             complex<double>(1./(grid->delta_y * grid->delta_y), 0.) *
               (psi_down + psi_up - psi_center * complex<double> (2., 0.))));
        }
    }
    double kinetic_energy =  real(sum / norm2[which]) * grid->delta_x * grid->delta_y;
#ifdef HAVE_MPI
    double *sums = new double[grid->mpi_procs];
    MPI_Allgather(&kinetic_energy, 1, MPI_DOUBLE, sums, 1, MPI_DOUBLE, grid->cartcomm);
    kinetic_energy = 0.;
    for(int i = 0; i < grid->mpi_procs; i++)
        kinetic_energy += sums[i];
    delete [] sums;
#endif
    return kinetic_energy;
}

double Solver::calculate_rotational_energy(int which, double _norm2) {

    int ini_halo_x = grid->inner_start_x - grid->start_x;
    int ini_halo_y = grid->inner_start_y - grid->start_y;
    int end_halo_x = grid->end_x - grid->inner_end_x;
    int end_halo_y = grid->end_y - grid->inner_end_y;
    int tile_width = grid->end_x - grid->start_x;
    State *current_state;
    if (which == 0) {
        current_state = state;
    } else {
        current_state = state_b;
    }
    if (_norm2 == 0 && norm2[which] == -1) {
        norm2[which] = current_state->calculate_squared_norm();
    } else if (_norm2 != 0) {
        norm2[which] = _norm2;
    }

    complex<double> sum = 0;
    complex<double> psi_up, psi_down, psi_center, psi_left, psi_right;
    complex<double> rot_y, rot_x;
    double cost_rot_x = 0.5 * hamiltonian->angular_velocity * grid->delta_y / grid->delta_x;
    double cost_rot_y = 0.5 * hamiltonian->angular_velocity * grid->delta_x / grid->delta_y;
    for(int i = grid->inner_start_y - grid->start_y + (ini_halo_y == 0),
          y = grid->inner_start_y + (ini_halo_y == 0);
          i <grid->inner_end_y - grid->start_y - (end_halo_y == 0); i++, y++) {
        for(int j = grid->inner_start_x - grid->start_x + (ini_halo_x == 0),
            x = grid->inner_start_x + (ini_halo_x == 0);
            j < grid->inner_end_x - grid->start_x - (end_halo_x == 0); j++, x++) {
            psi_center = complex<double> (current_state->p_real[i * tile_width + j],
                                             current_state->p_imag[i * tile_width + j]);
            psi_up = complex<double> (current_state->p_real[(i - 1) * tile_width + j],
                                         current_state->p_imag[(i - 1) * tile_width + j]);
            psi_down = complex<double> (current_state->p_real[(i + 1) * tile_width + j],
                                           current_state->p_imag[(i + 1) * tile_width + j]);
            psi_right = complex<double> (current_state->p_real[i * tile_width + j + 1],
                                            current_state->p_imag[i * tile_width + j + 1]);
            psi_left = complex<double> (current_state->p_real[i * tile_width + j - 1],
                                           current_state->p_imag[i * tile_width + j - 1]);

            rot_x = complex<double>(0., cost_rot_x * (y - hamiltonian->rot_coord_y));
            rot_y = complex<double>(0., cost_rot_y * (x - hamiltonian->rot_coord_x));
            sum += conj(psi_center) * (rot_y * (psi_down - psi_up) - rot_x * (psi_right - psi_left)) ;
        }
    }
    double energy_rot = real(sum / norm2[which]) * grid->delta_x * grid->delta_y;

#ifdef HAVE_MPI
    double *sums = new double[grid->mpi_procs];
    MPI_Allgather(&energy_rot, 1, MPI_DOUBLE, sums, 1, MPI_DOUBLE, grid->cartcomm);
    energy_rot = 0.;
    for(int i = 0; i < grid->mpi_procs; i++)
        energy_rot += sums[i];
    delete [] sums;
#endif
    return energy_rot;
}
     
double Solver::calculate_rabi_coupling_energy(double _norm2) {
    int ini_halo_x = grid->inner_start_x - grid->start_x;
    int ini_halo_y = grid->inner_start_y - grid->start_y;
    int end_halo_x = grid->end_x - grid->inner_end_x;
    int end_halo_y = grid->end_y - grid->inner_end_y;
    int tile_width = grid->end_x - grid->start_x;

    double total_norm2;
    if (_norm2 != 0) {
        total_norm2 = _norm2;
    } else {
        if (norm2[0] == -1) {
            norm2[0] = state->calculate_squared_norm();
        }
        if (norm2[1] == -1) {
            norm2[1] = state_b->calculate_squared_norm();
        }
        total_norm2 = norm2[0] + norm2[1];
    }


    complex<double> sum = 0;
    complex<double> psi_center_a, psi_center_b;
    complex<double> omega = complex<double> (static_cast<Hamiltonian2Component*>(hamiltonian)->omega_r,
                                             static_cast<Hamiltonian2Component*>(hamiltonian)->omega_i);

    for (int i = grid->inner_start_y - grid->start_y + (ini_halo_y == 0),
       y = grid->inner_start_y + (ini_halo_y == 0);
       i < grid->inner_end_y - grid->start_y - (end_halo_y == 0); i++, y++) {
        for (int j = grid->inner_start_x - grid->start_x + (ini_halo_x == 0),
             x = grid->inner_start_x + (ini_halo_x == 0);
             j < grid->inner_end_x - grid->start_x - (end_halo_x == 0); j++, x++) {
            psi_center_a = complex<double> (state->p_real[i * tile_width + j],
                                                 state->p_imag[i * tile_width + j]);
            psi_center_b = complex<double> (state_b->p_real[i * tile_width + j],
                                                 state_b->p_imag[i * tile_width + j]);
            sum += conj(psi_center_a) * psi_center_b * omega +  conj(psi_center_b) * psi_center_a * conj(omega);
        }
    }

    double energy_rabi = real(sum / total_norm2) * grid->delta_x * grid->delta_y;
#ifdef HAVE_MPI
    double *sums = new double[grid->mpi_procs];
    MPI_Allgather(&energy_rabi, 1, MPI_DOUBLE, sums, 1, MPI_DOUBLE, grid->cartcomm);
    energy_rabi = 0.;
    for(int i = 0; i < grid->mpi_procs; i++)
        energy_rabi += sums[i];
    delete [] sums;
#endif
    return energy_rabi;
}

double Solver::calculate_ab_energy(double _norm2) {
    int ini_halo_x = grid->inner_start_x - grid->start_x;
    int ini_halo_y = grid->inner_start_y - grid->start_y;
    int end_halo_x = grid->end_x - grid->inner_end_x;
    int end_halo_y = grid->end_y - grid->inner_end_y;
    int tile_width = grid->end_x - grid->start_x;

    double total_norm2;
    if (_norm2 != 0) {
        total_norm2 = _norm2;
    } else {
        if (norm2[0] == -1) {
            norm2[0] = state->calculate_squared_norm();
        }
        if (norm2[1] == -1) {
            norm2[1] = state_b->calculate_squared_norm();
        }
        total_norm2 = norm2[0] + norm2[1];
    }

    complex<double> sum = 0;
    complex<double> psi_center_a, psi_center_b;

    for (int i = grid->inner_start_y - grid->start_y + (ini_halo_y == 0),
       y = grid->inner_start_y + (ini_halo_y == 0);
       i < grid->inner_end_y - grid->start_y - (end_halo_y == 0); i++, y++) {
        for (int j = grid->inner_start_x - grid->start_x + (ini_halo_x == 0),
             x = grid->inner_start_x + (ini_halo_x == 0);
             j < grid->inner_end_x - grid->start_x - (end_halo_x == 0); j++, x++) {
            psi_center_a = complex<double> (state->p_real[i * tile_width + j], state->p_imag[i * tile_width + j]);
            psi_center_b = complex<double> (state_b->p_real[i * tile_width + j], state_b->p_imag[i * tile_width + j]);
            sum += conj(psi_center_a) * psi_center_a * conj(psi_center_b) * psi_center_b * complex<double> (static_cast<Hamiltonian2Component*>(hamiltonian)->coupling_ab);
        }
    }
    double energy_ab = real(sum / total_norm2) * grid->delta_x * grid->delta_y;
#ifdef HAVE_MPI
    double *sums = new double[grid->mpi_procs];
    MPI_Allgather(&energy_ab, 1, MPI_DOUBLE, sums, 1, MPI_DOUBLE, grid->cartcomm);
    energy_ab = 0.;
    for(int i = 0; i < grid->mpi_procs; i++)
        energy_ab += sums[i];
    delete [] sums;
#endif
    return energy_ab;
}

double Solver::calculate_total_energy_single_state(int which, double (*hamilt_pot)(double x, double y), double _norm2) {

    int ini_halo_x = grid->inner_start_x - grid->start_x;
    int ini_halo_y = grid->inner_start_y - grid->start_y;
    int end_halo_x = grid->end_x - grid->inner_end_x;
    int end_halo_y = grid->end_y - grid->inner_end_y;
    int tile_width = grid->end_x - grid->start_x;
    State *current_state;
    double *external_pot;
    double coupling;
    double mass;
    if (which == 0) {
        current_state = state;
        external_pot = hamiltonian->external_pot;
        coupling = hamiltonian->coupling_a;
        mass = hamiltonian->mass;
    } else {
        current_state = state_b;
        external_pot = static_cast<Hamiltonian2Component*>(hamiltonian)->external_pot_b;
        coupling = static_cast<Hamiltonian2Component*>(hamiltonian)->coupling_b;
        mass = static_cast<Hamiltonian2Component*>(hamiltonian)->mass_b;
    }
    if (_norm2 == 0 && norm2[which] == -1) {
        norm2[which] = current_state->calculate_squared_norm();
    } else if (_norm2 != 0) {
        norm2[which] = _norm2;
    }
    complex<double> sum = 0;
    complex<double> cost_E = -1. / (2. * mass);
    complex<double> psi_up, psi_down, psi_center, psi_left, psi_right;
    complex<double> rot_y, rot_x;
    double cost_rot_x = 0.5 * hamiltonian->angular_velocity * grid->delta_y / grid->delta_x;
    double cost_rot_y = 0.5 * hamiltonian->angular_velocity * grid->delta_x / grid->delta_y;
    
    double delta_x = grid->delta_x, delta_y = grid->delta_y;
    for (int i = grid->inner_start_y - grid->start_y + (ini_halo_y == 0),
         y = grid->inner_start_y + (ini_halo_y == 0);
         i < grid->inner_end_y - grid->start_y - (end_halo_y == 0); i++, y++) {
        for (int j = grid->inner_start_x - grid->start_x + (ini_halo_x == 0),
             x = grid->inner_start_x + (ini_halo_x == 0);
             j < grid->inner_end_x - grid->start_x - (end_halo_x == 0); j++, x++) {
            complex<double> potential_term;
            if(external_pot == NULL) {
                potential_term = complex<double> (hamilt_pot(x * delta_x, y * delta_y), 0.);
            } else {
                potential_term = complex<double> (external_pot[y * grid->global_dim_x + x], 0.);
            }
            psi_center = complex<double> (current_state->p_real[i * tile_width + j],
                                               current_state->p_imag[i * tile_width + j]);
            psi_up = complex<double> (current_state->p_real[(i - 1) * tile_width + j],
                                           current_state->p_imag[(i - 1) * tile_width + j]);
            psi_down = complex<double> (current_state->p_real[(i + 1) * tile_width + j],
                                             current_state->p_imag[(i + 1) * tile_width + j]);
            psi_right = complex<double> (current_state->p_real[i * tile_width + j + 1],
                                              current_state->p_imag[i * tile_width + j + 1]);
            psi_left = complex<double> (current_state->p_real[i * tile_width + j - 1],
                                             current_state->p_imag[i * tile_width + j - 1]);

            rot_x = complex<double>(0., cost_rot_x * (y - hamiltonian->rot_coord_y));
            rot_y = complex<double>(0., cost_rot_y * (x - hamiltonian->rot_coord_x));
            sum += conj(psi_center) * (cost_E * (
                complex<double> (1. / (grid->delta_x * grid->delta_x), 0.) *
                  (psi_right + psi_left - psi_center * complex<double> (2., 0.)) +
                complex<double> (1. / (grid->delta_y * grid->delta_y), 0.) *
                  (psi_down + psi_up - psi_center * complex<double> (2., 0.))) +
                psi_center * potential_term +
                psi_center * psi_center * conj(psi_center) * complex<double> (0.5 * coupling, 0.) +
                rot_y * (psi_down - psi_up) - rot_x * (psi_right - psi_left));
        }
    }
    double total_energy = real(sum / norm2[which]) * grid->delta_x * grid->delta_y;
#ifdef HAVE_MPI
    double *sums = new double[grid->mpi_procs];
    MPI_Allgather(&total_energy, 1, MPI_DOUBLE, sums, 1, MPI_DOUBLE, grid->cartcomm);
    total_energy = 0.;
    for(int i = 0; i < grid->mpi_procs; i++)
        total_energy += sums[i];
    delete [] sums;
#endif
    return total_energy;
}


double Solver::calculate_total_energy(double _norm2,
                                      double (*hamilt_pot_a)(double x, double y),
                                      double (*hamilt_pot_b)(double x, double y)) {

    double sum = 0;
    double total_norm2;
    if (_norm2 != 0) {
        total_norm2 = _norm2;
    } else {
        if (norm2[0] == -1) {
            norm2[0] = state->calculate_squared_norm();
        }
        if (norm2[1] == -1) {
            norm2[1] = state_b->calculate_squared_norm();
        }
        total_norm2 = norm2[0] + norm2[1];
    }
    sum += calculate_total_energy_single_state(0, hamilt_pot_a, total_norm2);
    if (!single_component) {
        sum += calculate_total_energy_single_state(1, hamilt_pot_b, total_norm2);
        sum += calculate_ab_energy(total_norm2);
        sum += calculate_rabi_coupling_energy(total_norm2);
    }
    return sum;
}
