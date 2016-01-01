/**
 * Distributed Trotter-Suzuki solver
 * Copyright (C) 2015 Luca Calderaro, 2012-2015 Peter Wittek,
 * 2010-2012 Carlos Bederián
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

double calculate_total_energy(Lattice *grid, State *state,
                              Hamiltonian *hamiltonian,
                              double (*hamilt_pot)(int x, int y, Lattice *grid),
                              double * external_pot, double norm2, bool global) {

    int ini_halo_x = grid->inner_start_x - grid->start_x;
    int ini_halo_y = grid->inner_start_y - grid->start_y;
    int end_halo_x = grid->end_x - grid->inner_end_x;
    int end_halo_y = grid->end_y - grid->inner_end_y;
    int tile_width = grid->end_x - grid->start_x;

    if(norm2 == 0)
        norm2 = state->calculate_squared_norm(false);

    complex<double> sum = 0;
    complex<double> cost_E = -1. / (2. * hamiltonian->mass);
    complex<double> psi_up, psi_down, psi_center, psi_left, psi_right;
    complex<double> rot_y, rot_x;
    double cost_rot_x = 0.5 * hamiltonian->omega * grid->delta_y / grid->delta_x;
    double cost_rot_y = 0.5 * hamiltonian->omega * grid->delta_x / grid->delta_y;

    for (int i = grid->inner_start_y - grid->start_y + (ini_halo_y == 0),
         y = grid->inner_start_y + (ini_halo_y == 0);
         i < grid->inner_end_y - grid->start_y - (end_halo_y == 0); i++, y++) {
        for (int j = grid->inner_start_x - grid->start_x + (ini_halo_x == 0),
             x = grid->inner_start_x + (ini_halo_x == 0);
             j < grid->inner_end_x - grid->start_x - (end_halo_x == 0); j++, x++) {
            complex<double> potential_term;
            if(external_pot == NULL) {
                potential_term = complex<double> (hamilt_pot(x, y, grid), 0.);
            } else {
                potential_term = complex<double> (external_pot[y * grid->global_dim_x + x], 0.);
            }
            psi_center = complex<double> (state->p_real[i * tile_width + j],
                                               state->p_imag[i * tile_width + j]);
            psi_up = complex<double> (state->p_real[(i - 1) * tile_width + j],
                                           state->p_imag[(i - 1) * tile_width + j]);
            psi_down = complex<double> (state->p_real[(i + 1) * tile_width + j],
                                             state->p_imag[(i + 1) * tile_width + j]);
            psi_right = complex<double> (state->p_real[i * tile_width + j + 1],
                                              state->p_imag[i * tile_width + j + 1]);
            psi_left = complex<double> (state->p_real[i * tile_width + j - 1],
                                             state->p_imag[i * tile_width + j - 1]);

            rot_x = complex<double>(0., cost_rot_x * (y - hamiltonian->rot_coord_y));
            rot_y = complex<double>(0., cost_rot_y * (x - hamiltonian->rot_coord_x));
            sum += conj(psi_center) * (cost_E * (
                complex<double> (1. / (grid->delta_x * grid->delta_x), 0.) *
                  (psi_right + psi_left - psi_center * complex<double> (2., 0.)) +
                complex<double> (1. / (grid->delta_y * grid->delta_y), 0.) *
                  (psi_down + psi_up - psi_center * complex<double> (2., 0.))) +
                psi_center * potential_term +
                psi_center * psi_center * conj(psi_center) * complex<double> (0.5 * hamiltonian->coupling_a, 0.) +
                rot_y * (psi_down - psi_up) - rot_x * (psi_right - psi_left));
        }
    }
  double total_energy = real(sum / norm2) * grid->delta_x * grid->delta_y;
#ifdef HAVE_MPI
    if (global) {
        double *sums = new double[grid->mpi_procs];
        MPI_Allgather(&total_energy, 1, MPI_DOUBLE, sums, 1, MPI_DOUBLE, grid->cartcomm);
        sums[0] = total_energy;
        total_energy = 0.;
        for(int i = 0; i < grid->mpi_procs; i++)
            total_energy += sums[i];
        delete [] sums;
    }
#endif
  return total_energy;
}

double calculate_kinetic_energy(Lattice *grid, State *state, Hamiltonian *hamiltonian, double norm2, bool global) {

  int ini_halo_x = grid->inner_start_x - grid->start_x;
  int ini_halo_y = grid->inner_start_y - grid->start_y;
  int end_halo_x = grid->end_x - grid->inner_end_x;
  int end_halo_y = grid->end_y - grid->inner_end_y;
  int tile_width = grid->end_x - grid->start_x;

  if(norm2 == 0)
      norm2 = state->calculate_squared_norm(false);

  complex<double> sum = 0;
  complex<double> cost_E = -1. / (2. * hamiltonian->mass);
  complex<double> psi_up, psi_down, psi_center, psi_left, psi_right;
  for (int i = grid->inner_start_y - grid->start_y + (ini_halo_y == 0);
       i < grid->inner_end_y - grid->start_y - (end_halo_y == 0); i++) {
      for (int j = grid->inner_start_x - grid->start_x + (ini_halo_x == 0);
           j < grid->inner_end_x - grid->start_x - (end_halo_x == 0); j++) {
          psi_center = complex<double> (state->p_real[i * tile_width + j],
                                             state->p_imag[i * tile_width + j]);
          psi_up = complex<double> (state->p_real[(i - 1) * tile_width + j],
                                         state->p_imag[(i - 1) * tile_width + j]);
          psi_down = complex<double> (state->p_real[(i + 1) * tile_width + j],
                                           state->p_imag[(i + 1) * tile_width + j]);
          psi_right = complex<double> (state->p_real[i * tile_width + j + 1],
                                            state->p_imag[i * tile_width + j + 1]);
          psi_left = complex<double> (state->p_real[i * tile_width + j - 1],
                                           state->p_imag[i * tile_width + j - 1]);
          sum += conj(psi_center) * (cost_E *
            (complex<double>(1./(grid->delta_x * grid->delta_x), 0.) *
               (psi_right + psi_left - psi_center * complex<double>(2., 0.)) +
             complex<double>(1./(grid->delta_y * grid->delta_y), 0.) *
               (psi_down + psi_up - psi_center * complex<double> (2., 0.))));
    }
  }
  double kinetic_energy =  real(sum / norm2) * grid->delta_x * grid->delta_y;
#ifdef HAVE_MPI
    if (global) {
        double *sums = new double[grid->mpi_procs];
        MPI_Allgather(&kinetic_energy, 1, MPI_DOUBLE, sums, 1, MPI_DOUBLE, grid->cartcomm);
        sums[0] = kinetic_energy;
        kinetic_energy = 0.;
        for(int i = 0; i < grid->mpi_procs; i++)
            kinetic_energy += sums[i];
        delete [] sums;
    }
#endif
  return kinetic_energy;
}

double calculate_rotational_energy(Lattice *grid, State *state, Hamiltonian *hamiltonian, double norm2, bool global) {

  int ini_halo_x = grid->inner_start_x - grid->start_x;
  int ini_halo_y = grid->inner_start_y - grid->start_y;
  int end_halo_x = grid->end_x - grid->inner_end_x;
  int end_halo_y = grid->end_y - grid->inner_end_y;
  int tile_width = grid->end_x - grid->start_x;

  if(norm2 == 0)
      norm2 = state->calculate_squared_norm(false);
  complex<double> sum = 0;
  complex<double> psi_up, psi_down, psi_center, psi_left, psi_right;
  complex<double> rot_y, rot_x;
  double cost_rot_x = 0.5 * hamiltonian->omega * grid->delta_y / grid->delta_x;
  double cost_rot_y = 0.5 * hamiltonian->omega * grid->delta_x / grid->delta_y;
  for(int i = grid->inner_start_y - grid->start_y + (ini_halo_y == 0),
          y = grid->inner_start_y + (ini_halo_y == 0);
          i <grid->inner_end_y - grid->start_y - (end_halo_y == 0); i++, y++) {
      for(int j = grid->inner_start_x - grid->start_x + (ini_halo_x == 0),
              x = grid->inner_start_x + (ini_halo_x == 0);
              j < grid->inner_end_x - grid->start_x - (end_halo_x == 0); j++, x++) {
          psi_center = complex<double> (state->p_real[i * tile_width + j],
                                             state->p_imag[i * tile_width + j]);
          psi_up = complex<double> (state->p_real[(i - 1) * tile_width + j],
                                         state->p_imag[(i - 1) * tile_width + j]);
          psi_down = complex<double> (state->p_real[(i + 1) * tile_width + j],
                                           state->p_imag[(i + 1) * tile_width + j]);
          psi_right = complex<double> (state->p_real[i * tile_width + j + 1],
                                            state->p_imag[i * tile_width + j + 1]);
          psi_left = complex<double> (state->p_real[i * tile_width + j - 1],
                                           state->p_imag[i * tile_width + j - 1]);

          rot_x = complex<double>(0., cost_rot_x * (y - hamiltonian->rot_coord_y));
          rot_y = complex<double>(0., cost_rot_y * (x - hamiltonian->rot_coord_x));
          sum += conj(psi_center) * (rot_y * (psi_down - psi_up) - rot_x * (psi_right - psi_left)) ;
    }
  }

  double energy_rot = real(sum / norm2) * grid->delta_x * grid->delta_y;
#ifdef HAVE_MPI
    if (global) {
        double *sums = new double[grid->mpi_procs];
        MPI_Allgather(&energy_rot, 1, MPI_DOUBLE, sums, 1, MPI_DOUBLE, grid->cartcomm);
        sums[0] = energy_rot;
        energy_rot = 0.;
        for(int i = 0; i < grid->mpi_procs; i++)
            energy_rot += sums[i];
        delete [] sums;
    }
#endif
    return energy_rot;
}

void calculate_mean_position(Lattice *grid, State *state, int grid_origin_x, int grid_origin_y,
                             double *results, double norm2) {

    int ini_halo_x = grid->inner_start_x - grid->start_x;
    int ini_halo_y = grid->inner_start_y - grid->start_y;
    int end_halo_x = grid->end_x - grid->inner_end_x;
    int end_halo_y = grid->end_y - grid->inner_end_y;
    int tile_width = grid->end_x - grid->start_x;

    if(norm2 == 0)
        norm2 = state->calculate_squared_norm(false);

    complex<double> sum_x_mean = 0, sum_xx_mean = 0, sum_y_mean = 0, sum_yy_mean = 0;
    complex<double> psi_center;
    for (int i = grid->inner_start_y - grid->start_y + (ini_halo_y == 0);
         i < grid->inner_end_y - grid->start_y - (end_halo_y == 0); i++) {
        for (int j = grid->inner_start_x - grid->start_x + (ini_halo_x == 0);
             j < grid->inner_end_x - grid->start_x - (end_halo_x == 0); j++) {
            psi_center = complex<double> (state->p_real[i * tile_width + j], state->p_imag[i * tile_width + j]);
            sum_x_mean += conj(psi_center) * psi_center * complex<double>(grid->delta_x * (j - grid_origin_x), 0.);
            sum_y_mean += conj(psi_center) * psi_center * complex<double>(grid->delta_y * (i - grid_origin_y), 0.);
            sum_xx_mean += conj(psi_center) * psi_center * complex<double>(grid->delta_x * (j - grid_origin_x), 0.) * complex<double>(grid->delta_x * (j - grid_origin_x), 0.);
            sum_yy_mean += conj(psi_center) * psi_center * complex<double>(grid->delta_y * (i - grid_origin_y), 0.) * complex<double>(grid->delta_y * (i - grid_origin_y), 0.);
      }
    }

    results[0] = real(sum_x_mean / norm2) * grid->delta_x * grid->delta_y;
    results[2] = real(sum_y_mean / norm2) * grid->delta_x * grid->delta_y;
    results[1] = real(sum_xx_mean / norm2) * grid->delta_x * grid->delta_y - results[0] * results[0];
    results[3] = real(sum_yy_mean / norm2) * grid->delta_x * grid->delta_y - results[2] * results[2];
}

void calculate_mean_momentum(Lattice *grid, State *state, double *results,
                             double norm2) {

    int ini_halo_x = grid->inner_start_x - grid->start_x;
    int ini_halo_y = grid->inner_start_y - grid->start_y;
    int end_halo_x = grid->end_x - grid->inner_end_x;
    int end_halo_y = grid->end_y - grid->inner_end_y;
    int tile_width = grid->end_x - grid->start_x;

    if(norm2 == 0)
        norm2 = state->calculate_squared_norm(false);

    complex<double> sum_px_mean = 0, sum_pxpx_mean = 0, sum_py_mean = 0,
                         sum_pypy_mean = 0,
                         var_px = complex<double>(0., - 0.5 / grid->delta_x),
                         var_py = complex<double>(0., - 0.5 / grid->delta_y);
    complex<double> psi_up, psi_down, psi_center, psi_left, psi_right;
    for (int i = grid->inner_start_y - grid->start_y + (ini_halo_y == 0);
         i < grid->inner_end_y - grid->start_y - (end_halo_y == 0); i++) {
        for (int j = grid->inner_start_x - grid->start_x + (ini_halo_x == 0);
             j < grid->inner_end_x - grid->start_x - (end_halo_x == 0); j++) {
            psi_center = complex<double> (state->p_real[i * tile_width + j],
                                               state->p_imag[i * tile_width + j]);
            psi_up = complex<double> (state->p_real[(i - 1) * tile_width + j],
                                           state->p_imag[(i - 1) * tile_width + j]);
            psi_down = complex<double> (state->p_real[(i + 1) * tile_width + j],
                                             state->p_imag[(i + 1) * tile_width + j]);
            psi_right = complex<double> (state->p_real[i * tile_width + j + 1],
                                              state->p_imag[i * tile_width + j + 1]);
            psi_left = complex<double> (state->p_real[i * tile_width + j - 1],
                                             state->p_imag[i * tile_width + j - 1]);

            sum_px_mean += conj(psi_center) * (psi_right - psi_left);
            sum_py_mean += conj(psi_center) * (psi_up - psi_down);
            sum_pxpx_mean += conj(psi_center) * (psi_right - 2. * psi_center + psi_left);
            sum_pypy_mean += conj(psi_center) * (psi_up - 2. * psi_center + psi_down);
        }
    }

    sum_px_mean = sum_px_mean * var_px;
    sum_py_mean = sum_py_mean * var_py;
    sum_pxpx_mean = sum_pxpx_mean * (-1.)/(grid->delta_x * grid->delta_x);
    sum_pypy_mean = sum_pypy_mean * (-1.)/(grid->delta_y * grid->delta_y);

    results[0] = real(sum_px_mean / norm2) * grid->delta_x * grid->delta_y;
    results[2] = real(sum_py_mean / norm2) * grid->delta_x * grid->delta_y;
    results[1] = real(sum_pxpx_mean / norm2) * grid->delta_x * grid->delta_y - results[0] * results[0];
    results[3] = real(sum_pypy_mean / norm2) * grid->delta_x * grid->delta_y - results[2] * results[2];
}

double calculate_rabi_coupling_energy(Lattice *grid, State *state1, State *state2, double omega_r, double omega_i, double norm2) {
  int ini_halo_x = grid->inner_start_x - grid->start_x;
  int ini_halo_y = grid->inner_start_y - grid->start_y;
  int end_halo_x = grid->end_x - grid->inner_end_x;
  int end_halo_y = grid->end_y - grid->inner_end_y;
  int tile_width = grid->end_x - grid->start_x;

  if(norm2 == 0)
      norm2 = state1->calculate_squared_norm(false) +
              state2->calculate_squared_norm(false);

  complex<double> sum = 0;
  complex<double> psi_center_a, psi_center_b;
  complex<double> omega = complex<double> (omega_r, omega_i);

  for (int i = grid->inner_start_y - grid->start_y + (ini_halo_y == 0),
       y = grid->inner_start_y + (ini_halo_y == 0);
       i < grid->inner_end_y - grid->start_y - (end_halo_y == 0); i++, y++) {
      for (int j = grid->inner_start_x - grid->start_x + (ini_halo_x == 0),
           x = grid->inner_start_x + (ini_halo_x == 0);
           j < grid->inner_end_x - grid->start_x - (end_halo_x == 0); j++, x++) {
          psi_center_a = complex<double> (state1->p_real[i * tile_width + j],
                                               state1->p_imag[i * tile_width + j]);
          psi_center_b = complex<double> (state2->p_real[i * tile_width + j],
                                               state2->p_imag[i * tile_width + j]);
          sum += conj(psi_center_a) * psi_center_b * omega +  conj(psi_center_b) * psi_center_a * conj(omega);
    }
  }

  return real(sum / norm2) * grid->delta_x * grid->delta_y;
}

double calculate_ab_energy(Lattice *grid, State *state1, State *state2,
                 double coupling_const_ab, double norm2) {
  int ini_halo_x = grid->inner_start_x - grid->start_x;
  int ini_halo_y = grid->inner_start_y - grid->start_y;
  int end_halo_x = grid->end_x - grid->inner_end_x;
  int end_halo_y = grid->end_y - grid->inner_end_y;
  int tile_width = grid->end_x - grid->start_x;

  if(norm2 == 0)
      norm2 = state1->calculate_squared_norm(false) +
              state2->calculate_squared_norm(false);

  complex<double> sum = 0;
  complex<double> psi_center_a, psi_center_b;

  for (int i = grid->inner_start_y - grid->start_y + (ini_halo_y == 0),
       y = grid->inner_start_y + (ini_halo_y == 0);
       i < grid->inner_end_y - grid->start_y - (end_halo_y == 0); i++, y++) {
      for (int j = grid->inner_start_x - grid->start_x + (ini_halo_x == 0),
           x = grid->inner_start_x + (ini_halo_x == 0);
           j < grid->inner_end_x - grid->start_x - (end_halo_x == 0); j++, x++) {
          psi_center_a = complex<double> (state1->p_real[i * tile_width + j], state1->p_imag[i * tile_width + j]);
          psi_center_b = complex<double> (state2->p_real[i * tile_width + j], state2->p_imag[i * tile_width + j]);
          sum += conj(psi_center_a) * psi_center_a * conj(psi_center_b) * psi_center_b * complex<double> (coupling_const_ab);
    }
  }
  return real(sum / norm2) * grid->delta_x * grid->delta_y;
}

double calculate_total_energy(Lattice *grid, State *state1, State *state2,
                              Hamiltonian2Component *hamiltonian,
                              double (*hamilt_pot_a)(int x, int y, Lattice *grid),
                              double (*hamilt_pot_b)(int x, int y, Lattice *grid),
                              double **external_pot, double norm2, bool global) {

  if(external_pot == NULL) {
    external_pot = new double* [2];
    external_pot[0] = NULL;
    external_pot[1] = NULL;
  }
  double sum = 0;
  if(norm2 == 0)
    norm2 = state1->calculate_squared_norm() + state2->calculate_squared_norm();

  Hamiltonian *hamiltonian_b = new Hamiltonian(grid, hamiltonian->mass_b, hamiltonian->coupling_b,
                hamiltonian->coupling_ab, hamiltonian->angular_velocity,
                hamiltonian->rot_coord_x, hamiltonian->rot_coord_y,
                hamiltonian->omega, hamiltonian->external_pot);
  sum += calculate_total_energy(grid, state1, hamiltonian, hamilt_pot_a, external_pot[0], norm2);
  sum += calculate_total_energy(grid, state2, hamiltonian_b, hamilt_pot_b, external_pot[1], norm2);
  sum += calculate_ab_energy(grid, state1, state2, hamiltonian->coupling_ab, norm2);
  sum += calculate_rabi_coupling_energy(grid, state1, state2, hamiltonian->omega_r, hamiltonian->omega_i, norm2);
  delete hamiltonian_b;
  return sum;
}

void trotter(Lattice *grid, State *state, Hamiltonian *hamiltonian,
             double h_a, double h_b,
             double *external_pot_real, double *external_pot_imag,
             double delta_t,
             const int iterations,
             string kernel_type, double norm, bool imag_time) {
    // Initialize kernel
    ITrotterKernel * kernel;
    if (kernel_type == "cpu") {
        kernel = new CPUBlock(grid, state, hamiltonian, external_pot_real, external_pot_imag, h_a, h_b, delta_t, norm, imag_time);
    } else if (kernel_type == "gpu") {
#ifdef CUDA
        kernel = new CC2Kernel(grid, state, hamiltonian, external_pot_real, external_pot_imag, h_a, h_b, norm, imag_time);
#else
        if (grid->mpi_coords[0] == 0 && grid->mpi_coords[1] == 0) {
            cerr << "Compiled without CUDA\n";
        }
#ifdef HAVE_MPI
        MPI_Abort(MPI_COMM_WORLD, 2);
#else
        abort ();
#endif
#endif
    } else if (kernel_type == "hybrid") {
#ifdef CUDA
        kernel = new HybridKernel(grid, state, external_pot_real, external_pot_imag, h_a, h_b, coupling_const, norm, imag_time);
#else
        if (grid->mpi_coords[0] == 0 && grid->mpi_coords[1] == 0) {
            cerr << "Compiled without CUDA\n";
        }
#ifdef HAVE_MPI
        MPI_Abort(MPI_COMM_WORLD, 2);
#else
        abort();
#endif
#endif
    }

    // Main loop
    for (int i = 0; i < iterations; i++) {
        kernel->run_kernel_on_halo();
        if (i != iterations - 1) {
            kernel->start_halo_exchange();
        }
        kernel->run_kernel();
        if (i != iterations - 1) {
            kernel->finish_halo_exchange();
        }
        kernel->wait_for_completion();
    }

    kernel->get_sample(grid->dim_x, 0, 0, grid->dim_x, grid->dim_y, state->p_real, state->p_imag);

    delete kernel;
}

void trotter(Lattice *grid, State *state1, State *state2,
             Hamiltonian2Component *hamiltonian,
             double *h_a, double *h_b,
             double **external_pot_real, double **external_pot_imag,
             double delta_t,
             const int iterations,
             string kernel_type, double *norm, bool imag_time) {
    // Initialize kernel
    ITrotterKernel * kernel;
    if (kernel_type == "cpu") {
        kernel = new CPUBlock(grid, state1, state2, hamiltonian, external_pot_real, external_pot_imag, h_a, h_b, delta_t, norm, imag_time);
    }
    double var = 0.5;
    kernel->rabi_coupling(var, delta_t);
    var = 1.;

    // Main loop
    for (int i = 0; i < iterations; i++) {
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

    if (i == iterations - 1)
      var = 0.5;
    kernel->rabi_coupling(var, delta_t);

        kernel->normalization();
    }

    kernel->get_sample(grid->dim_x, 0, 0, grid->dim_x, grid->dim_y, state1->p_real, state1->p_imag, state2->p_real, state2->p_imag);

    delete kernel;
}

Solver::Solver(Lattice *_grid, State *_state, Hamiltonian *_hamiltonian, 
               double _delta_t, string _kernel_type):
    grid(_grid), state(_state), hamiltonian(_hamiltonian), delta_t(_delta_t),
    kernel_type(_kernel_type) {
    external_pot_real = new double* [2];
    external_pot_imag = new double* [2];
    external_pot_real[0] = NULL;
    external_pot_imag[0] = NULL;
    external_pot_real[1] = NULL;
    external_pot_imag[1] = NULL;
    norm2[0] = 0.;
    state_b = NULL;
    first_run = true;
    single_component = true;
}

Solver::~Solver() {
    delete [] external_pot_real[0];
    delete [] external_pot_imag[0];
    delete [] external_pot_real[1];
    delete [] external_pot_imag[1];
    delete [] external_pot_real;
    delete [] external_pot_imag;
}

void Solver::evolve(int iterations, bool _imag_time) {
    if (first_run) {
        first_run = false;
        imag_time = !_imag_time; 
    }
    if (_imag_time != imag_time) {
        imag_time = _imag_time;
        double time_single_it = delta_t / 2.;  //second approx trotter-suzuki: time/2
        if(imag_time) {
            h_a[0] = cosh(time_single_it / (2. * hamiltonian->mass * grid->delta_x * grid->delta_y));
            h_b[0] = sinh(time_single_it / (2. * hamiltonian->mass * grid->delta_x * grid->delta_y));
            if (!single_component) {
                h_a[1] = cosh(time_single_it / (2. * static_cast<Hamiltonian2Component*>(hamiltonian)->mass_b * grid->delta_x * grid->delta_y));
                h_b[1] = sinh(time_single_it / (2. * static_cast<Hamiltonian2Component*>(hamiltonian)->mass_b * grid->delta_x * grid->delta_y));
            }
        }
        else {
            h_a[0] = cos(time_single_it / (2. * hamiltonian->mass * grid->delta_x * grid->delta_y));
            h_b[0] = sin(time_single_it / (2. * hamiltonian->mass * grid->delta_x * grid->delta_y));
            if (!single_component) {
                h_a[1] = cos(time_single_it / (2. * static_cast<Hamiltonian2Component*>(hamiltonian)->mass_b * grid->delta_x * grid->delta_y));
                h_b[1] = sin(time_single_it / (2. * static_cast<Hamiltonian2Component*>(hamiltonian)->mass_b * grid->delta_x * grid->delta_y));
            }
        }
    }
    if (single_component) {
        trotter(grid, state, hamiltonian, h_a[0], h_b[0], 
                external_pot_real[0], external_pot_imag[0], 
                delta_t, iterations, kernel_type, norm2[0], imag_time);  
    } else {
        trotter(grid, state, state_b, static_cast<Hamiltonian2Component*>(hamiltonian), h_a, h_b, 
                external_pot_real, external_pot_imag, 
                delta_t, iterations, kernel_type, norm2, imag_time);  
    }
  
}
