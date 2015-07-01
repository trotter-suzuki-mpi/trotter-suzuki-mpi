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

#ifndef __TROTTER_H
#define __TROTTER_H

/**
    API call to calculate the evolution through the Trotter-Suzuki decomposition.

    @param h_a               Kinetic term of the Hamiltonian (cosine part)
    @param h_b               Kinetic term of the Hamiltonian (sine part)
    @param external_pot_real External potential, real part
    @param external_pot_imag External potential, imaginary part
    @param p_real            Initial state, real part
    @param p_imag            Initial state, imaginary part
    @param matrix_width      The width of the initial state
    @param matrix_height     The height of the initial state
    @param iterations        Number of iterations to be calculated
    @param snapshots         Number of iterations between taking snapshots
                             (0 means no snapshots)
    @param kernel_type       The kernel type:
                              0: CPU block kernel
                              1: CPU SSE block kernel
                              2: GPU kernel
                              3: Hybrid kernel
    @param periods            Whether the grid is periodic in any of the directions
    @param output_folder      The folder to write the snapshots in
    @param verbose            Optional verbosity parameter
    @param imag_time          Optional parameter to calculate imaginary time evolution
    @param particle_tag       Optional parameter to tag a particle in the snapshots

*/

void trotter(double h_a, double h_b,
             double * external_pot_real, double * external_pot_imag,
             double * p_real, double * p_imag, 
             const int matrix_width, const int matrix_height, 
             const int iterations, const int kernel_type,
             int *periods, bool imag_time, int * time);
             
struct energy_momentum_statistics {
    double mean_E, mean_Px, mean_Py;
    double var_E, var_Px, var_Py;
    energy_momentum_statistics() : mean_E(0.), mean_Px(0.), mean_Py(0.),
        var_E(0.), var_Px(0.), var_Py(0.) {}
};

void expect_values(int dim, int iterations, int snapshots, double * hamilt_pot, double particle_mass,
                   const char *dirname, int *periods, int halo_x, int halo_y, energy_momentum_statistics *sample);
void stamp(double * p_real, double * p_imag, int matrix_width, int matrix_height, int halo_x, int halo_y, int start_x, int inner_start_x, int inner_end_x,
           int start_y, int inner_start_y, int inner_end_y, int * dims, int * coords, int * periods, 
           int tag_particle, int iterations, int count_snap, const char * output_folder
#ifdef HAVE_MPI
           , MPI_Comm cartcomm
#endif
           );
#endif
