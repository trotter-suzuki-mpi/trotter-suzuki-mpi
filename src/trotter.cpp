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

#include <cstring>
#include <string>
#include <sstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#if HAVE_CONFIG_H
#include "config.h"
#endif
#include "common.h"
#include "trotter.h"
#include "cpublock.h"
#ifdef SSE
#include "cpublocksse.h"
#endif
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#ifdef CUDA
#include "cc2kernel.h"
#include "hybrid.h"
#endif

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
            std::cerr << "Compiled without CUDA\n";
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
            std::cerr << "Compiled without CUDA\n";
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
