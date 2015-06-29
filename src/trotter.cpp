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
#include <sys/time.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#if HAVE_CONFIG_H
#include <config.h>
#endif
#include "common.h"
#include "trotter.h"
#include "cpublock.h"
#include "cpublocksse.h"
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#ifdef CUDA
#include "cc2kernel.h"
#include "hybrid.h"
#endif

void trotter(double h_a, double h_b,
             double * external_pot_real, double * external_pot_imag,
             double * p_real, double * p_imag, 
             const int matrix_width, const int matrix_height, 
             const int iterations, const int kernel_type,
             int *periods, bool imag_time, long * time) {
    
    int start_x, end_x, inner_start_x, inner_end_x,
        start_y, end_y, inner_start_y, inner_end_y;

    int coords[2], dims[2] = {0, 0};
    int rank;
    int nProcs;

#ifdef HAVE_MPI
    MPI_Comm cartcomm;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    MPI_Dims_create(nProcs, 2, dims);  //partition all the processes (the size of MPI_COMM_WORLD's group) into an 2-dimensional topology
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cartcomm);
    MPI_Comm_rank(cartcomm, &rank);
    MPI_Cart_coords(cartcomm, rank, 2, coords); //Determines process coords in cartesian topology given rank in group
#else
    nProcs = 1;
    rank = 0;
    dims[0] = dims[1] = 1;
    coords[0] = coords[1] = 0;
#endif

    int halo_x = (kernel_type == 2 ? 3 : 4);
    int halo_y = 4;
    calculate_borders(coords[1], dims[1], &start_x, &end_x, &inner_start_x, &inner_end_x, matrix_width - 2 * periods[1]*halo_x, halo_x, periods[1]);
    calculate_borders(coords[0], dims[0], &start_y, &end_y, &inner_start_y, &inner_end_y, matrix_height - 2 * periods[0]*halo_y, halo_y, periods[0]);
    int width = end_x - start_x;
    int height = end_y - start_y;
    
    // Initialize kernel
    ITrotterKernel * kernel;
    switch (kernel_type) {
    case 0:
        kernel = new CPUBlock(p_real, p_imag, external_pot_real, external_pot_imag, h_a, h_b, matrix_width, matrix_height, halo_x, halo_y, periods, imag_time
#ifdef HAVE_MPI
                              , cartcomm
#endif
                              );
        break;

    case 1:
        kernel = new CPUBlockSSEKernel(p_real, p_imag, external_pot_real, external_pot_imag, h_a, h_b, matrix_width, matrix_height, halo_x, halo_y, periods, imag_time
#ifdef HAVE_MPI
                              , cartcomm
#endif
                              );
        break;

    case 2:
#ifdef CUDA
        kernel = new CC2Kernel(p_real, p_imag, external_pot_real, external_pot_imag, h_a, h_b, matrix_width, matrix_height, halo_x, halo_y, periods, imag_time
#ifdef HAVE_MPI
                              , cartcomm
#endif
                              );
#else
        if (coords[0] == 0 && coords[1] == 0) {
            std::cerr << "Compiled without CUDA\n";
        }
#ifdef HAVE_MPI
        MPI_Abort(MPI_COMM_WORLD, 2);
#else
        abort ();
#endif
#endif
        break;

    case 3:
#ifdef CUDA
        kernel = new HybridKernel(p_real, p_imag, external_pot_real, external_pot_imag, h_a, h_b, matrix_width, matrix_height, halo_x, halo_y, periods, imag_time
#ifdef HAVE_MPI
                              , cartcomm
#endif
                              );
#else
        if (coords[0] == 0 && coords[1] == 0) {
            std::cerr << "Compiled without CUDA\n";
        }
#ifdef HAVE_MPI
        MPI_Abort(MPI_COMM_WORLD, 2);
#else
        abort();
#endif
#endif
        break;

    default:
        kernel = new CPUBlock(p_real, p_imag, external_pot_real, external_pot_imag, h_a, h_b, matrix_width, matrix_height, halo_x, halo_y, periods, imag_time
#ifdef HAVE_MPI
                              , cartcomm
#endif
                              );
        break;
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);

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
        kernel->wait_for_completion(i);
    }
    
    kernel->get_sample(width, 0, 0, width, height, p_real, p_imag);
    
    gettimeofday(&end, NULL);
    *time = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
    /*if (coords[0] == 0 && coords[1] == 0 && verbose == true) {
        long time = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
        std::cout << "TROTTER " << matrix_width - periods[1] * 2 * halo_x << "x" << matrix_height - periods[0] * 2 * halo_y << " " << kernel->get_name() << " " << nProcs << " " << time << std::endl;
    }*/
/*
#ifdef HAVE_MPI    
    MPI_Type_free(&localarray);
    MPI_Type_free(&num_as_string);
    MPI_Type_free(&complex_localarray);
    MPI_Type_free(&complex_num_as_string);
#endif
*/ 
    //delete kernel;
}
