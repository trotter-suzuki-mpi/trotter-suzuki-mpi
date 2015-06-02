/**
 * Distributed Trotter-Suzuki solver
 * Copyright (C) 2012 Peter Wittek, 2010-2012 Carlos Bederián, 2015 Luca Calderaro
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

#include <fstream>

#include <sstream>
#include <string>

#include <sys/time.h>
#include <mpi.h>
#include <unistd.h>

#include "trotter.h"
#include "common.h"
#include "cpublock.h"
#include "cpublocksse.h"
#ifdef CUDA
#include "cc2kernel.h"
#include "hybrid.h"
#endif

void trotter(double h_a, double h_b,
             double * external_pot_real, double * external_pot_imag,
             double * p_real, double * p_imag, const int matrix_width,
             const int matrix_height, const int iterations, const int snapshots, const int kernel_type,
             int *periods, int argc, char** argv, const char *dirname, bool show_time_sim, bool imag_time, int particle_tag) {

    double *_p_real, *_p_imag;
    double *_external_pot_real, *_external_pot_imag;
    std::stringstream filename;
    int start_x, end_x, inner_start_x, inner_end_x,
        start_y, end_y, inner_start_y, inner_end_y;

    int coords[2], dims[2] = {0, 0};
    int rank;
    int nProcs;

    MPI_Comm cartcomm;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    MPI_Dims_create(nProcs, 2, dims);  //partition all the processes (the size of MPI_COMM_WORLD's group) into an 2-dimensional topology
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cartcomm);
    MPI_Comm_rank(cartcomm, &rank);
    MPI_Cart_coords(cartcomm, rank, 2, coords); //Determines process coords in cartesian topology given rank in group

    int halo_x = (kernel_type == 2 ? 3 : 4);
    int halo_y = 4;
    calculate_borders(coords[1], dims[1], &start_x, &end_x, &inner_start_x, &inner_end_x, matrix_width - 2 * periods[1]*halo_x, halo_x, periods[1]);
    calculate_borders(coords[0], dims[0], &start_y, &end_y, &inner_start_y, &inner_end_y, matrix_height - 2 * periods[0]*halo_y, halo_y, periods[0]);
    int width = end_x - start_x;
    int height = end_y - start_y;

#ifdef DEBUG
    std::cout << "Coord_x: " << coords[1] << " start_x: " << start_x << \
              " end_x: " << end_x << " inner_start_x " << inner_start_x << " inner_end_x " << inner_end_x << "\n";
    std::cout << "Coord_y: " << coords[0] << " start_y: " << start_y << \
              " end_y: " << end_y << " inner_start_y " << inner_start_y << " inner_end_y " << inner_end_y << "\n";
#endif

    // Allocate matrices
    _p_real = new double[width * height];
    _p_imag = new double[width * height];
    _external_pot_real = new double[width * height];
    _external_pot_imag = new double[width * height];
    double * matr_real, * matr_imag;
    double * matrix_snap_real;
    double * matrix_snap_imag;
    if(rank == 0) {
        matr_real = new double [nProcs * (inner_end_x - inner_start_x) * (inner_end_y - inner_start_y)];
        matr_imag = new double [nProcs * (inner_end_x - inner_start_x) * (inner_end_y - inner_start_y)];
        matrix_snap_real = new double [(matrix_width - 2 * periods[1]*halo_x) * (matrix_height - 2 * periods[0]*halo_y)];
        matrix_snap_imag = new double [(matrix_width - 2 * periods[1]*halo_x) * (matrix_height - 2 * periods[0]*halo_y)];
    }

    memcpy2D(_p_real, width * sizeof(double), &p_real[(start_y + periods[0]*halo_y) * matrix_width + start_x + periods[1]*halo_x], matrix_width * sizeof(double), width * sizeof(double), height);
    memcpy2D(_p_imag, width * sizeof(double), &p_imag[(start_y + periods[0]*halo_y) * matrix_width + start_x + periods[1]*halo_x], matrix_width * sizeof(double), width * sizeof(double), height);
    memcpy2D(_external_pot_real, width * sizeof(double), &external_pot_real[(start_y + periods[0]*halo_y) * matrix_width + start_x + periods[1]*halo_x], matrix_width * sizeof(double), width * sizeof(double), height);
    memcpy2D(_external_pot_imag, width * sizeof(double), &external_pot_imag[(start_y + periods[0]*halo_y) * matrix_width + start_x + periods[1]*halo_x], matrix_width * sizeof(double), width * sizeof(double), height);

    // Initialize kernel
    ITrotterKernel * kernel;
    switch (kernel_type) {
    case 0:
        kernel = new CPUBlock(_p_real, _p_imag, _external_pot_real, _external_pot_imag, h_a, h_b, matrix_width, matrix_height, halo_x, halo_y, periods, cartcomm, imag_time);
        break;

    case 1:
        kernel = new CPUBlockSSEKernel(_p_real, _p_imag, h_a, h_b, matrix_width, matrix_height, halo_x, halo_y, periods, cartcomm);
        break;

    case 2:
#ifdef CUDA
        kernel = new CC2Kernel(_p_real, _p_imag, _external_pot_real, _external_pot_imag, h_a, h_b, matrix_width, matrix_height, halo_x, halo_y, periods, cartcomm, imag_time);
#else
        if (coords[0] == 0 && coords[1] == 0) {
            std::cerr << "Compiled without CUDA\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
#endif
        break;

    case 3:
#ifdef CUDA
        kernel = new HybridKernel(_p_real, _p_imag, _external_pot_real, _external_pot_imag, h_a, h_b, matrix_width, matrix_height, halo_x, halo_y, cartcomm);
#else
        if (coords[0] == 0 && coords[1] == 0) {
            std::cerr << "Compiled without CUDA\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
#endif
        break;

    default:
        kernel = new CPUBlock(_p_real, _p_imag, _external_pot_real, _external_pot_imag, h_a, h_b, matrix_width, matrix_height, halo_x, halo_y, periods, cartcomm, imag_time);
        break;
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Main loop
    for (int i = 0; i < iterations; i++) {
        if ( (snapshots > 0) && (i % snapshots == 0) ) {
            kernel->get_sample(inner_end_x - inner_start_x, inner_start_x - start_x, inner_start_y - start_y,
                               inner_end_x - inner_start_x, inner_end_y - inner_start_y, _p_real, _p_imag);
            MPI_Gather(_p_real, (inner_end_x - inner_start_x) * (inner_end_y - inner_start_y), MPI_DOUBLE,
                       matr_real, (inner_end_x - inner_start_x) * (inner_end_y - inner_start_y), MPI_DOUBLE, 0, cartcomm);
            MPI_Gather(_p_imag, (inner_end_x - inner_start_x) * (inner_end_y - inner_start_y), MPI_DOUBLE,
                       matr_imag, (inner_end_x - inner_start_x) * (inner_end_y - inner_start_y), MPI_DOUBLE, 0, cartcomm);


            if(rank == 0) {
                int _start_x, _end_x, _inner_start_x, _inner_end_x,
                    _start_y, _end_y, _inner_start_y, _inner_end_y;
                int _coords[2];
                int set = 0;

                for(int _rank = 0; _rank < nProcs; _rank++) {
                    MPI_Cart_coords(cartcomm, _rank, 2, _coords);
                    calculate_borders(_coords[0], dims[0], &_start_y, &_end_y, &_inner_start_y, &_inner_end_y, matrix_height - 2 * periods[0]*halo_y, halo_y, periods[0]);
                    calculate_borders(_coords[1], dims[1], &_start_x, &_end_x, &_inner_start_x, &_inner_end_x, matrix_width - 2 * periods[1]*halo_x, halo_x, periods[1]);
                    for(int y = 0 ; y < _inner_end_y - _inner_start_y; y++) {
                        for(int x = 0 ; x < _inner_end_x - _inner_start_x; x++) {
                            matrix_snap_real[x + _inner_start_x + (y + _inner_start_y) * (matrix_width - 2 * periods[1]*halo_x)] = matr_real[set + x + y * (_inner_end_x - _inner_start_x)];
                            matrix_snap_imag[x + _inner_start_x + (y + _inner_start_y) * (matrix_width - 2 * periods[1]*halo_x)] = matr_imag[set + x + y * (_inner_end_x - _inner_start_x)];
                        }
                    }
                    set += (_inner_end_y - _inner_start_y) * (_inner_end_x - _inner_start_x);
                }

                filename.str("");
                filename << dirname << "/" << particle_tag << "-" << i << "-iter-real.dat";
                print_matrix(filename.str(), matrix_snap_real, matrix_width - 2 * periods[1]*halo_x,
                             matrix_width - 2 * periods[1]*halo_x, matrix_height - 2 * periods[0]*halo_y);

                filename.str("");
                filename << dirname << "/" << particle_tag << "-" << i << "-iter-comp.dat";
                print_complex_matrix(filename.str(), matrix_snap_real, matrix_snap_imag, matrix_width - 2 * periods[1]*halo_x,
                                     matrix_width - 2 * periods[1]*halo_x, matrix_height - 2 * periods[0]*halo_y);
            }
        }
        kernel->run_kernel_on_halo();
        if (i != iterations - 1) {
            kernel->start_halo_exchange();
        }
        kernel->run_kernel();
        if (i != iterations - 1) {
            kernel->finish_halo_exchange();
        }
        kernel->wait_for_completion(i, snapshots);
    }

    gettimeofday(&end, NULL);
    if (coords[0] == 0 && coords[1] == 0 && show_time_sim == true) {
        long time = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
        std::cout << "TROTTER " << matrix_width - periods[1] * 2 * halo_x << "x" << matrix_height - periods[0] * 2 * halo_y << " " << kernel->get_name() << " " << nProcs << " " << time << std::endl;
    }
    delete[] _p_real;
    delete[] _p_imag;
    delete[] _external_pot_real;
    delete[] _external_pot_imag;
    delete kernel;
}
