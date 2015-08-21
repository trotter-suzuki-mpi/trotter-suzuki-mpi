/**
 * Distributed Trotter-Suzuki solver
 * Copyright (C) 2012 Peter Wittek, 2010-2012 Carlos Bederián
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
 
#include <sstream>
#include <string>

#include <sys/time.h>
#include <mpi.h>
#include <unistd.h>

#include "common.h"
#include "cpublock.h"
//#include "cpublocksse.h"
#ifdef CUDA
#include "cc2kernel.h"
#endif

#define DIM 640
#define ITERATIONS 1000
#define KERNEL_TYPE 0
#define SNAPSHOTS 0


void trotter(const int matrix_width, const int matrix_height, const int iterations, const int snapshots, const int kernel_type) {

    float * p_real, * p_imag;
    std::stringstream filename;
    int start_x, end_x, inner_start_x, inner_end_x,
        start_y, end_y, inner_start_y, inner_end_y;
    MPI_Comm cartcomm;
    int coords[2], dims[2]= {0,0};
    int periods[2]= {0, 0};
    int rank;
    int nProcs;

    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    MPI_Dims_create(nProcs, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cartcomm);
    MPI_Comm_rank(cartcomm, &rank);
    MPI_Cart_coords(cartcomm, rank, 2, coords);

    int halo_x=(kernel_type==2 ? 3 : 4);
    int halo_y=4;
    calculate_borders(coords[1], dims[1], &start_x, &end_x, &inner_start_x, &inner_end_x, matrix_width, halo_x);
    calculate_borders(coords[0], dims[0], &start_y, &end_y, &inner_start_y, &inner_end_y, matrix_height, halo_y);
    int width=end_x-start_x;
    int height=end_y-start_y;
#ifdef DEBUG
    std::cout << "Coord_x: "<< coords[1] << " start_x: " << start_x << \
              " end_x: "<< end_x << " inner_start_x " << inner_start_x << " inner_end_x " << inner_end_x << "\n";
    std::cout << "Coord_y: "<< coords[0] << " start_y: " << start_y << \
              " end_y: "<< end_y << " inner_start_y " << inner_start_y << " inner_end_y " << inner_end_y<< "\n";
#endif

    // Allocate and initialize matrices
    p_real = new float[width * height];
    p_imag = new float[width * height];
    init_p(p_real, p_imag, start_x, end_x, start_y, end_y);

    // Initialize kernel
    ITrotterKernel * kernel;
    switch (kernel_type) {
    case 0:
        kernel=new CPUBlock(p_real, p_imag, h_a, h_b, width, height, halo_x, halo_y);
        break;

    case 1:
  //      kernel=new CPUBlockSSEKernel(p_real, p_imag, h_a, h_b, width, height, halo_x, halo_y);
        break;

    case 2:
#ifdef CUDA
        kernel=new CC2Kernel(p_real, p_imag, h_a, h_b, width, height, halo_x, halo_y);
#else
        if (coords[0]==0 && coords[1]==0) {
            std::cerr << "Compiled without CUDA\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
#endif
        break;

    default:
        kernel=new CPUBlock(p_real, p_imag, h_a, h_b, width, height, halo_x, halo_y);
    }

    kernel->initialize_MPI(cartcomm, start_x, inner_end_x, start_y, inner_start_y, inner_end_y);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Main loop
    for (int i = 0; i < iterations; i++) {
        if ( (snapshots>0) && (i % snapshots == 0) ) {
            kernel->get_sample(width, 0, 0, width, height, p_real, p_imag);
            filename.str("");
            filename << i << "-iter-" << coords[1] << "-" << coords[0] << "-real.dat";
            print_matrix(filename.str(), p_real+((inner_start_y-start_y)*width+inner_start_x-start_x),
                         width, inner_end_x-inner_start_x, inner_end_y-inner_start_y);
        }
        kernel->run_kernel_on_halo();
        if (i!=iterations-1) {
          kernel->start_halo_exchange();
        }
        kernel->run_kernel();
        if (i!=iterations-1) {
            kernel->finish_halo_exchange();
        }
        kernel->wait_for_completion();
    }

    gettimeofday(&end, NULL);
    if (coords[0]==0 && coords[1]==0) {
        long time = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
        std::cout << matrix_width << "x" << matrix_height << " " << kernel->get_name() << " " << matrix_width * matrix_height << " "<< time << std::endl;
    }
    delete[] p_real;
    delete[] p_imag;
    delete kernel;
}

int main(int argc, char** argv) {
    int dim=640, iterations=1000, snapshots=100, kernel_type=0;
    int rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    trotter(dim, dim, iterations, snapshots, kernel_type);

    MPI_Finalize();
    return 0;
}
