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
 
#ifndef __CC2KERNEL_H
#define __CC2KERNEL_H

#include <cstdio>

#include <cuda.h>
#include <cuda_runtime.h>

#include "trotterkernel.h"

#define DISABLE_FMA

// NOTE: NEVER USE ODD NUMBERS FOR BLOCK DIMENSIONS
// thread block / shared memory block width
#define BLOCK_X 32
// shared memory block height
#define BLOCK_Y  (sizeof(float) == 8 ? 32 : 96)


#define STRIDE_Y 16
#define STEPS 1

#define CUT_CHECK_ERROR(errorMessage) {                                    \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    }

#define CUDA_SAFE_CALL(call) \
  if((call) != cudaSuccess) { \
    cudaError_t err = cudaGetLastError(); \
    fprintf (stderr, "CUDA error: %d\n", err ); \
    exit(-1);                                    \
  }

class CC2Kernel: public ITrotterKernel {
public:
    CC2Kernel(float *p_real, float *p_imag, float a, float b, int tile_width, int tile_height, int halo_x, int halo_y);
    ~CC2Kernel();
    void run_kernels();
    void run_kernel(int k);
    void wait_for_completion();
    void copy_results();
    void get_sample(size_t dest_stride, size_t x, size_t y, size_t width, size_t height, float * dest_real, float * dest_imag) const;

    bool runs_in_place() const {
        return false;
    }
    std::string get_name() const {
        return "CUDA CC 2.x 8-step kernel";
    }

    void initialize_MPI(MPI_Comm cartcomm, int _start_x, int _inner_end_x, int _start_y, int _inner_start_y, int _inner_end_y);
    void exchange_borders();

private:
    dim3 numBlocks;
    dim3 threadsPerBlock;

    float a;
    float b;
    float *p_real;
    float *p_imag;
    float *pdev_real[2];
    float *pdev_imag[2];
    int sense;
    int halo_x, halo_y, tile_width, tile_height;

    MPI_Comm cartcomm;
    int neighbors[4];
    int start_x, inner_end_x, start_y, inner_start_y,  inner_end_y;
    float *left_real_receive;
    float *left_real_send;
    float *right_real_receive;
    float *right_real_send;
    float *left_imag_receive;
    float *left_imag_send;
    float *right_imag_receive;
    float *right_imag_send;
    float *bottom_real_receive;
    float *bottom_real_send;
    float *top_real_receive;
    float *top_real_send;
    float *bottom_imag_receive;
    float *bottom_imag_send;
    float *top_imag_receive;
    float *top_imag_send;

};
#endif
