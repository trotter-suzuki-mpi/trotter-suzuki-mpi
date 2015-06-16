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

#ifndef __CC2KERNEL_H
#define __CC2KERNEL_H

#include <cuda.h>
#include <cuda_runtime.h>

#if HAVE_CONFIG_H
#include <config.h>
#endif
#include "kernel.h"
#ifdef HAVE_MPI
#include <mpi.h>
#endif


#define DISABLE_FMA

// NOTE: NEVER USE ODD NUMBERS FOR BLOCK DIMENSIONS
// thread block / shared memory block width
#define BLOCK_X 32
// shared memory block height
#define BLOCK_Y  (sizeof(double) == 8 ? 32 : 96)

#define STRIDE_Y 16

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

void setDevice(int commRank
#ifdef HAVE_MPI
               , MPI_Comm cartcomm
#endif
               );
               
void cc2kernel_wrapper(size_t tile_width, size_t tile_height, size_t offset_x, size_t offset_y, size_t halo_x, size_t halo_y, dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t stream, double a, double b, const double * __restrict__ dev_external_pot_real, const double * __restrict__ dev_external_pot_imag, const double * __restrict__ pdev_real, const double * __restrict__ pdev_imag, double * __restrict__ pdev2_real, double * __restrict__ pdev2_imag, int inner, int horizontal, int vertical, bool imag_time);

class CC2Kernel: public ITrotterKernel {
public:
    CC2Kernel(double *p_real, double *p_imag, double *_external_pot_real, double *_external_pot_imag, double a, double b,
              int matrix_width, int matrix_height, int halo_x, int halo_y, int *_periods,
#ifdef HAVE_MPI
              MPI_Comm cartcomm,
#endif
              bool _imag_time);
    ~CC2Kernel();
    void run_kernel();
    void run_kernel_on_halo();
    void wait_for_completion(int iteration, int snapshots);
    void copy_results();
    void get_sample(size_t dest_stride, size_t x, size_t y, size_t width, size_t height, double * dest_real, double * dest_imag) const;

    bool runs_in_place() const {
        return false;
    }
    std::string get_name() const {
        return "CUDA";
    }

    void start_halo_exchange();
    void finish_halo_exchange();

private:
    dim3 numBlocks;
    dim3 threadsPerBlock;
    cudaStream_t stream1, stream2;

    bool imag_time;
    double *p_real;
    double *p_imag;
    double *external_pot_real;
    double *external_pot_imag;
    double *dev_external_pot_real;
    double *dev_external_pot_imag;
    double *pdev_real[2];
    double *pdev_imag[2];
    double a;
    double b;
    int sense;
    size_t halo_x, halo_y, tile_width, tile_height;

#ifdef HAVE_MPI
    MPI_Comm cartcomm;
#else
    int *periods;
#endif
    int neighbors[4];
    int start_x, inner_end_x, start_y, inner_start_y,  inner_end_y;
    double *left_real_receive;
    double *left_real_send;
    double *right_real_receive;
    double *right_real_send;
    double *left_imag_receive;
    double *left_imag_send;
    double *right_imag_receive;
    double *right_imag_send;
    double *bottom_real_receive;
    double *bottom_real_send;
    double *top_real_receive;
    double *top_real_send;
    double *bottom_imag_receive;
    double *bottom_imag_send;
    double *top_imag_receive;
    double *top_imag_send;

};
#endif
