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
#include "config.h"
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


/**
 * \brief This class define the GPU kernel.
 *
 * This kernel provides real time and imaginary time evolution exploiting GPUs.
 * It implements a solver for a single wave function, whose evolution is governed by linear Schrodinger equation. The Hamiltonian of the physical system includes:
 *  - static external potential
 */
 
class CC2Kernel: public ITrotterKernel {
public:
    CC2Kernel(double *p_real, double *p_imag, double *_external_pot_real, double *_external_pot_imag, double a, double b, double _delta_x, double _delta_y,
              int matrix_width, int matrix_height, int halo_x, int halo_y, int *_periods, double _norm, bool _imag_time
#ifdef HAVE_MPI
              , MPI_Comm cartcomm
#endif
             );
    ~CC2Kernel();
    void run_kernel_on_halo();				    ///< Evolve blocks of wave function at the edge of the tile. This comprises the halos.
    void run_kernel();							///< Evolve the remaining blocks in the inner part of the tile.
    void wait_for_completion(int iteration);	///< Sincronize all the processes at the end of halos communication. Perform normalization for imaginary time evolution.
    void copy_results();						///< Copy wave function from buffer pointed by pdev_real and pdev_imag to buffers pointed by p_real and p_imag.
    void get_sample(size_t dest_stride, size_t x, size_t y, size_t width, size_t height, double * dest_real, double * dest_imag) const;  ///< Copy the wave function from the two buffers pointed by pdev_real and pdev_imag, without halos, to dest_real and dest_imag.

    bool runs_in_place() const {
        return false;
    }
    /// Get kernel name.
    std::string get_name() const {
        return "CUDA";
    }

    void start_halo_exchange();		///< Empty function.
    void finish_halo_exchange();	///< Exchange halos.

private:
    dim3 numBlocks;						///< Number of blocks exploited in the lattice.
    dim3 threadsPerBlock;				///< Number of lattice dots in a block.
    cudaStream_t stream1;				///< Stream of sequential instructions performing evolution and communication on the halos blocks.
	cudaStream_t stream2;				///< Stream of sequential instructions performing evolution on the inner blocks.
	
    bool imag_time;						///< True: imaginary time evolution; False: real time evolution.
    double *p_real;						///< Point to  the real part of the wave function (stored in Host).
    double *p_imag;						///< Point to  the imaginary part of the wave function (stored in Host).
    double *external_pot_real;			///< Points to the matrix representation (real entries) of the operator given by the exponential of external potential (stored in Host).
    double *external_pot_imag;			///< Points to the matrix representation (imaginary entries) of the operator given by the exponential of external potential (stored in Host).
    double *dev_external_pot_real;		///< Points to the matrix representation (real entries) of the operator given by the exponential of external potential (stored in Device).
    double *dev_external_pot_imag;		///< Points to the matrix representation (imaginary entries) of the operator given by the exponential of external potential (stored in Device).
    double *pdev_real[2];				///< Array of two pointers that point to two buffers used to store the real part of the wave function at i-th time step and (i+1)-th time step (stored in Device).
    double *pdev_imag[2];				///< Array of two pointers that point to two buffers used to store the imaginary part of the wave function at i-th time step and (i+1)-th time step (stored in Device).
    double a;							///< Diagonal value of the matrix representation of the operator given by the exponential of kinetic operator.
    double b;							///< Off diagonal value of the matrix representation of the operator given by the exponential of kinetic operator.
    double delta_x;						///< Physical length between two neighbour along x axis dots of the lattice.
    double delta_y;						///< Physical length between two neighbour along y axis dots of the lattice.
    double norm;						///< Squared norm of the wave function.
    int sense;							///< Takes values 0 or 1 and tells which of the two buffers pointed by p_real and p_imag is used to calculate the next time step.
    size_t halo_x;						///< Thickness of the vertical halos (number of lattice's dots).
    size_t halo_y;						///< Thickness of the horizontal halos (number of lattice's dots).
    size_t tile_width;					///< Width of the tile (number of lattice's dots).
    size_t tile_height;					///< Height of the tile (number of lattice's dots).
	int *periods;						///< Two dimensional array which takes entries 0 or 1. 1: periodic boundary condition along the corresponding axis; 0: closed boundary condition along the corresponding axis.
#ifdef HAVE_MPI
    MPI_Comm cartcomm;
#endif
    int neighbors[4];					///< Array that stores the processes' rank neighbour of the current process.
    int start_x;						///< X axis coordinate of the first dot of the processed tile.
    int start_y;						///< Y axis coordinate of the first dot of the processed tile.
    int end_x;							///< X axis coordinate of the last dot of the processed tile.
    int end_y;							///< Y axis coordinate of the last dot of the processed tile.
    int inner_start_x;					///< X axis coordinate of the first dot of the processed tile, which is not in the halo.
    int inner_start_y;					///< Y axis coordinate of the first dot of the processed tile, which is not in the halo.
    int inner_end_x;					///< X axis coordinate of the last dot of the processed tile, which is not in the halo.
    int inner_end_y;					///< Y axis coordinate of the last dot of the processed tile, which is not in the halo.
    double *left_real_receive;			///< Point to the buffer used to receive the real left vertical halo from the corresponding neighbour.
    double *left_real_send;				///< Point to the buffer used to send the real left vertical halo to the corresponding neighbour.
    double *right_real_receive;			///< Point to the buffer used to receive the real right vertical halo from the corresponding neighbour.
    double *right_real_send;			///< Point to the buffer used to send the real right vertical halo to the corresponding neighbour.
    double *left_imag_receive;			///< Point to the buffer used to receive the imaginary left vertical halo from the corresponding neighbour.
    double *left_imag_send;				///< Point to the buffer used to send the imaginary left vertical halo to the corresponding neighbour.
    double *right_imag_receive;			///< Point to the buffer used to receive the imaginary right vertical halo from the corresponding neighbour.
    double *right_imag_send;			///< Point to the buffer used to send the imaginary right vertical halo to the corresponding neighbour.
    double *bottom_real_receive;		///< Point to the buffer used to receive the real bottom horizontal halo from the corresponding neighbour.
    double *bottom_real_send;			///< Point to the buffer used to send the real bottom horizontal halo to the corresponding neighbour.
    double *top_real_receive;			///< Point to the buffer used to receive the real top horizontal halo from the corresponding neighbour.
    double *top_real_send;				///< Point to the buffer used to send the real top horizontal halo to the corresponding neighbour.
    double *bottom_imag_receive;		///< Point to the buffer used to receive the imaginary bottom horizontal halo from the corresponding neighbour.
    double *bottom_imag_send;			///< Point to the buffer used to send the imaginary bottom horizontal halo to the corresponding neighbour.
    double *top_imag_receive;			///< Point to the buffer used to receive the real top horizontal halo from the corresponding neighbour.
    double *top_imag_send;				///< Point to the buffer used to send the real bottom horizontal halo to the corresponding neighbour.

};
#endif
