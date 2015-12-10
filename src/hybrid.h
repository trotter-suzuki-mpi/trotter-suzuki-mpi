/**
 * Distributed Trotter-Suzuki solver
 * Copyright (C) 2015 Luca Calderaro, 2012-2015 Peter Wittek
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

#ifndef __HYBRID_H
#define __HYBRID_H

#if HAVE_CONFIG_H
#include "config.h"
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#include "kernel.h"
#include "cpublock.h"
#include "cc2kernel.h"

/**
 * \brief This class define the Hybrid kernel.
 *
 * This kernel provides real time and imaginary time evolution exploiting GPUs and CPUs.
 * It implements a solver for a single wave function, whose evolution is governed by linear Schrodinger equation. The Hamiltonian of the physical system includes:
 *  - static external potential
 */
 
class HybridKernel: public ITrotterKernel {
public:
    HybridKernel(double *p_real, double *p_imag, double *_external_pot_real, double *_external_pot_imag, double a, double b, double _coupling_const, double _delta_x, double _delta_y,
                 int matrix_width, int matrix_height, int halo_x, int halo_y, int * _periods, double _norm, bool _imag_time
#ifdef HAVE_MPI
                 , MPI_Comm cartcomm
#endif
                );
    ~HybridKernel();
    void run_kernel_on_halo();					///< Evolve blocks of wave function at the edge of the tile. This comprises the halos.
    void run_kernel();							///< Evolve the remaining blocks in the inner part of the tile.
    void wait_for_completion(int iteration);	///< Sincronize all the processes at the end of halos communication. Perform normalization for imaginary time evolution.
    void get_sample(size_t dest_stride, size_t x, size_t y, size_t width, size_t height, double * dest_real, double * dest_imag) const;		///< Copy the wave function from the buffers pointed by p_real, p_imag, pdev_real and pdev_imag, without halos, to dest_real and dest_imag.
    void normalization() {};
    void rabi_coupling(double var, double delta_t) {};
    
    bool runs_in_place() const {
        return false;
    }
    /// Get kernel name.
    std::string get_name() const {
        std::stringstream name;
        name << "Hybrid";
#ifdef _OPENMP
        name << "-OpenMP-" << omp_get_max_threads();
#endif
        return name.str();
    };

    void start_halo_exchange();				///< Start vertical halos exchange (between Hosts).
    void finish_halo_exchange();			///< Start horizontal halos exchange (between Hosts).

private:
    dim3 numBlocks;							///< Number of blocks exploited in the lattice.
    dim3 threadsPerBlock;					///< Number of lattice dots in a block.
    cudaStream_t stream;					///< Stream of sequential instructions performing evolution and communication on the Device lattice part.

    bool imag_time;							///< True: imaginary time evolution; False: real time evolution.
    double *p_real[2];						///< Array of two pointers that point to two buffers used to store the real part of the wave function at i-th time step and (i+1)-th time step (Host part). 
    double *p_imag[2];						///< Array of two pointers that point to two buffers used to store the imaginary part of the wave function at i-th time step and (i+1)-th time step (Host part).
    double *pdev_real[2];					///< Array of two pointers that point to two buffers used to store the real part of the wave function at i-th time step and (i+1)-th time step (Device part).
    double *pdev_imag[2];					///< Array of two pointers that point to two buffers used to store the imaginary part of the wave function at i-th time step and (i+1)-th time step (Device part).
    double *external_pot_real;				///< Points to the matrix representation (real entries) of the operator given by the exponential of external potential (Host part).
    double *external_pot_imag;				///< Points to the matrix representation (immaginary entries) of the operator given by the exponential of external potential (Host part).
    double *dev_external_pot_real;			///< Points to the matrix representation (real entries) of the operator given by the exponential of external potential (Device part).
    double *dev_external_pot_imag;			///< Points to the matrix representation (imaginary entries) of the operator given by the exponential of external potential (Device part).
    double a;								///< Diagonal value of the matrix representation of the operator given by the exponential of kinetic operator.
    double b;								///< Off diagonal value of the matrix representation of the operator given by the exponential of kinetic operator.
    double coupling_const;    				///< Coupling constant of the density self-interacting term (not implemented in this kernel).
    double delta_x;							///< Physical length between two neighbour along x axis dots of the lattice.
    double delta_y;							///< Physical length between two neighbour along y axis dots of the lattice.
    double norm;							///< Squared norm of the wave function.
    int sense;								///< Takes values 0 or 1 and tells which of the two buffers pointed by p_real and p_imag is used to calculate the next time step.
    size_t halo_x;							///< Thickness of the vertical halos (number of lattice's dots).
    size_t halo_y;							///< Thickness of the horizontal halos (number of lattice's dots).
    size_t tile_width;						///< Width of the tile (number of lattice's dots).
    size_t tile_height;						///< Height of the tile (number of lattice's dots).
    static const size_t block_width = BLOCK_WIDTH;		///< Width of the lattice block which is cached (number of lattice's dots).
    static const size_t block_height = BLOCK_HEIGHT;	///< Height of the lattice block which is cached (number of lattice's dots).
    size_t gpu_tile_width;					///< Tile width processes by Device.
    size_t gpu_tile_height;					///< Tile height processes by Device.
    size_t gpu_start_x;						///< X axis coordinate of the first dot of the processed tile in Device.
    size_t gpu_start_y;						///< Y axis coordinate of the first dot of the processed tile in Device.
    size_t n_bands_on_cpu;					///< Number of blocks in a column in Device's tile.

    int neighbors[4];						///< Array that stores the processes' rank neighbour of the current process.
    int start_x;							///< X axis coordinate of the first dot of the processed tile.
    int start_y;							///< Y axis coordinate of the first dot of the processed tile.
    int end_x;								///< X axis coordinate of the last dot of the processed tile.
    int end_y;								///< Y axis coordinate of the last dot of the processed tile.
    int inner_start_x;						///< X axis coordinate of the first dot of the processed tile, which is not in the halo.
    int inner_start_y;						///< Y axis coordinate of the first dot of the processed tile, which is not in the halo.
    int inner_end_x;						///< X axis coordinate of the last dot of the processed tile, which is not in the halo.
    int inner_end_y;						///< Y axis coordinate of the last dot of the processed tile, which is not in the halo.
    int *periods;							///< Two dimensional array which takes entries 0 or 1. 1: periodic boundary condition along the corresponding axis; 0: closed boundary condition along the corresponding axis.
#ifdef HAVE_MPI
    MPI_Comm cartcomm;						///< Ensemble of processes communicating the halos and evolving the tiles.
    MPI_Request req[8];						///< Variable to manage MPI communication (between Hosts).
    MPI_Status statuses[8];					///< Variable to manage MPI communication (between Hosts).
    MPI_Datatype horizontalBorder;			///< Datatype for the horizontal halos (Host halos).
    MPI_Datatype verticalBorder;			///< Datatype for the vertical halos (Host halos).
#endif
};

#endif
