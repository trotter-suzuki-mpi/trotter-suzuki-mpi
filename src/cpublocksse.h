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

#ifndef __CPUBLOCKSSE_H
#define __CPUBLOCKSSE_H

#if HAVE_CONFIG_H
#include "config.h"
#endif
#include "kernel.h"
#ifdef HAVE_MPI
#include <mpi.h>
#endif

#define BLOCK_WIDTH 128u
#define BLOCK_HEIGHT 128u

#ifdef WIN32
template <int offset_y>
inline void update_shifty_sse(size_t stride, size_t width, size_t height, double a, double b, double * __restrict r1, double * __restrict i1, double * __restrict r2, double * __restrict i2);
template <int offset_x>
inline void update_shiftx_sse(size_t stride, size_t width, size_t height, double a, double b, double * __restrict r1, double * __restrict i1, double * __restrict r2, double * __restrict i2);
#else
template <int offset_y>
inline void update_shifty_sse(size_t stride, size_t width, size_t height, double a, double b, double * __restrict__ r1, double * __restrict__ i1, double * __restrict__ r2, double * __restrict__ i2);
template <int offset_x>
inline void update_shiftx_sse(size_t stride, size_t width, size_t height, double a, double b, double * __restrict__ r1, double * __restrict__ i1, double * __restrict__ r2, double * __restrict__ i2);
#endif
void process_sides_sse( double *var, size_t tile_width, size_t block_width, size_t block_height, size_t halo_x, size_t read_y, size_t read_height, size_t write_offset, size_t write_height, double a, double b,
                        const double *ext_pot_r00, const double *ext_pot_r01, const double *ext_pot_r10, const double *ext_pot_r11,
                        const double *ext_pot_i00, const double *ext_pot_i01, const double *ext_pot_i10, const double *ext_pot_i11,
                        double *block_ext_pot_r00, double *block_ext_pot_r01, double *block_ext_pot_r10, double *block_ext_pot_r11,
                        double *block_ext_pot_i00, double *block_ext_pot_i01, double *block_ext_pot_i10, double *block_ext_pot_i11,
                        const double * r00, const double * r01, const double * r10, const double * r11,
                        const double * i00, const double * i01, const double * i10, const double * i11,
                        double * next_r00, double * next_r01, double * next_r10, double * next_r11,
                        double * next_i00, double * next_i01, double * next_i10, double * next_i11,
                        double * block_r00, double * block_r01, double * block_r10, double * block_r11,
                        double * block_i00, double * block_i01, double * block_i10, double * block_i11);

void process_band_sse(double *var,   size_t tile_width, size_t block_width, size_t block_height, size_t halo_x, size_t read_y, size_t read_height, size_t write_offset, size_t write_height, double a, double b,
                      const double *ext_pot_r00, const double *ext_pot_r01, const double *ext_pot_r10, const double *ext_pot_r11,
                      const double *ext_pot_i00, const double *ext_pot_i01, const double *ext_pot_i10, const double *ext_pot_i11,
                      const double * r00, const double * r01, const double * r10, const double * r11,
                      const double * i00, const double * i01, const double * i10, const double * i11,
                      double * next_r00, double * next_r01, double * next_r10, double * next_r11,
                      double * next_i00, double * next_i01, double * next_i10, double * next_i11, int inner, int sides);

/**
 * \brief This class define the SSE kernel.
 *
 * This kernel provides real time and imaginary time evolution exploiting CPUs.
 * It implements a solver for a single wave function, whose evolution is governed by linear Schrodinger equation. The Hamiltonian of the physical system includes:
 *  - static external potential
 * 
 * NB: the single N x N lattice tile is divided in N/2 x N/2 small 2 x 2 squared blocks. Each top-left complex number is stored in buffers pointed by r00[0] and i00[0]; 
 * Each top-right complex number is stored in buffers pointed by r01[0] and i01[0]; Each bottom-left complex number is stored in buffers pointed by r10[0] and i10[0]; 
 * Each bottom-right complex number is stored in buffers pointed by r11[0] and i11[0]. The same logic is applied to the storage of the matrix representation of the operator 
 * given by the exponential of external potential.
 */

class CPUBlockSSEKernel: public ITrotterKernel {
public:
    CPUBlockSSEKernel(double *p_real, double *p_imag, double *external_potential_real, double *external_potential_imag, double a, double b, double _delta_x, double _delta_y, int matrix_width, int matrix_height, int halo_x, int halo_y, int *_periods, double _norm, bool _imag_time
#ifdef HAVE_MPI
                      , MPI_Comm cartcomm
#endif
                     );
    ~CPUBlockSSEKernel();
    void run_kernel_on_halo();				    ///< Evolve blocks of wave function at the edge of the tile. This comprises the halos.
    void run_kernel();							///< Evolve the remaining blocks in the inner part of the tile.
    void wait_for_completion(int iteration);	///< Sincronize all the processes at the end of halos communication. Perform normalization for imaginary time evolution.
    void get_sample(size_t dest_stride, size_t x, size_t y, size_t width, size_t height, double * dest_real, double * dest_imag) const;  ///< Copy the wave function from the eight buffers pointed by r00, r01, r10, r11, i00, i01, i10 and i11, without halos, to dest_real and dest_imag.
	void get_sample2(size_t dest_stride, size_t x, size_t y, size_t width, size_t height, double ** dest_real, double ** dest_imag) const;
	void normalization() {};
	void rabi_coupling(double var, double delta_t) {};
	
    bool runs_in_place() const {
        return false;
    }
    /// Get kernel name.
    std::string get_name() const {
        return "SSE";
    };

    void start_halo_exchange();					///< Start vertical halos exchange.
    void finish_halo_exchange();				///< Start horizontal halos exchange.


private:
    double *p_real;				///< Point to  the real part of the wave function.
    double *p_imag;				///< Point to  the imaginary part of the wave function.
    double *r00[2];				///< Array of two pointers that point to two buffers used to store the real part of the wave function at i-th time step and (i+1)-th time step (top-left part of the small blocks).
    double *r01[2];				///< Array of two pointers that point to two buffers used to store the real part of the wave function at i-th time step and (i+1)-th time step (top-right part of the small blocks).
    double *r10[2];				///< Array of two pointers that point to two buffers used to store the real part of the wave function at i-th time step and (i+1)-th time step (bottom-left part of the small blocks).
    double *r11[2];				///< Array of two pointers that point to two buffers used to store the real part of the wave function at i-th time step and (i+1)-th time step (bottom-right part of the small blocks).
    double *i00[2];				///< Array of two pointers that point to two buffers used to store the imaginary part of the wave function at i-th time step and (i+1)-th time step (top-left part of the small blocks).
    double *i01[2];				///< Array of two pointers that point to two buffers used to store the imaginary part of the wave function at i-th time step and (i+1)-th time step (top-right part of the small blocks).
    double *i10[2];				///< Array of two pointers that point to two buffers used to store the imaginary part of the wave function at i-th time step and (i+1)-th time step (bottom-left part of the small blocks).
    double *i11[2];				///< Array of two pointers that point to two buffers used to store the imaginary part of the wave function at i-th time step and (i+1)-th time step (bottom-right part of the small blocks).
    double *ext_pot_r00;		///< Points to the matrix representation (real entries) of the operator given by the exponential of external potential (top-left part of the small blocks).
    double *ext_pot_r01;		///< Points to the matrix representation (real entries) of the operator given by the exponential of external potential (top-right part of the small blocks).
    double *ext_pot_r10;		///< Points to the matrix representation (real entries) of the operator given by the exponential of external potential (bottom-left part of the small blocks).
    double *ext_pot_r11;		///< Points to the matrix representation (real entries) of the operator given by the exponential of external potential (bottom-right part of the small blocks).
    double *ext_pot_i00;		///< Points to the matrix representation (imaginary entries) of the operator given by the exponential of external potential (top-left part of the small blocks).
    double *ext_pot_i01;		///< Points to the matrix representation (imaginary entries) of the operator given by the exponential of external potential (top-right part of the small blocks).
    double *ext_pot_i10;		///< Points to the matrix representation (imaginary entries) of the operator given by the exponential of external potential (bottom-left part of the small blocks).
    double *ext_pot_i11;		///< Points to the matrix representation (imaginary entries) of the operator given by the exponential of external potential (bottom-right part of the small blocks).
    double a;						///< Diagonal value of the matrix representation of the operator given by the exponential of kinetic operator.
    double b;						///< Off diagonal value of the matrix representation of the operator given by the exponential of kinetic operator.
    double delta_x;					///< Physical length between two neighbour along x axis dots of the lattice.
    double delta_y;					///< Physical length between two neighbour along y axis dots of the lattice.
    double norm;					///< Squared norm of the wave function.
    int sense;						///< Takes values 0 or 1 and tells which of the two buffers pointed by r00, r01, r10, r11, i00, i01, i10 and i11 is used to calculate the next time step.
    size_t halo_x;					///< Thickness of the vertical halos (number of lattice's dots).
    size_t halo_y;					///< Thickness of the horizontal halos (number of lattice's dots).
    size_t tile_width;				///< Width of the tile (number of lattice's dots).
    size_t tile_height;				///< Height of the tile (number of lattice's dots).
    bool imag_time;					///< True: imaginary time evolution; False: real time evolution.
    // NOTE: block rows must be 16 byte aligned
    //       block height must be even
    static const size_t block_width = BLOCK_WIDTH;		///< Width of the lattice block which is cached (number of lattice's dots).
    static const size_t block_height = BLOCK_HEIGHT;	///< Height of the lattice block which is cached (number of lattice's dots).

    int start_x;					///< X axis coordinate of the first dot of the processed tile.
    int start_y;					///< Y axis coordinate of the first dot of the processed tile.
    int end_x;						///< X axis coordinate of the last dot of the processed tile.
    int end_y;						///< Y axis coordinate of the last dot of the processed tile.
    int inner_start_x;				///< X axis coordinate of the first dot of the processed tile, which is not in the halo.
    int inner_start_y;				///< Y axis coordinate of the first dot of the processed tile, which is not in the halo.
    int inner_end_x;				///< X axis coordinate of the last dot of the processed tile, which is not in the halo.
    int inner_end_y;				///< Y axis coordinate of the last dot of the processed tile, which is not in the halo.
    int *periods;					///< Two dimensional array which takes entries 0 or 1. 1: periodic boundary condition along the corresponding axis; 0: closed boundary condition along the corresponding axis.
#ifdef HAVE_MPI
    MPI_Comm cartcomm;				///< Ensemble of processes communicating the halos and evolving the tiles.
    int neighbors[4];				///< Array that stores the processes' rank neighbour of the current process.
    MPI_Request req[32];			///< Variable to manage MPI communication.
    MPI_Status statuses[32];		///< Variable to manage MPI communication.
    MPI_Datatype horizontalBorder;	///< Datatype for the horizontal halos.
    MPI_Datatype verticalBorder;	///< Datatype for the vertical halos.
#endif
};

#endif
