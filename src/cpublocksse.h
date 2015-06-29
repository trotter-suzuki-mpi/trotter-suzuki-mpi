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
#include <config.h>
#endif
#include "kernel.h"
#ifdef HAVE_MPI
#include <mpi.h>
#endif

#define BLOCK_WIDTH 128u
#define BLOCK_HEIGHT 128u

template <int offset_y>
inline void update_shifty_sse(size_t stride, size_t width, size_t height, double a, double b, double * __restrict__ r1, double * __restrict__ i1, double * __restrict__ r2, double * __restrict__ i2);
template <int offset_x>
inline void update_shiftx_sse(size_t stride, size_t width, size_t height, double a, double b, double * __restrict__ r1, double * __restrict__ i1, double * __restrict__ r2, double * __restrict__ i2);

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


class CPUBlockSSEKernel: public ITrotterKernel {
public:
    CPUBlockSSEKernel(double *p_real, double *p_imag, double *external_potential_real, double *external_potential_imag, double a, double b, int matrix_width, int matrix_height, int halo_x, int halo_y, int *_periods, bool _imag_time
#ifdef HAVE_MPI
                      , MPI_Comm cartcomm
#endif
                      );
    ~CPUBlockSSEKernel();
    void run_kernel();
    void run_kernel_on_halo();
    void wait_for_completion(int iteration);
    void get_sample(size_t dest_stride, size_t x, size_t y, size_t width, size_t height, double * dest_real, double * dest_imag) const;

    bool runs_in_place() const {
        return false;
    }
    std::string get_name() const {
        return "SSE";
    };

    void start_halo_exchange();
    void finish_halo_exchange();


private:
    double *p_real;
    double *p_imag;
    double *r00[2], *r01[2], *r10[2], *r11[2];
    double *i00[2], *i01[2], *i10[2], *i11[2];
    double *ext_pot_r00, *ext_pot_r01, *ext_pot_r10, *ext_pot_r11;
    double *ext_pot_i00, *ext_pot_i01, *ext_pot_i10, *ext_pot_i11;
    double a;
    double b;
    int sense;
    size_t halo_x, halo_y, tile_width, tile_height;
    bool imag_time;
    // NOTE: block rows must be 16 byte aligned
    //       block height must be even
    static const size_t block_width = BLOCK_WIDTH;
    static const size_t block_height = BLOCK_HEIGHT;

    int start_x, inner_end_x, start_y, inner_start_y,  inner_end_y;
    int *periods;
#ifdef HAVE_MPI
    MPI_Comm cartcomm;
    int neighbors[4];
    MPI_Request req[32];
    MPI_Status statuses[32];
    MPI_Datatype horizontalBorder, verticalBorder;
#endif
};


#endif
