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
 
#ifndef __CPUBLOCKSSE_H
#define __CPUBLOCKSSE_H

#include <iostream>
#include <sstream>
#include <cassert>
#include <cstdlib>
#include <xmmintrin.h>
#include <emmintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "trotterkernel.h"

#define BLOCK_WIDTH 128u
#define BLOCK_HEIGHT 128u

void process_sides_sse(size_t tile_width, size_t block_width, size_t block_height, size_t halo_x, size_t read_y, size_t read_height, size_t write_offset, size_t write_height, float a, float b, const float * r00, const float * r01, const float * r10, const float * r11, const float * i00, const float * i01, const float * i10, const float * i11, float * next_r00, float * next_r01, float * next_r10, float * next_r11, float * next_i00, float * next_i01, float * next_i10, float * next_i11, float * block_r00, float * block_r01, float * block_r10, float * block_r11, float * block_i00, float * block_i01, float * block_i10, float * block_i11);

void process_band_sse(size_t read_y, size_t read_height, size_t write_offset, size_t write_height, size_t block_width, size_t block_height, size_t tile_width, size_t halo_x, float a, float b, const float * r00, const float * r01, const float * r10, const float * r11, const float * i00, const float * i01, const float * i10, const float * i11, float * next_r00, float * next_r01, float * next_r10, float * next_r11, float * next_i00, float * next_i01, float * next_i10, float * next_i11, int inner, int sides);

class CPUBlockSSEKernel: public ITrotterKernel{
public:
    CPUBlockSSEKernel(float *p_real, float *p_imag, float _a, float _b, int tile_width, int tile_height, int halo_x, int halo_y);
    ~CPUBlockSSEKernel();
    void run_kernel();
    void run_kernel_on_halo();    
    void wait_for_completion();
    void get_sample(size_t dest_stride, size_t x, size_t y, size_t width, size_t height, float * dest_real, float * dest_imag) const;

    bool runs_in_place() const { return false; }
    std::string get_name() const {
#ifdef _OPENMP
        std::stringstream name;
        name << "OpenMP Block SSE kernel (" << omp_get_max_threads() << " threads)";
        return name.str();
#else
        return "CPU Block SSE kernel";
#endif
    };

    void initialize_MPI(MPI_Comm cartcomm, int _start_x, int _inner_end_x, int _start_y, int _inner_start_y, int _inner_end_y);
    void start_halo_exchange();
    void finish_halo_exchange();


private:
    float *p_real;
    float *p_imag;
    float *r00[2], *r01[2], *r10[2], *r11[2];
    float *i00[2], *i01[2], *i10[2], *i11[2];
    float a;
    float b;
    int sense;
    int halo_x, halo_y, tile_width, tile_height;
    // NOTE: block rows must be 16 byte aligned
    //       block height must be even
    static const size_t block_width=BLOCK_WIDTH;
    static const size_t block_height=BLOCK_HEIGHT;
    
    MPI_Comm cartcomm;
    int neighbors[4];
    int start_x, inner_end_x, start_y, inner_start_y,  inner_end_y;
    MPI_Request req[32];
    MPI_Status statuses[32];    
    MPI_Datatype horizontalBorder, verticalBorder;    

};


#endif
