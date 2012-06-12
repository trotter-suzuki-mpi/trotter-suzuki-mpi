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
 
#ifndef __CPUBLOCK_H
#define __CPUBLOCK_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include "trotterkernel.h"

#define BLOCK_WIDTH 128u
#define BLOCK_HEIGHT 128u

//These are for the MPI NEIGHBOURS
#define UP    0
#define DOWN  1
#define LEFT  2
#define RIGHT 3

class CPUBlock: public ITrotterKernel {
public:
    CPUBlock(float *p_real, float *p_imag, float a, float b, size_t tile_width, size_t tile_height, int halo_x, int halo_y);
    ~CPUBlock();
    void run_kernel();
    void run_kernel_on_halo();
    void wait_for_completion();
    void copy_results();
    void get_sample(size_t dest_stride, size_t x, size_t y, size_t width, size_t height, float * dest_real, float * dest_imag) const;

    bool runs_in_place() const {
        return false;
    }
    std::string get_name() const {
#ifdef _OPENMP
        std::stringstream name;
        name << "OpenMP block kernel (" << omp_get_max_threads() << " threads)";
        return name.str();
#else
        return "CPU block kernel";
#endif
    };

    void initialize_MPI(MPI_Comm cartcomm, int _start_x, int _inner_end_x, int _start_y, int _inner_start_y, int _inner_end_y);
    void start_halo_exchange();
    void finish_halo_exchange();



private:
    void kernel8(const float *p_real, const float *p_imag, float * next_real, float * next_imag);
    void process_band(size_t, size_t, size_t, size_t, float, float, const float *, const float *, float *, float *, int, int);
    void process_sides(size_t read_y, size_t read_height, size_t write_offset, size_t write_height, float a, float b, const float * p_real, const float * p_imag, float * next_real, float * next_imag, float * block_real, float * block_imag);

    float *orig_real;
    float *orig_imag;
    float *p_real[2];
    float *p_imag[2];
    float a;
    float b;
    int sense;
    int halo_x, halo_y, tile_width, tile_height;
    static const size_t block_width=BLOCK_WIDTH;
    static const size_t block_height=BLOCK_HEIGHT;

    MPI_Comm cartcomm;
    int neighbors[4];
    MPI_Request req[8];
    MPI_Status statuses[8];    
    MPI_Datatype horizontalBorder, verticalBorder;
    int start_x, inner_end_x, start_y, inner_start_y,  inner_end_y;
};

#endif
