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
 
#ifndef __TROTTERKERNEL_H
#define __TROTTERKERNEL_H

#include <string>

#include <mpi.h>

//These are for the MPI NEIGHBOURS
#define UP    0
#define DOWN  1
#define LEFT  2
#define RIGHT 3

class ITrotterKernel {
public:
    virtual void run_kernels() = 0;
    virtual void wait_for_completion() = 0;
    virtual void get_sample(size_t dest_stride, size_t x, size_t y, size_t width, size_t height, float * dest_real, float * dest_imag) const = 0;

    virtual bool runs_in_place() const = 0;
    virtual std::string get_name() const = 0;
    virtual void initialize_MPI(MPI_Comm cartcomm, int _start_x, int _inner_end_x, int _start_y, int _inner_start_y, int _inner_end_y) = 0;
    virtual void exchange_borders() = 0;

};

#endif
