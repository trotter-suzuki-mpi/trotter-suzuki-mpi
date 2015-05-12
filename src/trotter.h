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

//These are for the MPI NEIGHBOURS
#define UP    0
#define DOWN  1
#define LEFT  2
#define RIGHT 3

struct procs_topology {
    int rank, dimsx, dimsy;
};

procs_topology trotter(double h_a, double h_b, float * external_pot_real, float * external_pot_imag, float * p_real, float * p_imag, const int matrix_width, const int matrix_height, const int iterations, const int snapshots, const int kernel_type, int *periods, int argc, char** argv, const char *dirname);

class ITrotterKernel {
public:
    virtual ~ITrotterKernel() {};
    virtual void run_kernel() = 0;
    virtual void run_kernel_on_halo() = 0;
    virtual void wait_for_completion() = 0;
    virtual void get_sample(size_t dest_stride, size_t x, size_t y, size_t width, size_t height, float * dest_real, float * dest_imag) const = 0;

    virtual bool runs_in_place() const = 0;
    virtual std::string get_name() const = 0;

    virtual void start_halo_exchange() = 0;
    virtual void finish_halo_exchange() = 0;

};

#endif
