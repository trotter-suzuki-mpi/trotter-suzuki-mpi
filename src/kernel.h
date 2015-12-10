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

#ifndef __KERNEL_H
#define __KERNEL_H

#if HAVE_CONFIG_H
#include "config.h"
#endif
#include <string>

//These are for the MPI NEIGHBOURS
#define UP    0
#define DOWN  1
#define LEFT  2
#define RIGHT 3

/**
 * \brief This class define the prototipe of the kernel classes: CPU, SSE, GPU, Hybrid.
 */

class ITrotterKernel {
public:
    virtual ~ITrotterKernel() {};
    virtual void run_kernel() = 0;							///< Evolve the remaining blocks in the inner part of the tile.
    virtual void run_kernel_on_halo() = 0;					///< Evolve blocks of wave function at the edge of the tile. This comprises the halos.
    virtual void wait_for_completion(int iteration) = 0;	///< Sincronize all the processes at the end of halos communication. Perform normalization for imaginary time evolution.
    virtual void get_sample(size_t dest_stride, size_t x, size_t y, size_t width, size_t height, double * dest_real, double * dest_imag) const = 0;					///< Get the evolved wave function.
    virtual void get_sample2(size_t dest_stride, size_t x, size_t y, size_t width, size_t height, double ** dest_real, double ** dest_imag) const = 0;
    //virtual void get_sample() const = 0;
    virtual void normalization() = 0;
    virtual void rabi_coupling(double var, double delta_t) = 0;
    
    virtual bool runs_in_place() const = 0;
    virtual std::string get_name() const = 0;				///< Get kernel name.

    virtual void start_halo_exchange() = 0;					///< Exchange halos between processes.
    virtual void finish_halo_exchange() = 0;				///< Exchange halos between processes.

};

#endif
