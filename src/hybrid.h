/**
 * Distributed Trotter-Suzuki solver
 * Copyright (C) 2012 Peter Wittek
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

#ifdef _OPENMP
#include <omp.h>
#endif

#include "trotter.h"
#include "cpublock.h"
#include "cc2kernel.h"

class HybridKernel: public ITrotterKernel {
public:
    HybridKernel(float *p_real, float *p_imag, float a, float b, int matrix_width, int matrix_height, int halo_x, int halo_y, MPI_Comm cartcomm);
    ~HybridKernel();
    void run_kernel();
    void run_kernel_on_halo();
    void wait_for_completion();
    void get_sample(size_t dest_stride, size_t x, size_t y, size_t width, size_t height, float * dest_real, float * dest_imag) const;
    bool runs_in_place() const {
        return false;
    }
    std::string get_name() const {
        std::stringstream name;
        name << "Hybrid";
#ifdef _OPENMP
        name << "-OpenMP-" << omp_get_max_threads();
#endif
        return name.str();
    };

    void start_halo_exchange();
    void finish_halo_exchange();

private:
    dim3 numBlocks;
    dim3 threadsPerBlock;
    cudaStream_t stream;

    float *p_real[2];
    float *p_imag[2];
    float *pdev_real[2];
    float *pdev_imag[2];
    float a;
    float b;
    int sense;
    size_t halo_x, halo_y, tile_width, tile_height;
    static const size_t block_width = BLOCK_WIDTH;
    static const size_t block_height = BLOCK_HEIGHT;
    size_t gpu_tile_width, gpu_tile_height, gpu_start_x, gpu_start_y;
    size_t n_bands_on_cpu;

    MPI_Comm cartcomm;
    int neighbors[4];
    int start_x, inner_end_x, start_y, inner_start_y,  inner_end_y;
    MPI_Request req[8];
    MPI_Status statuses[8];
    MPI_Datatype horizontalBorder, verticalBorder;
};

#endif
