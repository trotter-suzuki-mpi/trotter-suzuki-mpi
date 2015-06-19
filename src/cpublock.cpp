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
#include <iostream>
#include "common.h"
#include "cpublock.h"
#ifdef HAVE_MPI
#include <mpi.h>
#endif

// Helpers
void block_kernel_vertical(size_t start_offset, size_t stride, size_t width, size_t height, double a, double b, double * p_real, double * p_imag) {
    for (size_t idx = start_offset, peer = idx + stride; idx < width; idx += 2, peer += 2) {
        double tmp_real = p_real[idx];
        double tmp_imag = p_imag[idx];
        p_real[idx] = a * tmp_real - b * p_imag[peer];
        p_imag[idx] = a * tmp_imag + b * p_real[peer];
        p_real[peer] = a * p_real[peer] - b * tmp_imag;
        p_imag[peer] = a * p_imag[peer] + b * tmp_real;
    }
    for (size_t y = 1; y < height - 1; ++y) {
        for (size_t idx = y * stride + (start_offset + y) % 2, peer = idx + stride; idx < y * stride + width; idx += 2, peer += 2) {
            double tmp_real = p_real[idx];
            double tmp_imag = p_imag[idx];
            p_real[idx] = a * tmp_real - b * p_imag[peer];
            p_imag[idx] = a * tmp_imag + b * p_real[peer];
            p_real[peer] = a * p_real[peer] - b * tmp_imag;
            p_imag[peer] = a * p_imag[peer] + b * tmp_real;
        }
    }
}

void block_kernel_vertical_imaginary(size_t start_offset, size_t stride, size_t width, size_t height, double a, double b, double * p_real, double * p_imag) {
    for (size_t idx = start_offset, peer = idx + stride; idx < width; idx += 2, peer += 2) {
        double tmp_real = p_real[idx];
        double tmp_imag = p_imag[idx];
        p_real[idx] = a * tmp_real + b * p_real[peer];
        p_imag[idx] = a * tmp_imag + b * p_imag[peer];
        p_real[peer] = a * p_real[peer] + b * tmp_real;
        p_imag[peer] = a * p_imag[peer] + b * tmp_imag;
    }
    for (size_t y = 1; y < height - 1; ++y) {
        for (size_t idx = y * stride + (start_offset + y) % 2, peer = idx + stride; idx < y * stride + width; idx += 2, peer += 2) {
            double tmp_real = p_real[idx];
            double tmp_imag = p_imag[idx];
            p_real[idx] = a * tmp_real + b * p_real[peer];
            p_imag[idx] = a * tmp_imag + b * p_imag[peer];
            p_real[peer] = a * p_real[peer] + b * tmp_real;
            p_imag[peer] = a * p_imag[peer] + b * tmp_imag;
        }
    }
}

void block_kernel_horizontal(size_t start_offset, size_t stride, size_t width, size_t height, double a, double b, double * p_real, double * p_imag) {
    for (size_t y = 0; y < height; ++y) {
        for (size_t idx = y * stride + (start_offset + y) % 2, peer = idx + 1; idx < y * stride + width - 1; idx += 2, peer += 2) {
            double tmp_real = p_real[idx];
            double tmp_imag = p_imag[idx];
            p_real[idx] = a * tmp_real - b * p_imag[peer];
            p_imag[idx] = a * tmp_imag + b * p_real[peer];
            p_real[peer] = a * p_real[peer] - b * tmp_imag;
            p_imag[peer] = a * p_imag[peer] + b * tmp_real;
        }
    }
}

void block_kernel_horizontal_imaginary(size_t start_offset, size_t stride, size_t width, size_t height, double a, double b, double * p_real, double * p_imag) {
    for (size_t y = 0; y < height; ++y) {
        for (size_t idx = y * stride + (start_offset + y) % 2, peer = idx + 1; idx < y * stride + width - 1; idx += 2, peer += 2) {
            double tmp_real = p_real[idx];
            double tmp_imag = p_imag[idx];
            p_real[idx] = a * tmp_real + b * p_real[peer];
            p_imag[idx] = a * tmp_imag + b * p_imag[peer];
            p_real[peer] = a * p_real[peer] + b * tmp_real;
            p_imag[peer] = a * p_imag[peer] + b * tmp_imag;
        }
    }
}

//double time potential
void block_kernel_potential(size_t stride, size_t width, size_t height, double a, double b, size_t tile_width, const double *external_pot_real, const double *external_pot_imag, double * p_real, double * p_imag) {
    for (size_t y = 0; y < height; ++y) {
        for (size_t idx = y * stride, idx_pot = y * tile_width; idx < y * stride + width; ++idx, ++idx_pot) {
            double tmp = p_real[idx];
            p_real[idx] = external_pot_real[idx_pot] * tmp - external_pot_imag[idx_pot] * p_imag[idx];
            p_imag[idx] = external_pot_real[idx_pot] * p_imag[idx] + external_pot_imag[idx_pot] * tmp;
        }
    }
}

//double time potential
void block_kernel_potential_imaginary(size_t stride, size_t width, size_t height, double a, double b, size_t tile_width, const double *external_pot_real, const double *external_pot_imag, double * p_real, double * p_imag) {
    for (size_t y = 0; y < height; ++y) {
        for (size_t idx = y * stride, idx_pot = y * tile_width; idx < y * stride + width; ++idx, ++idx_pot) {
            p_real[idx] = external_pot_real[idx_pot] * p_real[idx];
            p_imag[idx] = external_pot_real[idx_pot] * p_imag[idx];
        }
    }
}

void full_step(size_t stride, size_t width, size_t height, double a, double b, size_t tile_width, const double *external_pot_real, const double *external_pot_imag, double * real, double * imag) {
    block_kernel_vertical  (0u, stride, width, height, a, b, real, imag);
    block_kernel_horizontal(0u, stride, width, height, a, b, real, imag);
    block_kernel_vertical  (1u, stride, width, height, a, b, real, imag);
    block_kernel_horizontal(1u, stride, width, height, a, b, real, imag);
    block_kernel_potential (stride, width, height, a, b, tile_width, external_pot_real, external_pot_imag, real, imag);
    block_kernel_horizontal(1u, stride, width, height, a, b, real, imag);
    block_kernel_vertical  (1u, stride, width, height, a, b, real, imag);
    block_kernel_horizontal(0u, stride, width, height, a, b, real, imag);
    block_kernel_vertical  (0u, stride, width, height, a, b, real, imag);
}

void full_step_imaginary(size_t stride, size_t width, size_t height, double a, double b, size_t tile_width, const double *external_pot_real, const double *external_pot_imag, double * real, double * imag) {
    block_kernel_vertical_imaginary  (0u, stride, width, height, a, b, real, imag);
    block_kernel_horizontal_imaginary(0u, stride, width, height, a, b, real, imag);
    block_kernel_vertical_imaginary  (1u, stride, width, height, a, b, real, imag);
    block_kernel_horizontal_imaginary(1u, stride, width, height, a, b, real, imag);
    block_kernel_potential_imaginary (stride, width, height, a, b, tile_width, external_pot_real, external_pot_imag, real, imag);
    block_kernel_horizontal_imaginary(1u, stride, width, height, a, b, real, imag);
    block_kernel_vertical_imaginary  (1u, stride, width, height, a, b, real, imag);
    block_kernel_horizontal_imaginary(0u, stride, width, height, a, b, real, imag);
    block_kernel_vertical_imaginary  (0u, stride, width, height, a, b, real, imag);
}

void process_sides(size_t tile_width, size_t block_width, size_t halo_x, size_t read_y, size_t read_height, size_t write_offset, size_t write_height, double a, double b, const double *external_pot_real, const double *external_pot_imag, const double * p_real, const double * p_imag, double * next_real, double * next_imag, double * block_real, double * block_imag, bool imag_time) {
    // First block [0..block_width - halo_x]
    memcpy2D(block_real, block_width * sizeof(double), &p_real[read_y * tile_width], tile_width * sizeof(double), block_width * sizeof(double), read_height);
    memcpy2D(block_imag, block_width * sizeof(double), &p_imag[read_y * tile_width], tile_width * sizeof(double), block_width * sizeof(double), read_height);
    if(imag_time)
        full_step_imaginary(block_width, block_width, read_height, a, b, tile_width, &external_pot_real[read_y * tile_width], &external_pot_imag[read_y * tile_width], block_real, block_imag);
    else
        full_step(block_width, block_width, read_height, a, b, tile_width, &external_pot_real[read_y * tile_width], &external_pot_imag[read_y * tile_width], block_real, block_imag);
    memcpy2D(&next_real[(read_y + write_offset) * tile_width], tile_width * sizeof(double), &block_real[write_offset * block_width], block_width * sizeof(double), (block_width - halo_x) * sizeof(double), write_height);
    memcpy2D(&next_imag[(read_y + write_offset) * tile_width], tile_width * sizeof(double), &block_imag[write_offset * block_width], block_width * sizeof(double), (block_width - halo_x) * sizeof(double), write_height);

    size_t block_start = ((tile_width - block_width) / (block_width - 2 * halo_x) + 1) * (block_width - 2 * halo_x);
    // Last block
    memcpy2D(block_real, block_width * sizeof(double), &p_real[read_y * tile_width + block_start], tile_width * sizeof(double), (tile_width - block_start) * sizeof(double), read_height);
    memcpy2D(block_imag, block_width * sizeof(double), &p_imag[read_y * tile_width + block_start], tile_width * sizeof(double), (tile_width - block_start) * sizeof(double), read_height);
    if(imag_time)
        full_step_imaginary(block_width, tile_width - block_start, read_height, a, b, tile_width, &external_pot_real[read_y * tile_width + block_start], &external_pot_imag[read_y * tile_width + block_start], block_real, block_imag);
    else
        full_step(block_width, tile_width - block_start, read_height, a, b, tile_width, &external_pot_real[read_y * tile_width + block_start], &external_pot_imag[read_y * tile_width + block_start], block_real, block_imag);
    memcpy2D(&next_real[(read_y + write_offset) * tile_width + block_start + halo_x], tile_width * sizeof(double), &block_real[write_offset * block_width + halo_x], block_width * sizeof(double), (tile_width - block_start - halo_x) * sizeof(double), write_height);
    memcpy2D(&next_imag[(read_y + write_offset) * tile_width + block_start + halo_x], tile_width * sizeof(double), &block_imag[write_offset * block_width + halo_x], block_width * sizeof(double), (tile_width - block_start - halo_x) * sizeof(double), write_height);
}

void process_band(size_t tile_width, size_t block_width, size_t block_height, size_t halo_x, size_t read_y, size_t read_height, size_t write_offset, size_t write_height, double a, double b, const double *external_pot_real, const double *external_pot_imag, const double * p_real, const double * p_imag, double * next_real, double * next_imag, int inner, int sides, bool imag_time) {
    double *block_real = new double[block_height * block_width];
    double *block_imag = new double[block_height * block_width];

    if (tile_width <= block_width) {
        if (sides) {
            // One full block
            memcpy2D(block_real, block_width * sizeof(double), &p_real[read_y * tile_width], tile_width * sizeof(double), tile_width * sizeof(double), read_height);
            memcpy2D(block_imag, block_width * sizeof(double), &p_imag[read_y * tile_width], tile_width * sizeof(double), tile_width * sizeof(double), read_height);
            if(imag_time)
                full_step_imaginary(block_width, tile_width, read_height, a, b, tile_width, &external_pot_real[read_y * tile_width], &external_pot_imag[read_y * tile_width], block_real, block_imag);
            else
                full_step(block_width, tile_width, read_height, a, b, tile_width, &external_pot_real[read_y * tile_width], &external_pot_imag[read_y * tile_width], block_real, block_imag);
            memcpy2D(&next_real[(read_y + write_offset) * tile_width], tile_width * sizeof(double), &block_real[write_offset * block_width], block_width * sizeof(double), tile_width * sizeof(double), write_height);
            memcpy2D(&next_imag[(read_y + write_offset) * tile_width], tile_width * sizeof(double), &block_imag[write_offset * block_width], block_width * sizeof(double), tile_width * sizeof(double), write_height);
        }
    }
    else {
        if (sides) {
            process_sides(tile_width, block_width, halo_x, read_y, read_height, write_offset, write_height, a, b, external_pot_real, external_pot_imag, p_real, p_imag, next_real, next_imag, block_real, block_imag, imag_time);
        }
        if (inner) {
            for (size_t block_start = block_width - 2 * halo_x; block_start < tile_width - block_width; block_start += block_width - 2 * halo_x) {
                memcpy2D(block_real, block_width * sizeof(double), &p_real[read_y * tile_width + block_start], tile_width * sizeof(double), block_width * sizeof(double), read_height);
                memcpy2D(block_imag, block_width * sizeof(double), &p_imag[read_y * tile_width + block_start], tile_width * sizeof(double), block_width * sizeof(double), read_height);
                if(imag_time)
                    full_step_imaginary(block_width, block_width, read_height, a, b, tile_width, &external_pot_real[read_y * tile_width + block_start], &external_pot_imag[read_y * tile_width + block_start], block_real, block_imag);
                else
                    full_step(block_width, block_width, read_height, a, b, tile_width, &external_pot_real[read_y * tile_width + block_start], &external_pot_imag[read_y * tile_width + block_start], block_real, block_imag);
                memcpy2D(&next_real[(read_y + write_offset) * tile_width + block_start + halo_x], tile_width * sizeof(double), &block_real[write_offset * block_width + halo_x], block_width * sizeof(double), (block_width - 2 * halo_x) * sizeof(double), write_height);
                memcpy2D(&next_imag[(read_y + write_offset) * tile_width + block_start + halo_x], tile_width * sizeof(double), &block_imag[write_offset * block_width + halo_x], block_width * sizeof(double), (block_width - 2 * halo_x) * sizeof(double), write_height);
            }
        }
    }
    delete[] block_real;
    delete[] block_imag;
}

// Class methods
CPUBlock::CPUBlock(double *_p_real, double *_p_imag, double *_external_pot_real, double *_external_pot_imag, double _a, double _b, int matrix_width, int matrix_height, int _halo_x, int _halo_y, int *_periods, bool _imag_time
#ifdef HAVE_MPI
                   , MPI_Comm _cartcomm
#endif
                   ):
    a(_a),
    b(_b),
    sense(0),
    halo_x(_halo_x),
    halo_y(_halo_y),
    imag_time(_imag_time) {

    periods = _periods;
    int rank, coords[2], dims[2] = {0, 0};
#ifdef HAVE_MPI
    cartcomm = _cartcomm;
    MPI_Cart_shift(cartcomm, 0, 1, &neighbors[UP], &neighbors[DOWN]);
    MPI_Cart_shift(cartcomm, 1, 1, &neighbors[LEFT], &neighbors[RIGHT]);
    MPI_Comm_rank(cartcomm, &rank);
    MPI_Cart_get(cartcomm, 2, dims, periods, coords);
#else
    dims[0] = dims[1] = 1;
    rank = 0;
    coords[0] = coords[1] = 0;
#endif
    int inner_start_x = 0, end_x = 0, end_y = 0;
    calculate_borders(coords[1], dims[1], &start_x, &end_x, &inner_start_x, &inner_end_x, matrix_width - 2 * periods[1]*halo_x, halo_x, periods[1]);
    calculate_borders(coords[0], dims[0], &start_y, &end_y, &inner_start_y, &inner_end_y, matrix_height - 2 * periods[0]*halo_y, halo_y, periods[0]);
    tile_width = end_x - start_x;
    tile_height = end_y - start_y;

    p_real[0] = _p_real;
    p_imag[0] = _p_imag;
    p_real[1] = new double[tile_width * tile_height];
    p_imag[1] = new double[tile_width * tile_height];
    external_pot_real = _external_pot_real;
    external_pot_imag = _external_pot_imag;

#ifdef HAVE_MPI
    // Halo exchange uses wave pattern to communicate
    // halo_x-wide inner rows are sent first to left and right
    // Then full length rows are exchanged to the top and bottom
    int count = inner_end_y - inner_start_y;	// The number of rows in the halo submatrix
    int block_length = halo_x;	// The number of columns in the halo submatrix
    int stride = tile_width;	// The combined width of the matrix with the halo
    MPI_Type_vector (count, block_length, stride, MPI_DOUBLE, &verticalBorder);
    MPI_Type_commit (&verticalBorder);

    count = halo_y;	// The vertical halo in rows
    block_length = tile_width;	// The number of columns of the matrix
    stride = tile_width;	// The combined width of the matrix with the halo
    MPI_Type_vector (count, block_length, stride, MPI_DOUBLE, &horizontalBorder);
    MPI_Type_commit (&horizontalBorder);
#endif
}

CPUBlock::~CPUBlock() {
    delete[] p_real[0];
    delete[] p_real[1];
    delete[] p_imag[0];
    delete[] p_imag[1];
}

void CPUBlock::run_kernel() {
    kernel8(p_real[sense], p_imag[sense], p_real[1 - sense], p_imag[1 - sense]);
    sense = 1 - sense;
}

void CPUBlock::run_kernel_on_halo() {
    int inner = 0, sides = 0;
    if (tile_height <= block_height) {
        // One full band
        inner = 1;
        sides = 1;
        process_band(tile_width, block_width, block_height, halo_x, 0, tile_height, 0, tile_height, a, b, external_pot_real, external_pot_imag, p_real[sense], p_imag[sense], p_real[1 - sense], p_imag[1 - sense], inner, sides, imag_time);
    }
    else {

        // Sides
        inner = 0;
        sides = 1;
        size_t block_start;
        for (block_start = block_height - 2 * halo_y; block_start < tile_height - block_height; block_start += block_height - 2 * halo_y) {
            process_band(tile_width, block_width, block_height, halo_x, block_start, block_height, halo_y, block_height - 2 * halo_y, a, b, external_pot_real, external_pot_imag, p_real[sense], p_imag[sense], p_real[1 - sense], p_imag[1 - sense], inner, sides, imag_time);
        }

        // First band
        inner = 1;
        sides = 1;
        process_band(tile_width, block_width, block_height, halo_x, 0, block_height, 0, block_height - halo_y, a, b, external_pot_real, external_pot_imag, p_real[sense], p_imag[sense], p_real[1 - sense], p_imag[1 - sense], inner, sides, imag_time);

        // Last band
        inner = 1;
        sides = 1;
        process_band(tile_width, block_width, block_height, halo_x, block_start, tile_height - block_start, halo_y, tile_height - block_start - halo_y, a, b, external_pot_real, external_pot_imag, p_real[sense], p_imag[sense], p_real[1 - sense], p_imag[1 - sense], inner, sides, imag_time);
    }
}

void CPUBlock::wait_for_completion(int iteration, int snapshots) {
    if(imag_time && ((iteration % 20) == 0 || ((snapshots > 0) && (iteration + 1) % snapshots == 0))) {
        //normalization
        int nProcs = 1;
#ifdef HAVE_MPI
        MPI_Comm_size(cartcomm, &nProcs);
#endif
        int height = tile_height - halo_y;
        int width = tile_width - halo_x;
        double sum = 0., sums[nProcs];
        for(int i = halo_y; i < height; i++) {
            for(int j = halo_x; j < width; j++) {
                sum += p_real[sense][j + i * tile_width] * p_real[sense][j + i * tile_width] + p_imag[sense][j + i * tile_width] * p_imag[sense][j + i * tile_width];
            }
        }
#ifdef HAVE_MPI
        MPI_Allgather(&sum, 1, MPI_DOUBLE, sums, 1, MPI_DOUBLE, cartcomm);
#else
        sums[0] = sum;
#endif
        double tot_sum = 0.;
        for(int i = 0; i < nProcs; i++)
            tot_sum += sums[i];
        double norm = sqrt(tot_sum);

        for(size_t i = 0; i < tile_height; i++) {
            for(size_t j = 0; j < tile_width; j++) {
                p_real[sense][j + i * tile_width] /= norm;
                p_imag[sense][j + i * tile_width] /= norm;
            }
        }
    }
}

void CPUBlock::get_sample(size_t dest_stride, size_t x, size_t y, size_t width, size_t height, double * dest_real, double * dest_imag) const {
    memcpy2D(dest_real, dest_stride * sizeof(double), &(p_real[sense][y * tile_width + x]), tile_width * sizeof(double), width * sizeof(double), height);
    memcpy2D(dest_imag, dest_stride * sizeof(double), &(p_imag[sense][y * tile_width + x]), tile_width * sizeof(double), width * sizeof(double), height);
}

void CPUBlock::kernel8(const double *p_real, const double *p_imag, double * next_real, double * next_imag) {
    // Inner part
    int inner = 1, sides = 0;
#ifndef HAVE_MPI
    #pragma omp parallel default(shared)
#endif
    {
#ifndef HAVE_MPI
        #pragma omp for
#endif
        for (size_t block_start = block_height - 2 * halo_y; block_start < tile_height - block_height; block_start += block_height - 2 * halo_y) {
            process_band(tile_width, block_width, block_height, halo_x, block_start, block_height, halo_y, block_height - 2 * halo_y, a, b, external_pot_real, external_pot_imag, p_real, p_imag, next_real, next_imag, inner, sides, imag_time);
        }
    }
}

void CPUBlock::start_halo_exchange() {
    // Halo exchange: LEFT/RIGHT
#ifdef HAVE_MPI
    int offset = (inner_start_y - start_y) * tile_width;
    MPI_Irecv(p_real[1 - sense] + offset, 1, verticalBorder, neighbors[LEFT], 1, cartcomm, req);
    MPI_Irecv(p_imag[1 - sense] + offset, 1, verticalBorder, neighbors[LEFT], 2, cartcomm, req + 1);
    offset = (inner_start_y - start_y) * tile_width + inner_end_x - start_x;
    MPI_Irecv(p_real[1 - sense] + offset, 1, verticalBorder, neighbors[RIGHT], 3, cartcomm, req + 2);
    MPI_Irecv(p_imag[1 - sense] + offset, 1, verticalBorder, neighbors[RIGHT], 4, cartcomm, req + 3);

    offset = (inner_start_y - start_y) * tile_width + inner_end_x - halo_x - start_x;
    MPI_Isend(p_real[1 - sense] + offset, 1, verticalBorder, neighbors[RIGHT], 1, cartcomm, req + 4);
    MPI_Isend(p_imag[1 - sense] + offset, 1, verticalBorder, neighbors[RIGHT], 2, cartcomm, req + 5);
    offset = (inner_start_y - start_y) * tile_width + halo_x;
    MPI_Isend(p_real[1 - sense] + offset, 1, verticalBorder, neighbors[LEFT], 3, cartcomm, req + 6);
    MPI_Isend(p_imag[1 - sense] + offset, 1, verticalBorder, neighbors[LEFT], 4, cartcomm, req + 7);
#else
    if(periods[1] != 0) {
        int offset = (inner_start_y - start_y) * tile_width;
        memcpy2D(&(p_real[1 - sense][offset]), tile_width * sizeof(double), &(p_real[1 - sense][offset + tile_width - 2 * halo_x]), tile_width * sizeof(double), halo_x * sizeof(double), tile_height - 2 * halo_y);
        memcpy2D(&(p_imag[1 - sense][offset]), tile_width * sizeof(double), &(p_imag[1 - sense][offset + tile_width - 2 * halo_x]), tile_width * sizeof(double), halo_x * sizeof(double), tile_height - 2 * halo_y);
        memcpy2D(&(p_real[1 - sense][offset + tile_width - halo_x]), tile_width * sizeof(double), &(p_real[1 - sense][offset + halo_x]), tile_width * sizeof(double), halo_x * sizeof(double), tile_height - 2 * halo_y);
        memcpy2D(&(p_imag[1 - sense][offset + tile_width - halo_x]), tile_width * sizeof(double), &(p_imag[1 - sense][offset + halo_x]), tile_width * sizeof(double), halo_x * sizeof(double), tile_height - 2 * halo_y);
    }
#endif
}

void CPUBlock::finish_halo_exchange() {
#ifdef HAVE_MPI
    MPI_Waitall(8, req, statuses);

    // Halo exchange: UP/DOWN
    int offset = 0;
    MPI_Irecv(p_real[sense] + offset, 1, horizontalBorder, neighbors[UP], 1, cartcomm, req);
    MPI_Irecv(p_imag[sense] + offset, 1, horizontalBorder, neighbors[UP], 2, cartcomm, req + 1);
    offset = (inner_end_y - start_y) * tile_width;
    MPI_Irecv(p_real[sense] + offset, 1, horizontalBorder, neighbors[DOWN], 3, cartcomm, req + 2);
    MPI_Irecv(p_imag[sense] + offset, 1, horizontalBorder, neighbors[DOWN], 4, cartcomm, req + 3);

    offset = (inner_end_y - halo_y - start_y) * tile_width;
    MPI_Isend(p_real[sense] + offset, 1, horizontalBorder, neighbors[DOWN], 1, cartcomm, req + 4);
    MPI_Isend(p_imag[sense] + offset, 1, horizontalBorder, neighbors[DOWN], 2, cartcomm, req + 5);
    offset = halo_y * tile_width;
    MPI_Isend(p_real[sense] + offset, 1, horizontalBorder, neighbors[UP], 3, cartcomm, req + 6);
    MPI_Isend(p_imag[sense] + offset, 1, horizontalBorder, neighbors[UP], 4, cartcomm, req + 7);

    MPI_Waitall(8, req, statuses);
#else
    if(periods[0] != 0) {
        int offset = (inner_end_y - start_y) * tile_width;
        memcpy2D(&(p_real[sense][0]), tile_width * sizeof(double), &(p_real[sense][offset - halo_y * tile_width]), tile_width * sizeof(double), tile_width * sizeof(double), halo_y);
        memcpy2D(&(p_imag[sense][0]), tile_width * sizeof(double), &(p_imag[sense][offset - halo_y * tile_width]), tile_width * sizeof(double), tile_width * sizeof(double), halo_y);
        memcpy2D(&(p_real[sense][offset]), tile_width * sizeof(double), &(p_real[sense][halo_y * tile_width]), tile_width * sizeof(double), tile_width * sizeof(double), halo_y);
        memcpy2D(&(p_imag[sense][offset]), tile_width * sizeof(double), &(p_imag[sense][halo_y * tile_width]), tile_width * sizeof(double), tile_width * sizeof(double), halo_y);
    }
#endif
}
