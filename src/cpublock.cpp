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
 
#include <mpi.h>

#include "common.h"
#include "cpublock.h"

// Helpers
static void block_kernel_vertical(size_t start_offset, size_t stride, size_t width, size_t height, float a, float b, float * p_real, float * p_imag) {
    for (size_t idx = start_offset, peer = idx + stride; idx < width; idx += 2, peer += 2) {
        float tmp_real = p_real[idx];
        float tmp_imag = p_imag[idx];
        p_real[idx] = a * tmp_real - b * p_imag[peer];
        p_imag[idx] = a * tmp_imag + b * p_real[peer];
        p_real[peer] = a * p_real[peer] - b * tmp_imag;
        p_imag[peer] = a * p_imag[peer] + b * tmp_real;
    }
    for (size_t y = 1; y < height - 1; ++y) {
        for (size_t idx = y * stride + (start_offset + y) % 2, peer = idx + stride; idx < y * stride + width; idx += 2, peer += 2) {
            float tmp_real = p_real[idx];
            float tmp_imag = p_imag[idx];
            p_real[idx] = a * tmp_real - b * p_imag[peer];
            p_imag[idx] = a * tmp_imag + b * p_real[peer];
            p_real[peer] = a * p_real[peer] - b * tmp_imag;
            p_imag[peer] = a * p_imag[peer] + b * tmp_real;
        }
    }
}

static void block_kernel_horizontal(size_t start_offset, size_t stride, size_t width, size_t height, float a, float b, float * p_real, float * p_imag) {
    for (size_t y = 0; y < height; ++y) {
        for (size_t idx = y * stride + (start_offset + y) % 2, peer = idx + 1; idx < y * stride + width - 1; idx += 2, peer += 2) {
            float tmp_real = p_real[idx];
            float tmp_imag = p_imag[idx];
            p_real[idx] = a * tmp_real - b * p_imag[peer];
            p_imag[idx] = a * tmp_imag + b * p_real[peer];
            p_real[peer] = a * p_real[peer] - b * tmp_imag;
            p_imag[peer] = a * p_imag[peer] + b * tmp_real;
        }
    }
}


static void full_step(size_t stride, size_t width, size_t height, float a, float b, float * real, float * imag) {
    block_kernel_vertical  (0u, stride, width, height, a, b, real, imag);
    block_kernel_horizontal(0u, stride, width, height, a, b, real, imag);
    block_kernel_vertical  (1u, stride, width, height, a, b, real, imag);
    block_kernel_horizontal(1u, stride, width, height, a, b, real, imag);
    block_kernel_horizontal(1u, stride, width, height, a, b, real, imag);
    block_kernel_vertical  (1u, stride, width, height, a, b, real, imag);
    block_kernel_horizontal(0u, stride, width, height, a, b, real, imag);
    block_kernel_vertical  (0u, stride, width, height, a, b, real, imag);
}

void CPUBlock::process_band(size_t read_y, size_t read_height, size_t write_offset, size_t write_height, float a, float b, const float * p_real, const float * p_imag, float * next_real, float * next_imag) {
    float *block_real=new float[block_height * block_width];
    float *block_imag=new float[block_height * block_width];

    if (tile_width <= block_width) {
        // One full block
        memcpy2D(block_real, block_width * sizeof(float), &p_real[read_y * tile_width], tile_width * sizeof(float), tile_width * sizeof(float), read_height);
        memcpy2D(block_imag, block_width * sizeof(float), &p_imag[read_y * tile_width], tile_width * sizeof(float), tile_width * sizeof(float), read_height);
        full_step(block_width, tile_width, read_height, a, b, block_real, block_imag);
        memcpy2D(&next_real[(read_y + write_offset) * tile_width], tile_width * sizeof(float), &block_real[write_offset * block_width], block_width * sizeof(float), tile_width * sizeof(float), write_height);
        memcpy2D(&next_imag[(read_y + write_offset) * tile_width], tile_width * sizeof(float), &block_imag[write_offset * block_width], block_width * sizeof(float), tile_width * sizeof(float), write_height);
    } else {

        // First block [0..block_width - halo_x]
        memcpy2D(block_real, block_width * sizeof(float), &p_real[read_y * tile_width], tile_width * sizeof(float), block_width * sizeof(float), read_height);
        memcpy2D(block_imag, block_width * sizeof(float), &p_imag[read_y * tile_width], tile_width * sizeof(float), block_width * sizeof(float), read_height);
        full_step(block_width, block_width, read_height, a, b, block_real, block_imag);
        memcpy2D(&next_real[(read_y + write_offset) * tile_width], tile_width * sizeof(float), &block_real[write_offset * block_width], block_width * sizeof(float), (block_width - halo_x) * sizeof(float), write_height);
        memcpy2D(&next_imag[(read_y + write_offset) * tile_width], tile_width * sizeof(float), &block_imag[write_offset * block_width], block_width * sizeof(float), (block_width - halo_x) * sizeof(float), write_height);

        size_t block_start;
        for (block_start = block_width - 2 * halo_x; block_start < tile_width - block_width; block_start += block_width - 2 * halo_x) {
            memcpy2D(block_real, block_width * sizeof(float), &p_real[read_y * tile_width + block_start], tile_width * sizeof(float), block_width * sizeof(float), read_height);
            memcpy2D(block_imag, block_width * sizeof(float), &p_imag[read_y * tile_width + block_start], tile_width * sizeof(float), block_width * sizeof(float), read_height);
            full_step(block_width, block_width, read_height, a, b, block_real, block_imag);
            memcpy2D(&next_real[(read_y + write_offset) * tile_width + block_start + halo_x], tile_width * sizeof(float), &block_real[write_offset * block_width + halo_x], block_width * sizeof(float), (block_width - 2 * halo_x) * sizeof(float), write_height);
            memcpy2D(&next_imag[(read_y + write_offset) * tile_width + block_start + halo_x], tile_width * sizeof(float), &block_imag[write_offset * block_width + halo_x], block_width * sizeof(float), (block_width - 2 * halo_x) * sizeof(float), write_height);
        }
        // Last block
        memcpy2D(block_real, block_width * sizeof(float), &p_real[read_y * tile_width + block_start], tile_width * sizeof(float), (tile_width - block_start) * sizeof(float), read_height);
        memcpy2D(block_imag, block_width * sizeof(float), &p_imag[read_y * tile_width + block_start], tile_width * sizeof(float), (tile_width - block_start) * sizeof(float), read_height);
        full_step(block_width, tile_width - block_start, read_height, a, b, block_real, block_imag);
        memcpy2D(&next_real[(read_y + write_offset) * tile_width + block_start + halo_x], tile_width * sizeof(float), &block_real[write_offset * block_width + halo_x], block_width * sizeof(float), (tile_width - block_start - halo_x) * sizeof(float), write_height);
        memcpy2D(&next_imag[(read_y + write_offset) * tile_width + block_start + halo_x], tile_width * sizeof(float), &block_imag[write_offset * block_width + halo_x], block_width * sizeof(float), (tile_width - block_start - halo_x) * sizeof(float), write_height);
    }
    delete[] block_real;
    delete[] block_imag;
}


// Class methods
CPUBlock::CPUBlock(float *_p_real, float *_p_imag, float _a, float _b, size_t _tile_width, size_t _tile_height, int _halo_x, int _halo_y):
    orig_real(_p_real),
    orig_imag(_p_imag),
    a(_a),
    b(_b),
    tile_width(_tile_width),
    tile_height(_tile_height),
    halo_x(_halo_x),
    halo_y(_halo_y),
    sense(0)
{
    p_real[0] = new float[tile_width * tile_height];
    p_real[1] = new float[tile_width * tile_height];
    p_imag[0] = new float[tile_width * tile_height];
    p_imag[1] = new float[tile_width * tile_height];

    memcpy(p_real[0], _p_real, tile_width * tile_height * sizeof(float));
    memcpy(p_imag[0], _p_imag, tile_width * tile_height * sizeof(float));
}

CPUBlock::~CPUBlock() {
    delete[] p_real[0];
    delete[] p_real[1];
    delete[] p_imag[0];
    delete[] p_imag[1];
}

void CPUBlock::run_kernel(int k) {
    switch (k) {
    case 1:
        kernel8(p_real[sense], p_imag[sense], p_real[1-sense], p_imag[1-sense]);
        break;

    default:
        // TODO: throw
        break;
    }
    sense = 1 - sense;
}

void CPUBlock::run_kernels() {
    run_kernel(1);
}

void CPUBlock::wait_for_completion() { }

void CPUBlock::copy_results() {
    memcpy(orig_real, p_real[sense], tile_width * tile_height * sizeof(float));
    memcpy(orig_imag, p_imag[sense], tile_width * tile_height * sizeof(float));
}

void CPUBlock::get_sample(size_t dest_stride, size_t x, size_t y, size_t width, size_t height, float * dest_real, float * dest_imag) const {
    memcpy2D(dest_real, dest_stride * sizeof(float), &(p_real[sense][y * tile_width + x]), tile_width * sizeof(float), width * sizeof(float), height);
    memcpy2D(dest_imag, dest_stride * sizeof(float), &(p_imag[sense][y * tile_width + x]), tile_width * sizeof(float), width * sizeof(float), height);
}

void CPUBlock::kernel8(const float *p_real, const float *p_imag, float * next_real, float * next_imag) {

    if (tile_height <= BLOCK_HEIGHT) {
        // One full band
        process_band(0, tile_height, 0, tile_height, a, b, p_real, p_imag, next_real, next_imag);
    } else {

#ifdef _OPENMP
        size_t block_start;
        #pragma omp parallel default(shared) private(block_start)
        {
            #pragma omp sections nowait
            {
                #pragma omp section
                {
                    // First band
                    process_band(0, BLOCK_HEIGHT, 0, BLOCK_HEIGHT - halo_y, a, b, p_real, p_imag, next_real, next_imag);
                }
                #pragma omp section
                {
                    // Last band
                    block_start = tile_height - BLOCK_HEIGHT + (tile_height - BLOCK_HEIGHT) % (BLOCK_HEIGHT - 2 * halo_y);
                    process_band(block_start, tile_height - block_start, halo_y, tile_height - block_start - halo_y, a, b, p_real, p_imag, next_real, next_imag);
                }
            }

            #pragma omp for schedule(runtime) nowait
            for (block_start = BLOCK_HEIGHT - 2 * halo_y; block_start < tile_height - BLOCK_HEIGHT; block_start += BLOCK_HEIGHT - 2 * halo_y) {
                process_band(block_start, BLOCK_HEIGHT, halo_y, BLOCK_HEIGHT - 2 * halo_y, a, b, p_real, p_imag, next_real, next_imag);
            }

            #pragma omp barrier
        }
#else

        // First band
        process_band(0, BLOCK_HEIGHT, 0, BLOCK_HEIGHT - halo_y, a, b, p_real, p_imag, next_real, next_imag);

        size_t block_start;
        for (block_start = BLOCK_HEIGHT - 2 * halo_y; block_start < tile_height - BLOCK_HEIGHT; block_start += BLOCK_HEIGHT - 2 * halo_y) {
            process_band(block_start, BLOCK_HEIGHT, halo_y, BLOCK_HEIGHT - 2 * halo_y, a, b, p_real, p_imag, next_real, next_imag);
        }

        // Last band
        process_band(block_start, tile_height - block_start, halo_y, tile_height - block_start - halo_y, a, b, p_real, p_imag, next_real, next_imag);
#endif /* _OPENMP */
    }
}

void CPUBlock::initialize_MPI(MPI_Comm _cartcomm, int _start_x, int _inner_end_x, int _start_y, int _inner_start_y, int _inner_end_y)
{
    cartcomm=_cartcomm;
    MPI_Cart_shift(cartcomm, 0, 1, &neighbors[UP], &neighbors[DOWN]);
    MPI_Cart_shift(cartcomm, 1, 1, &neighbors[LEFT], &neighbors[RIGHT]);
    start_x = _start_x;
    inner_end_x = _inner_end_x;
    start_y = _start_y;
    inner_start_y = _inner_start_y;
    inner_end_y = _inner_end_y;

    // Halo exchange uses wave pattern to communicate
    // halo_x-wide inner rows are sent first to left and right
    // Then full length rows are exchanged to the top and bottom
    int count = inner_end_y-inner_start_y;	// The number of rows in the halo submatrix
    int block_length = halo_x;	// The number of columns in the halo submatrix
    int stride = tile_width;	// The combined width of the matrix with the halo
    MPI_Type_vector (count, block_length, stride, MPI_FLOAT, &verticalBorder);
    MPI_Type_commit (&verticalBorder);

    count = halo_y;	// The vertical halo in rows
    block_length = tile_width;	// The number of columns of the matrix
    stride = tile_width;	// The combined width of the matrix with the halo
    MPI_Type_vector (count, block_length, stride, MPI_FLOAT, &horizontalBorder);
    MPI_Type_commit (&horizontalBorder);
}

void CPUBlock::exchange_borders() {
    MPI_Request req[8];
    MPI_Status statuses[8];
    int offset=0;
    // Halo exchange: LEFT/RIGHT
    offset = (inner_start_y-start_y)*tile_width;
    MPI_Irecv(p_real[sense]+offset, 1, verticalBorder, neighbors[LEFT], 1, cartcomm, req);
    MPI_Irecv(p_imag[sense]+offset, 1, verticalBorder, neighbors[LEFT], 2, cartcomm, req+1);
    offset = (inner_start_y-start_y)*tile_width+inner_end_x-start_x;
    MPI_Irecv(p_real[sense]+offset, 1, verticalBorder, neighbors[RIGHT], 3, cartcomm, req+2);
    MPI_Irecv(p_imag[sense]+offset, 1, verticalBorder, neighbors[RIGHT], 4, cartcomm, req+3);

    offset=(inner_start_y-start_y)*tile_width+inner_end_x-halo_x-start_x;
    MPI_Isend(p_real[sense]+offset, 1, verticalBorder, neighbors[RIGHT], 1, cartcomm,req+4);
    MPI_Isend(p_imag[sense]+offset, 1, verticalBorder, neighbors[RIGHT], 2, cartcomm,req+5);
    offset=(inner_start_y-start_y)*tile_width+halo_x;
    MPI_Isend(p_real[sense]+offset, 1, verticalBorder, neighbors[LEFT], 3, cartcomm,req+6);
    MPI_Isend(p_imag[sense]+offset, 1, verticalBorder, neighbors[LEFT], 4, cartcomm,req+7);

    MPI_Waitall(8, req, statuses);

    // Halo exchange: UP/DOWN
    offset = 0;
    MPI_Irecv(p_real[sense]+offset, 1, horizontalBorder, neighbors[UP], 1, cartcomm, req);
    MPI_Irecv(p_imag[sense]+offset, 1, horizontalBorder, neighbors[UP], 2, cartcomm, req+1);
    offset = (inner_end_y-start_y)*tile_width;
    MPI_Irecv(p_real[sense]+offset, 1, horizontalBorder, neighbors[DOWN], 3, cartcomm, req+2);
    MPI_Irecv(p_imag[sense]+offset, 1, horizontalBorder, neighbors[DOWN], 4, cartcomm, req+3);

    offset=(inner_end_y-halo_y-start_y)*tile_width;
    MPI_Isend(p_real[sense]+offset, 1, horizontalBorder, neighbors[DOWN], 1, cartcomm,req+4);
    MPI_Isend(p_imag[sense]+offset, 1, horizontalBorder, neighbors[DOWN], 2, cartcomm,req+5);
    offset=halo_y*tile_width;
    MPI_Isend(p_real[sense]+offset, 1, horizontalBorder, neighbors[UP], 3, cartcomm,req+6);
    MPI_Isend(p_imag[sense]+offset, 1, horizontalBorder, neighbors[UP], 4, cartcomm,req+7);

    MPI_Waitall(8, req, statuses);
}
