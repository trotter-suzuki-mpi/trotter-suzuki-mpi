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

#include <math.h>

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
void block_kernel_potential(size_t stride, size_t width, size_t height, double a, double b, double coupling_const, size_t tile_width, const double *external_pot_real, const double *external_pot_imag, double * p_real, double * p_imag) {
    for (size_t y = 0; y < height; ++y) {
        for (size_t idx = y * stride, idx_pot = y * tile_width; idx < y * stride + width; ++idx, ++idx_pot) {
            double norm_2 = p_real[idx] * p_real[idx] + p_imag[idx] * p_imag[idx];
            double tmp = p_real[idx];
            p_real[idx] = external_pot_real[idx_pot] * tmp - external_pot_imag[idx_pot] * p_imag[idx];
            p_imag[idx] = external_pot_real[idx_pot] * p_imag[idx] + external_pot_imag[idx_pot] * tmp;
            
            tmp = p_real[idx];
            p_real[idx] = cos(coupling_const * norm_2) * tmp + sin(coupling_const * norm_2) * p_imag[idx];
            p_imag[idx] = cos(coupling_const * norm_2) * p_imag[idx] - sin(coupling_const * norm_2) * tmp;
        }
    }
}

//double time potential
void block_kernel_potential_imaginary(size_t stride, size_t width, size_t height, double a, double b, double coupling_const, size_t tile_width, const double *external_pot_real, const double *external_pot_imag, double * p_real, double * p_imag) {
    for (size_t y = 0; y < height; ++y) {
        for (size_t idx = y * stride, idx_pot = y * tile_width; idx < y * stride + width; ++idx, ++idx_pot) {
			double tmp = exp(-1. * coupling_const * (p_real[idx] * p_real[idx] + p_imag[idx] * p_imag[idx]));
            p_real[idx] = external_pot_real[idx_pot] * p_real[idx];
            p_imag[idx] = external_pot_real[idx_pot] * p_imag[idx];
            
            p_real[idx] = tmp * p_real[idx];
            p_imag[idx] = tmp * p_imag[idx];
        }
    }
}

//rotation
void block_kernel_rotation(size_t stride, size_t width, size_t height, int offset_x, int offset_y, double alpha_x, double alpha_y, double * p_real, double * p_imag) {
	
	double tmp_r, tmp_i;
	
	for (int j = 0, y = offset_y; j < height; ++j, ++y) {
		double alpha_yy = - 0.5 * alpha_y * y;
		double a = cos(alpha_yy), b = sin(alpha_yy);
		for (int i = 0, idx = j * stride, peer = idx + 1; i < width - 1; i += 2, idx += 2, peer += 2) {
			tmp_r = p_real[idx], tmp_i = p_imag[idx];
			p_real[idx] = a * p_real[idx] + b * p_real[peer];
			p_imag[idx] = a * p_imag[idx] + b * p_imag[peer];
			p_real[peer] = a * p_real[peer] - b * tmp_r;
			p_imag[peer] = a * p_imag[peer] - b * tmp_i;
		}
		for (int i = 1, idx = j * stride + 1, peer = idx + 1; i < width - 1; i += 2, idx += 2, peer += 2) {
			tmp_r = p_real[idx], tmp_i = p_imag[idx];
			p_real[idx] = a * p_real[idx] + b * p_real[peer];
			p_imag[idx] = a * p_imag[idx] + b * p_imag[peer];
			p_real[peer] = a * p_real[peer] - b * tmp_r;
			p_imag[peer] = a * p_imag[peer] - b * tmp_i;
		}
	}
	
	for (int i = 0, x = offset_x; i < width; ++i, ++x) {
		double alpha_xx = alpha_x * x;
		double a = cos(alpha_xx), b = sin(alpha_xx);
		for (int j = 0, idx = i, peer = stride + idx; j < height - 1; j += 2, idx += 2 * stride, peer += 2 * stride) {
			tmp_r = p_real[idx], tmp_i = p_imag[idx];
			p_real[idx] = a * p_real[idx] + b * p_real[peer];
			p_imag[idx] = a * p_imag[idx] + b * p_imag[peer];
			p_real[peer] = a * p_real[peer] - b * tmp_r;
			p_imag[peer] = a * p_imag[peer] - b * tmp_i;
		}
		for (int j = 1, idx = j * stride + i, peer = stride + idx; j < height - 1; j += 2, idx += 2 * stride, peer += 2 * stride) {
			tmp_r = p_real[idx], tmp_i = p_imag[idx];
			p_real[idx] = a * p_real[idx] + b * p_real[peer];
			p_imag[idx] = a * p_imag[idx] + b * p_imag[peer];
			p_real[peer] = a * p_real[peer] - b * tmp_r;
			p_imag[peer] = a * p_imag[peer] - b * tmp_i;
		}
	}
	
	for (int j = 0, y = offset_y; j < height; ++j, ++y) {
		double alpha_yy = - 0.5 * alpha_y * y;
		double a = cos(alpha_yy), b = sin(alpha_yy);
		for (int i = 0, idx = j * stride, peer = idx + 1; i < width - 1; i += 2, idx += 2, peer += 2) {
			tmp_r = p_real[idx], tmp_i = p_imag[idx];
			p_real[idx] = a * p_real[idx] + b * p_real[peer];
			p_imag[idx] = a * p_imag[idx] + b * p_imag[peer];
			p_real[peer] = a * p_real[peer] - b * tmp_r;
			p_imag[peer] = a * p_imag[peer] - b * tmp_i;
		}
		for (int i = 1, idx = j * stride + 1, peer = idx + 1; i < width - 1; i += 2, idx += 2, peer += 2) {
			tmp_r = p_real[idx], tmp_i = p_imag[idx];
			p_real[idx] = a * p_real[idx] + b * p_real[peer];
			p_imag[idx] = a * p_imag[idx] + b * p_imag[peer];
			p_real[peer] = a * p_real[peer] - b * tmp_r;
			p_imag[peer] = a * p_imag[peer] - b * tmp_i;
		}
	}
}

void block_kernel_rotation_imaginary(size_t stride, size_t width, size_t height, int offset_x, int offset_y, double alpha_x, double alpha_y, double * p_real, double * p_imag) {
	
	double tmp_r, tmp_i;
	
	for (int j = 0, y = offset_y; j < height; ++j, ++y) {
		double alpha_yy = - 0.5 * alpha_y * y;
		double a = cosh(alpha_yy), b = sinh(alpha_yy);
		for (int i = 0, idx = j * stride, peer = idx + 1; i < width - 1; i += 2, idx += 2, peer += 2) {
			tmp_r = p_real[idx], tmp_i = p_imag[idx];
			p_real[idx] = a * p_real[idx] + b * p_imag[peer];
			p_imag[idx] = a * p_imag[idx] - b * p_real[peer];
			p_real[peer] = - b * tmp_i + a * p_real[peer];
			p_imag[peer] = b * tmp_r + a * p_imag[peer];
		}
		for (int i = 1, idx = j * stride + 1, peer = idx + 1; i < width - 1; i += 2, idx += 2, peer += 2) {
			tmp_r = p_real[idx], tmp_i = p_imag[idx];
			p_real[idx] = a * p_real[idx] + b * p_imag[peer];
			p_imag[idx] = a * p_imag[idx] - b * p_real[peer];
			p_real[peer] = - b * tmp_i + a * p_real[peer];
			p_imag[peer] = b * tmp_r + a * p_imag[peer];
		}
	}
	
	for (int i = 0, x = offset_x; i < width; ++i, ++x) {
		double alpha_xx = alpha_x * x;
		double a = cosh(alpha_xx), b = sinh(alpha_xx);
		for (int j = 0, idx = i, peer = stride + idx; j < height - 1; j += 2, idx += 2 * stride, peer += 2 * stride) {
			tmp_r = p_real[idx], tmp_i = p_imag[idx];
			p_real[idx] = a * p_real[idx] + b * p_imag[peer];
			p_imag[idx] = a * p_imag[idx] - b * p_real[peer];
			p_real[peer] = - b * tmp_i + a * p_real[peer];
			p_imag[peer] = b * tmp_r + a * p_imag[peer];
		}
		for (int j = 1, idx = j * stride + i, peer = stride + idx; j < height - 1; j += 2, idx += 2 * stride, peer += 2 * stride) {
			tmp_r = p_real[idx], tmp_i = p_imag[idx];
			p_real[idx] = a * p_real[idx] + b * p_imag[peer];
			p_imag[idx] = a * p_imag[idx] - b * p_real[peer];
			p_real[peer] = - b * tmp_i + a * p_real[peer];
			p_imag[peer] = b * tmp_r + a * p_imag[peer];
		}
	}
	
	for (int j = 0, y = offset_y; j < height; ++j, ++y) {
		double alpha_yy = - 0.5 * alpha_y * y;
		double a = cosh(alpha_yy), b = sinh(alpha_yy);
		for (int i = 0, idx = j * stride, peer = idx + 1; i < width - 1; i += 2, idx += 2, peer += 2) {
			tmp_r = p_real[idx], tmp_i = p_imag[idx];
			p_real[idx] = a * p_real[idx] + b * p_imag[peer];
			p_imag[idx] = a * p_imag[idx] - b * p_real[peer];
			p_real[peer] = - b * tmp_i + a * p_real[peer];
			p_imag[peer] = b * tmp_r + a * p_imag[peer];
		}
		for (int i = 1, idx = j * stride + 1, peer = idx + 1; i < width - 1; i += 2, idx += 2, peer += 2) {
			tmp_r = p_real[idx], tmp_i = p_imag[idx];
			p_real[idx] = a * p_real[idx] + b * p_imag[peer];
			p_imag[idx] = a * p_imag[idx] - b * p_real[peer];
			p_real[peer] = - b * tmp_i + a * p_real[peer];
			p_imag[peer] = b * tmp_r + a * p_imag[peer];
		}
	}
}

void full_step(size_t stride, size_t width, size_t height, int offset_x, int offset_y, double alpha_x, double alpha_y, double a, double b, double coupling_const, size_t tile_width, const double *external_pot_real, const double *external_pot_imag, double * real, double * imag) {
    block_kernel_vertical  (0u, stride, width, height, a, b, real, imag);
    block_kernel_horizontal(0u, stride, width, height, a, b, real, imag);
    block_kernel_vertical  (1u, stride, width, height, a, b, real, imag);
    block_kernel_horizontal(1u, stride, width, height, a, b, real, imag);
    block_kernel_potential (stride, width, height, a, b, coupling_const, tile_width, external_pot_real, external_pot_imag, real, imag);
    block_kernel_rotation  (stride, width, height, offset_x, offset_y, alpha_x, alpha_y, real, imag);
    block_kernel_horizontal(1u, stride, width, height, a, b, real, imag);
    block_kernel_vertical  (1u, stride, width, height, a, b, real, imag);
    block_kernel_horizontal(0u, stride, width, height, a, b, real, imag);
    block_kernel_vertical  (0u, stride, width, height, a, b, real, imag);
}

void full_step_imaginary(size_t stride, size_t width, size_t height, int offset_x, int offset_y, double alpha_x, double alpha_y, double a, double b, double coupling_const, size_t tile_width, const double *external_pot_real, const double *external_pot_imag, double * real, double * imag) {
    
    block_kernel_vertical_imaginary  (0u, stride, width, height, a, b, real, imag);
    block_kernel_horizontal_imaginary(0u, stride, width, height, a, b, real, imag);
    block_kernel_vertical_imaginary  (1u, stride, width, height, a, b, real, imag);
    block_kernel_horizontal_imaginary(1u, stride, width, height, a, b, real, imag);
    block_kernel_potential_imaginary (stride, width, height, a, b, coupling_const, tile_width, external_pot_real, external_pot_imag, real, imag);
    block_kernel_rotation_imaginary  (stride, width, height, offset_x, offset_y, alpha_x, alpha_y, real, imag);
    block_kernel_horizontal_imaginary(1u, stride, width, height, a, b, real, imag);
    block_kernel_vertical_imaginary  (1u, stride, width, height, a, b, real, imag);
    block_kernel_horizontal_imaginary(0u, stride, width, height, a, b, real, imag);
    block_kernel_vertical_imaginary  (0u, stride, width, height, a, b, real, imag);
}

void process_sides(int offset_tile_x, int offset_tile_y, double alpha_x, double alpha_y, size_t tile_width, size_t block_width, size_t halo_x, size_t read_y, size_t read_height, size_t write_offset, size_t write_height, double a, double b, double coupling_const, const double *external_pot_real, const double *external_pot_imag, const double * p_real, const double * p_imag, double * next_real, double * next_imag, double * block_real, double * block_imag, bool imag_time) {
	
    // First block [0..block_width - halo_x]
    memcpy2D(block_real, block_width * sizeof(double), &p_real[read_y * tile_width], tile_width * sizeof(double), block_width * sizeof(double), read_height);
    memcpy2D(block_imag, block_width * sizeof(double), &p_imag[read_y * tile_width], tile_width * sizeof(double), block_width * sizeof(double), read_height);
    if(imag_time)
        full_step_imaginary(block_width, block_width, read_height, offset_tile_x, offset_tile_y + read_y, alpha_x, alpha_y, a, b, coupling_const, tile_width, &external_pot_real[read_y * tile_width], &external_pot_imag[read_y * tile_width], block_real, block_imag);
    else
        full_step(block_width, block_width, read_height, offset_tile_x, offset_tile_y + read_y, alpha_x, alpha_y, a, b, coupling_const, tile_width, &external_pot_real[read_y * tile_width], &external_pot_imag[read_y * tile_width], block_real, block_imag);
	memcpy2D(&next_real[(read_y + write_offset) * tile_width], tile_width * sizeof(double), &block_real[write_offset * block_width], block_width * sizeof(double), (block_width - halo_x) * sizeof(double), write_height);
    memcpy2D(&next_imag[(read_y + write_offset) * tile_width], tile_width * sizeof(double), &block_imag[write_offset * block_width], block_width * sizeof(double), (block_width - halo_x) * sizeof(double), write_height);

    size_t block_start = ((tile_width - block_width) / (block_width - 2 * halo_x) + 1) * (block_width - 2 * halo_x);
    // Last block
    memcpy2D(block_real, block_width * sizeof(double), &p_real[read_y * tile_width + block_start], tile_width * sizeof(double), (tile_width - block_start) * sizeof(double), read_height);
    memcpy2D(block_imag, block_width * sizeof(double), &p_imag[read_y * tile_width + block_start], tile_width * sizeof(double), (tile_width - block_start) * sizeof(double), read_height);
    if(imag_time)
        full_step_imaginary(block_width, tile_width - block_start, read_height, offset_tile_x + block_start, offset_tile_y + read_y, alpha_x, alpha_y, a, b, coupling_const, tile_width, &external_pot_real[read_y * tile_width + block_start], &external_pot_imag[read_y * tile_width + block_start], block_real, block_imag);
    else
        full_step(block_width, tile_width - block_start, read_height, offset_tile_x + block_start, offset_tile_y + read_y, alpha_x, alpha_y, a, b, coupling_const, tile_width, &external_pot_real[read_y * tile_width + block_start], &external_pot_imag[read_y * tile_width + block_start], block_real, block_imag);
    memcpy2D(&next_real[(read_y + write_offset) * tile_width + block_start + halo_x], tile_width * sizeof(double), &block_real[write_offset * block_width + halo_x], block_width * sizeof(double), (tile_width - block_start - halo_x) * sizeof(double), write_height);
    memcpy2D(&next_imag[(read_y + write_offset) * tile_width + block_start + halo_x], tile_width * sizeof(double), &block_imag[write_offset * block_width + halo_x], block_width * sizeof(double), (tile_width - block_start - halo_x) * sizeof(double), write_height);
}

void process_band(int offset_tile_x, int offset_tile_y, double alpha_x, double alpha_y, size_t tile_width, size_t block_width, size_t block_height, size_t halo_x, size_t read_y, size_t read_height, size_t write_offset, size_t write_height, double a, double b, double coupling_const, const double *external_pot_real, const double *external_pot_imag, const double * p_real, const double * p_imag, double * next_real, double * next_imag, int inner, int sides, bool imag_time) {
    double *block_real = new double[block_height * block_width];
    double *block_imag = new double[block_height * block_width];

    if (tile_width <= block_width) {
        if (sides) {
            // One full block
            memcpy2D(block_real, block_width * sizeof(double), &p_real[read_y * tile_width], tile_width * sizeof(double), tile_width * sizeof(double), read_height);
            memcpy2D(block_imag, block_width * sizeof(double), &p_imag[read_y * tile_width], tile_width * sizeof(double), tile_width * sizeof(double), read_height);
            if(imag_time)
                full_step_imaginary(block_width, tile_width, read_height, offset_tile_x, offset_tile_y + read_y, alpha_x, alpha_y, a, b, coupling_const, tile_width, &external_pot_real[read_y * tile_width], &external_pot_imag[read_y * tile_width], block_real, block_imag);
            else
                full_step(block_width, tile_width, read_height, offset_tile_x, offset_tile_y + read_y, alpha_x, alpha_y, a, b, coupling_const, tile_width, &external_pot_real[read_y * tile_width], &external_pot_imag[read_y * tile_width], block_real, block_imag);
            memcpy2D(&next_real[(read_y + write_offset) * tile_width], tile_width * sizeof(double), &block_real[write_offset * block_width], block_width * sizeof(double), tile_width * sizeof(double), write_height);
            memcpy2D(&next_imag[(read_y + write_offset) * tile_width], tile_width * sizeof(double), &block_imag[write_offset * block_width], block_width * sizeof(double), tile_width * sizeof(double), write_height);
        }
    }
    else {
        if (sides) {
            process_sides(offset_tile_x, offset_tile_y, alpha_x, alpha_y, tile_width, block_width, halo_x, read_y, read_height, write_offset, write_height, a, b, coupling_const, external_pot_real, external_pot_imag, p_real, p_imag, next_real, next_imag, block_real, block_imag, imag_time);
        }
        if (inner) {
            for (size_t block_start = block_width - 2 * halo_x; block_start < tile_width - block_width; block_start += block_width - 2 * halo_x) {
                memcpy2D(block_real, block_width * sizeof(double), &p_real[read_y * tile_width + block_start], tile_width * sizeof(double), block_width * sizeof(double), read_height);
                memcpy2D(block_imag, block_width * sizeof(double), &p_imag[read_y * tile_width + block_start], tile_width * sizeof(double), block_width * sizeof(double), read_height);
                if(imag_time)
                    full_step_imaginary(block_width, block_width, read_height, offset_tile_x + block_start, offset_tile_y + read_y, alpha_x, alpha_y, a, b, coupling_const, tile_width, &external_pot_real[read_y * tile_width + block_start], &external_pot_imag[read_y * tile_width + block_start], block_real, block_imag);
                else
                    full_step(block_width, block_width, read_height, offset_tile_x + block_start, offset_tile_y + read_y, alpha_x, alpha_y, a, b, coupling_const, tile_width, &external_pot_real[read_y * tile_width + block_start], &external_pot_imag[read_y * tile_width + block_start], block_real, block_imag);
                memcpy2D(&next_real[(read_y + write_offset) * tile_width + block_start + halo_x], tile_width * sizeof(double), &block_real[write_offset * block_width + halo_x], block_width * sizeof(double), (block_width - 2 * halo_x) * sizeof(double), write_height);
                memcpy2D(&next_imag[(read_y + write_offset) * tile_width + block_start + halo_x], tile_width * sizeof(double), &block_imag[write_offset * block_width + halo_x], block_width * sizeof(double), (block_width - 2 * halo_x) * sizeof(double), write_height);
            }
        }
    }
    

    delete[] block_real;
    delete[] block_imag;
}

// Class methods
CPUBlock::CPUBlock(double *_p_real, double *_p_imag, double *_external_pot_real, double *_external_pot_imag, double _a, double _b, double _coupling_const, double _delta_x, double _delta_y, int matrix_width, int matrix_height, int _halo_x, int _halo_y, int *_periods, double _norm, bool _imag_time, double _alpha_x, double _alpha_y, int _rot_coord_x, int _rot_coord_y
#ifdef HAVE_MPI
                   , MPI_Comm _cartcomm
#endif
                  ):
    delta_x(_delta_x),
    delta_y(_delta_y),
    sense(0),
    halo_x(_halo_x),
    halo_y(_halo_y),
    imag_time(_imag_time),
    alpha_x(_alpha_x),
    alpha_y(_alpha_y),
    rot_coord_x(_rot_coord_x),
    rot_coord_y(_rot_coord_y) {

	a = new double [1];
	b = new double [1];
	coupling_const = new double [1];
	norm = new int [1];
	
	a[0] = _a;
    b[0] = _b;
    coupling_const[0] = _coupling_const;
    norm[0] = _norm;
    status = 0;
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

    calculate_borders(coords[1], dims[1], &start_x, &end_x, &inner_start_x, &inner_end_x, matrix_width - 2 * periods[1]*halo_x, halo_x, periods[1]);
    calculate_borders(coords[0], dims[0], &start_y, &end_y, &inner_start_y, &inner_end_y, matrix_height - 2 * periods[0]*halo_y, halo_y, periods[0]);
    tile_width = end_x - start_x;
    tile_height = end_y - start_y;

    p_real[0][0] = _p_real;
    p_imag[0][0] = _p_imag;
    p_real[1][0] = new double[tile_width * tile_height];
    p_imag[1][0] = new double[tile_width * tile_height];
    external_pot_real[0] = _external_pot_real;
    external_pot_imag[0] = _external_pot_imag;

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

CPUBlock::CPUBlock(double **_p_real, double **_p_imag, double **_external_pot_real, double **_external_pot_imag, double *_a, double *_b, double *_coupling_const, double _delta_x, double _delta_y, 
                   int matrix_width, int matrix_height, int _halo_x, int _halo_y, int *_periods, double *_norm, bool _imag_time, double _alpha_x, double _alpha_y, int _rot_coord_x, int _rot_coord_y
#ifdef HAVE_MPI
                   , MPI_Comm _cartcomm
#endif
                  ):    
    delta_x(_delta_x),
    delta_y(_delta_y),
    sense(0),
    state(0),
    halo_x(_halo_x),
    halo_y(_halo_y),
    imag_time(_imag_time),
    alpha_x(_alpha_x),
    alpha_y(_alpha_y),
    rot_coord_x(_rot_coord_x),
    rot_coord_y(_rot_coord_y) {
    
    a = _a;
    b = _b;
    norm = _norm;
    tot_norm = norm[0] + norm[1];

    coupling_const = _coupling_const;
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

    calculate_borders(coords[1], dims[1], &start_x, &end_x, &inner_start_x, &inner_end_x, matrix_width - 2 * periods[1]*halo_x, halo_x, periods[1]);
    calculate_borders(coords[0], dims[0], &start_y, &end_y, &inner_start_y, &inner_end_y, matrix_height - 2 * periods[0]*halo_y, halo_y, periods[0]);
    tile_width = end_x - start_x;
    tile_height = end_y - start_y;
    
    for(int i = 0; i < 2; i++) {
        p_real[i][0] = _p_real[i];
        p_imag[i][0] = _p_imag[i];
        p_real[i][1] = new double[tile_width * tile_height];
        p_imag[i][1] = new double[tile_width * tile_height];
        memcpy2D(p_real[i][1], tile_width * sizeof(double), p_real[i][0], tile_width * sizeof(double), tile_width * sizeof(double), tile_height);
        memcpy2D(p_imag[i][1], tile_width * sizeof(double), p_imag[i][0], tile_width * sizeof(double), tile_width * sizeof(double), tile_height);
        external_pot_real[i] = _external_pot_real[i];
        external_pot_imag[i] = _external_pot_imag[i];
    }

#ifdef HAVE_MPI
    // Halo exchange uses wave pattern to communicate
    // halo_x-wide inner rows are sent first to left and right
    // Then full length rows are exchanged to the top and bottom
    int count = inner_end_y - inner_start_y;    // The number of rows in the halo submatrix
    int block_length = halo_x;  // The number of columns in the halo submatrix
    int stride = tile_width;    // The combined width of the matrix with the halo
    MPI_Type_vector (count, block_length, stride, MPI_DOUBLE, &verticalBorder);
    MPI_Type_commit (&verticalBorder);

    count = halo_y; // The vertical halo in rows
    block_length = tile_width;  // The number of columns of the matrix
    stride = tile_width;    // The combined width of the matrix with the halo
    MPI_Type_vector (count, block_length, stride, MPI_DOUBLE, &horizontalBorder);
    MPI_Type_commit (&horizontalBorder);
#endif
}
            
CPUBlock::~CPUBlock() {
    delete[] p_real[1];
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
        process_band(start_x - rot_coord_x, start_y - rot_coord_y, alpha_x, alpha_y, tile_width, block_width, block_height, halo_x, 0, tile_height, 0, tile_height, a, b, coupling_const, external_pot_real, external_pot_imag, p_real[sense], p_imag[sense], p_real[1 - sense], p_imag[1 - sense], inner, sides, imag_time);
    }
    else {

        // Sides
        inner = 0;
        sides = 1;
        size_t block_start;
        for (block_start = block_height - 2 * halo_y; block_start < tile_height - block_height; block_start += block_height - 2 * halo_y) {
            process_band(start_x - rot_coord_x, start_y - rot_coord_y, alpha_x, alpha_y, tile_width, block_width, block_height, halo_x, block_start, block_height, halo_y, block_height - 2 * halo_y, a, b, coupling_const, external_pot_real, external_pot_imag, p_real[sense], p_imag[sense], p_real[1 - sense], p_imag[1 - sense], inner, sides, imag_time);
        }

        // First band
        inner = 1;
        sides = 1;
        process_band(start_x - rot_coord_x, start_y - rot_coord_y, alpha_x, alpha_y, tile_width, block_width, block_height, halo_x, 0, block_height, 0, block_height - halo_y, a, b, coupling_const, external_pot_real, external_pot_imag, p_real[sense], p_imag[sense], p_real[1 - sense], p_imag[1 - sense], inner, sides, imag_time);

        // Last band
        inner = 1;
        sides = 1;
        process_band(start_x - rot_coord_x, start_y - rot_coord_y, alpha_x, alpha_y, tile_width, block_width, block_height, halo_x, block_start, tile_height - block_start, halo_y, tile_height - block_start - halo_y, a, b, coupling_const, external_pot_real, external_pot_imag, p_real[sense], p_imag[sense], p_real[1 - sense], p_imag[1 - sense], inner, sides, imag_time);
    }
}

void CPUBlock::wait_for_completion(int iteration) {
    if(imag_time) {
        //normalization
        int nProcs = 1;
#ifdef HAVE_MPI
        MPI_Comm_size(cartcomm, &nProcs);
#endif
        
        double sum = 0., *sums;
        sums = new double[nProcs];
        for(int i = inner_start_y - start_y; i < inner_end_y - start_y; i++) {
            for(int j = inner_start_x - start_x; j < inner_end_x - start_x; j++) {
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
        double _norm = sqrt(tot_sum * delta_x * delta_y / norm);

        for(size_t i = 0; i < tile_height; i++) {
            for(size_t j = 0; j < tile_width; j++) {
                p_real[sense][j + i * tile_width] /= _norm;
                p_imag[sense][j + i * tile_width] /= _norm;
            }
        }
        delete[] sums;
    }
}

void CPUBlock::get_sample(size_t dest_stride, size_t x, size_t y, size_t width, size_t height, double * dest_real, double * dest_imag) const {
    memcpy2D(dest_real, dest_stride * sizeof(double), &(p_real[sense][y * tile_width + x]), tile_width * sizeof(double), width * sizeof(double), height);
    memcpy2D(dest_imag, dest_stride * sizeof(double), &(p_imag[sense][y * tile_width + x]), tile_width * sizeof(double), width * sizeof(double), height);
}

void CPUBlock::get_sample(size_t dest_stride, size_t x, size_t y, size_t width, size_t height, double ** dest_real, double ** dest_imag) const {
    memcpy2D(dest_real[0], dest_stride * sizeof(double), &(p_real[0][sense][y * tile_width + x]), tile_width * sizeof(double), width * sizeof(double), height);
    memcpy2D(dest_imag[0], dest_stride * sizeof(double), &(p_imag[0][sense][y * tile_width + x]), tile_width * sizeof(double), width * sizeof(double), height);
    
    memcpy2D(dest_real[1], dest_stride * sizeof(double), &(p_real[1][sense][y * tile_width + x]), tile_width * sizeof(double), width * sizeof(double), height);
    memcpy2D(dest_imag[1], dest_stride * sizeof(double), &(p_imag[1][sense][y * tile_width + x]), tile_width * sizeof(double), width * sizeof(double), height);
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
        for (int block_start = block_height - 2 * halo_y; block_start < int(tile_height - block_height); block_start += block_height - 2 * halo_y) {
            process_band(start_x - rot_coord_x, start_y - rot_coord_y, alpha_x, alpha_y, tile_width, block_width, block_height, halo_x, block_start, block_height, halo_y, block_height - 2 * halo_y, a, b, coupling_const, external_pot_real, external_pot_imag, p_real, p_imag, next_real, next_imag, inner, sides, imag_time);
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
