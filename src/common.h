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
 
#ifndef __COMMON_H
#define __COMMON_H

#include <string>
#include <complex>

//#define DEBUG
#define CUDA

static const double h_a = cos(0.02);
static const double h_b = sin(0.02);

void calculate_borders(int coord, int dim, int * start, int *end, int *inner_start, int *inner_end, int length, int halo);
void print_complex_matrix(std::string filename, float * matrix_real, float * matrix_imag, size_t stride, size_t width, size_t height);
void print_matrix(std::string filename, float * matrix, size_t stride, size_t width, size_t height);
void init_p(float *p_real, float *p_imag, int start_x, int end_x, int start_y, int end_y);
void memcpy2D(void * dst, size_t dstride, const void * src, size_t sstride, size_t width, size_t height);
void get_quadrant_sample(const float * r00, const float * r01, const float * r10, const float * r11,
                         const float * i00, const float * i01, const float * i10, const float * i11,
                         size_t src_stride, size_t dest_stride,
                         size_t x, size_t y, size_t width, size_t height,
                         float * dest_real, float * dest_imag);

#endif
