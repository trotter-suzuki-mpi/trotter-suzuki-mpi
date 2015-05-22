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
#include "trotter.h"

#if HAVE_CONFIG_H
#include <config.h>
#endif

struct MAGIC_NUMBER {
	float threshold_E, threshold_P;
    float expected_E;
    float expected_Px;
    float expected_Py;
    MAGIC_NUMBER();
};

void calculate_borders(int coord, int dim, int * start, int *end, int *inner_start, int *inner_end, int length, int halo, int periodic_bound);
void print_complex_matrix(std::string filename, float * matrix_real, float * matrix_imag, size_t stride, size_t width, size_t height);
void print_matrix(std::string filename, float * matrix, size_t stride, size_t width, size_t height);
void memcpy2D(void * dst, size_t dstride, const void * src, size_t sstride, size_t width, size_t height);
void get_quadrant_sample(const float * r00, const float * r01, const float * r10, const float * r11,
                         const float * i00, const float * i01, const float * i10, const float * i11,
                         size_t src_stride, size_t dest_stride,
                         size_t x, size_t y, size_t width, size_t height,
                         float * dest_real, float * dest_imag);
void get_quadrant_sample_to_buffer(const float * r00, const float * r01, const float * r10, const float * r11,
                                   const float * i00, const float * i01, const float * i10, const float * i11,
                                   size_t src_stride, size_t dest_stride,
                                   size_t x, size_t y, size_t width, size_t height,
                                   float * dest_real, float * dest_imag);

void expect_values(int dim, int iterations, int snapshots, float * hamilt_pot, float particle_mass, const char *dirname,
                   int *periods, int halo_x, int halo_y, MAGIC_NUMBER th_values);

#endif
