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

#include <cassert>
#include <emmintrin.h>
#include <stdlib.h>

#include "common.h"
#include "cpublocksse.h"
#ifdef HAVE_MPI
#include <mpi.h>
#endif

/***************
* SSE variants *
***************/
#ifdef WIN32
template <int offset_y>
inline void update_shifty_sse(size_t stride, size_t width, size_t height, double a, double b, double * __restrict r1, double * __restrict i1, double * __restrict r2, double * __restrict i2) {
#else
template <int offset_y>
inline void update_shifty_sse(size_t stride, size_t width, size_t height, double a, double b, double * __restrict__ r1, double * __restrict__ i1, double * __restrict__ r2, double * __restrict__ i2) {
#endif
	__m128d aq, bq;
    aq = _mm_load1_pd(&a);
    bq = _mm_load1_pd(&b);
    for (size_t i = 0; i < height - offset_y; i++) {
        int idx1 = i * stride;
        int idx2 = (i + offset_y) * stride;
        size_t j = 0;
        for (; j < width - width % 2; j += 2, idx1 += 2, idx2 += 2) {
            __m128d r1q = _mm_load_pd(&r1[idx1]);
            __m128d i1q = _mm_load_pd(&i1[idx1]);
            __m128d r2q = _mm_load_pd(&r2[idx2]);
            __m128d i2q = _mm_load_pd(&i2[idx2]);
            __m128d next_r1q = _mm_sub_pd(_mm_mul_pd(r1q, aq), _mm_mul_pd(i2q, bq));
            __m128d next_i1q = _mm_add_pd(_mm_mul_pd(i1q, aq), _mm_mul_pd(r2q, bq));
            __m128d next_r2q = _mm_sub_pd(_mm_mul_pd(r2q, aq), _mm_mul_pd(i1q, bq));
            __m128d next_i2q = _mm_add_pd(_mm_mul_pd(i2q, aq), _mm_mul_pd(r1q, bq));
            _mm_store_pd(&r1[idx1], next_r1q);
            _mm_store_pd(&i1[idx1], next_i1q);
            _mm_store_pd(&r2[idx2], next_r2q);
            _mm_store_pd(&i2[idx2], next_i2q);
        }
        for (; j < width; ++j, ++idx1, ++idx2) {
            double next_r1 = a * r1[idx1] - b * i2[idx2];
            double next_i1 = a * i1[idx1] + b * r2[idx2];
            double next_r2 = a * r2[idx2] - b * i1[idx1];
            double next_i2 = a * i2[idx2] + b * r1[idx1];
            r1[idx1] = next_r1;
            i1[idx1] = next_i1;
            r2[idx2] = next_r2;
            i2[idx2] = next_i2;
        }
    }
}

#ifdef WIN32
template <int offset_y>
inline void update_shifty_sse_imaginary(size_t stride, size_t width, size_t height, double a, double b, double * __restrict r1, double * __restrict i1, double * __restrict r2, double * __restrict i2) {
#else
template <int offset_y>
inline void update_shifty_sse_imaginary(size_t stride, size_t width, size_t height, double a, double b, double * __restrict__ r1, double * __restrict__ i1, double * __restrict__ r2, double * __restrict__ i2) {
#endif
	__m128d aq, bq;
    aq = _mm_load1_pd(&a);
    bq = _mm_load1_pd(&b);
    for (size_t i = 0; i < height - offset_y; i++) {
        int idx1 = i * stride;
        int idx2 = (i + offset_y) * stride;
        size_t j = 0;
        for (; j < width - width % 2; j += 2, idx1 += 2, idx2 += 2) {
            __m128d r1q = _mm_load_pd(&r1[idx1]);
            __m128d i1q = _mm_load_pd(&i1[idx1]);
            __m128d r2q = _mm_load_pd(&r2[idx2]);
            __m128d i2q = _mm_load_pd(&i2[idx2]);
            __m128d next_r1q = _mm_add_pd(_mm_mul_pd(r1q, aq), _mm_mul_pd(r2q, bq));
            __m128d next_i1q = _mm_add_pd(_mm_mul_pd(i1q, aq), _mm_mul_pd(i2q, bq));
            __m128d next_r2q = _mm_add_pd(_mm_mul_pd(r2q, aq), _mm_mul_pd(r1q, bq));
            __m128d next_i2q = _mm_add_pd(_mm_mul_pd(i2q, aq), _mm_mul_pd(i1q, bq));
            _mm_store_pd(&r1[idx1], next_r1q);
            _mm_store_pd(&i1[idx1], next_i1q);
            _mm_store_pd(&r2[idx2], next_r2q);
            _mm_store_pd(&i2[idx2], next_i2q);
        }
        for (; j < width; ++j, ++idx1, ++idx2) {
            double next_r1 = a * r1[idx1] + b * r2[idx2];
            double next_i1 = a * i1[idx1] + b * i2[idx2];
            double next_r2 = a * r2[idx2] + b * r1[idx1];
            double next_i2 = a * i2[idx2] + b * i1[idx1];
            r1[idx1] = next_r1;
            i1[idx1] = next_i1;
            r2[idx2] = next_r2;
            i2[idx2] = next_i2;
        }
    }
}

#ifdef WIN32
template <int offset_x>
inline void update_shiftx_sse(size_t stride, size_t width, size_t height, double a, double b, double * __restrict r1, double * __restrict i1, double * __restrict r2, double * __restrict i2) {
#else
template <int offset_x>
inline void update_shiftx_sse(size_t stride, size_t width, size_t height, double a, double b, double * __restrict__ r1, double * __restrict__ i1, double * __restrict__ r2, double * __restrict__ i2) {
#endif	
	__m128d aq, bq;
    aq = _mm_load1_pd(&a);
    bq = _mm_load1_pd(&b);
    for (size_t i = 0; i < height; i++) {
        int idx1 = i * stride;
        int idx2 = i * stride + offset_x;
        size_t j = 0;
        for (; j < width - offset_x - (width - offset_x) % 2; j += 2, idx1 += 2, idx2 += 2) {
            __m128d r1q = _mm_load_pd(&r1[idx1]);
            __m128d i1q = _mm_load_pd(&i1[idx1]);
            __m128d r2q;
            __m128d i2q;
            if (offset_x == 0) {
                r2q = _mm_load_pd(&r2[idx2]);
                i2q = _mm_load_pd(&i2[idx2]);
            }
            else {
                r2q = _mm_loadu_pd(&r2[idx2]);
                i2q = _mm_loadu_pd(&i2[idx2]);
            }
            __m128d next_r1q = _mm_sub_pd(_mm_mul_pd(r1q, aq), _mm_mul_pd(i2q, bq));
            __m128d next_i1q = _mm_add_pd(_mm_mul_pd(i1q, aq), _mm_mul_pd(r2q, bq));
            __m128d next_r2q = _mm_sub_pd(_mm_mul_pd(r2q, aq), _mm_mul_pd(i1q, bq));
            __m128d next_i2q = _mm_add_pd(_mm_mul_pd(i2q, aq), _mm_mul_pd(r1q, bq));
            _mm_store_pd(&r1[idx1], next_r1q);
            _mm_store_pd(&i1[idx1], next_i1q);
            if (offset_x == 0) {
                _mm_store_pd(&r2[idx2], next_r2q);
                _mm_store_pd(&i2[idx2], next_i2q);
            }
            else {
                _mm_storeu_pd(&r2[idx2], next_r2q);
                _mm_storeu_pd(&i2[idx2], next_i2q);
            }
        }
        for (; j < width - offset_x; ++j, ++idx1, ++idx2) {
            double next_r1 = a * r1[idx1] - b * i2[idx2];
            double next_i1 = a * i1[idx1] + b * r2[idx2];
            double next_r2 = a * r2[idx2] - b * i1[idx1];
            double next_i2 = a * i2[idx2] + b * r1[idx1];
            r1[idx1] = next_r1;
            i1[idx1] = next_i1;
            r2[idx2] = next_r2;
            i2[idx2] = next_i2;
        }
    }
}

#ifdef WIN32
template <int offset_x>
inline void update_shiftx_sse_imaginary(size_t stride, size_t width, size_t height, double a, double b, double * __restrict r1, double * __restrict i1, double * __restrict r2, double * __restrict i2) {
#else
template <int offset_x>
inline void update_shiftx_sse_imaginary(size_t stride, size_t width, size_t height, double a, double b, double * __restrict__ r1, double * __restrict__ i1, double * __restrict__ r2, double * __restrict__ i2) {
#endif
	__m128d aq, bq;
    aq = _mm_load1_pd(&a);
    bq = _mm_load1_pd(&b);
    for (size_t i = 0; i < height; i++) {
        int idx1 = i * stride;
        int idx2 = i * stride + offset_x;
        size_t j = 0;
        for (; j < width - offset_x - (width - offset_x) % 2; j += 2, idx1 += 2, idx2 += 2) {
            __m128d r1q = _mm_load_pd(&r1[idx1]);
            __m128d i1q = _mm_load_pd(&i1[idx1]);
            __m128d r2q;
            __m128d i2q;
            if (offset_x == 0) {
                r2q = _mm_load_pd(&r2[idx2]);
                i2q = _mm_load_pd(&i2[idx2]);
            }
            else {
                r2q = _mm_loadu_pd(&r2[idx2]);
                i2q = _mm_loadu_pd(&i2[idx2]);
            }
            __m128d next_r1q = _mm_add_pd(_mm_mul_pd(r1q, aq), _mm_mul_pd(r2q, bq));
            __m128d next_i1q = _mm_add_pd(_mm_mul_pd(i1q, aq), _mm_mul_pd(i2q, bq));
            __m128d next_r2q = _mm_add_pd(_mm_mul_pd(r2q, aq), _mm_mul_pd(r1q, bq));
            __m128d next_i2q = _mm_add_pd(_mm_mul_pd(i2q, aq), _mm_mul_pd(i1q, bq));
            _mm_store_pd(&r1[idx1], next_r1q);
            _mm_store_pd(&i1[idx1], next_i1q);
            if (offset_x == 0) {
                _mm_store_pd(&r2[idx2], next_r2q);
                _mm_store_pd(&i2[idx2], next_i2q);
            }
            else {
                _mm_storeu_pd(&r2[idx2], next_r2q);
                _mm_storeu_pd(&i2[idx2], next_i2q);
            }
        }
        for (; j < width - offset_x; ++j, ++idx1, ++idx2) {
            double next_r1 = a * r1[idx1] + b * r2[idx2];
            double next_i1 = a * i1[idx1] + b * i2[idx2];
            double next_r2 = a * r2[idx2] + b * r1[idx1];
            double next_i2 = a * i2[idx2] + b * i1[idx1];
            r1[idx1] = next_r1;
            i1[idx1] = next_i1;
            r2[idx2] = next_r2;
            i2[idx2] = next_i2;
        }
    }
}

#ifdef WIN32
void update_ext_pot_sse(size_t stride, size_t width, size_t height, double * __restrict pot_r, double * __restrict pot_i, double * __restrict real,
                        double * __restrict imag) {
#else
void update_ext_pot_sse(size_t stride, size_t width, size_t height, double * __restrict__ pot_r, double * __restrict__ pot_i, double * __restrict__ real,
	double * __restrict__ imag) {
#endif
    for (size_t i = 0; i < height; i++) {
        size_t j = 0;
        for (; j < width - width % 2; j += 2) {
            size_t idx = i * stride + j;
            __m128d rq = _mm_load_pd(&real[idx]);
            __m128d iq = _mm_load_pd(&imag[idx]);
            __m128d potrq = _mm_load_pd(&pot_r[idx]);
            __m128d potiq = _mm_load_pd(&pot_i[idx]);

            __m128d next_rq = _mm_sub_pd(_mm_mul_pd(rq, potrq), _mm_mul_pd(iq, potiq));
            __m128d next_iq = _mm_add_pd(_mm_mul_pd(iq, potrq), _mm_mul_pd(rq, potiq));

            _mm_store_pd(&real[idx], next_rq);
            _mm_store_pd(&imag[idx], next_iq);
        }

        for (; j < width; j++) {
            size_t idx = i * stride + j;
            double tmp = real[idx];
            real[idx] = pot_r[idx] * tmp - pot_i[idx] * imag[idx];
            imag[idx] = pot_r[idx] * imag[idx] + pot_i[idx] * tmp;
        }
    }
}

#ifdef WIN32
void update_ext_pot_sse_imaginary(size_t stride, size_t width, size_t height, double * __restrict pot_r, double * __restrict pot_i, double * __restrict real,
                                  double * __restrict imag) {
#else
void update_ext_pot_sse_imaginary(size_t stride, size_t width, size_t height, double * __restrict__ pot_r, double * __restrict__ pot_i, double * __restrict__ real,
	double * __restrict__ imag) {
#endif
    for (size_t i = 0; i < height; i++) {
        size_t j = 0;
        for (; j < width - width % 2; j += 2) {
            size_t idx = i * stride + j;
            __m128d rq = _mm_load_pd(&real[idx]);
            __m128d iq = _mm_load_pd(&imag[idx]);
            __m128d potrq = _mm_load_pd(&pot_r[idx]);

            __m128d next_rq = _mm_mul_pd(rq, potrq);
            __m128d next_iq = _mm_mul_pd(iq, potrq);

            _mm_store_pd(&real[idx], next_rq);
            _mm_store_pd(&imag[idx], next_iq);
        }
        for (; j < width; j++) {
            size_t idx = i * stride + j;
            real[idx] = pot_r[idx] * real[idx];
            imag[idx] = pot_r[idx] * imag[idx];
        }
    }
}

void full_step_sse(size_t stride, size_t width, size_t height, double a, double b,
                   double *ext_pot_r00, double *ext_pot_r01, double *ext_pot_r10, double *ext_pot_r11,
                   double *ext_pot_i00, double *ext_pot_i01, double *ext_pot_i10, double *ext_pot_i11,
                   double * r00, double * r01, double * r10, double * r11,
                   double * i00, double * i01, double * i10, double * i11) {
    // 1
    update_shifty_sse<0>(stride, width, height, a, b, r00, i00, r10, i10);
    update_shifty_sse<1>(stride, width, height, a, b, r11, i11, r01, i01);
    // 2
    update_shiftx_sse<0>(stride, width, height, a, b, r00, i00, r01, i01);
    update_shiftx_sse<1>(stride, width, height, a, b, r11, i11, r10, i10);
    // 3
    update_shifty_sse<0>(stride, width, height, a, b, r01, i01, r11, i11);
    update_shifty_sse<1>(stride, width, height, a, b, r10, i10, r00, i00);
    // 4
    update_shiftx_sse<0>(stride, width, height, a, b, r10, i10, r11, i11);
    update_shiftx_sse<1>(stride, width, height, a, b, r01, i01, r00, i00);
    //potential
    update_ext_pot_sse(stride, width, height, ext_pot_r00, ext_pot_i00, r00, i00);
    update_ext_pot_sse(stride, width, height, ext_pot_r10, ext_pot_i10, r10, i10);
    update_ext_pot_sse(stride, width, height, ext_pot_r01, ext_pot_i01, r01, i01);
    update_ext_pot_sse(stride, width, height, ext_pot_r11, ext_pot_i11, r11, i11);
    // 4
    update_shiftx_sse<0>(stride, width, height, a, b, r10, i10, r11, i11);
    update_shiftx_sse<1>(stride, width, height, a, b, r01, i01, r00, i00);
    // 3
    update_shifty_sse<0>(stride, width, height, a, b, r01, i01, r11, i11);
    update_shifty_sse<1>(stride, width, height, a, b, r10, i10, r00, i00);
    // 2
    update_shiftx_sse<0>(stride, width, height, a, b, r00, i00, r01, i01);
    update_shiftx_sse<1>(stride, width, height, a, b, r11, i11, r10, i10);
    // 1
    update_shifty_sse<0>(stride, width, height, a, b, r00, i00, r10, i10);
    update_shifty_sse<1>(stride, width, height, a, b, r11, i11, r01, i01);
}

void full_step_sse_imaginary(size_t stride, size_t width, size_t height, double a, double b,
                             double *ext_pot_r00, double *ext_pot_r01, double *ext_pot_r10, double *ext_pot_r11,
                             double *ext_pot_i00, double *ext_pot_i01, double *ext_pot_i10, double *ext_pot_i11,
                             double * r00, double * r01, double * r10, double * r11,
                             double * i00, double * i01, double * i10, double * i11) {
    // 1
    update_shifty_sse_imaginary<0>(stride, width, height, a, b, r00, i00, r10, i10);
    update_shifty_sse_imaginary<1>(stride, width, height, a, b, r11, i11, r01, i01);
    // 2
    update_shiftx_sse_imaginary<0>(stride, width, height, a, b, r00, i00, r01, i01);
    update_shiftx_sse_imaginary<1>(stride, width, height, a, b, r11, i11, r10, i10);
    // 3
    update_shifty_sse_imaginary<0>(stride, width, height, a, b, r01, i01, r11, i11);
    update_shifty_sse_imaginary<1>(stride, width, height, a, b, r10, i10, r00, i00);
    // 4
    update_shiftx_sse_imaginary<0>(stride, width, height, a, b, r10, i10, r11, i11);
    update_shiftx_sse_imaginary<1>(stride, width, height, a, b, r01, i01, r00, i00);
    //potential
    update_ext_pot_sse_imaginary(stride, width, height, ext_pot_r00, ext_pot_i00, r00, i00);
    update_ext_pot_sse_imaginary(stride, width, height, ext_pot_r10, ext_pot_i10, r10, i10);
    update_ext_pot_sse_imaginary(stride, width, height, ext_pot_r01, ext_pot_i01, r01, i01);
    update_ext_pot_sse_imaginary(stride, width, height, ext_pot_r11, ext_pot_i11, r11, i11);
    // 4
    update_shiftx_sse_imaginary<0>(stride, width, height, a, b, r10, i10, r11, i11);
    update_shiftx_sse_imaginary<1>(stride, width, height, a, b, r01, i01, r00, i00);
    // 3
    update_shifty_sse_imaginary<0>(stride, width, height, a, b, r01, i01, r11, i11);
    update_shifty_sse_imaginary<1>(stride, width, height, a, b, r10, i10, r00, i00);
    // 2
    update_shiftx_sse_imaginary<0>(stride, width, height, a, b, r00, i00, r01, i01);
    update_shiftx_sse_imaginary<1>(stride, width, height, a, b, r11, i11, r10, i10);
    // 1
    update_shifty_sse_imaginary<0>(stride, width, height, a, b, r00, i00, r10, i10);
    update_shifty_sse_imaginary<1>(stride, width, height, a, b, r11, i11, r01, i01);
}

void process_sides_sse( size_t tile_width, size_t block_width, size_t block_height, size_t halo_x, size_t read_y, size_t read_height, size_t write_offset, size_t write_height, double a, double b,
                        const double *ext_pot_r00, const double *ext_pot_r01, const double *ext_pot_r10, const double *ext_pot_r11,
                        const double *ext_pot_i00, const double *ext_pot_i01, const double *ext_pot_i10, const double *ext_pot_i11,
                        double *block_ext_pot_r00, double *block_ext_pot_r01, double *block_ext_pot_r10, double *block_ext_pot_r11,
                        double *block_ext_pot_i00, double *block_ext_pot_i01, double *block_ext_pot_i10, double *block_ext_pot_i11,
                        const double * r00, const double * r01, const double * r10, const double * r11,
                        const double * i00, const double * i01, const double * i10, const double * i11,
                        double * next_r00, double * next_r01, double * next_r10, double * next_r11,
                        double * next_i00, double * next_i01, double * next_i10, double * next_i11,
                        double * block_r00, double * block_r01, double * block_r10, double * block_r11,
                        double * block_i00, double * block_i01, double * block_i10, double * block_i11) {
    size_t read_idx;
    size_t read_width;
    size_t block_read_idx;
    size_t write_idx;
    size_t write_width;

    size_t block_stride = (block_width / 2) * sizeof(double);
    size_t matrix_stride = (tile_width / 2) * sizeof(double);

    // First block [0..block_width - halo_x]
    read_idx = (read_y / 2) * (tile_width / 2);
    read_width = (block_width / 2) * sizeof(double);
    memcpy2D(block_r00, block_stride, &r00[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_i00, block_stride, &i00[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_r01, block_stride, &r01[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_i01, block_stride, &i01[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_r10, block_stride, &r10[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_i10, block_stride, &i10[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_r11, block_stride, &r11[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_i11, block_stride, &i11[read_idx], matrix_stride, read_width, read_height / 2);

    memcpy2D(block_ext_pot_r00, block_stride, &ext_pot_r00[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_ext_pot_i00, block_stride, &ext_pot_i00[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_ext_pot_r01, block_stride, &ext_pot_r01[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_ext_pot_i01, block_stride, &ext_pot_i01[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_ext_pot_r10, block_stride, &ext_pot_r10[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_ext_pot_i10, block_stride, &ext_pot_i10[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_ext_pot_r11, block_stride, &ext_pot_r11[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_ext_pot_i11, block_stride, &ext_pot_i11[read_idx], matrix_stride, read_width, read_height / 2);

    full_step_sse(block_width / 2, block_width / 2, read_height / 2, a, b,
                  block_ext_pot_r00, block_ext_pot_r10, block_ext_pot_r01, block_ext_pot_r11,
                  block_ext_pot_i00, block_ext_pot_i10, block_ext_pot_i01, block_ext_pot_i11,
                  block_r00, block_r01, block_r10, block_r11,
                  block_i00, block_i01, block_i10, block_i11);

    block_read_idx = (write_offset / 2) * (block_width / 2);
    write_idx = (read_y / 2 + write_offset / 2) * (tile_width / 2);
    write_width = ((block_width - halo_x) / 2) * sizeof(double);
    memcpy2D(&next_r00[write_idx], matrix_stride, &block_r00[block_read_idx], block_stride, write_width, write_height / 2);
    memcpy2D(&next_i00[write_idx], matrix_stride, &block_i00[block_read_idx], block_stride, write_width, write_height / 2);
    memcpy2D(&next_r01[write_idx], matrix_stride, &block_r01[block_read_idx], block_stride, write_width, write_height / 2);
    memcpy2D(&next_i01[write_idx], matrix_stride, &block_i01[block_read_idx], block_stride, write_width, write_height / 2);
    memcpy2D(&next_r10[write_idx], matrix_stride, &block_r10[block_read_idx], block_stride, write_width, write_height / 2);
    memcpy2D(&next_i10[write_idx], matrix_stride, &block_i10[block_read_idx], block_stride, write_width, write_height / 2);
    memcpy2D(&next_r11[write_idx], matrix_stride, &block_r11[block_read_idx], block_stride, write_width, write_height / 2);
    memcpy2D(&next_i11[write_idx], matrix_stride, &block_i11[block_read_idx], block_stride, write_width, write_height / 2);

    // Last block
    size_t block_start = ((tile_width - block_width) / (block_width - 2 * halo_x) + 1) * (block_width - 2 * halo_x);
    read_idx = (read_y / 2) * (tile_width / 2) + block_start / 2;
    read_width = (tile_width / 2 - block_start / 2) * sizeof(double);
    memcpy2D(block_r00, block_stride, &r00[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_i00, block_stride, &i00[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_r01, block_stride, &r01[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_i01, block_stride, &i01[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_r10, block_stride, &r10[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_i10, block_stride, &i10[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_r11, block_stride, &r11[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_i11, block_stride, &i11[read_idx], matrix_stride, read_width, read_height / 2);

    full_step_sse(block_width / 2, tile_width / 2 - block_start / 2, read_height / 2, a, b,
                  block_ext_pot_r00, block_ext_pot_r10, block_ext_pot_r01, block_ext_pot_r11,
                  block_ext_pot_i00, block_ext_pot_i10, block_ext_pot_i01, block_ext_pot_i11,
                  block_r00, block_r01, block_r10, block_r11,
                  block_i00, block_i01, block_i10, block_i11);

    block_read_idx = (write_offset / 2) * (block_width / 2) + halo_x / 2;
    write_idx = (read_y / 2 + write_offset / 2) * (tile_width / 2) + (block_start + halo_x) / 2;
    write_width = (tile_width / 2 - block_start / 2 - halo_x / 2) * sizeof(double);
    memcpy2D(&next_r00[write_idx], matrix_stride, &block_r00[block_read_idx], block_stride, write_width, write_height / 2);
    memcpy2D(&next_i00[write_idx], matrix_stride, &block_i00[block_read_idx], block_stride, write_width, write_height / 2);
    memcpy2D(&next_r01[write_idx], matrix_stride, &block_r01[block_read_idx], block_stride, write_width, write_height / 2);
    memcpy2D(&next_i01[write_idx], matrix_stride, &block_i01[block_read_idx], block_stride, write_width, write_height / 2);
    memcpy2D(&next_r10[write_idx], matrix_stride, &block_r10[block_read_idx], block_stride, write_width, write_height / 2);
    memcpy2D(&next_i10[write_idx], matrix_stride, &block_i10[block_read_idx], block_stride, write_width, write_height / 2);
    memcpy2D(&next_r11[write_idx], matrix_stride, &block_r11[block_read_idx], block_stride, write_width, write_height / 2);
    memcpy2D(&next_i11[write_idx], matrix_stride, &block_i11[block_read_idx], block_stride, write_width, write_height / 2);
}

void process_sides_sse_imaginary( size_t tile_width, size_t block_width, size_t block_height, size_t halo_x, size_t read_y, size_t read_height, size_t write_offset, size_t write_height, double a, double b,
                                  const double *ext_pot_r00, const double *ext_pot_r01, const double *ext_pot_r10, const double *ext_pot_r11,
                                  const double *ext_pot_i00, const double *ext_pot_i01, const double *ext_pot_i10, const double *ext_pot_i11,
                                  double *block_ext_pot_r00, double *block_ext_pot_r01, double *block_ext_pot_r10, double *block_ext_pot_r11,
                                  double *block_ext_pot_i00, double *block_ext_pot_i01, double *block_ext_pot_i10, double *block_ext_pot_i11,
                                  const double * r00, const double * r01, const double * r10, const double * r11,
                                  const double * i00, const double * i01, const double * i10, const double * i11,
                                  double * next_r00, double * next_r01, double * next_r10, double * next_r11,
                                  double * next_i00, double * next_i01, double * next_i10, double * next_i11,
                                  double * block_r00, double * block_r01, double * block_r10, double * block_r11,
                                  double * block_i00, double * block_i01, double * block_i10, double * block_i11) {
    size_t read_idx;
    size_t read_width;
    size_t block_read_idx;
    size_t write_idx;
    size_t write_width;

    size_t block_stride = (block_width / 2) * sizeof(double);
    size_t matrix_stride = (tile_width / 2) * sizeof(double);

    // First block [0..block_width - halo_x]
    read_idx = (read_y / 2) * (tile_width / 2);
    read_width = (block_width / 2) * sizeof(double);
    memcpy2D(block_r00, block_stride, &r00[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_i00, block_stride, &i00[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_r01, block_stride, &r01[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_i01, block_stride, &i01[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_r10, block_stride, &r10[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_i10, block_stride, &i10[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_r11, block_stride, &r11[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_i11, block_stride, &i11[read_idx], matrix_stride, read_width, read_height / 2);

    memcpy2D(block_ext_pot_r00, block_stride, &ext_pot_r00[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_ext_pot_i00, block_stride, &ext_pot_i00[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_ext_pot_r01, block_stride, &ext_pot_r01[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_ext_pot_i01, block_stride, &ext_pot_i01[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_ext_pot_r10, block_stride, &ext_pot_r10[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_ext_pot_i10, block_stride, &ext_pot_i10[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_ext_pot_r11, block_stride, &ext_pot_r11[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_ext_pot_i11, block_stride, &ext_pot_i11[read_idx], matrix_stride, read_width, read_height / 2);

    full_step_sse_imaginary(block_width / 2, block_width / 2, read_height / 2, a, b,
                            block_ext_pot_r00, block_ext_pot_r10, block_ext_pot_r01, block_ext_pot_r11,
                            block_ext_pot_i00, block_ext_pot_i10, block_ext_pot_i01, block_ext_pot_i11,
                            block_r00, block_r01, block_r10, block_r11,
                            block_i00, block_i01, block_i10, block_i11);

    block_read_idx = (write_offset / 2) * (block_width / 2);
    write_idx = (read_y / 2 + write_offset / 2) * (tile_width / 2);
    write_width = ((block_width - halo_x) / 2) * sizeof(double);
    memcpy2D(&next_r00[write_idx], matrix_stride, &block_r00[block_read_idx], block_stride, write_width, write_height / 2);
    memcpy2D(&next_i00[write_idx], matrix_stride, &block_i00[block_read_idx], block_stride, write_width, write_height / 2);
    memcpy2D(&next_r01[write_idx], matrix_stride, &block_r01[block_read_idx], block_stride, write_width, write_height / 2);
    memcpy2D(&next_i01[write_idx], matrix_stride, &block_i01[block_read_idx], block_stride, write_width, write_height / 2);
    memcpy2D(&next_r10[write_idx], matrix_stride, &block_r10[block_read_idx], block_stride, write_width, write_height / 2);
    memcpy2D(&next_i10[write_idx], matrix_stride, &block_i10[block_read_idx], block_stride, write_width, write_height / 2);
    memcpy2D(&next_r11[write_idx], matrix_stride, &block_r11[block_read_idx], block_stride, write_width, write_height / 2);
    memcpy2D(&next_i11[write_idx], matrix_stride, &block_i11[block_read_idx], block_stride, write_width, write_height / 2);

    // Last block
    size_t block_start = ((tile_width - block_width) / (block_width - 2 * halo_x) + 1) * (block_width - 2 * halo_x);
    read_idx = (read_y / 2) * (tile_width / 2) + block_start / 2;
    read_width = (tile_width / 2 - block_start / 2) * sizeof(double);
    memcpy2D(block_r00, block_stride, &r00[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_i00, block_stride, &i00[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_r01, block_stride, &r01[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_i01, block_stride, &i01[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_r10, block_stride, &r10[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_i10, block_stride, &i10[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_r11, block_stride, &r11[read_idx], matrix_stride, read_width, read_height / 2);
    memcpy2D(block_i11, block_stride, &i11[read_idx], matrix_stride, read_width, read_height / 2);

    full_step_sse_imaginary(block_width / 2, tile_width / 2 - block_start / 2, read_height / 2, a, b,
                            block_ext_pot_r00, block_ext_pot_r10, block_ext_pot_r01, block_ext_pot_r11,
                            block_ext_pot_i00, block_ext_pot_i10, block_ext_pot_i01, block_ext_pot_i11,
                            block_r00, block_r01, block_r10, block_r11,
                            block_i00, block_i01, block_i10, block_i11);

    block_read_idx = (write_offset / 2) * (block_width / 2) + halo_x / 2;
    write_idx = (read_y / 2 + write_offset / 2) * (tile_width / 2) + (block_start + halo_x) / 2;
    write_width = (tile_width / 2 - block_start / 2 - halo_x / 2) * sizeof(double);
    memcpy2D(&next_r00[write_idx], matrix_stride, &block_r00[block_read_idx], block_stride, write_width, write_height / 2);
    memcpy2D(&next_i00[write_idx], matrix_stride, &block_i00[block_read_idx], block_stride, write_width, write_height / 2);
    memcpy2D(&next_r01[write_idx], matrix_stride, &block_r01[block_read_idx], block_stride, write_width, write_height / 2);
    memcpy2D(&next_i01[write_idx], matrix_stride, &block_i01[block_read_idx], block_stride, write_width, write_height / 2);
    memcpy2D(&next_r10[write_idx], matrix_stride, &block_r10[block_read_idx], block_stride, write_width, write_height / 2);
    memcpy2D(&next_i10[write_idx], matrix_stride, &block_i10[block_read_idx], block_stride, write_width, write_height / 2);
    memcpy2D(&next_r11[write_idx], matrix_stride, &block_r11[block_read_idx], block_stride, write_width, write_height / 2);
    memcpy2D(&next_i11[write_idx], matrix_stride, &block_i11[block_read_idx], block_stride, write_width, write_height / 2);
}

void process_band_sse(size_t tile_width, size_t block_width, size_t block_height, size_t halo_x, size_t read_y, size_t read_height, size_t write_offset, size_t write_height, double a, double b,
                      const double *ext_pot_r00, const double *ext_pot_r01, const double *ext_pot_r10, const double *ext_pot_r11,
                      const double *ext_pot_i00, const double *ext_pot_i01, const double *ext_pot_i10, const double *ext_pot_i11,
                      const double * r00, const double * r01, const double * r10, const double * r11,
                      const double * i00, const double * i01, const double * i10, const double * i11,
                      double * next_r00, double * next_r01, double * next_r10, double * next_r11,
                      double * next_i00, double * next_i01, double * next_i10, double * next_i11, int inner, int sides, bool imag_time) {
    double *block_r00 = new double[(block_height / 2) * (block_width / 2)];
	double *block_r01 = new double[(block_height / 2) * (block_width / 2)];
	double *block_r10 = new double[(block_height / 2) * (block_width / 2)];
	double *block_r11 = new double[(block_height / 2) * (block_width / 2)];
	double *block_i00 = new double[(block_height / 2) * (block_width / 2)];
	double *block_i01 = new double[(block_height / 2) * (block_width / 2)];
	double *block_i10 = new double[(block_height / 2) * (block_width / 2)];
	double *block_i11 = new double[(block_height / 2) * (block_width / 2)];

	double *block_ext_pot_r00 = new double[(block_height / 2) * (block_width / 2)];
	double *block_ext_pot_r01 = new double[(block_height / 2) * (block_width / 2)];
	double *block_ext_pot_r10 = new double[(block_height / 2) * (block_width / 2)];
	double *block_ext_pot_r11 = new double[(block_height / 2) * (block_width / 2)];
	double *block_ext_pot_i00 = new double[(block_height / 2) * (block_width / 2)];
	double *block_ext_pot_i01 = new double[(block_height / 2) * (block_width / 2)];
	double *block_ext_pot_i10 = new double[(block_height / 2) * (block_width / 2)];
	double *block_ext_pot_i11 = new double[(block_height / 2) * (block_width / 2)];

    size_t read_idx;
    size_t read_width;
    size_t block_read_idx;
    size_t write_idx;
    size_t write_width;

    size_t block_stride = (block_width / 2) * sizeof(double);
    size_t matrix_stride = (tile_width / 2) * sizeof(double);

    if (tile_width <= block_width) {
        if (sides) {
            // One full block
            read_idx = (read_y / 2) * (tile_width / 2);
            read_width = (tile_width / 2) * sizeof(double);
            memcpy2D(block_r00, block_stride, &r00[read_idx], matrix_stride, read_width, read_height / 2);
            memcpy2D(block_i00, block_stride, &i00[read_idx], matrix_stride, read_width, read_height / 2);
            memcpy2D(block_r01, block_stride, &r01[read_idx], matrix_stride, read_width, read_height / 2);
            memcpy2D(block_i01, block_stride, &i01[read_idx], matrix_stride, read_width, read_height / 2);
            memcpy2D(block_r10, block_stride, &r10[read_idx], matrix_stride, read_width, read_height / 2);
            memcpy2D(block_i10, block_stride, &i10[read_idx], matrix_stride, read_width, read_height / 2);
            memcpy2D(block_r11, block_stride, &r11[read_idx], matrix_stride, read_width, read_height / 2);
            memcpy2D(block_i11, block_stride, &i11[read_idx], matrix_stride, read_width, read_height / 2);

            memcpy2D(block_ext_pot_r00, block_stride, &ext_pot_r00[read_idx], matrix_stride, read_width, read_height / 2);
            memcpy2D(block_ext_pot_i00, block_stride, &ext_pot_i00[read_idx], matrix_stride, read_width, read_height / 2);
            memcpy2D(block_ext_pot_r01, block_stride, &ext_pot_r01[read_idx], matrix_stride, read_width, read_height / 2);
            memcpy2D(block_ext_pot_i01, block_stride, &ext_pot_i01[read_idx], matrix_stride, read_width, read_height / 2);
            memcpy2D(block_ext_pot_r10, block_stride, &ext_pot_r10[read_idx], matrix_stride, read_width, read_height / 2);
            memcpy2D(block_ext_pot_i10, block_stride, &ext_pot_i10[read_idx], matrix_stride, read_width, read_height / 2);
            memcpy2D(block_ext_pot_r11, block_stride, &ext_pot_r11[read_idx], matrix_stride, read_width, read_height / 2);
            memcpy2D(block_ext_pot_i11, block_stride, &ext_pot_i11[read_idx], matrix_stride, read_width, read_height / 2);

            if(imag_time) {
                full_step_sse_imaginary(block_width / 2, tile_width / 2, read_height / 2, a, b,
                                        block_ext_pot_r00, block_ext_pot_r10, block_ext_pot_r01, block_ext_pot_r11,
                                        block_ext_pot_i00, block_ext_pot_i10, block_ext_pot_i01, block_ext_pot_i11,
                                        block_r00, block_r01, block_r10, block_r11,
                                        block_i00, block_i01, block_i10, block_i11);
            }
            else {
                full_step_sse(block_width / 2, tile_width / 2, read_height / 2, a, b,
                              block_ext_pot_r00, block_ext_pot_r10, block_ext_pot_r01, block_ext_pot_r11,
                              block_ext_pot_i00, block_ext_pot_i10, block_ext_pot_i01, block_ext_pot_i11,
                              block_r00, block_r01, block_r10, block_r11,
                              block_i00, block_i01, block_i10, block_i11);
            }

            block_read_idx = (write_offset / 2) * (block_width / 2);
            write_idx = (read_y / 2 + write_offset / 2) * (tile_width / 2);
            write_width = read_width;
            memcpy2D(&next_r00[write_idx], matrix_stride, &block_r00[block_read_idx], block_stride, write_width, write_height / 2);
            memcpy2D(&next_i00[write_idx], matrix_stride, &block_i00[block_read_idx], block_stride, write_width, write_height / 2);
            memcpy2D(&next_r01[write_idx], matrix_stride, &block_r01[block_read_idx], block_stride, write_width, write_height / 2);
            memcpy2D(&next_i01[write_idx], matrix_stride, &block_i01[block_read_idx], block_stride, write_width, write_height / 2);
            memcpy2D(&next_r10[write_idx], matrix_stride, &block_r10[block_read_idx], block_stride, write_width, write_height / 2);
            memcpy2D(&next_i10[write_idx], matrix_stride, &block_i10[block_read_idx], block_stride, write_width, write_height / 2);
            memcpy2D(&next_r11[write_idx], matrix_stride, &block_r11[block_read_idx], block_stride, write_width, write_height / 2);
            memcpy2D(&next_i11[write_idx], matrix_stride, &block_i11[block_read_idx], block_stride, write_width, write_height / 2);
        }
    }
    else {
        if (sides) {
            if(imag_time) {
                process_sides_sse_imaginary(tile_width, block_width, block_height, halo_x, read_y, read_height, write_offset, write_height, a, b,
                                            ext_pot_r00, ext_pot_r10, ext_pot_r01, ext_pot_r11,
                                            ext_pot_i00, ext_pot_i10, ext_pot_i01, ext_pot_i11,
                                            block_ext_pot_r00, block_ext_pot_r10, block_ext_pot_r01, block_ext_pot_r11,
                                            block_ext_pot_i00, block_ext_pot_i10, block_ext_pot_i01, block_ext_pot_i11,
                                            r00, r01, r10, r11,
                                            i00, i01, i10, i11,
                                            next_r00, next_r01, next_r10, next_r11,
                                            next_i00, next_i01, next_i10, next_i11,
                                            block_r00, block_r01, block_r10, block_r11,
                                            block_i00, block_i01, block_i10, block_i11);
            }
            else {
                process_sides_sse(tile_width, block_width, block_height, halo_x, read_y, read_height, write_offset, write_height, a, b,
                                  ext_pot_r00, ext_pot_r10, ext_pot_r01, ext_pot_r11,
                                  ext_pot_i00, ext_pot_i10, ext_pot_i01, ext_pot_i11,
                                  block_ext_pot_r00, block_ext_pot_r10, block_ext_pot_r01, block_ext_pot_r11,
                                  block_ext_pot_i00, block_ext_pot_i10, block_ext_pot_i01, block_ext_pot_i11,
                                  r00, r01, r10, r11,
                                  i00, i01, i10, i11,
                                  next_r00, next_r01, next_r10, next_r11,
                                  next_i00, next_i01, next_i10, next_i11,
                                  block_r00, block_r01, block_r10, block_r11,
                                  block_i00, block_i01, block_i10, block_i11);
            }
        }
        if (inner) {
            // Regular blocks in the middle
            size_t block_start;
            read_width = (block_width / 2) * sizeof(double);
            block_read_idx = (write_offset / 2) * (block_width / 2) + halo_x / 2;
            write_width = ((block_width - 2 * halo_x) / 2) * sizeof(double);
            for (block_start = block_width - 2 * halo_x; block_start < tile_width - block_width; block_start += block_width - 2 * halo_x) {
                read_idx = (read_y / 2) * (tile_width / 2) + block_start / 2;
                memcpy2D(block_r00, block_stride, &r00[read_idx], matrix_stride, read_width, read_height / 2);
                memcpy2D(block_i00, block_stride, &i00[read_idx], matrix_stride, read_width, read_height / 2);
                memcpy2D(block_r01, block_stride, &r01[read_idx], matrix_stride, read_width, read_height / 2);
                memcpy2D(block_i01, block_stride, &i01[read_idx], matrix_stride, read_width, read_height / 2);
                memcpy2D(block_r10, block_stride, &r10[read_idx], matrix_stride, read_width, read_height / 2);
                memcpy2D(block_i10, block_stride, &i10[read_idx], matrix_stride, read_width, read_height / 2);
                memcpy2D(block_r11, block_stride, &r11[read_idx], matrix_stride, read_width, read_height / 2);
                memcpy2D(block_i11, block_stride, &i11[read_idx], matrix_stride, read_width, read_height / 2);

                memcpy2D(block_ext_pot_r00, block_stride, &ext_pot_r00[read_idx], matrix_stride, read_width, read_height / 2);
                memcpy2D(block_ext_pot_i00, block_stride, &ext_pot_i00[read_idx], matrix_stride, read_width, read_height / 2);
                memcpy2D(block_ext_pot_r01, block_stride, &ext_pot_r01[read_idx], matrix_stride, read_width, read_height / 2);
                memcpy2D(block_ext_pot_i01, block_stride, &ext_pot_i01[read_idx], matrix_stride, read_width, read_height / 2);
                memcpy2D(block_ext_pot_r10, block_stride, &ext_pot_r10[read_idx], matrix_stride, read_width, read_height / 2);
                memcpy2D(block_ext_pot_i10, block_stride, &ext_pot_i10[read_idx], matrix_stride, read_width, read_height / 2);
                memcpy2D(block_ext_pot_r11, block_stride, &ext_pot_r11[read_idx], matrix_stride, read_width, read_height / 2);
                memcpy2D(block_ext_pot_i11, block_stride, &ext_pot_i11[read_idx], matrix_stride, read_width, read_height / 2);

                if(imag_time) {
                    full_step_sse_imaginary(block_width / 2, block_width / 2, read_height / 2, a, b,
                                            block_ext_pot_r00, block_ext_pot_r10, block_ext_pot_r01, block_ext_pot_r11,
                                            block_ext_pot_i00, block_ext_pot_i10, block_ext_pot_i01, block_ext_pot_i11,
                                            block_r00, block_r01, block_r10, block_r11,
                                            block_i00, block_i01, block_i10, block_i11);
                }
                else {
                    full_step_sse(block_width / 2, block_width / 2, read_height / 2, a, b,
                                  block_ext_pot_r00, block_ext_pot_r10, block_ext_pot_r01, block_ext_pot_r11,
                                  block_ext_pot_i00, block_ext_pot_i10, block_ext_pot_i01, block_ext_pot_i11,
                                  block_r00, block_r01, block_r10, block_r11,
                                  block_i00, block_i01, block_i10, block_i11);
                }

                write_idx = (read_y / 2 + write_offset / 2) * (tile_width / 2) + (block_start + halo_x) / 2;
                memcpy2D(&next_r00[write_idx], matrix_stride, &block_r00[block_read_idx], block_stride, write_width, write_height / 2);
                memcpy2D(&next_i00[write_idx], matrix_stride, &block_i00[block_read_idx], block_stride, write_width, write_height / 2);
                memcpy2D(&next_r01[write_idx], matrix_stride, &block_r01[block_read_idx], block_stride, write_width, write_height / 2);
                memcpy2D(&next_i01[write_idx], matrix_stride, &block_i01[block_read_idx], block_stride, write_width, write_height / 2);
                memcpy2D(&next_r10[write_idx], matrix_stride, &block_r10[block_read_idx], block_stride, write_width, write_height / 2);
                memcpy2D(&next_i10[write_idx], matrix_stride, &block_i10[block_read_idx], block_stride, write_width, write_height / 2);
                memcpy2D(&next_r11[write_idx], matrix_stride, &block_r11[block_read_idx], block_stride, write_width, write_height / 2);
                memcpy2D(&next_i11[write_idx], matrix_stride, &block_i11[block_read_idx], block_stride, write_width, write_height / 2);
            }
        }
    }
	delete[] block_r00;
	delete[] block_r01;
	delete[] block_r10;
	delete[] block_r11;
	delete[] block_i00;
	delete[] block_i01;
	delete[] block_i10;
	delete[] block_i11;

	delete[] block_ext_pot_r00;
	delete[] block_ext_pot_r01;
	delete[] block_ext_pot_r10;
	delete[] block_ext_pot_r11;
	delete[] block_ext_pot_i00;
	delete[] block_ext_pot_i01;
	delete[] block_ext_pot_i10;
	delete[] block_ext_pot_i11;
}

CPUBlockSSEKernel::CPUBlockSSEKernel(double *_p_real, double *_p_imag, double *external_potential_real, double *external_potential_imag,
                                     double _a, double _b, int matrix_width, int matrix_height, int _halo_x, int _halo_y, int *_periods, bool _imag_time
#ifdef HAVE_MPI
                                     , MPI_Comm _cartcomm
#endif
                                     ):
    p_real(_p_real),
    p_imag(_p_imag),
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

    assert (tile_width % 2 == 0);
    assert (tile_height % 2 == 0);
	/*
    posix_memalign(reinterpret_cast<void**>(&r00[0]), 64, ((tile_width * tile_height) / 4) * sizeof(double));
    posix_memalign(reinterpret_cast<void**>(&r00[1]), 64, ((tile_width * tile_height) / 4) * sizeof(double));
    posix_memalign(reinterpret_cast<void**>(&r01[0]), 64, ((tile_width * tile_height) / 4) * sizeof(double));
    posix_memalign(reinterpret_cast<void**>(&r01[1]), 64, ((tile_width * tile_height) / 4) * sizeof(double));
    posix_memalign(reinterpret_cast<void**>(&r10[0]), 64, ((tile_width * tile_height) / 4) * sizeof(double));
    posix_memalign(reinterpret_cast<void**>(&r10[1]), 64, ((tile_width * tile_height) / 4) * sizeof(double));
    posix_memalign(reinterpret_cast<void**>(&r11[0]), 64, ((tile_width * tile_height) / 4) * sizeof(double));
    posix_memalign(reinterpret_cast<void**>(&r11[1]), 64, ((tile_width * tile_height) / 4) * sizeof(double));
    posix_memalign(reinterpret_cast<void**>(&i00[0]), 64, ((tile_width * tile_height) / 4) * sizeof(double));
    posix_memalign(reinterpret_cast<void**>(&i00[1]), 64, ((tile_width * tile_height) / 4) * sizeof(double));
    posix_memalign(reinterpret_cast<void**>(&i01[0]), 64, ((tile_width * tile_height) / 4) * sizeof(double));
    posix_memalign(reinterpret_cast<void**>(&i01[1]), 64, ((tile_width * tile_height) / 4) * sizeof(double));
    posix_memalign(reinterpret_cast<void**>(&i10[0]), 64, ((tile_width * tile_height) / 4) * sizeof(double));
    posix_memalign(reinterpret_cast<void**>(&i10[1]), 64, ((tile_width * tile_height) / 4) * sizeof(double));
    posix_memalign(reinterpret_cast<void**>(&i11[0]), 64, ((tile_width * tile_height) / 4) * sizeof(double));
    posix_memalign(reinterpret_cast<void**>(&i11[1]), 64, ((tile_width * tile_height) / 4) * sizeof(double));

    posix_memalign(reinterpret_cast<void**>(&ext_pot_r00), 64, ((tile_width * tile_height) / 4) * sizeof(double));
    posix_memalign(reinterpret_cast<void**>(&ext_pot_r01), 64, ((tile_width * tile_height) / 4) * sizeof(double));
    posix_memalign(reinterpret_cast<void**>(&ext_pot_r10), 64, ((tile_width * tile_height) / 4) * sizeof(double));
    posix_memalign(reinterpret_cast<void**>(&ext_pot_r11), 64, ((tile_width * tile_height) / 4) * sizeof(double));
    posix_memalign(reinterpret_cast<void**>(&ext_pot_i00), 64, ((tile_width * tile_height) / 4) * sizeof(double));
    posix_memalign(reinterpret_cast<void**>(&ext_pot_i01), 64, ((tile_width * tile_height) / 4) * sizeof(double));
    posix_memalign(reinterpret_cast<void**>(&ext_pot_i10), 64, ((tile_width * tile_height) / 4) * sizeof(double));
    posix_memalign(reinterpret_cast<void**>(&ext_pot_i11), 64, ((tile_width * tile_height) / 4) * sizeof(double));
	*/
	
#ifdef WIN32
	r00[0] = reinterpret_cast<double*>(_aligned_malloc(((tile_width * tile_height) / 4) * sizeof(double), 64));
	r00[1] = reinterpret_cast<double*>(_aligned_malloc(((tile_width * tile_height) / 4) * sizeof(double), 64));
	r01[0] = reinterpret_cast<double*>(_aligned_malloc(((tile_width * tile_height) / 4) * sizeof(double), 64));
	r01[1] = reinterpret_cast<double*>(_aligned_malloc(((tile_width * tile_height) / 4) * sizeof(double), 64));
	r10[0] = reinterpret_cast<double*>(_aligned_malloc(((tile_width * tile_height) / 4) * sizeof(double), 64));
	r10[1] = reinterpret_cast<double*>(_aligned_malloc(((tile_width * tile_height) / 4) * sizeof(double), 64));
	r11[0] = reinterpret_cast<double*>(_aligned_malloc(((tile_width * tile_height) / 4) * sizeof(double), 64));
	r11[1] = reinterpret_cast<double*>(_aligned_malloc(((tile_width * tile_height) / 4) * sizeof(double), 64));
	i00[0] = reinterpret_cast<double*>(_aligned_malloc(((tile_width * tile_height) / 4) * sizeof(double), 64));
	i00[1] = reinterpret_cast<double*>(_aligned_malloc(((tile_width * tile_height) / 4) * sizeof(double), 64));
	i01[0] = reinterpret_cast<double*>(_aligned_malloc(((tile_width * tile_height) / 4) * sizeof(double), 64));
	i01[1] = reinterpret_cast<double*>(_aligned_malloc(((tile_width * tile_height) / 4) * sizeof(double), 64));
	i10[0] = reinterpret_cast<double*>(_aligned_malloc(((tile_width * tile_height) / 4) * sizeof(double), 64));
	i10[1] = reinterpret_cast<double*>(_aligned_malloc(((tile_width * tile_height) / 4) * sizeof(double), 64));
	i11[0] = reinterpret_cast<double*>(_aligned_malloc(((tile_width * tile_height) / 4) * sizeof(double), 64));
	i11[1] = reinterpret_cast<double*>(_aligned_malloc(((tile_width * tile_height) / 4) * sizeof(double), 64));
	ext_pot_r00 = reinterpret_cast<double*>(_aligned_malloc(((tile_width * tile_height) / 4) * sizeof(double), 64));
	ext_pot_r01 = reinterpret_cast<double*>(_aligned_malloc(((tile_width * tile_height) / 4) * sizeof(double), 64));
	ext_pot_r10 = reinterpret_cast<double*>(_aligned_malloc(((tile_width * tile_height) / 4) * sizeof(double), 64));
	ext_pot_r11 = reinterpret_cast<double*>(_aligned_malloc(((tile_width * tile_height) / 4) * sizeof(double), 64));
	ext_pot_i00 = reinterpret_cast<double*>(_aligned_malloc(((tile_width * tile_height) / 4) * sizeof(double), 64));
	ext_pot_i01 = reinterpret_cast<double*>(_aligned_malloc(((tile_width * tile_height) / 4) * sizeof(double), 64));
	ext_pot_i10 = reinterpret_cast<double*>(_aligned_malloc(((tile_width * tile_height) / 4) * sizeof(double), 64));
	ext_pot_i11 = reinterpret_cast<double*>(_aligned_malloc(((tile_width * tile_height) / 4) * sizeof(double), 64));
#else
	r00[0] = reinterpret_cast<double*>(aligned_alloc(64, ((tile_width * tile_height) / 4) * sizeof(double)));
	r00[1] = reinterpret_cast<double*>(aligned_alloc(64, ((tile_width * tile_height) / 4) * sizeof(double)));
	r01[0] = reinterpret_cast<double*>(aligned_alloc(64, ((tile_width * tile_height) / 4) * sizeof(double)));
	r01[1] = reinterpret_cast<double*>(aligned_alloc(64, ((tile_width * tile_height) / 4) * sizeof(double)));
	r10[0] = reinterpret_cast<double*>(aligned_alloc(64, ((tile_width * tile_height) / 4) * sizeof(double)));
	r10[1] = reinterpret_cast<double*>(aligned_alloc(64, ((tile_width * tile_height) / 4) * sizeof(double)));
	r11[0] = reinterpret_cast<double*>(aligned_alloc(64, ((tile_width * tile_height) / 4) * sizeof(double)));
	r11[1] = reinterpret_cast<double*>(aligned_alloc(64, ((tile_width * tile_height) / 4) * sizeof(double)));
	i00[0] = reinterpret_cast<double*>(aligned_alloc(64, ((tile_width * tile_height) / 4) * sizeof(double)));
	i00[1] = reinterpret_cast<double*>(aligned_alloc(64, ((tile_width * tile_height) / 4) * sizeof(double)));
	i01[0] = reinterpret_cast<double*>(aligned_alloc(64, ((tile_width * tile_height) / 4) * sizeof(double)));
	i01[1] = reinterpret_cast<double*>(aligned_alloc(64, ((tile_width * tile_height) / 4) * sizeof(double)));
	i10[0] = reinterpret_cast<double*>(aligned_alloc(64, ((tile_width * tile_height) / 4) * sizeof(double)));
	i10[1] = reinterpret_cast<double*>(aligned_alloc(64, ((tile_width * tile_height) / 4) * sizeof(double)));
	i11[0] = reinterpret_cast<double*>(aligned_alloc(64, ((tile_width * tile_height) / 4) * sizeof(double)));
	i11[1] = reinterpret_cast<double*>(aligned_alloc(64, ((tile_width * tile_height) / 4) * sizeof(double)));
	ext_pot_r00 = reinterpret_cast<double*>(aligned_alloc(64, ((tile_width * tile_height) / 4) * sizeof(double)));
	ext_pot_r01 = reinterpret_cast<double*>(aligned_alloc(64, ((tile_width * tile_height) / 4) * sizeof(double)));
	ext_pot_r10 = reinterpret_cast<double*>(aligned_alloc(64, ((tile_width * tile_height) / 4) * sizeof(double)));
	ext_pot_r11 = reinterpret_cast<double*>(aligned_alloc(64, ((tile_width * tile_height) / 4) * sizeof(double)));
	ext_pot_i00 = reinterpret_cast<double*>(aligned_alloc(64, ((tile_width * tile_height) / 4) * sizeof(double)));
	ext_pot_i01 = reinterpret_cast<double*>(aligned_alloc(64, ((tile_width * tile_height) / 4) * sizeof(double)));
	ext_pot_i10 = reinterpret_cast<double*>(aligned_alloc(64, ((tile_width * tile_height) / 4) * sizeof(double)));
	ext_pot_i11 = reinterpret_cast<double*>(aligned_alloc(64, ((tile_width * tile_height) / 4) * sizeof(double)));
#endif

    for (size_t i = 0; i < tile_height / 2; i++) {
        for (size_t j = 0; j < tile_width / 2; j++) {
            r00[0][i * tile_width / 2 + j] = p_real[2 * i * tile_width + 2 * j];
            i00[0][i * tile_width / 2 + j] = p_imag[2 * i * tile_width + 2 * j];
            r01[0][i * tile_width / 2 + j] = p_real[2 * i * tile_width + 2 * j + 1];
            i01[0][i * tile_width / 2 + j] = p_imag[2 * i * tile_width + 2 * j + 1];

            ext_pot_r00[i * tile_width / 2 + j] = external_potential_real[2 * i * tile_width + 2 * j];
            ext_pot_i00[i * tile_width / 2 + j] = external_potential_imag[2 * i * tile_width + 2 * j];
            ext_pot_r01[i * tile_width / 2 + j] = external_potential_real[2 * i * tile_width + 2 * j + 1];
            ext_pot_i01[i * tile_width / 2 + j] = external_potential_imag[2 * i * tile_width + 2 * j + 1];
        }
        for (size_t j = 0; j < tile_width / 2; j++) {
            r10[0][i * tile_width / 2 + j] = p_real[(2 * i + 1) * tile_width + 2 * j];
            i10[0][i * tile_width / 2 + j] = p_imag[(2 * i + 1) * tile_width + 2 * j];
            r11[0][i * tile_width / 2 + j] = p_real[(2 * i + 1) * tile_width + 2 * j + 1];
            i11[0][i * tile_width / 2 + j] = p_imag[(2 * i + 1) * tile_width + 2 * j + 1];

            ext_pot_r10[i * tile_width / 2 + j] = external_potential_real[(2 * i + 1) * tile_width + 2 * j];
            ext_pot_i10[i * tile_width / 2 + j] = external_potential_imag[(2 * i + 1) * tile_width + 2 * j];
            ext_pot_r11[i * tile_width / 2 + j] = external_potential_real[(2 * i + 1) * tile_width + 2 * j + 1];
            ext_pot_i11[i * tile_width / 2 + j] = external_potential_imag[(2 * i + 1) * tile_width + 2 * j + 1];
        }
    }
#ifdef HAVE_MPI
    // Halo exchange uses wave pattern to communicate
    // halo_x-wide inner rows are sent first to left and right
    // Then full length rows are exchanged to the top and bottom
    int count = (inner_end_y - inner_start_y) / 2;	// The number of rows in the halo submatrix
    int block_length = halo_x / 2;	// The number of columns in the halo submatrix
    int stride = tile_width / 2;	// The combined width of the matrix with the halo
    MPI_Type_vector (count, block_length, stride, MPI_DOUBLE, &verticalBorder);
    MPI_Type_commit (&verticalBorder);

    count = halo_y / 2;	// The vertical halo in rows
    block_length = tile_width / 2;	// The number of columns of the matrix
    stride = tile_width / 2;	// The combined width of the matrix with the halo
    MPI_Type_vector (count, block_length, stride, MPI_DOUBLE, &horizontalBorder);
    MPI_Type_commit (&horizontalBorder);
#endif
}

CPUBlockSSEKernel::~CPUBlockSSEKernel() {

    free(r00[0]);
    free(r00[1]);
    free(r01[0]);
    free(r01[1]);
    free(r10[0]);
    free(r10[1]);
    free(r11[0]);
    free(r11[1]);
    free(i00[0]);
    free(i00[1]);
    free(i01[0]);
    free(i01[1]);
    free(i10[0]);
    free(i10[1]);
    free(i11[0]);
    free(i11[1]);
}


void CPUBlockSSEKernel::run_kernel_on_halo() {
    int inner = 0, sides = 0;
    if (tile_height <= block_height) {
        // One full band
        inner = 1;
        sides = 1;
        process_band_sse( tile_width, block_width, block_height, halo_x, 0, tile_height, 0, tile_height, a, b,
                          ext_pot_r00, ext_pot_r10, ext_pot_r01, ext_pot_r11,
                          ext_pot_i00, ext_pot_i10, ext_pot_i01, ext_pot_i11,
                          r00[sense], r01[sense], r10[sense], r11[sense],
                          i00[sense], i01[sense], i10[sense], i11[sense],
                          r00[1 - sense], r01[1 - sense], r10[1 - sense], r11[1 - sense],
                          i00[1 - sense], i01[1 - sense], i10[1 - sense], i11[1 - sense], inner, sides, imag_time);
    }
    else {

        // Sides
        inner = 0;
        sides = 1;
        size_t block_start;
        for (block_start = block_height - 2 * halo_y; block_start < tile_height - block_height; block_start += block_height - 2 * halo_y) {
            process_band_sse(tile_width, block_width, block_height, halo_x, block_start, block_height, halo_y, block_height - 2 * halo_y, a, b,
                             ext_pot_r00, ext_pot_r10, ext_pot_r01, ext_pot_r11,
                             ext_pot_i00, ext_pot_i10, ext_pot_i01, ext_pot_i11,
                             r00[sense], r01[sense], r10[sense], r11[sense],
                             i00[sense], i01[sense], i10[sense], i11[sense],
                             r00[1 - sense], r01[1 - sense], r10[1 - sense], r11[1 - sense],
                             i00[1 - sense], i01[1 - sense], i10[1 - sense], i11[1 - sense], inner, sides, imag_time);
        }

        // First band
        inner = 1;
        sides = 1;
        process_band_sse(tile_width, block_width, block_height, halo_x,  0, block_height, 0, block_height - halo_y, a, b,
                         ext_pot_r00, ext_pot_r10, ext_pot_r01, ext_pot_r11,
                         ext_pot_i00, ext_pot_i10, ext_pot_i01, ext_pot_i11,
                         r00[sense], r01[sense], r10[sense], r11[sense],
                         i00[sense], i01[sense], i10[sense], i11[sense],
                         r00[1 - sense], r01[1 - sense], r10[1 - sense], r11[1 - sense],
                         i00[1 - sense], i01[1 - sense], i10[1 - sense], i11[1 - sense], inner, sides, imag_time);

        // Last band
        inner = 1;
        sides = 1;
        process_band_sse(tile_width, block_width, block_height, halo_x, block_start, tile_height - block_start, halo_y, tile_height - block_start - halo_y, a, b,
                         ext_pot_r00, ext_pot_r10, ext_pot_r01, ext_pot_r11,
                         ext_pot_i00, ext_pot_i10, ext_pot_i01, ext_pot_i11,
                         r00[sense], r01[sense], r10[sense], r11[sense],
                         i00[sense], i01[sense], i10[sense], i11[sense],
                         r00[1 - sense], r01[1 - sense], r10[1 - sense], r11[1 - sense],
                         i00[1 - sense], i01[1 - sense], i10[1 - sense], i11[1 - sense], inner, sides, imag_time);
    }

}

void CPUBlockSSEKernel::run_kernel() {
    int inner = 1, sides = 0;
#ifndef HAVE_MPI
    #pragma omp parallel default(shared)
#endif
    {
#ifndef HAVE_MPI
        #pragma omp for
#endif
        for (int block_start = block_height - 2 * halo_y; block_start < tile_height - block_height; block_start += block_height - 2 * halo_y) {
            process_band_sse(tile_width, block_width, block_height, halo_x, block_start, block_height, halo_y, block_height - 2 * halo_y, a, b,
                             ext_pot_r00, ext_pot_r10, ext_pot_r01, ext_pot_r11,
                             ext_pot_i00, ext_pot_i10, ext_pot_i01, ext_pot_i11,
                             r00[sense], r01[sense], r10[sense], r11[sense],
                             i00[sense], i01[sense], i10[sense], i11[sense],
                             r00[1 - sense], r01[1 - sense], r10[1 - sense], r11[1 - sense],
                             i00[1 - sense], i01[1 - sense], i10[1 - sense], i11[1 - sense], inner, sides, imag_time);
        }
    }
    sense = 1 - sense;
}


void CPUBlockSSEKernel::wait_for_completion(int iteration) {
    if(imag_time && ((iteration % 20) == 0 )) {
        //normalization
        int nProcs = 1;
#ifdef HAVE_MPI
        MPI_Comm_size(cartcomm, &nProcs);
#endif
        int height = (tile_height - halo_y) / 2;
        int width = (tile_width - halo_x) / 2;
        double sum = 0., *sums;
		sums = new double[nProcs];
        for(int i = halo_y / 2; i < height; i++) {
            for(int j = halo_x / 2; j < width; j++) {
                int idx = j + i * tile_width / 2;
                sum += r00[sense][idx] * r00[sense][idx] + i00[sense][idx] * i00[sense][idx] +
                       r10[sense][idx] * r10[sense][idx] + i10[sense][idx] * i10[sense][idx] +
                       r01[sense][idx] * r01[sense][idx] + i01[sense][idx] * i01[sense][idx] +
                       r11[sense][idx] * r11[sense][idx] + i11[sense][idx] * i11[sense][idx];
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

        for(size_t i = 0; i < tile_height / 2; i++) {
            for(size_t j = 0; j < tile_width / 2; j++) {
                int idx = j + i * tile_width / 2;
                r00[sense][idx] /= norm;
                i00[sense][idx] /= norm;
                r10[sense][idx] /= norm;
                i10[sense][idx] /= norm;
                r01[sense][idx] /= norm;
                i01[sense][idx] /= norm;
                r11[sense][idx] /= norm;
                i11[sense][idx] /= norm;
            }
        }
		delete[] sums;
	}
}


void CPUBlockSSEKernel::get_sample(size_t dest_stride, size_t x, size_t y, size_t width, size_t height, double * dest_real, double * dest_imag) const {
    get_quadrant_sample(r00[sense], r01[sense], r10[sense], r11[sense], i00[sense], i01[sense], i10[sense], i11[sense], tile_width / 2, dest_stride, x, y, width, height, dest_real, dest_imag);
}

void CPUBlockSSEKernel::start_halo_exchange() {
#ifdef HAVE_MPI
    // Halo exchange: LEFT/RIGHT
    int offset = (inner_start_y - start_y) * tile_width / 4;
    MPI_Irecv(r00[1 - sense] + offset, 1, verticalBorder, neighbors[LEFT], 0, cartcomm, req + 0);
    MPI_Irecv(r01[1 - sense] + offset, 1, verticalBorder, neighbors[LEFT], 1, cartcomm, req + 1);
    MPI_Irecv(r10[1 - sense] + offset, 1, verticalBorder, neighbors[LEFT], 2, cartcomm, req + 2);
    MPI_Irecv(r11[1 - sense] + offset, 1, verticalBorder, neighbors[LEFT], 3, cartcomm, req + 3);
    MPI_Irecv(i00[1 - sense] + offset, 1, verticalBorder, neighbors[LEFT], 4, cartcomm, req + 4);
    MPI_Irecv(i01[1 - sense] + offset, 1, verticalBorder, neighbors[LEFT], 5, cartcomm, req + 5);
    MPI_Irecv(i10[1 - sense] + offset, 1, verticalBorder, neighbors[LEFT], 6, cartcomm, req + 6);
    MPI_Irecv(i11[1 - sense] + offset, 1, verticalBorder, neighbors[LEFT], 7, cartcomm, req + 7);
    offset = (inner_start_y - start_y) * tile_width / 4 + (inner_end_x - start_x) / 2;
    MPI_Irecv(r00[1 - sense] + offset, 1, verticalBorder, neighbors[RIGHT], 8, cartcomm, req + 8);
    MPI_Irecv(r01[1 - sense] + offset, 1, verticalBorder, neighbors[RIGHT], 9, cartcomm, req + 9);
    MPI_Irecv(r10[1 - sense] + offset, 1, verticalBorder, neighbors[RIGHT], 10, cartcomm, req + 10);
    MPI_Irecv(r11[1 - sense] + offset, 1, verticalBorder, neighbors[RIGHT], 11, cartcomm, req + 11);
    MPI_Irecv(i00[1 - sense] + offset, 1, verticalBorder, neighbors[RIGHT], 12, cartcomm, req + 12);
    MPI_Irecv(i01[1 - sense] + offset, 1, verticalBorder, neighbors[RIGHT], 13, cartcomm, req + 13);
    MPI_Irecv(i10[1 - sense] + offset, 1, verticalBorder, neighbors[RIGHT], 14, cartcomm, req + 14);
    MPI_Irecv(i11[1 - sense] + offset, 1, verticalBorder, neighbors[RIGHT], 15, cartcomm, req + 15);

    offset = (inner_start_y - start_y) * tile_width / 4 + (inner_end_x - halo_x - start_x) / 2;
    MPI_Isend(r00[1 - sense] + offset, 1, verticalBorder, neighbors[RIGHT], 0, cartcomm, req + 16);
    MPI_Isend(r01[1 - sense] + offset, 1, verticalBorder, neighbors[RIGHT], 1, cartcomm, req + 17);
    MPI_Isend(r10[1 - sense] + offset, 1, verticalBorder, neighbors[RIGHT], 2, cartcomm, req + 18);
    MPI_Isend(r11[1 - sense] + offset, 1, verticalBorder, neighbors[RIGHT], 3, cartcomm, req + 19);
    MPI_Isend(i00[1 - sense] + offset, 1, verticalBorder, neighbors[RIGHT], 4, cartcomm, req + 20);
    MPI_Isend(i01[1 - sense] + offset, 1, verticalBorder, neighbors[RIGHT], 5, cartcomm, req + 21);
    MPI_Isend(i10[1 - sense] + offset, 1, verticalBorder, neighbors[RIGHT], 6, cartcomm, req + 22);
    MPI_Isend(i11[1 - sense] + offset, 1, verticalBorder, neighbors[RIGHT], 7, cartcomm, req + 23);
    offset = (inner_start_y - start_y) * tile_width / 4 + halo_x / 2;
    MPI_Isend(r00[1 - sense] + offset, 1, verticalBorder, neighbors[LEFT], 8, cartcomm, req + 24);
    MPI_Isend(r01[1 - sense] + offset, 1, verticalBorder, neighbors[LEFT], 9, cartcomm, req + 25);
    MPI_Isend(r10[1 - sense] + offset, 1, verticalBorder, neighbors[LEFT], 10, cartcomm, req + 26);
    MPI_Isend(r11[1 - sense] + offset, 1, verticalBorder, neighbors[LEFT], 11, cartcomm, req + 27);
    MPI_Isend(i00[1 - sense] + offset, 1, verticalBorder, neighbors[LEFT], 12, cartcomm, req + 28);
    MPI_Isend(i01[1 - sense] + offset, 1, verticalBorder, neighbors[LEFT], 13, cartcomm, req + 29);
    MPI_Isend(i10[1 - sense] + offset, 1, verticalBorder, neighbors[LEFT], 14, cartcomm, req + 30);
    MPI_Isend(i11[1 - sense] + offset, 1, verticalBorder, neighbors[LEFT], 15, cartcomm, req + 31);
#else
    if(periods[1] != 0) {
        int offset = (inner_start_y - start_y) * tile_width / 4;
        memcpy2D(&(r00[1 - sense][offset]), tile_width * sizeof(double) / 2, &(r00[1 - sense][offset + tile_width / 2 - halo_x]), tile_width * sizeof(double) / 2, halo_x * sizeof(double)  / 2, tile_height / 2);
        memcpy2D(&(i00[1 - sense][offset]), tile_width * sizeof(double) / 2, &(i00[1 - sense][offset + tile_width / 2 - halo_x]), tile_width * sizeof(double) / 2, halo_x * sizeof(double)  / 2, tile_height / 2);
        memcpy2D(&(r01[1 - sense][offset]), tile_width * sizeof(double) / 2, &(r01[1 - sense][offset + tile_width / 2 - halo_x]), tile_width * sizeof(double) / 2, halo_x * sizeof(double)  / 2, tile_height / 2);
        memcpy2D(&(i01[1 - sense][offset]), tile_width * sizeof(double) / 2, &(i01[1 - sense][offset + tile_width / 2 - halo_x]), tile_width * sizeof(double) / 2, halo_x * sizeof(double)  / 2, tile_height / 2);
        memcpy2D(&(r10[1 - sense][offset]), tile_width * sizeof(double) / 2, &(r10[1 - sense][offset + tile_width / 2 - halo_x]), tile_width * sizeof(double) / 2, halo_x * sizeof(double)  / 2, tile_height / 2);
        memcpy2D(&(i10[1 - sense][offset]), tile_width * sizeof(double) / 2, &(i10[1 - sense][offset + tile_width / 2 - halo_x]), tile_width * sizeof(double) / 2, halo_x * sizeof(double)  / 2, tile_height / 2);
        memcpy2D(&(r11[1 - sense][offset]), tile_width * sizeof(double) / 2, &(r11[1 - sense][offset + tile_width / 2 - halo_x]), tile_width * sizeof(double) / 2, halo_x * sizeof(double)  / 2, tile_height / 2);
        memcpy2D(&(i11[1 - sense][offset]), tile_width * sizeof(double) / 2, &(i11[1 - sense][offset + tile_width / 2 - halo_x]), tile_width * sizeof(double) / 2, halo_x * sizeof(double)  / 2, tile_height / 2);

        memcpy2D(&(r00[1 - sense][offset + (tile_width - halo_x) / 2]), tile_width * sizeof(double) / 2, &(r00[1 - sense][offset + halo_x / 2]), tile_width * sizeof(double) / 2, halo_x * sizeof(double) / 2, tile_height / 2);
        memcpy2D(&(i00[1 - sense][offset + (tile_width - halo_x) / 2]), tile_width * sizeof(double) / 2, &(i00[1 - sense][offset + halo_x / 2]), tile_width * sizeof(double) / 2, halo_x * sizeof(double) / 2, tile_height / 2);
        memcpy2D(&(r01[1 - sense][offset + (tile_width - halo_x) / 2]), tile_width * sizeof(double) / 2, &(r01[1 - sense][offset + halo_x / 2]), tile_width * sizeof(double) / 2, halo_x * sizeof(double) / 2, tile_height / 2);
        memcpy2D(&(i01[1 - sense][offset + (tile_width - halo_x) / 2]), tile_width * sizeof(double) / 2, &(i01[1 - sense][offset + halo_x / 2]), tile_width * sizeof(double) / 2, halo_x * sizeof(double) / 2, tile_height / 2);
        memcpy2D(&(r10[1 - sense][offset + (tile_width - halo_x) / 2]), tile_width * sizeof(double) / 2, &(r10[1 - sense][offset + halo_x / 2]), tile_width * sizeof(double) / 2, halo_x * sizeof(double) / 2, tile_height / 2);
        memcpy2D(&(i10[1 - sense][offset + (tile_width - halo_x) / 2]), tile_width * sizeof(double) / 2, &(i10[1 - sense][offset + halo_x / 2]), tile_width * sizeof(double) / 2, halo_x * sizeof(double) / 2, tile_height / 2);
        memcpy2D(&(r11[1 - sense][offset + (tile_width - halo_x) / 2]), tile_width * sizeof(double) / 2, &(r11[1 - sense][offset + halo_x / 2]), tile_width * sizeof(double) / 2, halo_x * sizeof(double) / 2, tile_height / 2);
        memcpy2D(&(i11[1 - sense][offset + (tile_width - halo_x) / 2]), tile_width * sizeof(double) / 2, &(i11[1 - sense][offset + halo_x / 2]), tile_width * sizeof(double) / 2, halo_x * sizeof(double) / 2, tile_height / 2);
    }
#endif
}

void CPUBlockSSEKernel::finish_halo_exchange() {
#ifdef HAVE_MPI
    MPI_Waitall(32, req, statuses);

    // Halo exchange: UP/DOWN
    int offset = 0;
    MPI_Irecv(r00[sense] + offset, 1, horizontalBorder, neighbors[UP], 0, cartcomm, req + 0);
    MPI_Irecv(r01[sense] + offset, 1, horizontalBorder, neighbors[UP], 1, cartcomm, req + 1);
    MPI_Irecv(r10[sense] + offset, 1, horizontalBorder, neighbors[UP], 2, cartcomm, req + 2);
    MPI_Irecv(r11[sense] + offset, 1, horizontalBorder, neighbors[UP], 3, cartcomm, req + 3);
    MPI_Irecv(i00[sense] + offset, 1, horizontalBorder, neighbors[UP], 4, cartcomm, req + 4);
    MPI_Irecv(i01[sense] + offset, 1, horizontalBorder, neighbors[UP], 5, cartcomm, req + 5);
    MPI_Irecv(i10[sense] + offset, 1, horizontalBorder, neighbors[UP], 6, cartcomm, req + 6);
    MPI_Irecv(i11[sense] + offset, 1, horizontalBorder, neighbors[UP], 7, cartcomm, req + 7);
    offset = (inner_end_y - start_y) * tile_width / 4;
    MPI_Irecv(r00[sense] + offset, 1, horizontalBorder, neighbors[DOWN], 8, cartcomm, req + 8);
    MPI_Irecv(r01[sense] + offset, 1, horizontalBorder, neighbors[DOWN], 9, cartcomm, req + 9);
    MPI_Irecv(r10[sense] + offset, 1, horizontalBorder, neighbors[DOWN], 10, cartcomm, req + 10);
    MPI_Irecv(r11[sense] + offset, 1, horizontalBorder, neighbors[DOWN], 11, cartcomm, req + 11);
    MPI_Irecv(i00[sense] + offset, 1, horizontalBorder, neighbors[DOWN], 12, cartcomm, req + 12);
    MPI_Irecv(i01[sense] + offset, 1, horizontalBorder, neighbors[DOWN], 13, cartcomm, req + 13);
    MPI_Irecv(i10[sense] + offset, 1, horizontalBorder, neighbors[DOWN], 14, cartcomm, req + 14);
    MPI_Irecv(i11[sense] + offset, 1, horizontalBorder, neighbors[DOWN], 15, cartcomm, req + 15);

    offset = (inner_end_y - halo_y - start_y) * tile_width / 4;
    MPI_Isend(r00[sense] + offset, 1, horizontalBorder, neighbors[DOWN], 0, cartcomm, req + 16);
    MPI_Isend(r01[sense] + offset, 1, horizontalBorder, neighbors[DOWN], 1, cartcomm, req + 17);
    MPI_Isend(r10[sense] + offset, 1, horizontalBorder, neighbors[DOWN], 2, cartcomm, req + 18);
    MPI_Isend(r11[sense] + offset, 1, horizontalBorder, neighbors[DOWN], 3, cartcomm, req + 19);
    MPI_Isend(i00[sense] + offset, 1, horizontalBorder, neighbors[DOWN], 4, cartcomm, req + 20);
    MPI_Isend(i01[sense] + offset, 1, horizontalBorder, neighbors[DOWN], 5, cartcomm, req + 21);
    MPI_Isend(i10[sense] + offset, 1, horizontalBorder, neighbors[DOWN], 6, cartcomm, req + 22);
    MPI_Isend(i11[sense] + offset, 1, horizontalBorder, neighbors[DOWN], 7, cartcomm, req + 23);
    offset = halo_y * tile_width / 4;
    MPI_Isend(r00[sense] + offset, 1, horizontalBorder, neighbors[UP], 8, cartcomm, req + 24);
    MPI_Isend(r01[sense] + offset, 1, horizontalBorder, neighbors[UP], 9, cartcomm, req + 25);
    MPI_Isend(r10[sense] + offset, 1, horizontalBorder, neighbors[UP], 10, cartcomm, req + 26);
    MPI_Isend(r11[sense] + offset, 1, horizontalBorder, neighbors[UP], 11, cartcomm, req + 27);
    MPI_Isend(i00[sense] + offset, 1, horizontalBorder, neighbors[UP], 12, cartcomm, req + 28);
    MPI_Isend(i01[sense] + offset, 1, horizontalBorder, neighbors[UP], 13, cartcomm, req + 29);
    MPI_Isend(i10[sense] + offset, 1, horizontalBorder, neighbors[UP], 14, cartcomm, req + 30);
    MPI_Isend(i11[sense] + offset, 1, horizontalBorder, neighbors[UP], 15, cartcomm, req + 31);

    MPI_Waitall(32, req, statuses);
#else
    if(periods[0] != 0) {
        int offset = (inner_end_y - start_y) * tile_width / 4;
        memcpy2D(r00[sense], tile_width * sizeof(double) / 2, &(r00[sense][offset - halo_y * tile_width / 4]), tile_width * sizeof(double) / 2, tile_width * sizeof(double) / 2, halo_y / 2);
        memcpy2D(i00[sense], tile_width * sizeof(double) / 2, &(i00[sense][offset - halo_y * tile_width / 4]), tile_width * sizeof(double) / 2, tile_width * sizeof(double) / 2, halo_y / 2);
        memcpy2D(r01[sense], tile_width * sizeof(double) / 2, &(r01[sense][offset - halo_y * tile_width / 4]), tile_width * sizeof(double) / 2, tile_width * sizeof(double) / 2, halo_y / 2);
        memcpy2D(i01[sense], tile_width * sizeof(double) / 2, &(i01[sense][offset - halo_y * tile_width / 4]), tile_width * sizeof(double) / 2, tile_width * sizeof(double) / 2, halo_y / 2);
        memcpy2D(r10[sense], tile_width * sizeof(double) / 2, &(r10[sense][offset - halo_y * tile_width / 4]), tile_width * sizeof(double) / 2, tile_width * sizeof(double) / 2, halo_y / 2);
        memcpy2D(i10[sense], tile_width * sizeof(double) / 2, &(i10[sense][offset - halo_y * tile_width / 4]), tile_width * sizeof(double) / 2, tile_width * sizeof(double) / 2, halo_y / 2);
        memcpy2D(r11[sense], tile_width * sizeof(double) / 2, &(r11[sense][offset - halo_y * tile_width / 4]), tile_width * sizeof(double) / 2, tile_width * sizeof(double) / 2, halo_y / 2);
        memcpy2D(i11[sense], tile_width * sizeof(double) / 2, &(i11[sense][offset - halo_y * tile_width / 4]), tile_width * sizeof(double) / 2, tile_width * sizeof(double) / 2, halo_y / 2);

        memcpy2D(&(r00[sense][offset]), tile_width * sizeof(double) / 2, &(r00[sense][halo_y * tile_width / 4]), tile_width * sizeof(double) / 2, tile_width * sizeof(double) / 2, halo_y / 2);
        memcpy2D(&(i00[sense][offset]), tile_width * sizeof(double) / 2, &(i00[sense][halo_y * tile_width / 4]), tile_width * sizeof(double) / 2, tile_width * sizeof(double) / 2, halo_y / 2);
        memcpy2D(&(r01[sense][offset]), tile_width * sizeof(double) / 2, &(r01[sense][halo_y * tile_width / 4]), tile_width * sizeof(double) / 2, tile_width * sizeof(double) / 2, halo_y / 2);
        memcpy2D(&(i01[sense][offset]), tile_width * sizeof(double) / 2, &(i01[sense][halo_y * tile_width / 4]), tile_width * sizeof(double) / 2, tile_width * sizeof(double) / 2, halo_y / 2);
        memcpy2D(&(r10[sense][offset]), tile_width * sizeof(double) / 2, &(r10[sense][halo_y * tile_width / 4]), tile_width * sizeof(double) / 2, tile_width * sizeof(double) / 2, halo_y / 2);
        memcpy2D(&(i10[sense][offset]), tile_width * sizeof(double) / 2, &(i10[sense][halo_y * tile_width / 4]), tile_width * sizeof(double) / 2, tile_width * sizeof(double) / 2, halo_y / 2);
        memcpy2D(&(r11[sense][offset]), tile_width * sizeof(double) / 2, &(r11[sense][halo_y * tile_width / 4]), tile_width * sizeof(double) / 2, tile_width * sizeof(double) / 2, halo_y / 2);
        memcpy2D(&(i11[sense][offset]), tile_width * sizeof(double) / 2, &(i11[sense][halo_y * tile_width / 4]), tile_width * sizeof(double) / 2, tile_width * sizeof(double) / 2, halo_y / 2);
    }
#endif
}
