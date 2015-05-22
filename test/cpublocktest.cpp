/**
 * Distributed Trotter-Suzuki solver
 * Copyright (C) 2012 Peter Wittek, 2010-2012 Carlos Bederián, 2015 Luca Calderaro
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

#include "cpublocktest.h"
#include "cpublocksse.h"
#include "cpublock.h"
#include "common.h"
#include <iostream>

CPPUNIT_TEST_SUITE_REGISTRATION( CPUBlockTest );

void CPUBlockTest::setUp() {}
void CPUBlockTest::tearDown() {}

//cpublocksse's functions
template <int offset_y>
inline void update_shifty_sse(size_t stride, size_t width, size_t height, float a, float b, float * __restrict__ r1, float * __restrict__ i1, float * __restrict__ r2, float * __restrict__ i2) {
    __m128 aq, bq;
    aq = _mm_load1_ps(&a);
    bq = _mm_load1_ps(&b);
    for (size_t i = 0; i < height - offset_y; i++) {
        int idx1 = i * stride;
        int idx2 = (i + offset_y) * stride;
        size_t j = 0;
        for (; j < width - width % 4; j += 4, idx1 += 4, idx2 += 4) {
            __m128 r1q = _mm_load_ps(&r1[idx1]);
            __m128 i1q = _mm_load_ps(&i1[idx1]);
            __m128 r2q = _mm_load_ps(&r2[idx2]);
            __m128 i2q = _mm_load_ps(&i2[idx2]);
            __m128 next_r1q = _mm_sub_ps(_mm_mul_ps(r1q, aq), _mm_mul_ps(i2q, bq));
            __m128 next_i1q = _mm_add_ps(_mm_mul_ps(i1q, aq), _mm_mul_ps(r2q, bq));
            __m128 next_r2q = _mm_sub_ps(_mm_mul_ps(r2q, aq), _mm_mul_ps(i1q, bq));
            __m128 next_i2q = _mm_add_ps(_mm_mul_ps(i2q, aq), _mm_mul_ps(r1q, bq));
            _mm_store_ps(&r1[idx1], next_r1q);
            _mm_store_ps(&i1[idx1], next_i1q);
            _mm_store_ps(&r2[idx2], next_r2q);
            _mm_store_ps(&i2[idx2], next_i2q);
        }
        for (; j < width; ++j, ++idx1, ++idx2) {
            float next_r1 = a * r1[idx1] - b * i2[idx2];
            float next_i1 = a * i1[idx1] + b * r2[idx2];
            float next_r2 = a * r2[idx2] - b * i1[idx1];
            float next_i2 = a * i2[idx2] + b * r1[idx1];
            r1[idx1] = next_r1;
            i1[idx1] = next_i1;
            r2[idx2] = next_r2;
            i2[idx2] = next_i2;
        }
    }
}

template <int offset_x>
inline void update_shiftx_sse(size_t stride, size_t width, size_t height, float a, float b, float * __restrict__ r1, float * __restrict__ i1, float * __restrict__ r2, float * __restrict__ i2) {
    __m128 aq, bq;
    aq = _mm_load1_ps(&a);
    bq = _mm_load1_ps(&b);
    for (size_t i = 0; i < height; i++) {
        int idx1 = i * stride;
        int idx2 = i * stride + offset_x;
        size_t j = 0;
        for (; j < width - offset_x - (width - offset_x) % 4; j += 4, idx1 += 4, idx2 += 4) {
            __m128 r1q = _mm_load_ps(&r1[idx1]);
            __m128 i1q = _mm_load_ps(&i1[idx1]);
            __m128 r2q;
            __m128 i2q;
            if (offset_x == 0) {
                r2q = _mm_load_ps(&r2[idx2]);
                i2q = _mm_load_ps(&i2[idx2]);
            }
            else {
                r2q = _mm_loadu_ps(&r2[idx2]);
                i2q = _mm_loadu_ps(&i2[idx2]);
            }
            __m128 next_r1q = _mm_sub_ps(_mm_mul_ps(r1q, aq), _mm_mul_ps(i2q, bq));
            __m128 next_i1q = _mm_add_ps(_mm_mul_ps(i1q, aq), _mm_mul_ps(r2q, bq));
            __m128 next_r2q = _mm_sub_ps(_mm_mul_ps(r2q, aq), _mm_mul_ps(i1q, bq));
            __m128 next_i2q = _mm_add_ps(_mm_mul_ps(i2q, aq), _mm_mul_ps(r1q, bq));
            _mm_store_ps(&r1[idx1], next_r1q);
            _mm_store_ps(&i1[idx1], next_i1q);
            if (offset_x == 0) {
                _mm_store_ps(&r2[idx2], next_r2q);
                _mm_store_ps(&i2[idx2], next_i2q);
            }
            else {
                _mm_storeu_ps(&r2[idx2], next_r2q);
                _mm_storeu_ps(&i2[idx2], next_i2q);
            }
        }
        for (; j < width - offset_x; ++j, ++idx1, ++idx2) {
            float next_r1 = a * r1[idx1] - b * i2[idx2];
            float next_i1 = a * i1[idx1] + b * r2[idx2];
            float next_r2 = a * r2[idx2] - b * i1[idx1];
            float next_i2 = a * i2[idx2] + b * r1[idx1];
            r1[idx1] = next_r1;
            i1[idx1] = next_i1;
            r2[idx2] = next_r2;
            i2[idx2] = next_i2;
        }
    }
}


void CPUBlockTest::test_block_kernel_vertical() {
    //Set Up
    int DIM = 640;
    int offset = 0;
    float a = h_a, b = h_b;

    float *block_real = new float[DIM * DIM];
    float *block_imag = new float[DIM * DIM];
    float *block_real_expected = new float[DIM * DIM];
    float *block_imag_expected = new float[DIM * DIM];

    //initialize block_real, block_imag
    for(int i = 0; i < DIM; i++) {
        for(int j = 0; j < DIM; j++) {
            block_real[i * DIM + j] = 1.;
            block_imag[i * DIM + j] = 0.;
        }
    }

    //inizialize block_real_expected, block_imag_expected
    for(int i = 0; i < DIM; i++) {
        for(int j = 0; j < DIM; j++) {
            if((i == 0 || i == DIM - 1) && ((j + offset + 1) % 2) == 0) {
                block_real_expected[i * DIM + j] = 1.;
                block_imag_expected[i * DIM + j] = 0.;
            }
            else {
                block_real_expected[i * DIM + j] = a;
                block_imag_expected[i * DIM + j] = b;
            }
        }
    }

    //Process block_real, block_imag
    block_kernel_vertical(offset, DIM, DIM, DIM, a , b, block_real, block_imag);

    Matrix matrix_processed(block_real, block_imag, DIM, DIM);
    Matrix matrix_expected(block_real_expected, block_imag_expected, DIM, DIM);

    //Check
    CPPUNIT_ASSERT( matrix_processed == matrix_expected );
    std::cout << "TEST FUNCTION: block_kernel_vertical -> PASSED! " << std::endl;
}

void CPUBlockTest::test_block_kernel_horizontal() {
    //Set Up
    int DIM = 640;
    int offset = 0;
    float a = h_a, b = h_b;
    float *block_real = new float[DIM * DIM];
    float *block_imag = new float[DIM * DIM];
    float *block_real_expected = new float[DIM * DIM];
    float *block_imag_expected = new float[DIM * DIM];

    //initialize block_real, block_imag
    for(int i = 0; i < DIM; i++) {
        for(int j = 0; j < DIM; j++) {
            block_real[i * DIM + j] = 1.;
            block_imag[i * DIM + j] = 0.;
        }
    }

    //inizialize block_real_expected, block_imag_expected
    for(int i = 0; i < DIM; i++) {
        for(int j = 0; j < DIM; j++) {
            if((j == 0 || j == DIM - 1) && ((i + offset + 1) % 2) == 0) {
                block_real_expected[i * DIM + j] = 1.;
                block_imag_expected[i * DIM + j] = 0.;
            }
            else {
                block_real_expected[i * DIM + j] = a;
                block_imag_expected[i * DIM + j] = b;
            }
        }
    }

    //Process block_real, block_imag
    block_kernel_horizontal(offset, DIM, DIM, DIM, a , b, block_real, block_imag);

    Matrix matrix_processed(block_real, block_imag, DIM, DIM);
    Matrix matrix_expected(block_real_expected, block_imag_expected, DIM, DIM);

    //Check
    CPPUNIT_ASSERT( matrix_processed == matrix_expected );
    std::cout << "TEST FUNCTION: block_kernel_horizontal -> PASSED! " << std::endl;
}

//TEST cpublocksse

void CPUBlockTest::test_update_shifty_sse() {
    //Set Up
    int DIM = 640;
    int offset = 0;
    float a = h_a, b = h_b;
    float *block_r00 = new float[DIM * DIM];
    float *block_i00 = new float[DIM * DIM];
    float *block_r10 = new float[DIM * DIM];
    float *block_i10 = new float[DIM * DIM];
    float *block_r00_expected = new float[DIM * DIM];
    float *block_i00_expected = new float[DIM * DIM];
    float *block_r10_expected = new float[DIM * DIM];
    float *block_i10_expected = new float[DIM * DIM];

    //initialize block_r00, block_i00, block_r10, block_i10
    for(int i = 0; i < DIM; i++) {
        for(int j = 0; j < DIM; j++) {
            block_r00[i * DIM + j] = 1.;
            block_i00[i * DIM + j] = 0.;
            block_r10[i * DIM + j] = 1.;
            block_i10[i * DIM + j] = 0.;
        }
    }

    //inizialize block_r00_expected, block_i00_expected
    for(int i = 0; i < DIM; i++) {
        for(int j = 0; j < DIM; j++) {
            block_r00_expected[i * DIM + j] = a;
            block_i00_expected[i * DIM + j] = b;
            block_r10_expected[i * DIM + j] = a;
            block_i10_expected[i * DIM + j] = b;
        }
    }

    //Process block_r00, block_i00, block_r10, block_i10
    update_shifty_sse<0>(DIM, DIM, DIM, a, b, block_r00, block_i00, block_r10, block_i10);

    Matrix matrix_00_processed(block_r00, block_i00, DIM, DIM);
    Matrix matrix_10_processed(block_r10, block_i10, DIM, DIM);
    Matrix matrix_00_expected(block_r00_expected, block_i00_expected, DIM, DIM);
    Matrix matrix_10_expected(block_r10_expected, block_i10_expected, DIM, DIM);

    //Check
    CPPUNIT_ASSERT( matrix_00_processed == matrix_00_expected );
    CPPUNIT_ASSERT( matrix_10_processed == matrix_10_expected );
    std::cout << "TEST FUNCTION: update_shifty_sse -> PASSED! " << std::endl;
}

void CPUBlockTest::test_update_shiftx_sse() {
    //Set Up
    int DIM = 640;
    int offset = 0;
    float a = h_a, b = h_b;
    float *block_r00 = new float[DIM * DIM];
    float *block_i00 = new float[DIM * DIM];
    float *block_r10 = new float[DIM * DIM];
    float *block_i10 = new float[DIM * DIM];
    float *block_r00_expected = new float[DIM * DIM];
    float *block_i00_expected = new float[DIM * DIM];
    float *block_r10_expected = new float[DIM * DIM];
    float *block_i10_expected = new float[DIM * DIM];

    //initialize block_r00, block_i00, block_r10, block_i10
    for(int i = 0; i < DIM; i++) {
        for(int j = 0; j < DIM; j++) {
            block_r00[i * DIM + j] = 1.;
            block_i00[i * DIM + j] = 0.;
            block_r10[i * DIM + j] = 1.;
            block_i10[i * DIM + j] = 0.;
        }
    }

    //inizialize block_r00_expected, block_i00_expected
    for(int i = 0; i < DIM; i++) {
        for(int j = 0; j < DIM; j++) {
            block_r00_expected[i * DIM + j] = a;
            block_i00_expected[i * DIM + j] = b;
            block_r10_expected[i * DIM + j] = a;
            block_i10_expected[i * DIM + j] = b;
        }
    }

    //Process block_r00, block_i00, block_r10, block_i10
    update_shiftx_sse<0>(DIM, DIM, DIM, a, b, block_r00, block_i00, block_r10, block_i10);

    Matrix matrix_00_processed(block_r00, block_i00, DIM, DIM);
    Matrix matrix_10_processed(block_r10, block_i10, DIM, DIM);
    Matrix matrix_00_expected(block_r00_expected, block_i00_expected, DIM, DIM);
    Matrix matrix_10_expected(block_r10_expected, block_i10_expected, DIM, DIM);

    //Check
    CPPUNIT_ASSERT( matrix_00_processed == matrix_00_expected );
    CPPUNIT_ASSERT( matrix_10_processed == matrix_10_expected );
    std::cout << "TEST FUNCTION: update_shiftx_sse -> PASSED! " << std::endl;
}


//Members of class Matrix
Matrix::Matrix(float *matrix_real, float *matrix_imag, int width, int height) {
    m_real = new float[width * height];
    m_imag = new float[width * height];
    m_width = width;
    m_height = height;

    for(int i = 0; i < m_height; i++) {
        for(int j = 0; j < m_width; j++) {
            m_real[i * m_width + j] = matrix_real[i * m_width + j];
            m_imag[i * m_width + j] = matrix_imag[i * m_width + j];
        }
    }
}

void Matrix::show_matrix() {
    for(int i = 0; i < m_height; i++) {
        for(int j = 0; j < m_width; j++) {
            std::cout << "(" << m_real[i * m_height + j] << " , " << m_imag[i * m_height + j] << ") ";
        }
        std::cout << std::endl;
    }
}

bool Matrix::operator ==(const Matrix &other) const {
    bool var = false;
    if((m_height == other.m_height) && (m_width == other.m_width)) {
        var = true;
        for(int i = 0; i < m_height; i++) {
            for(int j = 0; j < m_width; j++) {
                if((m_real[i * m_width + j] != other.m_real[i * m_width + j]) || (m_imag[i * m_width + j] != other.m_imag[i * m_width + j]))
                    var = false;
            }
        }
    }
    return var;
}

