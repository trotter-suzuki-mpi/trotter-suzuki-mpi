/**
 * Massively Parallel Trotter-Suzuki Solver
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
#include "cpublocktest.h"
#include "kernel.h"
#include "common.h"
#include "cpublock.cpp"

CPPUNIT_TEST_SUITE_REGISTRATION( CPUBlockTest );

void CPUBlockTest::setUp() {}
void CPUBlockTest::tearDown() {}

void CPUBlockTest::test_block_kernel_vertical() {
    //Set Up
    int DIM = 640;
    int offset = 0;
    double a = h_a, b = h_b;

    double *block_real = new double[DIM * DIM];
    double *block_imag = new double[DIM * DIM];
    double *block_real_expected = new double[DIM * DIM];
    double *block_imag_expected = new double[DIM * DIM];

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
    double a = h_a, b = h_b;
    double *block_real = new double[DIM * DIM];
    double *block_imag = new double[DIM * DIM];
    double *block_real_expected = new double[DIM * DIM];
    double *block_imag_expected = new double[DIM * DIM];

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



//Members of class Matrix
Matrix::Matrix(double *matrix_real, double *matrix_imag, int width, int height) {
    m_real = new double[width * height];
    m_imag = new double[width * height];
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
