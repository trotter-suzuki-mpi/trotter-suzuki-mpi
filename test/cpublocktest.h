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

#ifndef __CPUBLOCKTEST_H
#define __CPUBLOCKTEST_H

#include <cmath>
#include <cppunit/extensions/HelperMacros.h>

static const double h_a = cos(0.02);
static const double h_b = sin(0.02);

class CPUBlockTest: public CppUnit::TestFixture {
    CPPUNIT_TEST_SUITE( CPUBlockTest );
    CPPUNIT_TEST( test_block_kernel_vertical );
    CPPUNIT_TEST( test_block_kernel_horizontal );
    CPPUNIT_TEST_SUITE_END();

public:
    void setUp();
    void tearDown();
    void test_block_kernel_vertical();
    void test_block_kernel_horizontal();
};

class Matrix {
public:
    Matrix(double *matrix_real, double *matrix_imag, int width, int height);
    void show_matrix();
    bool operator ==(const Matrix &other) const;

private:
    double *m_real;
    double *m_imag;
    int m_width, m_height;
};
#endif
