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

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>


#include <sstream>
#include <unistd.h>
#include <complex>

#include "common.h"

void calculate_borders(int coord, int dim, int * start, int *end, int *inner_start, int *inner_end, int length, int halo, int periodic_bound) {
    int inner = (int)ceil((double)length / (double)dim);
    *inner_start = coord * inner;
    if(periodic_bound != 0)
		*start = *inner_start - halo;
	else
		*start = ( coord == 0 ? 0 : *inner_start - halo );
    *end = *inner_start + (inner + halo);
    
    if (*end > length) {
		if(periodic_bound != 0)
			*end = length + halo;
		else
			*end = length;
    }
    if(periodic_bound != 0)
		*inner_end = *end - halo;
	else
		*inner_end = ( *end == length ? *end : *end - halo );
}

void print_complex_matrix(std::string filename, float * matrix_real, float * matrix_imag, size_t stride, size_t width, size_t height) {
    std::ofstream out(filename.c_str(), std::ios::out | std::ios::trunc);
    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            out << "(" << matrix_real[i * stride + j] << "," << matrix_imag[i * stride + j] << ") ";
        }
        out << std::endl;
    }
    out.close();
}

void print_matrix(std::string filename, float * matrix, size_t stride, size_t width, size_t height) {
    std::ofstream out(filename.c_str(), std::ios::out | std::ios::trunc);
    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            out << matrix[i * stride + j] << " ";
        }
        out << std::endl;
    }
    out.close();
}

void memcpy2D(void * dst, size_t dstride, const void * src, size_t sstride, size_t width, size_t height) {
    char *d = reinterpret_cast<char *>(dst);
    const char *s = reinterpret_cast<const char *>(src);
    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            d[i * dstride + j] = s[i * sstride + j];
        }
    }
}

void merge_line(const float * evens, const float * odds, size_t x, size_t width, float * dest) {

    const float * odds_p = odds + (x / 2);
    const float * evens_p = evens + (x / 2);

    size_t dest_x = x;
    if (x % 2 == 1) {
        dest[dest_x++] = *(odds_p++);
    }
    while (dest_x < (x + width) - (x + width) % 2) {
        dest[dest_x++] = *(evens_p++);
        dest[dest_x++] = *(odds_p++);
    }
    if (dest_x < x + width) {
        dest[dest_x++] = *evens_p;
    }
    assert(dest_x == x + width);
}


void get_quadrant_sample(const float * r00, const float * r01, const float * r10, const float * r11,
                         const float * i00, const float * i01, const float * i10, const float * i11,
                         size_t src_stride, size_t dest_stride,
                         size_t x, size_t y, size_t width, size_t height,
                         float * dest_real, float * dest_imag) {
    size_t dest_y = y;
    if (y % 2 == 1) {
        merge_line(&r10[(y / 2) * src_stride], &r11[(y / 2) * src_stride], x, width, dest_real);
        merge_line(&i10[(y / 2) * src_stride], &i11[(y / 2) * src_stride], x, width, dest_imag);
        ++dest_y;
    }
    while (dest_y < (y + height) - (y + height) % 2) {
        merge_line(&r00[(dest_y / 2) * src_stride], &r01[(dest_y / 2) * src_stride], x, width, &dest_real[dest_y * dest_stride]);
        merge_line(&i00[(dest_y / 2) * src_stride], &i01[(dest_y / 2) * src_stride], x, width, &dest_imag[dest_y * dest_stride]);
        ++dest_y;
        merge_line(&r10[(dest_y / 2) * src_stride], &r11[(dest_y / 2) * src_stride], x, width, &dest_real[dest_y * dest_stride]);
        merge_line(&i10[(dest_y / 2) * src_stride], &i11[(dest_y / 2) * src_stride], x, width, &dest_imag[dest_y * dest_stride]);
        ++dest_y;
    }
    if (dest_y < y + height) {
        merge_line(&r00[(dest_y / 2) * src_stride], &r01[(dest_y / 2) * src_stride], x, width, &dest_real[dest_y * dest_stride]);
        merge_line(&i00[(dest_y / 2) * src_stride], &i01[(dest_y / 2) * src_stride], x, width, &dest_imag[dest_y * dest_stride]);
    }
    assert (dest_y == y + height);
}

void merge_line_to_buffer(const float * evens, const float * odds, size_t x, size_t width, float * dest) {

    const float * odds_p = odds + (x / 2);
    const float * evens_p = evens + (x / 2);

    size_t dest_x = x;
    size_t buffer_x = 0;
    if (x % 2 == 1) {
        dest[buffer_x++] = *(odds_p++);
        dest_x++;
    }
    while (dest_x < (x + width) - (x + width) % 2) {
        dest[buffer_x++] = *(evens_p++);
        dest[buffer_x++] = *(odds_p++);
        dest_x++;
        dest_x++;
    }
    if (dest_x < x + width) {
        dest[buffer_x++] = *evens_p;
        dest_x++;
    }
    assert(dest_x == x + width);
}


void get_quadrant_sample_to_buffer(const float * r00, const float * r01, const float * r10, const float * r11,
                                   const float * i00, const float * i01, const float * i10, const float * i11,
                                   size_t src_stride, size_t dest_stride,
                                   size_t x, size_t y, size_t width, size_t height,
                                   float * dest_real, float * dest_imag) {
    size_t dest_y = y;
    size_t buffer_y = 0;
    if (y % 2 == 1) {
        merge_line_to_buffer(&r10[(y / 2) * src_stride], &r11[(y / 2) * src_stride], x, width, dest_real);
        merge_line_to_buffer(&i10[(y / 2) * src_stride], &i11[(y / 2) * src_stride], x, width, dest_imag);
        ++dest_y;
        ++buffer_y;
    }
    while (dest_y < (y + height) - (y + height) % 2) {
        merge_line_to_buffer(&r00[(dest_y / 2) * src_stride], &r01[(dest_y / 2) * src_stride], x, width, &dest_real[buffer_y * dest_stride]);
        merge_line_to_buffer(&i00[(dest_y / 2) * src_stride], &i01[(dest_y / 2) * src_stride], x, width, &dest_imag[buffer_y * dest_stride]);
        ++dest_y;
        ++buffer_y;
        merge_line_to_buffer(&r10[(dest_y / 2) * src_stride], &r11[(dest_y / 2) * src_stride], x, width, &dest_real[buffer_y * dest_stride]);
        merge_line_to_buffer(&i10[(dest_y / 2) * src_stride], &i11[(dest_y / 2) * src_stride], x, width, &dest_imag[buffer_y * dest_stride]);
        ++dest_y;
        ++buffer_y;
    }
    if (dest_y < y + height) {
        merge_line_to_buffer(&r00[(dest_y / 2) * src_stride], &r01[(dest_y / 2) * src_stride], x, width, &dest_real[buffer_y * dest_stride]);
        merge_line_to_buffer(&i00[(dest_y / 2) * src_stride], &i01[(dest_y / 2) * src_stride], x, width, &dest_imag[buffer_y * dest_stride]);
    }
    assert (dest_y == y + height);
}

void expect_values(int dim, int iterations, int snapshots, float * hamilt_pot, float particle_mass, const char *dirname) {

    if(snapshots == 0)
        return;

    int N_files = iterations / snapshots;
    int N_name[N_files];
    int DIM = dim;
    N_name[0] = 0;
    for(int i = 1; i < N_files; i++) {
        N_name[i] = N_name[i - 1] + snapshots;
    }

    std::complex<float> sum_E = 0;
    std::complex<float> sum_Px = 0, sum_Py = 0;
    std::complex<float> sum_psi = 0;

    std::complex<float> potential[DIM][DIM];
    std::complex<float> psi[DIM][DIM];
    std::complex<float> cost_E = -1. / (2.*particle_mass), cost_P;
    cost_P = std::complex<float>(0., -0.5);

    std::stringstream filename;
    std::string filenames;

    filename.str("");
    filename << dirname << "/exp_val_D" << dim << "_I" << iterations << "_S" << snapshots << ".dat";
    filenames = filename.str();
    std::ofstream out(filenames.c_str());

    for(int i = 0; i < DIM; i++) {
        for(int j = 0; j < DIM; j++) {
            potential[j][i] = std::complex<float> (hamilt_pot[i * DIM + j], 0.);
        }
    }

    out << "#time\tEnergy\t\tPx\tPy\tP**2\tnorm(psi(t))" << std::endl;
    for(int i = 0; i < N_files; i++) {
        filename.str("");
        filename << dirname << "/" << N_name[i] << "-iter-0-0-comp.dat";
        filenames = filename.str();
        std::ifstream in_compl(filenames.c_str());

        for(int j = 0; j < DIM; j++) {
            for(int k = 0; k < DIM; k++) {
                in_compl >> psi[k][j];
            }
        }
        in_compl.close();

        for(int j = 1; j < DIM - 1; j++) {
            for(int k = 1; k < DIM - 1; k++) {
                sum_E += conj(psi[k][j]) * (cost_E * (psi[k + 1][j] + psi[k - 1][j] + psi[k][j + 1] + psi[k][j - 1] - psi[k][j] * std::complex<float> (4., 0.)) + potential[k][j] * psi[k][j]) ;
                sum_Px += conj(psi[k][j]) * (psi[k + 1][j] - psi[k - 1][j]);
                sum_Py += conj(psi[k][j]) * (psi[k][j + 1] - psi[k][j - 1]);
                sum_psi += conj(psi[k][j]) * psi[k][j];
            }
        }

        out << N_name[i] << "\t" << real(sum_E / sum_psi) << "\t" << real(cost_P * sum_Px / sum_psi) << "\t" << real(cost_P * sum_Py / sum_psi) << "\t"
            << real(cost_P * sum_Px / sum_psi)*real(cost_P * sum_Px / sum_psi) + real(cost_P * sum_Py / sum_psi)*real(cost_P * sum_Py / sum_psi) << "\t" << real(sum_psi) << std::endl;
        sum_E = 0;
        sum_Px = 0;
        sum_Py = 0;
        sum_psi = 0;
    }
}

