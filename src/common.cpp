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

#include "common.h"

void calculate_borders(int coord, int dim, int * start, int *end, int *inner_start, int *inner_end, int length, int halo)
{
    int inner=(int)ceil((double)length/(double)dim);
    *inner_start=coord*inner;
    *start = ( coord == 0 ? 0 : *inner_start - halo );
    *end = *inner_start + (inner+halo);
    if (*end>length)
    {
        *end=length;
    }
    *inner_end = ( *end == length ? *end : *end - halo );
}

void print_complex_matrix(std::string filename, float * matrix_real, float * matrix_imag, size_t stride, size_t width, size_t height)
{
    std::ofstream out(filename.c_str(), std::ios::out | std::ios::trunc);
    for (size_t i = 0; i < height; ++i)
    {
        for (size_t j = 0; j < width; ++j)
        {
            out << "(" << matrix_real[i * stride + j] << "," << matrix_imag[i * stride + j] << ") ";
        }
        out << std::endl;
    }
    out.close();
}

void print_matrix(std::string filename, float * matrix, size_t stride, size_t width, size_t height)
{
    std::ofstream out(filename.c_str(), std::ios::out | std::ios::trunc);
    for (size_t i = 0; i < height; ++i)
    {
        for (size_t j = 0; j < width; ++j)
        {
            out << matrix[i * stride + j] << " ";
        }
        out << std::endl;
    }
    out.close();
}

void init_p(float *p_real, float *p_imag, int start_x, int end_x, int start_y, int end_y)
{
    double s = 64.0; // FIXME: y esto?
    for (int y = start_y+1, j=0; y <= end_y; y++,j++)
    {
        for (int x = start_x+1, i=0; x <= end_x; x++,i++)
        {
            std::complex<float> tmp = std::complex<float>(exp(-(pow(x - 180.0, 2.0) + pow(y - 300.0, 2.0)) / (2.0 * pow(s, 2.0))), 0.0)
                                      * exp(std::complex<float>(0.0, 0.4 * (x + y - 480.0)));

            p_real[j * (end_x-start_x) + i] = real(tmp);
            p_imag[j * (end_x-start_x) + i] = imag(tmp);
        }
    }
}

void memcpy2D(void * dst, size_t dstride, const void * src, size_t sstride, size_t width, size_t height)
{
    char *d = reinterpret_cast<char *>(dst);
    const char *s = reinterpret_cast<const char *>(src);
    for (size_t i = 0; i < height; ++i)
    {
        for (size_t j = 0; j < width; ++j)
        {
            d[i * dstride + j] = s[i * sstride + j];
        }
    }
}

void merge_line(const float * evens, const float * odds, size_t x, size_t width, float * dest)
{

    const float * odds_p = odds + (x/2);
    const float * evens_p = evens + (x/2);

    size_t dest_x = x;
    if (x % 2 == 1)
    {
        dest[dest_x++] = *(odds_p++);
    }
    while (dest_x < (x + width) - (x + width) % 2)
    {
        dest[dest_x++] = *(evens_p++);
        dest[dest_x++] = *(odds_p++);
    }
    if (dest_x < x + width)
    {
        dest[dest_x++] = *evens_p;
    }
    assert(dest_x == x + width);
}


void get_quadrant_sample(const float * r00, const float * r01, const float * r10, const float * r11,
                         const float * i00, const float * i01, const float * i10, const float * i11,
                         size_t src_stride, size_t dest_stride,
                         size_t x, size_t y, size_t width, size_t height,
                         float * dest_real, float * dest_imag)
{
    size_t dest_y = y;
    if (y % 2 == 1)
    {
        merge_line(&r10[(y/2) * src_stride], &r11[(y/2) * src_stride], x, width, dest_real);
        merge_line(&i10[(y/2) * src_stride], &i11[(y/2) * src_stride], x, width, dest_imag);
        ++dest_y;
    }
    while (dest_y < (y + height) - (y + height) % 2)
    {
        merge_line(&r00[(dest_y/2) * src_stride], &r01[(dest_y/2) * src_stride], x, width, &dest_real[dest_y * dest_stride]);
        merge_line(&i00[(dest_y/2) * src_stride], &i01[(dest_y/2) * src_stride], x, width, &dest_imag[dest_y * dest_stride]);
        ++dest_y;
        merge_line(&r10[(dest_y/2) * src_stride], &r11[(dest_y/2) * src_stride], x, width, &dest_real[dest_y * dest_stride]);
        merge_line(&i10[(dest_y/2) * src_stride], &i11[(dest_y/2) * src_stride], x, width, &dest_imag[dest_y * dest_stride]);
        ++dest_y;
    }
    if (dest_y < y + height)
    {
        merge_line(&r00[(dest_y/2) * src_stride], &r01[(dest_y/2) * src_stride], x, width, &dest_real[dest_y * dest_stride]);
        merge_line(&i00[(dest_y/2) * src_stride], &i01[(dest_y/2) * src_stride], x, width, &dest_imag[dest_y * dest_stride]);
    }
    assert (dest_y == y + height);
}

void merge_line_to_buffer(const float * evens, const float * odds, size_t x, size_t width, float * dest)
{

    const float * odds_p = odds + (x/2);
    const float * evens_p = evens + (x/2);

    size_t dest_x = x;
    size_t buffer_x = 0;
    if (x % 2 == 1)
    {
        dest[buffer_x++] = *(odds_p++);
        dest_x++;
    }
    while (dest_x < (x + width) - (x + width) % 2)
    {
        dest[buffer_x++] = *(evens_p++);
        dest[buffer_x++] = *(odds_p++);
        dest_x++;
        dest_x++;
    }
    if (dest_x < x + width)
    {
        dest[buffer_x++] = *evens_p;
        dest_x++;
    }
    assert(dest_x == x + width);
}


void get_quadrant_sample_to_buffer(const float * r00, const float * r01, const float * r10, const float * r11,
                                   const float * i00, const float * i01, const float * i10, const float * i11,
                                   size_t src_stride, size_t dest_stride,
                                   size_t x, size_t y, size_t width, size_t height,
                                   float * dest_real, float * dest_imag)
{
    size_t dest_y = y;
    size_t buffer_y = 0;
    if (y % 2 == 1)
    {
        merge_line_to_buffer(&r10[(y/2) * src_stride], &r11[(y/2) * src_stride], x, width, dest_real);
        merge_line_to_buffer(&i10[(y/2) * src_stride], &i11[(y/2) * src_stride], x, width, dest_imag);
        ++dest_y;
        ++buffer_y;
    }
    while (dest_y < (y + height) - (y + height) % 2)
    {
        merge_line_to_buffer(&r00[(dest_y/2) * src_stride], &r01[(dest_y/2) * src_stride], x, width, &dest_real[buffer_y * dest_stride]);
        merge_line_to_buffer(&i00[(dest_y/2) * src_stride], &i01[(dest_y/2) * src_stride], x, width, &dest_imag[buffer_y * dest_stride]);
        ++dest_y;
        ++buffer_y;
        merge_line_to_buffer(&r10[(dest_y/2) * src_stride], &r11[(dest_y/2) * src_stride], x, width, &dest_real[buffer_y * dest_stride]);
        merge_line_to_buffer(&i10[(dest_y/2) * src_stride], &i11[(dest_y/2) * src_stride], x, width, &dest_imag[buffer_y * dest_stride]);
        ++dest_y;
        ++buffer_y;
    }
    if (dest_y < y + height)
    {
        merge_line_to_buffer(&r00[(dest_y/2) * src_stride], &r01[(dest_y/2) * src_stride], x, width, &dest_real[buffer_y * dest_stride]);
        merge_line_to_buffer(&i00[(dest_y/2) * src_stride], &i01[(dest_y/2) * src_stride], x, width, &dest_imag[buffer_y * dest_stride]);
    }
    assert (dest_y == y + height);
}
