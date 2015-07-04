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
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <stdlib.h>
#include <string.h>

#if HAVE_CONFIG_H
#include "config.h"
#endif
#ifdef HAVE_MPI
#include <mpi.h>
#endif

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

void read_initial_state(double * p_real, double * p_imag, int tile_width, int tile_height,
                        char * file_name, int matrix_width, int matrix_height, int start_x, int start_y,
                        int * periods, int * coords, int * dims, int halo_x, int halo_y, int read_offset) {
    std::ifstream input(file_name);

    int in_width = matrix_width - 2 * periods[1] * halo_x;
    int in_height = matrix_height - 2 * periods[0] * halo_y;
    std::complex<double> tmp;
    for(int i = 0; i < read_offset; i++)
        input >> tmp;
    for(int i = 0; i < in_height; i++) {
        for(int j = 0; j < in_width; j++) {
            input >> tmp;
            if((i - start_y) >= 0 && (i - start_y) < tile_height && (j - start_x) >= 0 && (j - start_x) < tile_width) {
                p_real[(i - start_y) * tile_width + j - start_x] = real(tmp);
                p_imag[(i - start_y) * tile_width + j - start_x] = imag(tmp);
            }

            //Down band
            if(i < halo_y && coords[0] == dims[0] - 1 && periods[0] != 0) {
                if((j - start_x) >= 0 && (j - start_x) < tile_width) {
                    p_real[(i + tile_height - halo_y) * tile_width + j - start_x] = real(tmp);
                    p_imag[(i + tile_height - halo_y) * tile_width + j - start_x] = imag(tmp);
                }
                //Down right corner
                if(j < halo_x && periods[1] != 0 && coords[1] == dims[1] - 1) {
                    p_real[(i + tile_height - halo_y) * tile_width + j + tile_width - halo_x] = real(tmp);
                    p_imag[(i + tile_height - halo_y) * tile_width + j + tile_width - halo_x] = imag(tmp);
                }
                //Down left corner
                if(j >= in_width - halo_x && periods[1] != 0 && coords[1] == 0) {
                    p_real[(i + tile_height - halo_y) * tile_width + j - (in_width - halo_x)] = real(tmp);
                    p_imag[(i + tile_height - halo_y) * tile_width + j - (in_width - halo_x)] = imag(tmp);
                }
            }

            //Upper band
            if(i >= in_height - halo_y && periods[0] != 0 && coords[0] == 0) {
                if((j - start_x) >= 0 && (j - start_x) < tile_width) {
                    p_real[(i - (in_height - halo_y)) * tile_width + j - start_x] = real(tmp);
                    p_imag[(i - (in_height - halo_y)) * tile_width + j - start_x] = imag(tmp);
                }
                //Up right corner
                if(j < halo_x && periods[1] != 0 && coords[1] == dims[1] - 1) {
                    p_real[(i - (in_height - halo_y)) * tile_width + j + tile_width - halo_x] = real(tmp);
                    p_imag[(i - (in_height - halo_y)) * tile_width + j + tile_width - halo_x] = imag(tmp);
                }
                //Up left corner
                if(j >= in_width - halo_x && periods[1] != 0 && coords[1] == 0) {
                    p_real[(i - (in_height - halo_y)) * tile_width + j - (in_width - halo_x)] = real(tmp);
                    p_imag[(i - (in_height - halo_y)) * tile_width + j - (in_width - halo_x)] = imag(tmp);
                }
            }

            //Right band
            if(j < halo_x && periods[1] != 0 && coords[1] == dims[1] - 1) {
                if((i - start_y) >= 0 && (i - start_y) < tile_height) {
                    p_real[(i - start_y) * tile_width + j + tile_width - halo_x] = real(tmp);
                    p_imag[(i - start_y) * tile_width + j + tile_width - halo_x] = imag(tmp);
                }
            }

            //Left band
            if(j >= in_width - halo_x && periods[1] != 0 && coords[1] == 0) {
                if((i - start_y) >= 0 && (i - start_y) < tile_height) {
                    p_real[(i - start_y) * tile_width + j - (in_width - halo_x)] = real(tmp);
                    p_imag[(i - start_y) * tile_width + j - (in_width - halo_x)] = imag(tmp);
                }
            }
        }
    }
    input.close();
}

void read_potential(double * external_pot_real, double * external_pot_imag, int tile_width, int tile_height,
                    char * pot_name, int matrix_width, int matrix_height, int start_x, int start_y,
                    int * periods, int * coords, int * dims, int halo_x, int halo_y, double time_single_it, double particle_mass, bool imag_time) {
    std::ifstream input(pot_name);

    int in_width = matrix_width - 2 * periods[1] * halo_x;
    int in_height = matrix_height - 2 * periods[0] * halo_y;
    std::complex<double> tmp;
    double read;
    double order_approx = 2.;
    double CONST_1 = -1. * time_single_it * order_approx;
    double CONST_2 = 2. * time_single_it / particle_mass * order_approx;		//CONST_2: discretization of momentum operator and the only effect is to produce a scalar operator, so it could be omitted

    for(int i = 0; i < in_height; i++) {
        for(int j = 0; j < in_width; j++) {
            input >> read;
            if(imag_time)
                tmp = exp(std::complex<double> (CONST_1 * read , CONST_2));
            else
                tmp = exp(std::complex<double> (0., CONST_1 * read + CONST_2));

            if((i - start_y) >= 0 && (i - start_y) < tile_height && (j - start_x) >= 0 && (j - start_x) < tile_width) {
                external_pot_real[(i - start_y) * tile_width + j - start_x] = real(tmp);
                external_pot_imag[(i - start_y) * tile_width + j - start_x] = imag(tmp);
            }

            //Down band
            if(i < halo_y && coords[0] == dims[0] - 1 && periods[0] != 0) {
                if((j - start_x) >= 0 && (j - start_x) < tile_width) {
                    external_pot_real[(i + tile_height - halo_y) * tile_width + j - start_x] = real(tmp);
                    external_pot_imag[(i + tile_height - halo_y) * tile_width + j - start_x] = imag(tmp);
                }
                //Down right corner
                if(j < halo_x && periods[1] != 0 && coords[1] == dims[1] - 1) {
                    external_pot_real[(i + tile_height - halo_y) * tile_width + j + tile_width - halo_x] = real(tmp);
                    external_pot_imag[(i + tile_height - halo_y) * tile_width + j + tile_width - halo_x] = imag(tmp);
                }
                //Down left corner
                if(j >= in_width - halo_x && periods[1] != 0 && coords[1] == 0) {
                    external_pot_real[(i + tile_height - halo_y) * tile_width + j - (in_width - halo_x)] = real(tmp);
                    external_pot_imag[(i + tile_height - halo_y) * tile_width + j - (in_width - halo_x)] = imag(tmp);
                }
            }

            //Upper band
            if(i >= in_height - halo_y && periods[0] != 0 && coords[0] == 0) {
                if((j - start_x) >= 0 && (j - start_x) < tile_width) {
                    external_pot_real[(i - (in_height - halo_y)) * tile_width + j - start_x] = real(tmp);
                    external_pot_imag[(i - (in_height - halo_y)) * tile_width + j - start_x] = imag(tmp);
                }
                //Up right corner
                if(j < halo_x && periods[1] != 0 && coords[1] == dims[1] - 1) {
                    external_pot_real[(i - (in_height - halo_y)) * tile_width + j + tile_width - halo_x] = real(tmp);
                    external_pot_imag[(i - (in_height - halo_y)) * tile_width + j + tile_width - halo_x] = imag(tmp);
                }
                //Up left corner
                if(j >= in_width - halo_x && periods[1] != 0 && coords[1] == 0) {
                    external_pot_real[(i - (in_height - halo_y)) * tile_width + j - (in_width - halo_x)] = real(tmp);
                    external_pot_imag[(i - (in_height - halo_y)) * tile_width + j - (in_width - halo_x)] = imag(tmp);
                }
            }

            //Right band
            if(j < halo_x && periods[1] != 0 && coords[1] == dims[1] - 1) {
                if((i - start_y) >= 0 && (i - start_y) < tile_height) {
                    external_pot_real[(i - start_y) * tile_width + j + tile_width - halo_x] = real(tmp);
                    external_pot_imag[(i - start_y) * tile_width + j + tile_width - halo_x] = imag(tmp);
                }
            }

            //Left band
            if(j >= in_width - halo_x && periods[1] != 0 && coords[1] == 0) {
                if((i - start_y) >= 0 && (i - start_y) < tile_height) {
                    external_pot_real[(i - start_y) * tile_width + j - (in_width - halo_x)] = real(tmp);
                    external_pot_imag[(i - start_y) * tile_width + j - (in_width - halo_x)] = imag(tmp);
                }
            }
        }
    }
    input.close();
}

void initialize_state(double * p_real, double * p_imag, char * filename, std::complex<double> (*ini_state)(int x, int y, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y),
                      int tile_width, int tile_height, int matrix_width, int matrix_height, int start_x, int start_y,
                      int * periods, int * coords, int * dims, int halo_x, int halo_y, int read_offset) {
    if(filename[0] != '\0')
        read_initial_state(p_real, p_imag, tile_width, tile_height, filename, matrix_width, matrix_height, start_x, start_y, periods, coords, dims, halo_x, halo_y, read_offset);
    else if(ini_state != NULL) {
        std::complex<double> tmp;
        for (int y = 0, idy = start_y; y < tile_height; y++, idy++) {
            for (int x = 0, idx = start_x; x < tile_width; x++, idx++) {
                tmp = ini_state(idx, idy, matrix_width, matrix_height, periods, halo_x, halo_y);
                p_real[y * tile_width + x] = real(tmp);
                p_imag[y * tile_width + x] = imag(tmp);
            }
        }
    }
}

double const_potential(int x, int y, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y) {
    return 0.;
}

void initialize_exp_potential(double * external_pot_real, double * external_pot_imag, char * pot_name, double (*hamilt_pot)(int x, int y, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y),
                              int tile_width, int tile_height, int matrix_width, int matrix_height, int start_x, int start_y,
                              int * periods, int * coords, int * dims, int halo_x, int halo_y, double time_single_it, double particle_mass, bool imag_time) {
    if(pot_name[0] != '\0')
        read_potential(external_pot_real, external_pot_imag, tile_width, tile_height, pot_name, matrix_width, matrix_height, start_x, start_y, periods, coords, dims, halo_x, halo_y, time_single_it, particle_mass, imag_time);
    else if(hamilt_pot != NULL) {
        double order_approx = 2.;
        double CONST_1 = -1. * time_single_it * order_approx;
        double CONST_2 = 2. * time_single_it / particle_mass * order_approx;		//CONST_2: discretization of momentum operator and the only effect is to produce a scalar operator, so it could be omitted

        std::complex<double> tmp;
        for (int y = 0, idy = start_y; y < tile_height; y++, idy++) {
            for (int x = 0, idx = start_x; x < tile_width; x++, idx++) {
                if(imag_time)
                    tmp = exp(std::complex<double> (CONST_1 * hamilt_pot(idx, idy, matrix_width, matrix_height, periods, halo_x, halo_y) , CONST_2));
                else
                    tmp = exp(std::complex<double> (0., CONST_1 * hamilt_pot(idx, idy, matrix_width, matrix_height, periods, halo_x, halo_y) + CONST_2));
                external_pot_real[y * tile_width + x] = real(tmp);
                external_pot_imag[y * tile_width + x] = imag(tmp);
            }
        }
    }
}

void initialize_potential(double * hamilt_pot, double (*hamiltonian_pot)(int x, int y, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y),
                          int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y) {
    for(int y = 0; y < matrix_height; y++) {
        for(int x = 0; x < matrix_width; x++) {
            hamilt_pot[y * matrix_width + x] = 0.;//hamiltonian_pot(x, y, matrix_width, matrix_height, periods, halo_x, halo_y);
        }
    }
}

void print_complex_matrix(char * filename, double * matrix_real, double * matrix_imag, size_t stride, size_t width, size_t height) {
    std::ofstream out(filename, std::ios::out | std::ios::trunc);
    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            out << "(" << matrix_real[i * stride + j] << "," << matrix_imag[i * stride + j] << ") ";
        }
        out << std::endl;
    }
    out.close();
}

void print_matrix(char * filename, double * matrix, size_t stride, size_t width, size_t height) {
    std::ofstream out(filename, std::ios::out | std::ios::trunc);
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

void merge_line(const double * evens, const double * odds, size_t x, size_t width, double * dest) {

    const double * odds_p = odds + (x / 2);
    const double * evens_p = evens + (x / 2);

    size_t dest_x = x;
    if (x % 2 == 1) {
        dest[dest_x++ - x] = *(odds_p++);
    }
    while (dest_x < (x + width) - (x + width) % 2) {
        dest[dest_x++ - x] = *(evens_p++);
        dest[dest_x++ - x] = *(odds_p++);
    }
    if (dest_x < x + width) {
        dest[dest_x++ - x] = *evens_p;
    }
    assert(dest_x == x + width);
}


void get_quadrant_sample(const double * r00, const double * r01, const double * r10, const double * r11,
                         const double * i00, const double * i01, const double * i10, const double * i11,
                         size_t src_stride, size_t dest_stride,
                         size_t x, size_t y, size_t width, size_t height,
                         double * dest_real, double * dest_imag) {
    size_t dest_y = y;
    if (y % 2 == 1) {
        merge_line(&r10[(y / 2) * src_stride], &r11[(y / 2) * src_stride], x, width, dest_real);
        merge_line(&i10[(y / 2) * src_stride], &i11[(y / 2) * src_stride], x, width, dest_imag);
        ++dest_y;
    }
    while (dest_y < (y + height) - (y + height) % 2) {
        merge_line(&r00[(dest_y / 2) * src_stride], &r01[(dest_y / 2) * src_stride], x, width, &dest_real[(dest_y - y) * dest_stride]);
        merge_line(&i00[(dest_y / 2) * src_stride], &i01[(dest_y / 2) * src_stride], x, width, &dest_imag[(dest_y - y) * dest_stride]);
        ++dest_y;
        merge_line(&r10[(dest_y / 2) * src_stride], &r11[(dest_y / 2) * src_stride], x, width, &dest_real[(dest_y - y) * dest_stride]);
        merge_line(&i10[(dest_y / 2) * src_stride], &i11[(dest_y / 2) * src_stride], x, width, &dest_imag[(dest_y - y) * dest_stride]);
        ++dest_y;
    }
    if (dest_y < y + height) {
        merge_line(&r00[(dest_y / 2) * src_stride], &r01[(dest_y / 2) * src_stride], x, width, &dest_real[(dest_y - y) * dest_stride]);
        merge_line(&i00[(dest_y / 2) * src_stride], &i01[(dest_y / 2) * src_stride], x, width, &dest_imag[(dest_y - y) * dest_stride]);
    }
    assert (dest_y == y + height);
}

void merge_line_to_buffer(const double * evens, const double * odds, size_t x, size_t width, double * dest) {

    const double * odds_p = odds + (x / 2);
    const double * evens_p = evens + (x / 2);

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


void get_quadrant_sample_to_buffer(const double * r00, const double * r01, const double * r10, const double * r11,
                                   const double * i00, const double * i01, const double * i10, const double * i11,
                                   size_t src_stride, size_t dest_stride,
                                   size_t x, size_t y, size_t width, size_t height,
                                   double * dest_real, double * dest_imag) {
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

void stamp(double * p_real, double * p_imag, int matrix_width, int matrix_height, int halo_x, int halo_y, int start_x, int inner_start_x, int inner_end_x,
           int start_y, int inner_start_y, int inner_end_y, int * dims, int * coords, int * periods,
           int tag_particle, int iterations, int count_snap, const char * output_folder
#ifdef HAVE_MPI
           , MPI_Comm cartcomm
#endif
          ) {

    char * output_filename;
    output_filename = new char[51];
#ifdef HAVE_MPI
    // Set variables for mpi output
    char *data_as_txt;
    int count;

    MPI_File   file;
    MPI_Status status;

    // each number is represented by charspernum chars
    const int chars_per_complex_num = 28;
    MPI_Datatype complex_num_as_string;
    MPI_Type_contiguous(chars_per_complex_num, MPI_CHAR, &complex_num_as_string);
    MPI_Type_commit(&complex_num_as_string);

    const int charspernum = 14;
    MPI_Datatype num_as_string;
    MPI_Type_contiguous(charspernum, MPI_CHAR, &num_as_string);
    MPI_Type_commit(&num_as_string);

    // create a type describing our piece of the array
    int globalsizes[2] = {matrix_height - 2 * periods[0] * halo_y, matrix_width - 2 * periods[1] * halo_x};
    int localsizes [2] = {inner_end_y - inner_start_y, inner_end_x - inner_start_x};
    int starts[2]      = {inner_start_y, inner_start_x};
    int order          = MPI_ORDER_C;

    MPI_Datatype complex_localarray;
    MPI_Type_create_subarray(2, globalsizes, localsizes, starts, order, complex_num_as_string, &complex_localarray);
    MPI_Type_commit(&complex_localarray);

    MPI_Datatype localarray;
    MPI_Type_create_subarray(2, globalsizes, localsizes, starts, order, num_as_string, &localarray);
    MPI_Type_commit(&localarray);

    // output complex matrix
    // conversion
    data_as_txt = new char[(inner_end_x - inner_start_x) * (inner_end_y - inner_start_y) * chars_per_complex_num];
    count = 0;
    for (int j = inner_start_y - start_y; j < inner_end_y - start_y; j++) {
        for (int k = inner_start_x - start_x; k < inner_end_x - start_x - 1; k++) {
            sprintf(&data_as_txt[count * chars_per_complex_num], "(%+.5e,%+.5e) ", p_real[j * matrix_width + k], p_imag[j * matrix_width + k]);
            count++;
        }
        if(coords[1] == dims[1] - 1) {
            sprintf(&data_as_txt[count * chars_per_complex_num], "(%+.5e,%+.5e)\n", p_real[j * matrix_width + (inner_end_x - start_x) - 1], p_imag[j * matrix_width + (inner_end_x - start_x) - 1]);
            count++;
        }
        else {
            sprintf(&data_as_txt[count * chars_per_complex_num], "(%+.5e,%+.5e) ", p_real[j * matrix_width + (inner_end_x - start_x) - 1], p_imag[j * matrix_width + (inner_end_x - start_x) - 1]);
            count++;
        }
    }

    // open the file, and set the view
    sprintf(output_filename, "%s/%i-%i-iter-comp.dat", output_folder, tag_particle + 1, iterations * count_snap);
    MPI_File_open(cartcomm, output_filename,
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &file);

    MPI_File_set_view(file, 0,  MPI_CHAR, complex_localarray, "native", MPI_INFO_NULL);

    MPI_File_write_all(file, data_as_txt, (inner_end_x - inner_start_x) * (inner_end_y - inner_start_y), complex_num_as_string, &status);
    MPI_File_close(&file);
    delete [] data_as_txt;

    // output real matrix
    //conversion
    data_as_txt = new char[(inner_end_x - inner_start_x) * (inner_end_y - inner_start_y) * charspernum];
    count = 0;
    for (int j = inner_start_y - start_y; j < inner_end_y - start_y; j++) {
        for (int k = inner_start_x - start_x; k < inner_end_x - start_x - 1; k++) {
            sprintf(&data_as_txt[count * charspernum], "%+.6e ", p_real[j * matrix_width + k]);
            count++;
        }
        if(coords[1] == dims[1] - 1) {
            sprintf(&data_as_txt[count * charspernum], "%+.6e\n", p_real[j * matrix_width + (inner_end_x - start_x) - 1]);
            count++;
        }
        else {
            sprintf(&data_as_txt[count * charspernum], "%+.6e ", p_real[j * matrix_width + (inner_end_x - start_x) - 1]);
            count++;
        }
    }

    // open the file, and set the view
    sprintf(output_filename, "%s/%i-%i-iter-real.dat", output_folder, tag_particle + 1, iterations * count_snap);
    MPI_File_open(cartcomm, output_filename,
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &file);

    MPI_File_set_view(file, 0,  MPI_CHAR, localarray, "native", MPI_INFO_NULL);

    MPI_File_write_all(file, data_as_txt, (inner_end_x - inner_start_x) * ( inner_end_y - inner_start_y), num_as_string, &status);
    MPI_File_close(&file);
    delete [] data_as_txt;
#else
    sprintf(output_filename, "%s/%i-%i-iter-real.dat", output_folder, tag_particle + 1, iterations * count_snap);
    print_matrix(output_filename, &(p_real[matrix_width * (inner_start_y - start_y) + inner_start_x - start_x]), matrix_width,
                 matrix_width - 2 * periods[1]*halo_x, matrix_height - 2 * periods[0]*halo_y);

    sprintf(output_filename, "%s/%i-%i-iter-comp.dat", output_folder, tag_particle + 1, iterations * count_snap);
    print_complex_matrix(output_filename, &(p_real[matrix_width * (inner_start_y - start_y) + inner_start_x - start_x]), &(p_imag[matrix_width * (inner_start_y - start_y) + inner_start_x - start_x]), matrix_width,
                         matrix_width - 2 * periods[1]*halo_x, matrix_height - 2 * periods[0]*halo_y);
#endif
    return;
}

void expect_values(int dim, int iterations, int snapshots, double * hamilt_pot, double particle_mass,
                   const char *dirname, int *periods, int halo_x, int halo_y, energy_momentum_statistics *sample) {

    if(snapshots == 0)
        return;

    int N_files = snapshots + 1;
    int *N_name = new int[N_files];

    N_name[0] = 0;
    for(int i = 1; i < N_files; i++) {
        N_name[i] = N_name[i - 1] + iterations;
    }

    std::complex<double> sum_E = 0;
    std::complex<double> sum_Px = 0, sum_Py = 0;
    std::complex<double> sum_pdi = 0;
    double *energy = new double[N_files];
    double *momentum_x = new double[N_files];
    double *momentum_y = new double[N_files];

    std::complex<double> cost_E = -1. / (2.*particle_mass), cost_P;
    cost_P = std::complex<double>(0., -0.5);

    std::stringstream filename;
    std::string filenames;

    filename.str("");
    filename << dirname << "/exp_val_D" << dim << "_I" << iterations << "_S" << snapshots << ".dat";
    filenames = filename.str();
    std::ofstream out(filenames.c_str());

    out << "#time\tEnergy\t\tPx\tPy\tP**2\tnorm(psi(t))" << std::endl;
    for(int i = 0; i < N_files; i++) {

        filename.str("");
        filename << dirname << "/" << "1-" << N_name[i] << "-iter-comp.dat";
        filenames = filename.str();
        std::ifstream up(filenames.c_str()), center(filenames.c_str()), down(filenames.c_str());
        std::complex<double> psi_up, psi_down, psi_center, psi_left, psi_right;

        for (int j = 0; j < dim; j++)
            center >> psi_center;
        for (int j = 0; j < 2 * dim; j++)
            down >> psi_center;
        up >> psi_up;
        down >> psi_down;
        center >> psi_left >> psi_center;
        for (int j = 1; j < dim - 1; j++) {
            if(j != 1) {
                up >> psi_up >> psi_up;
                down >> psi_down >> psi_down;
                center >> psi_left >> psi_center;
            }
            for (int k = 1; k < dim - 1; k++) {
                up >> psi_up;
                center >> psi_right;
                down >> psi_down;

                sum_E += conj(psi_center) * (cost_E * (psi_right + psi_left + psi_down + psi_up - psi_center * std::complex<double> (4., 0.)) + psi_center * std::complex<double> (hamilt_pot[j * dim + k], 0.)) ;
                sum_Px += conj(psi_center) * (psi_right - psi_left);
                sum_Py += conj(psi_center) * (psi_down - psi_up);
                sum_pdi += conj(psi_center) * psi_center;

                psi_left = psi_center;
                psi_center = psi_right;
            }

        }
        up.close();
        center.close();
        down.close();

        out << N_name[i] << "\t" << real(sum_E / sum_pdi) << "\t" << real(cost_P * sum_Px / sum_pdi) << "\t" << real(cost_P * sum_Py / sum_pdi) << "\t"
            << real(cost_P * sum_Px / sum_pdi)*real(cost_P * sum_Px / sum_pdi) + real(cost_P * sum_Py / sum_pdi)*real(cost_P * sum_Py / sum_pdi) << "\t" << real(sum_pdi) << std::endl;

        energy[i] = real(sum_E / sum_pdi);
        momentum_x[i] = real(cost_P * sum_Px / sum_pdi);
        momentum_y[i] = real(cost_P * sum_Py / sum_pdi);

        sum_E = 0;
        sum_Px = 0;
        sum_Py = 0;
        sum_pdi = 0;
    }

    //calculate sample mean and sample variance
    for(int i = 0; i < N_files; i++) {
        sample->mean_E += energy[i];
        sample->mean_Px += momentum_x[i];
        sample->mean_Py += momentum_y[i];
    }
    sample->mean_E /= N_files;
    sample->mean_Px /= N_files;
    sample->mean_Py /= N_files;

    for(int i = 0; i < N_files; i++) {
        sample->var_E += (energy[i] - sample->mean_E) * (energy[i] - sample->mean_E);
        sample->var_Px += (momentum_x[i] - sample->mean_Px) * (momentum_x[i] - sample->mean_Px);
        sample->var_Py += (momentum_y[i] - sample->mean_Py) * (momentum_y[i] - sample->mean_Py);
    }
    sample->var_E /= N_files - 1;
    sample->var_E = sqrt(sample->var_E);
    sample->var_Px /= N_files - 1;
    sample->var_Px = sqrt(sample->var_Px);
    sample->var_Py /= N_files - 1;
    sample->var_Py = sqrt(sample->var_Py);
}
