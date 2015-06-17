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

/*
 * Initial state functions
 */

std::complex<double> gauss_state(int x, int y, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y) {
    double s = 64.0; // FIXME: y esto?
    return std::complex<double>(exp(-(pow(x - 180.0, 2.0) + pow(y - 300.0, 2.0)) / (2.0 * pow(s, 2.0))), 0.0)
           * exp(std::complex<double>(0.0, 0.4 * (x + y - 480.0)));
}

std::complex<double> sinus_state(int x, int y, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y) {
    double L_x = matrix_width - periods[1] * 2 * halo_x;
    double L_y = matrix_height - periods[0] * 2 * halo_y;

    return std::complex<double> (sin(2 * 3.14159 / L_x * (x - periods[1] * halo_x)) * sin(2 * 3.14159 / L_y * (y - periods[0] * halo_y)), 0.0);
}

std::complex<double> exp_state(int x, int y, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y) {
    double L_x = matrix_width - periods[1] * 2 * halo_x;
    double L_y = matrix_height - periods[0] * 2 * halo_y;

    return exp(std::complex<double>(0. , 2 * 3.14159 / L_x * (x - periods[1] * halo_x) + 2 * 3.14159 / L_y * (y - periods[0] * halo_y) ));
}

std::complex<double> super_position_two_exp_state(int x, int y, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y) {
    double L_x = matrix_width - periods[1] * 2 * halo_x;
    double L_y = matrix_height - periods[0] * 2 * halo_y;

    return exp(std::complex<double>(0. , 2. * 3.14159 / L_x * (x - periods[1] * halo_x))) +
           exp(std::complex<double>(0. , 10. * 2. * 3.14159 / L_x * (x - periods[1] * halo_x)));
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

void expect_values(int dim, int iterations, int snapshots, double * hamilt_pot, double particle_mass,
                   const char *dirname, int *periods, int halo_x, int halo_y, energy_momentum_statistics *sample) {

    if(snapshots == 0)
        return;

    int N_files = (int)ceil(double(iterations) / double(snapshots));
    int N_name[N_files];

    N_name[0] = 0;
    for(int i = 1; i < N_files; i++) {
        N_name[i] = N_name[i - 1] + snapshots;
    }

    std::complex<double> sum_E = 0;
    std::complex<double> sum_Px = 0, sum_Py = 0;
    std::complex<double> sum_pdi = 0;
    double energy[N_files], momentum_x[N_files], momentum_y[N_files];

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
