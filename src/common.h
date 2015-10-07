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

#ifndef __COMMON_H
#define __COMMON_H

#include <string>
#include <complex>

#if HAVE_CONFIG_H
#include "config.h"
#endif
#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include "trotter.h"

void calculate_borders(int coord, int dim, int * start, int *end, int *inner_start, int *inner_end, int length, int halo, int periodic_bound);
void print_complex_matrix(char * filename, double * matrix_real, double * matrix_imag, size_t stride, size_t width, size_t height);
void print_matrix(const char * filename, double * matrix, size_t stride, size_t width, size_t height);
void memcpy2D(void * dst, size_t dstride, const void * src, size_t sstride, size_t width, size_t height);
void get_quadrant_sample(const double * r00, const double * r01, const double * r10, const double * r11,
                         const double * i00, const double * i01, const double * i10, const double * i11,
                         size_t src_stride, size_t dest_stride,
                         size_t x, size_t y, size_t width, size_t height,
                         double * dest_real, double * dest_imag);
void get_quadrant_sample_to_buffer(const double * r00, const double * r01, const double * r10, const double * r11,
                                   const double * i00, const double * i01, const double * i10, const double * i11,
                                   size_t src_stride, size_t dest_stride,
                                   size_t x, size_t y, size_t width, size_t height,
                                   double * dest_real, double * dest_imag);

void expect_values(int dimx, int dimy, double delta_x, double delta_y, double delta_t, double coupling_const, int iterations, int snapshots, double * hamilt_pot, double particle_mass,
                   const char *dirname, int *periods, int halo_x, int halo_y, energy_momentum_statistics *sample);

void initialize_state(double * p_real, double * p_imag, char * filename, std::complex<double> (*ini_state)(int x, int y, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y),
                      int tile_width, int tile_height, int matrix_width, int matrix_height, int start_x, int start_y,
                      int * periods, int * coords, int * dims, int halo_x, int halo_y, int read_offset = 0);
void initialize_exp_potential(double * external_pot_real, double * external_pot_imag, char * pot_name, double (*hamilt_pot)(int x, int y, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y),
                              int tile_width, int tile_height, int matrix_width, int matrix_height, int start_x, int start_y,
                              int * periods, int * coords, int * dims, int halo_x, int halo_y, double time_single_it, double particle_mass, bool imag_time);
void initialize_potential(double * hamilt_pot, double (*hamiltonian_pot)(int x, int y, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y),
                          int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y);

double const_potential(int x, int y, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y);
void stamp(double * p_real, double * p_imag, int matrix_width, int matrix_height, int halo_x, int halo_y, int start_x, int inner_start_x, int inner_end_x, int end_x,
           int start_y, int inner_start_y, int inner_end_y, int * dims, int * coords, int * periods,
           int tag_particle, int iterations, int count_snap, const char * output_folder
#ifdef HAVE_MPI
           , MPI_Comm cartcomm
#endif
          );
#endif
