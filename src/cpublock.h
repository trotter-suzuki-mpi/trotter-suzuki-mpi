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

#ifndef __CPUBLOCK_H
#define __CPUBLOCK_H

#include "trotter.h"
#include <mpi.h>

#define BLOCK_WIDTH 128u
#define BLOCK_HEIGHT 128u

//Helpers
void block_kernel_vertical(size_t start_offset, size_t stride, size_t width, size_t height, double a, double b, double * p_real, double * p_imag);
void block_kernel_horizontal(size_t start_offset, size_t stride, size_t width, size_t height, double a, double b, double * p_real, double * p_imag);

void block_kernel_vertical_imaginary(size_t start_offset, size_t stride, size_t width, size_t height, double a, double b, double * p_real, double * p_imag);
void block_kernel_horizontal_imaginary(size_t start_offset, size_t stride, size_t width, size_t height, double a, double b, double * p_real, double * p_imag);

void process_sides(size_t tile_width, size_t block_width, size_t halo_x, size_t read_y, size_t read_height, size_t write_offset, size_t write_height, double a, double b, const double *external_pot_real, const double *external_pot_imag, const double * p_real, const double * p_imag, double * next_real, double * next_imag, double * block_real, double * block_imag, bool imag_time);
void process_band(size_t tile_width, size_t block_width, size_t block_height, size_t halo_x, size_t read_y, size_t read_height, size_t write_offset, size_t write_height, double a, double b, const double *external_pot_real, const double *external_pot_imag, const double * p_real, const double * p_imag, double * next_real, double * next_imag, int inner, int sides, bool imag_time);

class CPUBlock: public ITrotterKernel {
public:
    CPUBlock(double *p_real, double *p_imag, double *_external_pot_real, double *_external_pot_imag, double a, double b, int matrix_width, int matrix_height, int halo_x, int halo_y, int *periods, MPI_Comm cartcomm, bool imag_time);
    ~CPUBlock();
    void run_kernel();
    void run_kernel_on_halo();
    void wait_for_completion(int iteration, int snapshosts);
    void get_sample(size_t dest_stride, size_t x, size_t y, size_t width, size_t height, double * dest_real, double * dest_imag) const;

    bool runs_in_place() const {
        return false;
    }
    std::string get_name() const {
        return "CPU";
    };

    void start_halo_exchange();
    void finish_halo_exchange();



private:
    void kernel8(const double *p_real, const double *p_imag, double * next_real, double * next_imag);
    bool imag_time;
    double *p_real[2];
    double *p_imag[2];
    double *external_pot_real;
    double *external_pot_imag;
    double a;
    double b;
    int sense;
    size_t halo_x, halo_y, tile_width, tile_height;
    static const size_t block_width = BLOCK_WIDTH;
    static const size_t block_height = BLOCK_HEIGHT;

    MPI_Comm cartcomm;
    int neighbors[4];
    int start_x, inner_end_x, start_y, inner_start_y,  inner_end_y;
    MPI_Request req[8];
    MPI_Status statuses[8];
    MPI_Datatype horizontalBorder, verticalBorder;
};

#endif
