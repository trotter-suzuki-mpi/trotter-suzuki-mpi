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
 
 //added delta_x, delta_y, coupling_const

#ifndef __CPUBLOCK_H
#define __CPUBLOCK_H

#if HAVE_CONFIG_H
#include "config.h"
#endif
#include "kernel.h"
#ifdef HAVE_MPI
#include <mpi.h>
#endif

#define BLOCK_WIDTH 128u
#define BLOCK_HEIGHT 128u

//Helpers
void block_kernel_vertical(size_t start_offset, size_t stride, size_t width, size_t height, double a, double b, double * p_real, double * p_imag);
void block_kernel_horizontal(size_t start_offset, size_t stride, size_t width, size_t height, double a, double b, double * p_real, double * p_imag);

void block_kernel_vertical_imaginary(size_t start_offset, size_t stride, size_t width, size_t height, double a, double b, double * p_real, double * p_imag);
void block_kernel_horizontal_imaginary(size_t start_offset, size_t stride, size_t width, size_t height, double a, double b, double * p_real, double * p_imag);

void process_sides(int offset_tile_x, int offset_tile_y, double alpha_x, double alpha_y, size_t tile_width, size_t block_width, size_t halo_x, size_t read_y, size_t read_height, size_t write_offset, size_t write_height, double a, double b, double coupling_const, const double *external_pot_real, const double *external_pot_imag, const double * p_real, const double * p_imag, double * next_real, double * next_imag, double * block_real, double * block_imag, bool imag_time);
void process_band(int offset_tile_x, int offset_tile_y, double alpha_x, double alpha_y, size_t tile_width, size_t block_width, size_t block_height, size_t halo_x, size_t read_y, size_t read_height, size_t write_offset, size_t write_height, double a, double b, double coupling_const, const double *external_pot_real, const double *external_pot_imag, const double * p_real, const double * p_imag, double * next_real, double * next_imag, int inner, int sides, bool imag_time);

/**
 * \brief This class define the CPU kernel.
 *
 * This kernel provides real time and imaginary time evolution exploiting CPUs.
 * It implements a solver for a single wave function, whose evolution is governed by nonlinear Schrodinger equation. The Hamiltonian of the physical system includes:
 *  - static external potential
 *  - density self-interacting term
 *  - rotational energy
 */

class CPUBlock: public ITrotterKernel {
public:
    CPUBlock(double *_p_real, double *_p_imag, double *_external_pot_real, double *_external_pot_imag, double _a, double _b, double _coupling_const, double _delta_x, double _delta_y, int matrix_width, int matrix_height, int _halo_x, int _halo_y, int *_periods, double _norm, bool _imag_time, double _alpha_x, double _alpha_y, int _rot_coord_x, int _rot_coord_y
#ifdef HAVE_MPI
             , MPI_Comm cartcomm
#endif
            );
            
    CPUBlock(double *_p_real, double *_p_imag, double *_external_pot_real, double *_external_pot_imag, double _a, double _b, double _coupling_const, double _delta_x, double _delta_y, int matrix_width, int matrix_height, int _halo_x, int _halo_y, int *_periods, double _norm, bool _imag_time, double _alpha_x, double _alpha_y, int _rot_coord_x, int _rot_coord_y
#ifdef HAVE_MPI
             , MPI_Comm cartcomm
#endif
            );
    
    CPUBlock(double **_p_real, double **_p_imag, double **_external_pot_real, double **_external_pot_imag, double *_a, double *_b, double *_coupling_const, double _delta_x, double _delta_y, 
             int matrix_width, int matrix_height, int _halo_x, int _halo_y, int *_periods, double *_norm, bool _imag_time, double _alpha_x, double _alpha_y, int _rot_coord_x, int _rot_coord_y
#ifdef HAVE_MPI
             , MPI_Comm cartcomm
#endif
            );
    
    ~CPUBlock();
    void run_kernel_on_halo();					///< Evolve blocks of wave function at the edge of the tile. This comprises the halos.
    void run_kernel();							///< Evolve the remaining blocks in the inner part of the tile.
    void wait_for_completion(int iteration);	///< Sincronize all the processes at the end of halos communication. Perform normalization for imaginary time evolution.
    void get_sample(size_t dest_stride, size_t x, size_t y, size_t width, size_t height, double * dest_real, double * dest_imag) const;  ///< Copy the wave function from the two buffers pointed by p_real and p_imag, without halos, to dest_real and dest_imag.
	void normalization();
	void rabi_coupling(double var, double delta_t);
	
    bool runs_in_place() const {
        return false;
    }
    /// Get kernel name.
    std::string get_name() const {
        return "CPU";
    };

    void start_halo_exchange();					///< Start vertical halos exchange.
    void finish_halo_exchange();				///< Start horizontal halos exchange.



private:
    double *p_real[2][2];				///< Array of two pointers that point to two buffers used to store the real part of the wave function at i-th time step and (i+1)-th time step. 
    double *p_imag[2][2];				///< Array of two pointers that point to two buffers used to store the imaginary part of the wave function at i-th time step and (i+1)-th time step.
    double *external_pot_real[2];		///< Points to the matrix representation (real entries) of the operator given by the exponential of external potential.
    double *external_pot_imag[2];		///< Points to the matrix representation (immaginary entries) of the operator given by the exponential of external potential.
    double *a;						///< Diagonal value of the matrix representation of the operator given by the exponential of kinetic operator.
    double *b;						///< Off diagonal value of the matrix representation of the operator given by the exponential of kinetic operator.
    double delta_x;					///< Physical length between two neighbour along x axis dots of the lattice.
    double delta_y;					///< Physical length between two neighbour along y axis dots of the lattice.
    double *norm;					///< Squared norm of the wave function.
    double tot_norm;
    double *coupling_const;			///< Coupling constant of the density self-interacting term.
    int sense;						///< Takes values 0 or 1 and tells which of the two buffers pointed by p_real and p_imag is used to calculate the next time step.
    int state;
    size_t halo_x;					///< Thickness of the vertical halos (number of lattice's dots).
    size_t halo_y;					///< Thickness of the horizontal halos (number of lattice's dots).
    size_t tile_width;				///< Width of the tile (number of lattice's dots).
    size_t tile_height;				///< Height of the tile (number of lattice's dots).
    bool imag_time;					///< True: imaginary time evolution; False: real time evolution.
    static const size_t block_width = BLOCK_WIDTH;			///< Width of the lattice block which is cached (number of lattice's dots).
    static const size_t block_height = BLOCK_HEIGHT;		///< Height of the lattice block which is cached (number of lattice's dots).

	double alpha_x;					///< Real coupling constant associated to the X*P_y operator, part of the angular momentum.
	double alpha_y;					///< Real coupling constant associated to the Y*P_x operator, part of the angular momentum.
	int rot_coord_x;				///< X axis coordinate of the center of rotation.
	int rot_coord_y;				///< Y axis coordinate of the center of rotation.
    int start_x;					///< X axis coordinate of the first dot of the processed tile.
    int start_y;					///< Y axis coordinate of the first dot of the processed tile.
    int end_x;						///< X axis coordinate of the last dot of the processed tile.
    int end_y;						///< Y axis coordinate of the last dot of the processed tile.
    int inner_start_x;				///< X axis coordinate of the first dot of the processed tile, which is not in the halo.
    int inner_start_y;				///< Y axis coordinate of the first dot of the processed tile, which is not in the halo.
    int inner_end_x;				///< X axis coordinate of the last dot of the processed tile, which is not in the halo.
    int inner_end_y;				///< Y axis coordinate of the last dot of the processed tile, which is not in the halo.
    int *periods;					///< Two dimensional array which takes entries 0 or 1. 1: periodic boundary condition along the corresponding axis; 0: closed boundary condition along the corresponding axis.
#ifdef HAVE_MPI
    MPI_Comm cartcomm;				///< Ensemble of processes communicating the halos and evolving the tiles.
    int neighbors[4];				///< Array that stores the processes' rank neighbour of the current process.
    MPI_Request req[8];				///< Variable to manage MPI communication.
    MPI_Status statuses[8];			///< Variable to manage MPI communication.
    MPI_Datatype horizontalBorder;	///< Datatype for the horizontal halos.
    MPI_Datatype verticalBorder;	///< Datatype for the vertical halos.
#endif
};

#endif
