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

#include <cstring>
#include <string>
#include <sstream>
#include <sys/time.h>
#include <mpi.h>

#include "common.h"
#include "trotter.h"
#include "cpublock.h"
#include "cpublocksse.h"
#ifdef CUDA
#include "cc2kernel.h"
#include "hybrid.h"
#endif

void trotter(double h_a, double h_b,
             double * external_pot_real, double * external_pot_imag,
             double * p_real, double * p_imag, const int matrix_width,
             const int matrix_height, const int iterations, const int snapshots, const int kernel_type,
             int *periods, const char *output_folder, bool verbose, 
             bool imag_time, int particle_tag) {

    
    int start_x, end_x, inner_start_x, inner_end_x,
        start_y, end_y, inner_start_y, inner_end_y;

    int coords[2], dims[2] = {0, 0};
    int rank;
    int nProcs;

    MPI_Comm cartcomm;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    MPI_Dims_create(nProcs, 2, dims);  //partition all the processes (the size of MPI_COMM_WORLD's group) into an 2-dimensional topology
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cartcomm);
    MPI_Comm_rank(cartcomm, &rank);
    MPI_Cart_coords(cartcomm, rank, 2, coords); //Determines process coords in cartesian topology given rank in group

    int halo_x = (kernel_type == 2 ? 3 : 4);
    int halo_y = 4;
    calculate_borders(coords[1], dims[1], &start_x, &end_x, &inner_start_x, &inner_end_x, matrix_width - 2 * periods[1]*halo_x, halo_x, periods[1]);
    calculate_borders(coords[0], dims[0], &start_y, &end_y, &inner_start_y, &inner_end_y, matrix_height - 2 * periods[0]*halo_y, halo_y, periods[0]);
    int width = end_x - start_x;
    int height = end_y - start_y;
    
    // Set variables for mpi output
    char *data_as_txt, *filename;
    filename = new char[strlen(output_folder) + 50];
    double *_p_real, *_p_imag;
    int count;
    
    MPI_File   file;
    MPI_Status status;
    
    // each number is represented by charspernum chars 
    const int chars_per_complex_num=30;
    MPI_Datatype complex_num_as_string;
    MPI_Type_contiguous(chars_per_complex_num, MPI_CHAR, &complex_num_as_string); 
    MPI_Type_commit(&complex_num_as_string);
    
    const int charspernum=14;
    MPI_Datatype num_as_string;
    MPI_Type_contiguous(charspernum, MPI_CHAR, &num_as_string); 
    MPI_Type_commit(&num_as_string);
    
    // create a type describing our piece of the array 
    int globalsizes[2] = {matrix_height - 2 * periods[0]*halo_y, matrix_width - 2 * periods[1]*halo_x};
    int localsizes [2] = {inner_end_y - inner_start_y, inner_end_x - inner_start_x};
    int starts[2]      = {inner_start_y, inner_start_x};
    int order          = MPI_ORDER_C;
	
	MPI_Datatype complex_localarray;
    MPI_Type_create_subarray(2, globalsizes, localsizes, starts, order, complex_num_as_string, &complex_localarray);
    MPI_Type_commit(&complex_localarray);
    
    MPI_Datatype localarray;
    MPI_Type_create_subarray(2, globalsizes, localsizes, starts, order, num_as_string, &localarray);
    MPI_Type_commit(&localarray);
    
#ifdef DEBUG
    std::cout << "Coord_x: " << coords[1] << " start_x: " << start_x << \
              " end_x: " << end_x << " inner_start_x " << inner_start_x << " inner_end_x " << inner_end_x << "\n";
    std::cout << "Coord_y: " << coords[0] << " start_y: " << start_y << \
              " end_y: " << end_y << " inner_start_y " << inner_start_y << " inner_end_y " << inner_end_y << "\n";
#endif

    // Initialize kernel
    ITrotterKernel * kernel;
    switch (kernel_type) {
    case 0:
        kernel = new CPUBlock(p_real, p_imag, external_pot_real, external_pot_imag, h_a, h_b, matrix_width, matrix_height, halo_x, halo_y, periods, cartcomm, imag_time);
        break;

    case 1:
        kernel = new CPUBlockSSEKernel(p_real, p_imag, external_pot_real, external_pot_imag, h_a, h_b, matrix_width, matrix_height, halo_x, halo_y, periods, cartcomm, imag_time);
        break;

    case 2:
#ifdef CUDA
        kernel = new CC2Kernel(p_real, p_imag, external_pot_real, external_pot_imag, h_a, h_b, matrix_width, matrix_height, halo_x, halo_y, periods, cartcomm, imag_time);
#else
        if (coords[0] == 0 && coords[1] == 0) {
            std::cerr << "Compiled without CUDA\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
#endif
        break;

    case 3:
#ifdef CUDA
        kernel = new HybridKernel(p_real, p_imag, external_pot_real, external_pot_imag, h_a, h_b, matrix_width, matrix_height, halo_x, halo_y, periods, cartcomm, imag_time);
#else
        if (coords[0] == 0 && coords[1] == 0) {
            std::cerr << "Compiled without CUDA\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 2);
#endif
        break;

    default:
        kernel = new CPUBlock(p_real, p_imag, external_pot_real, external_pot_imag, h_a, h_b, matrix_width, matrix_height, halo_x, halo_y, periods, cartcomm, imag_time);
        break;
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Main loop
    for (int i = 0; i < iterations; i++) {
        if ( (snapshots > 0) && (i % snapshots == 0) ) {
            
            _p_real = new double[(inner_end_x - inner_start_x) * (inner_end_y - inner_start_y)];
            _p_imag = new double[(inner_end_x - inner_start_x) * (inner_end_y - inner_start_y)];
            kernel->get_sample(inner_end_x - inner_start_x, inner_start_x - start_x, inner_start_y - start_y,
                               inner_end_x - inner_start_x, inner_end_y - inner_start_y, _p_real, _p_imag);
            
            // output complex matrix
			// conversion
			data_as_txt = new char[(inner_end_x - inner_start_x)*( inner_end_y - inner_start_y)*chars_per_complex_num];
			count = 0;
			for (int j=0; j<(inner_end_y - inner_start_y); j++) {
				for (int k=0; k<(inner_end_x - inner_start_x)-1; k++) {
					sprintf(&data_as_txt[count*chars_per_complex_num], "(%+.6e,%+.6e) ", _p_real[j*(inner_end_x - inner_start_x) + k], _p_imag[j*(inner_end_x - inner_start_x) + k]);
					count++;
				}
				if(coords[1] == dims[1]-1) {
					sprintf(&data_as_txt[count*chars_per_complex_num], "(%+.6e,%+.6e)\n", _p_real[j*(inner_end_x - inner_start_x) + (inner_end_x - inner_start_x)-1], _p_imag[j*(inner_end_x - inner_start_x) + (inner_end_x - inner_start_x)-1]);
					count++;
				}
				else {
					sprintf(&data_as_txt[count*chars_per_complex_num], "(%+.6e,%+.6e) ", _p_real[j*(inner_end_x - inner_start_x) + (inner_end_x - inner_start_x)-1], _p_imag[j*(inner_end_x - inner_start_x) + (inner_end_x - inner_start_x)-1]);
					count++;
				}
			}

			// open the file, and set the view 
            sprintf(filename, "%s/%i-%i-iter-comp.dat", output_folder, particle_tag, i);
			MPI_File_open(cartcomm, filename, 
						  MPI_MODE_CREATE|MPI_MODE_WRONLY,
						  MPI_INFO_NULL, &file);

			MPI_File_set_view(file, 0,  MPI_CHAR, complex_localarray, 
								   "native", MPI_INFO_NULL);

			MPI_File_write_all(file, data_as_txt, (inner_end_x - inner_start_x)*( inner_end_y - inner_start_y), complex_num_as_string, &status);
			MPI_File_close(&file);
			delete [] data_as_txt;
			
			// output real matrix
			//conversion
			data_as_txt = new char[(inner_end_x - inner_start_x)*( inner_end_y - inner_start_y)*charspernum];
			count = 0;
			for (int j=0; j<(inner_end_y - inner_start_y); j++) {
				for (int k=0; k<(inner_end_x - inner_start_x)-1; k++) {
					sprintf(&data_as_txt[count*charspernum], "%+.6e ", _p_real[j*(inner_end_x - inner_start_x) + k]);
					count++;
				}
				if(coords[1] == dims[1]-1) {
					sprintf(&data_as_txt[count*charspernum], "%+.6e\n", _p_real[j*(inner_end_x - inner_start_x) + (inner_end_x - inner_start_x)-1]);
					count++;
				}
				else {
					sprintf(&data_as_txt[count*charspernum], "%+.6e ", _p_real[j*(inner_end_x - inner_start_x) + (inner_end_x - inner_start_x)-1]);
					count++;
				}
			}
			
			// open the file, and set the view 
            sprintf(filename, "%s/%i-%i-iter-real.dat", output_folder, particle_tag, i);
			MPI_File_open(cartcomm, filename, 
						  MPI_MODE_CREATE|MPI_MODE_WRONLY,
						  MPI_INFO_NULL, &file);

			MPI_File_set_view(file, 0,  MPI_CHAR, localarray, 
								   "native", MPI_INFO_NULL);

			MPI_File_write_all(file, data_as_txt, (inner_end_x - inner_start_x)*( inner_end_y - inner_start_y), num_as_string, &status);
			MPI_File_close(&file);
			delete [] data_as_txt;
            
			/*
			//NO MPI
			
			sprintf(filename, "%s/%i-%i-iter-real.dat", output_folder, particle_tag, i);
			print_matrix(filename, _p_real, matrix_width - 2 * periods[1]*halo_x,
						 matrix_width - 2 * periods[1]*halo_x, matrix_height - 2 * periods[0]*halo_y);

			sprintf(filename, "%s/%i-%i-iter-comp.dat", output_folder, particle_tag, i);
			print_complex_matrix(filename, _p_real, _p_imag, matrix_width - 2 * periods[1]*halo_x,
								 matrix_width - 2 * periods[1]*halo_x, matrix_height - 2 * periods[0]*halo_y);
            
            */
            delete [] _p_real;
            delete [] _p_imag;
        }
        kernel->run_kernel_on_halo();
        if (i != iterations - 1) {
            kernel->start_halo_exchange();
        }
        kernel->run_kernel();
        if (i != iterations - 1) {
            kernel->finish_halo_exchange();
        }
        kernel->wait_for_completion(i, snapshots);
    }

    gettimeofday(&end, NULL);
    if (coords[0] == 0 && coords[1] == 0 && verbose == true) {
        long time = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
        std::cout << "TROTTER " << matrix_width - periods[1] * 2 * halo_x << "x" << matrix_height - periods[0] * 2 * halo_y << " " << kernel->get_name() << " " << nProcs << " " << time << std::endl;
    }
    
    MPI_Type_free(&localarray);
    MPI_Type_free(&num_as_string);
    MPI_Type_free(&complex_localarray);
    MPI_Type_free(&complex_num_as_string);
    
    delete kernel;
}
