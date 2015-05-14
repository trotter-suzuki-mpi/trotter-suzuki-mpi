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
#include <cmath>

#include "common.h"
#include "trotter.h"

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

void stick_files(int N_files, int N_name, std::complex<float> *psi, const char *dirname, procs_topology var, int dim,
				 int *periods, int halo_x, int halo_y) {
					 
	int start_x, end_x, inner_start_x, inner_end_x,
		start_y, end_y, inner_start_y, inner_end_y;
	
	std::stringstream filename;
	std::string filenames;
		
	//read wave function
	for(int idy = 0; idy < var.dimsy; idy++) {
		for(int idx = 0; idx < var.dimsx; idx++) {
			filename.str("");
			filename << dirname << "/" << N_name << "-iter-" << idx << "-" << idy << "-comp.dat";
			filenames = filename.str();
			std::ifstream in_compl(filenames.c_str());

			//get dimension of input file
			calculate_borders(idx, var.dimsx, &start_x, &end_x, &inner_start_x, &inner_end_x, dim, halo_x, periods[1]);
			calculate_borders(idy, var.dimsy, &start_y, &end_y, &inner_start_y, &inner_end_y, dim, halo_y, periods[0]);
			int width = inner_end_x - inner_start_x;
			int height = inner_end_y - inner_start_y;

			//read file
			for(int j = 0; j < height; j++) {
				for(int k = 0; k < width; k++) {
					in_compl >> psi[(j + inner_start_y) * dim + k + inner_start_x];
				}
			}
			in_compl.close();
		}
	}
	
	//output an unique file with the entire wavefunction
	filename.str("");
	filename << dirname << "/" << N_name << "-iter-comp.dat";
	filenames = filename.str();
	std::ofstream union_out_comp(filenames.c_str());
	
	filename.str("");
	filename << dirname << "/" << N_name << "-iter-real.dat";
	filenames = filename.str();
	std::ofstream union_out_real(filenames.c_str());
	
	for(int j = 0; j < dim; j++) {
		for(int k = 0; k < dim; k++) {
			union_out_comp << psi[j * dim + k] << " ";
			union_out_real << real(psi[j * dim + k]) << " ";
		}
		union_out_comp << std::endl;
		union_out_real << std::endl;
	}
	union_out_comp.close();
	union_out_real.close();
}

void expect_values(int dim, int iterations, int snapshots, float * hamilt_pot, float particle_mass,
				   const char *dirname, procs_topology var, int *periods, int halo_x, int halo_y) {

    if(snapshots == 0)
        return;

    int N_files = (int)ceil(double(iterations) / double(snapshots));
    int N_name[N_files];
    int DIM = dim;

    int start_x, end_x, inner_start_x, inner_end_x,
        start_y, end_y, inner_start_y, inner_end_y;

    N_name[0] = 0;
    for(int i = 1; i < N_files; i++) {
        N_name[i] = N_name[i - 1] + snapshots;
    }

    std::complex<float> sum_E = 0;
    std::complex<float> sum_Px = 0, sum_Py = 0;
    std::complex<float> sum_psi = 0;
    float energy[N_files], momentum_x[N_files], momentum_y[N_files], norm[N_files];
	const float threshold_E = 3, threshold_P = 1;
	const float expected_E = (2. * M_PI / dim) * (2. * M_PI / dim);
	const float expected_Px = 0.;
	const float expected_Py = 0.;
	
    std::complex<float> potential[DIM][DIM];
    std::complex<float> psi[DIM*DIM];
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
		stick_files(N_files, N_name[i], psi, dirname, var, dim, periods, halo_x, halo_y);
		
		if(var.dimsy != 1 || var.dimsx != 1) {
			for(int idy = 0; idy < var.dimsy; idy++) {
				for(int idx = 0; idx < var.dimsx; idx++) {
					filename.str("");
					filename << dirname << "/" << N_name[i] << "-iter-" << idx << "-" << idy << "-comp.dat";
					filenames = filename.str();
					remove(filenames.c_str());
					filename.str("");
					filename << dirname << "/" << N_name[i] << "-iter-" << idx << "-" << idy << "-real.dat";
					filenames = filename.str();
					remove(filenames.c_str());
				}
			}	
		}
		
        for(int j = 1; j < DIM - 1; j++) {
            for(int k = 1; k < DIM - 1; k++) {
                sum_E += conj(psi[k + j * dim]) * (cost_E * (psi[k + 1 + j * dim] + psi[k - 1 + j * dim] + psi[k + (j + 1) * dim] + psi[k + (j - 1) * dim] - psi[k + j * dim] * std::complex<float> (4., 0.)) + potential[k][j] * psi[k + j * dim]) ;
                sum_Px += conj(psi[k + j * dim]) * (psi[k + 1 + j * dim] - psi[k - 1 + j * dim]);
                sum_Py += conj(psi[k + j * dim]) * (psi[k + (j + 1) * dim] - psi[k + (j - 1) * dim]);
                sum_psi += conj(psi[k + j * dim]) * psi[k + j * dim];
            }
        }

        out << N_name[i] << "\t" << real(sum_E / sum_psi) << "\t" << real(cost_P * sum_Px / sum_psi) << "\t" << real(cost_P * sum_Py / sum_psi) << "\t"
            << real(cost_P * sum_Px / sum_psi)*real(cost_P * sum_Px / sum_psi) + real(cost_P * sum_Py / sum_psi)*real(cost_P * sum_Py / sum_psi) << "\t" << real(sum_psi) << std::endl;
        
        energy[i] = real(sum_E / sum_psi);
        momentum_x[i] = real(cost_P * sum_Px / sum_psi);
        momentum_y[i] = real(cost_P * sum_Py / sum_psi);
        norm[i] = real(cost_P * sum_Px / sum_psi)*real(cost_P * sum_Px / sum_psi) + real(cost_P * sum_Py / sum_psi)*real(cost_P * sum_Py / sum_psi);
        
        sum_E = 0;
        sum_Px = 0;
        sum_Py = 0;
        sum_psi = 0;
    }
    
    //calculate sample variance
    float var_E = 0, var_Px = 0, var_Py = 0;
    float mean_E = 0, mean_Px = 0, mean_Py = 0;
    
    for(int i = 0; i < N_files; i++) {
		mean_E += energy[i];
		mean_Px += momentum_x[i];
		mean_Py += momentum_y[i];
	}
	mean_E /= N_files;
	mean_Px /= N_files;
	mean_Py /= N_files;
	
	for(int i = 0; i < N_files; i++) {
		var_E += (energy[i] - mean_E) * (energy[i] - mean_E);
		var_Px += (momentum_x[i] - mean_Px) * (momentum_x[i] - mean_Px);
		var_Py += (momentum_y[i] - mean_Py) * (momentum_y[i] - mean_Py);
	}
	var_E /= N_files - 1; var_E = sqrt(var_E);
	var_Px /= N_files - 1; var_Px = sqrt(var_Px);
	var_Py /= N_files - 1; var_Py = sqrt(var_Py);
	
	//std::cout << "Sample mean Energy: " << mean_E << "  Sample variance Energy: " << var_E << std::endl;
	//std::cout << "Sample mean Momentum Px: " << mean_Px << "  Sample variance Momentum Px: " << var_Px << std::endl;
	//std::cout << "Sample mean Momentum Py: " << mean_Py << "  Sample variance Momentum Py: " << var_Py << std::endl;
	
	if(std::abs(mean_E - expected_E) / var_E < threshold_E)
		std::cout << "Energy -> OK\tsigma: " << std::abs(mean_E - expected_E) / var_E << std::endl;
	else
		std::cout << "Energy value is not the one theoretically expected: sigma " << std::abs(mean_E - expected_E) / var_E << std::endl;
	if(std::abs(mean_Px - expected_Px) / var_Px < threshold_P)
		std::cout << "Momentum Px -> OK\tsigma: " << std::abs(mean_Px - expected_Px) / var_Px << std::endl;
	else
		std::cout << "Momentum Px value is not the one theoretically expected: sigma " << std::abs(mean_Px - expected_Px) / var_Px << std::endl;
	if(std::abs(mean_Py - expected_Py) / var_Py < threshold_P)
		std::cout << "Momentum Py -> OK\tsigma: " << std::abs(mean_Py - expected_Py) / var_Py << std::endl;
	else
		std::cout << "Momentum Py value is not the one theoretically expected: sigma " << std::abs(mean_Py - expected_Py) / var_Py << std::endl;
	
}

