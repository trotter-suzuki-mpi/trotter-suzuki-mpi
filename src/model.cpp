/**
 * Massively Parallel Trotter-Suzuki Solver
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
#include <fstream>
#include <iostream>
#include "trottersuzuki.h"
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

double const_potential(double x, double y) {
    return 0.;
}

Lattice::Lattice(int dim, double _length,
                 bool periodic_x_axis, bool periodic_y_axis,
                 double angular_velocity): length_x(_length), length_y(_length) {
    delta_x = length_x / double(dim);
    delta_y = length_y / double(dim);
    periods[0] = (int) periodic_y_axis;
    periods[1] = (int) periodic_x_axis;
    mpi_dims[0] = mpi_dims[1] = 0;
#ifdef HAVE_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_procs);
    MPI_Dims_create(mpi_procs, 2, mpi_dims);  //partition all the processes (the size of MPI_COMM_WORLD's group) into an 2-dimensional topology
    MPI_Cart_create(MPI_COMM_WORLD, 2, mpi_dims, periods, 0, &cartcomm);
    MPI_Comm_rank(cartcomm, &mpi_rank);
    MPI_Cart_coords(cartcomm, mpi_rank, 2, mpi_coords);
#else
    mpi_procs = 1;
    mpi_rank = 0;
    mpi_dims[0] = mpi_dims[1] = 1;
    mpi_coords[0] = mpi_coords[1] = 0;
#endif
    halo_x = (angular_velocity == 0. ? 4 : 8);
    halo_y = (angular_velocity == 0. ? 4 : 8);
    global_dim_x = dim + periods[1] * 2 * halo_x;
    global_dim_y = dim + periods[0] * 2 * halo_y;
    global_no_halo_dim_x = dim;
    global_no_halo_dim_y = dim;
    //set dimension of tiles and offsets
    calculate_borders(mpi_coords[1], mpi_dims[1], &start_x, &end_x,
                      &inner_start_x, &inner_end_x,
                      dim, halo_x, periods[1]);
    calculate_borders(mpi_coords[0], mpi_dims[0], &start_y, &end_y,
                      &inner_start_y, &inner_end_y,
                      dim, halo_y, periods[0]);
    dim_x = end_x - start_x;
    dim_y = end_y - start_y;
}


State::State(Lattice *_grid, double *_p_real, double *_p_imag): grid(_grid) {
    expected_values_updated = false;
    if (_p_real == 0) {
        self_init = true;
        p_real = new double[grid->dim_x * grid->dim_y];
        for (int i = 0; i < grid->dim_x * grid->dim_y; i++) {
            p_real[i] = 0;
        }
    }
    else {
        self_init = false;
        p_real = _p_real;
    }
    if (_p_imag == 0) {
        p_imag = new double[grid->dim_x * grid->dim_y];
        for (int i = 0; i < grid->dim_x * grid->dim_y; i++) {
            p_imag[i] = 0;
        }
    }
    else {
        p_imag = _p_imag;
    }
}

State::State(const State &obj): grid(obj.grid), expected_values_updated(obj.expected_values_updated), self_init(obj.self_init),
    mean_X(obj.mean_X), mean_XX(obj.mean_XX), mean_Y(obj.mean_Y), mean_YY(obj.mean_YY),
    mean_Px(obj.mean_Px), mean_PxPx(obj.mean_PxPx), mean_Py(obj.mean_Py), mean_PyPy(obj.mean_PyPy),
    norm2(obj.norm2) {
    p_real = new double[grid->dim_x * grid->dim_y];
    p_imag = new double[grid->dim_x * grid->dim_y];
    for (int y = 0; y < grid->dim_y; y++) {
        for (int x = 0; x < grid->dim_x; x++) {
            p_real[y * grid->dim_x + x] = obj.p_real[y * grid->dim_x + x];
            p_imag[y * grid->dim_x + x] = obj.p_imag[y * grid->dim_x + x];
        }
    }
}

State::~State() {
    if (self_init) {
        delete [] p_real;
        delete [] p_imag;
    }
}

void State::imprint(complex<double> (*function)(double x, double y)) {
    double delta_x = grid->delta_x, delta_y = grid->delta_y;
    double idy = grid->start_y * delta_y + 0.5 * delta_y, idx;
    double x_c = grid->global_no_halo_dim_x * grid->delta_x * 0.5;
    double y_c = grid->global_no_halo_dim_y * grid->delta_y * 0.5;
    for (int y = 0; y < grid->dim_y; y++, idy += delta_y) {
        idx = grid->start_x * delta_x + 0.5 * delta_x;
        for (int x = 0; x < grid->dim_x; x++, idx += delta_x) {
            complex<double> tmp = function(idx - x_c, idy - y_c);
            double tmp_p_real = p_real[y * grid->dim_x + x];
            p_real[y * grid->dim_x + x] = tmp_p_real * real(tmp) - p_imag[y * grid->dim_x + x] * imag(tmp);
            p_imag[y * grid->dim_x + x] = tmp_p_real * imag(tmp) + p_imag[y * grid->dim_x + x] * real(tmp);
        }
    }
}

void State::init_state(complex<double> (*ini_state)(double x, double y)) {
    complex<double> tmp;
    double delta_x = grid->delta_x, delta_y = grid->delta_y;
    double idy = grid->start_y * delta_y + 0.5 * delta_y, idx;
    double x_c = grid->global_no_halo_dim_x * grid->delta_x * 0.5;
    double y_c = grid->global_no_halo_dim_y * grid->delta_y * 0.5;
    for (int y = 0; y < grid->dim_y; y++, idy += delta_y) {
        idx = grid->start_x * delta_x + 0.5 * delta_x;
        for (int x = 0; x < grid->dim_x; x++, idx += delta_x) {
            tmp = ini_state(idx - x_c, idy - y_c);
            p_real[y * grid->dim_x + x] = real(tmp);
            p_imag[y * grid->dim_x + x] = imag(tmp);
        }
    }
}

void State::loadtxt(char *file_name) {
    ifstream input(file_name);
    int in_width = grid->global_no_halo_dim_x;
    int in_height = grid->global_no_halo_dim_y;
    complex<double> tmp;
    for(int i = 0; i < in_height; i++) {
        for(int j = 0; j < in_width; j++) {
            input >> tmp;

            if((i - grid->start_y) >= 0 && (i - grid->start_y) < grid->dim_y && (j - grid->start_x) >= 0 && (j - grid->start_x) < grid->dim_x) {
                p_real[(i - grid->start_y) * grid->dim_x + j - grid->start_x] = real(tmp);
                p_imag[(i - grid->start_y) * grid->dim_x + j - grid->start_x] = imag(tmp);
            }

            //Down band
            if(i < grid->halo_y && grid->mpi_coords[0] == grid->mpi_dims[0] - 1 && grid->periods[0] != 0) {
                if((j - grid->start_x) >= 0 && (j - grid->start_x) < grid->dim_x) {
                    p_real[(i + grid->dim_y - grid->halo_y) * grid->dim_x + j - grid->start_x] = real(tmp);
                    p_imag[(i + grid->dim_y - grid->halo_y) * grid->dim_x + j - grid->start_x] = imag(tmp);
                }
                //Down right corner
                if(j < grid->halo_x && grid->periods[1] != 0 && grid->mpi_coords[1] == grid->mpi_dims[1] - 1) {
                    p_real[(i + grid->dim_y - grid->halo_y) * grid->dim_x + j + grid->dim_x - grid->halo_x] = real(tmp);
                    p_imag[(i + grid->dim_y - grid->halo_y) * grid->dim_x + j + grid->dim_x - grid->halo_x] = imag(tmp);
                }
                //Down left corner
                if(j >= in_width - grid->halo_x && grid->periods[1] != 0 && grid->mpi_coords[1] == 0) {
                    p_real[(i + grid->dim_y - grid->halo_y) * grid->dim_x + j - (in_width - grid->halo_x)] = real(tmp);
                    p_imag[(i + grid->dim_y - grid->halo_y) * grid->dim_x + j - (in_width - grid->halo_x)] = imag(tmp);
                }
            }

            //Upper band
            if(i >= in_height - grid->halo_y && grid->periods[0] != 0 && grid->mpi_coords[0] == 0) {
                if((j - grid->start_x) >= 0 && (j - grid->start_x) < grid->dim_x) {
                    p_real[(i - (in_height - grid->halo_y)) * grid->dim_x + j - grid->start_x] = real(tmp);
                    p_imag[(i - (in_height - grid->halo_y)) * grid->dim_x + j - grid->start_x] = imag(tmp);
                }
                //Up right corner
                if(j < grid->halo_x && grid->periods[1] != 0 && grid->mpi_coords[1] == grid->mpi_dims[1] - 1) {
                    p_real[(i - (in_height - grid->halo_y)) * grid->dim_x + j + grid->dim_x - grid->halo_x] = real(tmp);
                    p_imag[(i - (in_height - grid->halo_y)) * grid->dim_x + j + grid->dim_x - grid->halo_x] = imag(tmp);
                }
                //Up left corner
                if(j >= in_width - grid->halo_x && grid->periods[1] != 0 && grid->mpi_coords[1] == 0) {
                    p_real[(i - (in_height - grid->halo_y)) * grid->dim_x + j - (in_width - grid->halo_x)] = real(tmp);
                    p_imag[(i - (in_height - grid->halo_y)) * grid->dim_x + j - (in_width - grid->halo_x)] = imag(tmp);
                }
            }

            //Right band
            if(j < grid->halo_x && grid->periods[1] != 0 && grid->mpi_coords[1] == grid->mpi_dims[1] - 1) {
                if((i - grid->start_y) >= 0 && (i - grid->start_y) < grid->dim_y) {
                    p_real[(i - grid->start_y) * grid->dim_x + j + grid->dim_x - grid->halo_x] = real(tmp);
                    p_imag[(i - grid->start_y) * grid->dim_x + j + grid->dim_x - grid->halo_x] = imag(tmp);
                }
            }

            //Left band
            if(j >= in_width - grid->halo_x && grid->periods[1] != 0 && grid->mpi_coords[1] == 0) {
                if((i - grid->start_y) >= 0 && (i - grid->start_y) < grid->dim_y) {
                    p_real[(i - grid->start_y) * grid->dim_x + j - (in_width - grid->halo_x)] = real(tmp);
                    p_imag[(i - grid->start_y) * grid->dim_x + j - (in_width - grid->halo_x)] = imag(tmp);
                }
            }
        }
    }
    input.close();
}

double *State::get_particle_density(double *_density) {
    double *density;
    if (_density == 0) {
        density = new double[grid->dim_x * grid->dim_y];
    }
    else {
        density = _density;
    }
    for(int j = grid->inner_start_y - grid->start_y; j < grid->inner_end_y - grid->start_y; j++) {
        for(int i = grid->inner_start_x - grid->start_x; i < grid->inner_end_x - grid->start_x; i++) {
            density[j * grid->dim_x + i] = p_real[j * grid->dim_x + i] * p_real[j * grid->dim_x + i] + p_imag[j * grid->dim_x + i] * p_imag[j * grid->dim_x + i];
        }
    }
    return density;
}

void State::write_particle_density(string fileprefix) {
    double *density = new double[grid->dim_x * grid->dim_y];
    stringstream filename;
    filename << fileprefix << "-density";
    stamp_matrix(grid, get_particle_density(density), filename.str());
    delete [] density;
}

double *State::get_phase(double *_phase) {
    double *phase;
    if (_phase == 0) {
        phase = new double[grid->dim_x * grid->dim_y];
    }
    else {
        phase = _phase;
    }
    double norm;
    for(int j = grid->inner_start_y - grid->start_y; j < grid->inner_end_y - grid->start_y; j++) {
        for(int i = grid->inner_start_x - grid->start_x; i < grid->inner_end_x - grid->start_x; i++) {
            norm = sqrt(p_real[j * grid->dim_x + i] * p_real[j * grid->dim_x + i] + p_imag[j * grid->dim_x + i] * p_imag[j * grid->dim_x + i]);
            if(norm == 0)
                phase[j * grid->dim_x + i] = 0;
            else
                phase[j * grid->dim_x + i] = acos(p_real[j * grid->dim_x + i] / norm) * ((p_imag[j * grid->dim_x + i] >= 0) - (p_imag[j * grid->dim_x + i] < 0));
        }
    }
    return phase;
}

void State::write_phase(string fileprefix) {
    double *phase = new double[grid->dim_x * grid->dim_y];
    stringstream filename;
    filename << fileprefix << "-phase";
    stamp_matrix(grid, get_phase(phase), filename.str());
    delete [] phase;
}

void State::calculate_expected_values(void) {
    int ini_halo_x = grid->inner_start_x - grid->start_x;
    int ini_halo_y = grid->inner_start_y - grid->start_y;
    int end_halo_x = grid->end_x - grid->inner_end_x;
    int end_halo_y = grid->end_y - grid->inner_end_y;
    int tile_width = grid->end_x - grid->start_x;

    double grid_center_x = grid->length_x * 0.5 - grid->delta_x * 0.5;
    double grid_center_y = grid->length_y * 0.5 - grid->delta_y * 0.5;

    double sum_norm2 = 0.;
    double sum_x_mean = 0, sum_xx_mean = 0, sum_y_mean = 0, sum_yy_mean = 0;
    double sum_px_mean = 0, sum_pxpx_mean = 0, sum_py_mean = 0,
           sum_pypy_mean = 0,
           param_px = - 1. / grid->delta_x,
           param_py = 1. / grid->delta_y;
    complex<double> const_1 = -1. / 12., const_2 = 4. / 3., const_3 = -2.5;
    complex<double> derivate1_1 = 1. / 6., derivate1_2 = - 1., derivate1_3 = 0.5, derivate1_4 = 1. / 3.;

#ifndef HAVE_MPI
    #pragma omp parallel for reduction(+:sum_norm2,sum_x_mean,sum_y_mean,sum_xx_mean,sum_yy_mean,sum_px_mean,sum_py_mean,sum_pxpx_mean,sum_pypy_mean)
#endif
    for (int i = grid->inner_start_y - grid->start_y; i < grid->inner_end_y - grid->start_y; ++i) {
        int y = grid->inner_start_y + i - (grid->inner_start_y - grid->start_y);
        complex<double> psi_up, psi_down, psi_center, psi_left, psi_right;
        complex<double> psi_up_up, psi_down_down, psi_left_left, psi_right_right;
        int x = grid->inner_start_x;
        for (int j = grid->inner_start_x - grid->start_x; j < grid->inner_end_x - grid->start_x; ++j) {

            psi_center = complex<double> (p_real[i * tile_width + j], p_imag[i * tile_width + j]);
            sum_norm2 += real(conj(psi_center) * psi_center);
            sum_x_mean += real(conj(psi_center) * psi_center * complex<double>(grid->delta_x * x - grid_center_x, 0.));
            sum_y_mean += real(conj(psi_center) * psi_center * complex<double>(grid->delta_y * y - grid_center_y, 0.));
            sum_xx_mean += real(conj(psi_center) * psi_center * complex<double>(grid->delta_x * x - grid_center_x, 0.) * complex<double>(grid->delta_x * x - grid_center_x, 0.));
            sum_yy_mean += real(conj(psi_center) * psi_center * complex<double>(grid->delta_y * y - grid_center_y, 0.) * complex<double>(grid->delta_y * y - grid_center_y, 0.));

            if (i - (grid->inner_start_y - grid->start_y) >= (ini_halo_y == 0) * 2 &&
                    i < grid->inner_end_y - grid->start_y - (end_halo_y == 0) * 2 &&
                    j - (grid->inner_start_x - grid->start_x) >= (ini_halo_x == 0) * 2 &&
                    j < grid->inner_end_x - grid->start_x - (end_halo_x == 0) * 2) {

                psi_up = complex<double> (p_real[(i - 1) * tile_width + j],
                                          p_imag[(i - 1) * tile_width + j]);
                psi_down = complex<double> (p_real[(i + 1) * tile_width + j],
                                            p_imag[(i + 1) * tile_width + j]);
                psi_right = complex<double> (p_real[i * tile_width + j + 1],
                                             p_imag[i * tile_width + j + 1]);
                psi_left = complex<double> (p_real[i * tile_width + j - 1],
                                            p_imag[i * tile_width + j - 1]);
                psi_up_up = complex<double> (p_real[(i - 2) * tile_width + j],
                                             p_imag[(i - 2) * tile_width + j]);
                psi_down_down = complex<double> (p_real[(i + 2) * tile_width + j],
                                                 p_imag[(i + 2) * tile_width + j]);
                psi_right_right = complex<double> (p_real[i * tile_width + j + 2],
                                                   p_imag[i * tile_width + j + 2]);
                psi_left_left = complex<double> (p_real[i * tile_width + j - 2],
                                                 p_imag[i * tile_width + j - 2]);

                sum_px_mean += imag(conj(psi_center) * (derivate1_4 * psi_right + derivate1_3 * psi_center + derivate1_2 * psi_left + derivate1_1 * psi_left_left));
                sum_py_mean += imag(conj(psi_center) * (derivate1_4 * psi_up + derivate1_3 * psi_center + derivate1_2 * psi_down + derivate1_1 * psi_down_down));
                sum_pxpx_mean += real(conj(psi_center) * (const_1 * psi_right_right + const_2 * psi_right + const_2 * psi_left + const_1 * psi_left_left + const_3 * psi_center));
                sum_pypy_mean += real(conj(psi_center) * (const_1 * psi_down_down + const_2 * psi_down + const_2 * psi_up + const_1 * psi_up_up + const_3 * psi_center));
            }
            ++x;
        }
    }
    norm2 = sum_norm2;
    mean_X = sum_x_mean;
    mean_Y = sum_y_mean;
    mean_XX = sum_xx_mean;
    mean_YY = sum_yy_mean;
    mean_Px = - sum_px_mean * param_px;
    mean_Py = - sum_py_mean * param_py;
    mean_PxPx = - sum_pxpx_mean * param_px * param_px;
    mean_PyPy = - sum_pypy_mean * param_py * param_py;

#ifdef HAVE_MPI
    double *norm2_mpi = new double[grid->mpi_procs];
    double *mean_X_mpi = new double[grid->mpi_procs];
    double *mean_Y_mpi = new double[grid->mpi_procs];
    double *mean_XX_mpi = new double[grid->mpi_procs];
    double *mean_YY_mpi = new double[grid->mpi_procs];
    double *mean_Px_mpi = new double[grid->mpi_procs];
    double *mean_Py_mpi = new double[grid->mpi_procs];
    double *mean_PxPx_mpi = new double[grid->mpi_procs];
    double *mean_PyPy_mpi = new double[grid->mpi_procs];

    MPI_Allgather(&norm2, 1, MPI_DOUBLE, norm2_mpi, 1, MPI_DOUBLE, grid->cartcomm);
    MPI_Allgather(&mean_X, 1, MPI_DOUBLE, mean_X_mpi, 1, MPI_DOUBLE, grid->cartcomm);
    MPI_Allgather(&mean_Y, 1, MPI_DOUBLE, mean_Y_mpi, 1, MPI_DOUBLE, grid->cartcomm);
    MPI_Allgather(&mean_XX, 1, MPI_DOUBLE, mean_XX_mpi, 1, MPI_DOUBLE, grid->cartcomm);
    MPI_Allgather(&mean_YY, 1, MPI_DOUBLE, mean_YY_mpi, 1, MPI_DOUBLE, grid->cartcomm);
    MPI_Allgather(&mean_Px, 1, MPI_DOUBLE, mean_Px_mpi, 1, MPI_DOUBLE, grid->cartcomm);
    MPI_Allgather(&mean_Py, 1, MPI_DOUBLE, mean_Py_mpi, 1, MPI_DOUBLE, grid->cartcomm);
    MPI_Allgather(&mean_PxPx, 1, MPI_DOUBLE, mean_PxPx_mpi, 1, MPI_DOUBLE, grid->cartcomm);
    MPI_Allgather(&mean_PyPy, 1, MPI_DOUBLE, mean_PyPy_mpi, 1, MPI_DOUBLE, grid->cartcomm);

    norm2 = 0.;
    mean_X = 0.;
    mean_Y = 0.;
    mean_XX = 0.;
    mean_YY = 0.;
    mean_Px = 0.;
    mean_Py = 0.;
    mean_PxPx = 0.;
    mean_PyPy = 0.;

    for(int i = 0; i < grid->mpi_procs; i++) {
        norm2 += norm2_mpi[i];
        mean_X += mean_X_mpi[i];
        mean_Y += mean_Y_mpi[i];
        mean_XX += mean_XX_mpi[i];
        mean_YY += mean_YY_mpi[i];
        mean_Px += mean_Px_mpi[i];
        mean_Py += mean_Py_mpi[i];
        mean_PxPx += mean_PxPx_mpi[i];
        mean_PyPy += mean_PyPy_mpi[i];
    }
    delete [] norm2_mpi;
    delete [] mean_X_mpi;
    delete [] mean_Y_mpi;
    delete [] mean_XX_mpi;
    delete [] mean_YY_mpi;
    delete [] mean_Px_mpi;
    delete [] mean_Py_mpi;
    delete [] mean_PxPx_mpi;
    delete [] mean_PyPy_mpi;
#endif
    mean_X = mean_X / norm2;
    mean_Y = mean_Y / norm2;
    mean_XX = mean_XX / norm2;
    mean_YY = mean_YY / norm2;
    mean_Px = mean_Px / norm2;
    mean_Py = mean_Py / norm2;
    mean_PxPx = mean_PxPx / norm2;
    mean_PyPy = mean_PyPy / norm2;
    norm2 *= grid->delta_x * grid->delta_y;

    expected_values_updated = true;
}

double State::get_mean_x(void) {
    if(!expected_values_updated)
        calculate_expected_values();
    return mean_X;
}

double State::get_mean_xx(void) {
    if(!expected_values_updated)
        calculate_expected_values();
    return mean_XX;
}

double State::get_mean_y(void) {
    if(!expected_values_updated)
        calculate_expected_values();
    return mean_Y;
}

double State::get_mean_yy(void) {
    if(!expected_values_updated)
        calculate_expected_values();
    return mean_YY;
}

double State::get_mean_px(void) {
    if(!expected_values_updated)
        calculate_expected_values();
    return mean_Px;
}

double State::get_mean_pxpx(void) {
    if(!expected_values_updated)
        calculate_expected_values();
    return mean_PxPx;
}

double State::get_mean_py(void) {
    if(!expected_values_updated)
        calculate_expected_values();
    return mean_Py;
}

double State::get_mean_pypy(void) {
    if(!expected_values_updated)
        calculate_expected_values();
    return mean_PyPy;
}

double State::get_squared_norm(void) {
    if(!expected_values_updated)
        calculate_expected_values();
    return norm2;
}

void State::write_to_file(string filename) {
    stamp(grid, this, filename);
}

ExponentialState::ExponentialState(Lattice *_grid, int _n_x, int _n_y, double _norm, double _phase, double *_p_real, double *_p_imag):
    State(_grid, _p_real, _p_imag), n_x(_n_x), n_y(_n_y), norm(_norm), phase(_phase) {
    complex<double> tmp;
    double delta_x = grid->delta_x, delta_y = grid->delta_y;
    double idy = grid->start_y * delta_y + 0.5 * delta_y, idx;
    for (int y = 0; y < grid->dim_y; y++, idy += delta_y) {
        idx = grid->start_x * delta_x + 0.5 * delta_x;
        for (int x = 0; x < grid->dim_x; x++, idx += delta_x) {
            tmp = exp_state(idx, idy);
            p_real[y * grid->dim_x + x] = real(tmp);
            p_imag[y * grid->dim_x + x] = imag(tmp);
        }
    }
}

complex<double> ExponentialState::exp_state(double x, double y) {
    double L_x = grid->global_no_halo_dim_x * grid->delta_x;
    double L_y = grid->global_no_halo_dim_x * grid->delta_y;
    return sqrt(norm / (L_x * L_y)) * exp(complex<double>(0., phase)) * exp(complex<double>(0., 2 * M_PI * double(n_x) / L_x * x + 2 * M_PI * double(n_y) / L_y * y));
}

GaussianState::GaussianState(Lattice *_grid, double _omega_x, double _omega_y, double _mean_x, double _mean_y,
                             double _norm, double _phase, double *_p_real, double *_p_imag):
    State(_grid, _p_real, _p_imag), mean_x(_mean_x),
    mean_y(_mean_y), omega_x(_omega_x), omega_y(_omega_y), norm(_norm), phase(_phase) {
    if (omega_y == -1.)
        omega_y = omega_x;
	complex<double> tmp;
    double delta_x = grid->delta_x, delta_y = grid->delta_y;
    double idy = grid->start_y * delta_y + 0.5 * delta_y, idx;
    for (int y = 0; y < grid->dim_y; y++, idy += delta_y) {
        idx = grid->start_x * delta_x + 0.5 * delta_x;
        for (int x = 0; x < grid->dim_x; x++, idx += delta_x) {
            tmp = gauss_state(idx, idy);
            p_real[y * grid->dim_x + x] = real(tmp);
            p_imag[y * grid->dim_x + x] = imag(tmp);
        }
    }
}

complex<double> GaussianState::gauss_state(double x, double y) {
    double x_c = grid->global_no_halo_dim_x * grid->delta_x * 0.5;
    double y_c = grid->global_no_halo_dim_y * grid->delta_y * 0.5;
    return complex<double>(sqrt(norm * sqrt(omega_x * omega_y) / M_PI) * exp(-(omega_x * pow(x - mean_x - x_c, 2.0) + omega_y * pow(y - mean_y - y_c, 2.0)) * 0.5), 0.) * exp(complex<double>(0., phase));
}

SinusoidState::SinusoidState(Lattice *_grid, int _n_x, int _n_y, double _norm, double _phase, double *_p_real, double *_p_imag):
    State(_grid, _p_real, _p_imag), n_x(_n_x), n_y(_n_y), norm(_norm), phase(_phase)  {
    complex<double> tmp;
    double delta_x = grid->delta_x, delta_y = grid->delta_y;
    double idy = grid->start_y * delta_y + 0.5 * delta_y, idx;
    for (int y = 0; y < grid->dim_y; y++, idy += delta_y) {
        idx = grid->start_x * delta_x + 0.5 * delta_x;
        for (int x = 0; x < grid->dim_x; x++, idx += delta_x) {
            tmp = sinusoid_state(idx, idy);
            p_real[y * grid->dim_x + x] = real(tmp);
            p_imag[y * grid->dim_x + x] = imag(tmp);
        }
    }
}

complex<double> SinusoidState::sinusoid_state(double x, double y) {
    double L_x = grid->global_no_halo_dim_x * grid->delta_x;
    double L_y = grid->global_no_halo_dim_y * grid->delta_y;
    return sqrt(norm / (L_x * L_y)) * 2.* exp(complex<double>(0., phase)) * complex<double> (sin(2 * M_PI * double(n_x) / L_x * x) * sin(2 * M_PI * double(n_y) / L_y * y), 0.0);
}

Potential::Potential(Lattice *_grid, char *filename): grid(_grid) {
    matrix = new double[grid->dim_y * grid->dim_x];
    self_init = true;
    is_static = true;
    ifstream input(filename);
    double tmp;
    for(int y = 0; y < grid->dim_y; y++) {
        for(int x = 0; x < grid->dim_x; x++) {
            input >> tmp;
            matrix[y * grid->dim_y + x] = tmp;
        }
    }
    input.close();
}

Potential::Potential(Lattice *_grid, double *_external_pot): grid(_grid) {
    if (_external_pot == 0) {
        self_init = true;
        matrix = new double[grid->dim_x * grid->dim_y];
    }
    else {
        matrix = _external_pot;
    }
    self_init = false;
    is_static = true;
    evolving_potential = NULL;
    static_potential = NULL;
}

Potential::Potential(Lattice *_grid, double (*potential_fuction)(double x, double y)): grid(_grid) {
    is_static = true;
    self_init = false;
    evolving_potential = NULL;
    static_potential = potential_fuction;
    matrix = NULL;
}

Potential::Potential(Lattice *_grid, double (*potential_function)(double x, double y, double t), int _t): grid(_grid) {
    is_static = false;
    self_init = false;
    evolving_potential = potential_function;
    static_potential = NULL;
    matrix = NULL;
}

double Potential::get_value(int x, int y) {
    if (matrix != NULL) {
        return matrix[y * grid->dim_x + x];
    }
    else {
        double idy = grid->start_y * grid->delta_y + y * grid->delta_y + 0.5 * grid->delta_y;
        double idx = grid->start_x * grid->delta_x + x * grid->delta_x + 0.5 * grid->delta_x;
        if (is_static) {
            return static_potential(idx, idy);
        }
        else {
            return evolving_potential(idx, idy, current_evolution_time);
        }
    }
}

bool Potential::update(double t) {
    current_evolution_time = t;
    if (!is_static) {
        if (matrix != NULL) {
            double delta_x = grid->delta_x, delta_y = grid->delta_y;
            double idy = grid->start_y * delta_y + 0.5 * delta_y, idx;
            for (int y = 0; y < grid->dim_y; y++, idy += delta_y) {
                idx = grid->start_x * delta_x + 0.5 * delta_x;
                for (int x = 0; x < grid->dim_x; x++, idx += delta_x) {
                    matrix[y * grid->dim_x + x] = evolving_potential(idx, idy, t);
                }
            }
        }
        return true;
    }
    return false;
}

Potential::~Potential() {
    if (self_init) {
        delete [] matrix;
    }
}

HarmonicPotential::HarmonicPotential(Lattice *_grid, double _omegax, double _omegay, double _mass, double _mean_x, double _mean_y):
    Potential(_grid, const_potential), omegax(_omegax), omegay(_omegay),
    mass(_mass), mean_x(_mean_x), mean_y(_mean_y) {
    is_static = true;
    self_init = false;
    evolving_potential = NULL;
    static_potential = NULL;
    matrix = NULL;
}

double HarmonicPotential::get_value(int x, int y) {
    double idy = (grid->start_y + y) * grid->delta_y + mean_x + 0.5 * grid->delta_y;
    double idx = (grid->start_x + x) * grid->delta_x + mean_y + 0.5 * grid->delta_x;
    double x_c = (grid->global_dim_x - 2.*grid->halo_x * grid->periods[1]) * grid->delta_x * 0.5;
    double y_c = (grid->global_dim_y - 2.*grid->halo_y * grid->periods[1]) * grid->delta_y * 0.5;
    double x_r = idx - x_c;
    double y_r = idy - y_c;
    return 0.5 * mass * (omegax * omegax * x_r * x_r + omegay * omegay * y_r * y_r);
}

HarmonicPotential::~HarmonicPotential() {
}

Hamiltonian::Hamiltonian(Lattice *_grid, Potential *_potential,
                         double _mass, double _coupling_a,
                         double _angular_velocity,
                         double _rot_coord_x, double _rot_coord_y): mass(_mass),
    coupling_a(_coupling_a), angular_velocity(_angular_velocity), grid(_grid) {
    if (angular_velocity != 0.) {
        if (grid->periods[0] != 0 || grid->periods[1] != 0) {
            cout << "Boundary conditions must be closed for rotating frame of refernce\n";
            return;
        }
        if (grid->mpi_procs == 1) {
            grid->halo_x = 8;
            grid->halo_y = 8;
        }
        if (grid->mpi_procs > 1 && (grid->halo_x == 4 || grid->halo_y == 4)) {
            cout << "Halos must be of 8 points width\n";
            return;
        }
    }
    rot_coord_x = (grid->global_dim_x - grid->periods[1] * 2 * grid->halo_x) * 0.5 + _rot_coord_x / grid->delta_x;
    rot_coord_y = (grid->global_dim_y - grid->periods[1] * 2 * grid->halo_y) * 0.5 + _rot_coord_y / grid->delta_y;
    if (_potential == NULL) {
        self_init = true;
        potential = new Potential(grid, const_potential);
    }
    else {
        self_init = false;
        potential = _potential;
    }
}

Hamiltonian::~Hamiltonian() {
    if (self_init) {
        delete potential;
    }
}

Hamiltonian2Component::Hamiltonian2Component(Lattice *_grid,
        Potential *_potential,
        Potential *_potential_b,
        double _mass,
        double _mass_b, double _coupling_a,
        double _coupling_ab, double _coupling_b,
        double _omega_r, double _omega_i,
        double _angular_velocity,
        double _rot_coord_x, double _rot_coord_y):
    Hamiltonian(_grid, _potential, _mass, _coupling_a, _angular_velocity, _rot_coord_x, rot_coord_y), mass_b(_mass_b),
    coupling_ab( _coupling_ab), coupling_b(_coupling_b), omega_r(_omega_r), omega_i(_omega_i) {

    if (_potential_b == NULL) {
        potential_b = _potential;
    }
    else {
        potential_b = _potential_b;
    }
}

Hamiltonian2Component::~Hamiltonian2Component() {

}
