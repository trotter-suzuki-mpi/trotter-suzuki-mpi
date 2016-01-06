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


Lattice::Lattice(int dim, double _length_x, double _length_y, int _periods[2],
                 double omega): length_x(_length_x), length_y(_length_y) {
    delta_x = length_x / double(dim);
    delta_y = length_y / double(dim);
    if (_periods == 0) {
          periods[0] = 0;
          periods[1] = 0;
    } else {
          periods[0] = _periods[0];
          periods[1] = _periods[1];
    }
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
    halo_x = (omega == 0. ? 4 : 8);
    halo_y = (omega == 0. ? 4 : 8);
    global_dim_x = dim + periods[1] * 2 * halo_x;
    global_dim_y = dim + periods[0] * 2 * halo_y;
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


State::State(Lattice *_grid, double *_p_real, double *_p_imag): grid(_grid){
    if (_p_real == 0) {
        self_init = true;
        p_real = new double[grid->dim_x * grid->dim_y];
    } else {
        self_init = false;
        p_real = _p_real;
    }
    if (_p_imag == 0) {
        p_imag = new double[grid->dim_x * grid->dim_y];
    } else {
        p_imag = _p_imag;
    }
}

State::~State() {
    if (self_init) {
        delete p_real;
        delete p_imag;
    }
}

void State::init_state(complex<double> (*ini_state)(double x, double y)) {
    complex<double> tmp;
    double delta_x = grid->delta_x, delta_y = grid->delta_y;
    double idy = grid->start_y * delta_y, idx;
    for (int y = 0; y < grid->dim_y; y++, idy += delta_y) {
        idx = grid->start_x * delta_x;
        for (int x = 0; x < grid->dim_x; x++, idx += delta_x) {
            tmp = ini_state(idx, idy);
            p_real[y * grid->dim_x + x] = real(tmp);
            p_imag[y * grid->dim_x + x] = imag(tmp);
        }
    }
}


void State::read_state(char *file_name, int read_offset) {
    ifstream input(file_name);
    int in_width = grid->global_dim_x - 2 * grid->periods[1] * grid->halo_x;
    int in_height = grid->global_dim_y - 2 * grid->periods[0] * grid->halo_y;
    complex<double> tmp;
    for(int i = 0; i < read_offset; i++)
        input >> tmp;
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

double State::calculate_squared_norm(bool global) {
    double norm2 = 0;
    int tile_width = grid->end_x - grid->start_x;
    for(int i = grid->inner_start_y - grid->start_y; i < grid->inner_end_y - grid->start_y; i++) {
      for(int j = grid->inner_start_x - grid->start_x; j < grid->inner_end_x - grid->start_x; j++) {
        norm2 += p_real[j + i * tile_width] * p_real[j + i * tile_width] + p_imag[j + i * tile_width] * p_imag[j + i * tile_width];
      }
    }
#ifdef HAVE_MPI
    if (global) {
        double *sums = new double[grid->mpi_procs];
        MPI_Allgather(&norm2, 1, MPI_DOUBLE, sums, 1, MPI_DOUBLE, grid->cartcomm);
        norm2 = 0.;
        for(int i = 0; i < grid->mpi_procs; i++)
            norm2 += sums[i];
        delete [] sums;
    }
#endif
    return norm2 * grid->delta_x * grid->delta_y;
}

double *State::get_particle_density(double *_density) {
    double *density;
    if (_density == 0) {
      density = new double[grid->dim_x * grid->dim_y];
    } else {
      density = _density;
    }
    for(int j = grid->inner_start_y - grid->start_y; j < grid->inner_end_y - grid->start_y; j++) {
        for(int i = grid->inner_start_x - grid->start_x; i < grid->inner_end_x - grid->start_x; i++) {
            density[j * grid->dim_x + i] = p_real[j * grid->dim_x + i] * p_real[j * grid->dim_x + i] + p_imag[j * grid->dim_x + i] * p_imag[j * grid->dim_x + i];
      }
    }
    return density;
}

double *State::get_phase(double *_phase) {
    double *phase;
    if (_phase == 0) {
      phase = new double[grid->dim_x * grid->dim_y];
    } else {
      phase = _phase;
    }
    double norm;
    for(int j = grid->inner_start_y - grid->start_y; j < grid->inner_end_y - grid->start_y; j++) {
        for(int i = grid->inner_start_x - grid->start_x; i < grid->inner_end_x - grid->start_x; i++) {
            norm = sqrt(p_real[j * grid->dim_x + i] * p_real[j * grid->dim_x + i] + p_imag[j * grid->dim_x + i] * p_imag[j * grid->dim_x + i]);
            if(norm == 0)
                phase[j * grid->dim_x + i] = 0;
            else
                phase[j * grid->dim_x + i] = acos(p_real[j * grid->dim_x + i] / norm) * ((p_imag[j * grid->dim_x + i] > 0) - (p_imag[j * grid->dim_x + i] < 0));
        }
    }
    return phase;
}

Hamiltonian::Hamiltonian(Lattice *_grid, double _mass, double _coupling_a,
                         double _angular_velocity,
                         double _rot_coord_x, double _rot_coord_y,
                         double *_external_pot): grid(_grid), mass(_mass),
                         coupling_a(_coupling_a), angular_velocity(_angular_velocity) {
    if (_rot_coord_x == DBL_MAX) {
        rot_coord_x = (grid->global_dim_x - grid->periods[1] * 2 * grid->halo_x) * 0.5;
    } else {
        rot_coord_x = _rot_coord_x;
    }
    if (_rot_coord_y == DBL_MAX) {
        rot_coord_y = (grid->global_dim_y - grid->periods[1] * 2 * grid->halo_y) * 0.5;
    } else {
        rot_coord_y = _rot_coord_y;
    }
    if (_external_pot == 0) {
        external_pot = new double[grid->dim_y * grid->dim_x];
        self_init = true;
    } else {
        external_pot = _external_pot;
        self_init = false;
    }
    has_changed = true;
}

Hamiltonian::~Hamiltonian() {
    if (self_init) {
        delete [] external_pot;
    }
}

void Hamiltonian::initialize_potential(double (*hamiltonian_pot)(double x, double y)) {
    has_changed = true;
    double delta_x = grid->delta_x, delta_y = grid->delta_y;
    double idy = grid->start_y * delta_y, idx;
    for (int y = 0; y < grid->dim_y; y++, idy += delta_y) {
        idx = grid->start_x * delta_x;
        for (int x = 0; x < grid->dim_x; x++, idx += delta_x) {
            external_pot[y * grid->dim_x + x] = hamiltonian_pot(idx, idy);
        }
    }
}

void Hamiltonian::read_potential(char *pot_name) {
    has_changed = true;
    ifstream input(pot_name);
    double tmp;
    for(int y = 0; y < grid->dim_y; y++) {
        for(int x = 0; x < grid->dim_x; x++) {
            input >> tmp;
            external_pot[y * grid->dim_y + x] = tmp;
        }
    }
    input.close();
}

void Hamiltonian::set_potential(double *_external_pot) {
    external_pot = _external_pot;
    has_changed = true;
}

double *Hamiltonian::get_potential() {
    return external_pot;
}

double Hamiltonian::get_potential_value(int x, int y) {
    return external_pot[y * grid->dim_x + x];
}

Hamiltonian2Component::Hamiltonian2Component(Lattice *_grid, double _mass,
                         double _mass_b, double _coupling_a,
                         double _coupling_ab, double _coupling_b,
                         double _omega_r, double _omega_i,
                         double _angular_velocity,
                         double _rot_coord_x, double _rot_coord_y,
                         double *_external_pot, double *_external_pot_b):
                         Hamiltonian(_grid, _mass, _coupling_a, _angular_velocity, _rot_coord_x, rot_coord_y, _external_pot),
                         coupling_ab( _coupling_ab), coupling_b(_coupling_b), omega_r(_omega_r), omega_i(_omega_i) {
    if (_external_pot_b == 0) {
        external_pot_b = new double[grid->dim_y * grid->dim_x];
        self_init = true;
    } else {
        external_pot_b = _external_pot_b;
        self_init = false;
    }
}

Hamiltonian2Component::~Hamiltonian2Component() {
    if (self_init) {
        delete [] external_pot;
        delete [] external_pot_b;
    }
}

void Hamiltonian2Component::initialize_potential(double (*hamiltonian_pot)(double x, double y), int which) {
    double *tmp;
    if (which == 0) {
      tmp = external_pot;
    } else {
      tmp = external_pot_b;
    }
    double delta_x = grid->delta_x, delta_y = grid->delta_y;
    double idy = grid->start_y * delta_y, idx;
    for (int y = 0; y < grid->dim_y; y++, idy += delta_y) {
        idx = grid->start_x * delta_x;
        for (int x = 0; x < grid->dim_x; x++, idx += delta_x) {
            tmp[y * grid->dim_x + x] = hamiltonian_pot(idx, idy);
        }
    }
}

void Hamiltonian2Component::set_potential(double *_external_pot, int which) {
    if (which == 0) {
      external_pot = _external_pot;
    } else {
      external_pot_b = _external_pot;
    }
    has_changed = true;
}

double Hamiltonian2Component::get_potential_value(int x, int y, int which) {
    if (which == 0) {
        return external_pot[y * grid->dim_x + x];
    } else {
        return external_pot_b[y * grid->dim_x + x];
    }
}

double *Hamiltonian2Component::get_potential(int which) {
    if (which == 0) {
        return external_pot;
    } else {
        return external_pot_b;
    }
}
