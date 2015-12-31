#include <cmath>
#include <complex>

#include "trotter.h"

void add_padding(double *padded_matrix, double *matrix, 
                 int padded_dim_x, int padded_dim_y, 
                 int halo_x, int halo_y, 
                 const int dim_x, const int dim_y, int *periods) {
    for (int i = 0; i < dim_y; i++) {
        for (int j = 0; j < dim_x; j++) {
            double element = matrix[j + i * dim_x];
            padded_matrix[(i + halo_y * periods[0]) * padded_dim_x + j + halo_x * periods[1]] = element;
            //Down band
            if (i < halo_y && periods[0] != 0) {
                padded_matrix[(i + padded_dim_y - halo_y) * padded_dim_x + j + halo_x * periods[1]] = element;
                //Down right corner
                if (j < halo_x && periods[1] != 0) {
                    padded_matrix[(i + padded_dim_y - halo_y) * padded_dim_x + j + padded_dim_x - halo_x] = element;
                }
                //Down left corner
                if(j >= dim_x - halo_x && periods[1] != 0) {
                    padded_matrix[(i + padded_dim_y - halo_y) * padded_dim_x + j - (dim_x - halo_x)] = element;
                }
            }
            //Upper band
            if (i >= dim_y - halo_y && periods[0] != 0) {
                padded_matrix[(i - (dim_y - halo_y)) * padded_dim_x + j + halo_x * periods[1]] = element;
                //Up right corner
                if (j < halo_x && periods[1] != 0) {
                    padded_matrix[(i - (dim_y - halo_y)) * padded_dim_x + j + padded_dim_x - halo_x] = element;
                }
                //Up left corner
                if (j >= dim_x - halo_x && periods[1] != 0) {
                    padded_matrix[(i - (dim_y - halo_y)) * padded_dim_x + j - (dim_x - halo_x)] = element;
                }
            }
            //Right band
            if (j < halo_x && periods[1] != 0) {
                padded_matrix[(i + halo_y * periods[0]) * padded_dim_x + j + padded_dim_x - halo_x] = element;
            }
            //Left band
            if (j >= dim_x - halo_x && periods[1] != 0) {
                padded_matrix[(i + halo_y * periods[0]) * padded_dim_x + j - (dim_x - halo_x)] = element;
            }
        }
    }
}


void ini_state_padding(double *_p_real, double *_p_imag, double *p_real, double *p_imag, int dimx, int dimy, int halo_x, int halo_y, const int matrix_width, const int matrix_height, int *periods) {
  for(int i = 0; i < matrix_height; i++) {
        for(int j = 0; j < matrix_width; j++) {
            
      _p_real[(i + halo_y * periods[0]) * dimx + j + halo_x * periods[1]] = p_real[j + i * matrix_width];
      _p_imag[(i + halo_y * periods[0]) * dimx + j + halo_x * periods[1]] = p_imag[j + i * matrix_width];
      
            //Down band
            if(i < halo_y && periods[0] != 0) {
                
        _p_real[(i + dimy - halo_y) * dimx + j + halo_x * periods[1]] = p_real[j + i * matrix_width];
        _p_imag[(i + dimy - halo_y) * dimx + j + halo_x * periods[1]] = p_imag[j + i * matrix_width];
                
                //Down right corner
                if(j < halo_x && periods[1] != 0) {
                    _p_real[(i + dimy - halo_y) * dimx + j + dimx - halo_x] = p_real[j + i * matrix_width];
                    _p_imag[(i + dimy - halo_y) * dimx + j + dimx - halo_x] = p_imag[j + i * matrix_width];
                }
                //Down left corner
                if(j >= matrix_width - halo_x && periods[1] != 0) {
                    _p_real[(i + dimy - halo_y) * dimx + j - (matrix_width - halo_x)] = p_real[j + i * matrix_width];
                    _p_imag[(i + dimy - halo_y) * dimx + j - (matrix_width - halo_x)] = p_imag[j + i * matrix_width];
                }
            }

            //Upper band
            if(i >= matrix_height - halo_y && periods[0] != 0) {
                
        _p_real[(i - (matrix_height - halo_y)) * dimx + j + halo_x * periods[1]] = p_real[j + i * matrix_width];
        _p_imag[(i - (matrix_height - halo_y)) * dimx + j + halo_x * periods[1]] = p_imag[j + i * matrix_width];
      
                //Up right corner
                if(j < halo_x && periods[1] != 0) {
                    _p_real[(i - (matrix_height - halo_y)) * dimx + j + dimx - halo_x] = p_real[j + i * matrix_width];
                    _p_imag[(i - (matrix_height - halo_y)) * dimx + j + dimx - halo_x] = p_imag[j + i * matrix_width];
                }
                //Up left corner
                if(j >= matrix_width - halo_x && periods[1] != 0) {
                    _p_real[(i - (matrix_height - halo_y)) * dimx + j - (matrix_width - halo_x)] = p_real[j + i * matrix_width];
                    _p_imag[(i - (matrix_height - halo_y)) * dimx + j - (matrix_width - halo_x)] = p_imag[j + i * matrix_width];
                }
            }

            //Right band
            if(j < halo_x && periods[1] != 0) {
        _p_real[(i + halo_y * periods[0]) * dimx + j + dimx - halo_x] = p_real[j + i * matrix_width];
        _p_imag[(i + halo_y * periods[0]) * dimx + j + dimx - halo_x] = p_imag[j + i * matrix_width];
            }

            //Left band
            if(j >= matrix_width - halo_x && periods[1] != 0) {
        _p_real[(i + halo_y * periods[0]) * dimx + j - (matrix_width - halo_x)] = p_real[j + i * matrix_width];
        _p_imag[(i + halo_y * periods[0]) * dimx + j - (matrix_width - halo_x)] = p_imag[j + i * matrix_width];
            }
        }
    }
}

void ini_external_pot(double *external_pot_real, double *external_pot_imag, double *external_pot, double delta_t, int dimx, int dimy, int halo_x, int halo_y, const int matrix_width, const int matrix_height, int *periods, bool imag_time) {
  std::complex<double> tmp;    
    for(int i = 0; i < matrix_height; i++) {
        for(int j = 0; j < matrix_width; j++) {
            
            if(imag_time)
        tmp = exp(std::complex<double> (-1. * delta_t * external_pot[j + i * matrix_width] , 0.));
      else
        tmp = exp(std::complex<double> (0., -1. * delta_t * external_pot[j + i * matrix_width]));
        
      external_pot_real[(i + halo_y * periods[0]) * dimx + j + halo_x * periods[1]] = real(tmp);
      external_pot_imag[(i + halo_y * periods[0]) * dimx + j + halo_x * periods[1]] = imag(tmp);
      
            //Down band
            if(i < halo_y && periods[0] != 0) {
                
        external_pot_real[(i + dimy - halo_y) * dimx + j + halo_x * periods[1]] = real(tmp);
        external_pot_imag[(i + dimy - halo_y) * dimx + j + halo_x * periods[1]] = imag(tmp);
                
                //Down right corner
                if(j < halo_x && periods[1] != 0) {
                    external_pot_real[(i + dimy - halo_y) * dimx + j + dimx - halo_x] = real(tmp);
                    external_pot_imag[(i + dimy - halo_y) * dimx + j + dimx - halo_x] = imag(tmp);
                }
                //Down left corner
                if(j >= matrix_width - halo_x && periods[1] != 0) {
                    external_pot_real[(i + dimy - halo_y) * dimx + j - (matrix_width - halo_x)] = real(tmp);
                    external_pot_imag[(i + dimy - halo_y) * dimx + j - (matrix_width - halo_x)] = imag(tmp);
                }
            }

            //Upper band
            if(i >= matrix_height - halo_y && periods[0] != 0) {
                
        external_pot_real[(i - (matrix_height - halo_y)) * dimx + j + halo_x * periods[1]] = real(tmp);
        external_pot_imag[(i - (matrix_height - halo_y)) * dimx + j + halo_x * periods[1]] = imag(tmp);
      
                //Up right corner
                if(j < halo_x && periods[1] != 0) {
                    external_pot_real[(i - (matrix_height - halo_y)) * dimx + j + dimx - halo_x] = real(tmp);
                    external_pot_imag[(i - (matrix_height - halo_y)) * dimx + j + dimx - halo_x] = imag(tmp);
                }
                //Up left corner
                if(j >= matrix_width - halo_x && periods[1] != 0) {
                    external_pot_real[(i - (matrix_height - halo_y)) * dimx + j - (matrix_width - halo_x)] = real(tmp);
                    external_pot_imag[(i - (matrix_height - halo_y)) * dimx + j - (matrix_width - halo_x)] = imag(tmp);
                }
            }

            //Right band
            if(j < halo_x && periods[1] != 0) {
        external_pot_real[(i + halo_y * periods[0]) * dimx + j + dimx - halo_x] = real(tmp);
        external_pot_imag[(i + halo_y * periods[0]) * dimx + j + dimx - halo_x] = imag(tmp);
            }

            //Left band
            if(j >= matrix_width - halo_x && periods[1] != 0) {
        external_pot_real[(i + halo_y * periods[0]) * dimx + j - (matrix_width - halo_x)] = real(tmp);
        external_pot_imag[(i + halo_y * periods[0]) * dimx + j - (matrix_width - halo_x)] = imag(tmp);
            }
        }
    }
}

void solver(Lattice *grid, State *state, Hamiltonian *hamiltonian,
            double delta_t, const int iterations, string kernel_type, bool imag_time) {
  
  int halo_x = (kernel_type == "sse" ? 3 : 4);
    halo_x = (hamiltonian->omega == 0. ? halo_x : 8);
    int halo_y = (hamiltonian->omega == 0. ? 4 : 8);
  int dimx = grid->global_dim_x + 2 * halo_x * grid->periods[1];
  int dimy = grid->global_dim_y + 2 * halo_y * grid->periods[0];

  Lattice *padded_grid = new Lattice(dimx*grid->delta_x, dimy*grid->delta_y, dimx, dimy, dimx, dimy, grid->periods);
  //padding of the initial state
  double *_p_real = new double [dimx * dimy];
  double *_p_imag = new double [dimx * dimy];
  ini_state_padding(_p_real, _p_imag, state->p_real, state->p_imag, dimx, dimy, halo_x, halo_y, grid->global_dim_x, grid->global_dim_y, grid->periods);
  State *padded_state = new State(padded_grid, _p_real, _p_imag);
    
    //calculate norm of the state
    double norm = 0;
    if(imag_time == true) {
    for(int i = 0; i < grid->global_dim_y; i++) {
      for(int j = 0; j < grid->global_dim_x; j++) {
        norm += grid->delta_x * grid->delta_y * (state->p_real[j + i * grid->global_dim_x] * state->p_real[j + i * grid->global_dim_x] + state->p_imag[j + i * grid->global_dim_x] * state->p_imag[j + i * grid->global_dim_x]);
      }
    }
  }
  
  //calculate parameters for evolution operator
    double *external_pot_real = new double[dimx * dimy];
    double *external_pot_imag = new double[dimx * dimy];
    ini_external_pot(external_pot_real, external_pot_imag, hamiltonian->external_pot, delta_t, dimx, dimy, halo_x, halo_y, grid->global_dim_x, grid->global_dim_y, grid->periods, imag_time);
    
    double h_a, h_b;
    if(imag_time) {
    h_a = cosh(delta_t / (4. * hamiltonian->mass * grid->delta_x * grid->delta_y));
    h_b = sinh(delta_t / (4. * hamiltonian->mass * grid->delta_x * grid->delta_y)); 
    }
    else {
    h_a = cos(delta_t / (4. * hamiltonian->mass * grid->delta_x * grid->delta_y));
    h_b = sin(delta_t / (4. * hamiltonian->mass * grid->delta_x * grid->delta_y));
    }
  
  //launch kernel
  trotter(padded_grid, padded_state, hamiltonian, h_a, h_b, external_pot_real, external_pot_imag, delta_t, iterations, kernel_type, norm, imag_time);

  
  //copy back the final state
  for(int i = 0; i < grid->global_dim_y; i++) {
        for(int j = 0; j < grid->global_dim_x; j++) {
      state->p_real[i * grid->global_dim_x + j] = padded_state->p_real[(i + halo_y * grid->periods[0]) * dimx + j + halo_x * grid->periods[1]];
      state->p_imag[i * grid->global_dim_x + j] = padded_state->p_imag[(i + halo_y * grid->periods[0]) * dimx + j + halo_x * grid->periods[1]];
    }
  }
}


void solver(Lattice *grid, State *state1, State *state2, Hamiltonian2Component *hamiltonian,
      double * external_pot, double * external_pot_b, double delta_t, const int iterations, string kernel_type, bool imag_time) {
  
  int halo_x = (kernel_type == "sse" ? 3 : 4);
    halo_x = (hamiltonian->omega == 0. ? halo_x : 8);
    int halo_y = (hamiltonian->omega == 0. ? 4 : 8);
  int dimx = grid->global_dim_x + 2 * halo_x * grid->periods[1];
  int dimy = grid->global_dim_y + 2 * halo_y * grid->periods[0];
  
  //padding of the initial state
  double *_p_real[2];
  double *_p_imag[2];
  _p_real[0] = new double [dimx * dimy];
  _p_imag[0] = new double [dimx * dimy];    
  _p_real[1] = new double [dimx * dimy];
  _p_imag[1] = new double [dimx * dimy];
  ini_state_padding(_p_real[0], _p_imag[0], state1->p_real, state1->p_imag, dimx, dimy, halo_x, halo_y, grid->global_dim_x, grid->global_dim_y, grid->periods);
  ini_state_padding(_p_real[1], _p_imag[1], state2->p_real, state2->p_imag, dimx, dimy, halo_x, halo_y, grid->global_dim_x, grid->global_dim_y, grid->periods);

  Lattice *padded_grid = new Lattice(dimx*grid->delta_x, dimy*grid->delta_y, dimx, dimy, dimx, dimy, grid->periods);
  State *padded_state1 = new State(padded_grid, _p_real[0], _p_imag[0]);
  State *padded_state2 = new State(padded_grid, _p_real[1], _p_imag[1]);
    
    //calculate norm of the state
    double norm[2];
    norm[0] = 0;
    norm[1] = 0;
    if(imag_time == true) {
    for(int i = 0; i < grid->global_dim_y; i++) {
      for(int j = 0; j < grid->global_dim_x; j++) {
        norm[0] += grid->delta_x * grid->delta_y * (state1->p_real[j + i * grid->global_dim_x] * state1->p_real[j + i * grid->global_dim_x] + state1->p_imag[j + i * grid->global_dim_x] * state1->p_imag[j + i * grid->global_dim_x]);
      }
    }   
    for(int i = 0; i < grid->global_dim_y; i++) {
      for(int j = 0; j < grid->global_dim_x; j++) {
        norm[1] += grid->delta_x * grid->delta_y * (state2->p_real[j + i * grid->global_dim_x] * state2->p_real[j + i * grid->global_dim_x] + state2->p_imag[j + i * grid->global_dim_x] * state2->p_imag[j + i * grid->global_dim_x]);
      }
    }
  }
  
  //calculate parameters for evolution operator
    double *external_pot_real[2];
    double *external_pot_imag[2];
    external_pot_real[0] = new double[dimx * dimy];
    external_pot_imag[0] = new double[dimx * dimy];
  external_pot_real[1] = new double[dimx * dimy];
  external_pot_imag[1] = new double[dimx * dimy];
  ini_external_pot(external_pot_real[0], external_pot_imag[0], external_pot, delta_t, dimx, dimy, halo_x, halo_y, grid->global_dim_x, grid->global_dim_y, grid->periods, imag_time);
  ini_external_pot(external_pot_real[1], external_pot_imag[1], external_pot_b, delta_t, dimx, dimy, halo_x, halo_y, grid->global_dim_x, grid->global_dim_y, grid->periods, imag_time);
  
    double h_a[2], h_b[2];
    if(imag_time) {
    h_a[0] = cosh(delta_t / (4. * hamiltonian->mass * grid->delta_x * grid->delta_y));
    h_b[0] = sinh(delta_t / (4. * hamiltonian->mass * grid->delta_x * grid->delta_y));  
    h_a[1] = cosh(delta_t / (4. * hamiltonian->mass_b * grid->delta_x * grid->delta_y));
    h_b[1] = sinh(delta_t / (4. * hamiltonian->mass_b * grid->delta_x * grid->delta_y));  
    }
    else {
    h_a[0] = cos(delta_t / (4. * hamiltonian->mass * grid->delta_x * grid->delta_y));
    h_b[0] = sin(delta_t / (4. * hamiltonian->mass * grid->delta_x * grid->delta_y));
    h_a[1] = cos(delta_t / (4. * hamiltonian->mass_b * grid->delta_x * grid->delta_y));
    h_b[1] = sin(delta_t / (4. * hamiltonian->mass_b * grid->delta_x * grid->delta_y));
    }
  
  //launch kernel
  trotter(padded_grid, padded_state1, padded_state2, hamiltonian, h_a, h_b, external_pot_real, external_pot_imag, delta_t, iterations, kernel_type, norm, imag_time);
  
  //copy back the final state
  for(int i = 0; i < grid->global_dim_y; i++) {
        for(int j = 0; j < grid->global_dim_x; j++) {
      state1->p_real[i * grid->global_dim_x + j] = _p_real[0][(i + halo_y * grid->periods[0]) * dimx + j + halo_x * grid->periods[1]];
      state1->p_imag[i * grid->global_dim_x + j] = _p_imag[0][(i + halo_y * grid->periods[0]) * dimx + j + halo_x * grid->periods[1]];
    }
  }
  for(int i = 0; i < grid->global_dim_y; i++) {
    for(int j = 0; j < grid->global_dim_x; j++) {
      state2->p_real[i * grid->global_dim_x + j] = _p_real[1][(i + halo_y * grid->periods[0]) * dimx + j + halo_x * grid->periods[1]];
      state2->p_imag[i * grid->global_dim_x + j] = _p_imag[1][(i + halo_y * grid->periods[0]) * dimx + j + halo_x * grid->periods[1]];
    }
  }
}
