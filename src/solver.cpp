#include <cmath>
#include <complex>

#include "trotter.h"

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

void solver(double * p_real, double * p_imag,
			double particle_mass, double coupling_const, double * external_pot, double omega, int rot_coord_x, int rot_coord_y,
            const int matrix_width, const int matrix_height, double delta_x, double delta_y, double delta_t, const int iterations, string kernel_type, int *periods, bool imag_time) {
	
	int halo_x = (kernel_type == "sse" ? 3 : 4);
    halo_x = (omega == 0. ? halo_x : 8);
    int halo_y = (omega == 0. ? 4 : 8);
	int dimx = matrix_width + 2 * halo_x * periods[1];
	int dimy = matrix_height + 2 * halo_y * periods[0];
	
	//padding of the initial state
	double *_p_real = new double [dimx * dimy];
	double *_p_imag = new double [dimx * dimy];
	ini_state_padding(_p_real, _p_imag, p_real, p_imag, dimx, dimy, halo_x, halo_y, matrix_width, matrix_height, periods);
    
    //calculate norm of the state
    double norm = 0;
    if(imag_time == true) {
		for(int i = 0; i < matrix_height; i++) {
			for(int j = 0; j < matrix_width; j++) {
				norm += delta_x * delta_y * (p_real[j + i * matrix_width] * p_real[j + i * matrix_width] + p_imag[j + i * matrix_width] * p_imag[j + i * matrix_width]);
			}
		}
	}
	
	//calculate parameters for evolution operator
    double *external_pot_real = new double[dimx * dimy];
    double *external_pot_imag = new double[dimx * dimy];
    ini_external_pot(external_pot_real, external_pot_imag, external_pot, delta_t, dimx, dimy, halo_x, halo_y, matrix_width, matrix_height, periods, imag_time);
    
    double h_a, h_b;
    if(imag_time) {
		h_a = cosh(delta_t / (4. * particle_mass * delta_x * delta_y));
		h_b = sinh(delta_t / (4. * particle_mass * delta_x * delta_y));	
    }
    else {
		h_a = cos(delta_t / (4. * particle_mass * delta_x * delta_y));
		h_b = sin(delta_t / (4. * particle_mass * delta_x * delta_y));
    }
	
	//launch kernel
	trotter(h_a, h_b, coupling_const, external_pot_real, external_pot_imag, _p_real, _p_imag, delta_x, delta_y, dimx, dimy, delta_t, iterations, omega, rot_coord_x, rot_coord_y, kernel_type, norm, imag_time, periods);
	
	//copy back the final state
	for(int i = 0; i < matrix_height; i++) {
        for(int j = 0; j < matrix_width; j++) {
			p_real[i * matrix_width + j] = _p_real[(i + halo_y * periods[0]) * dimx + j + halo_x * periods[1]];
			p_imag[i * matrix_width + j] = _p_imag[(i + halo_y * periods[0]) * dimx + j + halo_x * periods[1]];
		}
	}
}


void solver(double * p_real, double * p_imag, double * pb_real, double * pb_imag,
			double particle_mass_a, double particle_mass_b, double *coupling_const, double * external_pot, double * external_pot_b, double omega, int rot_coord_x, int rot_coord_y,
            const int matrix_width, const int matrix_height, double delta_x, double delta_y, double delta_t, const int iterations, string kernel_type, int *periods, bool imag_time) {
	
	int halo_x = (kernel_type == "sse" ? 3 : 4);
    halo_x = (omega == 0. ? halo_x : 8);
    int halo_y = (omega == 0. ? 4 : 8);
	int dimx = matrix_width + 2 * halo_x * periods[1];
	int dimy = matrix_height + 2 * halo_y * periods[0];
	
	//padding of the initial state
	double *_p_real[2];
	double *_p_imag[2];
	_p_real[0] = new double [dimx * dimy];
	_p_imag[0] = new double [dimx * dimy];		
	_p_real[1] = new double [dimx * dimy];
	_p_imag[1] = new double [dimx * dimy];
	ini_state_padding(_p_real[0], _p_imag[0], p_real, p_imag, dimx, dimy, halo_x, halo_y, matrix_width, matrix_height, periods);
	ini_state_padding(_p_real[1], _p_imag[1], pb_real, pb_imag, dimx, dimy, halo_x, halo_y, matrix_width, matrix_height, periods);
    
    //calculate norm of the state
    double norm[2];
    norm[0] = 0;
    norm[1] = 0;
    if(imag_time == true) {
		for(int i = 0; i < matrix_height; i++) {
			for(int j = 0; j < matrix_width; j++) {
				norm[0] += delta_x * delta_y * (p_real[j + i * matrix_width] * p_real[j + i * matrix_width] + p_imag[j + i * matrix_width] * p_imag[j + i * matrix_width]);
			}
		}		
		for(int i = 0; i < matrix_height; i++) {
			for(int j = 0; j < matrix_width; j++) {
				norm[1] += delta_x * delta_y * (pb_real[j + i * matrix_width] * pb_real[j + i * matrix_width] + pb_imag[j + i * matrix_width] * pb_imag[j + i * matrix_width]);
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
	ini_external_pot(external_pot_real[0], external_pot_imag[0], external_pot, delta_t, dimx, dimy, halo_x, halo_y, matrix_width, matrix_height, periods, imag_time);
	ini_external_pot(external_pot_real[1], external_pot_imag[1], external_pot_b, delta_t, dimx, dimy, halo_x, halo_y, matrix_width, matrix_height, periods, imag_time);
  
    double h_a[2], h_b[2];
    if(imag_time) {
		h_a[0] = cosh(delta_t / (4. * particle_mass_a * delta_x * delta_y));
		h_b[0] = sinh(delta_t / (4. * particle_mass_a * delta_x * delta_y));	
		h_a[1] = cosh(delta_t / (4. * particle_mass_b * delta_x * delta_y));
		h_b[1] = sinh(delta_t / (4. * particle_mass_b * delta_x * delta_y));	
    }
    else {
		h_a[0] = cos(delta_t / (4. * particle_mass_a * delta_x * delta_y));
		h_b[0] = sin(delta_t / (4. * particle_mass_a * delta_x * delta_y));
		h_a[1] = cos(delta_t / (4. * particle_mass_b * delta_x * delta_y));
		h_b[1] = sin(delta_t / (4. * particle_mass_b * delta_x * delta_y));
    }
 	
	//launch kernel
	trotter(h_a, h_b, coupling_const, external_pot_real, external_pot_imag, _p_real, _p_imag, delta_x, delta_y, dimx, dimy, delta_t, iterations, omega, rot_coord_x, rot_coord_y, kernel_type, norm, imag_time, periods);
	
	//copy back the final state
	for(int i = 0; i < matrix_height; i++) {
        for(int j = 0; j < matrix_width; j++) {
			p_real[i * matrix_width + j] = _p_real[0][(i + halo_y * periods[0]) * dimx + j + halo_x * periods[1]];
			p_imag[i * matrix_width + j] = _p_imag[0][(i + halo_y * periods[0]) * dimx + j + halo_x * periods[1]];
		}
	}
	for(int i = 0; i < matrix_height; i++) {
		for(int j = 0; j < matrix_width; j++) {
			pb_real[i * matrix_width + j] = _p_real[1][(i + halo_y * periods[0]) * dimx + j + halo_x * periods[1]];
			pb_imag[i * matrix_width + j] = _p_imag[1][(i + halo_y * periods[0]) * dimx + j + halo_x * periods[1]];
		}
	}
}
