/*ss*/

%module trottersuzuki
%{
#define SWIG_FILE_WITH_INIT
#include "trotter.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* external_pot, int width_ext_r, int height_ext_r)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* p_real, int width_p_r, int height_p_r)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* p_imag, int width_p_i, int height_p_i)}
%apply (int* IN_ARRAY1, int DIM1) {(int * periods, int size)}

%inline %{
   void solver(double * p_real, int width_p_r, int height_p_r,
             double * p_imag, int width_p_i, int height_p_i,
             double particle_mass, double coupling_const, double * external_pot, int width_ext_r, int height_ext_r,
             double delta_x, double delta_y, double delta_t,
             const int iterations, const int kernel_type,
             int *periods, int size, bool imag_time) {
       
       solver(p_real, p_imag,
             particle_mass, coupling_const, external_pot,
             width_p_r, height_p_r, delta_x, delta_y, delta_t, iterations, kernel_type, periods, imag_time);
    }

    double H(double * p_real, int width_p_r, int height_p_r,
			double * p_imag, int width_p_i, int height_p_i,
			double particle_mass, double coupling_const, double * external_pot, int width_ext_r, int height_ext_r,
			double delta_x, double delta_y) {

    	return Energy_tot(p_real, p_imag, particle_mass, coupling_const, external_pot, width_ext_r, height_ext_r, delta_x, delta_y);
    }

    double K(double * p_real, int width_p_r, int height_p_r,
			double * p_imag, int width_p_i, int height_p_i,
			double particle_mass, double delta_x, double delta_y) {
    	return Energy_kin(p_real, p_imag, particle_mass, width_p_r, height_p_r, delta_x, delta_y);
    }

    double Norm2(double * p_real, int width_p_r, int height_p_r,
			double * p_imag, int width_p_i, int height_p_i,
			double delta_x, double delta_y) {
    	return Norm2(p_real, p_imag, width_p_r, height_p_r, delta_x, delta_y);
    }
%}
