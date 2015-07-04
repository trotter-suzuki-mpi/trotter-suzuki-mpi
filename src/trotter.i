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

%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* external_pot_real, int width_ext_r, int height_ext_r)}
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* external_pot_imag, int width_ext_i, int height_ext_i)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* p_real, int width_p_r, int height_p_r)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* p_imag, int width_p_i, int height_p_i)}
%apply (int* IN_ARRAY1, int DIM1) {(int * periods, int size)}
%apply (int* INPLACE_ARRAY1, int DIM1) {(int * time, int size_2)}

%inline %{
   void trotter_func(double h_a, double h_b,
             double * external_pot_real, int width_ext_r, int height_ext_r,
             double * external_pot_imag, int width_ext_i, int height_ext_i,
             double * p_real, int width_p_r, int height_p_r,
	     double * p_imag, int width_p_i, int height_p_i, 
             const int iterations, const int kernel_type,
             int *periods, int size, bool imag_time, int * time, int size_2) {
       
       trotter(h_a, h_b,
             external_pot_real, external_pot_imag,
             p_real, p_imag, 
             width_ext_r, height_ext_i, 
             iterations, kernel_type,
             periods, imag_time, time);
    }
%}
