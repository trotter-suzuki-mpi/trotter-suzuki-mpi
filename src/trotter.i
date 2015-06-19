%module trottersuzuki
%{
#define SWIG_FILE_WITH_INIT
#include "trotter.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* external_pot_real, int matrix_width, int matrix_height)}
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* external_pot_imag, int matrix_width, int matrix_height)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* p_real, int matrix_width, int matrix_height)}
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* p_imag, int matrix_width, int matrix_height)}
%include "trotter.h"
