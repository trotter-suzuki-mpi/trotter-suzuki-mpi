/*ss*/

%module trottersuzuki
%{
#define SWIG_FILE_WITH_INIT
#include "src/trotter.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* external_pot, int width_ext_r, int height_ext_r)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* p_real, int width_p_r, int height_p_r)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* p_imag, int width_p_i, int height_p_i)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* phase, int width_ph, int height_ph)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* density, int width_de, int height_de)}
%apply (int* IN_ARRAY1, int DIM1) {(int * periods, int size)}

%inline %{
   void solver(double * p_real, int width_p_r, int height_p_r,
               double * p_imag, int width_p_i, int height_p_i,
               double particle_mass, double coupling_const, double * external_pot, int width_ext_r, int height_ext_r,
               double omega, double rot_coord_x, double rot_coord_y, double delta_x, double delta_y, double delta_t,
               const int iterations, const int kernel_type,
               int *periods, int size, bool imag_time) {
       
        solver(p_real, p_imag,
               particle_mass, coupling_const, external_pot, omega, rot_coord_x, rot_coord_y,
               width_p_r, height_p_r, delta_x, delta_y, delta_t, iterations, kernel_type, periods, imag_time);
    }

    double H(double * p_real, int width_p_r, int height_p_r,
             double * p_imag, int width_p_i, int height_p_i,
             double particle_mass, double coupling_const, double * external_pot, int width_ext_r, int height_ext_r, double omega, double coord_rot_x, double coord_rot_y,
             double delta_x, double delta_y) {
        
        return Energy_tot(p_real, p_imag, particle_mass, coupling_const, NULL, external_pot, omega, coord_rot_x, coord_rot_y, delta_x, delta_y, 0, 0, 0, width_p_r, width_p_r, 0, 0, height_p_r, height_p_r, width_ext_r, height_ext_r, 0, 0, NULL);
    }

    double K(double * p_real, int width_p_r, int height_p_r,
             double * p_imag, int width_p_i, int height_p_i,
             double particle_mass, double delta_x, double delta_y) {
        return Energy_kin(p_real, p_imag, particle_mass, delta_x, delta_y, 0, 0, 0, width_p_r, width_p_r, 0, 0, height_p_r, height_p_r);
    }

    double Lz(double * p_real, int width_p_r, int height_p_r,
              double * p_imag, int width_p_i, int height_p_i,
              double omega, double coord_rot_x, double coord_rot_y,
              double delta_x, double delta_y){

        return Energy_rot(p_real, p_imag, omega, coord_rot_x, coord_rot_y, delta_x, delta_y, 0, 0, 0, width_p_r, width_p_r, 0, 0, height_p_r, height_p_r);
    }

    double Norm2(double * p_real, int width_p_r, int height_p_r,
                 double * p_imag, int width_p_i, int height_p_i,
                 double delta_x, double delta_y) {
        return Norm2(p_real, p_imag, delta_x, delta_y, 0, 0, width_p_r, width_p_r, 0, 0, height_p_r, height_p_r);
    }

    void phase(double * phase, int width_ph, int height_ph,
               double * p_real, int width_p_r, int height_p_r,
               double * p_imag, int width_p_i, int height_p_i) {

        get_wave_function_phase(phase, p_real, p_imag, 0, 0, width_p_r, width_p_r, 0, 0, height_p_r, height_p_r);
    }

    void density(double * density, int width_de, int height_de,
                 double * p_real, int width_p_r, int height_p_r,
                 double * p_imag, int width_p_i, int height_p_i) {

        get_wave_function_density(density, p_real, p_imag, 0, 0, width_p_r, width_p_r, 0, 0, height_p_r, height_p_r);
    }
%}
