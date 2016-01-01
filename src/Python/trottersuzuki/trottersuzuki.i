%module trottersuzuki
%include <std_string.i>
%{
#define SWIG_FILE_WITH_INIT
#include "src/trottersuzuki.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* external_pot, int width_ext_r, int height_ext_r)}
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* external_pot_b, int width_extb_r, int height_extb_r)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* p_real, int width_p_r, int height_p_r)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* p_imag, int width_p_i, int height_p_i)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* pb_real, int width_pb_r, int height_pb_r)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* pb_imag, int width_pb_i, int height_pb_i)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* phase, int width_ph, int height_ph)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* density, int width_de, int height_de)}
%apply (int* IN_ARRAY1, int DIM1) {(int * periods, int size)}
%apply (double* IN_ARRAY1, int DIM1) {(double * coupling, int size_c)}

%inline %{
   void solver(double * p_real, int width_p_r, int height_p_r,
               double * p_imag, int width_p_i, int height_p_i,
               double particle_mass, double coupling_const, double * external_pot, int width_ext_r, int height_ext_r,
               double omega, double rot_coord_x, double rot_coord_y, double delta_x, double delta_y, double delta_t,
               const int iterations, std::string kernel_type,
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
    
    double H_2GPE(double * p_real, int width_p_r, int height_p_r,
                  double * p_imag, int width_p_i, int height_p_i,
                  double * pb_real, int width_pb_r, int height_pb_r,
                  double * pb_imag, int width_pb_i, int height_pb_i,
                  double particle_mass_a, double particle_mass_b, double *coupling, int size_c,
                  double * external_pot, int width_ext_r, int height_ext_r,
                  double * external_pot_b, int width_extb_r, int height_extb_r,
                  double omega, double coord_rot_x, double coord_rot_y,
                  double delta_x, double delta_y) {

        double * real[2], * imag[2], *external_pot2[2];
        real[0] = p_real;
        real[1] = pb_real;
        imag[0] = p_imag;
        imag[1] = pb_imag;
        external_pot2[0] = external_pot;
        external_pot2[1] = external_pot_b;
        return Energy_tot(real, imag,
			       particle_mass_a, particle_mass_b, coupling, 
			       NULL, NULL, external_pot2, 
			       omega, coord_rot_x, coord_rot_y,
			       delta_x, delta_y, 0, 0, 0, width_p_r, width_p_r, 0, 0, height_p_r, height_p_r, width_ext_r, height_ext_r, 0, 0, NULL);
    }
    
    double Norm2_2GPE(double * p_real, int width_p_r, int height_p_r,
                      double * p_imag, int width_p_i, int height_p_i,
                      double * pb_real, int width_pb_r, int height_pb_r,
                      double * pb_imag, int width_pb_i, int height_pb_i,
                      double delta_x, double delta_y) {
        double * real[2], * imag[2];
        real[0] = p_real;
        real[1] = pb_real;
        imag[0] = p_imag;
        imag[1] = pb_imag;
        return Norm2(real, imag, delta_x, delta_y, 0, 0, width_p_r, width_p_r, 0, 0, height_p_r, height_p_r);
    }
    
    void solver_2GPE(double * p_real, int width_p_r, int height_p_r,
               double * p_imag, int width_p_i, int height_p_i,
               double * pb_real, int width_pb_r, int height_pb_r,
               double * pb_imag, int width_pb_i, int height_pb_i,
               double particle_mass_a, double particle_mass_b,
               double * coupling, int size_c, 
               double * external_pot, int width_ext_r, int height_ext_r,
               double * external_pot_b, int width_extb_r, int height_extb_r,
               double omega, double rot_coord_x, double rot_coord_y, double delta_x, double delta_y, double delta_t,
               const int iterations, std::string kernel_type,
               int *periods, int size, bool imag_time) {
       
        solver(p_real, p_imag, pb_real, pb_imag,
               particle_mass_a, particle_mass_b, coupling, external_pot, external_pot_b, omega, rot_coord_x, rot_coord_y,
               width_p_r, height_p_r, delta_x, delta_y, delta_t, iterations, kernel_type, periods, imag_time);
    }
%}
