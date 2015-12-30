/**
 * Distributed Trotter-Suzuki solver
 * Copyright (C) 2015 Luca Calderaro, 2012-2015 Peter Wittek,
 * 2010-2012 Carlos Bederián
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
#include <string>
#include <cfloat>
using namespace std;
#ifndef __TROTTER_H
#define __TROTTER_H

class Lattice {
public:
    Lattice(double _length_x=20., double _length_y=20., 
            int _dim_x=100, int _dim_y=100, 
            int _global_dim_x=0, int _global_dim_y=0, 
            int _periods[2]=0);
    double length_x, length_y;
    int dim_x, dim_y;
    int global_dim_x, global_dim_y;
    double delta_x, delta_y;
    int periods[2];
  
};

class State{
public:
    double *p_real;
    double *p_imag;

    State(Lattice *_grid, double *_p_real=0, double *_p_imag=0);
    ~State();
    void init_state(std::complex<double> (*ini_state)(int x, int y, Lattice *grid, int halo_x, int halo_y),
                    int start_x, int start_y, int halo_x, int halo_y);
    void read_state(char *file_name, int start_x, int start_y,
                    int *coords, int *dims, int halo_x, int halo_y, int read_offset);

    double calculate_squared_norm(State *psi_b=0);
    double *get_particle_density(double *density=0, 
                                 int inner_start_x=0, int start_x=0, 
                                 int inner_end_x=0, int end_x=0, 
                                 int inner_start_y=0, int start_y=0, 
                                 int inner_end_y=0, int end_y=0);
    double *get_phase(double *phase=0, int inner_start_x=0, int start_x=0, 
                      int inner_end_x=0, int end_x=0, 
                      int inner_start_y=0, int start_y=0, 
                      int inner_end_y=0, int end_y=0);

private:
    Lattice *grid;
    bool self_init;
};


class Hamiltonian {
public:
    Lattice *grid;
    double mass;
    double coupling_a;
    double coupling_ab;
    double angular_velocity;
    double rot_coord_x;
    double rot_coord_y;
    double omega;
    double *external_pot;
    bool self_init;
    
    Hamiltonian(Lattice *_grid, double _mass=1., double _coupling_a=0., 
                double coupling_ab=0., double _angular_velocity=0., 
                double _rot_coord_x=DBL_MAX, double _rot_coord_y=DBL_MAX, 
                double _omega=0.,
                double *_external_pot=0);
    ~Hamiltonian();
    void initialize_potential(double (*hamiltonian_pot)(int x, int y, Lattice *grid, int halo_x, int halo_y),
                              int halo_x, int halo_y);
    
};

class Hamiltonian2Component: public Hamiltonian {
public:
    double mass_b;
    double coupling_b;
    double *external_pot_b;

    Hamiltonian2Component(Lattice *_grid, double _mass=1., double _mass_b=1., 
                          double _coupling_a=0., double coupling_ab=0., 
                          double _coupling_b=0.,
                          double _angular_velocity=0., 
                          double _rot_coord_x=DBL_MAX, 
                          double _rot_coord_y=DBL_MAX, 
                          double _omega=0,
                          double *_external_pot=0, 
                          double *_external_pot_b=0);
    ~Hamiltonian2Component();
    void initialize_potential_b(double (*hamiltonian_pot)(int x, int y, Lattice *grid, int halo_x, int halo_y),
                                int halo_x, int halo_y);
    
};


/**
    API call to calculate the evolution through the Trotter-Suzuki decomposition.

    @param h_a               Kinetic term of the Hamiltonian (cosine part)
    @param h_b               Kinetic term of the Hamiltonian (sine part)
    @param external_pot_real External potential, real part
    @param external_pot_imag External potential, imaginary part
    @param p_real            Initial state, real part
    @param p_imag            Initial state, imaginary part
    @param matrix_width      The width of the initial state
    @param matrix_height     The height of the initial state
    @param iterations        Number of iterations to be calculated
    @param snapshots         Number of iterations between taking snapshots
                             (0 means no snapshots)
    @param kernel_type       The kernel type: "cpu", "gpu", or "hybrid"
    @param periods            Whether the grid is periodic in any of the directions
    @param output_folder      The folder to write the snapshots in
    @param verbose            Optional verbosity parameter
    @param imag_time          Optional parameter to calculate imaginary time evolution
    @param particle_tag       Optional parameter to tag a particle in the snapshots

*/

void trotter(Lattice *grid, State *state, Hamiltonian *hamiltonian,
             double h_a, double h_b, 
             double * external_pot_real, double * external_pot_imag,
             double delta_t,
             const int iterations,
             string kernel_type = "cpu", double norm = 1., bool imag_time = false);             

void trotter(Lattice *grid, State *state1, State *state2, 
             Hamiltonian2Component *hamiltonian,
             double *h_a, double *h_b, 
             double **external_pot_real, double **external_pot_imag,
             double delta_t,
             const int iterations, 
             string kernel_type, double *norm, bool imag_time);

void solver(Lattice *grid, State *state, Hamiltonian *hamiltonian,
            double delta_t, const int iterations, string kernel_type, bool imag_time);

void solver(double * p_real, double * p_imag, double * pb_real, double * pb_imag,
			double particle_mass_a, double particle_mass_b, double *coupling_const, double * external_pot, double * external_pot_b, double omega, int rot_coord_x, int rot_coord_y,
            const int matrix_width, const int matrix_height, double delta_x, double delta_y, double delta_t, const int iterations, string kernel_type, int *periods, bool imag_time);
            
/**
 * \brief Structure defining expected values calculated by expect_values().
 */
struct energy_momentum_statistics {
    double mean_E;	///< Expected total energy.
    double mean_Px; ///< Expected momentum along x axis.
    double mean_Py; ///< Expected momentum along y axis.
    double var_E;	///< Expected total energy variation.
    double var_Px;	///< Expected momentum along x axis variation.
    double var_Py;	///< Expected momentum along y axis variation.
    energy_momentum_statistics() : mean_E(0.), mean_Px(0.), mean_Py(0.),
        var_E(0.), var_Px(0.), var_Py(0.) {}
};

double Energy_rot(double * p_real, double * p_imag,
				  double omega, double coord_rot_x, double coord_rot_y, double delta_x, double delta_y,
				  double norm2, int inner_start_x, int start_x, int inner_end_x, int end_x, int inner_start_y, int start_y, int inner_end_y, int end_y);
double Energy_kin(double * p_real, double * p_imag, double particle_mass, double delta_x, double delta_y,
                  double norm2, int inner_start_x, int start_x, int inner_end_x, int end_x, int inner_start_y, int start_y, int inner_end_y, int end_y);
double Energy_tot(double * p_real, double * p_imag,
				  double particle_mass, double coupling_const, double (*hamilt_pot)(int x, int y, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y), double * external_pot, double omega, double coord_rot_x, double coord_rot_y,
				  double delta_x, double delta_y, double norm2, int inner_start_x, int start_x, int inner_end_x, int end_x, int inner_start_y, int start_y, int inner_end_y, int end_y,
				  int matrix_width, int matrix_height, int halo_x, int halo_y, int * periods);
double Energy_tot(double ** p_real, double ** p_imag,
				       double particle_mass_a, double particle_mass_b, double *coupling_const, 
				       double (*hamilt_pot_a)(int x, int y, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y),
				       double (*hamilt_pot_b)(int x, int y, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y), 
				       double ** external_pot, 
				       double omega, double coord_rot_x, double coord_rot_y,
				       double delta_x, double delta_y, double norm2, int inner_start_x, int start_x, int inner_end_x, int end_x, int inner_start_y, int start_y, int inner_end_y, int end_y,
				       int matrix_width, int matrix_height, int halo_x, int halo_y, int * periods);
void expect_values(int dim, int iterations, int snapshots, double * hamilt_pot, double particle_mass,
                   const char *dirname, int *periods, int halo_x, int halo_y, energy_momentum_statistics *sample);
#endif
