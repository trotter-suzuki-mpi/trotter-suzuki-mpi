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
#ifndef __TROTTERSUZUKI_H
#define __TROTTERSUZUKI_H

#include <string>
#include <cfloat>
#include <complex>
#if HAVE_CONFIG_H
#include "config.h"
#endif
#ifdef HAVE_MPI
#include <mpi.h>
#endif

using namespace std;

class Lattice {
public:
    Lattice(int _dim=100, double _length_x=20., double _length_y=20., 
            int _periods[2]=0, double omega=0.);
    double length_x, length_y;
    double delta_x, delta_y;
    int dim_x, dim_y;
    int global_dim_x, global_dim_y;
    int periods[2];
    
    // Computational topology
    int halo_x, halo_y;
    int start_x, end_x, inner_start_x, inner_end_x,
        start_y, end_y, inner_start_y, inner_end_y;
    int mpi_coords[2], mpi_dims[2];
    int mpi_rank;
    int mpi_procs;
#ifdef HAVE_MPI
    MPI_Comm cartcomm;
#endif  
};

class State{
public:
    double *p_real;
    double *p_imag;

    State(Lattice *_grid, double *_p_real=0, double *_p_imag=0);
    ~State();
    void init_state(complex<double> (*ini_state)(double x, double y));
    void read_state(char *file_name, int read_offset);

    double calculate_squared_norm(bool global=true);
    double *get_particle_density(double *density=0);
    double *get_phase(double *phase=0);

private:
    Lattice *grid;
    bool self_init;
};


class Hamiltonian {
public:
    Lattice *grid;
    double mass;
    double coupling_a;
    double angular_velocity;
    double rot_coord_x;
    double rot_coord_y;
    double *external_pot;
    bool self_init;
    
    Hamiltonian(Lattice *_grid, double _mass=1., double _coupling_a=0., 
                double _angular_velocity=0., 
                double _rot_coord_x=DBL_MAX, double _rot_coord_y=DBL_MAX, 
                double *_external_pot=0);
    ~Hamiltonian();
    void initialize_potential(double (*hamiltonian_pot)(double x, double y));
    void read_potential(char *pot_name);    
};

class Hamiltonian2Component: public Hamiltonian {
public:
    double mass_b;
    double coupling_ab;
    double coupling_b;
    double *external_pot_b;
    double omega_r;
    double omega_i;
    
    Hamiltonian2Component(Lattice *_grid, double _mass=1., double _mass_b=1., 
                          double _coupling_a=0., double coupling_ab=0., 
                          double _coupling_b=0.,
                          double _omega_r=0, double _omega_i=0,
                          double _angular_velocity=0., 
                          double _rot_coord_x=DBL_MAX, 
                          double _rot_coord_y=DBL_MAX, 
                          double *_external_pot=0, 
                          double *_external_pot_b=0);
    ~Hamiltonian2Component();
    void initialize_potential(double (*hamiltonian_pot)(double x, double y), int which);
    
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

class Solver {
public:
    Lattice *grid;
    State *state;
    State *state_b;
    Hamiltonian *hamiltonian;
    Solver(Lattice *grid, State *state, Hamiltonian *hamiltonian, double delta_t, 
           string kernel_type="cpu");
    Solver(Lattice *_grid, State *state1, State *state2, 
           Hamiltonian2Component *_hamiltonian,
           double _delta_t, string _kernel_type="cpu");
    ~Solver();
    void evolve(int iterations, bool imag_time=false);
private:
    bool imag_time;
    double h_a[2];
    double h_b[2];
    double **external_pot_real;
    double **external_pot_imag;
    double delta_t;
    double norm2[2];
    bool single_component;
    bool first_run;
    string kernel_type;
    void initialize_exp_potential(double time_single_it, int which);
};

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
double calculate_rotational_energy(Lattice *grid, State *state, 
                                   Hamiltonian *hamiltonian, double norm2, 
                                   bool global=true);
double calculate_kinetic_energy(Lattice *grid, State *state, 
                                Hamiltonian *hamiltonian, double norm2, 
                                bool global=true);
double calculate_total_energy(Lattice *grid, State *state, 
                              Hamiltonian *hamiltonian, 
                              double (*hamilt_pot)(double x, double y)=0, 
                              double * external_pot=0, double norm2=0, bool global=true);
double calculate_total_energy(Lattice *grid, State *state1, State *state2, 
                              Hamiltonian2Component *hamiltonian,
                              double (*hamilt_pot_a)(double x, double y)=0,
                              double (*hamilt_pot_b)(double x, double y)=0, 
                              double **external_pot=0, double norm2=0, bool global=true);
void calculate_mean_position(Lattice *grid, State *state, int grid_origin_x, int grid_origin_y, 
                             double *results, double norm2=0);
void calculate_mean_momentum(Lattice *grid, State *state, double *results,
                             double norm2=0);
void expect_values(int dim, int iterations, int snapshots, double * hamilt_pot, double particle_mass,
                   const char *dirname, int *periods, int halo_x, int halo_y, energy_momentum_statistics *sample);
void initialize_exp_potential(Lattice *grid, double * external_pot_real, double * external_pot_imag, double (*hamilt_pot)(int x, int y, Lattice *grid),
                              double time_single_it, double particle_mass, bool imag_time);
double const_potential(double x, double y);
void stamp(Lattice *grid, State *state, int tag_particle, int iterations, int count_snap, const char * output_folder);
void stamp_real(Lattice *grid, double *matrix, int iterations, const char * output_folder, const char * file_tag);

#endif // __TROTTERSUZUKI_H
