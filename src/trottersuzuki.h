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
    Lattice(int dim=100, double length_x=20., double length_y=20.,
            bool periodic_x_axis=false, bool periodic_y_axis=false, double omega=0.);
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

    State(Lattice *grid, double *p_real=0, double *p_imag=0);
    ~State();
    void init_state(complex<double> (*ini_state)(double x, double y));
    void read_state(char *file_name, int read_offset);

    double calculate_squared_norm(bool global=true);
    double *get_particle_density(double *density=0);
    double *get_phase(double *phase=0);
    double get_squared_norm(void);
    double get_mean_x(void);
    double get_mean_xx(void);
    double get_mean_y(void);
    double get_mean_yy(void);
    double get_mean_px(void);
    double get_mean_pxpx(void);
    double get_mean_py(void);
    double get_mean_pypy(void);
    void write_to_file(string fileprefix);
    void write_particle_density(string fileprefix);
    void write_phase(string fileprefix);
    bool expected_values_updated;

protected:
    Lattice *grid;
    bool self_init;
    void calculate_expected_values(void);
    double mean_X, mean_XX;
    double mean_Y, mean_YY;
    double mean_Px, mean_PxPx;
    double mean_Py, mean_PyPy;
    double norm2;
};


class ExponentialState: public State {
public:
    ExponentialState(Lattice *grid, int n_x=1, int n_y=1, double norm=1, double phase=0, double *p_real=0, double *p_imag=0);

private:
    int n_x, n_y;
    double norm, phase;
    complex<double> exp_state(double x, double y);
};

class GaussianState: public State {
public:
    GaussianState(Lattice *grid, double omega, double mean_x=0, double mean_y=0, double norm=1, double phase=0,
                  double *p_real=0, double *p_imag=0);

private:
    double mean_x, mean_y, omega, norm, phase;
    complex<double> gauss_state(double x, double y);
};

class SinusoidState: public State {
public:
    SinusoidState(Lattice *grid, int n_x=1, int n_y=1, double norm=1, double phase=0, double *p_real=0, double *p_imag=0);

private:
    int n_x, n_y;
    double norm, phase;
    complex<double> sinusoid_state(double x, double y);
};


class Potential {
public:
    Potential(Lattice *grid, char *filename);
    Potential(Lattice *grid, double *external_pot);
    Potential(Lattice *grid, double (*potential_function)(double x, double y));
    Potential(Lattice *grid, double (*potential_function)(double x, double y, double t), int t=0);
    ~Potential();
    virtual double get_value(int x, int y);
    bool update(double t);

protected:
    double current_evolution_time;
    Lattice *grid;
    double (*static_potential)(double x, double y);
    double (*evolving_potential)(double x, double y, double t);
    double *matrix;
    bool self_init;
    bool is_static;
};

class ParabolicPotential: public Potential {
public:
    ParabolicPotential(Lattice *grid, double omegax, double omegay, double mass=1., double mean_x = 0., double mean_y = 0.);
    ~ParabolicPotential();
    double get_value(int x, int y);

private:
    double omegax, omegay;
    double mean_x, mean_y;
    double mass;
};

class Hamiltonian {
public:
    Potential *potential;
    double mass;
    double coupling_a;
    double angular_velocity;
    double rot_coord_x;
    double rot_coord_y;

    Hamiltonian(Lattice *grid, Potential *potential=0, double mass=1., double coupling_a=0.,
                double angular_velocity=0.,
                double rot_coord_x=0, double rot_coord_y=0);
    ~Hamiltonian();

protected:
    bool self_init;
    Lattice *grid;
};

class Hamiltonian2Component: public Hamiltonian {
public:
    double mass_b;
    double coupling_ab;
    double coupling_b;
    double omega_r;
    double omega_i;
    Potential *potential_b;

    Hamiltonian2Component(Lattice *grid, Potential *potential=0,
                          Potential *potential_b=0,
                          double mass=1., double mass_b=1.,
                          double coupling_a=0., double coupling_ab=0.,
                          double coupling_b=0.,
                          double omega_r=0, double omega_i=0,
                          double angular_velocity=0.,
                          double rot_coord_x=0,
                          double rot_coord_y=0);
    ~Hamiltonian2Component();
};


/**
 * \brief This class define the prototipe of the kernel classes: CPU, GPU, Hybrid.
 */

class ITrotterKernel {
public:
    virtual ~ITrotterKernel() {};
    virtual void run_kernel() = 0;							///< Evolve the remaining blocks in the inner part of the tile.
    virtual void run_kernel_on_halo() = 0;					///< Evolve blocks of wave function at the edge of the tile. This comprises the halos.
    virtual void wait_for_completion() = 0;	                ///< Sincronize all the processes at the end of halos communication. Perform normalization for imaginary time evolution.
    virtual void get_sample(size_t dest_stride, size_t x, size_t y, size_t width, size_t height, double * dest_real, double * dest_imag, double * dest_real2=0, double * dest_imag2=0) const = 0;					///< Get the evolved wave function.
    virtual void normalization() = 0;
    virtual void rabi_coupling(double var, double delta_t) = 0;
    virtual double calculate_squared_norm(bool global=true) = 0;
    virtual bool runs_in_place() const = 0;
    virtual string get_name() const = 0;				///< Get kernel name.
    virtual void update_potential(double *_external_pot_real, double *_external_pot_imag) = 0;

    virtual void start_halo_exchange() = 0;					///< Exchange halos between processes.
    virtual void finish_halo_exchange() = 0;				///< Exchange halos between processes.

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
    double current_evolution_time;
    Solver(Lattice *grid, State *state, Hamiltonian *hamiltonian, double delta_t,
           string kernel_type="cpu");
    Solver(Lattice *grid, State *state1, State *state2,
           Hamiltonian2Component *hamiltonian,
           double delta_t, string kernel_type="cpu");
    ~Solver();
    void evolve(int iterations, bool imag_time=false);
    double get_total_energy(void);
    double get_squared_norm(size_t which=3);
    double get_kinetic_energy(size_t which=3);
    double get_potential_energy(size_t which=3);
    double get_rotational_energy(size_t which=3);
    double get_intra_species_energy(size_t which=3);
    double get_inter_species_energy(void);
    double get_rabi_energy(void);
private:
    bool imag_time;
    double h_a[2];
    double h_b[2];
    double **external_pot_real;
    double **external_pot_imag;
    double delta_t;
    double norm2[2];
    bool single_component;
    string kernel_type;
    ITrotterKernel * kernel;
    void initialize_exp_potential(double time_single_it, int which);
    void init_kernel();
    double total_energy;
    double kinetic_energy[2];
    double tot_kinetic_energy;
    double potential_energy[2];
    double tot_potential_energy;
    double rotational_energy[2];
    double tot_rotational_energy;
    double intra_species_energy[2];
    double tot_intra_species_energy;
    double inter_species_energy;
    double rabi_energy;
    bool energy_expected_values_updated;
    void calculate_energy_expected_values(void);
};

double const_potential(double x, double y);

#endif // __TROTTERSUZUKI_H
