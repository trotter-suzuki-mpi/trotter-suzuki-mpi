%module trottersuzuki
%include <std_string.i>
%include "docstring.i"
%{
#define SWIG_FILE_WITH_INIT
#include "src/trottersuzuki.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* state_real, int state_real_width, int state_real_height)}
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* state_imag, int state_imag_width, int state_imag_height)}
%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* _potential, int _potential_width, int _potential_height)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* p_real, int p_r_width, int p_r_height)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double* p_imag, int p_i_width, int p_i_height)}
%apply (double** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {(double **density_out, int *de_dim1_out, int *de_dim2_out)}
%apply (double** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {(double **phase_out, int *ph_dim1_out, int *ph_dim2_out)}

%exception Solver::init_kernel {
   try {
      $action
   } catch (runtime_error &e) {
      PyErr_SetString(PyExc_RuntimeError, const_cast<char*>(e.what()));
      return NULL;
   }
}

class Lattice {
public:
    Lattice(int dim=100, double length=20.,
            bool periodic_x_axis=false, bool periodic_y_axis=false, double angular_velocity=0.);
    double length_x, length_y;
    double delta_x, delta_y;
    int dim_x, dim_y;
    int global_no_halo_dim_x, global_no_halo_dim_y;
    int global_dim_x, global_dim_y;
    int periods[2];

    int halo_x;
    int halo_y;
    int start_x, start_y;
    int end_x, end_y;
    int inner_start_x, inner_start_y;
    int inner_end_x, inner_end_y;
    int mpi_coords[2], mpi_dims[2];
    int mpi_rank;
    int mpi_procs;
};

class State{
public:
    double *p_real;
    double *p_imag;
    Lattice *grid;

    State(Lattice *grid, double *p_real=0, double *p_imag=0);
    State(const State &obj /**< [in] State object. */);
    ~State();
    %extend {
        void init_state_matrix(double* state_real, int state_real_width, int state_real_height,
                        double* state_imag, int state_imag_width, int state_imag_height) {
            //check that p_real and p_imag have been allocated

            for (int y = 0; y < self->grid->dim_y; y++) {
                for (int x = 0; x < self->grid->dim_x; x++) {
                    self->p_real[y * self->grid->dim_x + x] = state_real[y * self->grid->dim_x + x];
                    self->p_imag[y * self->grid->dim_x + x] = state_imag[y * self->grid->dim_x + x];
                }
            }
        }
    }
    void loadtxt(char *file_name /**< [in] Name of the file. */);
    %extend {
        void imprint_matrix(double* state_real, int state_real_width, int state_real_height,
                            double* state_imag, int state_imag_width, int state_imag_height) {
            for (int y = 0; y < self->grid->dim_y; y++) {
                for (int x = 0; x < self->grid->dim_x; x++) {
                    double tmp = self->p_real[y * self->grid->dim_x + x];
                    self->p_real[y * self->grid->dim_x + x] = self->p_real[y * self->grid->dim_x + x] * state_real[y * self->grid->dim_x + x] -
                                                              self->p_imag[y * self->grid->dim_x + x] * state_imag[y * self->grid->dim_x + x];
                    self->p_imag[y * self->grid->dim_x + x] = tmp * state_imag[y * self->grid->dim_x + x] +
                                                              self->p_imag[y * self->grid->dim_x + x] * state_real[y * self->grid->dim_x + x];
                }
            }
            self->expected_values_updated = false;
        }
    }
    %extend {
        void get_particle_density(double **density_out, int *de_dim1_out, int *de_dim2_out) {
            double *_density;
            _density = self->get_particle_density();
        end:
           *de_dim1_out = self->grid->dim_x;
           *de_dim2_out = self->grid->dim_y;
           *density_out = _density;
	    }
    }
    %extend {
        void get_phase(double **phase_out, int *ph_dim1_out, int *ph_dim2_out) {
            double *_phase;
            _phase = self->get_phase();
        end:
           *ph_dim1_out = self->grid->dim_x;
           *ph_dim2_out = self->grid->dim_y;
           *phase_out = _phase;
        }
    }
    double get_squared_norm(void);
    double get_mean_x(void);
    double get_mean_xx(void);
    double get_mean_y(void);
    double get_mean_yy(void);
    double get_mean_px(void);
    double get_mean_pxpx(void);
    double get_mean_py(void);
    double get_mean_pypy(void);
    void write_to_file(std::string fileprefix /** [in] prefix name of the file */);
    void write_particle_density(std::string fileprefix /** [in] prefix name of the file */);
    void write_phase(std::string fileprefix /** [in] prefix name of the file */);
    bool expected_values_updated;

protected:

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
    ExponentialState(Lattice *_grid, int _n_x=1, int _n_y=1, double _norm=1, double _phase=0, double *_p_real=0, double *_p_imag=0);

private:
    int n_x, n_y;
    double norm, phase;
    complex<double> exp_state(double x, double y);
};

class GaussianState: public State {
public:
    GaussianState(Lattice *grid, double omega_x, double omega_y = -1., double mean_x = 0, double mean_y = 0, double norm = 1, double phase = 0,
                  double *p_real = 0, double *p_imag = 0);

private:
    double mean_x;
    double mean_y;
    double omega_x;
    double omega_y;
    double norm;
    double phase;
    complex<double> gauss_state(double x, double y);
};

class SinusoidState: public State {
public:
    SinusoidState(Lattice *_grid, int _n_x=1, int _n_y=1, double _norm=1, double _phase=0, double *_p_real=0, double *_p_imag=0);

private:
    int n_x, n_y;
    double norm, phase;
    complex<double> sinusoid_state(double x, double y);
};

class Potential {
public:
    Lattice *grid;    ///< Object that defines the lattice structure.
    double *matrix;    ///< Matrix storing the potential.

    Potential(Lattice *grid, char *filename);
    Potential(Lattice *grid, double *external_pot=0);
    Potential(Lattice *grid, double (*potential_function)(double x, double y));
    Potential(Lattice *grid, double (*potential_function)(double x, double y, double t), int t=0);
    ~Potential();
    %extend {
        void init_potential_matrix(double* _potential, int _potential_width, int _potential_height) {
            for (int y = 0; y < self->grid->dim_y; y++) {
                for (int x = 0; x < self->grid->dim_x; x++) {
                    self->matrix[y * self->grid->dim_x + x] = _potential[y * self->grid->dim_x + x];
                }
            }
        }
    }
    virtual double get_value(int x, int y);
    bool update(double t);

protected:
    double current_evolution_time;
    double (*static_potential)(double x, double y);
    double (*evolving_potential)(double x, double y, double t);
    bool self_init;
    bool is_static;
};

class HarmonicPotential: public Potential {
public:

    HarmonicPotential(Lattice *_grid, double _omegax, double _omegay, double _mass=1., double _mean_x = 0., double _mean_y = 0.);
    ~HarmonicPotential();
    double get_value(int x, int y);

private:
    double omegax, omegay;
    double mass;
    double mean_x, mean_y;
};

class Hamiltonian {
public:
    Potential *potential;
    double mass;
    double coupling_a;
    double angular_velocity;
    double rot_coord_x;
    double rot_coord_y;
    Hamiltonian(Lattice *_grid, Potential *_potential=0, double _mass=1., double _coupling_a=0.,
                double _angular_velocity=0.,
                double _rot_coord_x=0, double _rot_coord_y=0);
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

    Hamiltonian2Component(Lattice *_grid, Potential *_potential=0,
                          Potential *_potential_b=0,
                          double _mass=1., double _mass_b=1.,
                          double _coupling_a=0., double coupling_ab=0.,
                          double _coupling_b=0.,
                          double _omega_r=0, double _omega_i=0,
                          double _angular_velocity=0.,
                          double _rot_coord_x=0,
                          double _rot_coord_y=0);
    ~Hamiltonian2Component();
};

class Solver {
public:
    Lattice *grid;
    State *state;
    State *state_b;
    Hamiltonian *hamiltonian;
    double current_evolution_time;

    Solver(Lattice *grid, State *state, Hamiltonian *hamiltonian, double delta_t,
           std::string kernel_type="cpu");
    Solver(Lattice *grid, State *state1, State *state2,
           Hamiltonian2Component *hamiltonian,
           double delta_t, std::string kernel_type="cpu");
    ~Solver();
    void evolve(int iterations, bool imag_time=false);
    void update_parameters();
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
    std::string kernel_type;
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
    bool has_parameters_changed;
    bool energy_expected_values_updated;
    void calculate_energy_expected_values(void);
};
