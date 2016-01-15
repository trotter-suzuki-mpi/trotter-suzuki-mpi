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
    ExponentialState(Lattice *_grid, int _n_x=1, int _n_y=1, double _norm=1, double _phase=0, double *_p_real=0, double *_p_imag=0);

private:
    int n_x, n_y;
    double norm, phase;
    complex<double> exp_state(double x, double y);
};

class GaussianState: public State {
public:
    GaussianState(Lattice *_grid, double _omega, double _mean_x=0, double _mean_y=0, double _norm=1, double _phase=0,
                  double *_p_real=0, double *_p_imag=0);

private:
    double mean_x, mean_y, omega, norm, phase;
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
    Potential(Lattice *_grid, char *filename);
    Potential(Lattice *_grid, double *_external_pot);
    Potential(Lattice *_grid, double (*potential_function)(double x, double y));
    Potential(Lattice *_grid, double (*potential_function)(double x, double y, double t), int _t=0);
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
    ParabolicPotential(Lattice *_grid, double _omegax, double _omegay, double _mass=1., double _mean_x = 0., double _mean_y = 0.);
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
           string kernel_type="cpu");
    Solver(Lattice *_grid, State *state1, State *state2,
           Hamiltonian2Component *_hamiltonian,
           double _delta_t, string _kernel_type="cpu");
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
