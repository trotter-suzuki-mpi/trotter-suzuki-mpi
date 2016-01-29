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
    /**
        Lattice constructor.

        @param [in] dim              Linear dimension of the squared lattice.
        @param [in] length_x         Physical length of the lattice's side along the x axis.
        @param [in] length_y         Physical length of the lattice's side along the y axis.
        @param [in] periodic_x_axis  Boundary condition along the x axis (false=closed, true=periodic).
        @param [in] periodic_y_axis  Boundary condition along the y axis (false=closed, true=periodic).
        @param [in] angular_velocity Angular velocity of the frame of reference.
     */
    Lattice(int dim=100, double length_x=20., double length_y=20.,
            bool periodic_x_axis=false, bool periodic_y_axis=false, double angular_velocity=0.);
    double length_x, length_y;    ///< Physical length of the lattice's sides.
    double delta_x, delta_y;    ///< Physical distance between two consecutive point of the grid, along the x and y axes.
    int dim_x, dim_y;    ///< Linear dimension of the tile along x and y axes.
    int global_no_halo_dim_x, global_no_halo_dim_y;    ///< Linear dimension of the lattice, excluding the eventual surrounding halo, along x and y axes.
    int global_dim_x, global_dim_y;    ///< Linear dimension of the lattice, comprising the eventual surrounding halo, along x and y axes.
    int periods[2];    ///< Whether the grid is periodic in any of the directions.

    // Computational topology
    int halo_x, halo_y;    ///< Halo length along the x and y halos.
    int start_x, start_y;    ///< Spatial coordinates (not physical) of the first element of the tile.
    int end_x, end_y;    ///< Spatial coordinates (not physical) of the last element of the tile.
    int inner_start_x, inner_start_y;    ///< Spatial coordinates (not physical) of the first element of the tile, excluding the eventual surrounding halo.
    int inner_end_x, inner_end_y;    ///< Spatial coordinates (not physical) of the last element of the tile, excluding the eventual surrounding halo.
    int mpi_coords[2], mpi_dims[2];    ///< Coordinate of the process in the MPI topology and structure of the MPI topology.
    int mpi_rank;    ///< Rank of the process in the MPI topology.
    int mpi_procs;    ///< Number of processes in the MPI topology.
};

class State{
public:
    double *p_real;    ///< Real part of the wave function.
    double *p_imag;    ///< Imaginary part of the wave function.
    Lattice *grid;    ///< Object that defines the lattice structure.

    /**
        Construct the state from given matrices if they are provided, otherwise construct a state with null wave function, initializing p_real and p_imag.

        @param [in] grid             Lattice object.
        @param [in] p_real           Pointer to the real part of the wave function.
        @param [in] p_imag           Pointer to the imaginary part of the wave function.
     */
    State(Lattice *grid, double *p_real=0, double *p_imag=0);
    State(const State &obj /**< [in] State object. */);    ///< Copy constructor: copy the state object.
    ~State();    ///< Destructor.
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
    void loadtxt(char *file_name /**< [in] Name of the file. */);    ///< Load the wave function from a file.
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
    double get_squared_norm(void);    ///< Return the squared norm of the quantum state.
    double get_mean_x(void);    ///< Return the expected value of the X operator.
    double get_mean_xx(void);    ///< Return the expected value of the X^2 operator.
    double get_mean_y(void);    ///< Return the expected value of the Y operator.
    double get_mean_yy(void);    ///< Return the expected value of the Y^2 operator.
    double get_mean_px(void);    ///< Return the expected value of the P_x operator.
    double get_mean_pxpx(void);    ///< Return the expected value of the P_x^2 operator.
    double get_mean_py(void);    ///< Return the expected value of the P_y operator.
    double get_mean_pypy(void);    ///< Return the expected value of the P_y^2 operator.
    void write_to_file(std::string fileprefix /** [in] prefix name of the file */);    ///< Write to a file the wave function.
    void write_particle_density(std::string fileprefix /** [in] prefix name of the file */);    ///< Write to a file the squared norm of the wave function.
    void write_phase(std::string fileprefix /** [in] prefix name of the file */);    ///< Write to a file the phase of the wave function.
    bool expected_values_updated;    ///< Whether the expected values of the state object are updated with respect to the last evolution.

protected:
    
    bool self_init;    ///< Whether the p_real and p_imag matrices have been initialized from the State constructor or not.
    void calculate_expected_values(void);    ///< Calculate squared norm and expected values.
    double mean_X, mean_XX;    ///< Expected values of the X and X^2 operators.
    double mean_Y, mean_YY;    ///< Expected values of the Y and Y^2 operators.
    double mean_Px, mean_PxPx;    ///< Expected values of the P_x and P_x^2 operators.
    double mean_Py, mean_PyPy;    ///< Expected values of the P_y and P_y^2 operators.
    double norm2;    ///< Squared norm of the state.
};

/**
 * \brief This class defines a quantum state with exponential like wave function.
 *
 * This class is a child of State class.
 */
class ExponentialState: public State {
public:
    /**
        Construct the quantum state with exponential like wave function.

        @param [in] grid             Lattice object.
        @param [in] n_x              First quantum number.
        @param [in] n_y              Second quantum number.
        @param [in] norm             Squared norm of the quantum state.
        @param [in] phase            Relative phase of the wave function.
        @param [in] p_real           Pointer to the real part of the wave function.
        @param [in] p_imag           Pointer to the imaginary part of the wave function.
     */
    ExponentialState(Lattice *_grid, int _n_x=1, int _n_y=1, double _norm=1, double _phase=0, double *_p_real=0, double *_p_imag=0);

private:
    int n_x, n_y;    ///< First and second quantum number.
    double norm, phase;    ///< Norm and phase of the state.
    complex<double> exp_state(double x, double y);    ///< Exponential wave function.
};

/**
 * \brief This class defines a quantum state with gaussian like wave function.
 *
 * This class is a child of State class.
 */
class GaussianState: public State {
public:
    /**
        Construct the quantum state with gaussian like wave function.

        @param [in] grid             Lattice object.
        @param [in] omega            Gaussian coefficient.
        @param [in] mean_x           X coordinate of the gaussian function's center.
        @param [in] mean_y           Y coordinate of the gaussian function's center.
        @param [in] norm             Squared norm of the state.
        @param [in] phase            Relative phase of the wave function.
        @param [in] p_real           Pointer to the real part of the wave function.
        @param [in] p_imag           Pointer to the imaginary part of the wave function.
     */
    GaussianState(Lattice *_grid, double _omega, double _mean_x=0, double _mean_y=0, double _norm=1, double _phase=0,
                  double *_p_real=0, double *_p_imag=0);

private:
    double mean_x;    ///< X coordinate of the gaussian function's center.
    double mean_y;    ///< Y coordinate of the gaussian function's center.
    double omega;    ///< Gaussian coefficient.
    double norm;    ///< Norm of the state.
    double phase;    ///< Relative phase of the wave function.
    complex<double> gauss_state(double x, double y);    ///< Gaussian function.
};

/**
 * \brief This class defines a quantum state with sinusoidal like wave function.
 *
 * This class is a child of State class.
 */
class SinusoidState: public State {
public:
    /**
        Construct the quantum state with sinusoidal like wave function.

        @param [in] grid             Lattice object.
        @param [in] n_x              First quantum number.
        @param [in] n_y              Second quantum number.
        @param [in] norm             Squared norm of the quantum state.
        @param [in] phase            Relative phase of the wave function.
        @param [in] p_real           Pointer to the real part of the wave function.
        @param [in] p_imag           Pointer to the imaginary part of the wave function.
     */
    SinusoidState(Lattice *_grid, int _n_x=1, int _n_y=1, double _norm=1, double _phase=0, double *_p_real=0, double *_p_imag=0);

private:
    int n_x, n_y;    ///< First and second quantum number.
    double norm, phase;    ///< Norm and phase of the state.
    complex<double> sinusoid_state(double x, double y);    ///< Sinusoidal function.
};

/**
 * \brief This class defines the external potential, that is used for Hamiltonian class.
 */
class Potential {
public:
    Lattice *grid;    ///< Object that defines the lattice structure.
    double *matrix;    ///< Matrix storing the potential.

    /**
        Construct the external potential.

        @param [in] grid             Lattice object.
        @param [in] filename         Name of the file that stores the external potential matrix.
     */
    Potential(Lattice *grid, char *filename);
    /**
        Construct the external potential.

        @param [in] grid             Lattice object.
        @param [in] external_pot     Pointer to the external potential matrix.
     */
    Potential(Lattice *grid, double *external_pot=0);
    /**
        Construct the external potential.

        @param [in] grid                   Lattice object.
        @param [in] potential_function     Pointer to the static external potential function.
     */
    Potential(Lattice *grid, double (*potential_function)(double x, double y));
    /**
        Construct the external potential.

        @param [in] grid                   Lattice object.
        @param [in] potential_function     Pointer to the time-dependent external potential function.
     */
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
    virtual double get_value(int x, int y);    ///< Get the value at the coordinate (x,y).
    bool update(double t);    ///< Update the potential matrix at time t.

protected:
    double current_evolution_time;    ///< Amount of time evolved since the beginning of the evolution.
    double (*static_potential)(double x, double y);    ///< Function of the static external potential.
    double (*evolving_potential)(double x, double y, double t);    ///< Function of the time-dependent external potential.
    bool self_init;    ///< Whether the external potential matrix has been initialized from the Potential constructor or not.
    bool is_static;    ///< Whether the external potential is static or time-dependent.
};

/**
 * \brief This class defines the external potential, that is used for Hamiltonian class.
 *
 * This class is a child of Potential class.
 */
class HarmonicPotential: public Potential {
public:
    /**
        Construct the harmonic external potential.

        @param [in] grid       Lattice object.
        @param [in] omegax     Frequency along x axis.
        @param [in] omegay     Frequency along y axis.
        @param [in] mass       Mass of the particle.
        @param [in] mean_x     Minimum of the potential along x axis.
        @param [in] mean_y     Minimum of the potential along y axis.
     */
    HarmonicPotential(Lattice *_grid, double _omegax, double _omegay, double _mass=1., double _mean_x = 0., double _mean_y = 0.);
    ~HarmonicPotential();
    double get_value(int x, int y);    ///< Return the value of the external potential at coordinate (x,y)

private:
    double omegax, omegay;    ///< Frequencies along x and y axis.
    double mass;    ///< Mass of the particle.
    double mean_x, mean_y;    ///< Minimum of the potential along x and y axis.
};

/**
 * \brief This class defines the Hamiltonian of a single component system.
 */
class Hamiltonian {
public:
    Potential *potential;   ///< Potential object.
    double mass;    ///< Mass of the particle.
    double coupling_a;    ///< Coupling constant of intra-particle interaction.
    double angular_velocity;    ///< The frame of reference rotates with this angular velocity.
    double rot_coord_x;    ///< X coordinate of the center of rotation.
    double rot_coord_y;    ///< Y coordinate of the center of rotation.

    /**
        Construct the Hamiltonian of a single component system.

        @param [in] grid                Lattice object.
        @param [in] potential           Potential object.
        @param [in] mass                Mass of the particle.
        @param [in] coupling_a          Coupling constant of intra-particle interaction.
        @param [in] angular_velocity    The frame of reference rotates with this angular velocity.
        @param [in] rot_coord_x         X coordinate of the center of rotation.
        @param [in] rot_coord_y         Y coordinate of the center of rotation.
     */
    Hamiltonian(Lattice *_grid, Potential *_potential=0, double _mass=1., double _coupling_a=0.,
                double _angular_velocity=0.,
                double _rot_coord_x=0, double _rot_coord_y=0);
    ~Hamiltonian();

protected:
    bool self_init;    ///< Whether the potential is initialized in the Hamiltonian constructor or not.
    Lattice *grid;    ///< Lattice object.
};

/**
 * \brief This class defines the Hamiltonian of a two component system.
 */
class Hamiltonian2Component: public Hamiltonian {
public:
    double mass_b;    ///< Mass of the second component.
    double coupling_ab;    ///< Coupling constant of the inter-particles interaction.
    double coupling_b;    ///< Coupling constant of the intra-particles interaction of the second component.
    double omega_r;    ///< Real part of the Rabi coupling.
    double omega_i;    ///< Imaginary part of the Rabi coupling.
    Potential *potential_b;    ///< External potential for the second component.

    /**
        Construct the Hamiltonian of a two component system.

        @param [in] grid                Lattice object.
        @param [in] potential           Potential of the first component.
        @param [in] potential_b         Potential of the second component.
        @param [in] mass                Mass of the first-component's particles.
        @param [in] mass_b              Mass of the second-component's particles.
        @param [in] coupling_a          Coupling constant of intra-particle interaction for the first component.
        @param [in] coupling_ab         Coupling constant of inter-particle interaction between the two components.
        @param [in] coupling_b          Coupling constant of intra-particle interaction for the second component.
        @param [in] omega_r             Real part of the Rabi coupling.
        @param [in] omega_i             Imaginary part of the Rabi coupling.
        @param [in] angular_velocity    The frame of reference rotates with this angular velocity.
        @param [in] rot_coord_x         X coordinate of the center of rotation.
        @param [in] rot_coord_y         Y coordinate of the center of rotation.
     */
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

/**
 * \brief This class defines the evolution tasks.
 */
class Solver {
public:
    Lattice *grid;    ///< Lattice object.
    State *state;    ///< State of the first component.
    State *state_b;    ///< State of the second component.
    Hamiltonian *hamiltonian;    ///< Hamiltonian of the system; either single component or two components.
    double current_evolution_time;    ///< Amount of time evolved since the beginning of the evolution.
    /**
        Construct the Solver object for a single-component system.

        @param [in] grid                Lattice object.
        @param [in] state               State of the system.
        @param [in] hamiltonian         Hamiltonian of the system.
        @param [in] delta_t             A single evolution iteration, evolves the state for this time.
        @param [in] kernel_type         Which kernel to use (either cpu or gpu).
     */
    Solver(Lattice *grid, State *state, Hamiltonian *hamiltonian, double delta_t,
           std::string kernel_type="cpu");
    /**
        Construct the Solver object for a two-component system.

        @param [in] grid                Lattice object.
        @param [in] state1              First component's state of the system.
        @param [in] state2              Second component's state of the system.
        @param [in] hamiltonian         Hamiltonian of the two-component system.
        @param [in] delta_t             A single evolution iteration, evolves the state for this time.
        @param [in] kernel_type         Which kernel to use (either cpu or gpu).
     */
    Solver(Lattice *grid, State *state1, State *state2,
           Hamiltonian2Component *hamiltonian,
           double delta_t, std::string kernel_type="cpu");
    ~Solver();
    void evolve(int iterations, bool imag_time=false);    ///< Evolve the state of the system.
    double get_total_energy(void);    ///< Get the total energy of the system.
    double get_squared_norm(size_t which=3 /** [in] Which = 1(first component); 2 (second component); 3(total state) */);    ///< Get the squared norm of the state (default: total wave-function).
    double get_kinetic_energy(size_t which=3 /** [in] Which = 1(first component); 2 (second component); 3(total state) */);    ///< Get the kinetic energy of the system.
    double get_potential_energy(size_t which=3 /** [in] Which = 1(first component); 2 (second component); 3(total state) */);    ///< Get the potential energy of the system.
    double get_rotational_energy(size_t which=3 /** [in] Which = 1(first component); 2 (second component); 3(total state) */);    ///< Get the rotational energy of the system.
    double get_intra_species_energy(size_t which=3 /** [in] Which = 1(first component); 2 (second component); 3(total state) */);    ///< Get the intra-particles interaction energy of the system.
    double get_inter_species_energy(void);    ///< Get the inter-particles interaction energy of the system.
    double get_rabi_energy(void);    ///< Get the Rabi energy of the system.
private:
    bool imag_time;    ///< Whether the time of evolution is imaginary(true) or real(false).
    double h_a[2];    ///< Parameters of the evolution operator regarding the kinetic operator of the first-component.
    double h_b[2];    ///< Parameters of the evolution operator regarding the kinetic operator of the second-component.
    double **external_pot_real;    ///< Real part of the evolution operator regarding the external potential.
    double **external_pot_imag;    ///< Imaginary part of the evolution operator regarding the external potential.
    double delta_t;    ///< A single evolution iteration, evolves the state for this time.
    double norm2[2];    ///< Squared norms of the two wave function.
    bool single_component;    ///< Whether the system is single-component(true) or two-components(false).
    std::string kernel_type;    ///< Which kernel are being used (cpu or gpu).
    void initialize_exp_potential(double time_single_it, int which);    ///< Initialize the evolution operator regarding the external potential.
    void init_kernel();    ///< Initialize the kernel (cpu or gpu).
    double total_energy;    ///< Total energy of the system.
    double kinetic_energy[2];    ///< Kinetic energy for the single components.
    double tot_kinetic_energy;    ///< Total kinetic energy of the system.
    double potential_energy[2];    ///< Potential energy for the single components.
    double tot_potential_energy;    ///< Total potential energy of the system.
    double rotational_energy[2];    ///< Rotational energy for the single components.
    double tot_rotational_energy;    ///< Total Rotational energy of the system.
    double intra_species_energy[2];    ///< Intra-particles interaction energy for the single components.
    double tot_intra_species_energy;    ///< Total intra-particles interaction energy of the system.
    double inter_species_energy;    ///< Inter-particles interaction energy of the system.
    double rabi_energy;    ///< Rabi energy of the system.
    bool energy_expected_values_updated;    ///< Whether the expectation values are updated or not.
    void calculate_energy_expected_values(void);    ///< Calculate all the expectation values and the state's norm.
};
