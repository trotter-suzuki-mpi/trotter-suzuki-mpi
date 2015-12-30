from __future__ import print_function
import numpy as np
from .trottersuzuki import solver, H, K, Lz, Norm2, phase, density, H_2GPE, \
                           Norm2_2GPE, solver_2GPE


class Lattice(object):

    def __init__(self, length_x=20, length_y=20, dim_x=100, dim_y=100,
                 periods=None):

        if dim_y == 0 or dim_x == 0:
            raise ValueError("Not a valid lattice: (", self.grid.dim_y, ",",
                             self.grid.dim_x, ")")
        self.length_y = length_y
        self.length_x = length_x
        self.dim_y = dim_y
        self.dim_x = dim_x
        self.delta_y = length_y / dim_y
        self.delta_x = length_x / dim_x
        if periods is None:
            self.periods = [0, 0]
        else:
            self.periods = periods


class State(object):

    def __init__(self, grid, p_real=None, p_imag=None):

        if not isinstance(grid, Lattice):
            raise TypeError("First parameter is not a instance of Lattice "
                            "class.")

        self.grid = grid
        if p_real is None:
            self.p_real = np.zeros((self.grid.dim_y, self.grid.dim_x))
        else:
            self.p_real = p_real
        if p_imag is None:
            self.p_imag = np.zeros((self.grid.dim_y, self.grid.dim_x))
        else:
            self.p_imag = p_imag

    def init_state(self, function):
        for i in range(0, self.grid.dim_y):
            for j in range(0, self.grid.dim_x):
                self.p_real[i, j] = np.real(function(self.grid.delta_y *
                                                     (i-self.grid.dim_y*0.5),
                                                     self.grid.delta_x *
                                                     (j-self.grid.dim_x*0.5)))
                self.p_imag[i, j] = np.imag(function(self.grid.delta_y *
                                                     (i-self.grid.dim_y*0.5),
                                                     self.grid.delta_x *
                                                     (j-self.grid.dim_x*0.5)))

    def calculate_squared_norm(self, psi_b=None):
        """Function for calculating the squared norm of the state. """
        if psi_b is None:
            return Norm2(self.p_real, self.p_imag,
                         self.grid.delta_x, self.grid.delta_y)
        else:
            if not isinstance(psi_b, State):
                raise TypeError("Parameter is not a state")
            return Norm2_2GPE(self.p_real, self.p_imag,
                              psi_b.p_real, psi_b.p_imag,
                              self.grid.delta_x, self.grid.delta_y)

    def get_particle_density(self):
        """Returns the particle density of the wave function"""
        density_matrix = np.zeros(self.p_real.shape)
        density(density_matrix, self.p_real, self.p_imag)
        return density_matrix

    def get_phase(self):
        phase_matrix = np.zeros(self.p_real.shape)
        phase(phase_matrix, self.p_real, self.p_imag)
        return phase_matrix


class Hamiltonian(object):

    def __init__(self, grid, mass=1., coupling_a=0., coupling_ab=0.,
                 angular_velocity=0., rot_coord_x=None,
                 rot_coord_y=None, omega=0., external_pot=None):
        if not isinstance(grid, Lattice):
            raise TypeError("First parameter is not a instance of Lattice "
                            "class.")
        self.grid = grid
        self.mass = mass
        self.external_pot = external_pot
        if self.external_pot is None:
            self.external_pot = np.zeros((self.grid.dim_y, self.grid.dim_x))

        self.coupling_a = coupling_a
        self.coupling_ab = coupling_ab
        self.angular_velocity = angular_velocity

        if rot_coord_y is None:
            self.rot_coord_y = grid.dim_y * 0.5
        else:
            self.rot_coord_y = rot_coord_y

        if rot_coord_x is None:
            self.rot_coord_x = grid.dim_x * 0.5
        else:
            self.rot_coord_x = rot_coord_x
        self.omega = omega

    def init_external_pot(self, function):

        for i in range(0, self.grid.dim_y):
            for j in range(0, self.grid.dim_x):
                self.external_pot[i, j] = function(self.grid.delta_y *
                                                   (i - self.grid.dim_y * 0.5),
                                                   self.grid.delta_x *
                                                   (j - self.grid.dim_x * 0.5))

    def init_harmonic_external_pot(self, omega_y, omega_x):

        def harmonic_pot(y, x):
            return 0.5 * self.mass * (omega_y * omega_y * y * y +
                                      omega_x * omega_x * x * x)

        self.init_external_pot(harmonic_pot)


class Hamiltonian2Component(Hamiltonian):

    def __init__(self, grid, mass_a=1., mass_b=1.,
                 coupling_a=0., coupling_ab=0., coupling_b=0.,
                 angular_velocity=0., rot_coord_x=None,
                 rot_coord_y=None, omega=0.,
                 external_pot_a=None, external_pot_b=None):
        super(Hamiltonian2Component, self).__init__(grid, coupling_a,
                                                    coupling_ab, mass_a,
                                                    angular_velocity,
                                                    rot_coord_y, rot_coord_x,
                                                    external_pot_a)
        self.mass_b = mass_b
        self.external_pot_b = external_pot_b
        self.coupling_b = coupling_b
        self.omega = omega

    def init_external_pot_b(self, function):

        if self.external_pot_b is None:
            self.external_pot_b = np.zeros((self.grid.dim_y, self.grid.dim_x))

        for i in range(0, self.grid.dim_y):
            for j in range(0, self.grid.dim_x):
                self.external_pot_b[i, j] = function(self.grid.delta_y *
                                                     (i - self.grid.dim_y*0.5),
                                                     self.grid.delta_x *
                                                     (j - self.grid.dim_x*0.5))

    def init_harmonic_external_pot_b(self, omega_x, omega_y):

        def harmonic_pot(y, x):
            return 0.5 * self.mass_b * (omega_y * omega_y * y * y +
                                        omega_x * omega_x * x * x)

        self.init_external_pot_b(harmonic_pot)


class Solver(object):

    def __init__(self, grid, hamiltonian, state, delta_t=1e-4,
                 kernel_type='cpu'):

        if not isinstance(grid, Lattice):
            raise TypeError("First parameter is not a instance of Lattice "
                            "class.")
        self.grid = grid

        if isinstance(hamiltonian, Hamiltonian2Component):
            self.single_component = False
        elif isinstance(hamiltonian, Hamiltonian):
            self.single_component = True
        else:
            raise TypeError("Second parameter is not a instance of Hamiltonian"
                            " or Hamiltonian2Component classes.")
        self.hamiltonian = hamiltonian

        if isinstance(state, list) and len(state) == 2 and \
                not self.single_component:
            self.state = state[0]
            self.state_b = state[1]
        elif isinstance(state, list):
            raise ValueError("Number of states does not match Hamiltonian")
        elif isinstance(state, State):
            self.state = state
            self.state_b = None
        else:
            raise TypeError("Third parameter is not a instance of or a "
                            "list of State class.")

        self.delta_t = delta_t
        self.kernel_type = kernel_type

    def evolve(self, iterations, imag_time=False):

        if self.single_component:
            solver(self.state.p_real, self.state.p_imag,
                   self.hamiltonian.mass,
                   self.hamiltonian.coupling_a,
                   self.hamiltonian.external_pot,
                   self.hamiltonian.angular_velocity,
                   self.hamiltonian.rot_coord_x,
                   self.hamiltonian.rot_coord_y,
                   self.grid.delta_x, self.grid.delta_y, self.delta_t,
                   iterations, self.kernel_type, self.grid.periods, imag_time)
        else:
            solver_2GPE(self.state.p_real, self.state.p_imag,
                        self.state_b.p_real, self.state_b.p_imag,
                        self.hamiltonian.mass,
                        self.hamiltonian.mass_b,
                        [self.hamiltonian.coupling_a,
                         self.hamiltonian.coupling_b,
                         self.hamiltonian.coupling_ab],
                        self.hamiltonian.external_pot,
                        self.hamiltonian.external_pot_b,
                        self.hamiltonian.angular_velocity,
                        self.hamiltonian.rot_coord_x,
                        self.hamiltonian.rot_coord_y,
                        self.grid.delta_x, self.grid.delta_y,
                        self.delta_t, iterations, self.kernel_type,
                        self.grid.periods, imag_time)

    def calculate_total_energy(self):
        """Function for calculating the expectation value of the Hamiltonian.
        """
        if self.single_component:
            return H(self.state.p_real, self.state.p_imag,
                     self.hamiltonian.mass,
                     self.hamiltonian.coupling_a,
                     self.hamiltonian.external_pot,
                     self.hamiltonian.angular_velocity,
                     self.hamiltonian.rot_coord_x,
                     self.hamiltonian.rot_coord_y,
                     self.grid.delta_x, self.grid.delta_y)
        else:
            return H_2GPE(self.state.p_real, self.state.p_imag,
                          self.state_b.p_real, self.state_b.p_imag,
                          self.hamiltonian.mass,
                          self.hamiltonian.mass_b,
                          [self.hamiltonian.coupling_a,
                           self.hamiltonian.coupling_b,
                           self.hamiltonian.coupling_ab],
                          self.hamiltonian.external_pot,
                          self.hamiltonian.external_pot_b,
                          self.hamiltonian.angular_velocity,
                          self.hamiltonian.rot_coord_x,
                          self.hamiltonian.rot_coord_y,
                          self.grid.delta_x, self.grid.delta_y)

    def calculate_kinetic_energy(self):
        """Function for calculating the expectation value of the kinetic
        energy.
        """
        return K(self.state.p_real, self.state.p_imag,
                 self.hamiltonian.mass,
                 self.grid.delta_x, self.grid.delta_y)

    def calculate_rotational_energy(self):
        """Function for calculating rotational energy of a system in a rotating
        frame of reference. The axis of rotation is parallel to z.
        """
        return Lz(self.state.p_real, self.state.p_imag,
                  self.hamiltonian.angular_velocity,
                  self.hamiltonian.rot_coord_x, self.hamiltonian.rot_coord_y,
                  self.grid.delta_x, self.grid.delta_y)
