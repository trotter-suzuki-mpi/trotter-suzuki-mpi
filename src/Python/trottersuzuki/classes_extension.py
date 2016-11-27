import numpy as np
from .trottersuzuki import Lattice1D as _Lattice1D
from .trottersuzuki import Lattice2D as _Lattice2D
from .trottersuzuki import State as _State
from .trottersuzuki import GaussianState as _GaussianState
from .trottersuzuki import SinusoidState as _SinusoidState
from .trottersuzuki import ExponentialState as _ExponentialState
from .trottersuzuki import Potential as _Potential
from .trottersuzuki import Solver as _Solver
from .tools import center_coordinates, imprint


class Lattice1D(_Lattice1D):

    def get_x_axis(self):
        """
        Get the x-axis of the lattice.

        Returns
        -------
        * `x_axis` : numpy array
            X-axis of the lattice
        """
        x_axis = np.arange(self.global_no_halo_dim_x) - self.global_no_halo_dim_x * 0.5 + 0.5
        x_axis *= self.delta_x

        return x_axis


class Lattice2D(_Lattice2D):

    def __init__(self, dim_x, length_x, dim_y=None, length_y=None,
                 periodic_x_axis=False, periodic_y_axis=False,
                 angular_velocity=0.):
        if dim_y is None:
            dim_y = dim_x
        if length_y is None:
            length_y = length_x
        super(Lattice2D, self).__init__(dim_x, length_x, dim_y, length_y,
                                        periodic_x_axis, periodic_y_axis,
                                        angular_velocity)

    def get_x_axis(self):
        """
        Get the x-axis of the lattice.

        Returns
        -------
        * `x_axis` : numpy array
            X-axis of the lattice
        """
        x_axis = np.arange(self.global_no_halo_dim_x) - self.global_no_halo_dim_x * 0.5 + 0.5
        x_axis *= self.delta_x

        return x_axis

    def get_y_axis(self):
        """
        Get the y-axis of the lattice

        Returns
        -------
        * `y_axis` : numpy array
            Y-axis of the lattice
        """
        y_axis = np.arange(self.global_no_halo_dim_y) - self.global_no_halo_dim_y * 0.5 + 0.5
        y_axis *= self.delta_y

        return y_axis


class State(_State):

    def init_state(self, state_function):
        """
        Initialize the wave function of the state using a function.

        Parameters
        ----------
        * `state_function` : python function
            Python function defining the wave function of the state :math:`\psi`.

        Notes
        -----
        The input arguments of the python function must be (x,y).

        Example
        -------

            >>> import trottersuzuki as ts  # import the module
            >>> grid = ts.Lattice2D()  # Define the simulation's geometry
            >>> def wave_function(x,y):  # Define a flat wave function
            >>>     return 1.
            >>> state = ts.State(grid)  # Create the system's state
            >>> state.ini_state(wave_function)  # Initialize the wave function of the state
        """
        try:
            state_function(0)

            def function(x, y):
                return state_function(x)
        except TypeError:
            function = state_function

        state = np.zeros((self.grid.dim_y, self.grid.dim_x),
                         dtype=np.complex128)

        for y in range(self.grid.dim_y):
            for x in range(self.grid.dim_x):
                state[y, x] = function(*center_coordinates(self.grid, x, y))

        self.init_state_matrix(state.real, state.imag)

    def imprint(self, function):
        """
        Multiply the wave function of the state by the function provided.

        Parameters
        ----------
        * `function` : python function
            Function to be printed on the state.

        Notes
        -----
        Useful, for instance, to imprint solitons and vortices on a condensate.
        Generally, it performs a transformation of the state whose wave function becomes

        .. math:: \psi(x,y)' = f(x,y) \psi(x,y)

        being :math:`f(x,y)` the input function and :math:`\psi(x,y)` the initial wave function.

        Example
        -------

            >>> import trottersuzuki as ts  # import the module
            >>> grid = ts.Lattice2D()  # Define the simulation's geometry
            >>> def vortex(x,y):  # Vortex function
            >>>     z = x + 1j*y
            >>>     angle = np.angle(z)
            >>>     return np.exp(1j * angle)
            >>> state = ts.GaussianState(grid, 1.)  # Create the system's state
            >>> state.imprint(vortex)  # Imprint a vortex on the state
        """
        imprint(self, function)


class GaussianState(_GaussianState):

    def imprint(self, function):
        """
        Multiply the wave function of the state by the function provided.

        Parameters
        ----------
        * `function` : python function
            Function to be printed on the state.

        Notes
        -----
        Useful, for instance, to imprint solitons and vortices on a condensate.
        Generally, it performs a transformation of the state whose wave function becomes

        .. math:: \psi(x,y)' = f(x,y) \psi(x,y)

        being :math:`f(x,y)` the input function and :math:`\psi(x,y)` the initial wave function.

        Example
        -------

            >>> import trottersuzuki as ts  # import the module
            >>> grid = ts.Lattice2D()  # Define the simulation's geometry
            >>> def vortex(x,y):  # Vortex function
            >>>     z = x + 1j*y
            >>>     angle = np.angle(z)
            >>>     return np.exp(1j * angle)
            >>> state = ts.GaussianState(grid, 1.)  # Create the system's state
            >>> state.imprint(vortex)  # Imprint a vortex on the state
        """
        imprint(self, function)


class SinusoidState(_SinusoidState):

    def imprint(self, function):
        """
        Multiply the wave function of the state by the function provided.

        Parameters
        ----------
        * `function` : python function
            Function to be printed on the state.

        Notes
        -----
        Useful, for instance, to imprint solitons and vortices on a condensate.
        Generally, it performs a transformation of the state whose wave function becomes

        .. math:: \psi(x,y)' = f(x,y) \psi(x,y)

        being :math:`f(x,y)` the input function and :math:`\psi(x,y)` the initial wave function.

        Example
        -------

            >>> import trottersuzuki as ts  # import the module
            >>> grid = ts.Lattice2D()  # Define the simulation's geometry
            >>> def vortex(x,y):  # Vortex function
            >>>     z = x + 1j*y
            >>>     angle = np.angle(z)
            >>>     return np.exp(1j * angle)
            >>> state = ts.GaussianState(grid, 1.)  # Create the system's state
            >>> state.imprint(vortex)  # Imprint a vortex on the state
        """
        imprint(self, function)


class ExponentialState(_ExponentialState):

    def imprint(self, function):
        """
        Multiply the wave function of the state by the function provided.

        Parameters
        ----------
        * `function` : python function
            Function to be printed on the state.

        Notes
        -----
        Useful, for instance, to imprint solitons and vortices on a condensate.
        Generally, it performs a transformation of the state whose wave function becomes

        .. math:: \psi(x,y)' = f(x,y) \psi(x,y)

        being :math:`f(x,y)` the input function and :math:`\psi(x,y)` the initial wave function.

        Example
        -------

            >>> import trottersuzuki as ts  # import the module
            >>> grid = ts.Lattice2D()  # Define the simulation's geometry
            >>> def vortex(x,y):  # Vortex function
            >>>     z = x + 1j*y
            >>>     angle = np.angle(z)
            >>>     return np.exp(1j * angle)
            >>> state = ts.GaussianState(grid, 1.)  # Create the system's state
            >>> state.imprint(vortex)  # Imprint a vortex on the state
        """
        imprint(self, function)


class Potential(_Potential):

    def init_potential(self, pot_function):
        """
        Initialize the external potential.

        Parameters
        ----------
        * `potential_function` : python function
            Define the external potential function.

        Example
        -------

            >>> import trottersuzuki as ts  # import the module
            >>> grid = ts.Lattice2D()  # Define the simulation's geometry
            >>> # Define a constant external potential
            >>> def external_potential_function(x,y):
            >>>     return 1.
            >>> potential = ts.Potential(grid)  # Create the external potential
            >>> potential.init_potential(external_potential_function)  # Initialize the external potential

        """
        try:
            pot_function(0)

            def _pot_function(x, y):
                return pot_function(x)
        except TypeError:
            try:
                pot_function(0, 0, 0)

                def _pot_function(x, y):
                    return pot_function(x, y, 0)
                self.updated_potential_matrix = True
                self.pot_function = pot_function
                self.exp_potential_matrix = np.zeros((self.grid.dim_y,
                                                      self.grid.dim_x),
                                                     dtype=np.complex128)
            except TypeError:
                _pot_function = pot_function

        self.potential_matrix = np.zeros((self.grid.dim_y, self.grid.dim_x))

        for y in range(self.grid.dim_y):
            for x in range(self.grid.dim_x):
                self.potential_matrix[y, x] = \
                    _pot_function(*center_coordinates(self.grid, x, y))

        self.init_potential_matrix(self.potential_matrix)

    def exponential_update(self, delta_t, t):
        for y in range(self.grid.dim_y):
            for x in range(self.grid.dim_x):
                self.exp_potential_matrix[y, x] = \
                    np.exp(-1j*delta_t*self.pot_function(*center_coordinates(self.grid, x, y), t))
        return self.exp_potential_matrix

class Solver(_Solver):

    def __init__(self, Lattice, State, Hamiltonian, delta_t, Potential=None,
                 State2=None, Potential2=None, kernel_type="cpu",):
        if State2 is None:
            super(Solver, self).__init__(Lattice, State, Hamiltonian,
                                         delta_t, kernel_type)
        else:
            super(Solver, self).__init__(Lattice, State, State2,
                                         Hamiltonian, delta_t, kernel_type)
        self.delta_t = delta_t
        self.potential = Potential
        if State2 is not None and Potential2 is None:
            self.potential2 = Potential
        elif State2 is not None:
            self.potential2 = Potential2
        else:
            self.potential2 = None

    def evolve(self, iterations, imag_time=False):
        if not self.hamiltonian.potential.updated_potential_matrix or \
                imag_time:
            super(Solver, self).evolve(iterations, imag_time)
            return
        for _ in range(iterations-1):
            exp_pot = self.potential.exponential_update(self.delta_t,
                                                        self.current_evolution_time)
            super(Solver, self).set_exp_potential(np.ravel(exp_pot.real),
                                                  np.ravel(exp_pot.imag), 0)
            super(Solver, self).evolve(-1, imag_time)
        exp_pot = self.potential.exponential_update(self.delta_t,
                                                    self.current_evolution_time)
        super(Solver, self).set_exp_potential(np.ravel(exp_pot.real),
                                              np.ravel(exp_pot.imag), 0)
        super(Solver, self).evolve(1, imag_time)
