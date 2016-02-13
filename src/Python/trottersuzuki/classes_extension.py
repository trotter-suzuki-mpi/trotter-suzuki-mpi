import numpy as np
from .trottersuzuki import Lattice as _Lattice
from .trottersuzuki import State as _State
from .trottersuzuki import GaussianState as _GaussianState
from .trottersuzuki import SinusoidState as _SinusoidState
from .trottersuzuki import ExponentialState as _ExponentialState
from .trottersuzuki import Potential as _Potential

class Lattice(_Lattice):
    
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
            >>> grid = ts.Lattice()  # Define the simulation's geometry
            >>> def wave_function(x,y):  # Define a flat wave function
            >>>     return 1.
            >>> state = ts.State(grid)  # Create the system's state
            >>> state.ini_state(wave_function)  # Initialize the wave function of the state
        """
        real = np.zeros((self.grid.dim_y, self.grid.dim_x))
        imag = np.zeros((self.grid.dim_y, self.grid.dim_x))
        
        delta_x = self.grid.delta_x 
        delta_y = self.grid.delta_y
        idy = self.grid.start_y * delta_y + 0.5 * delta_y
        x_c = self.grid.global_no_halo_dim_x * self.grid.delta_x * 0.5
        y_c = self.grid.global_no_halo_dim_y * self.grid.delta_y * 0.5
        
        for y in range(0, self.grid.dim_y):
            y_r = idy - y_c
            idx = self.grid.start_x * delta_x + 0.5 * delta_x
            for x in range(0, self.grid.dim_x):
                x_r = idx - x_c
                tmp = state_function(x_r, y_r)
                real[y, x] = np.real(tmp)
                imag[y, x] = np.imag(tmp)
                idx += delta_x
            idy += delta_y
            
        self.init_state_matrix(real, imag)
        
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
            >>> grid = ts.Lattice()  # Define the simulation's geometry
            >>> def vortex(x,y):  # Vortex function
            >>>     z = x + 1j*y
            >>>     angle = np.angle(z)
            >>>     return np.exp(1j * angle)
            >>> state = ts.GaussianState(grid, 1.)  # Create the system's state
            >>> state.imprint(vortex)  # Imprint a vortex on the state
        """
        real = np.zeros((self.grid.dim_y, self.grid.dim_x))
        imag = np.zeros((self.grid.dim_y, self.grid.dim_x))
        
        delta_x = self.grid.delta_x 
        delta_y = self.grid.delta_y
        
        x_c = self.grid.global_no_halo_dim_x * self.grid.delta_x * 0.5
        y_c = self.grid.global_no_halo_dim_y * self.grid.delta_y * 0.5
        idy = self.grid.start_y * delta_y + 0.5 * delta_y
        for y in range(0, self.grid.dim_y):
            y_r = idy - y_c
            idx = self.grid.start_x * delta_x + 0.5 * delta_x
            for x in range(0, self.grid.dim_x):
                x_r = idx - x_c
                tmp = function(x_r, y_r)
                real[y, x] = np.real(tmp)
                imag[y, x] = np.imag(tmp)
                idx += delta_x
            idy += delta_y
            
        self.imprint_matrix(real, imag)
        
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
            >>> grid = ts.Lattice()  # Define the simulation's geometry
            >>> def vortex(x,y):  # Vortex function
            >>>     z = x + 1j*y
            >>>     angle = np.angle(z)
            >>>     return np.exp(1j * angle)
            >>> state = ts.GaussianState(grid, 1.)  # Create the system's state
            >>> state.imprint(vortex)  # Imprint a vortex on the state
        """
        real = np.zeros((self.grid.dim_y, self.grid.dim_x))
        imag = np.zeros((self.grid.dim_y, self.grid.dim_x))
        
        delta_x = self.grid.delta_x 
        delta_y = self.grid.delta_y
        idy = self.grid.start_y * delta_y + 0.5 * delta_y
        x_c = self.grid.global_no_halo_dim_x * self.grid.delta_x * 0.5
        y_c = self.grid.global_no_halo_dim_y * self.grid.delta_y * 0.5
        
        for y in range(0, self.grid.dim_y):
            y_r = idy - y_c
            idx = self.grid.start_x * delta_x + 0.5 * delta_x
            for x in range(0, self.grid.dim_x):
                x_r = idx - x_c
                tmp = function(x_r, y_r)
                real[y, x] = np.real(tmp)
                imag[y, x] = np.imag(tmp)
                idx += delta_x
            idy += delta_y
        
        self.imprint_matrix(real, imag)
        
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
            >>> grid = ts.Lattice()  # Define the simulation's geometry
            >>> def vortex(x,y):  # Vortex function
            >>>     z = x + 1j*y
            >>>     angle = np.angle(z)
            >>>     return np.exp(1j * angle)
            >>> state = ts.GaussianState(grid, 1.)  # Create the system's state
            >>> state.imprint(vortex)  # Imprint a vortex on the state
        """
        real = np.zeros((self.grid.dim_y, self.grid.dim_x))
        imag = np.zeros((self.grid.dim_y, self.grid.dim_x))
        
        delta_x = self.grid.delta_x 
        delta_y = self.grid.delta_y
        idy = self.grid.start_y * delta_y + 0.5 * delta_y
        x_c = self.grid.global_no_halo_dim_x * self.grid.delta_x * 0.5
        y_c = self.grid.global_no_halo_dim_y * self.grid.delta_y * 0.5
        
        for y in range(0, self.grid.dim_y):
            y_r = idy - y_c
            idx = self.grid.start_x * delta_x + 0.5 * delta_x
            for x in range(0, self.grid.dim_x):
                x_r = idx - x_c
                tmp = function(x_r, y_r)
                real[y, x] = np.real(tmp)
                imag[y, x] = np.imag(tmp)
                idx += delta_x
            idy += delta_y
            
        self.imprint_matrix(real, imag)

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
            >>> grid = ts.Lattice()  # Define the simulation's geometry
            >>> def vortex(x,y):  # Vortex function
            >>>     z = x + 1j*y
            >>>     angle = np.angle(z)
            >>>     return np.exp(1j * angle)
            >>> state = ts.GaussianState(grid, 1.)  # Create the system's state
            >>> state.imprint(vortex)  # Imprint a vortex on the state
        """
        real = np.zeros((self.grid.dim_y, self.grid.dim_x))
        imag = np.zeros((self.grid.dim_y, self.grid.dim_x))
        
        delta_x = self.grid.delta_x 
        delta_y = self.grid.delta_y
        idy = self.grid.start_y * delta_y + 0.5 * delta_y
        x_c = self.grid.global_no_halo_dim_x * self.grid.delta_x * 0.5
        y_c = self.grid.global_no_halo_dim_y * self.grid.delta_y * 0.5
        
        for y in range(0, self.grid.dim_y):
            y_r = idy - y_c
            idx = self.grid.start_x * delta_x + 0.5 * delta_x
            for x in range(0, self.grid.dim_x):
                x_r = idx - x_c
                tmp = function(x_r, y_r)
                real[y, x] = np.real(tmp)
                imag[y, x] = np.imag(tmp)
                idx += delta_x
            idy += delta_y
        
        self.imprint_matrix(real, imag)
        
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
            >>> grid = ts.Lattice()  # Define the simulation's geometry
            >>> # Define a constant external potential
            >>> def external_potential_function(x,y):
            >>>     return 1.
            >>> potential = ts.Potential(grid)  # Create the external potential
            >>> potential.init_potential(external_potential_function)  # Initialize the external potential

        """
        potential = np.zeros((self.grid.dim_y, self.grid.dim_x))
        
        delta_x = self.grid.delta_x 
        delta_y = self.grid.delta_y
        idy = self.grid.start_y * delta_y + 0.5 * delta_y
        x_c = self.grid.global_no_halo_dim_x * self.grid.delta_x * 0.5
        y_c = self.grid.global_no_halo_dim_y * self.grid.delta_y * 0.5
        
        for y in range(0, self.grid.dim_y):
            y_r = idy - y_c    
            idx = self.grid.start_x * delta_x + 0.5 * delta_x            
            for x in range(0, self.grid.dim_x):
                x_r = idx - x_c
                potential[y, x] = pot_function(x_r, y_r)
                idx += delta_x
            idy += delta_y
        
        self.init_potential_matrix(potential)
