import numpy as np
from .trottersuzuki import solver, H, K, Lz, Norm2, phase, density, H_2GPE, Norm2_2GPE, solver_2GPE

class Lattice(object):
    
    def __init__(self, length_y = 20, length_x = 20, dim_y = 100, dim_x = 100):
        
        self.length_y = length_y
        self.length_x = length_x
        self.dim_y = dim_y
        self.dim_x = dim_x
        self.delta_y = length_y / dim_y
        self.delta_x = length_x / dim_x
        
class State(Lattice):
        
    def __init__(self, grid = Lattice(), periods = [0, 0], p_real = None, p_imag = None, pb_real = None, pb_imag = None):
        
        if not(isinstance(grid, Lattice)):
            print "First parameter is not a instance of Lattice class."
            return
        
        Lattice.__init__(self, grid.length_y, grid.length_x, grid.dim_y, grid.dim_x)
        self.periods = periods
        self.p_real = p_real
        self.p_imag = p_imag
        self.pb_real = pb_real
        self.pb_imag = pb_imag
        
    def init_state(self, function, functionb = None):
        
        if self.dim_y == 0 or self.dim_x == 0:
            print "No Valid Lattice: (", self.dim_y, ",", self.dim_x, ")"
            return
            
        if self.p_real == None:
            self.p_real = np.zeros((self.dim_y, self.dim_x))
        
        if self.p_imag == None:
            self.p_imag = np.zeros((self.dim_y, self.dim_x))
        
        for i in range(0, self.dim_y):
            for j in range(0, self.dim_x):
                self.p_real[i, j] = np.real(function(self.delta_y * (i - self.dim_y * 0.5), self.delta_x * (j - self.dim_x * 0.5)))
                self.p_imag[i, j] = np.imag(function(self.delta_y * (i - self.dim_y * 0.5), self.delta_x * (j - self.dim_x * 0.5)))
        
        if functionb != None:
            
            if self.pb_real == None:
                self.pb_real = np.zeros((self.dim_y, self.dim_x))
            
            if self.pb_imag == None:
                self.pb_imag = np.zeros((self.dim_y, self.dim_x))
            
            for i in range(0, self.dim_y):
                for j in range(0, self.dim_x):
                    self.pb_real[i, j] = np.real(functionb(self.delta_y * (i - self.dim_y * 0.5), self.delta_x * (j - self.dim_x * 0.5)))
                    self.pb_imag[i, j] = np.imag(functionb(self.delta_y * (i - self.dim_y * 0.5), self.delta_x * (j - self.dim_x * 0.5)))
    
    def calculate_squared_norm(self):
        """Function for calculating the squared norm of the state. """
        return Norm2(self.p_real, self.p_imag, self.delta_x, self.delta_y)
    
    def calculate_squared_norm_b(self):
        """Function for calculating the squared norm of the state. """
        if self.pb_real != None and self.pb_imag != None:
            return Norm2(self.pb_real, self.pb_imag, self.delta_x, self.delta_y)
        else:
            print "Second wave function is not defined."
        
    def calculate_squared_norm_2(self):
        if self.pb_real != None and self.pb_imag != None:
            return Norm2_2GPE(self.p_real, self.p_imag, self.pb_real, self.pb_imag, self.delta_x, self.delta_y)
        else:
            print "Second wave function is not defined."
            
    def get_particles_density(self):
        """Function that return the particle denity of the wave function"""
        density_matrix = np.zeros(self.p_real.shape)
        density(density_matrix, self.p_real, self.p_imag)
        return density_matrix
    
    def get_particles_density_b(self):
        """Function that return the particle denity of the wave function"""
        if self.pb_real != None and self.pb_imag != None:
            density_matrix = np.zeros(self.p_real.shape)
            density(density_matrix, self.p_real, self.p_imag)
            return density_matrix
        else:
            print "Second wave function is not defined."
    
    def get_phase(self):
        phase_matrix = np.zeros(self.p_real.shape)
        phase(phase_matrix, self.p_real, self.p_imag)
        return phase_matrix
    
    def get_phase_b(self):
        if self.pb_real != None and self.pb_imag != None:
            phase_matrix = np.zeros(self.pb_real.shape)
            phase(phase_matrix, self.pb_real, self.pb_imag)
            return phase_matrix
        else:
            print "Second wave function is not defined."
            
    def calculate_total_energy(self, hamiltonian):
        """Function for calculating the expectation value of the Hamiltonian."""
        if not(isinstance(hamiltonian, Hamiltonian)):
            print "First parameter is not a instance of Hamiltonian class."
            return
        else:
            return H(self.p_real, self.p_imag, hamiltonian.mass, hamiltonian.coupling_a, hamiltonian.external_pot, hamiltonian.angular_velocity, 
                     hamiltonian.rot_coord_x, hamiltonian.rot_coord_y, self.delta_x, self.delta_y)
    
    def calculate_kinetic_energy(self, hamiltonian):
        """Function for calculating the expectation value of the kinetic energy."""
        if not(isinstance(hamiltonian, Hamiltonian)):
            print "First parameter is not a instance of Hamiltonian class."
            return
        else:
            return K(self.p_real, self.p_imag, hamiltonian.mass, self.delta_x, self.delta_y)

    def calculate_rotational_energy(self, hamiltonian):
        """Function for calculating rotational energy of a system in a rotating frame of reference. The axis of rotation is parallel to z."""
        if not(isinstance(hamiltonian, Hamiltonian)):
            print "First parameter is not a instance of Hamiltonian class."
            return
        else:
            return Lz(self.p_real, self.p_imag, hamiltonian.angular_velocity, hamiltonian.rot_coord_x, hamiltonian.rot_coord_y, self.delta_x, self.delta_y)

    def calculate_total_energy_2GPE(self, hamiltonian):
        """Function for calculating the expectation value of the two component Hamiltonian."""
        if not(isinstance(hamiltonian, Hamiltonian2Component)):
            print "First parameter is not a instance of Hamiltonian class."
            return
        else:
            return H_2GPE(self.p_real, self.p_imag, self.pb_real, self.pb_imag, hamiltonian.hamiltonian1.mass, hamiltonian.hamiltonian2.mass, 
                          [hamiltonian.hamiltonian1.coupling_a, hamiltonian.hamiltonian2.coupling_a, hamiltonian.hamiltonian1.coupling_ab, np.real(hamiltonian.omega), np.imag(hamiltonian.omega)], 
                          hamiltonian.hamiltonian1.external_pot, hamiltonian.hamiltonian2.external_pot, hamiltonian.hamiltonian1.angular_velocity, 
                          hamiltonian.hamiltonian1.rot_coord_x, hamiltonian.hamiltonian1.rot_coord_y, self.delta_x, self.delta_y)

class Hamiltonian(Lattice):
    
    def __init__(self, grid = Lattice(), coupling_a = 0., coupling_ab = 0, mass = 1., angular_velocity = 0., rot_coord_y = None, rot_coord_x = None, external_pot = None):
        if not(isinstance(grid, Lattice)):
            print "First parameter is not a instance of Lattice class."
            return
        
        Lattice.__init__(self, grid.length_y, grid.length_x, grid.dim_y, grid.dim_x)
        self.mass = mass
        self.external_pot = external_pot
        self.coupling_a = coupling_a
        self.coupling_ab = coupling_ab
        self.angular_velocity = angular_velocity
        
        if rot_coord_y == None:
            self.rot_coord_y = grid.dim_y * 0.5
        else:
            self.rot_coord_y = rot_coord_y
        
        if rot_coord_x == None:
            self.rot_coord_x = grid.dim_x * 0.5
        else:
            self.rot_coord_x = rot_coord_x

    def init_external_pot(self, function):
        
        if self.dim_y == 0 or self.dim_x == 0:
            print "No Valid Lattice: (", self.dim_y, ",", self.dim_x, ")"
            return
            
        if self.external_pot == None:
            self.external_pot = np.zeros((self.dim_y, self.dim_x))
        
        for i in range(0, self.dim_y):
            for j in range(0, self.dim_x):
                self.external_pot[i, j] = function(self.delta_y * (i - self.dim_y * 0.5), self.delta_x * (j - self.dim_x * 0.5))
    
    def harmonic_external_pot(self, omega_y, omega_x):
        
        def harmonic_pot(y, x):
            return 0.5 * self.mass * (omega_y * omega_y * y * y + omega_x * omega_x * x * x)
        
        self.init_external_pot(harmonic_pot)
                
class Hamiltonian2Component(object):
    
    def __init__(self, hamiltonian1 = Hamiltonian(), hamiltonian2 = Hamiltonian(), omega = 0.):
        if not(isinstance(hamiltonian1, Hamiltonian)):
            print "First parameter is not a instance of Hamiltonian class."
            return
        
        if not(isinstance(hamiltonian2, Hamiltonian)):
            print "Second parameter is not a instance of Hamiltonian class."
            return
        
        self.hamiltonian1 = hamiltonian1
        self.hamiltonian2 = hamiltonian2
        self.omega = omega
        
class Solver(State):
    
    def __init__(self, grid, hamiltonian, state = State(), imag_time = False, iterations = 100, delta_t = 1e-4, kernel_type = 'cpu'):
            
        if not(isinstance(grid, Lattice)):
            print "First parameter is not a instance of Lattice class."
            return
        
        if isinstance(hamiltonian, Hamiltonian):
            
            self.mass_a = hamiltonian.mass
            self.external_pot_a = hamiltonian.external_pot
            self.coupling_a = hamiltonian.coupling_a
            self.coupling_ab = hamiltonian.coupling_ab
            self.angular_velocity = hamiltonian.angular_velocity
            self.rot_coord_y = hamiltonian.rot_coord_y
            self.rot_coord_x = hamiltonian.rot_coord_x
            
            self.single_component = True
        
        elif isinstance(hamiltonian, Hamiltonian2Component):
            
            self.mass_a = hamiltonian.hamiltonian1.mass
            self.external_pot_a = hamiltonian.hamiltonian1.external_pot
            self.coupling_a = hamiltonian.hamiltonian1.coupling_a
            self.coupling_ab = hamiltonian.hamiltonian1.coupling_ab
            self.angular_velocity = hamiltonian.hamiltonian1.angular_velocity
            self.rot_coord_y = hamiltonian.hamiltonian1.rot_coord_y
            self.rot_coord_x = hamiltonian.hamiltonian1.rot_coord_x
            
            self.mass_b = hamiltonian.hamiltonian2.mass
            self.external_pot_b = hamiltonian.hamiltonian2.external_pot
            self.coupling_b = hamiltonian.hamiltonian2.coupling_a
            self.coupling_ab = hamiltonian.hamiltonian2.coupling_ab
            self.angular_velocity = hamiltonian.hamiltonian2.angular_velocity
            self.rot_coord_y = hamiltonian.hamiltonian2.rot_coord_y
            self.rot_coord_x = hamiltonian.hamiltonian2.rot_coord_x
            
            self.single_component = False
            
        else:
            print "Second parameter is not a instance of Hamiltonian or Hamiltonian2Component classes."
            return
        
        if not(isinstance(state, State)):
            print "Third parameter is not a instance of State class."
            return
        
        
        State.__init__(self, grid, state.periods, state.p_real, state.p_imag, state.pb_real, state.pb_imag)
        Lattice.__init__(self, grid.length_y, grid.length_x, grid.dim_y, grid.dim_x)
        self.iterations = iterations
        self.delta_t = delta_t
        self.imag_time = imag_time
        self.kernel_type = kernel_type
        
    def evolve(self, imag_time = None, iterations = None, delta_t = None, kernel_type = None):
        
        if imag_time == None:
            evo_imag_time = self.imag_time
        else:
            evo_imag_time = imag_time
        
        if iterations == None:
            evo_iterations = self.iterations
        else:
            evo_iterations = iterations
        
        if delta_t == None:
            evo_delta_t = self.delta_t
        else:
            evo_delta_t = delta_t
        
        if kernel_type == None:
            evo_kernel_type = self.kernel_type
        else:
            evo_kernel_type = kernel_type
        
        if self.single_component == True:
            solver(self.p_real, self.p_imag, self.mass_a, self.coupling_a, self.external_pot_a, self.angular_velocity, self.rot_coord_x, self.rot_coord_y, 
                   self.delta_x, self.delta_y, evo_delta_t, evo_iterations, evo_kernel_type, self.periods, evo_imag_time)
        else:
            solver_2GPE(self.p_real, self.p_imag, self.pb_real, self.pb_imag, self.mass_a, self.mass_b, [self.coupling_a, self.coupling_b, self.coupling_ab], 
                        self.external_pot_a, self.external_pot_b, self.angular_velocity, self.rot_coord_x, self.rot_coord_y, 
                        self.delta_x, self.delta_y, evo_delta_t, evo_iterations, evo_kernel_type, self.periods, evo_imag_time)
        
