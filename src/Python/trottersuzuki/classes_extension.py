import numpy as np
from .trottersuzuki import State as _State
from .trottersuzuki import GaussianState as _GaussianState
from .trottersuzuki import SinusoidState as _SinusoidState
from .trottersuzuki import ExponentialState as _ExponentialState
from .trottersuzuki import Potential as _Potential

class State(_State):
    
    def init_state(self, state_function):
        
        real = np.zeros((self.grid.dim_x, self.grid.dim_y))
        imag = np.zeros((self.grid.dim_x ,self.grid.dim_y))
        
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
                real[x, y] = np.real(tmp)
                imag[x, y] = np.imag(tmp)
                idx += delta_x
            idy += delta_y
            
        self.init_state_matrix(real, imag)
        
    def imprint(self, function):
        
        real = np.zeros((self.grid.dim_x, self.grid.dim_y))
        imag = np.zeros((self.grid.dim_x ,self.grid.dim_y))
        
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
                real[x, y] = np.real(tmp)
                imag[x, y] = np.imag(tmp)
                idx += delta_x
            idy += delta_y
            
        self.imprint_matrix(real, imag)
        
class GaussianState(_GaussianState):
    
    def imprint(self, function):
        
        real = np.zeros((self.grid.dim_x, self.grid.dim_y))
        imag = np.zeros((self.grid.dim_x ,self.grid.dim_y))
        
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
                real[x, y] = np.real(tmp)
                imag[x, y] = np.imag(tmp)
                idx += delta_x
            idy += delta_y
            
        self.imprint_matrix(real, imag)
        
class SinusoidState(_SinusoidState):
    
    def imprint(self, function):
        
        real = np.zeros((self.grid.dim_x, self.grid.dim_y))
        imag = np.zeros((self.grid.dim_x ,self.grid.dim_y))
        
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
                real[x, y] = np.real(tmp)
                imag[x, y] = np.imag(tmp)
                idx += delta_x
            idy += delta_y
            
        self.imprint_matrix(real, imag)

class ExponentialState(_ExponentialState):
    
    def imprint(self, function):
        
        real = np.zeros((self.grid.dim_x, self.grid.dim_y))
        imag = np.zeros((self.grid.dim_x ,self.grid.dim_y))
        
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
                real[x, y] = np.real(tmp)
                imag[x, y] = np.imag(tmp)
                idx += delta_x
            idy += delta_y
            
        self.imprint_matrix(real, imag)
        
class Potential(_Potential):
    
    def init_potential(self, pot_function):
        
        potential = np.zeros((self.grid.dim_x, self.grid.dim_y))
        
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
                potential[x, y] = pot_function(x_r, y_r)
                idx += delta_x
            idy += delta_y
            
        self.init_potential_matrix(potential)
