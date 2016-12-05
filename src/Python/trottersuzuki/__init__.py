"""Trotter-Suzuki-MPI
=====

Provides a massively parallel implementation of the Trotter-Suzuki
decomposition for simulation of quantum systems
"""

from .trottersuzuki import HarmonicPotential, \
                           Hamiltonian, Hamiltonian2Component
from .classes_extension import Lattice1D, Lattice2D, State, GaussianState, \
    SinusoidState, ExponentialState, Potential, Solver
from .tools import center_coordinates, get_vortex_position

__version__ = "1.6.1"

__all__ = ['Lattice1D', 'Lattice2D', 'State', 'ExponentialState',
           'GaussianState', 'SinusoidState', 'Potential', 'HarmonicPotential',
           'Hamiltonian', 'Hamiltonian2Component', 'Solver',
           'center_coordinates', 'get_vortex_position']
