"""Trotter-Suzuki-MPI
=====

Provides a massively parallel implementation of the Trotter-Suzuki
decomposition for simulation of quantum systems
"""

from .trottersuzuki import HarmonicPotential, \
                           Hamiltonian, Hamiltonian2Component
from .classes_extension import Lattice1D, Lattice2D, State, GaussianState, \
    SinusoidState, ExponentialState, BesselState, Potential, Solver
from .tools import map_lattice_to_coordinate_space, get_vortex_position

__version__ = "1.6.2"

__all__ = ['Lattice1D', 'Lattice2D', 'State', 'ExponentialState',
           'GaussianState', 'SinusoidState', 'BesselState', 'Potential', 'HarmonicPotential',
           'Hamiltonian', 'Hamiltonian2Component', 'Solver',
           'map_lattice_to_coordinate_space', 'get_vortex_position']
