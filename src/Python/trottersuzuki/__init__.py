"""Trotter-Suzuki-MPI
=====

Provides a massively parallel implementation of the Trotter-Suzuki
decomposition for simulation of quantum systems
"""

from .trottersuzuki import Lattice, HarmonicPotential, \
                           Hamiltonian, Hamiltonian2Component, Solver
from .classes_extension import State, GaussianState, SinusoidState, \
                               ExponentialState, Potential

__all__ = ['Lattice', 'State', 'ExponentialState', 'GaussianState',
           'SinusoidState', 'Potential', 'HarmonicPotential', 'Hamiltonian',
           'Hamiltonian2Component', 'Solver']
