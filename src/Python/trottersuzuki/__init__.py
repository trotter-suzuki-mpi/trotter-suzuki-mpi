"""Trotter-Suzuki-MPI
=====

Provides a massively parallel implementation of the Trotter-Suzuki
decomposition for simulation of quantum systems
"""

from .trottersuzuki import Lattice, ExponentialState, GaussianState, \
                           SinusoidState, ParabolicPotential, \
                           Hamiltonian, Hamiltonian2Component, Solver
from .classes_extension import State, Potential

__all__ = ['Lattice', 'State', 'ExponentialState', 'GaussianState',
           'SinusoidState', 'Potential', 'ParabolicPotential', 'Hamiltonian',
           'Hamiltonian2Component', 'Solver']
