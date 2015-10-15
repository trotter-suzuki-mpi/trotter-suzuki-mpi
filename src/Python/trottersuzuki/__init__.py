"""Trotter-Suzuki-MPI
=====

Provides a massively parallel implementation of the Trotter-Suzuki
decomposition for simulation of quantum systems
"""

from .evolution import evolve, calculate_total_energy, \
                       calculate_kinetic_energy, calculate_norm2

__all__ = ['evolve', 'calculate_total_energy', 'calculate_kinetic_energy',
           'calculate_norm2']
