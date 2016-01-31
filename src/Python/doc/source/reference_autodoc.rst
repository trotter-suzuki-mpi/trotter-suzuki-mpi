******************
Function Reference
******************

Lattice Class
=============
.. autoclass:: trottersuzuki.Lattice


State Classes
=============
.. autoclass:: trottersuzuki.State
   :members: get_particle_density, get_phase, get_squared_norm, get_mean_x, get_mean_xx, get_mean_y, get_mean_yy, get_mean_px, get_mean_pxpx, get_mean_py, get_mean_pypy, init_state, imprint, write_to_file, write_particle_density, write_phase

.. autoclass:: trottersuzuki.ExponentialState
  :members: get_particle_density, get_phase, get_squared_norm, get_mean_x, get_mean_xx, get_mean_y, get_mean_yy, get_mean_px, get_mean_pxpx, get_mean_py, get_mean_pypy, imprint, write_to_file, write_particle_density, write_phase

.. autoclass:: trottersuzuki.GaussianState
  :members: get_particle_density, get_phase, get_squared_norm, get_mean_x, get_mean_xx, get_mean_y, get_mean_yy, get_mean_px, get_mean_pxpx, get_mean_py, get_mean_pypy, imprint, write_to_file, write_particle_density, write_phase

.. autoclass:: trottersuzuki.SinusoidState
  :members: get_particle_density, get_phase, get_squared_norm, get_mean_x, get_mean_xx, get_mean_y, get_mean_yy, get_mean_px, get_mean_pxpx, get_mean_py, get_mean_pypy, imprint, write_to_file, write_particle_density, write_phase

Potential Classes
=================
.. autoclass:: trottersuzuki.Potential
   :members: init_potential, get_value

.. autoclass:: trottersuzuki.HarmonicPotential
   :members: get_value

Hamiltonian Classes
===================
.. autoclass:: trottersuzuki.Hamiltonian

.. autoclass:: trottersuzuki.Hamiltonian2Component

Solver Class
===================
.. autoclass:: trottersuzuki.Solver
   :members: evolve, get_total_energy, get_squared_norm, get_kinetic_energy, get_potential_energy, get_rotational_energy, get_intra_species_energy, get_inter_species_energy, get_rabi_energy
