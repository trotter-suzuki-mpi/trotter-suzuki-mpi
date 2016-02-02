******************
Function Reference
******************

Lattice Class
=============
.. py:class:: Lattice(dim=100, length_x=20.0, length_y=20.0, periodic_x_axis=False, periodic_y_axis=False, angular_velocity=0.0)
   :module: trottersuzuki

   This class defines the lattice structure over which the state and potential
   matrices are defined.

   As to single-process execution, the lattice is a single tile which can be
   surrounded by a halo, in the case of periodic boundary conditions.

   Constructors
   ------------
   * `Lattice(dim=100, length_x=20.0, length_y=20.0, periodic_x_axis=False, periodic_y_axis=False, angular_velocity=0.0)`

       Construct the Lattice.

       Parameters
       ----------
       * `dim` :
           Linear dimension of the squared lattice.
       * `length_x` :
           Physical length of the lattice's side along the x axis.
       * `length_y` :
           Physical length of the lattice's side along the y axis.
       * `periodic_x_axis` :
           Boundary condition along the x axis (false=closed, true=periodic).
       * `periodic_y_axis` :
           Boundary condition along the y axis (false=closed, true=periodic).
       * `angular_velocity` :
           Angular velocity of the frame of reference.

   C++ includes: trottersuzuki.h


State Classes
=============
.. py:class:: State(*args)
   :module: trottersuzuki

   This class defines the quantum state.

   C++ includes: trottersuzuki.h

   .. py:method:: State.get_mean_px()
      :module: trottersuzuki

      Return the expected value of the P_x operator.



   .. py:method:: State.get_mean_pxpx()
      :module: trottersuzuki

      Return the expected value of the P_x^2 operator.



   .. py:method:: State.get_mean_py()
      :module: trottersuzuki

      Return the expected value of the P_y operator.



   .. py:method:: State.get_mean_pypy()
      :module: trottersuzuki

      Return the expected value of the P_y^2 operator.



   .. py:method:: State.get_mean_x()
      :module: trottersuzuki

      Return the expected value of the X operator.



   .. py:method:: State.get_mean_xx()
      :module: trottersuzuki

      Return the expected value of the X^2 operator.



   .. py:method:: State.get_mean_y()
      :module: trottersuzuki

      Return the expected value of the Y operator.



   .. py:method:: State.get_mean_yy()
      :module: trottersuzuki

      Return the expected value of the Y^2 operator.



   .. py:method:: State.get_particle_density()
      :module: trottersuzuki

      Return a matrix storing the squared norm of the wave function.



   .. py:method:: State.get_phase()
      :module: trottersuzuki

      Return a matrix storing the phase of the wave function.



   .. py:method:: State.get_squared_norm()
      :module: trottersuzuki

      Return the squared norm of the quantum state.



   .. py:method:: State.write_particle_density(fileprefix)
      :module: trottersuzuki

      Write to a file the squared norm of the wave function.



   .. py:method:: State.write_phase(fileprefix)
      :module: trottersuzuki

      Write to a file the phase of the wave function.



   .. py:method:: State.write_to_file(fileprefix)
      :module: trottersuzuki

      Write to a file the wave function.



.. py:class:: ExponentialState(_grid, _n_x=1, _n_y=1, _norm=1, _phase=0, _p_real=None, _p_imag=None)
   :module: trottersuzuki

   This class defines a quantum state with exponential like wave function.

   This class is a child of State class.

   Constructors
   ------------
   * `ExponentialState(_grid, _n_x=1, _n_y=1, _norm=1, _phase=0, _p_real=None, _p_imag=None)`

       Construct the Lattice.

       Construct the quantum state with exponential like wave function.

       Parameters
       ----------
       * `grid` :
           Lattice object.
       * `n_x` :
           First quantum number.
       * `n_y` :
           Second quantum number.
       * `norm` :
           Squared norm of the quantum state.
       * `phase` :
           Relative phase of the wave function.
       * `p_real` :
           Pointer to the real part of the wave function.
       * `p_imag` :
           Pointer to the imaginary part of the wave function.

   C++ includes: trottersuzuki.h


   .. py:method:: ExponentialState.get_mean_px()
      :module: trottersuzuki

      Return the expected value of the P_x operator.



   .. py:method:: ExponentialState.get_mean_pxpx()
      :module: trottersuzuki

      Return the expected value of the P_x^2 operator.



   .. py:method:: ExponentialState.get_mean_py()
      :module: trottersuzuki

      Return the expected value of the P_y operator.



   .. py:method:: ExponentialState.get_mean_pypy()
      :module: trottersuzuki

      Return the expected value of the P_y^2 operator.



   .. py:method:: ExponentialState.get_mean_x()
      :module: trottersuzuki

      Return the expected value of the X operator.



   .. py:method:: ExponentialState.get_mean_xx()
      :module: trottersuzuki

      Return the expected value of the X^2 operator.



   .. py:method:: ExponentialState.get_mean_y()
      :module: trottersuzuki

      Return the expected value of the Y operator.



   .. py:method:: ExponentialState.get_mean_yy()
      :module: trottersuzuki

      Return the expected value of the Y^2 operator.



   .. py:method:: ExponentialState.get_particle_density()
      :module: trottersuzuki

      Return a matrix storing the squared norm of the wave function.



   .. py:method:: ExponentialState.get_phase()
      :module: trottersuzuki

      Return a matrix storing the phase of the wave function.



   .. py:method:: ExponentialState.get_squared_norm()
      :module: trottersuzuki

      Return the squared norm of the quantum state.



   .. py:method:: ExponentialState.write_particle_density(fileprefix)
      :module: trottersuzuki

      Write to a file the squared norm of the wave function.



   .. py:method:: ExponentialState.write_phase(fileprefix)
      :module: trottersuzuki

      Write to a file the phase of the wave function.



   .. py:method:: ExponentialState.write_to_file(fileprefix)
      :module: trottersuzuki

      Write to a file the wave function.



.. py:class:: GaussianState(_grid, _omega, _mean_x=0, _mean_y=0, _norm=1, _phase=0, _p_real=None, _p_imag=None)
   :module: trottersuzuki

   This class defines a quantum state with gaussian like wave function.

   This class is a child of State class.

   Constructors
   ------------
   * `GaussianState(_grid, _omega, _mean_x=0, _mean_y=0, _norm=1, _phase=0, _p_real=None, _p_imag=None)`

       Construct the quantum state with gaussian like wave function.

       Parameters
       ----------
       * `grid` :
           Lattice object.
       * `omega` :
           Gaussian coefficient.
       * `mean_x` :
           X coordinate of the gaussian function's center.
       * `mean_y` :
           Y coordinate of the gaussian function's center.
       * `norm` :
           Squared norm of the state.
       * `phase` :
           Relative phase of the wave function.
       * `p_real` :
           Pointer to the real part of the wave function.
       * `p_imag` :
           Pointer to the imaginary part of the wave function.

   C++ includes: trottersuzuki.h

   .. py:method:: GaussianState.get_mean_px()
      :module: trottersuzuki

      Return the expected value of the P_x operator.



   .. py:method:: GaussianState.get_mean_pxpx()
      :module: trottersuzuki

      Return the expected value of the P_x^2 operator.



   .. py:method:: GaussianState.get_mean_py()
      :module: trottersuzuki

      Return the expected value of the P_y operator.



   .. py:method:: GaussianState.get_mean_pypy()
      :module: trottersuzuki

      Return the expected value of the P_y^2 operator.



   .. py:method:: GaussianState.get_mean_x()
      :module: trottersuzuki

      Return the expected value of the X operator.



   .. py:method:: GaussianState.get_mean_xx()
      :module: trottersuzuki

      Return the expected value of the X^2 operator.



   .. py:method:: GaussianState.get_mean_y()
      :module: trottersuzuki

      Return the expected value of the Y operator.



   .. py:method:: GaussianState.get_mean_yy()
      :module: trottersuzuki

      Return the expected value of the Y^2 operator.



   .. py:method:: GaussianState.get_particle_density()
      :module: trottersuzuki

      Return a matrix storing the squared norm of the wave function.



   .. py:method:: GaussianState.get_phase()
      :module: trottersuzuki

      Return a matrix storing the phase of the wave function.



   .. py:method:: GaussianState.get_squared_norm()
      :module: trottersuzuki

      Return the squared norm of the quantum state.



   .. py:method:: GaussianState.write_particle_density(fileprefix)
      :module: trottersuzuki

      Write to a file the squared norm of the wave function.



   .. py:method:: GaussianState.write_phase(fileprefix)
      :module: trottersuzuki

      Write to a file the phase of the wave function.



   .. py:method:: GaussianState.write_to_file(fileprefix)
      :module: trottersuzuki

      Write to a file the wave function.



.. py:class:: SinusoidState(_grid, _n_x=1, _n_y=1, _norm=1, _phase=0, _p_real=None, _p_imag=None)
   :module: trottersuzuki

   This class defines a quantum state with sinusoidal like wave function.

   This class is a child of State class.

   C++ includes: trottersuzuki.h

   Constructors
   ------------
   * `SinusoidState(_grid, _n_x=1, _n_y=1, _norm=1, _phase=0, _p_real=None, _p_imag=None)`

       Construct the quantum state with sinusoidal like wave function.

       Parameters
       ----------
       * `grid` :
           Lattice object.
       * `n_x` :
           First quantum number.
       * `n_y` :
           Second quantum number.
       * `norm` :
           Squared norm of the quantum state.
       * `phase` :
           Relative phase of the wave function.
       * `p_real` :
           Pointer to the real part of the wave function.
       * `p_imag` :
           Pointer to the imaginary part of the wave function.

   C++ includes: trottersuzuki.h

   .. py:method:: SinusoidState.get_mean_px()
      :module: trottersuzuki

      Return the expected value of the P_x operator.



   .. py:method:: SinusoidState.get_mean_pxpx()
      :module: trottersuzuki

      Return the expected value of the P_x^2 operator.



   .. py:method:: SinusoidState.get_mean_py()
      :module: trottersuzuki

      Return the expected value of the P_y operator.



   .. py:method:: SinusoidState.get_mean_pypy()
      :module: trottersuzuki

      Return the expected value of the P_y^2 operator.



   .. py:method:: SinusoidState.get_mean_x()
      :module: trottersuzuki

      Return the expected value of the X operator.



   .. py:method:: SinusoidState.get_mean_xx()
      :module: trottersuzuki

      Return the expected value of the X^2 operator.



   .. py:method:: SinusoidState.get_mean_y()
      :module: trottersuzuki

      Return the expected value of the Y operator.



   .. py:method:: SinusoidState.get_mean_yy()
      :module: trottersuzuki

      Return the expected value of the Y^2 operator.



   .. py:method:: SinusoidState.get_particle_density()
      :module: trottersuzuki

      Return a matrix storing the squared norm of the wave function.



   .. py:method:: SinusoidState.get_phase()
      :module: trottersuzuki

      Return a matrix storing the phase of the wave function.



   .. py:method:: SinusoidState.get_squared_norm()
      :module: trottersuzuki

      Return the squared norm of the quantum state.



   .. py:method:: SinusoidState.write_particle_density(fileprefix)
      :module: trottersuzuki

      Write to a file the squared norm of the wave function.



   .. py:method:: SinusoidState.write_phase(fileprefix)
      :module: trottersuzuki

      Write to a file the phase of the wave function.



   .. py:method:: SinusoidState.write_to_file(fileprefix)
      :module: trottersuzuki

      Write to a file the wave function.


Potential Classes
=================
.. py:class:: Potential(*args)
   :module: trottersuzuki

   This class defines the external potential that is used for Hamiltonian class.

   C++ includes: trottersuzuki.h

   Constructors
   ------------
   * `Potential(*args)`

       Construct the external potential.

       Parameters
       ----------
       * `grid` :
           Lattice object.
       * `filename` :
           Name of the file that stores the external potential matrix.

   C++ includes: trottersuzuki.h


   .. py:method:: Potential.get_value(x, y)
      :module: trottersuzuki

      Get the value at the coordinate (x,y).



.. py:class:: HarmonicPotential(_grid, _omegax, _omegay, _mass=1.0, _mean_x=0.0, _mean_y=0.0)
   :module: trottersuzuki

   `HarmonicPotential(grid, omegax, omegay, mass=1., mean_x=0., mean_y=0.)`

   This class defines the external potential, that is used for Hamiltonian class.

   This class is a child of Potential class.

   Constructors
   ------------
   * `HarmonicPotential(grid, omegax, omegay, mass=1., mean_x=0., mean_y=0.)`

       Construct the harmonic external potential.

       Parameters:
       * `grid` :
           Lattice object.
       * `omegax` :
           Frequency along x axis.
       * `omegay` :
           Frequency along y axis.
       * `mass` :
           Mass of the particle.
       * `mean_x` :
           Minimum of the potential along x axis.
       * `mean_y` :
           Minimum of the potential along y axis.

   C++ includes: trottersuzuki.h



   .. py:method:: HarmonicPotential.get_value(x, y)
      :module: trottersuzuki

      Return the value of the external potential at coordinate (x,y)


Hamiltonian Classes
===================
.. py:class:: Hamiltonian(_grid, _potential=None, _mass=1.0, _coupling_a=0.0, _angular_velocity=0.0, _rot_coord_x=0, _rot_coord_y=0)
   :module: trottersuzuki

   `Hamiltonian(grid, potential=0, mass=1., coupling_a=0., angular_velocity=0.,
       rot_coord_x=0, rot_coord_y=0)`

   This class defines the Hamiltonian of a single component system.

   Constructors
   ------------
   * `Hamiltonian(grid, potential=0, mass=1., coupling_a=0., angular_velocity=0.,
       rot_coord_x=0, rot_coord_y=0)`

       Construct the Hamiltonian of a single component system.

       Parameters:
       * `grid` :
           Lattice object.
       * `potential` :
           Potential object.
       * `mass` :
           Mass of the particle.
       * `coupling_a` :
           Coupling constant of intra-particle interaction.
       * `angular_velocity` :
           The frame of reference rotates with this angular velocity.
       * `rot_coord_x` :
           X coordinate of the center of rotation.
       * `rot_coord_y` :
           Y coordinate of the center of rotation.

   C++ includes: trottersuzuki.h



.. py:class:: Hamiltonian2Component(_grid, _potential=None, _potential_b=None, _mass=1.0, _mass_b=1.0, _coupling_a=0.0, coupling_ab=0.0, _coupling_b=0.0, _omega_r=0, _omega_i=0, _angular_velocity=0.0, _rot_coord_x=0, _rot_coord_y=0)
   :module: trottersuzuki

   `Hamiltonian2Component(grid, potential=0, potential_b=0, mass=1., mass_b=1.,
       coupling_a=0., coupling_ab=0., coupling_b=0., omega_r=0, omega_i=0,
       angular_velocity=0., rot_coord_x=0, rot_coord_y=0)`

   This class defines the Hamiltonian of a two component system.

   Constructors
   ------------
   * `Hamiltonian2Component(grid, potential=0, potential_b=0, mass=1., mass_b=1.,
       coupling_a=0., coupling_ab=0., coupling_b=0., omega_r=0, omega_i=0,
       angular_velocity=0., rot_coord_x=0, rot_coord_y=0)`

       Construct the Hamiltonian of a two component system.

       Parameters:
       * `grid` :
           Lattice object.
       * `potential` :
           Potential of the first component.
       * `potential_b` :
           Potential of the second component.
       * `mass` :
           Mass of the first-component's particles.
       * `mass_b` :
           Mass of the second-component's particles.
       * `coupling_a` :
           Coupling constant of intra-particle interaction for the first component.
       * `coupling_ab` :
           Coupling constant of inter-particle interaction between the two
           components.
       * `coupling_b` :
           Coupling constant of intra-particle interaction for the second
           component.
       * `omega_r` :
           Real part of the Rabi coupling.
       * `omega_i` :
           Imaginary part of the Rabi coupling.
       * `angular_velocity` :
           The frame of reference rotates with this angular velocity.
       * `rot_coord_x` :
           X coordinate of the center of rotation.
       * `rot_coord_y` :
           Y coordinate of the center of rotation.

   C++ includes: trottersuzuki.h

Solver Class
============
.. py:class:: Solver(*args)
   :module: trottersuzuki

   `Solver(grid, state, hamiltonian, delta_t, kernel_type="cpu")`
   `Solver(grid, state1, state2, hamiltonian, delta_t, kernel_type="cpu")`

   This class defines the evolution tasks.

   Constructors
   ------------
   * `Solver(grid, state, hamiltonian, delta_t, kernel_type="cpu")`

       Construct the Solver object for a single-component system.

       Parameters:
       * `grid` :
           Lattice object.
       * `state` :
           State of the system.
       * `hamiltonian` :
           Hamiltonian of the system.
       * `delta_t` :
           A single evolution iteration, evolves the state for this time.
       * `kernel_type` :
           Which kernel to use (either cpu or gpu).

       Massively Parallel Trotter-Suzuki Solver

       This program is free software: you can redistribute it and/or modify it
       under the terms of the GNU General Public License as published by the Free
       Software Foundation, either version 3 of the License, or (at your option)
       any later version.

       This program is distributed in the hope that it will be useful, but WITHOUT
       ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
       FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
       more details.

       You should have received a copy of the GNU General Public License along with
       this program. If not, see http://www.gnu.org/licenses/.

   * `Solver(grid, state1, state2, hamiltonian, delta_t, kernel_type="cpu")`

       Construct the Solver object for a two-component system.

       Parameters:
       * `grid` :
           Lattice object.
       * `state1` :
           First component's state of the system.
       * `state2` :
           Second component's state of the system.
       * `hamiltonian` :
           Hamiltonian of the two-component system.
       * `delta_t` :
           A single evolution iteration, evolves the state for this time.
       * `kernel_type` :
           Which kernel to use (either cpu or gpu).

   C++ includes: trottersuzuki.h



   .. py:method:: Solver.evolve(iterations, imag_time=False)
      :module: trottersuzuki

      Evolve the state of the system.



   .. py:method:: Solver.get_inter_species_energy()
      :module: trottersuzuki

      Get the inter-particles interaction energy of the system.



   .. py:method:: Solver.get_intra_species_energy(which=3)
      :module: trottersuzuki

      Get the intra-particles interaction energy of the system.



   .. py:method:: Solver.get_kinetic_energy(which=3)
      :module: trottersuzuki

      Get the kinetic energy of the system.



   .. py:method:: Solver.get_potential_energy(which=3)
      :module: trottersuzuki

      Get the potential energy of the system.



   .. py:method:: Solver.get_rabi_energy()
      :module: trottersuzuki

      Get the Rabi energy of the system.



   .. py:method:: Solver.get_rotational_energy(which=3)
      :module: trottersuzuki

      Get the rotational energy of the system.



   .. py:method:: Solver.get_squared_norm(which=3)
      :module: trottersuzuki

      Get the squared norm of the state (default: total wave-function).



   .. py:method:: Solver.get_total_energy()
      :module: trottersuzuki

      Get the total energy of the system.
