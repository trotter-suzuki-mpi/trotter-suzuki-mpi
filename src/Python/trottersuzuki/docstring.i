
// File: index.xml

// File: classCPUBlock.xml


%feature("docstring") CPUBlock "

This class defines the CPU kernel.  

This kernel provides real time and imaginary time evolution of a quantum state,
using CPUs. It implements a solver for a single or two wave functions, whose
evolution is governed by nonlinear Schrodinger equation (Gross Pitaevskii
equation). The Hamiltonian of the physical system includes:  

*   time-dependent external potential  
*   rotating system of reference  
*   intra species interaction  
*   extra species interaction  
*   Rabi coupling  

C++ includes: kernel.h
";

%feature("docstring") CPUBlock::run_kernel_on_halo "

Evolve blocks of wave function at the edge of the tile. This comprises the
halos.  
";

%feature("docstring") CPUBlock::~CPUBlock "
";

%feature("docstring") CPUBlock::start_halo_exchange "

Start vertical halos exchange.  
";

%feature("docstring") CPUBlock::calculate_squared_norm "

Calculate squared norm of the state.  
";

%feature("docstring") CPUBlock::normalization "

Normalize the state when performing an imaginary time evolution (only two wave-
function evolution).  
";

%feature("docstring") CPUBlock::get_sample "

Copy the wave function from the two buffers pointed by p_real and p_imag,
without halos, to dest_real and dest_imag.  
";

%feature("docstring") CPUBlock::run_kernel "

Evolve the remaining blocks in the inner part of the tile.  
";

%feature("docstring") CPUBlock::CPUBlock "

Instantiate the kernel for single wave functions state evolution.  
";

%feature("docstring") CPUBlock::CPUBlock "

Instantiate the kernel for two wave functions state evolution.  
";

%feature("docstring") CPUBlock::finish_halo_exchange "

Start horizontal halos exchange.  
";

%feature("docstring") CPUBlock::wait_for_completion "

Synchronize all the processes at the end of halos communication. Perform
normalization for imaginary time evolution in the case of single wave-function
evolution.  
";

%feature("docstring") CPUBlock::get_name "

Get kernel name.  
";

%feature("docstring") CPUBlock::runs_in_place "
";

%feature("docstring") CPUBlock::rabi_coupling "

Evolution corresponding to the Rabi coupling term of the Hamiltonian (only two
wave-function evolution).  
";

%feature("docstring") CPUBlock::update_potential "

Update memory pointed by external_potential_real and external_potential_imag
(only non static external potential).  
";

// File: classExponentialState.xml


%feature("docstring") ExponentialState "

This class defines a quantum state with exponential like wave function.  

This class is a child of State class.  

C++ includes: trottersuzuki.h
";

%feature("docstring") ExponentialState::ExponentialState "

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
";

// File: classGaussianState.xml


%feature("docstring") GaussianState "

This class defines a quantum state with gaussian like wave function.  

This class is a child of State class.  

C++ includes: trottersuzuki.h
";

%feature("docstring") GaussianState::GaussianState "

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
";

// File: classHamiltonian.xml


%feature("docstring") Hamiltonian "

This class defines the Hamiltonian of a single component system.  

C++ includes: trottersuzuki.h
";

%feature("docstring") Hamiltonian::~Hamiltonian "
";

%feature("docstring") Hamiltonian::Hamiltonian "

Construct the Hamiltonian of a single component system.  

Parameters
----------
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
";

// File: classHamiltonian2Component.xml


%feature("docstring") Hamiltonian2Component "

This class defines the Hamiltonian of a two component system.  

C++ includes: trottersuzuki.h
";

%feature("docstring") Hamiltonian2Component::Hamiltonian2Component "

Construct the Hamiltonian of a two component system.  

Parameters
----------
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
    Coupling constant of inter-particle interaction between the two components.  
* `coupling_b` :  
    Coupling constant of intra-particle interaction for the second component.  
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
";

%feature("docstring") Hamiltonian2Component::~Hamiltonian2Component "
";

// File: classHarmonicPotential.xml


%feature("docstring") HarmonicPotential "

This class defines the external potential, that is used for Hamiltonian class.  

This class is a child of Potential class.  

C++ includes: trottersuzuki.h
";

%feature("docstring") HarmonicPotential::get_value "

Return the value of the external potential at coordinate (x,y)  
";

%feature("docstring") HarmonicPotential::HarmonicPotential "

Construct the harmonic external potential.  

Parameters
----------
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
";

%feature("docstring") HarmonicPotential::~HarmonicPotential "
";

// File: classITrotterKernel.xml


%feature("docstring") ITrotterKernel "

This class defines the prototipe of the kernel classes: CPU, GPU, Hybrid.  

C++ includes: trottersuzuki.h
";

%feature("docstring") ITrotterKernel::calculate_squared_norm "

Calculate the squared norm of the wave function.  
";

%feature("docstring") ITrotterKernel::rabi_coupling "

Perform the evolution regarding the Rabi coupling.  
";

%feature("docstring") ITrotterKernel::update_potential "

Update the evolution matrix, regarding the external potential, at time t.  
";

%feature("docstring") ITrotterKernel::runs_in_place "
";

%feature("docstring") ITrotterKernel::run_kernel "

Evolve the remaining blocks in the inner part of the tile.  
";

%feature("docstring") ITrotterKernel::finish_halo_exchange "

Exchange halos between processes.  
";

%feature("docstring") ITrotterKernel::get_sample "

Get the evolved wave function.  
";

%feature("docstring") ITrotterKernel::get_name "

Get kernel name.  
";

%feature("docstring") ITrotterKernel::run_kernel_on_halo "

Evolve blocks of wave function at the edge of the tile. This comprises the
halos.  
";

%feature("docstring") ITrotterKernel::start_halo_exchange "

Exchange halos between processes.  
";

%feature("docstring") ITrotterKernel::wait_for_completion "

Sincronize all the processes at the end of halos communication. Perform
normalization for imaginary time evolution.  
";

%feature("docstring") ITrotterKernel::~ITrotterKernel "
";

%feature("docstring") ITrotterKernel::normalization "

Normalization of the two components wave function.  
";

// File: classLattice.xml


%feature("docstring") Lattice "

This class defines the lattice structure over which the state and potential
matrices are defined.  

As to single-process execution, the lattice is a single tile which can be
surrounded by a halo, in the case of periodic boundary conditions. As to multi-
process execution, the lattice is divided in smaller lattices, dubbed tiles, one
for each process. Each of the tiles is surrounded by a halo.  

C++ includes: trottersuzuki.h
";

%feature("docstring") Lattice::Lattice "

Lattice constructor.  

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
";

// File: structoption.xml


%feature("docstring") option "
";

// File: classPotential.xml


%feature("docstring") Potential "

This class defines the external potential, that is used for Hamiltonian class.  

C++ includes: trottersuzuki.h
";

%feature("docstring") Potential::Potential "

Construct the external potential.  

Parameters
----------
* `grid` :  
    Lattice object.  
* `filename` :  
    Name of the file that stores the external potential matrix.  
";

%feature("docstring") Potential::Potential "

Construct the external potential.  

Parameters
----------
* `grid` :  
    Lattice object.  
* `external_pot` :  
    Pointer to the external potential matrix.  
";

%feature("docstring") Potential::Potential "

Construct the external potential.  

Parameters
----------
* `grid` :  
    Lattice object.  
* `potential_function` :  
    Pointer to the static external potential function.  
";

%feature("docstring") Potential::Potential "

Construct the external potential.  

Parameters
----------
* `grid` :  
    Lattice object.  
* `potential_function` :  
    Pointer to the time-dependent external potential function.  
";

%feature("docstring") Potential::update "

Update the potential matrix at time t.  
";

%feature("docstring") Potential::~Potential "
";

%feature("docstring") Potential::get_value "

Get the value at the coordinate (x,y).  
";

// File: classSinusoidState.xml


%feature("docstring") SinusoidState "

This class defines a quantum state with sinusoidal like wave function.  

This class is a child of State class.  

C++ includes: trottersuzuki.h
";

%feature("docstring") SinusoidState::SinusoidState "

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
";

// File: classSolver.xml


%feature("docstring") Solver "

This class defines the evolution tasks.  

C++ includes: trottersuzuki.h
";

%feature("docstring") Solver::get_kinetic_energy "

Get the kinetic energy of the system.  
";

%feature("docstring") Solver::get_rabi_energy "

Get the Rabi energy of the system.  
";

%feature("docstring") Solver::get_squared_norm "

Get the squared norm of the state (default: total wave-function).  
";

%feature("docstring") Solver::~Solver "
";

%feature("docstring") Solver::get_intra_species_energy "

Get the intra-particles interaction energy of the system.  
";

%feature("docstring") Solver::get_rotational_energy "

Get the rotational energy of the system.  
";

%feature("docstring") Solver::get_inter_species_energy "

Get the inter-particles interaction energy of the system.  
";

%feature("docstring") Solver::get_total_energy "

Get the total energy of the system.  
";

%feature("docstring") Solver::evolve "

Evolve the state of the system.  
";

%feature("docstring") Solver::Solver "

Construct the Solver object for a single-component system.  

Parameters
----------
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

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.  

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.  

You should have received a copy of the GNU General Public License along with
this program. If not, see http://www.gnu.org/licenses/.  
";

%feature("docstring") Solver::Solver "

Construct the Solver object for a two-component system.  

Parameters
----------
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
";

%feature("docstring") Solver::get_potential_energy "

Get the potential energy of the system.  
";

// File: classState.xml


%feature("docstring") State "

This class defines the quantum state.  

C++ includes: trottersuzuki.h
";

%feature("docstring") State::write_particle_density "

Write to a file the squared norm of the wave function.  
";

%feature("docstring") State::init_state "

Write the wave function from a C++ function to p_real and p_imag matrices.  
";

%feature("docstring") State::get_phase "

Return a matrix storing the phase of the wave function.  
";

%feature("docstring") State::write_phase "

Write to a file the phase of the wave function.  
";

%feature("docstring") State::get_mean_yy "

Return the expected value of the Y^2 operator.  
";

%feature("docstring") State::~State "

Destructor.  
";

%feature("docstring") State::get_mean_px "

Return the expected value of the P_x operator.  
";

%feature("docstring") State::get_mean_py "

Return the expected value of the P_y operator.  
";

%feature("docstring") State::State "

Construct the state from given matrices if they are provided, otherwise
construct a state with null wave function, initializing p_real and p_imag.  

Parameters
----------
* `grid` :  
    Lattice object.  
* `p_real` :  
    Pointer to the real part of the wave function.  
* `p_imag` :  
    Pointer to the imaginary part of the wave function.  
";

%feature("docstring") State::State "

Copy constructor: copy the state object.  
";

%feature("docstring") State::imprint "

Multiply the wave function of the state by the function provided.  
";

%feature("docstring") State::get_mean_pxpx "

Return the expected value of the P_x^2 operator.  
";

%feature("docstring") State::get_squared_norm "

Return the squared norm of the quantum state.  
";

%feature("docstring") State::get_mean_xx "

Return the expected value of the X^2 operator.  
";

%feature("docstring") State::loadtxt "

Load the wave function from a file to p_real and p_imag matrices.  
";

%feature("docstring") State::get_particle_density "

Return a matrix storing the squared norm of the wave function.  
";

%feature("docstring") State::get_mean_y "

Return the expected value of the Y operator.  
";

%feature("docstring") State::get_mean_x "

Return the expected value of the X operator.  
";

%feature("docstring") State::get_mean_pypy "

Return the expected value of the P_y^2 operator.  
";

%feature("docstring") State::write_to_file "

Write to a file the wave function.  
";

// File: namespacestd.xml

// File: config_8h.xml

// File: download_8md.xml

// File: examples_8md.xml

// File: tutorial_8md.xml

// File: README_8md.xml

// File: common_8cpp.xml

%feature("docstring") memcpy2D "
";

%feature("docstring") stamp_matrix "
";

%feature("docstring") stamp "

Massively Parallel Trotter-Suzuki Solver  

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.  

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.  

You should have received a copy of the GNU General Public License along with
this program. If not, see http://www.gnu.org/licenses/.  
";

%feature("docstring") add_padding "
";

%feature("docstring") print_matrix "
";

%feature("docstring") my_abort "

Massively Parallel Trotter-Suzuki Solver  

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.  

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.  

You should have received a copy of the GNU General Public License along with
this program. If not, see http://www.gnu.org/licenses/.  
";

%feature("docstring") print_complex_matrix "
";

// File: common_8h.xml

%feature("docstring") stamp "

Massively Parallel Trotter-Suzuki Solver  

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.  

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.  

You should have received a copy of the GNU General Public License along with
this program. If not, see http://www.gnu.org/licenses/.  
";

%feature("docstring") my_abort "

Massively Parallel Trotter-Suzuki Solver  

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.  

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.  

You should have received a copy of the GNU General Public License along with
this program. If not, see http://www.gnu.org/licenses/.  
";

%feature("docstring") memcpy2D "
";

%feature("docstring") stamp_matrix "
";

// File: cpublock_8cpp.xml

%feature("docstring") block_kernel_potential_imaginary "
";

%feature("docstring") full_step_imaginary "
";

%feature("docstring") block_kernel_vertical_imaginary "
";

%feature("docstring") process_sides "
";

%feature("docstring") rabi_coupling_real "
";

%feature("docstring") full_step "
";

%feature("docstring") block_kernel_rotation "
";

%feature("docstring") block_kernel_vertical "

Massively Parallel Trotter-Suzuki Solver  

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.  

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.  

You should have received a copy of the GNU General Public License along with
this program. If not, see http://www.gnu.org/licenses/.  
";

%feature("docstring") process_band "
";

%feature("docstring") block_kernel_horizontal "
";

%feature("docstring") block_kernel_rotation_imaginary "
";

%feature("docstring") block_kernel_potential "
";

%feature("docstring") rabi_coupling_imaginary "
";

%feature("docstring") block_kernel_horizontal_imaginary "
";

// File: getopt_8c.xml

%feature("docstring") warnx "
";

%feature("docstring") _vwarnx "
";

%feature("docstring") gcd "
";

%feature("docstring") parse_long_options "
";

%feature("docstring") getopt_long_only "
";

%feature("docstring") getopt_internal "
";

%feature("docstring") getopt "
";

%feature("docstring") getopt_long "
";

%feature("docstring") permute_args "
";

// File: getopt_8h.xml

%feature("docstring") getopt "
";

%feature("docstring") getopt_long "
";

%feature("docstring") getopt_long_only "
";

// File: kernel_8h.xml

%feature("docstring") process_band "
";

// File: main_8cpp.xml

%feature("docstring") process_command_line "
";

%feature("docstring") main "
";

%feature("docstring") print_usage "
";

// File: model_8cpp.xml

%feature("docstring") calculate_borders "

Massively Parallel Trotter-Suzuki Solver  

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.  

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.  

You should have received a copy of the GNU General Public License along with
this program. If not, see http://www.gnu.org/licenses/.  
";

%feature("docstring") const_potential "

Defines the null potential function.  
";

// File: solver_8cpp.xml

// File: trottersuzuki_8h.xml

%feature("docstring") std::const_potential "

Defines the null potential function.  
";

// File: unistd_8h.xml

// File: md_download.xml

// File: md_examples.xml

// File: md_tutorial.xml

// File: dir_68267d1309a1af8e8297ef4c3efbcdba.xml

// File: indexpage.xml

