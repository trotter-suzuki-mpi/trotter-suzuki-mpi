
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

";

%feature("docstring") ExponentialState::ExponentialState "

Construct the quantum state with exponential like wave function.  

Parameters
----------
* `grid` : Lattice object 
    Defines the geometry of the simulation.  
* `n_x` : integer,optional (default: 1)
    First quantum number.  
* `n_y` : integer,optional (default: 1)
    Second quantum number.  
* `norm` : float,optional (default: 1)
    Squared norm of the quantum state.  
* `phase` : float,optional (default: 0)
    Relative phase of the wave function. 

Returns
-------
* `ExponentialState` : State object. 
    Quantum state with exponential like wave function. The wave function is give by:\n
    
    .. math:: \psi(x,y) = \sqrt{N}/L \mathrm{e}^{i 2 \pi (n_x x + n_y y) / L} \mathrm{e}^{i \phi}
    
    being :math:`N` the norm of the state, :math:`L` 
    the length of the lattice edge, :math:`n_x` and :math:`n_y` the quantum numbers 
    and :math:`\phi` the relative phase.

Notes
-----
The geometry of the simulation has to have periodic boundary condition 
to use Exponential state as initial state of a real time evolution. 
Indeed, the wave function is not null at the edges of the space.

Example
-------

    >>> import trottersuzuki as ts  # import the module
    >>> grid = ts.Lattice(300, 30., True, True)  # Define the simulation's geometry
    >>> state = ts.ExponentialState(grid, 2, 1)  # Create the system's state
";

// File: classGaussianState.xml


%feature("docstring") GaussianState " 

";

%feature("docstring") GaussianState::GaussianState "

Construct the quantum state with gaussian like wave function.  

Parameters
----------
* `grid` : Lattice object 
    Defines the geometry of the simulation.  
* `omega_x` : float
    Inverse of the variance along x-axis.  
* `omega_y` : float, optional (default: omega_x) 
    Inverse of the variance along y-axis.
* `mean_x` : float, optional (default: 0)
    X coordinate of the gaussian function's peak.  
* `mean_y` : float, optional (default: 0)
    Y coordinate of the gaussian function's peak.  
* `norm` : float, optional (default: 1) 
    Squared norm of the state.  
* `phase` : float, optional (default: 0) 
    Relative phase of the wave function. 

Returns
-------
* `GaussianState` : State object. 
    Quantum state with gaussian like wave function. The wave function is given by:\n
    
    .. math:: \psi(x,y) = (N/\pi)^{1/2} (\omega_x \omega_y)^{1/4} \mathrm{e}^{-(\omega_x(x-\mu_x)^2 + \omega_y(y-\mu_y)^2)/2} \mathrm{e}^{i \phi}
    
    being :math:`N` the norm of the state, :math:`\omega_x` and :math:`\omega_y` 
    the inverse of the variances, :math:`\mu_x` and :math:`\mu_y` the coordinates of the
    function's peak and :math:`\phi` the relative phase.

Notes
-----
The physical dimensions of the Lattice have to be enough to ensure that 
the wave function is almost zero at the edges.

Example
-------

    >>> import trottersuzuki as ts  # import the module
    >>> grid = ts.Lattice(300, 30.)  # Define the simulation's geometry
    >>> state = ts.GaussianState(grid, 2.)  # Create the system's state
";

// File: classHamiltonian.xml


%feature("docstring") Hamiltonian "

";

%feature("docstring") Hamiltonian::~Hamiltonian "
";

%feature("docstring") Hamiltonian::Hamiltonian "

Construct the Hamiltonian of a single component system.  

Parameters
----------
* `grid` : Lattice object 
    Define the geometry of the simulation.  
* `potential` : Potential object 
    Define the external potential of the Hamiltonian (:math:`V`).  
* `mass` : float,optional (default: 1.) 
    Mass of the particle (:math:`m`).  
* `coupling` : float,optional (default: 0.) 
    Coupling constant of intra-particle interaction (:math:`g`).  
* `angular_velocity` : float,optional (default: 0.) 
    The frame of reference rotates with this angular velocity (:math:`\omega`).  
* `rot_coord_x` : float,optional (default: 0.) 
    X coordinate of the center of rotation.  
* `rot_coord_y` : float,optional (default: 0.)
    Y coordinate of the center of rotation.

Returns
-------
* `Hamiltonian` : Hamiltonian object
    Hamiltonian of the system to be simulated: 
    
    .. math:: H(x,y) = 1/(2m)(P_x^2 + P_y^2)  + V(x,y) + g |\psi(x,y)|^2 + \omega L_z
    
    being :math:`m` the particle mass, :math:`V(x,y)` the external potential, 
    :math:`g` the coupling constant of intra-particle interaction, :math:`\omega` 
    the angular velocity of the frame of reference  and :math:`L_z` the angular momentum operator along the z-axis.

Example
-------

    >>> import trottersuzuki as ts  # import the module
    >>> grid = ts.Lattice()  # Define the simulation's geometry
    >>> potential = ts.HarmonicPotential(grid, 1., 1.)  # Create an harmonic external potential
    >>> hamiltonian = ts.Hamiltonian(grid, potential)  # Create the Hamiltonian of an harmonic oscillator
";

// File: classHamiltonian2Component.xml

%feature("docstring") Hamiltonian2Component "

";

%feature("docstring") Hamiltonian2Component::Hamiltonian2Component "

Construct the Hamiltonian of a two component system.  

Parameters
----------
* `grid` : Lattice object  
    Define the geometry of the simulation.  
* `potential_1` : Potential object 
    External potential to which the first state is subjected (:math:`V_1`).  
* `potential_2` : Potential object 
    External potential to which the second state is subjected (:math:`V_2`).  
* `mass_1` : float,optional (default: 1.) 
    Mass of the first-component's particles (:math:`m_1`).  
* `mass_2` : float,optional (default: 1.) 
    Mass of the second-component's particles (:math:`m_2`).  
* `coupling_1` : float,optional (default: 0.) 
    Coupling constant of intra-particle interaction for the first component (:math:`g_1`).  
* `coupling_12` : float,optional (default: 0.) 
    Coupling constant of inter-particle interaction between the two components (:math:`g_{12}`).  
* `coupling_2` : float,optional (default: 0.) 
    Coupling constant of intra-particle interaction for the second component (:math:`g_2`).  
* `omega_r` : float,optional (default: 0.) 
    Real part of the Rabi coupling (:math:`\mathrm{Re}(\Omega)`).  
* `omega_i` : float,optional (default: 0.) 
    Imaginary part of the Rabi coupling (:math:`\mathrm{Im}(\Omega)`).  
* `angular_velocity` : float,optional (default: 0.) 
    The frame of reference rotates with this angular velocity (:math:`\omega`).  
* `rot_coord_x` : float,optional (default: 0.) 
    X coordinate of the center of rotation.  
* `rot_coord_y` : float,optional (default: 0.) 
    Y coordinate of the center of rotation.  

Returns
-------
* `Hamiltonian2Component` : Hamiltonian2Component object 
    Hamiltonian of the two-component system to be simulated.
    
    .. math::
    
       H(x,y)(\psi_1,\psi_2) &= \n
    
       &(1/(2m_1)(P_x^2 + P_y^2) + V_1(x,y) + g_1 |\psi_1(x,y)|^2 + g_{12} |\psi_2(x,y)|^2 + \omega L_z)\psi_1 + \Omega \psi_2 /2 \n
       &(1/(2m_2)(P_x^2 + P_y^2) + V_2(x,y) + g_2 |\psi_2(x,y)|^2 + g_{12} |\psi_1(x,y)|^2 + \omega L_z)\psi_2 + \Omega \psi_1 /2
    
    
    being, for the i-th component, :math:`m_i` the particle mass, :math:`V_i(x,y)` the external potential, 
    :math:`g_i` the coupling constant of intra-particle interaction; 
    :math:`g_{12}` the coupling constant of inter-particle interaction 
    :math:`\omega` the angular velocity of the frame of reference, :math:`L_z` the angular momentum operator along the z-axis 
    and :math:`\Omega` the Rabi coupling.
    
Example
-------

    >>> import trottersuzuki as ts  # import the module
    >>> grid = ts.Lattice()  # Define the simulation's geometry
    >>> potential = ts.HarmonicPotential(grid, 1., 1.)  # Create an harmonic external potential
    >>> hamiltonian = ts.Hamiltonian2Component(grid, potential, potential)  # Create the Hamiltonian of an harmonic oscillator for a two-component system
";

%feature("docstring") Hamiltonian2Component::~Hamiltonian2Component "
";

// File: classHarmonicPotential.xml


%feature("docstring") HarmonicPotential "

";

%feature("docstring") HarmonicPotential::get_value "

Return the value of the external potential at coordinate (x,y)  
";

%feature("docstring") HarmonicPotential::HarmonicPotential "

Construct the harmonic external potential.  

Parameters
----------
* `grid` : Lattice object  
    Define the geometry of the simulation.  
* `omegax` : float
    Frequency along x-axis.  
* `omegay` : float 
    Frequency along y-axis.  
* `mass` : float,optional (default: 1.) 
    Mass of the particle.  
* `mean_x` : float,optional (default: 0.) 
    Minimum of the potential along x axis.  
* `mean_y` : float,optional (default: 0.) 
    Minimum of the potential along y axis.  

Returns
-------
* `HarmonicPotential` : Potential object 
    Harmonic external potential.

Notes
-----
External potential function:\n

.. math:: V(x,y) = 1/2 m (\omega_x^2  x^2 + \omega_y^2 y^2)

being :math:`m` the particle mass, :math:`\omega_x` and :math:`\omega_y` the potential frequencies.

Example
-------

    >>> import trottersuzuki as ts  # import the module
    >>> grid = ts.Lattice()  # Define the simulation's geometry
    >>> potential = ts.HarmonicPotential(grid, 2., 1.)  # Create an harmonic external potential
    
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

";

%feature("docstring") Lattice::Lattice "

Lattice constructor.  

Parameters
----------
* `dim` : integer,optional (default: 100)
    Linear dimension of the squared lattice.  
* `length` : float,optional (default: 20.)
    Physical length of the lattice's side.   
* `periodic_x_axis` : bool,optional (default: False)
    Boundary condition along the x axis (false=closed, true=periodic).  
* `periodic_y_axis` : bool,optional (default: False) 
    Boundary condition along the y axis (false=closed, true=periodic).

Returns
-------
* `Lattice` : Lattice object 
    Define the geometry of the simulation.

Notes
-----
The lattice created is squared.

Example
-------

    >>> import trottersuzuki as ts  # import the module
    >>> # Generate a 200x200 Lattice with physical dimensions of 30x30
    >>> # and closed boundary conditions.
    >>> grid = ts.Lattice(200, 30.)
  
";

// File: structoption.xml


%feature("docstring") option "
";

// File: classPotential.xml


%feature("docstring") Potential "

";

%feature("docstring") Potential::Potential "

Construct the external potential.  

Parameters
----------
* `grid` : Lattice object 
    Define the geometry of the simulation.  
* `filename` :  string,optional
    Name of the file that stores the external potential matrix.  

Returns
-------
* `Potential` : Potential object 
    Create external potential.

Example
-------

    >>> import trottersuzuki as ts  # import the module
    >>> grid = ts.Lattice()  # Define the simulation's geometry
    >>> # Define a constant external potential
    >>> def external_potential_function(x,y):
    >>>     return 1.
    >>> potential = ts.Potential(grid)  # Create the external potential
    >>> potential.init_potential(external_potential_function)  # Initialize the external potential

";

%feature("docstring") Potential::Potential "

Construct the external potential.  

Parameters
----------
* `grid` : Lattice object 
    Define the geometry of the simulation.  

Returns
-------
* `Potential` : Potential object 
    Create external potential.

Example
-------

    >>> import trottersuzuki as ts  # import the module
    >>> grid = ts.Lattice()  # Define the simulation's geometry
    >>> # Define a constant external potential
    >>> def external_potential_function(x,y):
    >>>     return 1.
    >>> potential = ts.Potential(grid)  # Create the external potential
    >>> potential.init_potential(external_potential_function)  # Initialize the external potential
";

%feature("docstring") Potential::Potential "

Construct the external potential.  

Parameters
----------
* `grid` : Lattice object 
    Define the geometry of the simulation. 
  
Returns
-------
* `Potential` : Potential object 
    Create external potential.

Example
-------

    >>> import trottersuzuki as ts  # import the module
    >>> grid = ts.Lattice()  # Define the simulation's geometry
    >>> # Define a constant external potential
    >>> def external_potential_function(x,y):
    >>>     return 1.
    >>> potential = ts.Potential(grid)  # Create the external potential
    >>> potential.init_potential(external_potential_function)  # Initialize the external potential 
";

%feature("docstring") Potential::Potential "

Construct the external potential.  

Parameters
----------
* `grid` : Lattice object 
    Define the geometry of the simulation.  

Returns
-------
* `Potential` : Potential object 
    Create external potential.

Example
-------

    >>> import trottersuzuki as ts  # import the module
    >>> grid = ts.Lattice()  # Define the simulation's geometry
    >>> # Define a constant external potential
    >>> def external_potential_function(x,y):
    >>>     return 1.
    >>> potential = ts.Potential(grid)  # Create the external potential
    >>> potential.init_potential(external_potential_function)  # Initialize the external potential
";

%feature("docstring") Potential::update "

Update the potential matrix at time t.  
";

%feature("docstring") Potential::~Potential "
";

%feature("docstring") Potential::get_value "

Get the value at the lattice's coordinate (x,y).

Returns
-------
* `value` : float
    Value of the external potential.
";

// File: classSinusoidState.xml


%feature("docstring") SinusoidState "

";

%feature("docstring") SinusoidState::SinusoidState "

Construct the quantum state with sinusoidal like wave function.  

Parameters
----------
* `grid` : Lattice object  
    Define the geometry of the simulation.  
* `n_x` : integer, optional (default: 1) 
    First quantum number.  
* `n_y` : integer, optional (default: 1)  
    Second quantum number.  
* `norm` : float, optional (default: 1)  
    Squared norm of the quantum state.  
* `phase` : float, optional (default: 1) 
    Relative phase of the wave function.

Returns
-------
* `SinusoidState` : State object. 
    Quantum state with sinusoidal like wave function. The wave function is given by:
    
    .. math:: \psi(x,y) = 2\sqrt{N}/L \sin(2\pi n_x x / L) \sin(2\pi n_y y / L) \mathrm{e}^{(i \phi)}
    
    being :math:`N` the norm of the state, :math:`L` 
    the length of the lattice edge, :math:`n_x` and :math:`n_y` the quantum numbers 
    and :math:`\phi` the relative phase.

Example
-------

    >>> import trottersuzuki as ts  # import the module
    >>> grid = ts.Lattice(300, 30., True, True)  # Define the simulation's geometry
    >>> state = ts.SinusoidState(grid, 2, 0)  # Create the system's state

";

// File: classSolver.xml


%feature("docstring") Solver "

";

%feature("docstring") Solver::get_kinetic_energy "

Get the kinetic energy of the system.

Parameters
----------
* `which` : integer,optional (default: 3)
    Which kinetic energy to return: total system (default, which=3), first component (which=1), second component (which=2). 

Returns
-------
* `get_kinetic_energy` : float
    kinetic energy of the system.

Example
-------

    >>> import trottersuzuki as ts  # import the module
    >>> grid = ts.Lattice()  # Define the simulation's geometry
    >>> state = ts.GaussianState(grid, 1.)  # Create the system's state
    >>> potential = ts.HarmonicPotential(grid, 1., 1.)  # Create harmonic potential
    >>> hamiltonian = ts.Hamiltonian(grid, potential)  # Create a harmonic oscillator Hamiltonian
    >>> solver = ts.Solver(grid, state, hamiltonian, 1e-2)  # Create the solver
    >>> solver.get_kinetic_energy()  # Get the kinetic energy
    0.5
";

%feature("docstring") Solver::get_rabi_energy "

Get the Rabi energy of the system.

Returns
-------
* `get_rabi_energy` : float
    Rabi energy of the system.  
";

%feature("docstring") Solver::get_squared_norm "

Get the squared norm of the state (default: total wave-function).

Parameters
----------
* `which` : integer,optional (default: 3)
    Which squared state norm to return: total system (default, which=3), first component (which=1), second component (which=2). 

Returns
-------
* `get_squared_norm` : float
    Squared norm of the state.   
";

%feature("docstring") Solver::~Solver "
";

%feature("docstring") Solver::get_intra_species_energy "

Get the intra-particles interaction energy of the system.  

Parameters
----------
* `which` : integer,optional (default: 3)
    Which intra-particles interaction energy to return: total system (default, which=3), first component (which=1), second component (which=2). 

Returns
-------
* `get_intra_species_energy` : float
    Intra-particles interaction energy of the system. 
";

%feature("docstring") Solver::get_rotational_energy "

Get the rotational energy of the system.

Parameters
----------
* `which` : integer,optional (default: 3)
    Which rotational energy to return: total system (default, which=3), first component (which=1), second component (which=2). 

Returns
-------
* `get_rotational_energy` : float
    Rotational energy of the system.
";

%feature("docstring") Solver::get_inter_species_energy "

Get the inter-particles interaction energy of the system.  

Returns
-------
* `get_inter_species_energy` : float
    Inter-particles interaction energy of the system. 
";

%feature("docstring") Solver::get_total_energy "

Get the total energy of the system.

Returns
-------
* `get_total_energy` : float
    Total energy of the system.  

Example
-------

    >>> import trottersuzuki as ts  # import the module
    >>> grid = ts.Lattice()  # Define the simulation's geometry
    >>> state = ts.GaussianState(grid, 1.)  # Create the system's state
    >>> potential = ts.HarmonicPotential(grid, 1., 1.)  # Create harmonic potential
    >>> hamiltonian = ts.Hamiltonian(grid, potential)  # Create a harmonic oscillator Hamiltonian
    >>> solver = ts.Solver(grid, state, hamiltonian, 1e-2)  # Create the solver
    >>> solver.get_total_energy()  # Get the total energy
    1
";

%feature("docstring") Solver::evolve "

Evolve the state of the system.

Parameters
----------
* `iterations` : integer 
    Number of iterations.
* `imag_time` : bool,optional (default: False)  
    Whether to perform imaginary time evolution (True) or real time evolution (False).    

Notes
-----

The norm of the state is preserved both in real-time and in imaginary-time evolution.

Example
-------

    >>> import trottersuzuki as ts  # import the module
    >>> grid = ts.Lattice()  # Define the simulation's geometry
    >>> state = ts.GaussianState(grid, 1.)  # Create the system's state
    >>> potential = ts.HarmonicPotential(grid, 1., 1.)  # Create harmonic potential
    >>> hamiltonian = ts.Hamiltonian(grid, potential)  # Create a harmonic oscillator Hamiltonian
    >>> solver = ts.Solver(grid, state, hamiltonian, 1e-2)  # Create the solver
    >>> solver.evolve(1000)  # perform 1000 iteration in real time evolution
";

%feature("docstring") Solver::update_parameters "

Notify the solver if any parameter changed in the Hamiltonian

";

%feature("docstring") Solver::Solver "

Construct the Solver object for a single-component system.  

Parameters
----------
* `grid` : Lattice object  
    Define the geometry of the simulation.  
* `state` : State object 
    State of the system.  
* `hamiltonian` : Hamiltonian object 
    Hamiltonian of the system.  
* `delta_t` : float 
    A single evolution iteration, evolves the state for this time.  
* `kernel_type` : string,optional (default: 'cpu') 
    Which kernel to use (either cpu or gpu).  

Returns
-------
* `Solver` : Solver object  
    Solver object for the simulation of a single-component system.
    
Example
-------

    >>> import trottersuzuki as ts  # import the module
    >>> grid = ts.Lattice()  # Define the simulation's geometry
    >>> state = ts.GaussianState(grid, 1.)  # Create the system's state
    >>> potential = ts.HarmonicPotential(grid, 1., 1.)  # Create harmonic potential
    >>> hamiltonian = ts.Hamiltonian(grid, potential)  # Create a harmonic oscillator Hamiltonian
    >>> solver = ts.Solver(grid, state, hamiltonian, 1e-2)  # Create the solver

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
* `grid` : Lattice object  
    Define the geometry of the simulation.
* `state1` : State object
    First component's state of the system.  
* `state2` : State object 
    Second component's state of the system.  
* `hamiltonian` : Hamiltonian object
    Hamiltonian of the two-component system.  
* `delta_t` : float
    A single evolution iteration, evolves the state for this time.  
* `kernel_type` : string,optional (default: 'cpu') 
    Which kernel to use (either cpu or gpu).  

Returns
-------
* `Solver` : Solver object  
    Solver object for the simulation of a two-component system.

Example
-------

    >>> import trottersuzuki as ts  # import the module
    >>> grid = ts.Lattice()  # Define the simulation's geometry
    >>> state_1 = ts.GaussianState(grid, 1.)  # Create first-component system's state
    >>> state_2 = ts.GaussianState(grid, 1.)  # Create second-component system's state
    >>> potential = ts.HarmonicPotential(grid, 1., 1.)  # Create harmonic potential
    >>> hamiltonian = ts.Hamiltonian2Component(grid, potential, potential)  # Create an harmonic oscillator Hamiltonian
    >>> solver = ts.Solver(grid, state_1, state_2, hamiltonian, 1e-2)  # Create the solver

";

%feature("docstring") Solver::get_potential_energy "

Get the potential energy of the system.  

Parameters
----------
* `which` : integer,optional (default: 3)
    Which potential energy to return: total system (default, which=3), first component (which=1), second component (which=2). 

Returns
-------
* `get_potential_energy` : float
    Potential energy of the system.
";

// File: classState.xml


%feature("docstring") State "

";

%feature("docstring") State::write_particle_density "

Write to a file the particle density matrix of the wave function.

Parameters
----------
* `file_name` : string
    Name of the file.  
";

%feature("docstring") State::init_state "

Initialize the wave function of the state using a function.

Parameters
----------
* `function` : python function
    Python function defining the wave function of the state.

Notes
-----
The input arguments of the python function must be (x,y).

Example
-------

    >>> import trottersuzuki as ts  # import the module
    >>> grid = ts.Lattice()  # Define the simulation's geometry
    >>> def wave_function(x,y):  # Define a flat wave function
    >>>     return 1.
    >>> state = ts.State(grid)  # Create the system's state
    >>> state.ini_state(wave_function)  # Initialize the wave function of the state
    
";

%feature("docstring") State::get_phase "

Return a matrix of the wave function's phase.

Returns
-------
* `get_phase` : numpy matrix
    Matrix of the wave function's phase :math:`\phi(x,y) = \log(\psi(x,y))`
";

%feature("docstring") State::write_phase "

Write to a file the phase of the wave function. 

Parameters
----------
* `file_name` : string
    Name of the file.
";

%feature("docstring") State::get_mean_yy "

Return the expected value of the :math:`Y^2` operator.

Returns
-------
* `mean_yy` : float
      Expected value of the :math:`Y^2` operator.
";

%feature("docstring") State::~State "

Destructor.  
";

%feature("docstring") State::get_mean_px "

Return the expected value of the :math:`P_x` operator.  

Returns
-------
* `mean_px` : float
      Expected value of the :math:`P_x` operator.
";

%feature("docstring") State::get_mean_py "

Return the expected value of the :math:`P_y` operator.  

Returns
-------
* `mean_py` : float
      Expected value of the :math:`P_y` operator.
";

%feature("docstring") State::State "

Create a quantum state.  

Parameters
----------
* `grid` : Lattice object  
    Define the geometry of the simulation.

Returns
-------
* `state` : State object
    Quantum state.

Notes
-----
It may be used to copy a quantum state.

    >>> import trottersuzuki as ts  # import the module
    >>> grid = ts.Lattice()  # Define the simulation's geometry
    >>> state = ts.GaussianState(grid, 1.)  # Create the system's state with a gaussian wave function
    >>> state2 = ts.State(state)  # Copy state into state2

Example
-------

    >>> import trottersuzuki as ts  # import the module
    >>> grid = ts.Lattice()  # Define the simulation's geometry
    >>> def wave_function(x,y):  # Define a flat wave function
    >>>     return 1.
    >>> state = ts.State(grid)  # Create the system's state
    >>> state.ini_state(wave_function)  # Initialize the wave function of the state
";

%feature("docstring") State::State "

Create a quantum state.  

Parameters
----------
* `grid` : Lattice object  
    Define the geometry of the simulation.

Returns
-------
* `state` : State object
    Quantum state.

Notes
-----
It may be used to copy a quantum state.

    >>> import trottersuzuki as ts  # import the module
    >>> grid = ts.Lattice()  # Define the simulation's geometry
    >>> state = ts.GaussianState(grid, 1.)  # Create the system's state with a gaussian wave function
    >>> state2 = ts.State(state)  # Copy state into state2

Example
-------

    >>> import trottersuzuki as ts  # import the module
    >>> grid = ts.Lattice()  # Define the simulation's geometry
    >>> def wave_function(x,y):  # Define a flat wave function
    >>>     return 1.
    >>> state = ts.State(grid)  # Create the system's state
    >>> state.ini_state(wave_function)  # Initialize the wave function of the state
";

%feature("docstring") State::imprint "

Multiply the wave function of the state by the function provided.

Parameters
----------
* `function` : python function
    Function to be printed on the state.

Notes
-----
Useful for instance to imprint solitons and vortices on a condensate. 
Generally, it performs a transformation of the state whose wave function 
risult in the multiplication of the prior wave function by the input function.

Example
-------

    >>> import trottersuzuki as ts  # import the module
    >>> grid = ts.Lattice()  # Define the simulation's geometry
    >>> def vortex(x,y):  # Vortex function
    >>>     z = x + 1j*y
    >>>     angle = np.angle(z)
    >>>     return np.exp(1j * angle)
    >>> state = ts.GaussianState(grid, 1.)  # Create the system's state
    >>> state.imprint(vortex)  # Imprint a vortex on the state
     
";

%feature("docstring") State::get_mean_pxpx "

Return the expected value of the :math:`P_x^2` operator.

Returns
-------
* `mean_pxpx` : float
      Expected value of the :math:`P_x^2` operator.  
";

%feature("docstring") State::get_squared_norm "

Return the squared norm of the quantum state.

Returns
-------
* `squared_norm` : float
      Squared norm of the quantum state.
";

%feature("docstring") State::get_mean_xx "

Return the expected value of the :math:`X^2` operator.

Returns
-------
* `mean_xx` : float
      Expected value of the :math:`X^2` operator.   
";

%feature("docstring") State::loadtxt "

Load the wave function from a file.

Parameters
----------
* `file_name` : string
      Name of the file to be written.

Example
-------

    >>> import trottersuzuki as ts  # import the module
    >>> grid = ts.Lattice()  # Define the simulation's geometry
    >>> state = ts.GaussianState(grid, 1.)  # Create the system's state
    >>> state.write_to_file('wave_function.txt')  # Write to a file the wave function
    >>> state2 = ts.State(grid)  # Create a quantum state
    >>> state2.loadtxt('wave_function.txt')  # Load the wave function
";

%feature("docstring") State::get_particle_density "

Return a matrix storing the squared norm of the wave function.

Returns
-------
* `particle_density` : numpy matrix
    Particle density of the state :math:`|\psi(x,y)|^2` 
";

%feature("docstring") State::get_mean_y "

Return the expected value of the :math:`Y` operator.

Returns
-------
* `mean_y` : float
      Expected value of the :math:`Y` operator.   
";

%feature("docstring") State::get_mean_x "

Return the expected value of the :math:`X` operator.  

Returns
-------
* `mean_x` : float
      Expected value of the :math:`X` operator. 
";

%feature("docstring") State::get_mean_pypy "

Return the expected value of the :math:`P_y^2` operator.  

Returns
-------
* `mean_pypy` : float
      Expected value of the :math:`P_y^2` operator. 
";

%feature("docstring") State::write_to_file "

Write to a file the wave function.  

Parameters
----------
* `file_name` : string
      Name of the file to be written. 

Example
-------

    >>> import trottersuzuki as ts  # import the module
    >>> grid = ts.Lattice()  # Define the simulation's geometry
    >>> state = ts.GaussianState(grid, 1.)  # Create the system's state
    >>> state.write_to_file('wave_function.txt')  # Write to a file the wave function
    >>> state2 = ts.State(grid)  # Create a quantum state
    >>> state2.loadtxt('wave_function.txt')  # Load the wave function
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

