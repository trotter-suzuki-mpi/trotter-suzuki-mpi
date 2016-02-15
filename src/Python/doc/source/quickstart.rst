
Quick Start Guide
=================

Simulation Set up
-----------------

Start by importing the module:

.. code:: python

    import trottersuzuki as ts

To set up the simulation of a quantum system, we need only a few lines of code.
First of all, we create the lattice over which the physical system is
defined. All information about the discretized space is collected in a
single object. Say we want a squared lattice of 300x300 nodes, with a
physical area of 20x20, then we have to specify these in the constructor of the ``Lattice`` class:

.. code:: python

    grid = ts.Lattice(300, 20.)

The object ``grid`` defines the geometry of the system and it
will be used throughout the simulations. Note that the origin of the lattice is at its centre.

The physics of the problem is described by the Hamiltonian. A single
object is going to store all the information regarding the Hamiltonian.
The module is able to deal with two physical models: Gross-Pitaevskii
equation of a single or two-component wave function, namely (in units
:math:`\hbar=1`):

.. math::

   i \frac{\partial}{\partial t} \psi(t) = H \psi(t)


being

.. math::

   H = \frac{1}{2m}(P_x^2 + P_y^2) + V(x,y) + g|\psi(x,y)|^2 + \omega L_z

and :math:`\psi(t) = \psi_t(x,y)` for the single component wave
function, or

.. math::

   H = \begin{bmatrix} H_1 &  \frac{\Omega}{2} \\ \frac{\Omega}{2} & H_2 \end{bmatrix} 

where

.. math::

   H_1 = \frac{1}{2m_1}(P_x^2 + P_y^2) + V_1(x,y) + g_1|\psi(x,y)_1|^2 + g_{12}|\psi(x,y)_2|^2 + \omega L_z  

   H_2 = \frac{1}{2m_2}(P_x^2 + P_y^2) + V_2(x,y) + g_2|\psi(x,y)_2|^2 + g_{12}|\psi(x,y)_1|^2 + \omega L_z  


and
:math:`\psi(t) = \begin{bmatrix} \psi_1(t) \\ \psi_2(t) \end{bmatrix}`,
for the two component wave function.

First we define the object for the external potential :math:`V(x,y)`. A
general external potential function can be defined by a Python
function, for instance, the harmonic potential can be defined as follows:

.. code:: python

    def harmonic_potential(x,y):
        return 0.5 * (x**2 + y**2)

Now we create the external potential object using the ``Potential``
class and then we initialize it with the function above:

.. code:: python

    potential = ts.Potential(grid)  # Create the potential object
    potential.init_potential(harmonic_potential)  # Initialize it using a python function

Note that the module provides a quick way to define the harmonic
potential, as it is fequently used:

.. code:: python

    omegax = omegay = 1.
    harmonicpotential = ts.HarmonicPotential(grid, omegax, omegay)

We are ready to create the ``Hamiltonian`` object. For the sake of simplicity, let us create the Hamiltonian of the harmonic oscillator:

.. code:: python

    particle_mass = 1. # Mass of the particle
    hamiltonian = ts.Hamiltonian(grid, potential, particle_mass)  # Create the Hamiltonian object

The quantum state is created by the ``State`` class; it resembles the way the potential is defined. Here we create the ground state of the
harmonic oscillator:

.. code:: python

    import numpy as np  # Import the module numpy for the exponential and sqrt functions

    def state_wave_function(x,y):  # Wave function
        return np.exp(-0.5*(x**2 + y**2)) / np.sqrt(np.pi)

    state = ts.State(grid)  # Create the quantum state
    state.init_state(state_wave_function)  # Initialize the state

The module provides several predefined quantum states as well. In this
case, we could have used the ``GaussianState`` class:

.. code:: python

    omega = 1.
    gaussianstate = ts.GaussianState(grid, omega)  # Create a quantum state whose wave function is Gaussian-like

We are left with the creation of the last object: the ``Solver`` class gathers all the objects we defined so far and it is used to perform the evolution and analyze the expectation values:

.. code:: python

    delta_t = 1e-3  # Physical time of a single iteration
    solver = ts.Solver(grid, state, hamiltonian, delta_t)  # Creating the solver object

Finally we can perform both real-time and imaginary-time evolution using
the method ``evolve``:

.. code:: python

    iterations = 100  # Number of iterations to be performed
    solver.evolve(iterations, True)  # Perform imaginary-time evolution
    solver.evolve(iterations)  # Perform real-time evolution

Analysis
--------

The classes we have seen so far implement several members useful to
analyze the system (see the function reference section for a complete
list).

Expectation values
~~~~~~~~~~~~~~~~~~

The solver class provides members for the energy calculations. For
instance, the total energy can be calculated using the
``get_total_energy`` member. We expect it to be :math:`1`
(:math:`\hbar =1`), and indeed we get the right result up to a small
error which depends on the lattice approximation:

.. code:: python

    tot_energy = solver.get_total_energy()
    print(tot_energy)


.. parsed-literal::

    1.00146456951


The expected values of the :math:`X`, :math:`Y`, :math:`P_x`,
:math:`P_y` operators are calculated using the members in the ``State``
class

.. code:: python

    mean_x = state.get_mean_x()  # Get the expected value of X operator
    print(mean_x)


.. parsed-literal::

    1.39431975344e-14


Norm of the state
~~~~~~~~~~~~~~~~~

The squared norm of the state can be calculated by means of both
``State`` and ``Solver`` classes

.. code:: python

    snorm = state.get_squared_norm()
    print(snorm)


.. parsed-literal::

    1.0


Particle density and Phase
~~~~~~~~~~~~~~~~~~~~~~~~~~

Very often one is interested in the phase and particle density of the
state. Two members of ``State`` class provide these features

.. code:: python

    density = state.get_particle_density()  # Return a numpy matrix of the particle density
    phase = state.get_phase()  # Return a numpy matrix of the phase

Imprinting
~~~~~~~~~~

The member ``imprint``, in the ``State`` class, applies the following transformation to the state:

.. math::

   \psi(x,y) \rightarrow \psi'(x,y) = f(x,y)  \psi(x,y)

being :math:`f(x,y)` a general complex-valued function. This comes in
handy when we want to imprint, for instance, vortices or solitons:

.. code:: python

    def vortex(x, y):  # Function defining a vortex
        z = x + 1j*y
        angle = np.angle(z)
        return np.exp(1j * angle)
    
    state.imprint(vortex)  # Imprint the vortex on the state

File Input and Output
~~~~~~~~~~~~~~~~~~~~~

``write_to_files`` and ``loadtxt`` members, in ``State`` class, provide
a simple way to handle file I/O. The former writes the wave function
arranged as a complex matrix, in a plain text; the latter loads the wave
function from a file to the ``state`` object. The following code
provides an example:

.. code:: python

    state.write_to_file("file_name")  # Write the wave function to a file
    state2 = ts.State(grid)  # Create a new state
    state2.loadtxt("file_name")  # Load the wave function from the file

For a complete list of methods see the function reference.
