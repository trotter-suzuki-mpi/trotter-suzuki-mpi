import numpy as np
from .trottersuzuki import solver, H, K, Norm2


def evolve(p_real, p_imag, particle_mass, external_potential, delta_x, delta_y,
           delta_t, iterations, coupling_const=0.0, kernel_type=0,
           periods=None, imag_time=False):
    """Function for evolving a quantum state.

    :param p_real: The real part of the initial quantum state.
    :type p_real: 2D numpy.array of float64.
    :param p_imag: The imaginary part of the initial quantum state.
    :type p_imag: 2D numpy.array of float64.
    :param particle_mass: Mass of the particle.
    :type particle_mass: float.
    :param external_potential: External potential.
    :type external_potential: 2D numpy.array of float64.
    :param delta_x: Relative grid distance in the x direction.
    :type delta_x: float.
    :param delta_y: Relative grid distance in the y direction.
    :type delta_y: float.
    :param delta_t: Time step.
    :type delta_t: float.
    :param iterations: Number of iterations in the simulation.
    :type iterations: int.
    :param coupling_const: Optional coupling constant between parameters.
    :type coupling_const: float.
    :param kernel_type: Optional parameter to specify which kernel to use:

                           * 0: CPU kernel (default)
                           * 1: CPU SSE kernel (if compiled with it)
    :type kernel_type: int.
    :param periods: Optional parameter to specify periodicity in x and y
                    directions.
    :type periods: [int, int]
    :param imag_time: Optional parameter to request imaginary time evolution.
                      Default: False.
    :type imag_time: bool.
    """
    if external_potential is None:
        external_potential = np.zeros(p_real.shape)
    if periods is None:
        periods = [0, 0]
    solver(p_real, p_imag, particle_mass, coupling_const, external_potential,
           delta_x, delta_y, delta_t, iterations, kernel_type, periods,
           imag_time)


def calculate_total_energy(p_real, p_imag, particle_mass, external_potential,
                           delta_x, delta_y, coupling_const=0.0):
    """Function for calculating the expectation value of the Hamiltonian.

    :param p_real: The real part of the quantum state.
    :type p_real: 2D numpy.array of float64.
    :param p_imag: The imaginary part of the quantum state.
    :type p_imag: 2D numpy.array of float64.
    :param particle_mass: Mass of the particle.
    :type particle_mass: float.
    :param external_potential: External potential.
    :type external_potential: 2D numpy.array of float64.
    :param delta_x: Relative grid distance in the x direction.
    :type delta_x: float.
    :param delta_y: Relative grid distance in the y direction.
    :type delta_y: float.
    :param coupling_const: Optional coupling constant between parameters.
    :type coupling_const: float.
    """
    if external_potential is None:
        external_potential = np.zeros(p_real.shape)
    return H(p_real, p_imag, particle_mass, coupling_const, external_potential,
             delta_x, delta_y)


def calculate_kinetic_energy(p_real, p_imag, particle_mass, delta_x, delta_y):
    """Function for calculating the expectation value of the kinetic energy.

    :param p_real: The real part of the quantum state.
    :type p_real: 2D numpy.array of float64.
    :param p_imag: The imaginary part of the quantum state.
    :type p_imag: 2D numpy.array of float64.
    :param particle_mass: Mass of the particle.
    :type particle_mass: float.
    :param delta_x: Relative grid distance in the x direction.
    :type delta_x: float.
    :param delta_y: Relative grid distance in the y direction.
    :type delta_y: float.
    """
    return K(p_real, p_imag, particle_mass, delta_x, delta_y)


def calculate_norm2(p_real, p_imag, delta_x, delta_y):
    """Function for calculating the expectation value of the kinetic energy.

    :param p_real: The real part of the quantum state.
    :type p_real: 2D numpy.array of float64.
    :param p_imag: The imaginary part of the quantum state.
    :type p_imag: 2D numpy.array of float64.
    :param delta_x: Relative grid distance in the x direction.
    :type delta_x: float.
    :param delta_y: Relative grid distance in the y direction.
    :type delta_y: float.
    """
    return Norm2(p_real, p_imag, delta_x, delta_y)
