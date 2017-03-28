Mathematical Details
====================
What follows is a brief description of the approximation used to calculate the evolution of the wave function. Formulas of the evolution operator are provided.

Evolution operator
------------------
The evolution operator is calculated using the Trotter-Suzuki approximation. Given an Hamiltonian as a sum of hermitian operators, for instance :math:`H = H_1 + H_2 + H_3`, the evolution is approximated as

.. math::

    e^{-i\Delta tH} = e^{-i\frac{\Delta t}{2} H_1} e^{-i\frac{\Delta t}{2} H_2} e^{-i\frac{\Delta t}{2} H_3} e^{-i\frac{\Delta t}{2} H_3} e^{-i\frac{\Delta t}{2} H_2} e^{-i\frac{\Delta t}{2} H_1}.


Since the wavefunction is discretized in the space coordinate representation, to avoid Fourier transformation, the derivatives are approximated using finite differences.

Kinetic operators
-----------------
In cartesian coordinates, the kinetic term is :math:`K = -\frac{1}{2m} \left( \partial_x^2 + \partial_y^2 \right)`. The discrete form of the second derivative is

.. math::

   \partial_x^2 \psi(x) = \frac{\psi(x + \Delta x) - 2 \psi(x) + \psi(x - \Delta x)}{\Delta x^2}

It is useful to express the above equation in a matrix form. The wave function can be vectorized as it is discrete, hence the partial derivative is the matrix

.. math::

    \frac{1}{\Delta x^2} \left(\begin{array}{ccc}
    -2 & 1 & 0 \\
    1 & -2 & 1 \\
    0 & 1 & -2 \end{array} \right)
    =&
    \frac{1}{\Delta x^2} \left(\begin{array}{ccc}
    0 & 1 & 0 \\
    1 & 0 & 0 \\
    0 & 0 & 0 \end{array} \right)
    +
    \frac{1}{\Delta x^2} \left(\begin{array}{ccc}
    0 & 0 & 0 \\
    0 & 0 & 1 \\
    0 & 1 & 0 \end{array} \right)\\
    &-
    \frac{2}{\Delta x^2} \left(\begin{array}{ccc}
    1 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & 1 \end{array} \right).


On the right-hand side of the above equation the last matrix is the identity and in the approximation of Trotter-Suzuki it gives only a global shift of the wave function, which can be ignored. The other two matrices are in the diagonal block form and can be easily exponentiated. Indeed, for the real time evolution:

.. math::

    \exp\left[i\frac{\Delta t}{4m \Delta x^2} \left(\begin{array}{cc}
    0 & 1 \\
    1 & 0 \end{array} \right)\right] =
    \left(\begin{array}{cc}
    \cos\beta & i\sin\beta \\
    i\sin\beta & \cos\beta \end{array} \right).

While, for imaginary time evolution:

.. math::

    \exp\left[\frac{\Delta t}{4m \Delta x^2} \left(\begin{array}{cc}
    0 & 1 \\
    1 & 0 \end{array} \right)\right] =
    \left(\begin{array}{cc}
    \cosh\beta & \sinh\beta \\
    \sinh\beta & \cosh\beta \end{array} \right),

with :math:`\beta = \frac{\Delta t}{4m \Delta x^2}`.

In cylindrical coordinates, the kinetic operator has an additional term, :math:`K = -\frac{1}{2m} \left( \partial_r^2 + \frac{1}{r} \partial_r+ \partial_z^2 \right)`. The first derivative is discretized as

.. math::

    \frac{1}{r}\partial_r \psi(r) = \frac{\psi(r + \Delta r) - \psi(r - \Delta r)}{2 r \Delta r},

and in matrix form

.. math::

    \frac{1}{2 \Delta r} \left(\begin{array}{ccc}
    0 & \frac{1}{r_0} & 0 \\
    -\frac{1}{r_1} & 0 & \frac{1}{r_1} \\
    0 & -\frac{1}{r_2} & 0 \end{array} \right)
    =
    \frac{1}{2 \Delta r} \left(\begin{array}{ccc}
    0 & \frac{1}{r_0} & 0 \\
    -\frac{1}{r_1} & 0 & 0 \\
    0 & 0 & 0 \end{array} \right)
    +
    \frac{1}{2 \Delta r} \left(\begin{array}{ccc}
    0 & 0 & 0 \\
    0 & 0 & \frac{1}{r_1} \\
    0 & -\frac{1}{r_2} & 0 \end{array} \right).

The exponentiation of a block is, for real-time evolution:

.. math::

    \exp\left[i\frac{\Delta t}{8m \Delta r} \left(\begin{array}{cc}
    0 & \frac{1}{r_1} \\
    -\frac{1}{r_2} & 0 \end{array} \right)\right] =
    \left(\begin{array}{cc}
    \cosh\beta & i\alpha\sinh\beta \\
    -i\frac{1}{\alpha}\sinh\beta & \cosh\beta \end{array} \right).

While, for imaginary time evolution:

.. math::

    \exp\left[\frac{\Delta t}{8m \Delta r} \left(\begin{array}{cc}
    0 & \frac{1}{r_1} \\
    -\frac{1}{r_2} & 0 \end{array} \right)\right] =
    \left(\begin{array}{cc}
    \cos\beta & \alpha\sin\beta \\
    -\frac{1}{\alpha}\sin\beta & \cos\beta \end{array} \right).

with :math:`\beta = \frac{\Delta t}{8m \Delta r \sqrt{r_1r_2}}`, :math:`\alpha = \sqrt{\frac{r_2}{r_1}}` and :math:`r_1, r_2 > 0`. However, the block matrix that contains :math:`1/r_0` has a different exponentiation, since :math:`r_0 < 0 `.

In particular :math:`r_0 = - r_1` and for the real-time evolution, the block is of the form

.. math::

    \exp\left[i\frac{\Delta t}{8m r_1\Delta r} \left(\begin{array}{cc}
    0 & -1 \\
    -1 & 0 \end{array} \right)\right] =
    \left(\begin{array}{cc}
    \cos\beta & i\sin\beta \\
    i\sin\beta & \cos\beta \end{array} \right)

for imaginary-time evolution

.. math::

    \exp\left[\frac{\Delta t}{8m r_1\Delta r} \left(\begin{array}{cc}
    0 & -1 \\
    -1 & 0 \end{array} \right)\right] =
    \left(\begin{array}{cc}
    \cosh\beta & \sinh\beta \\
    \sinh\beta & \cosh\beta \end{array} \right)

with :math:`\beta = -\frac{\Delta t}{8m r_1 \Delta r}`.

External potential
------------------
An external potential dependent on the coordinate space is trivial to calculate. For the discretization that we use, such external potential is approximated by a diagonal matrix. For real time evolution

.. math::

    \exp[-i\Delta t V] &=
    \exp\left[-i\Delta t \left(\begin{array}{ccc}
    V(x_0,y_0) & 0 & 0 \\
    0 & V(x_1,y_0) & 0 \\
    0 & 0 & V(x_2,y_0) \end{array} \right)\right] \\
    &= \left(\begin{array}{ccc}
    e^{-i\Delta t V(x_0,y_0)} & 0 & 0 \\
    0 & e^{-i\Delta t V(x_1,y_0)} & 0 \\
    0 & 0 & e^{-i\Delta t V(x_2,y_0)} \end{array} \right)


and for imaginary time evolution

.. math::

    \exp[-\Delta t V] &=
    \exp\left[-\Delta t \left(\begin{array}{ccc}
    V(x_0,y_0) & 0 & 0 \\
    0 & V(x_1,y_0) & 0 \\
    0 & 0 & V(x_2,y_0) \end{array} \right)\right] \\
    &= \left(\begin{array}{ccc}
    e^{-\Delta t V(x_0,y_0)} & 0 & 0 \\
    0 & e^{-\Delta t V(x_1,y_0)} & 0 \\
    0 & 0 & e^{-\Delta t V(x_2,y_0)} \end{array} \right)


Self interaction term
---------------------
The self interaction term of the wave function, :math:`g|\psi(x,y)|^2`, depends on the coordinate space, hence its discrete form is a diagonal matrix, as in the case of the external potential. In addition, the Lee-Huang-Yang term, :math:`g_{LHY}|\psi(x,y)|^3`, is implemented in the same way.

Angular momentum
----------------
For cartesian coordinates the Hamiltonian containing the angular momentum operator is

.. math::

    -i \omega\left( x\partial_y - y\partial_x \right).

For the trotter-suzuki approximation, the exponentiation is done separately for the two terms:

- First term, real-time evolution, :math:`\beta = \frac{\Delta t \omega x}{2\Delta y}`

.. math::

    \exp[-\Delta t \omega x\partial_y] =
    \exp\left[-\beta
    \left(\begin{array}{cc}
    0 & 1 \\
    -1 & 0 \end{array} \right)\right] =
    \left(\begin{array}{cc}
    \cos\beta & -\sin\beta \\
    \sin\beta & \cos\beta \end{array} \right)


- First term, imaginary-time evolution, :math:`\beta = \frac{\Delta t \omega x}{2\Delta y}`

.. math::

    \exp[i\Delta t \omega x\partial_y] =
    \exp\left[-\beta
    \left(\begin{array}{cc}
    0 & 1 \\
    -1 & 0 \end{array} \right)\right] =
    \left(\begin{array}{cc}
    \cosh\beta & i\sinh\beta \\
    -i\sinh\beta & \cosh\beta \end{array} \right)


- Second term, real-time evolution, :math:`\beta = \frac{\Delta t \omega y}{2\Delta x}`

.. math::

    \exp[\Delta t \omega y\partial_x] =
    \exp\left[-\beta
    \left(\begin{array}{cc}
    0 & 1 \\
    -1 & 0 \end{array} \right)\right] =
    \left(\begin{array}{cc}
    \cos\beta & \sin\beta \\
    -\sin\beta & \cos\beta \end{array} \right)


- Second term, imaginary-time evolution, :math:`\beta = \frac{\Delta t \omega y}{2\Delta x}`

.. math::

    \exp[-i\Delta t \omega y\partial_x] =
    \exp\left[-\beta
    \left(\begin{array}{cc}
    0 & 1 \\
    -1 & 0 \end{array} \right)\right] =
    \left(\begin{array}{cc}
    \cosh\beta & -i\sinh\beta \\
    i\sinh\beta & \cosh\beta \end{array} \right)



