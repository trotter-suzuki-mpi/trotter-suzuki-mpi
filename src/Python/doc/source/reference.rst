******************
Function Reference
******************

Lattice Class
=============
.. py:class:: Lattice
   :module: trottersuzuki

   This class defines the lattice structure over which the state and potential
   matrices are defined.

   **Constructors**
   
   .. py:method:: Lattice(dim=100, length=20.0, periodic_x_axis=False, periodic_y_axis=False)

      Construct the Lattice.

      **Parameters**
      
      * `dim` : integer,optional (default: 100)
          Linear dimension of the squared lattice.  
      * `length` : float,optional (default: 20.)
          Physical length of the lattice's side.   
      * `periodic_x_axis` : bool,optional (default: False)
          Boundary condition along the x axis (false=closed, true=periodic).  
      * `periodic_y_axis` : bool,optional (default: False) 
          Boundary condition along the y axis (false=closed, true=periodic).

      **Returns**

      * `Lattice` : Lattice object 
          Define the geometry of the simulation.

      **Notes**

      The lattice created is squared.

      **Example**


          >>> import trottersuzuki as ts  # import the module
          >>> # Generate a 200x200 Lattice with physical dimensions of 30x30
          >>> # and closed boundary conditions.
          >>> grid = ts.Lattice(200, 30.)

   **Members**

   .. py:method:: get_x_axis()
      :module: trottersuzuki

      Get the x-axis of the lattice.
        
      **Returns**

      * `x_axis` : numpy array
          X-axis of the lattice

   .. py:method:: get_y_axis()
      :module: trottersuzuki
      
      Get the y-axis of the lattice.
        
      **Returns**

      * `y_axis` : numpy array
          Y-axis of the lattice

   **Attributes**
   
   .. py:attribute:: length_x
      :module: trottersuzuki
      
      Physical length of the lattice along the X-axis.
      
   .. py:attribute:: length_y
      :module: trottersuzuki
      
      Physical length of the lattice along the Y-axis.
      
   .. py:attribute:: dim_x
      :module: trottersuzuki
      
      Number of dots of the lattice along the X-axis.

   .. py:attribute:: dim_y
      :module: trottersuzuki
      
      Number of dots of the lattice along the Y-axis.

   .. py:attribute:: delta_x
      :module: trottersuzuki
      
      Resolution of the lattice along the X-axis: ratio between `lenth_x` and `dim_x`.

   .. py:attribute:: delta_y
      :module: trottersuzuki
      
      Resolution of the lattice along the y-axis: ratio between `lenth_y` and `dim_y`.

State Classes
=============
.. py:class:: State
   :module: trottersuzuki

   This class defines the quantum state.

   **Constructors**

   .. py:method:: State(grid)
      
      Create a quantum state.  

      **Parameters**

      * `grid` : Lattice object  
          Define the geometry of the simulation.

      **Returns**

      * `state` : State object
          Quantum state.

      **Example**

          >>> import trottersuzuki as ts  # import the module
          >>> grid = ts.Lattice()  # Define the simulation's geometry
          >>> def wave_function(x,y):  # Define a flat wave function
          >>>     return 1.
          >>> state = ts.State(grid)  # Create the system's state
          >>> state.ini_state(wave_function)  # Initialize the wave function of the state

   .. py:method:: State(state)
   
      Copy a quantum state.
      
      **Parameters**

      * `state` : State object  
          Quantum state to be copied
      
      **Returns**

      * `state` : State object
          Quantum state.

      **Example**

          >>> import trottersuzuki as ts  # import the module
          >>> grid = ts.Lattice()  # Define the simulation's geometry
          >>> state = ts.GaussianState(grid, 1.)  # Create the system's state with a gaussian wave function
          >>> state2 = ts.State(state)  # Copy state into state2
   
   **Members**
   
   .. py:method:: State.init_state(state_function):
      :module: trottersuzuki
      
      Initialize the wave function of the state using a function.

      **Parameters**

      * `state_function` : python function
         Python function defining the wave function of the state :math:`\psi`.

      **Notes**

      The input arguments of the python function must be (x,y).

      **Example**

         >>> import trottersuzuki as ts  # import the module
         >>> grid = ts.Lattice()  # Define the simulation's geometry
         >>> def wave_function(x,y):  # Define a flat wave function
         >>>     return 1.
         >>> state = ts.State(grid)  # Create the system's state
         >>> state.ini_state(wave_function)  # Initialize the wave function of the state
   
   .. py:method:: State.imprint(function)
      :module: trottersuzuki
      
        Multiply the wave function of the state by the function provided.
        
        **Parameters**

        * `function` : python function
            Function to be printed on the state.
        
        **Notes**

        Useful, for instance, to imprint solitons and vortices on a condensate. 
        Generally, it performs a transformation of the state whose wave function becomes:
        
        .. math:: \psi(x,y)' = f(x,y) \psi(x,y)
        
        being :math:`f(x,y)` the input function and :math:`\psi(x,y)` the initial wave function.        
        
        **Example**
        
            >>> import trottersuzuki as ts  # import the module
            >>> grid = ts.Lattice()  # Define the simulation's geometry
            >>> def vortex(x,y):  # Vortex function
            >>>     z = x + 1j*y
            >>>     angle = np.angle(z)
            >>>     return np.exp(1j * angle)
            >>> state = ts.GaussianState(grid, 1.)  # Create the system's state
            >>> state.imprint(vortex)  # Imprint a vortex on the state
         
   .. py:method:: State.get_mean_px()
      :module: trottersuzuki

      Return the expected value of the :math:`P_x` operator.  

      **Returns**

      * `mean_px` : float
            Expected value of the :math:`P_x` operator.

   .. py:method:: State.get_mean_pxpx()
      :module: trottersuzuki

      Return the expected value of the :math:`P_x^2` operator.

      **Returns**

      * `mean_pxpx` : float
            Expected value of the :math:`P_x^2` operator.



   .. py:method:: State.get_mean_py()
      :module: trottersuzuki

      Return the expected value of the :math:`P_y` operator.  

      **Returns**

      * `mean_py` : float
            Expected value of the :math:`P_y` operator.

   .. py:method:: State.get_mean_pypy()
      :module: trottersuzuki

      Return the expected value of the :math:`P_y^2` operator.  

      **Returns**

      * `mean_pypy` : float
            Expected value of the :math:`P_y^2` operator. 

   .. py:method:: State.get_mean_x()
      :module: trottersuzuki

      Return the expected value of the :math:`X` operator.  

      **Returns**

      * `mean_x` : float
            Expected value of the :math:`X` operator. 

   .. py:method:: State.get_mean_xx()
      :module: trottersuzuki

      Return the expected value of the :math:`X^2` operator.

      **Returns**

      * `mean_xx` : float
            Expected value of the :math:`X^2` operator.   


   .. py:method:: State.get_mean_y()
      :module: trottersuzuki

      Return the expected value of the :math:`Y` operator.

      **Returns**
      
      * `mean_y` : float
            Expected value of the :math:`Y` operator.

   .. py:method:: State.get_mean_yy()
      :module: trottersuzuki

      Return the expected value of the :math:`Y^2` operator.

      **Returns**
      
      * `mean_yy` : float
            Expected value of the :math:`Y^2` operator.

   .. py:method:: State.get_particle_density()
      :module: trottersuzuki

      Return a matrix storing the squared norm of the wave function.

      **Returns**
      
      * `particle_density` : numpy matrix
          Particle density of the state :math:`|\psi(x,y)|^2` 


   .. py:method:: State.get_phase()
      :module: trottersuzuki

      Return a matrix of the wave function's phase.

      **Returns**

      * `get_phase` : numpy matrix
          Matrix of the wave function's phase :math:`\phi(x,y) = \log(\psi(x,y))`


   .. py:method:: State.get_squared_norm()
      :module: trottersuzuki

      Return the squared norm of the quantum state.

      **Returns**

      * `squared_norm` : float
            Squared norm of the quantum state.

   .. py:method:: State.loadtxt(file_name)
      :module: trottersuzuki
      
      Load the wave function from a file.

      **Parameters**

      * `file_name` : string
            Name of the file to be written.

      **Example**

          >>> import trottersuzuki as ts  # import the module
          >>> grid = ts.Lattice()  # Define the simulation's geometry
          >>> state = ts.GaussianState(grid, 1.)  # Create the system's state
          >>> state.write_to_file('wave_function.txt')  # Write to a file the wave function
          >>> state2 = ts.State(grid)  # Create a quantum state
          >>> state2.loadtxt('wave_function.txt')  # Load the wave function

   .. py:method:: State.write_particle_density(file_name)
      :module: trottersuzuki

      Write to a file the particle density matrix of the wave function.

      **Parameters**
      
      * `file_name` : string
          Name of the file. 

   .. py:method:: State.write_phase(file_name)
      :module: trottersuzuki

      Write to a file the wave function.  

      **Parameters**
      
      * `file_name` : string
            Name of the file to be written. 


   .. py:method:: State.write_to_file(file_name)
      :module: trottersuzuki

      Write to a file the wave function.  

      **Parameters**

      * `file_name` : string
            Name of the file to be written. 

      **Example**

          >>> import trottersuzuki as ts  # import the module
          >>> grid = ts.Lattice()  # Define the simulation's geometry
          >>> state = ts.GaussianState(grid, 1.)  # Create the system's state
          >>> state.write_to_file('wave_function.txt')  # Write to a file the wave function
          >>> state2 = ts.State(grid)  # Create a quantum state
          >>> state2.loadtxt('wave_function.txt')  # Load the wave function


.. py:class:: ExponentialState
   :module: trottersuzuki

   This class defines a quantum state with exponential like wave function.

   This class is a child of State class.

   **Constructors**

   .. py:method:: ExponentialState(grid, n_x=1, n_y=1, norm=1, phase=0)
      :module: trottersuzuki
      
      Construct the quantum state with exponential like wave function.

      **Parameters**

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

      **Returns**

      * `ExponentialState` : State object. 
          Quantum state with exponential like wave function. The wave function is give by:\n
          
          .. math:: \psi(x,y) = \sqrt{N}/L \mathrm{e}^{i 2 \pi (n_x x + n_y y) / L} \mathrm{e}^{i \phi}
          
          being :math:`N` the norm of the state, :math:`L` 
          the length of the lattice edge, :math:`n_x` and :math:`n_y` the quantum numbers 
          and :math:`\phi` the relative phase.

      **Notes**

      The geometry of the simulation has to have periodic boundary condition 
      to use Exponential state as initial state of a real time evolution. 
      Indeed, the wave function is not null at the edges of the space.

      **Example**

          >>> import trottersuzuki as ts  # import the module
          >>> grid = ts.Lattice(300, 30., True, True)  # Define the simulation's geometry
          >>> state = ts.ExponentialState(grid, 2, 1)  # Create the system's state

   **Member**

   .. py:method:: ExponentialState.imprint(function)
      :module: trottersuzuki
      
        Multiply the wave function of the state by the function provided.
        
        **Parameters**

        * `function` : python function
            Function to be printed on the state.
        
        **Notes**

        Useful, for instance, to imprint solitons and vortices on a condensate. 
        Generally, it performs a transformation of the state whose wave function becomes:
        
        .. math:: \psi(x,y)' = f(x,y) \psi(x,y)
        
        being :math:`f(x,y)` the input function and :math:`\psi(x,y)` the initial wave function.        
        
        **Example**
        
            >>> import trottersuzuki as ts  # import the module
            >>> grid = ts.Lattice()  # Define the simulation's geometry
            >>> def vortex(x,y):  # Vortex function
            >>>     z = x + 1j*y
            >>>     angle = np.angle(z)
            >>>     return np.exp(1j * angle)
            >>> state = ts.GaussianState(grid, 1.)  # Create the system's state
            >>> state.imprint(vortex)  # Imprint a vortex on the state

   .. py:method:: ExponentialState.get_mean_px()
      :module: trottersuzuki

      Return the expected value of the :math:`P_x` operator.  

      **Returns**

      * `mean_px` : float
            Expected value of the :math:`P_x` operator.

   .. py:method:: ExponentialState.get_mean_pxpx()
      :module: trottersuzuki

      Return the expected value of the :math:`P_x^2` operator.

      **Returns**

      * `mean_pxpx` : float
            Expected value of the :math:`P_x^2` operator.



   .. py:method:: ExponentialState.get_mean_py()
      :module: trottersuzuki

      Return the expected value of the :math:`P_y` operator.  

      **Returns**

      * `mean_py` : float
            Expected value of the :math:`P_y` operator.

   .. py:method:: ExponentialState.get_mean_pypy()
      :module: trottersuzuki

      Return the expected value of the :math:`P_y^2` operator.  

      **Returns**

      * `mean_pypy` : float
            Expected value of the :math:`P_y^2` operator. 

   .. py:method:: ExponentialState.get_mean_x()
      :module: trottersuzuki

      Return the expected value of the :math:`X` operator.  

      **Returns**

      * `mean_x` : float
            Expected value of the :math:`X` operator. 

   .. py:method:: ExponentialState.get_mean_xx()
      :module: trottersuzuki

      Return the expected value of the :math:`X^2` operator.

      **Returns**

      * `mean_xx` : float
            Expected value of the :math:`X^2` operator.   


   .. py:method:: ExponentialState.get_mean_y()
      :module: trottersuzuki

      Return the expected value of the :math:`Y` operator.

      **Returns**
      
      * `mean_y` : float
            Expected value of the :math:`Y` operator.

   .. py:method:: ExponentialState.get_mean_yy()
      :module: trottersuzuki

      Return the expected value of the :math:`Y^2` operator.

      **Returns**
      
      * `mean_yy` : float
            Expected value of the :math:`Y^2` operator.

   .. py:method:: ExponentialState.get_particle_density()
      :module: trottersuzuki

      Return a matrix storing the squared norm of the wave function.

      **Returns**
      
      * `particle_density` : numpy matrix
          Particle density of the state :math:`|\psi(x,y)|^2` 


   .. py:method:: ExponentialState.get_phase()
      :module: trottersuzuki

      Return a matrix of the wave function's phase.

      **Returns**

      * `get_phase` : numpy matrix
          Matrix of the wave function's phase :math:`\phi(x,y) = \log(\psi(x,y))`


   .. py:method:: ExponentialState.get_squared_norm()
      :module: trottersuzuki

      Return the squared norm of the quantum state.

      **Returns**

      * `squared_norm` : float
            Squared norm of the quantum state.

   .. py:method:: ExponentialState.loadtxt(file_name)
      :module: trottersuzuki
      
      Load the wave function from a file.

      **Parameters**

      * `file_name` : string
            Name of the file to be written.

      **Example**

          >>> import trottersuzuki as ts  # import the module
          >>> grid = ts.Lattice()  # Define the simulation's geometry
          >>> state = ts.GaussianState(grid, 1.)  # Create the system's state
          >>> state.write_to_file('wave_function.txt')  # Write to a file the wave function
          >>> state2 = ts.State(grid)  # Create a quantum state
          >>> state2.loadtxt('wave_function.txt')  # Load the wave function

   .. py:method:: ExponentialState.write_particle_density(file_name)
      :module: trottersuzuki

      Write to a file the particle density matrix of the wave function.

      **Parameters**
      
      * `file_name` : string
          Name of the file. 

   .. py:method:: ExponentialState.write_phase(file_name)
      :module: trottersuzuki

      Write to a file the wave function.  

      **Parameters**
      
      * `file_name` : string
            Name of the file to be written. 


   .. py:method:: ExponentialState.write_to_file(file_name)
      :module: trottersuzuki

      Write to a file the wave function.  

      **Parameters**

      * `file_name` : string
            Name of the file to be written. 

      **Example**

          >>> import trottersuzuki as ts  # import the module
          >>> grid = ts.Lattice()  # Define the simulation's geometry
          >>> state = ts.GaussianState(grid, 1.)  # Create the system's state
          >>> state.write_to_file('wave_function.txt')  # Write to a file the wave function
          >>> state2 = ts.State(grid)  # Create a quantum state
          >>> state2.loadtxt('wave_function.txt')  # Load the wave function



.. py:class:: GaussianState
   :module: trottersuzuki

   This class defines a quantum state with gaussian like wave function.

   This class is a child of State class.

   **Constructors**
   
   .. py:method:: GaussianState(grid, omega_x, omega_y=omega_x, mean_x=0, mean_y=0, norm=1, phase=0)

      Construct the quantum state with gaussian like wave function.  

      **Parameters**

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

      **Returns**

      * `GaussianState` : State object. 
          Quantum state with gaussian like wave function. The wave function is given by:\n
          
          .. math:: \psi(x,y) = (N/\pi)^{1/2} (\omega_x \omega_y)^{1/4} \mathrm{e}^{-(\omega_x(x-\mu_x)^2 + \omega_y(y-\mu_y)^2)/2} \mathrm{e}^{i \phi}
          
          being :math:`N` the norm of the state, :math:`\omega_x` and :math:`\omega_y` 
          the inverse of the variances, :math:`\mu_x` and :math:`\mu_y` the coordinates of the
          function's peak and :math:`\phi` the relative phase.

      **Notes**

      The physical dimensions of the Lattice have to be enough to ensure that 
      the wave function is almost zero at the edges.

      **Example**

          >>> import trottersuzuki as ts  # import the module
          >>> grid = ts.Lattice(300, 30.)  # Define the simulation's geometry
          >>> state = ts.GaussianState(grid, 2.)  # Create the system's state

   **Members**

   .. py:method:: GaussianState.imprint(function)
      :module: trottersuzuki
      
        Multiply the wave function of the state by the function provided.
        
        **Parameters**

        * `function` : python function
            Function to be printed on the state.
        
        **Notes**

        Useful, for instance, to imprint solitons and vortices on a condensate. 
        Generally, it performs a transformation of the state whose wave function becomes:
        
        .. math:: \psi(x,y)' = f(x,y) \psi(x,y)
        
        being :math:`f(x,y)` the input function and :math:`\psi(x,y)` the initial wave function.        
        
        **Example**
        
            >>> import trottersuzuki as ts  # import the module
            >>> grid = ts.Lattice()  # Define the simulation's geometry
            >>> def vortex(x,y):  # Vortex function
            >>>     z = x + 1j*y
            >>>     angle = np.angle(z)
            >>>     return np.exp(1j * angle)
            >>> state = ts.GaussianState(grid, 1.)  # Create the system's state
            >>> state.imprint(vortex)  # Imprint a vortex on the state

   .. py:method:: GaussianState.get_mean_px()
      :module: trottersuzuki

      Return the expected value of the :math:`P_x` operator.  

      **Returns**

      * `mean_px` : float
            Expected value of the :math:`P_x` operator.

   .. py:method:: GaussianState.get_mean_pxpx()
      :module: trottersuzuki

      Return the expected value of the :math:`P_x^2` operator.

      **Returns**

      * `mean_pxpx` : float
            Expected value of the :math:`P_x^2` operator.



   .. py:method:: GaussianState.get_mean_py()
      :module: trottersuzuki

      Return the expected value of the :math:`P_y` operator.  

      **Returns**

      * `mean_py` : float
            Expected value of the :math:`P_y` operator.

   .. py:method:: GaussianState.get_mean_pypy()
      :module: trottersuzuki

      Return the expected value of the :math:`P_y^2` operator.  

      **Returns**

      * `mean_pypy` : float
            Expected value of the :math:`P_y^2` operator. 

   .. py:method:: GaussianState.get_mean_x()
      :module: trottersuzuki

      Return the expected value of the :math:`X` operator.  

      **Returns**

      * `mean_x` : float
            Expected value of the :math:`X` operator. 

   .. py:method:: GaussianState.get_mean_xx()
      :module: trottersuzuki

      Return the expected value of the :math:`X^2` operator.

      **Returns**

      * `mean_xx` : float
            Expected value of the :math:`X^2` operator.   


   .. py:method:: GaussianState.get_mean_y()
      :module: trottersuzuki

      Return the expected value of the :math:`Y` operator.

      **Returns**
      
      * `mean_y` : float
            Expected value of the :math:`Y` operator.

   .. py:method:: GaussianState.get_mean_yy()
      :module: trottersuzuki

      Return the expected value of the :math:`Y^2` operator.

      **Returns**
      
      * `mean_yy` : float
            Expected value of the :math:`Y^2` operator.

   .. py:method:: GaussianState.get_particle_density()
      :module: trottersuzuki

      Return a matrix storing the squared norm of the wave function.

      **Returns**
      
      * `particle_density` : numpy matrix
          Particle density of the state :math:`|\psi(x,y)|^2` 


   .. py:method:: GaussianState.get_phase()
      :module: trottersuzuki

      Return a matrix of the wave function's phase.

      **Returns**

      * `get_phase` : numpy matrix
          Matrix of the wave function's phase :math:`\phi(x,y) = \log(\psi(x,y))`


   .. py:method:: GaussianState.get_squared_norm()
      :module: trottersuzuki

      Return the squared norm of the quantum state.

      **Returns**

      * `squared_norm` : float
            Squared norm of the quantum state.

   .. py:method:: GaussianState.loadtxt(file_name)
      :module: trottersuzuki
      
      Load the wave function from a file.

      **Parameters**

      * `file_name` : string
            Name of the file to be written.

      **Example**

          >>> import trottersuzuki as ts  # import the module
          >>> grid = ts.Lattice()  # Define the simulation's geometry
          >>> state = ts.GaussianState(grid, 1.)  # Create the system's state
          >>> state.write_to_file('wave_function.txt')  # Write to a file the wave function
          >>> state2 = ts.State(grid)  # Create a quantum state
          >>> state2.loadtxt('wave_function.txt')  # Load the wave function

   .. py:method:: GaussianState.write_particle_density(file_name)
      :module: trottersuzuki

      Write to a file the particle density matrix of the wave function.

      **Parameters**
      
      * `file_name` : string
          Name of the file. 

   .. py:method:: GaussianState.write_phase(file_name)
      :module: trottersuzuki

      Write to a file the wave function.  

      **Parameters**
      
      * `file_name` : string
            Name of the file to be written. 


   .. py:method:: GaussianState.write_to_file(file_name)
      :module: trottersuzuki

      Write to a file the wave function.  

      **Parameters**

      * `file_name` : string
            Name of the file to be written. 

      **Example**

          >>> import trottersuzuki as ts  # import the module
          >>> grid = ts.Lattice()  # Define the simulation's geometry
          >>> state = ts.GaussianState(grid, 1.)  # Create the system's state
          >>> state.write_to_file('wave_function.txt')  # Write to a file the wave function
          >>> state2 = ts.State(grid)  # Create a quantum state
          >>> state2.loadtxt('wave_function.txt')  # Load the wave function


.. py:class:: SinusoidState
   :module: trottersuzuki

   This class defines a quantum state with sinusoidal like wave function.

   This class is a child of State class.

   **Constructors**
   
   .. py:method:: SinusoidState(grid, n_x=1, n_y=1, norm=1, phase=0)
   
      Construct the quantum state with sinusoidal like wave function.  

      **Parameters**

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

      **Returns**

      * `SinusoidState` : State object. 
          Quantum state with sinusoidal like wave function. The wave function is given by:
          
          .. math:: \psi(x,y) = 2\sqrt{N}/L \sin(2\pi n_x x / L) \sin(2\pi n_y y / L) \mathrm{e}^{(i \phi)}
          
          being :math:`N` the norm of the state, :math:`L` 
          the length of the lattice edge, :math:`n_x` and :math:`n_y` the quantum numbers 
          and :math:`\phi` the relative phase.

      **Example**

          >>> import trottersuzuki as ts  # import the module
          >>> grid = ts.Lattice(300, 30., True, True)  # Define the simulation's geometry
          >>> state = ts.SinusoidState(grid, 2, 0)  # Create the system's state

   **Members**

   .. py:method:: SinusoidState.imprint(function)
      :module: trottersuzuki
      
        Multiply the wave function of the state by the function provided.
        
        **Parameters**

        * `function` : python function
            Function to be printed on the state.
        
        **Notes**

        Useful, for instance, to imprint solitons and vortices on a condensate. 
        Generally, it performs a transformation of the state whose wave function becomes:
        
        .. math:: \psi(x,y)' = f(x,y) \psi(x,y)
        
        being :math:`f(x,y)` the input function and :math:`\psi(x,y)` the initial wave function.        
        
        **Example**
        
            >>> import trottersuzuki as ts  # import the module
            >>> grid = ts.Lattice()  # Define the simulation's geometry
            >>> def vortex(x,y):  # Vortex function
            >>>     z = x + 1j*y
            >>>     angle = np.angle(z)
            >>>     return np.exp(1j * angle)
            >>> state = ts.GaussianState(grid, 1.)  # Create the system's state
            >>> state.imprint(vortex)  # Imprint a vortex on the state
 
   .. py:method:: SinusoidState.get_mean_px()
      :module: trottersuzuki

      Return the expected value of the :math:`P_x` operator.  

      **Returns**

      * `mean_px` : float
            Expected value of the :math:`P_x` operator.

   .. py:method:: SinusoidState.get_mean_pxpx()
      :module: trottersuzuki

      Return the expected value of the :math:`P_x^2` operator.

      **Returns**

      * `mean_pxpx` : float
            Expected value of the :math:`P_x^2` operator.



   .. py:method:: SinusoidState.get_mean_py()
      :module: trottersuzuki

      Return the expected value of the :math:`P_y` operator.  

      **Returns**

      * `mean_py` : float
            Expected value of the :math:`P_y` operator.

   .. py:method:: SinusoidState.get_mean_pypy()
      :module: trottersuzuki

      Return the expected value of the :math:`P_y^2` operator.  

      **Returns**

      * `mean_pypy` : float
            Expected value of the :math:`P_y^2` operator. 

   .. py:method:: SinusoidState.get_mean_x()
      :module: trottersuzuki

      Return the expected value of the :math:`X` operator.  

      **Returns**

      * `mean_x` : float
            Expected value of the :math:`X` operator. 

   .. py:method:: SinusoidState.get_mean_xx()
      :module: trottersuzuki

      Return the expected value of the :math:`X^2` operator.

      **Returns**

      * `mean_xx` : float
            Expected value of the :math:`X^2` operator.   


   .. py:method:: SinusoidState.get_mean_y()
      :module: trottersuzuki

      Return the expected value of the :math:`Y` operator.

      **Returns**
      
      * `mean_y` : float
            Expected value of the :math:`Y` operator.

   .. py:method:: SinusoidState.get_mean_yy()
      :module: trottersuzuki

      Return the expected value of the :math:`Y^2` operator.

      **Returns**
      
      * `mean_yy` : float
            Expected value of the :math:`Y^2` operator.

   .. py:method:: SinusoidState.get_particle_density()
      :module: trottersuzuki

      Return a matrix storing the squared norm of the wave function.

      **Returns**
      
      * `particle_density` : numpy matrix
          Particle density of the state :math:`|\psi(x,y)|^2` 


   .. py:method:: SinusoidState.get_phase()
      :module: trottersuzuki

      Return a matrix of the wave function's phase.

      **Returns**

      * `get_phase` : numpy matrix
          Matrix of the wave function's phase :math:`\phi(x,y) = \log(\psi(x,y))`


   .. py:method:: SinusoidState.get_squared_norm()
      :module: trottersuzuki

      Return the squared norm of the quantum state.

      **Returns**

      * `squared_norm` : float
            Squared norm of the quantum state.

   .. py:method:: SinusoidState.loadtxt(file_name)
      :module: trottersuzuki
      
      Load the wave function from a file.

      **Parameters**

      * `file_name` : string
            Name of the file to be written.

      **Example**

          >>> import trottersuzuki as ts  # import the module
          >>> grid = ts.Lattice()  # Define the simulation's geometry
          >>> state = ts.GaussianState(grid, 1.)  # Create the system's state
          >>> state.write_to_file('wave_function.txt')  # Write to a file the wave function
          >>> state2 = ts.State(grid)  # Create a quantum state
          >>> state2.loadtxt('wave_function.txt')  # Load the wave function

   .. py:method:: SinusoidState.write_particle_density(file_name)
      :module: trottersuzuki

      Write to a file the particle density matrix of the wave function.

      **Parameters**
      
      * `file_name` : string
          Name of the file. 

   .. py:method:: SinusoidState.write_phase(file_name)
      :module: trottersuzuki

      Write to a file the wave function.  

      **Parameters**
      
      * `file_name` : string
            Name of the file to be written. 


   .. py:method:: SinusoidState.write_to_file(file_name)
      :module: trottersuzuki

      Write to a file the wave function.  

      **Parameters**

      * `file_name` : string
            Name of the file to be written. 

      **Example**

          >>> import trottersuzuki as ts  # import the module
          >>> grid = ts.Lattice()  # Define the simulation's geometry
          >>> state = ts.GaussianState(grid, 1.)  # Create the system's state
          >>> state.write_to_file('wave_function.txt')  # Write to a file the wave function
          >>> state2 = ts.State(grid)  # Create a quantum state
          >>> state2.loadtxt('wave_function.txt')  # Load the wave function



Potential Classes
=================
.. py:class:: Potential
   :module: trottersuzuki

   This class defines the external potential that is used for Hamiltonian class.

   **Constructors**

   .. py:method:: Potential(grid)

      Construct the external potential.  

      **Parameters**
      
      * `grid` : Lattice object 
          Define the geometry of the simulation.  

      **Returns**

      * `Potential` : Potential object 
          Create external potential.

      **Example**

          >>> import trottersuzuki as ts  # import the module
          >>> grid = ts.Lattice()  # Define the simulation's geometry
          >>> # Define a constant external potential
          >>> def external_potential_function(x,y):
          >>>     return 1.
          >>> potential = ts.Potential(grid)  # Create the external potential
          >>> potential.init_potential(external_potential_function)  # Initialize the external potential

   **Members**

   .. py:method:: Potential.init_potential(potential_function)
      :module: trottersuzuki

      Initialize the external potential.  

      **Parameters**
      
      * `potential_function` : python function
         Define the external potential function.

      **Example**

         >>> import trottersuzuki as ts  # import the module
         >>> grid = ts.Lattice()  # Define the simulation's geometry
         >>> # Define a constant external potential
         >>> def external_potential_function(x,y):
         >>>     return 1.
         >>> potential = ts.Potential(grid)  # Create the external potential
         >>> potential.init_potential(external_potential_function)  # Initialize the external potential

   .. py:method:: Potential.get_value(x, y)
      :module: trottersuzuki

      Get the value at the lattice's coordinate (x,y).

      **Returns**
      
      * `value` : float
          Value of the external potential.


.. py:class:: HarmonicPotential
   :module: trottersuzuki

   This class defines the external potential, that is used for Hamiltonian class.

   This class is a child of Potential class.

   **Constructors**
   
   .. py:method:: HarmonicPotential(grid, omegax, omegay, mass=1., mean_x=0., mean_y=0.)`

      Construct the harmonic external potential.  

      **Parameters**

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

      **Returns**
      
      * `HarmonicPotential` : Potential object 
          Harmonic external potential.

      **Notes**

      External potential function:\n

      .. math:: V(x,y) = 1/2 m (\omega_x^2  x^2 + \omega_y^2 y^2)

      being :math:`m` the particle mass, :math:`\omega_x` and :math:`\omega_y` the potential frequencies.

      **Example**

          >>> import trottersuzuki as ts  # Import the module
          >>> grid = ts.Lattice()  # Define the simulation's geometry
          >>> potential = ts.HarmonicPotential(grid, 2., 1.)  # Create an harmonic external potential

   **Members**

   .. py:method:: HarmonicPotential.get_value(x, y)
      :module: trottersuzuki

      Get the value at the lattice's coordinate (x,y).

      **Returns**
      
      * `value` : float
          Value of the external potential.



Hamiltonian Classes
===================
.. py:class:: Hamiltonian
   :module: trottersuzuki

   This class defines the Hamiltonian of a single component system.

   **Constructors**
   
   .. py:method:: Hamiltonian(grid, potential=0, mass=1., coupling=0., angular_velocity=0., rot_coord_x=0, rot_coord_y=0)

      Construct the Hamiltonian of a single component system.  

      **Parameters**

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

      **Returns**
      
      * `Hamiltonian` : Hamiltonian object
          Hamiltonian of the system to be simulated: 
          
          .. math:: H(x,y) = \frac{1}{2m}(P_x^2 + P_y^2)  + V(x,y) + g |\psi(x,y)|^2 + \omega L_z
          
          being :math:`m` the particle mass, :math:`V(x,y)` the external potential, 
          :math:`g` the coupling constant of intra-particle interaction, :math:`\omega` 
          the angular velocity of the frame of reference and :math:`L_z` the angular momentum operator along the z-axis.

      **Example**

          >>> import trottersuzuki as ts  # import the module
          >>> grid = ts.Lattice()  # Define the simulation's geometry
          >>> potential = ts.HarmonicPotential(grid, 1., 1.)  # Create an harmonic external potential
          >>> hamiltonian = ts.Hamiltonian(grid, potential)  # Create the Hamiltonian of an harmonic oscillator


.. py:class:: Hamiltonian2Component
   :module: trottersuzuki

   This class defines the Hamiltonian of a two component system.

   **Constructors**

   .. py:method:: Hamiltonian2Component(grid, potential_1=0, potential_2=0, mass_1=1., mass_2=1., coupling_1=0., coupling_12=0., coupling_2=0., omega_r=0, omega_i=0, angular_velocity=0., rot_coord_x=0, rot_coord_y=0)

      Construct the Hamiltonian of a two component system.  

      **Parameters**

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

      **Returns**

      * `Hamiltonian2Component` : Hamiltonian2Component object 
          Hamiltonian of the two-component system to be simulated.
          
          .. math::

             H = \begin{bmatrix} H_1 &  \frac{\Omega}{2} \\ \frac{\Omega}{2} & H_2 \end{bmatrix} 

          being

          .. math::

             H_1 = \frac{1}{2m_1}(P_x^2 + P_y^2) + V_1(x,y) + g_1|\psi_1(x,y)|^2 + g_{12}|\psi_2(x,y)|^2 + \omega L_z  

             H_2 = \frac{1}{2m_2}(P_x^2 + P_y^2) + V_2(x,y) + g_2|\psi_2(x,y)|^2 + g_{12}|\psi_1(x,y)|^2 + \omega L_z  

          and, for the i-th component, :math:`m_i` the particle mass, :math:`V_i(x,y)` the external potential, 
          :math:`g_i` the coupling constant of intra-particle interaction; 
          :math:`g_{12}` the coupling constant of inter-particle interaction 
          :math:`\omega` the angular velocity of the frame of reference, :math:`L_z` the angular momentum operator along the z-axis 
          and :math:`\Omega` the Rabi coupling.
      
      **Example**

          >>> import trottersuzuki as ts  # import the module
          >>> grid = ts.Lattice()  # Define the simulation's geometry
          >>> potential = ts.HarmonicPotential(grid, 1., 1.)  # Create an harmonic external potential
          >>> hamiltonian = ts.Hamiltonian2Component(grid, potential, potential)  # Create the Hamiltonian of an harmonic oscillator for a two-component system

Solver Class
============
.. py:class:: Solver
   :module: trottersuzuki

   This class defines the evolution tasks.

   **Constructors**

   .. py:method:: Solver(grid, state, hamiltonian, delta_t, kernel_type="cpu")

      Construct the Solver object for a single-component system.  

      **Parameters**

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

      **Returns**

      * `Solver` : Solver object  
          Solver object for the simulation of a single-component system.
          
      **Example**

          >>> import trottersuzuki as ts  # import the module
          >>> grid = ts.Lattice()  # Define the simulation's geometry
          >>> state = ts.GaussianState(grid, 1.)  # Create the system's state
          >>> potential = ts.HarmonicPotential(grid, 1., 1.)  # Create harmonic potential
          >>> hamiltonian = ts.Hamiltonian(grid, potential)  # Create a harmonic oscillator Hamiltonian
          >>> solver = ts.Solver(grid, state, hamiltonian, 1e-2)  # Create the solver


   .. py:method:: Solver(grid, state1, state2, hamiltonian, delta_t, kernel_type="cpu")

      Construct the Solver object for a two-component system.  

      **Parameters**

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

      **Returns**

      * `Solver` : Solver object  
          Solver object for the simulation of a two-component system.

      **Example**

          >>> import trottersuzuki as ts  # import the module
          >>> grid = ts.Lattice()  # Define the simulation's geometry
          >>> state_1 = ts.GaussianState(grid, 1.)  # Create first-component system's state
          >>> state_2 = ts.GaussianState(grid, 1.)  # Create second-component system's state
          >>> potential = ts.HarmonicPotential(grid, 1., 1.)  # Create harmonic potential
          >>> hamiltonian = ts.Hamiltonian2Component(grid, potential, potential)  # Create an harmonic oscillator Hamiltonian
          >>> solver = ts.Solver(grid, state_1, state_2, hamiltonian, 1e-2)  # Create the solver

   **Members**

   .. py:method:: Solver.evolve(iterations, imag_time=False)
      :module: trottersuzuki

      Evolve the state of the system.

      **Parameters**

      * `iterations` : integer 
          Number of iterations.
      * `imag_time` : bool,optional (default: False)  
          Whether to perform imaginary time evolution (True) or real time evolution (False).    

      **Notes**
      
      The norm of the state is preserved both in real-time and in imaginary-time evolution.

      **Example**

          >>> import trottersuzuki as ts  # import the module
          >>> grid = ts.Lattice()  # Define the simulation's geometry
          >>> state = ts.GaussianState(grid, 1.)  # Create the system's state
          >>> potential = ts.HarmonicPotential(grid, 1., 1.)  # Create harmonic potential
          >>> hamiltonian = ts.Hamiltonian(grid, potential)  # Create a harmonic oscillator Hamiltonian
          >>> solver = ts.Solver(grid, state, hamiltonian, 1e-2)  # Create the solver
          >>> solver.evolve(1000)  # perform 1000 iteration in real time evolution

   .. py:method:: Solver.get_inter_species_energy()
      :module: trottersuzuki

      Get the inter-particles interaction energy of the system.  

      **Returns**
      
      * `get_inter_species_energy` : float
          Inter-particles interaction energy of the system. 


   .. py:method:: Solver.get_intra_species_energy(which=3)
      :module: trottersuzuki

      Get the intra-particles interaction energy of the system.  

      **Parameters**

      * `which` : integer,optional (default: 3)
          Which intra-particles interaction energy to return: total system (default, which=3), first component (which=1), second component (which=2). 


   .. py:method:: Solver.get_kinetic_energy(which=3)
      :module: trottersuzuki

      Get the kinetic energy of the system.

      **Parameters**

      * `which` : integer,optional (default: 3)
          Which kinetic energy to return: total system (default, which=3), first component (which=1), second component (which=2). 


   .. py:method:: Solver.get_potential_energy(which=3)
      :module: trottersuzuki

      Get the potential energy of the system.  

      **Parameters**

      * `which` : integer,optional (default: 3)
          Which potential energy to return: total system (default, which=3), first component (which=1), second component (which=2). 


   .. py:method:: Solver.get_rabi_energy()
      :module: trottersuzuki

      Get the Rabi energy of the system.

      **Returns**

      * `get_rabi_energy` : float
          Rabi energy of the system.  


   .. py:method:: Solver.get_rotational_energy(which=3)
      :module: trottersuzuki

      Get the rotational energy of the system.

      **Parameters**

      * `which` : integer,optional (default: 3)
          Which rotational energy to return: total system (default, which=3), first component (which=1), second component (which=2). 


   .. py:method:: Solver.get_squared_norm(which=3)
      :module: trottersuzuki

      Get the squared norm of the state (default: total wave-function).

      **Parameters**

      * `which` : integer,optional (default: 3)
          Which squared state norm to return: total system (default, which=3), first component (which=1), second component (which=2). 


   .. py:method:: Solver.get_total_energy()
      :module: trottersuzuki

      Get the total energy of the system.

      **Returns**

      * `get_total_energy` : float
          Total energy of the system.  

      **Example**

          >>> import trottersuzuki as ts  # import the module
          >>> grid = ts.Lattice()  # Define the simulation's geometry
          >>> state = ts.GaussianState(grid, 1.)  # Create the system's state
          >>> potential = ts.HarmonicPotential(grid, 1., 1.)  # Create harmonic potential
          >>> hamiltonian = ts.Hamiltonian(grid, potential)  # Create a harmonic oscillator Hamiltonian
          >>> solver = ts.Solver(grid, state, hamiltonian, 1e-2)  # Create the solver
          >>> solver.get_total_energy()  # Get the total energy
          1
          
   .. py:method:: Solver::update_parameters()
      :module: trottersuzuki

      Notify the solver if any parameter changed in the Hamiltonian


Tools
=====
.. py:method:: vortex_position(grid, state, approx_cloud_radius=0.)

    Get the position of a single vortex in the quantum state.
    
    **Parameters**
    
    * `grid` : Lattice object
        Define the geometry of the simulation.
    * `state` : State object
        System's state.
    * `approx_cloud_radius` : float, optional
        Radius of the circle, centered at the Lattice's origin, where the vortex core
        is expected to be. Need for a better accuracy.
    
    **Returns**

    * `coords` : numpy array
        Coordinates of the vortex core's position (coords[0]: x coordinate; coords[1]: y coordinate).
    
    **Notes**

    Only one vortex must be present in the state.
    
    **Example**
    
        >>> import trottersuzuki as ts  # import the module
        >>> import numpy as np
        >>> grid = ts.Lattice()  # Define the simulation's geometry
        >>> state = ts.GaussianState(grid, 1.)  # Create a state with gaussian wave function
        >>> def vortex_a(x, y):  # Define the vortex to be imprinted
        >>>     z = x + 1j*y
        >>>     angle = np.angle(z)
        >>>     return np.exp(1j * angle)
        >>> state.imprint(vortex)  # Imprint the vortex on the state
        >>> ts.vortex_position(grid, state)
        array([  8.88178420e-16,   8.88178420e-16])
    
