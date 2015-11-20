Trotter-Suzuki-MPI - MATLAB Interface
=====================================

The module is a massively parallel implementation of the Trotter-Suzuki approximation to simulate the evolution of quantum systems classically. It relies on interfacing with C++ code with OpenMP for multicore execution, and it can be accelerated by CUDA..

Key features of the MATLAB interface:

* Fast execution by parallelization: OpenMP and CUDA.
* Multi-platform: Linux, OS X, and Windows are supported.

Usage
------
The following code block gives a simple example of initializing a state and plotting the result of the evolution.

.. code-block:: python
		
  # lattice parameters
  dim = 200;					% linear dimensione of the lattice
  delta_x = 1.;				% physical resolution along the x axis
  delta_y = 1.;				% physical resolution along the y axis
  periods = zeros(1,2); 		% 0 for closed boundary conditions, 1 for periodic boundary conditions

  # Hamiltonian parameters
  particle_mass = 1;
  coupling_const = 0.;
  external_potential = zeros(dim, dim);

  % initial state.
  p_real = zeros(dim, dim);
  p_imag = zeros(dim, dim);
  for y = 1:dim
      for x = 1:dim
          p_real(y, x) = sin(2 * pi * x / dim) * sin(2 * pi * y / dim);
      end
  end

  # evolution parameters
  imag_time = 0;			% 0 for real time evolution, 1 for imaginary time evolution
  delta_t = 0.001;		
  iterations = 200;		% number of iteration
  kernel_type = 0;		% use CPU kernel

  [pf_real, pf_imag] = MexTrotter(p_real, p_imag, particle_mass, coupling_const, external_potential, delta_x, delta_y, delta_t, iterations, kernel_type, periods, imag_time);

  TotalEnergy = MexH(pf_real, pf_imag, particle_mass, coupling_const, external_potential, delta_x, delta_y);
  KineticEnergy = MexK(pf_real, pf_imag, particle_mass, delta_x, delta_y);
  Norm2 = MexNorm(pf_real, pf_imag, delta_x, delta_y);

  colormap('hot');
  imagesc(pf_real);
  colorbar
  
Trotter-Suzuki-MPI MATLAB Extension Build Guide (Linux/Mac):
================================

1. Referring to the installation instructions reported here https://github.com/trotter-suzuki-mpi/trotter-suzuki-mpi, first run

    $ ./autogen.sh

Then run ``configure`` disabling MPI. You can also supply the root directory of your MATLAB installation:

    $ ./configure --without-mpi --without-cuda --with-matlab=/usr/local/MATLAB/R2015a/bin/mex

If you want CUDA support, specify the CUDA directory as well.
 
2. Build the MATLAB Extension
   ::
      make matlab

3. Then ``MexTrotter.mexa64`` or ``MexTrotter.mexa32`` is generated for use, you can test by running the ``mex_trotter_interface_test.m``.

Building Mex Extension on Windows:
===================================

First, run ``mex -setup`` in CMD to see if MATLAB and Visual C++ compiler are installed properly; the CMD will prompt for available compilers and you can choose the appropriate version. If you do not have it, you should install some supported version of Visual Studio that includes the Visual C++ compiler by your MATLAB version like on `this <http://www.mathworks.com/support/compilers/R2015a/index.html?sec=win64/>`_ page.

Then run the script in this folder makeMex.bat in CMD and the ``MexTrotter.mexa64`` or ``MexTrotter.mexa32`` is generated for use, you can test by running the ``mex_trotter_interface_test.m``.
