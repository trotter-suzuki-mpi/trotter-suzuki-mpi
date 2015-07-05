Trotter-Suzuki-MPI - MATLAB Interface
=====================================

The module is a massively parallel implementation of the Trotter-Suzuki approximation to simulate the evolution of quantum systems classically. It relies on interfacing with C++ code with OpenMP for multicore execution.

Key features of the MATLAB interface:

* Fast execution by parallelization: OpenMP.
* Multi-platform: Linux, OS X, and Windows are supported.

Usage
------
The following code block gives a simple example of initializing a state and plotting the result of the evolution.

.. code-block:: python
		
  imag_time = 0;
  order_approx = 2;
  dim = 640;
  iterations = 600;
  kernel_type = 0;
  periods = zeros(1,2);
  
  particle_mass = 1;
  time_single_it = 0.08 * particle_mass / 2;
  
  h_a = cos(time_single_it / (2. * particle_mass));
  h_b = sin(time_single_it / (2. * particle_mass));
  
  p_real = zeros(dim, dim);
  p_imag = zeros(dim, dim);
  pot_r = zeros(dim, dim);
  pot_i = zeros(dim, dim);
  
  CONST = -1. * time_single_it * order_approx;
  for y = 1:dim
      for x = 1:dim
          p_real(y, x) = sin(2 * pi * x / dim) * sin(2 * pi * y / dim);
          pot_r(y, x) = cos(CONST * pot_r(y, x));
      pot_i(y, x) = sin(CONST * pot_i(y, x));
      end
  end
  
  [pf_real, pf_imag] = MexTrotter(h_a, h_b, pot_r, pot_i, p_real, p_imag, iterations, kernel_type, periods, imag_time);
  
  colormap('hot');
  imagesc(pf_real);
  colorbar
  
Trotter-Suzuki-MPI MATLAB Extension Build Guide (Linux/Mac):
================================

1. Referring to the installation instructions reported here https://github.com/peterwittek/trotter-suzuki-mpi, first run

    $ ./autogen.sh

   Then run ``configure`` disabling MPI and CUDA:

    $ ./configure --without-mpi --without-cuda
 
2. Build MATLAB Extension by running:
   ::
      MEX_BIN="/usr/local/MATLAB/R2015a/bin/mex" ./makeMex.sh
    
   where ``MEX_BIN`` is the path to the MATLAB installation mex binary.

3. Then ``MexTrotter.mexa64`` or ``MexTrotter.mexa32`` is generated for use, you can test by running the ``mex_trotter_interface_test.m``.

Building Mex Extension on Windows:
===================================

First, run ``mex -setup`` in CMD to see if MATLAB and Visual C++ compiler are installed properly; the CMD will prompt for available compilers and you can choose the appropriate version. If you do not have it, you should install some supported version of Visual Studio that includes the Visual C++ compiler by your MATLAB version like on `this <http://www.mathworks.com/support/compilers/R2015a/index.html?sec=win64/>`_ page.

Then run the script in this folder makeMex.bat in CMD and the ``MexTrotter.mexa64`` or ``MexTrotter.mexa32`` is generated for use, you can test by running the ``mex_trotter_interface_test.m``.
