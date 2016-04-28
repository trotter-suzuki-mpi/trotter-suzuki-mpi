Introduction
============
The module is a massively parallel implementation of the Trotter-Suzuki approximation to simulate the evolution of quantum systems classically.  It relies on interfacing with C++ code with OpenMP for multicore execution, and it can be accelerated by CUDA.

Key features of the Python interface:

* Simulation of 2D quantum systems.
* Fast execution by parallelization: OpenMP and CUDA are supported.
* Many-body simulations with non-interacting particles.
* Solving the Gross-Pitaevskii equation  (e.g., `dark soltions <https://github.com/Lucacalderaro/Master-Thesis/blob/master/Soliton%20generation%20on%20Bose-Einstein%20Condensate.ipynb>`_, `vortex dynamics in Bose-Einstein Condensates <http://nbviewer.jupyter.org/github/trotter-suzuki-mpi/notebooks/blob/master/Vortex%20Dynamics.ipynb>`_).
* Imaginary time evolution to approximate the ground state.
* Stationary and time-dependent external potential.
* NumPy arrays are supported for efficient data exchange.
* Multi-platform: Linux, OS X, and Windows are supported.

Copyright and License
---------------------
Trotter-Suzuki-MPI  is free software; you can redistribute it and/or modify it under the terms of the `GNU General Public License <http://www.gnu.org/licenses/gpl-3.0.html>`_ as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

Trotter-Suzuki-MPI is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the `GNU General Public License <http://www.gnu.org/licenses/gpl-3.0.html>`_ for more details.


Acknowledgement
---------------
The `original high-performance kernels <https://bitbucket.org/zzzoom/trottersuzuki>`_ were developed by Carlos Bederián. The distributed extension was carried out while `Peter Wittek <http://peterwittek.com/>`_ was visiting the `Department of Computer Applications in Science \& Engineering <http://www.bsc.es/computer-applications>`_ at the `Barcelona Supercomputing Center <http://www.bsc.es/>`_, funded by the "Access to BSC Facilities" project of the `HPC-Europe2 <http://www.hpc-europa.org/>`_ programme (contract no. 228398). Generalizing the capabilities of kernels was carried out by Luca Calderaro while visiting the `Quantum Information Theory Group <https://www.icfo.eu/research/group_details.php?id=19>`_ at `ICFO-The Institute of Photonic Sciences <https://www.icfo.eu/>`_, sponsored by the `Erasmus+ <http://ec.europa.eu/programmes/erasmus-plus/index_en.htm>`_ programme. `Pietro Massignan <http://users.icfo.es/Pietro.Massignan/>`_ has contributed to the project with extensive testing and suggestions of new features.

Citations
---------

1. Bederián, C. & Dente, A. (2011). Boosting quantum evolutions using Trotter-Suzuki algorithms on GPUs. *Proceedings of HPCLatAm-11, 4th High-Performance Computing Symposium*. `PDF <http://www.famaf.unc.edu.ar/grupos/GPGPU/boosting_trotter-suzuki.pdf>`_

2. Wittek, P. and Cucchietti, F.M. (2013). `A Second-Order Distributed Trotter-Suzuki Solver with a Hybrid CPU-GPU Kernel <http://dx.doi.org/10.1016/j.cpc.2012.12.008>`_. *Computer Physics Communications*, 184, pp. 1165-1171. `PDF <http://arxiv.org/pdf/1208.2407>`_

3. Wittek, P. and Calderaro, L. (2015). `Extended computational kernels in a massively parallel implementation of the Trotter-Suzuki approximation <http://dx.doi.org/10.1016/j.cpc.2015.07.017>`_. *Computer Physics Communications*, 197, pp. 339-340. `PDF <https://www.researchgate.net/profile/Peter_Wittek/publication/280962265_Extended_Computational_Kernels_in_a_Massively_Parallel_Implementation_of_the_TrotterSuzuki_Approximation/links/55cebd1f08aee19936fc5dcf.pdf>`_
