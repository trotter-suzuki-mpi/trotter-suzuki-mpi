Massively Parallel Trotter-Suzuki Solver
========================================

The Trotter-Suzuki approximation leads to an efficient algorithm for simulating quantum systems. This library provides a scalable, high-precision implementation that uses parallel and distributed computational resources. The implementation built on [single-node parallel kernels](https://bitbucket.org/zzzoom/trottersuzuki) [1], extending them to use distributed resources [2], and generalizing the kernels to be able to tackle a wider range of problems in quantum physics [3].

Key features:

  - Simulation of 2D quantum systems.
  - Arbitrary single-body initial state with closed and periodic boundary conditions.
  - Many-body simulations with non-interacting particles.
  - [Gross-Pitaevskii equation](https://github.com/Lucacalderaro/Master-Thesis/blob/master/Soliton%20generation%20on%20Bose-Einstein%20Condensate.ipynb).
  - Imaginary time evolution to calculate the ground state.
  - Stationary and time-dependent external potential.
  - A command-line interface (CLI) and a C++ application programming interface (API) for flexible use.
  - A [Python](https://trotter-suzuki-mpi.readthedocs.io/) wrapper is provided.
  - Cache optimized multi-core, GPU, and hybrid kernels.
  - Near-linear scaling across multiple nodes with computations overlapping communication.

The documentation of the CLI and the C++ interface is available at [trotter-suzuki-mpi.github.io](https://trotter-suzuki-mpi.github.io/). The Python documentation is on [Read the Docs](https://trotter-suzuki-mpi.readthedocs.io/).

Copyright and License
---------------------
Trotter-Suzuki-MPI  is free software; you can redistribute it and/or modify it under the terms of the [GNU General Public License](http://www.gnu.org/licenses/gpl-3.0.html) as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

Trotter-Suzuki-MPI is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the [GNU General Public License](http://www.gnu.org/licenses/gpl-3.0.html) for more details.

Acknowledgement
---------------
The [original high-performance kernels](https://bitbucket.org/zzzoom/trottersuzuki) were developed by Carlos Bederián. The distributed extension was carried out while [Peter Wittek](http://peterwittek.com/) was visiting the [Department of Computer Applications in Science \& Engineering](http://www.bsc.es/computer-applications) at the [Barcelona Supercomputing Center](http://www.bsc.es/), funded by the "Access to BSC Facilities" project of the [HPC-Europe2](http://www.hpc-europa.org/) programme (contract no. 228398). Generalizing the capabilities of kernels was carried out by Luca Calderaro while visiting the [Quantum Information Theory Group](https://www.icfo.eu/research/group_details.php?id=19) at [ICFO-The Institute of Photonic Sciences](https://www.icfo.eu/), sponsored by the [Erasmus+](http://ec.europa.eu/programmes/erasmus-plus/index_en.htm) programme.
[Pietro Massignan](http://users.icfo.es/Pietro.Massignan/) has contributed to the project with extensive testing and suggestions of new features.

References
----------

  1. Bederián, C. and Dente, A. (2011). Boosting quantum evolutions using Trotter-Suzuki algorithms on GPUs. *Proceedings of HPCLatAm-11, 4th High-Performance Computing Symposium*. [PDF](http://www.famaf.unc.edu.ar/grupos/GPGPU/boosting_trotter-suzuki.pdf)

  2. Wittek, P. and Cucchietti, F.M. (2013). [A Second-Order Distributed Trotter-Suzuki Solver with a Hybrid CPU-GPU Kernel](http://dx.doi.org/10.1016/j.cpc.2012.12.008). *Computer Physics Communications*, 184, pp. 1165-1171. [PDF](http://arxiv.org/pdf/1208.2407)

  3. Wittek, P. and Calderaro, L. (2015). [Extended computational kernels in a massively parallel implementation of the Trotter-Suzuki approximation](http://dx.doi.org/10.1016/j.cpc.2015.07.017). *Computer Physics Communications*, 197, pp. 339-340. [PDF](https://www.researchgate.net/profile/Peter_Wittek/publication/280962265_Extended_Computational_Kernels_in_a_Massively_Parallel_Implementation_of_the_TrotterSuzuki_Approximation/links/55cebd1f08aee19936fc5dcf.pdf)
