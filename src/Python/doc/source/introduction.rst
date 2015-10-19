============
Introduction
============
The module is a massively parallel implementation of the Trotter-Suzuki approximation to simulate the evolution of quantum systems classically. It relies on interfacing with C++ code with OpenMP for multicore execution.

Key features of the Python interface:

* Fast execution by parallelization: OpenMP.
* NumPy arrays are supported for efficient data exchange.
* Multi-platform: Linux, OS X, and Windows are supported.


Copyright and License
---------------------
Trotter-Suzuki-MPI  is free software; you can redistribute it and/or modify it under the terms of the `GNU General Public License <http://www.gnu.org/licenses/gpl-3.0.html>`_ as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

Trotter-Suzuki-MPI is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the `GNU General Public License <http://www.gnu.org/licenses/gpl-3.0.html>`_ for more details. 


Acknowledgement
---------------
The `original high-performance kernels <https://bitbucket.org/zzzoom/trottersuzuki>`_ were developed by Carlos Bederián. The distributed extension was carried out while `Peter Wittek <http://peterwittek.com/>`_ was visiting the `Department of Computer Applications in Science \& Engineering <http://www.bsc.es/computer-applications>`_ at the `Barcelona Supercomputing Center <http://www.bsc.es/>`_, funded by the "Access to BSC Facilities" project of the `HPC-Europe2 <http://www.hpc-europa.org/>`_ programme (contract no. 228398). Generalizing the capabilities of kernels was carried out by Luca Calderaro while visiting the `Quantum Information Theory Group <https://www.icfo.eu/research/group_details.php?id=19>`_ at `ICFO-The Institute of Photonic Sciences <https://www.icfo.eu/>`_, sponsored by the `Erasmus+ <http://ec.europa.eu/programmes/erasmus-plus/index_en.htm>`_ programme.
