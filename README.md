Distributed Trotter-Suzuki Solver
==

This is a distributed variant of efficient implementations of a simple Trotter-Suzuki solver. The kernels are based on the efficient [implementation](https://bitbucket.org/zzzoom/trottersuzuki) by 
Carlos Bederián

Dependencies
==
CUDA and an MPI implementation are required for compiling. It also require a GPU with at least Compute Cabapility 2.0 for the CUDA kernel to execute.

Usage
==

A test example is included. Usage: trottertest [OPTIONS]. 

Arguments:

    -d NUMBER     Matrix dimension (default: 640)
    -i NUMBER     Number of iterations (default: 1000)
    -k NUMBER     Kernel type (default: 0): 
                    0: CPU, cache-optimized
                    1: CPU, SSE and cache-optimized
                    2: GPU
                    3: Hybrid CPU-GPU (experimental)                    
    -s NUMBER     Snapshots are taken at every NUMBER of iterations.
                    Zero means no snapshots. Default: 0.

Example:

    mpirun -np 4 build/trottertest -k 0 -i 100 -d 640 -s 10

In case of the SSE kernel, the chunk of the matrix assigned to a node, that is a tile, must have a width that is divisible by two. This puts a constraint on the possible matrix sizes. For instance, running twelve MPI threads in a 4x3 configuration, the dimensions must be divisible by six and eight.

The hybrid kernel is experimental. It splits the work between the GPU and the CPU. It uses one MPI thread per GPU, and uses OpenMP to use parallelism on the CPU. It is efficient if the matrix does not fit the GPU memory. For instance, given twelve physical cores in a single node with two Tesla C2050 GPUs, a 14,000x14,000 would not fit the GPU memory. The following command would calculate the part that does not fit the device memory on the CPU:

    OMP_NUM_THREADS=6 mpirun -np 2 build/trottertest -k 3 -i 100 -d 14000

The included shell script single2double.sh generates a double precision variant. Executing it will overwrite the single precision source code.

Citation
==
Further details are available in the following paper:

Wittek, P. and Cucchietti, F.M. (2013). A Second-Order Distributed Trotter-Suzuki Solver with a Hybrid CPU-GPU Kernel. Computer Physics Communications, 184, pp. 1165-1171. http://arxiv.org/abs/1208.2407
