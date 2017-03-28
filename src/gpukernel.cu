/**
 * Massively Parallel Trotter-Suzuki Solver
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#undef _GLIBCXX_ATOMIC_BUILTINS
#include <stdio.h>
#include <sstream>
#include <vector>
#include <map>
#include <cassert>
#include "common.h"
#include "kernel.h"

#define CUT_CHECK_ERROR(errorMessage) {                                      \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
      stringstream sstm;                                                     \
      sstm << errorMessage <<"\n The error is " << cudaGetErrorString( err); \
      my_abort(sstm.str());  }}

#define CUDA_SAFE_CALL(call) \
  if ((call) != cudaSuccess) { \
      cudaError_t err = cudaGetLastError(); \
      stringstream sstm; \
      sstm << "CUDA error calling \""#call"\", code is " << err; \
      my_abort(sstm.str()); }

/** Check and initialize a device attached to a node
 *  @param commRank - the MPI rank of this process
 *  @param commSize - the size of MPI comm world
 *  This snippet is from GPMR:
 *  http://code.google.com/p/gpmr/
 */
void setDevice(int commRank
#ifdef HAVE_MPI
               , MPI_Comm cartcomm
#endif
              ) {
    int devCount;
    int deviceNum = 0; //-1;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&devCount));

#ifdef HAVE_MPI
    int commSize = 1;
    MPI_Comm_size(cartcomm, &commSize);
#ifdef _WIN32
    FILE * fp = popen("hostname.exe", "r");
#else
    FILE * fp = popen("/bin/hostname", "r");
#endif
    char buf[1024];
    if (fgets(buf, 1023, fp) == NULL) strcpy(buf, "localhost");
    pclose(fp);
    string host = buf;
    host = host.substr(0, host.size() - 1);
    strcpy(buf, host.c_str());

    if (commRank == 0) {
        map<string, vector<int> > hosts;
        map<string, int> devCounts;
        MPI_Status stat;
        MPI_Request req;

        hosts[buf].push_back(0);
        devCounts[buf] = devCount;
        for (int i = 1; i < commSize; ++i) {
            MPI_Recv(buf, 1024, MPI_CHAR, i, 0, cartcomm, &stat);
            MPI_Recv(&devCount, 1, MPI_INT, i, 0, cartcomm, &stat);

            // check to make sure each process on each node reports the same number of devices.
            hosts[buf].push_back(i);
            if (devCounts.find(buf) != devCounts.end()) {
                if (devCounts[buf] != devCount) {
                    printf("Error, device count mismatch %d != %d on %s\n", devCounts[buf], devCount, buf);
                    fflush(stdout);
                }
            }
            else devCounts[buf] = devCount;
        }
        // check to make sure that we don't have more jobs on a node than we have GPUs.
        for (map<string, vector<int> >::iterator it = hosts.begin(); it != hosts.end(); ++it) {
            if (it->second.size() > static_cast<unsigned int>(devCounts[it->first])) {
                printf("Error, more jobs running on '%s' than devices - %d jobs > %d devices.\n",
                       it->first.c_str(), static_cast<int>(it->second.size()), devCounts[it->first]);
                fflush(stdout);
                MPI_Abort(cartcomm, 1);
            }
        }

        // send out the device number for each process to use.
        MPI_Irecv(&deviceNum, 1, MPI_INT, 0, 0, cartcomm, &req);
        for (map<string, vector<int> >::iterator it = hosts.begin(); it != hosts.end(); ++it) {
            for (unsigned int i = 0; i < it->second.size(); ++i) {
                int devID = i;
                MPI_Send(&devID, 1, MPI_INT, it->second[i], 0, cartcomm);
            }
        }
        MPI_Wait(&req, &stat);
    }
    else {
        // send out the hostname and device count for your local node, then get back the device number you should use.
        MPI_Status stat;
        MPI_Send(buf, strlen(buf) + 1, MPI_CHAR, 0, 0, cartcomm);
        MPI_Send(&devCount, 1, MPI_INT, 0, 0, cartcomm);
        MPI_Recv(&deviceNum, 1, MPI_INT, 0, 0, cartcomm, &stat);
    }
    MPI_Barrier(cartcomm);
#endif
    CUDA_SAFE_CALL(cudaSetDevice(deviceNum));
}

CC2Kernel::CC2Kernel(Lattice *grid, State *state, Hamiltonian *hamiltonian,
                     double *_external_pot_real, double *_external_pot_imag,
                     double delta_t, double _norm, bool _imag_time):
    threadsPerBlock(BLOCK_X, STRIDE_Y),
    sense(0),
    state_index(0),
    imag_time(_imag_time) {

    halo_x = grid->halo_x;
    halo_y = grid->halo_y;
    p_real[0] = state->p_real;
    p_imag[0] = state->p_imag;
    p_real[1] = NULL;
    p_imag[1] = NULL;
    delta_x = grid->delta_x;
    delta_y = grid->delta_y;
    periods = grid->periods;
    alpha_x = hamiltonian->angular_velocity * delta_t * grid->delta_x / (2 * grid->delta_y);
    alpha_y = hamiltonian->angular_velocity * delta_t * grid->delta_y / (2 * grid->delta_x);

    coupling_const = new double[3];
    coupling_const[0] = hamiltonian->coupling_a * delta_t;
    coupling_const[1] = 0.;
    coupling_const[2] = 0.;

    aH = new double [1];
	bH = new double [1];
	aV = new double [1];
	bV = new double [1];
	if (imag_time) {
		aH[0] = cosh(delta_t / (4. * hamiltonian->mass * grid->delta_x * grid->delta_x));
		bH[0] = sinh(delta_t / (4. * hamiltonian->mass * grid->delta_x * grid->delta_x));
		aV[0] = cosh(delta_t / (4. * hamiltonian->mass * grid->delta_y * grid->delta_y));
		bV[0] = sinh(delta_t / (4. * hamiltonian->mass * grid->delta_y * grid->delta_y));
	}
	else {
		aH[0] = cos(delta_t / (4. * hamiltonian->mass * grid->delta_x * grid->delta_x));
		bH[0] = sin(delta_t / (4. * hamiltonian->mass * grid->delta_x * grid->delta_x));
		aV[0] = cos(delta_t / (4. * hamiltonian->mass * grid->delta_y * grid->delta_y));
		bV[0] = sin(delta_t / (4. * hamiltonian->mass * grid->delta_y * grid->delta_y));
	}
    norm = new double [1];
    norm[0] = _norm;
    tot_norm = norm[0];
    external_pot_real[0] = _external_pot_real;
    external_pot_imag[0] = _external_pot_imag;
    two_wavefunctions = false;

    int rank;
#ifdef HAVE_MPI
    cartcomm = grid->cartcomm;
    MPI_Cart_shift(cartcomm, 0, 1, &neighbors[UP], &neighbors[DOWN]);
    MPI_Cart_shift(cartcomm, 1, 1, &neighbors[LEFT], &neighbors[RIGHT]);
    MPI_Comm_rank(cartcomm, &rank);
#else
    neighbors[UP] = neighbors[DOWN] = neighbors[LEFT] = neighbors[RIGHT] = 0;
    rank = 0;
#endif
    start_x = grid->start_x;
    end_x = grid->end_x;
    inner_start_x = grid->inner_start_x;
    inner_end_x = grid->inner_end_x;
    start_y = grid->start_y;
    end_y = grid->end_y;
    inner_start_y = grid->inner_start_y;
    inner_end_y = grid->inner_end_y;
    tile_width = end_x - start_x;
    tile_height = end_y - start_y;

    setDevice(rank
#ifdef HAVE_MPI
              , cartcomm
#endif
             );
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dev_external_pot_real[0]), tile_width * tile_height * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dev_external_pot_imag[0]), tile_width * tile_height * sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpy(dev_external_pot_real[0], external_pot_real[0], tile_width * tile_height * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dev_external_pot_imag[0], external_pot_imag[0], tile_width * tile_height * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&pdev_real[0][0]), tile_width * tile_height * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&pdev_real[0][1]), tile_width * tile_height * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&pdev_imag[0][0]), tile_width * tile_height * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&pdev_imag[0][1]), tile_width * tile_height * sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpy(pdev_real[0][0], p_real[0], tile_width * tile_height * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(pdev_imag[0][0], p_imag[0], tile_width * tile_height * sizeof(double), cudaMemcpyHostToDevice));
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        my_abort("CuBLAS initialization error");
    }

    // Halo exchange uses wave pattern to communicate
    int height = inner_end_y - inner_start_y;	// The vertical halo in rows
    int width = halo_x;	// The number of columns of the matrix
    // Allocating pinned memory for the buffers
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &left_real_receive, height * width * sizeof(double), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &left_real_send, height * width * sizeof(double), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &right_real_receive, height * width * sizeof(double), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &right_real_send, height * width * sizeof(double), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &left_imag_receive, height * width * sizeof(double), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &left_imag_send, height * width * sizeof(double), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &right_imag_receive, height * width * sizeof(double), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &right_imag_send, height * width * sizeof(double), cudaHostAllocDefault));

    height = halo_y;	// The vertical halo in rows
    width = tile_width;	// The number of columns of the matrix
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &bottom_real_receive, height * width * sizeof(double), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &bottom_real_send, height * width * sizeof(double), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &top_real_receive, height * width * sizeof(double), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &top_real_send, height * width * sizeof(double), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &bottom_imag_receive, height * width * sizeof(double), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &bottom_imag_send, height * width * sizeof(double), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &top_imag_receive, height * width * sizeof(double), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &top_imag_send, height * width * sizeof(double), cudaHostAllocDefault));

}

CC2Kernel::CC2Kernel(Lattice *grid, State *state1, State *state2,
                     Hamiltonian2Component *hamiltonian,
                     double **_external_pot_real, double **_external_pot_imag,
                     double delta_t, double *_norm, bool _imag_time):
    threadsPerBlock(BLOCK_X, STRIDE_Y),
    sense(0),
    state_index(0),
    imag_time(_imag_time) {

    halo_x = grid->halo_x;
    halo_y = grid->halo_y;
    p_real[0] = state1->p_real;
    p_imag[0] = state1->p_imag;
    p_real[1] = state2->p_real;
    p_imag[1] = state2->p_imag;
    delta_x = grid->delta_x;
    delta_y = grid->delta_y;
    periods = grid->periods;
    alpha_x = hamiltonian->angular_velocity * delta_t * grid->delta_x / (2 * grid->delta_y);
    alpha_y = hamiltonian->angular_velocity * delta_t * grid->delta_y / (2 * grid->delta_x);

    coupling_const = new double[5];
    coupling_const[0] = delta_t * hamiltonian->coupling_a;
    coupling_const[1] = delta_t * hamiltonian->coupling_b;
    coupling_const[2] = delta_t * hamiltonian->coupling_ab;
    coupling_const[3] = 0.5 * hamiltonian->omega_r;
    coupling_const[4] = 0.5 * hamiltonian->omega_i;
    aH = new double [2];
	bH = new double [2];
	aV = new double [2];
	bV = new double [2];
    if (imag_time) {
    	aH[0] = cosh(delta_t / (4. * hamiltonian->mass * grid->delta_x * grid->delta_x));
		bH[0] = sinh(delta_t / (4. * hamiltonian->mass * grid->delta_x * grid->delta_x));
    	aH[1] = cosh(delta_t / (4. * hamiltonian->mass_b * grid->delta_x * grid->delta_x));
		bH[1] = sinh(delta_t / (4. * hamiltonian->mass_b * grid->delta_x * grid->delta_x));
		aV[0] = cosh(delta_t / (4. * hamiltonian->mass * grid->delta_y * grid->delta_y));
		bV[0] = sinh(delta_t / (4. * hamiltonian->mass * grid->delta_y * grid->delta_y));
		aV[1] = cosh(delta_t / (4. * hamiltonian->mass_b * grid->delta_y * grid->delta_y));
		bV[1] = sinh(delta_t / (4. * hamiltonian->mass_b * grid->delta_y * grid->delta_y));
	}
	else {
		aH[0] = cos(delta_t / (4. * hamiltonian->mass * grid->delta_x * grid->delta_x));
		bH[0] = sin(delta_t / (4. * hamiltonian->mass * grid->delta_x * grid->delta_x));
		aH[1] = cos(delta_t / (4. * hamiltonian->mass_b * grid->delta_x * grid->delta_x));
		bH[1] = sin(delta_t / (4. * hamiltonian->mass_b * grid->delta_x * grid->delta_x));
		aV[0] = cos(delta_t / (4. * hamiltonian->mass * grid->delta_y * grid->delta_y));
		bV[0] = sin(delta_t / (4. * hamiltonian->mass * grid->delta_y * grid->delta_y));
		aV[1] = cos(delta_t / (4. * hamiltonian->mass_b * grid->delta_y * grid->delta_y));
		bV[1] = sin(delta_t / (4. * hamiltonian->mass_b * grid->delta_y * grid->delta_y));
	}
    norm = _norm;
    tot_norm = norm[0] + norm[1];
    external_pot_real[0] = _external_pot_real[0];
    external_pot_imag[0] = _external_pot_imag[0];
    external_pot_real[1] = _external_pot_real[1];
    external_pot_imag[1] = _external_pot_imag[1];
    two_wavefunctions = true;

    int rank;
#ifdef HAVE_MPI
    cartcomm = grid->cartcomm;
    MPI_Cart_shift(cartcomm, 0, 1, &neighbors[UP], &neighbors[DOWN]);
    MPI_Cart_shift(cartcomm, 1, 1, &neighbors[LEFT], &neighbors[RIGHT]);
    MPI_Comm_rank(cartcomm, &rank);
#else
    neighbors[UP] = neighbors[DOWN] = neighbors[LEFT] = neighbors[RIGHT] = 0;
    rank = 0;
#endif
    start_x = grid->start_x;
    end_x = grid->end_x;
    inner_start_x = grid->inner_start_x;
    inner_end_x = grid->inner_end_x;
    start_y = grid->start_y;
    end_y = grid->end_y;
    inner_start_y = grid->inner_start_y;
    inner_end_y = grid->inner_end_y;
    tile_width = end_x - start_x;
    tile_height = end_y - start_y;

    setDevice(rank
#ifdef HAVE_MPI
              , cartcomm
#endif
             );
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dev_external_pot_real[0]), tile_width * tile_height * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dev_external_pot_imag[0]), tile_width * tile_height * sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpy(dev_external_pot_real[0], external_pot_real[0], tile_width * tile_height * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dev_external_pot_imag[0], external_pot_imag[0], tile_width * tile_height * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&pdev_real[0][0]), tile_width * tile_height * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&pdev_real[0][1]), tile_width * tile_height * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&pdev_imag[0][0]), tile_width * tile_height * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&pdev_imag[0][1]), tile_width * tile_height * sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpy(pdev_real[0][0], p_real[0], tile_width * tile_height * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(pdev_imag[0][0], p_imag[0], tile_width * tile_height * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dev_external_pot_real[1]), tile_width * tile_height * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&dev_external_pot_imag[1]), tile_width * tile_height * sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpy(dev_external_pot_real[1], external_pot_real[1], tile_width * tile_height * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dev_external_pot_imag[1], external_pot_imag[1], tile_width * tile_height * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&pdev_real[1][0]), tile_width * tile_height * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&pdev_real[1][1]), tile_width * tile_height * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&pdev_imag[1][0]), tile_width * tile_height * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&pdev_imag[1][1]), tile_width * tile_height * sizeof(double)));
    CUDA_SAFE_CALL(cudaMemcpy(pdev_real[1][0], p_real[1], tile_width * tile_height * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(pdev_imag[1][0], p_imag[1], tile_width * tile_height * sizeof(double), cudaMemcpyHostToDevice));

    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        my_abort("CuBLAS initialization error");
    }

    // Halo exchange uses wave pattern to communicate
    int height = inner_end_y - inner_start_y;	// The vertical halo in rows
    int width = halo_x;	// The number of columns of the matrix
    // Allocating pinned memory for the buffers
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &left_real_receive, height * width * sizeof(double), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &left_real_send, height * width * sizeof(double), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &right_real_receive, height * width * sizeof(double), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &right_real_send, height * width * sizeof(double), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &left_imag_receive, height * width * sizeof(double), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &left_imag_send, height * width * sizeof(double), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &right_imag_receive, height * width * sizeof(double), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &right_imag_send, height * width * sizeof(double), cudaHostAllocDefault));

    height = halo_y;	// The vertical halo in rows
    width = tile_width;	// The number of columns of the matrix
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &bottom_real_receive, height * width * sizeof(double), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &bottom_real_send, height * width * sizeof(double), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &top_real_receive, height * width * sizeof(double), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &top_real_send, height * width * sizeof(double), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &bottom_imag_receive, height * width * sizeof(double), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &bottom_imag_send, height * width * sizeof(double), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &top_imag_receive, height * width * sizeof(double), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &top_imag_send, height * width * sizeof(double), cudaHostAllocDefault));

}

void CC2Kernel::update_potential(double *_external_pot_real, double *_external_pot_imag, int which) {
    external_pot_real[which] = _external_pot_real;
    external_pot_imag[which] = _external_pot_imag;
    CUDA_SAFE_CALL(cudaMemcpy(dev_external_pot_real[which], external_pot_real[which], tile_width * tile_height * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dev_external_pot_imag[which], external_pot_imag[which], tile_width * tile_height * sizeof(double), cudaMemcpyHostToDevice));
}


CC2Kernel::~CC2Kernel() {
    CUDA_SAFE_CALL(cudaFreeHost(left_real_receive));
    CUDA_SAFE_CALL(cudaFreeHost(left_real_send));
    CUDA_SAFE_CALL(cudaFreeHost(right_real_receive));
    CUDA_SAFE_CALL(cudaFreeHost(right_real_send));
    CUDA_SAFE_CALL(cudaFreeHost(bottom_real_receive));
    CUDA_SAFE_CALL(cudaFreeHost(bottom_real_send));
    CUDA_SAFE_CALL(cudaFreeHost(top_real_receive));
    CUDA_SAFE_CALL(cudaFreeHost(top_real_send));
    CUDA_SAFE_CALL(cudaFreeHost(left_imag_receive));
    CUDA_SAFE_CALL(cudaFreeHost(left_imag_send));
    CUDA_SAFE_CALL(cudaFreeHost(right_imag_receive));
    CUDA_SAFE_CALL(cudaFreeHost(right_imag_send));
    CUDA_SAFE_CALL(cudaFreeHost(bottom_imag_receive));
    CUDA_SAFE_CALL(cudaFreeHost(bottom_imag_send));
    CUDA_SAFE_CALL(cudaFreeHost(top_imag_receive));
    CUDA_SAFE_CALL(cudaFreeHost(top_imag_send));

    CUDA_SAFE_CALL(cudaFree(pdev_real[0][0]));
    CUDA_SAFE_CALL(cudaFree(pdev_real[0][1]));
    CUDA_SAFE_CALL(cudaFree(pdev_imag[0][0]));
    CUDA_SAFE_CALL(cudaFree(pdev_imag[0][1]));
    CUDA_SAFE_CALL(cudaFree(dev_external_pot_real[0]));
    CUDA_SAFE_CALL(cudaFree(dev_external_pot_imag[0]));
    if (two_wavefunctions) {
        CUDA_SAFE_CALL(cudaFree(pdev_real[1][0]));
        CUDA_SAFE_CALL(cudaFree(pdev_real[1][1]));
        CUDA_SAFE_CALL(cudaFree(pdev_imag[1][0]));
        CUDA_SAFE_CALL(cudaFree(pdev_imag[1][1]));
        CUDA_SAFE_CALL(cudaFree(dev_external_pot_real[1]));
        CUDA_SAFE_CALL(cudaFree(dev_external_pot_imag[1]));
    }
    else {
        delete [] aH;
        delete [] bH;
        delete [] aV;
		delete [] bV;
        delete [] norm;
    }
    delete [] coupling_const;
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    cublasStatus_t status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        my_abort("CuBLAS shutdown error");
    }
}

void CC2Kernel::run_kernel_on_halo() {
    int inner = 0, horizontal = 0, vertical = 0;
    inner = 0;
    horizontal = 1;
    vertical = 0;
    numBlocks.x = (tile_width  + (BLOCK_X - 2 * halo_x) - 1) / (BLOCK_X - 2 * halo_x);
    numBlocks.y = 2;
    if(imag_time)
        imag_cc2kernel <<< numBlocks, threadsPerBlock, 0, stream1>>>(tile_width, tile_height, 0, 0, halo_x, halo_y, aH[state_index], bH[state_index], aV[state_index], bV[state_index], coupling_const[state_index], alpha_x, alpha_y, dev_external_pot_real[state_index], dev_external_pot_imag[state_index], pdev_real[state_index][sense], pdev_imag[state_index][sense], pdev_real[state_index][1 - sense], pdev_imag[state_index][1 - sense], inner, horizontal, vertical);
    else
        cc2kernel <<< numBlocks, threadsPerBlock, 0, stream1>>>(tile_width, tile_height, 0, 0, halo_x, halo_y, aH[state_index], bH[state_index], aV[state_index], bV[state_index], coupling_const[state_index], alpha_x, alpha_y, dev_external_pot_real[state_index], dev_external_pot_imag[state_index], pdev_real[state_index][sense], pdev_imag[state_index][sense], pdev_real[state_index][1 - sense], pdev_imag[state_index][1 - sense], inner, horizontal, vertical);

    inner = 0;
    horizontal = 0;
    vertical = 1;
    numBlocks.x = 2;
    numBlocks.y = (tile_height  + (BLOCK_Y - 2 * halo_y) - 1) / (BLOCK_Y - 2 * halo_y);
    if(imag_time)
        imag_cc2kernel <<< numBlocks, threadsPerBlock, 0, stream1>>>(tile_width, tile_height, 0, 0, halo_x, halo_y, aH[state_index], bH[state_index], aV[state_index], bV[state_index], coupling_const[state_index], alpha_x, alpha_y, dev_external_pot_real[state_index], dev_external_pot_imag[state_index], pdev_real[state_index][sense], pdev_imag[state_index][sense], pdev_real[state_index][1 - sense], pdev_imag[state_index][1 - sense], inner, horizontal, vertical);
    else
        cc2kernel <<< numBlocks, threadsPerBlock, 0, stream1>>>(tile_width, tile_height, 0, 0, halo_x, halo_y, aH[state_index], bH[state_index], aV[state_index], bV[state_index], coupling_const[state_index], alpha_x, alpha_y, dev_external_pot_real[state_index], dev_external_pot_imag[state_index], pdev_real[state_index][sense], pdev_imag[state_index][sense], pdev_real[state_index][1 - sense], pdev_imag[state_index][1 - sense], inner, horizontal, vertical);
}

void CC2Kernel::run_kernel() {
    int inner = 0, horizontal = 0, vertical = 0;
    inner = 1;
    horizontal = 0;
    vertical = 0;
    numBlocks.x = (tile_width  + (BLOCK_X - 2 * halo_x) - 1) / (BLOCK_X - 2 * halo_x) ;
    numBlocks.y = (tile_height + (BLOCK_Y - 2 * halo_y) - 1) / (BLOCK_Y - 2 * halo_y) - 2;

    if(imag_time)
        imag_cc2kernel <<< numBlocks, threadsPerBlock, 0, stream2>>>(tile_width, tile_height, 0, 0, halo_x, halo_y, aH[state_index], bH[state_index], aV[state_index], bV[state_index], coupling_const[state_index], alpha_x, alpha_y, dev_external_pot_real[state_index], dev_external_pot_imag[state_index], pdev_real[state_index][sense], pdev_imag[state_index][sense], pdev_real[state_index][1 - sense], pdev_imag[state_index][1 - sense], inner, horizontal, vertical);
    else
        cc2kernel <<< numBlocks, threadsPerBlock, 0, stream2>>>(tile_width, tile_height, 0, 0, halo_x, halo_y, aH[state_index], bH[state_index], aV[state_index], bV[state_index], coupling_const[state_index], alpha_x, alpha_y, dev_external_pot_real[state_index], dev_external_pot_imag[state_index], pdev_real[state_index][sense], pdev_imag[state_index][sense], pdev_real[state_index][1 - sense], pdev_imag[state_index][1 - sense], inner, horizontal, vertical);
    sense = 1 - sense;
    CUT_CHECK_ERROR("Kernel error in CC2Kernel::run_kernel");
}


double CC2Kernel::calculate_squared_norm(bool global) const {
    double norm2 = 0., result_imag = 0.;
    cublasStatus_t status = cublasDnrm2(handle, tile_width * tile_height, pdev_real[state_index][sense], 1, &norm2);
    status = cublasDnrm2(handle, tile_width * tile_height, pdev_imag[state_index][sense], 1, &result_imag);
    if (status != CUBLAS_STATUS_SUCCESS) {
        my_abort("CuBLAS error");
    }
    norm2 = norm2 * norm2 + result_imag * result_imag;
#ifdef HAVE_MPI
    if (global) {
        int nProcs = 1;
        MPI_Comm_size(cartcomm, &nProcs);
        double *sums = new double[nProcs];
        MPI_Allgather(&norm2, 1, MPI_DOUBLE, sums, 1, MPI_DOUBLE, cartcomm);
        norm2 = 0.;
        for(int i = 0; i < nProcs; i++)
            norm2 += sums[i];
        delete [] sums;
    }
#endif
    return norm2 * delta_x * delta_y;
}

void CC2Kernel::wait_for_completion() {
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //normalization
    if (imag_time && norm[state_index] != 0) {
        double tot_sum = calculate_squared_norm(true);
        double inverse_norm = 1. / sqrt(tot_sum / norm[state_index]);
        cublasStatus_t status = cublasDscal(handle, tile_width * tile_height, &inverse_norm, pdev_real[state_index][sense], 1);
        status = cublasDscal(handle, tile_width * tile_height, &inverse_norm, pdev_imag[state_index][sense], 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            my_abort("CuBLAS error");
        }
    }
    if (two_wavefunctions) {
        if (state_index == 0) {
            sense = 1 - sense;
        }
        state_index = 1 - state_index;
    }

}

void CC2Kernel::normalization() {
    if(imag_time && (coupling_const[3] != 0 || coupling_const[4] != 0)) {
        state_index = 0;
        double tot_sum_a = calculate_squared_norm(true);
        state_index = 1;
        double tot_sum_b = calculate_squared_norm(true);
        state_index = 0;
        double _norm = sqrt((tot_sum_a + tot_sum_b) / tot_norm);
        double inverse_norm = 1. / _norm;
        cublasStatus_t status = cublasDscal(handle, tile_width * tile_height, &inverse_norm, pdev_real[0][sense], 1);
        status = cublasDscal(handle, tile_width * tile_height, &inverse_norm, pdev_imag[0][sense], 1);
        if(p_real[1] != NULL) {
            status = cublasDscal(handle, tile_width * tile_height, &inverse_norm, pdev_real[1][sense], 1);
            status = cublasDscal(handle, tile_width * tile_height, &inverse_norm, pdev_imag[1][sense], 1);
        }
        norm[0] = tot_sum_a / (tot_sum_a + tot_sum_b) * tot_norm;
        norm[1] = tot_sum_b / (tot_sum_a + tot_sum_b) * tot_norm;
    }
}

void CC2Kernel::cpy_first_positive_to_first_negative() {

}

void CC2Kernel::get_sample(size_t dest_stride, size_t x, size_t y,
                           size_t width, size_t height,
                           double *dest_real, double *dest_imag,
                           double *dest_real2, double *dest_imag2) const {
    assert(x < tile_width);
    assert(y < tile_height);
    assert(x + width <= tile_width);
    assert(y + height <= tile_height);
    CUDA_SAFE_CALL(cudaMemcpy2D(dest_real, dest_stride * sizeof(double), &(pdev_real[0][sense][y * tile_width + x]), tile_width * sizeof(double), width * sizeof(double), height, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy2D(dest_imag, dest_stride * sizeof(double), &(pdev_imag[0][sense][y * tile_width + x]), tile_width * sizeof(double), width * sizeof(double), height, cudaMemcpyDeviceToHost));
    if (dest_real2 != 0) {
        CUDA_SAFE_CALL(cudaMemcpy2D(dest_real2, dest_stride * sizeof(double), &(pdev_real[1][sense][y * tile_width + x]), tile_width * sizeof(double), width * sizeof(double), height, cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy2D(dest_imag2, dest_stride * sizeof(double), &(pdev_imag[1][sense][y * tile_width + x]), tile_width * sizeof(double), width * sizeof(double), height, cudaMemcpyDeviceToHost));
    }
}

void CC2Kernel::start_halo_exchange() {

}

void CC2Kernel::finish_halo_exchange() {
#ifdef HAVE_MPI
    MPI_Request req[8];
    MPI_Status statuses[8];
#endif
    int offset = 0;

    // Halo copy: LEFT/RIGHT
    int height = inner_end_y - inner_start_y;	// The vertical halo in rows
    int width = halo_x;	// The number of columns of the matrix
    int stride = tile_width;	// The combined width of the matrix with the halo
    offset = (inner_start_y - start_y) * tile_width + inner_end_x - halo_x - start_x;
    CUDA_SAFE_CALL(cudaMemcpy2DAsync(right_real_send, width * sizeof(double), &(pdev_real[state_index][sense][offset]), stride * sizeof(double), width * sizeof(double), height, cudaMemcpyDeviceToHost, stream1));
    CUDA_SAFE_CALL(cudaMemcpy2DAsync(right_imag_send, width * sizeof(double), &(pdev_imag[state_index][sense][offset]), stride * sizeof(double), width * sizeof(double), height, cudaMemcpyDeviceToHost, stream1));
    offset = (inner_start_y - start_y) * tile_width + halo_x;
    CUDA_SAFE_CALL(cudaMemcpy2DAsync(left_real_send, width * sizeof(double), &(pdev_real[state_index][sense][offset]), stride * sizeof(double), width * sizeof(double), height, cudaMemcpyDeviceToHost, stream1));
    CUDA_SAFE_CALL(cudaMemcpy2DAsync(left_imag_send, width * sizeof(double), &(pdev_imag[state_index][sense][offset]), stride * sizeof(double), width * sizeof(double), height, cudaMemcpyDeviceToHost, stream1));

    // Halo copy: UP/DOWN
    height = halo_y;	// The vertical halo in rows
    width = tile_width;	// The number of columns of the matrix
    stride = tile_width;	// The combined width of the matrix with the halo

    offset = (inner_end_y - halo_y - start_y) * tile_width;
    CUDA_SAFE_CALL(cudaMemcpy2DAsync(bottom_real_send, width * sizeof(double), &(pdev_real[state_index][sense][offset]), stride * sizeof(double), width * sizeof(double), height, cudaMemcpyDeviceToHost, stream1));
    CUDA_SAFE_CALL(cudaMemcpy2DAsync(bottom_imag_send, width * sizeof(double), &(pdev_imag[state_index][sense][offset]), stride * sizeof(double), width * sizeof(double), height, cudaMemcpyDeviceToHost, stream1));
    offset = halo_y * tile_width;
    CUDA_SAFE_CALL(cudaMemcpy2DAsync(top_real_send, width * sizeof(double), &(pdev_real[state_index][sense][offset]), stride * sizeof(double), width * sizeof(double), height, cudaMemcpyDeviceToHost, stream1));
    CUDA_SAFE_CALL(cudaMemcpy2DAsync(top_imag_send, width * sizeof(double), &(pdev_imag[state_index][sense][offset]), stride * sizeof(double), width * sizeof(double), height, cudaMemcpyDeviceToHost, stream1));

    cudaStreamSynchronize(stream1);


    // Halo exchange: LEFT/RIGHT
    height = inner_end_y - inner_start_y;	// The vertical halo in rows
    width = halo_x;	// The number of columns of the matrix
    stride = tile_width;	// The combined width of the matrix with the halo

#ifdef HAVE_MPI
    MPI_Irecv(left_real_receive, height * width, MPI_DOUBLE, neighbors[LEFT], 1, cartcomm, req);
    MPI_Irecv(left_imag_receive, height * width, MPI_DOUBLE, neighbors[LEFT], 2, cartcomm, req + 1);
    MPI_Irecv(right_real_receive, height * width, MPI_DOUBLE, neighbors[RIGHT], 3, cartcomm, req + 2);
    MPI_Irecv(right_imag_receive, height * width, MPI_DOUBLE, neighbors[RIGHT], 4, cartcomm, req + 3);

    offset = (inner_start_y - start_y) * tile_width + inner_end_x - halo_x - start_x;
    MPI_Isend(right_real_send, height * width, MPI_DOUBLE, neighbors[RIGHT], 1, cartcomm, req + 4);
    MPI_Isend(right_imag_send, height * width, MPI_DOUBLE, neighbors[RIGHT], 2, cartcomm, req + 5);

    offset = (inner_start_y - start_y) * tile_width + halo_x;
    MPI_Isend(left_real_send, height * width, MPI_DOUBLE, neighbors[LEFT], 3, cartcomm, req + 6);
    MPI_Isend(left_imag_send, height * width, MPI_DOUBLE, neighbors[LEFT], 4, cartcomm, req + 7);

    MPI_Waitall(8, req, statuses);
#else
    if(periods[1] != 0) {
        memcpy2D(left_real_receive, height * width * sizeof(double), right_real_send, height * width * sizeof(double), height * width * sizeof(double), 1);
        memcpy2D(left_imag_receive, height * width * sizeof(double), right_imag_send, height * width * sizeof(double), height * width * sizeof(double), 1);
        memcpy2D(right_real_receive, height * width * sizeof(double) , left_real_send, height * width * sizeof(double), height * width * sizeof(double), 1);
        memcpy2D(right_imag_receive, height * width * sizeof(double) , left_imag_send, height * width * sizeof(double), height * width * sizeof(double), 1);
    }
#endif

    // Halo exchange: UP/DOWN
    height = halo_y;	// The vertical halo in rows
    width = tile_width;	// The number of columns of the matrix
    stride = tile_width;	// The combined width of the matrix with the halo

#ifdef HAVE_MPI
    MPI_Irecv(top_real_receive, height * width, MPI_DOUBLE, neighbors[UP], 1, cartcomm, req);
    MPI_Irecv(top_imag_receive, height * width, MPI_DOUBLE, neighbors[UP], 2, cartcomm, req + 1);
    MPI_Irecv(bottom_real_receive, height * width, MPI_DOUBLE, neighbors[DOWN], 3, cartcomm, req + 2);
    MPI_Irecv(bottom_imag_receive, height * width, MPI_DOUBLE, neighbors[DOWN], 4, cartcomm, req + 3);

    offset = (inner_end_y - halo_y - start_y) * tile_width;
    MPI_Isend(bottom_real_send, height * width, MPI_DOUBLE, neighbors[DOWN], 1, cartcomm, req + 4);
    MPI_Isend(bottom_imag_send, height * width, MPI_DOUBLE, neighbors[DOWN], 2, cartcomm, req + 5);

    offset = halo_y * tile_width;
    MPI_Isend(top_real_send, height * width, MPI_DOUBLE, neighbors[UP], 3, cartcomm, req + 6);
    MPI_Isend(top_imag_send, height * width, MPI_DOUBLE, neighbors[UP], 4, cartcomm, req + 7);

    MPI_Waitall(8, req, statuses);
    bool MPI = true;
#else
    if(periods[0] != 0) {
        memcpy2D(top_real_receive, height * width * sizeof(double), bottom_real_send, height * width  * sizeof(double), height * width * sizeof(double), 1);
        memcpy2D(top_imag_receive, height * width * sizeof(double), bottom_imag_send, height * width * sizeof(double), height * width * sizeof(double), 1);
        memcpy2D(bottom_real_receive, height * width * sizeof(double) , top_real_send, height * width * sizeof(double) , height * width * sizeof(double), 1);
        memcpy2D(bottom_imag_receive, height * width  * sizeof(double), top_imag_send, height * width * sizeof(double) , height * width * sizeof(double), 1);
    }
    bool MPI = false;
#endif
    // Copy back the halos to the GPU memory

    height = inner_end_y - inner_start_y;	// The vertical halo in rows
    width = halo_x;	// The number of columns of the matrix
    stride = tile_width;	// The combined width of the matrix with the halo

    if(periods[1] != 0 || MPI) {
        offset = (inner_start_y - start_y) * tile_width;
        if (neighbors[LEFT] >= 0) {
            CUDA_SAFE_CALL(cudaMemcpy2DAsync(&(pdev_real[state_index][sense][offset]), stride * sizeof(double), left_real_receive, width * sizeof(double), width * sizeof(double), height, cudaMemcpyHostToDevice, stream1));
            CUDA_SAFE_CALL(cudaMemcpy2DAsync(&(pdev_imag[state_index][sense][offset]), stride * sizeof(double), left_imag_receive, width * sizeof(double), width * sizeof(double), height, cudaMemcpyHostToDevice, stream1));
        }
        offset = (inner_start_y - start_y) * tile_width + inner_end_x - start_x;
        if (neighbors[RIGHT] >= 0) {
            CUDA_SAFE_CALL(cudaMemcpy2DAsync(&(pdev_real[state_index][sense][offset]), stride * sizeof(double), right_real_receive, width * sizeof(double), width * sizeof(double), height, cudaMemcpyHostToDevice, stream1));
            CUDA_SAFE_CALL(cudaMemcpy2DAsync(&(pdev_imag[state_index][sense][offset]), stride * sizeof(double), right_imag_receive, width * sizeof(double), width * sizeof(double), height, cudaMemcpyHostToDevice, stream1));
        }
    }

    height = halo_y;	// The vertical halo in rows
    width = tile_width;	// The number of columns of the matrix
    stride = tile_width;	// The combined width of the matrix with the halo

    if(periods[0] != 0 || MPI) {
        offset = 0;
        if (neighbors[UP] >= 0) {
            CUDA_SAFE_CALL(cudaMemcpy2DAsync(&(pdev_real[state_index][sense][offset]), stride * sizeof(double), top_real_receive, width * sizeof(double), width * sizeof(double), height, cudaMemcpyHostToDevice, stream1));
            CUDA_SAFE_CALL(cudaMemcpy2DAsync(&(pdev_imag[state_index][sense][offset]), stride * sizeof(double), top_imag_receive, width * sizeof(double), width * sizeof(double), height, cudaMemcpyHostToDevice, stream1));
        }

        offset = (inner_end_y - start_y) * tile_width;
        if (neighbors[DOWN] >= 0) {
            CUDA_SAFE_CALL(cudaMemcpy2DAsync(&(pdev_real[state_index][sense][offset]), stride * sizeof(double), bottom_real_receive, width * sizeof(double), width * sizeof(double), height, cudaMemcpyHostToDevice, stream1));
            CUDA_SAFE_CALL(cudaMemcpy2DAsync(&(pdev_imag[state_index][sense][offset]), stride * sizeof(double), bottom_imag_receive, width * sizeof(double), width * sizeof(double), height, cudaMemcpyHostToDevice, stream1));
        }
    }
}

void CC2Kernel::rabi_coupling(double var, double delta_t) {
    double norm_omega = sqrt(coupling_const[3] * coupling_const[3] + coupling_const[4] * coupling_const[4]);
    double cc, cs_r, cs_i;
    if(imag_time) {
        cc = cosh(- delta_t * var * norm_omega);
        if (norm_omega == 0) {
            cs_r = 0;
            cs_i = 0;
        }
        else {
            cs_r = coupling_const[3] / norm_omega * sinh(- delta_t * var * norm_omega);
            cs_i = coupling_const[4] / norm_omega * sinh(- delta_t * var * norm_omega);
        }
        gpu_rabi_coupling_imag <<< tile_height, tile_width>>>(tile_width, tile_height, cc, cs_r, cs_i,
                pdev_real[0][sense], pdev_imag[0][sense],
                pdev_real[1][sense], pdev_imag[1][sense]);
    }
    else {
        cc = cos(- delta_t * var * norm_omega);
        if (norm_omega == 0) {
            cs_r = 0;
            cs_i = 0;
        }
        else {
            cs_r = coupling_const[3] / norm_omega * sin(- delta_t * var * norm_omega);
            cs_i = coupling_const[4] / norm_omega * sin(- delta_t * var * norm_omega);
        }
        gpu_rabi_coupling_real <<< tile_height, tile_width>>>(tile_width, tile_height, cc, cs_r, cs_i,
                pdev_real[0][sense], pdev_imag[0][sense],
                pdev_real[1][sense], pdev_imag[1][sense]);
    }
}
