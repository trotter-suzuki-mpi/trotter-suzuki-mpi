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
#include <iostream>
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

//REAL TIME functions

template<int BLOCK_WIDTH, int BLOCK_HEIGHT, int BACKWARDS>
inline __device__ void gpu_kernel_vertical(double a, double b, int tile_height, double &cell_r, double &cell_i, int kx, int ky, int py, double rl[BLOCK_HEIGHT][BLOCK_WIDTH], double im[BLOCK_HEIGHT][BLOCK_WIDTH]) {
    double peer_r;
    double peer_i;

    const int ky_peer = ky + 1 - 2 * BACKWARDS;
    if (py >= BACKWARDS && py < tile_height - 1 + BACKWARDS && ky >= BACKWARDS && ky < BLOCK_HEIGHT - 1 + BACKWARDS) {
        peer_r = rl[ky_peer][kx];
        peer_i = im[ky_peer][kx];
#ifndef DISABLE_FMA
        rl[ky_peer][kx] = a * peer_r - b * cell_i;
        im[ky_peer][kx] = a * peer_i + b * cell_r;
        cell_r = a * cell_r - b * peer_i;
        cell_i = a * cell_i + b * peer_r;
#else
        // NOTE: disabling FMA has worse precision and performance
        //       use only for exact implementation verification against CPU results
        rl[ky_peer][kx] = __dadd_rn(a * peer_r, - b * cell_i);
        im[ky_peer][kx] = __dadd_rn(a * peer_i, b * cell_r);
        cell_r = __dadd_rn(a * cell_r, - b * peer_i);
        cell_i = __dadd_rn(a * cell_i, b * peer_r);
#endif
    }
}


template<int BLOCK_WIDTH, int BLOCK_HEIGHT, int BACKWARDS>
static  inline __device__ void gpu_kernel_horizontal(double a, double b,  int tile_width, double &cell_r, double &cell_i, int kx, int ky, int px, double rl[BLOCK_HEIGHT][BLOCK_WIDTH], double im[BLOCK_HEIGHT][BLOCK_WIDTH]) {
    double peer_r;
    double peer_i;

    const int kx_peer = kx + 1 - 2 * BACKWARDS;
    if (px >= BACKWARDS && px < tile_width - 1 + BACKWARDS && kx >= BACKWARDS && kx < BLOCK_WIDTH - 1 + BACKWARDS) {
        peer_r = rl[ky][kx_peer];
        peer_i = im[ky][kx_peer];
#ifndef DISABLE_FMA
        rl[ky][kx_peer] = a * peer_r - b * cell_i;
        im[ky][kx_peer] = a * peer_i + b * cell_r;
        cell_r = a * cell_r - b * peer_i;
        cell_i = a * cell_i + b * peer_r;
#else
        // NOTE: disabling FMA has worse precision and performance
        //       use only for exact implementation verification against CPU results
        rl[ky][kx_peer] = __dadd_rn(a * peer_r, - b * cell_i);
        im[ky][kx_peer] = __dadd_rn(a * peer_i, b * cell_r);
        cell_r = __dadd_rn(a * cell_r, - b * peer_i);
        cell_i = __dadd_rn(a * cell_i, b * peer_r);
#endif
    }
}

template<int BLOCK_WIDTH, int BLOCK_HEIGHT>
static  inline __device__ void gpu_kernel_potential(int tile_width, int tile_height, double &cell_r, double &cell_i,
        int kx, int ky, int px, int py,
        double rl[BLOCK_HEIGHT][BLOCK_WIDTH], double im[BLOCK_HEIGHT][BLOCK_WIDTH],
        double pot_r[BLOCK_HEIGHT][BLOCK_WIDTH], double pot_i[BLOCK_HEIGHT][BLOCK_WIDTH], double coupling_a) {
    double tmp;
    double peer_r;
    double peer_i;
    double pot_cell_r, pot_cell_i, pot_peer_r, pot_peer_i;
    double c_cos;
    double c_sin;

    const int ky_peer = ky + 1 - 2 * (kx % 2);
    if(ky >= 0 && ky < BLOCK_HEIGHT && ky_peer >= 0 && ky_peer < BLOCK_HEIGHT && kx >= 0 && kx < BLOCK_WIDTH) {
        pot_cell_r = pot_r[ky][kx];
        pot_cell_i = pot_i[ky][kx];
        pot_peer_r = pot_r[ky_peer][kx];
        pot_peer_i = pot_i[ky_peer][kx];

        peer_r = rl[ky_peer][kx];
        peer_i = im[ky_peer][kx];

#ifndef DISABLE_FMA
        tmp = cell_r * cell_r + cell_i * cell_i;
        c_cos = cos(coupling_a * tmp);
        c_sin = sin(coupling_a * tmp);

        tmp = cell_r;
        cell_r = pot_cell_r * tmp - pot_cell_i * cell_i;
        cell_i = pot_cell_r * cell_i + pot_cell_i * tmp;

        tmp = cell_r;
        cell_r = c_cos * cell_r + c_sin * cell_i;
        cell_i = c_cos * cell_i - c_sin * cell_r;

        tmp = peer_r * peer_r + peer_i * peer_i;
        c_cos = cos(coupling_a * tmp);
        c_sin = sin(coupling_a * tmp);

        tmp = peer_r;
        peer_r = pot_peer_r * tmp - pot_peer_i * peer_i;
        peer_i = pot_peer_r * peer_i + pot_peer_i * tmp;

        rl[ky_peer][kx] = c_cos * peer_r + c_sin * peer_i;
        im[ky_peer][kx] = c_cos * peer_i - c_sin * peer_r;
#else
        // NOTE: disabling FMA has worse precision and performance
        // use only for exact implementation verification against CPU results
        tmp = __dadd_rn(cell_r * cell_r, cell_i * cell_i);
        c_cos = cos(coupling_a * tmp);
        c_sin = sin(coupling_a * tmp);

        tmp = cell_r;
        cell_r = __dadd_rn(pot_cell_r * tmp, - pot_cell_i * cell_i);
        cell_i = __dadd_rn(pot_cell_r * cell_i, pot_cell_i * tmp);

        tmp = cell_r;
        cell_r = __dadd_rn(c_cos * cell_r, c_sin * cell_i);
        cell_i = __dadd_rn(c_cos * cell_i, - c_sin * cell_i);

        tmp = __dadd_rn(peer_r * peer_r, peer_i * peer_i);
        c_cos = cos(coupling_a * tmp);
        c_sin = sin(coupling_a * tmp);

        tmp = peer_r;
        peer_r = __dadd_rn(pot_peer_r * tmp, - pot_peer_i * peer_i);
        peer_i = __dadd_rn(pot_peer_r * peer_i, pot_peer_i * tmp);

        rl[ky_peer][kx] = __dadd_rn(c_cos * peer_r, c_sin * peer_i);
        im[ky_peer][kx] = __dadd_rn(c_cos * peer_i, - c_sin * peer_r);
#endif
    }
}

__launch_bounds__(BLOCK_X * STRIDE_Y)
__global__ void cc2kernel(size_t tile_width, size_t tile_height, size_t offset_x, size_t offset_y, size_t halo_x, size_t halo_y,
                          double a, double b, double coupling_a, double alpha_x, double alpha_y, const double * __restrict__ external_pot_real, const double * __restrict__ external_pot_imag,
                          const double * __restrict__ p_real, const double * __restrict__ p_imag,
                          double * __restrict__ p2_real, double * __restrict__ p2_imag,
                          int inner, int horizontal, int vertical) {

    __shared__ double rl[BLOCK_Y][BLOCK_X];
    __shared__ double im[BLOCK_Y][BLOCK_X];
    __shared__ double pot_r[BLOCK_Y][BLOCK_X];
    __shared__ double pot_i[BLOCK_Y][BLOCK_X];

    int blockIdxx = inner * (blockIdx.x + 1) + horizontal * (blockIdx.x) + vertical * (blockIdx.x * ((tile_width + (BLOCK_X - 2 * halo_x) - 1) / (BLOCK_X - 2 * halo_x) - 1));
    int blockIdxy = inner * (blockIdx.y + 1) + horizontal * (blockIdx.y * ((tile_height + (BLOCK_Y - 2 * halo_y) - 1) / (BLOCK_Y - 2 * halo_y) - 1)) + vertical * (blockIdx.y + 1);

    // The offsets are used by the hybrid kernel
    int px = offset_x + blockIdxx * (BLOCK_X - 2 * halo_x) + threadIdx.x - halo_x;
    int py = offset_y + blockIdxy * (BLOCK_Y - 2 * halo_y) + threadIdx.y - halo_y;

    // Read block from global into shared memory (state and potential)
    if (px >= 0 && px < tile_width) {
#pragma unroll
        for (int i = 0, pidx = py * tile_width + px; i < BLOCK_Y / STRIDE_Y; ++i, pidx += STRIDE_Y * tile_width) {
            if (py + i * STRIDE_Y >= 0 && py + i * STRIDE_Y < tile_height) {
                rl[threadIdx.y + i * STRIDE_Y][threadIdx.x] = p_real[pidx];
                im[threadIdx.y + i * STRIDE_Y][threadIdx.x] = p_imag[pidx];
                pot_r[threadIdx.y + i * STRIDE_Y][threadIdx.x] = external_pot_real[pidx];
                pot_i[threadIdx.y + i * STRIDE_Y][threadIdx.x] = external_pot_imag[pidx];
            }
        }
    }

    __syncthreads();

    // Place threads along the black cells of a checkerboard pattern
    int sx = threadIdx.x;
    int sy;
    if ((halo_x) % 2 == (halo_y) % 2) {
        sy = 2 * threadIdx.y + threadIdx.x % 2;
    }
    else {
        sy = 2 * threadIdx.y + 1 - threadIdx.x % 2;
    }

    // global y coordinate of the thread on the checkerboard (px remains the same)
    // used for range checks
    int checkerboard_py = offset_y + blockIdxy * (BLOCK_Y - 2 * halo_y) + sy - halo_y;

    // Keep the fixed black cells on registers, reds are updated in shared memory
    double cell_r[BLOCK_Y / (STRIDE_Y * 2)];
    double cell_i[BLOCK_Y / (STRIDE_Y * 2)];

#pragma unroll
    // Read black cells to registers
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        cell_r[part] = rl[sy + part * 2 * STRIDE_Y][sx];
        cell_i[part] = im[sy + part * 2 * STRIDE_Y][sx];
    }

    // 12344321
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        gpu_kernel_vertical<BLOCK_X, BLOCK_Y, 0>(a, b, tile_height, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, checkerboard_py + part * 2 * STRIDE_Y, rl, im);
    }
    __syncthreads();
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        gpu_kernel_horizontal<BLOCK_X, BLOCK_Y, 0>(a, b, tile_width, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, px, rl, im);
    }
    __syncthreads();
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        gpu_kernel_vertical<BLOCK_X, BLOCK_Y, 1>(a, b, tile_height, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, checkerboard_py + part * 2 * STRIDE_Y, rl, im);
    }
    __syncthreads();
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        gpu_kernel_horizontal<BLOCK_X, BLOCK_Y, 1>(a, b, tile_width, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, px, rl, im);
    }
    __syncthreads();
//potential
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        gpu_kernel_potential<BLOCK_X, BLOCK_Y>(tile_width, tile_height, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, px, checkerboard_py + part * 2 * STRIDE_Y, rl, im, pot_r, pot_i, coupling_a);
    }
    __syncthreads();

    if (alpha_x != 0. && alpha_y != 0.) {
        // TODO: Rotation kernel should come here
    }

#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        gpu_kernel_horizontal<BLOCK_X, BLOCK_Y, 1>(a, b, tile_width, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, px, rl, im);
    }
    __syncthreads();
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        gpu_kernel_vertical<BLOCK_X, BLOCK_Y, 1>(a, b, tile_height, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, checkerboard_py + part * 2 * STRIDE_Y, rl, im);
    }
    __syncthreads();
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        gpu_kernel_horizontal<BLOCK_X, BLOCK_Y, 0>(a, b, tile_width, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, px, rl, im);
    }
    __syncthreads();
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        gpu_kernel_vertical<BLOCK_X, BLOCK_Y, 0>(a, b, tile_height, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, checkerboard_py + part * 2 * STRIDE_Y, rl, im);
    }
    __syncthreads();


    // Write black cells in registers to shared memory
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        rl[sy + part * 2 * STRIDE_Y][sx] = cell_r[part];
        im[sy + part * 2 * STRIDE_Y][sx] = cell_i[part];
    }
    __syncthreads();

    // discard the halo and copy results from shared to global memory
    sx = threadIdx.x + halo_x;
    sy = threadIdx.y + halo_y;
    px += halo_x;
    py += halo_y;
    if (sx < BLOCK_X - halo_x && px < tile_width) {
#pragma unroll
        for (int i = 0, pidx = py * tile_width + px; i < BLOCK_Y / STRIDE_Y; ++i, pidx += STRIDE_Y * tile_width) {
            if (sy + i * STRIDE_Y < BLOCK_Y - halo_y && py + i * STRIDE_Y < tile_height) {
                p2_real[pidx] = rl[sy + i * STRIDE_Y][sx];
                p2_imag[pidx] = im[sy + i * STRIDE_Y][sx];
            }
        }
    }
}

//  IMAGINARY TIME functions

template<int BLOCK_WIDTH, int BLOCK_HEIGHT, int BACKWARDS>
inline __device__ void imag_gpu_kernel_vertical(double a, double b, int tile_height, double &cell_r, double &cell_i, int kx, int ky, int py, double rl[BLOCK_HEIGHT][BLOCK_WIDTH], double im[BLOCK_HEIGHT][BLOCK_WIDTH]) {
    double peer_r;
    double peer_i;

    const int ky_peer = ky + 1 - 2 * BACKWARDS;
    if (py >= BACKWARDS && py < tile_height - 1 + BACKWARDS && ky >= BACKWARDS && ky < BLOCK_HEIGHT - 1 + BACKWARDS) {
        peer_r = rl[ky_peer][kx];
        peer_i = im[ky_peer][kx];
#ifndef DISABLE_FMA
        rl[ky_peer][kx] = a * peer_r + b * cell_r;
        im[ky_peer][kx] = a * peer_i + b * cell_i;
        cell_r = a * cell_r + b * peer_r;
        cell_i = a * cell_i + b * peer_i;
#else
        // NOTE: disabling FMA has worse precision and performance
        //       use only for exact implementation verification against CPU results
        rl[ky_peer][kx] = __dadd_rn(a * peer_r, b * cell_r);
        im[ky_peer][kx] = __dadd_rn(a * peer_i, b * cell_i);
        cell_r = __dadd_rn(a * cell_r, b * peer_r);
        cell_i = __dadd_rn(a * cell_i, b * peer_i);
#endif
    }
}


template<int BLOCK_WIDTH, int BLOCK_HEIGHT, int BACKWARDS>
static  inline __device__ void imag_gpu_kernel_horizontal(double a, double b,  int tile_width, double &cell_r, double &cell_i, int kx, int ky, int px, double rl[BLOCK_HEIGHT][BLOCK_WIDTH], double im[BLOCK_HEIGHT][BLOCK_WIDTH]) {
    double peer_r;
    double peer_i;

    const int kx_peer = kx + 1 - 2 * BACKWARDS;
    if (px >= BACKWARDS && px < tile_width - 1 + BACKWARDS && kx >= BACKWARDS && kx < BLOCK_WIDTH - 1 + BACKWARDS) {
        peer_r = rl[ky][kx_peer];
        peer_i = im[ky][kx_peer];
#ifndef DISABLE_FMA
        rl[ky][kx_peer] = a * peer_r + b * cell_r;
        im[ky][kx_peer] = a * peer_i + b * cell_i;
        cell_r = a * cell_r + b * peer_r;
        cell_i = a * cell_i + b * peer_i;
#else
        // NOTE: disabling FMA has worse precision and performance
        //       use only for exact implementation verification against CPU results
        rl[ky][kx_peer] = __dadd_rn(a * peer_r, b * cell_r);
        im[ky][kx_peer] = __dadd_rn(a * peer_i, b * cell_i);
        cell_r = __dadd_rn(a * cell_r, b * peer_r);
        cell_i = __dadd_rn(a * cell_i, b * peer_i);
#endif
    }
}

template<int BLOCK_WIDTH, int BLOCK_HEIGHT>
static  inline __device__ void imag_gpu_kernel_potential(int tile_width, int tile_height, double &cell_r, double &cell_i,
        int kx, int ky, int px, int py,
        double rl[BLOCK_HEIGHT][BLOCK_WIDTH], double im[BLOCK_HEIGHT][BLOCK_WIDTH],
        double pot_r[BLOCK_HEIGHT][BLOCK_WIDTH], double coupling_a) {
    double peer_r;
    double peer_i;
    double pot_cell_r, pot_peer_r;
    double tmp;
    const int ky_peer = ky + 1 - 2 * (kx % 2);
    if(ky >= 0 && ky < BLOCK_HEIGHT && ky_peer >= 0 && ky_peer < BLOCK_HEIGHT && kx >= 0 && kx < BLOCK_WIDTH) {
        pot_cell_r = pot_r[ky][kx];
        pot_peer_r = pot_r[ky_peer][kx];
        peer_r = rl[ky_peer][kx];
        peer_i = im[ky_peer][kx];
        tmp = exp(-1. * coupling_a * (cell_r * cell_r + cell_i * cell_i));
        cell_r = tmp * pot_cell_r * cell_r;
        cell_i = tmp * pot_cell_r * cell_i;
        tmp = exp(-1. * coupling_a * (peer_r * peer_r + peer_i * peer_i));
        rl[ky_peer][kx] = tmp * pot_peer_r * peer_r;
        im[ky_peer][kx] = tmp * pot_peer_r * peer_i;
    }
}

__launch_bounds__(BLOCK_X * STRIDE_Y)
__global__ void imag_cc2kernel(size_t tile_width, size_t tile_height, size_t offset_x, size_t offset_y, size_t halo_x, size_t halo_y,
                               double a, double b, double coupling_a, double alpha_x, double alpha_y, const double * __restrict__ external_pot_real, const double * __restrict__ external_pot_imag,
                               const double * __restrict__ p_real, const double * __restrict__ p_imag,
                               double * __restrict__ p2_real, double * __restrict__ p2_imag,
                               int inner, int horizontal, int vertical) {

    __shared__ double rl[BLOCK_Y][BLOCK_X];
    __shared__ double im[BLOCK_Y][BLOCK_X];
    __shared__ double pot_r[BLOCK_Y][BLOCK_X];

    int blockIdxx = inner * (blockIdx.x + 1) + horizontal * (blockIdx.x) + vertical * (blockIdx.x * ((tile_width + (BLOCK_X - 2 * halo_x) - 1) / (BLOCK_X - 2 * halo_x) - 1));
    int blockIdxy = inner * (blockIdx.y + 1) + horizontal * (blockIdx.y * ((tile_height + (BLOCK_Y - 2 * halo_y) - 1) / (BLOCK_Y - 2 * halo_y) - 1)) + vertical * (blockIdx.y + 1);

    // The offsets are used by the hybrid kernel
    int px = offset_x + blockIdxx * (BLOCK_X - 2 * halo_x) + threadIdx.x - halo_x;
    int py = offset_y + blockIdxy * (BLOCK_Y - 2 * halo_y) + threadIdx.y - halo_y;

    // Read block from global into shared memory (state and potential)
    if (px >= 0 && px < tile_width) {
#pragma unroll
        for (int i = 0, pidx = py * tile_width + px; i < BLOCK_Y / STRIDE_Y; ++i, pidx += STRIDE_Y * tile_width) {
            if (py + i * STRIDE_Y >= 0 && py + i * STRIDE_Y < tile_height) {
                rl[threadIdx.y + i * STRIDE_Y][threadIdx.x] = p_real[pidx];
                im[threadIdx.y + i * STRIDE_Y][threadIdx.x] = p_imag[pidx];
                pot_r[threadIdx.y + i * STRIDE_Y][threadIdx.x] = external_pot_real[pidx];
            }
        }
    }

    __syncthreads();

    // Place threads along the black cells of a checkerboard pattern
    int sx = threadIdx.x;
    int sy;
    if ((halo_x) % 2 == (halo_y) % 2) {
        sy = 2 * threadIdx.y + threadIdx.x % 2;
    }
    else {
        sy = 2 * threadIdx.y + 1 - threadIdx.x % 2;
    }

    // global y coordinate of the thread on the checkerboard (px remains the same)
    // used for range checks
    int checkerboard_py = offset_y + blockIdxy * (BLOCK_Y - 2 * halo_y) + sy - halo_y;

    // Keep the fixed black cells on registers, reds are updated in shared memory
    double cell_r[BLOCK_Y / (STRIDE_Y * 2)];
    double cell_i[BLOCK_Y / (STRIDE_Y * 2)];

#pragma unroll
    // Read black cells to registers
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        cell_r[part] = rl[sy + part * 2 * STRIDE_Y][sx];
        cell_i[part] = im[sy + part * 2 * STRIDE_Y][sx];
    }

    // 12344321
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        imag_gpu_kernel_vertical<BLOCK_X, BLOCK_Y, 0>(a, b, tile_height, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, checkerboard_py + part * 2 * STRIDE_Y, rl, im);
    }
    __syncthreads();
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        imag_gpu_kernel_horizontal<BLOCK_X, BLOCK_Y, 0>(a, b, tile_width, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, px, rl, im);
    }
    __syncthreads();
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        imag_gpu_kernel_vertical<BLOCK_X, BLOCK_Y, 1>(a, b, tile_height, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, checkerboard_py + part * 2 * STRIDE_Y, rl, im);
    }
    __syncthreads();
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        imag_gpu_kernel_horizontal<BLOCK_X, BLOCK_Y, 1>(a, b, tile_width, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, px, rl, im);
    }
    __syncthreads();
//potential
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        imag_gpu_kernel_potential<BLOCK_X, BLOCK_Y>(tile_width, tile_height, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, px, checkerboard_py + part * 2 * STRIDE_Y, rl, im, pot_r, coupling_a);
    }
    __syncthreads();

    if (alpha_x != 0. && alpha_y != 0.) {
        // TODO: Rotation kernel should come here
    }

#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        imag_gpu_kernel_horizontal<BLOCK_X, BLOCK_Y, 1>(a, b, tile_width, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, px, rl, im);
    }
    __syncthreads();
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        imag_gpu_kernel_vertical<BLOCK_X, BLOCK_Y, 1>(a, b, tile_height, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, checkerboard_py + part * 2 * STRIDE_Y, rl, im);
    }
    __syncthreads();
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        imag_gpu_kernel_horizontal<BLOCK_X, BLOCK_Y, 0>(a, b, tile_width, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, px, rl, im);
    }
    __syncthreads();
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        imag_gpu_kernel_vertical<BLOCK_X, BLOCK_Y, 0>(a, b, tile_height, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, checkerboard_py + part * 2 * STRIDE_Y, rl, im);
    }
    __syncthreads();


    // Write black cells in registers to shared memory
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        rl[sy + part * 2 * STRIDE_Y][sx] = cell_r[part];
        im[sy + part * 2 * STRIDE_Y][sx] = cell_i[part];
    }
    __syncthreads();

    // discard the halo and copy results from shared to global memory
    sx = threadIdx.x + halo_x;
    sy = threadIdx.y + halo_y;
    px += halo_x;
    py += halo_y;
    if (sx < BLOCK_X - halo_x && px < tile_width) {
#pragma unroll
        for (int i = 0, pidx = py * tile_width + px; i < BLOCK_Y / STRIDE_Y; ++i, pidx += STRIDE_Y * tile_width) {
            if (sy + i * STRIDE_Y < BLOCK_Y - halo_y && py + i * STRIDE_Y < tile_height) {
                p2_real[pidx] = rl[sy + i * STRIDE_Y][sx];
                p2_imag[pidx] = im[sy + i * STRIDE_Y][sx];
            }
        }
    }
}

__global__ void gpu_rabi_coupling_real(size_t width, size_t height,
                                       double cc, double cs_r, double cs_i,
                                       double *p_real, double *p_imag,
                                       double *pb_real, double *pb_imag) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    double real, imag;
    /* The test shouldn't be necessary */
    if (blockIdx.x < height && threadIdx.x < width) {
        real = p_real[idx];
        imag = p_imag[idx];
        p_real[idx] = cc * real - cs_i * pb_real[idx] - cs_r * pb_imag[idx];
        p_imag[idx] = cc * imag + cs_r * pb_real[idx] - cs_i * pb_imag[idx];
        pb_real[idx] = cc * pb_real[idx] + cs_i * real - cs_r * imag;
        pb_imag[idx] = cc * pb_imag[idx] + cs_r * real + cs_i * imag;
    }
}

__global__ void gpu_rabi_coupling_imag(size_t width, size_t height,
                                       double cc, double cs_r, double cs_i,
                                       double *p_real, double *p_imag,
                                       double *pb_real, double *pb_imag) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    double real, imag;
    /* The test shouldn't be necessary */
    if (blockIdx.x < height && threadIdx.x < width) {
        real = p_real[idx];
        imag = p_imag[idx];
        p_real[idx] = cc * real + cs_r * pb_real[idx] - cs_i * pb_imag[idx];
        p_imag[idx] = cc * imag + cs_i * pb_real[idx] + cs_r * pb_imag[idx];
        pb_real[idx] = cc * pb_real[idx] + cs_r * real + cs_i * imag;
        pb_imag[idx] = cc * pb_imag[idx] - cs_i * real + cs_r * imag;
    }
}

// Wrapper function for the hybrid kernel
void cc2kernel_wrapper(size_t tile_width, size_t tile_height, size_t offset_x, size_t offset_y, size_t halo_x, size_t halo_y, dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t stream, double a, double b, double coupling_a, double alpha_x, double alpha_y, const double * __restrict__ dev_external_pot_real, const double * __restrict__ dev_external_pot_imag, const double * __restrict__ pdev_real, const double * __restrict__ pdev_imag, double * __restrict__ pdev2_real, double * __restrict__ pdev2_imag, int inner, int horizontal, int vertical, bool imag_time) {
    if(imag_time)
        imag_cc2kernel <<< numBlocks, threadsPerBlock, 0, stream>>>(tile_width, tile_height, offset_x, offset_y, halo_x, halo_y, a, b, coupling_a, alpha_x, alpha_y, dev_external_pot_real, dev_external_pot_imag, pdev_real, pdev_imag, pdev2_real, pdev2_imag, inner, horizontal, vertical);
    else
        cc2kernel <<< numBlocks, threadsPerBlock, 0, stream>>>(tile_width, tile_height, offset_x, offset_y, halo_x, halo_y, a, b, coupling_a, alpha_x, alpha_y, dev_external_pot_real, dev_external_pot_imag, pdev_real, pdev_imag, pdev2_real, pdev2_imag, inner, horizontal, vertical);
    CUT_CHECK_ERROR("Kernel error in cc2kernel_wrapper");
}

CC2Kernel::CC2Kernel(Lattice *grid, State *state, Hamiltonian *hamiltonian,
                     double *_external_pot_real, double *_external_pot_imag,
                     double _a, double _b, double delta_t,
                     double _norm, bool _imag_time):
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

    a = new double [1];
    b = new double [1];
    norm = new double [1];
    a[0] = _a;
    b[0] = _b;
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
                     double *_a, double *_b, double delta_t,
                     double *_norm, bool _imag_time):
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
    a = _a;
    b = _b;
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

void CC2Kernel::update_potential(double *_external_pot_real, double *_external_pot_imag) {
    external_pot_real[0] = _external_pot_real;
    external_pot_imag[0] = _external_pot_imag;
    CUDA_SAFE_CALL(cudaMemcpy(dev_external_pot_real[0], external_pot_real[0], tile_width * tile_height * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dev_external_pot_imag[0], external_pot_imag[0], tile_width * tile_height * sizeof(double), cudaMemcpyHostToDevice));
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
    } else {
        delete [] a;
        delete [] b;
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
        imag_cc2kernel <<< numBlocks, threadsPerBlock, 0, stream1>>>(tile_width, tile_height, 0, 0, halo_x, halo_y, a[state_index], b[state_index], coupling_const[state_index], alpha_x, alpha_y, dev_external_pot_real[state_index], dev_external_pot_imag[state_index], pdev_real[state_index][sense], pdev_imag[state_index][sense], pdev_real[state_index][1 - sense], pdev_imag[state_index][1 - sense], inner, horizontal, vertical);
    else
        cc2kernel <<< numBlocks, threadsPerBlock, 0, stream1>>>(tile_width, tile_height, 0, 0, halo_x, halo_y, a[state_index], b[state_index], coupling_const[state_index], alpha_x, alpha_y, dev_external_pot_real[state_index], dev_external_pot_imag[state_index], pdev_real[state_index][sense], pdev_imag[state_index][sense], pdev_real[state_index][1 - sense], pdev_imag[state_index][1 - sense], inner, horizontal, vertical);

    inner = 0;
    horizontal = 0;
    vertical = 1;
    numBlocks.x = 2;
    numBlocks.y = (tile_height  + (BLOCK_Y - 2 * halo_y) - 1) / (BLOCK_Y - 2 * halo_y);
    if(imag_time)
        imag_cc2kernel <<< numBlocks, threadsPerBlock, 0, stream1>>>(tile_width, tile_height, 0, 0, halo_x, halo_y, a[state_index], b[state_index], coupling_const[state_index], alpha_x, alpha_y, dev_external_pot_real[state_index], dev_external_pot_imag[state_index], pdev_real[state_index][sense], pdev_imag[state_index][sense], pdev_real[state_index][1 - sense], pdev_imag[state_index][1 - sense], inner, horizontal, vertical);
    else
        cc2kernel <<< numBlocks, threadsPerBlock, 0, stream1>>>(tile_width, tile_height, 0, 0, halo_x, halo_y, a[state_index], b[state_index], coupling_const[state_index], alpha_x, alpha_y, dev_external_pot_real[state_index], dev_external_pot_imag[state_index], pdev_real[state_index][sense], pdev_imag[state_index][sense], pdev_real[state_index][1 - sense], pdev_imag[state_index][1 - sense], inner, horizontal, vertical);
}

void CC2Kernel::run_kernel() {
    int inner = 0, horizontal = 0, vertical = 0;
    inner = 1;
    horizontal = 0;
    vertical = 0;
    numBlocks.x = (tile_width  + (BLOCK_X - 2 * halo_x) - 1) / (BLOCK_X - 2 * halo_x) ;
    numBlocks.y = (tile_height + (BLOCK_Y - 2 * halo_y) - 1) / (BLOCK_Y - 2 * halo_y) - 2;

    if(imag_time)
        imag_cc2kernel <<< numBlocks, threadsPerBlock, 0, stream2>>>(tile_width, tile_height, 0, 0, halo_x, halo_y, a[state_index], b[state_index], coupling_const[state_index], alpha_x, alpha_y, dev_external_pot_real[state_index], dev_external_pot_imag[state_index], pdev_real[state_index][sense], pdev_imag[state_index][sense], pdev_real[state_index][1 - sense], pdev_imag[state_index][1 - sense], inner, horizontal, vertical);
    else
        cc2kernel <<< numBlocks, threadsPerBlock, 0, stream2>>>(tile_width, tile_height, 0, 0, halo_x, halo_y, a[state_index], b[state_index], coupling_const[state_index], alpha_x, alpha_y, dev_external_pot_real[state_index], dev_external_pot_imag[state_index], pdev_real[state_index][sense], pdev_imag[state_index][sense], pdev_real[state_index][1 - sense], pdev_imag[state_index][1 - sense], inner, horizontal, vertical);
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
        double _norm = sqrt((tot_sum_a + tot_sum_b)/tot_norm);
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
