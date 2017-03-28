#undef _GLIBCXX_ATOMIC_BUILTINS
#include <stdio.h>
#include <sstream>
#include <vector>
#include <map>
#include <cassert>
#include "common.h"
#include "kernel.h"

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
                               double aH, double bH, double aV, double bV, double coupling_a, double alpha_x, double alpha_y, const double * __restrict__ external_pot_real, const double * __restrict__ external_pot_imag,
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
        imag_gpu_kernel_vertical<BLOCK_X, BLOCK_Y, 0>(aV, bV, tile_height, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, checkerboard_py + part * 2 * STRIDE_Y, rl, im);
    }
    __syncthreads();
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        imag_gpu_kernel_horizontal<BLOCK_X, BLOCK_Y, 0>(aH, bH, tile_width, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, px, rl, im);
    }
    __syncthreads();
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        imag_gpu_kernel_vertical<BLOCK_X, BLOCK_Y, 1>(aV, bV, tile_height, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, checkerboard_py + part * 2 * STRIDE_Y, rl, im);
    }
    __syncthreads();
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        imag_gpu_kernel_horizontal<BLOCK_X, BLOCK_Y, 1>(aH, bH, tile_width, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, px, rl, im);
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
        imag_gpu_kernel_horizontal<BLOCK_X, BLOCK_Y, 1>(aH, bH, tile_width, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, px, rl, im);
    }
    __syncthreads();
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        imag_gpu_kernel_vertical<BLOCK_X, BLOCK_Y, 1>(aV, bV, tile_height, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, checkerboard_py + part * 2 * STRIDE_Y, rl, im);
    }
    __syncthreads();
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        imag_gpu_kernel_horizontal<BLOCK_X, BLOCK_Y, 0>(aH, bH, tile_width, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, px, rl, im);
    }
    __syncthreads();
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        imag_gpu_kernel_vertical<BLOCK_X, BLOCK_Y, 0>(aV, bV, tile_height, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, checkerboard_py + part * 2 * STRIDE_Y, rl, im);
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

__launch_bounds__(BLOCK_X * STRIDE_Y)
__global__ void cc2kernel(size_t tile_width, size_t tile_height, size_t offset_x, size_t offset_y, size_t halo_x, size_t halo_y,
                          double aH, double bH, double aV, double bV, double coupling_a, double alpha_x, double alpha_y, const double * __restrict__ external_pot_real, const double * __restrict__ external_pot_imag,
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
        gpu_kernel_vertical<BLOCK_X, BLOCK_Y, 0>(aV, bV, tile_height, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, checkerboard_py + part * 2 * STRIDE_Y, rl, im);
    }
    __syncthreads();
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        gpu_kernel_horizontal<BLOCK_X, BLOCK_Y, 0>(aH, bH, tile_width, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, px, rl, im);
    }
    __syncthreads();
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        gpu_kernel_vertical<BLOCK_X, BLOCK_Y, 1>(aV, bV, tile_height, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, checkerboard_py + part * 2 * STRIDE_Y, rl, im);
    }
    __syncthreads();
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        gpu_kernel_horizontal<BLOCK_X, BLOCK_Y, 1>(aH, bH, tile_width, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, px, rl, im);
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
        gpu_kernel_horizontal<BLOCK_X, BLOCK_Y, 1>(aH, bH, tile_width, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, px, rl, im);
    }
    __syncthreads();
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        gpu_kernel_vertical<BLOCK_X, BLOCK_Y, 1>(aV, bV, tile_height, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, checkerboard_py + part * 2 * STRIDE_Y, rl, im);
    }
    __syncthreads();
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        gpu_kernel_horizontal<BLOCK_X, BLOCK_Y, 0>(aH, bH, tile_width, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, px, rl, im);
    }
    __syncthreads();
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        gpu_kernel_vertical<BLOCK_X, BLOCK_Y, 0>(aV, bV, tile_height, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, checkerboard_py + part * 2 * STRIDE_Y, rl, im);
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
