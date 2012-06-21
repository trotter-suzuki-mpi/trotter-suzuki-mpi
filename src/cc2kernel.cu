/**
 * Distributed Trotter-Suzuki solver
 * Copyright (C) 2012 Peter Wittek, 2010-2012 Carlos Bederián
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
#include <cassert>

#include "cc2kernel.h"

template<int BLOCK_WIDTH, int BLOCK_HEIGHT, int BACKWARDS>
inline __device__ void trotter_vert_pair_flexible_nosync(float a, float b, int tile_height, float &cell_r, float &cell_i, int kx, int ky, int py, float rl[BLOCK_HEIGHT][BLOCK_WIDTH], float im[BLOCK_HEIGHT][BLOCK_WIDTH]) {
    float peer_r;
    float peer_i;

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
        rl[ky_peer][kx] = __fadd_rn(a * peer_r, - b * cell_i);
        im[ky_peer][kx] = __fadd_rn(a * peer_i, b * cell_r);
        cell_r = __fadd_rn(a * cell_r, - b * peer_i);
        cell_i = __fadd_rn(a * cell_i, b * peer_r);
#endif
    }
}


template<int BLOCK_WIDTH, int BLOCK_HEIGHT, int BACKWARDS>
static  inline __device__ void trotter_horz_pair_flexible_nosync(float a, float b,  int tile_width, float &cell_r, float &cell_i, int kx, int ky, int px, float rl[BLOCK_HEIGHT][BLOCK_WIDTH], float im[BLOCK_HEIGHT][BLOCK_WIDTH]) {
    float peer_r;
    float peer_i;

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
        rl[ky][kx_peer] = __fadd_rn(a * peer_r, - b * cell_i);
        im[ky][kx_peer] = __fadd_rn(a * peer_i, b * cell_r);
        cell_r = __fadd_rn(a * cell_r, - b * peer_i);
        cell_i = __fadd_rn(a * cell_i, b * peer_r);
#endif
    }
}

__launch_bounds__(BLOCK_X * STRIDE_Y)
__global__ void cc2kernel(size_t tile_width, size_t tile_height, size_t block_width, size_t block_height, size_t halo_x, size_t halo_y, float a, float b, const float * __restrict__ p_real, const float * __restrict__ p_imag, float * __restrict__ p2_real, float * __restrict__ p2_imag, int inner, int horizontal, int vertical) {
    __shared__ float rl[BLOCK_Y][BLOCK_X];
    __shared__ float im[BLOCK_Y][BLOCK_X];
    
    int blockIdxx=inner*(blockIdx.x+1)+horizontal*(blockIdx.x)+vertical*(blockIdx.x*((tile_width + (BLOCK_X - 2 * halo_x) - 1) / (BLOCK_X - 2 * halo_x)-1));
    int blockIdxy=inner*(blockIdx.y+1)+horizontal*(blockIdx.y*((tile_height + (BLOCK_Y - 2 * halo_y) - 1) / (BLOCK_Y - 2 * halo_y)-1))+vertical*(blockIdx.y+1);
    
    // BLOCK_X can be different from block_width using the hybrid kernel
    int px = blockIdxx * (block_width - 2 * halo_x) + threadIdx.x - halo_x;
    int py = blockIdxy * (block_height - 2 * halo_y) + threadIdx.y - halo_y;

    // Read block from global into shared memory
    if (px >= 0 && px < tile_width) {
#pragma unroll
        for (int i = 0, pidx = py * tile_width + px; i < BLOCK_Y / STRIDE_Y; ++i, pidx += STRIDE_Y * tile_width) {
            if (py + i * STRIDE_Y >= 0 && py + i * STRIDE_Y < tile_height) {
                rl[threadIdx.y + i * STRIDE_Y][threadIdx.x] = p_real[pidx];
                im[threadIdx.y + i * STRIDE_Y][threadIdx.x] = p_imag[pidx];
            }
        }
    }

    __syncthreads();

    // Place threads along the black cells of a checkerboard pattern
    int sx = threadIdx.x;
    int sy;
    if ((halo_x) % 2 == (halo_y) % 2) {
        sy = 2 * threadIdx.y + threadIdx.x % 2;
    } else {
        sy = 2 * threadIdx.y + 1 - threadIdx.x % 2;
    }

    // global y coordinate of the thread on the checkerboard (px remains the same)
    // used for range checks
    int checkerboard_py = blockIdxy * (BLOCK_Y - 2 * halo_y) + sy - halo_y;

    // Keep the fixed black cells on registers, reds are updated in shared memory
    float cell_r[BLOCK_Y / (STRIDE_Y * 2)];
    float cell_i[BLOCK_Y / (STRIDE_Y * 2)];

#pragma unroll
    // Read black cells to registers
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        cell_r[part] = rl[sy + part * 2 * STRIDE_Y][sx];
        cell_i[part] = im[sy + part * 2 * STRIDE_Y][sx];
    }

    // 12344321
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        trotter_vert_pair_flexible_nosync<BLOCK_X, BLOCK_Y, 0>(a, b, tile_height, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, checkerboard_py + part * 2 * STRIDE_Y, rl, im);
    }
    __syncthreads();
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        trotter_horz_pair_flexible_nosync<BLOCK_X, BLOCK_Y, 0>(a, b, tile_width, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, px, rl, im);
    }
    __syncthreads();
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        trotter_vert_pair_flexible_nosync<BLOCK_X, BLOCK_Y, 1>(a, b, tile_height, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, checkerboard_py + part * 2 * STRIDE_Y, rl, im);
    }
    __syncthreads();
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        trotter_horz_pair_flexible_nosync<BLOCK_X, BLOCK_Y, 1>(a, b, tile_width, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, px, rl, im);
    }
    __syncthreads();

#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        trotter_horz_pair_flexible_nosync<BLOCK_X, BLOCK_Y, 1>(a, b, tile_width, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, px, rl, im);
    }
    __syncthreads();
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        trotter_vert_pair_flexible_nosync<BLOCK_X, BLOCK_Y, 1>(a, b, tile_height, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, checkerboard_py + part * 2 * STRIDE_Y, rl, im);
    }
    __syncthreads();
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        trotter_horz_pair_flexible_nosync<BLOCK_X, BLOCK_Y, 0>(a, b, tile_width, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, px, rl, im);
    }
    __syncthreads();
#pragma unroll
    for (int part = 0; part < BLOCK_Y / (STRIDE_Y * 2); ++part) {
        trotter_vert_pair_flexible_nosync<BLOCK_X, BLOCK_Y, 0>(a, b, tile_height, cell_r[part], cell_i[part], sx, sy + part * 2 * STRIDE_Y, checkerboard_py + part * 2 * STRIDE_Y, rl, im);
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

// Wrapper function for the hybrid kernel
void cc2kernel_wrapper(size_t tile_width, size_t tile_height, size_t block_width, size_t block_height, size_t halo_x, size_t halo_y, dim3 numBlocks, dim3 threadsPerBlock, cudaStream_t stream, float a, float b, const float * __restrict__ pdev_real, const float * __restrict__ pdev_imag, float * __restrict__ pdev2_real, float * __restrict__ pdev2_imag, int inner, int horizontal, int vertical) {
    cc2kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(tile_width, tile_height, block_width, block_height, halo_x, halo_y, a, b, pdev_real, pdev_imag, pdev2_real, pdev2_imag, inner, horizontal, vertical);
    CUT_CHECK_ERROR("Kernel error in cc2kernel_wrapper");
}


CC2Kernel::CC2Kernel(float *_p_real, float *_p_imag, float _a, float _b, int _tile_width, int _tile_height, int _halo_x, int _halo_y):
    p_real(_p_real),
    p_imag(_p_imag),
    threadsPerBlock(BLOCK_X, STRIDE_Y),
    numBlocks((_tile_width  + (BLOCK_X - 2 * _halo_x) - 1) / (BLOCK_X - 2 * _halo_x),
              (_tile_height + (BLOCK_Y - 2 * _halo_y) - 1) / (BLOCK_Y - 2 * _halo_y)),
    sense(0),
    a(_a),
    b(_b),
    tile_width(_tile_width),
    tile_height(_tile_height),
    halo_x(_halo_x),
    halo_y(_halo_y)
{

    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&pdev_real[0]), tile_width * tile_height * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&pdev_real[1]), tile_width * tile_height * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&pdev_imag[0]), tile_width * tile_height * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&pdev_imag[1]), tile_width * tile_height * sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpy(pdev_real[0], p_real, tile_width * tile_height * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(pdev_imag[0], p_imag, tile_width * tile_height * sizeof(float), cudaMemcpyHostToDevice));
    cudaStreamCreate(&stream1);
	  cudaStreamCreate(&stream2);
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

    CUDA_SAFE_CALL(cudaFree(pdev_real[0]));
    CUDA_SAFE_CALL(cudaFree(pdev_real[1]));
    CUDA_SAFE_CALL(cudaFree(pdev_imag[0]));
    CUDA_SAFE_CALL(cudaFree(pdev_imag[1]));
    
    cudaStreamDestroy(stream1);
	  cudaStreamDestroy(stream2);
}

void CC2Kernel::run_kernel_on_halo() {
    int inner=0, horizontal=0, vertical=0;  
    inner=0; horizontal=1; vertical=0;
    numBlocks.x=(tile_width  + (BLOCK_X - 2 * halo_x) - 1) / (BLOCK_X - 2 * halo_x);
    numBlocks.y=2;
    cc2kernel<<<numBlocks, threadsPerBlock, 0, stream1>>>(tile_width, tile_height, BLOCK_X, BLOCK_Y, halo_x, halo_y, a, b, pdev_real[sense], pdev_imag[sense], pdev_real[1-sense], pdev_imag[1-sense], inner, horizontal, vertical);

    inner=0; horizontal=0; vertical=1;
    numBlocks.x=2;
    numBlocks.y=(tile_height  + (BLOCK_Y - 2 * halo_y) - 1) / (BLOCK_Y - 2 * halo_y);
    cc2kernel<<<numBlocks, threadsPerBlock, 0, stream1>>>(tile_width, tile_height, BLOCK_X, BLOCK_Y, halo_x, halo_y, a, b, pdev_real[sense], pdev_imag[sense], pdev_real[1-sense], pdev_imag[1-sense], inner, horizontal, vertical);
}

void CC2Kernel::run_kernel() {
    int inner=0, horizontal=0, vertical=0;  
    inner=1; horizontal=0; vertical=0;  
    numBlocks.x=(tile_width  + (BLOCK_X - 2 * halo_x) - 1) / (BLOCK_X - 2 * halo_x);
    numBlocks.y=(tile_height + (BLOCK_Y - 2 * halo_y) - 1) / (BLOCK_Y - 2 * halo_y) - 2;
    cc2kernel<<<numBlocks, threadsPerBlock, 0, stream2>>>(tile_width, tile_height, BLOCK_X, BLOCK_Y, halo_x, halo_y, a, b, pdev_real[sense], pdev_imag[sense], pdev_real[1-sense], pdev_imag[1-sense], inner, horizontal, vertical);
    sense = 1 - sense;
    CUT_CHECK_ERROR("Kernel error in CC2Kernel::run_kernel");
}

void CC2Kernel::wait_for_completion() {
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

void CC2Kernel::copy_results() {
    CUDA_SAFE_CALL(cudaMemcpy(p_real, pdev_real[sense], tile_width * tile_height * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(p_imag, pdev_imag[sense], tile_width * tile_height * sizeof(float), cudaMemcpyDeviceToHost));
}

void CC2Kernel::get_sample(size_t dest_stride, size_t x, size_t y, size_t width, size_t height, float * dest_real, float * dest_imag) const {
    assert(x < tile_width);
    assert(y < tile_height);
    assert(x + width <= tile_width);
    assert(y + height <= tile_height);
    CUDA_SAFE_CALL(cudaMemcpy2D(dest_real, dest_stride * sizeof(float), &(pdev_real[sense][y * tile_width + x]), tile_width * sizeof(float), width * sizeof(float), height, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy2D(dest_imag, dest_stride * sizeof(float), &(pdev_imag[sense][y * tile_width + x]), tile_width * sizeof(float), width * sizeof(float), height, cudaMemcpyDeviceToHost));
}

void CC2Kernel::initialize_MPI(MPI_Comm _cartcomm, int _start_x, int _inner_end_x, int _start_y, int _inner_start_y, int _inner_end_y)
{
    cartcomm=_cartcomm;
    MPI_Cart_shift(cartcomm, 0, 1, &neighbors[UP], &neighbors[DOWN]);
    MPI_Cart_shift(cartcomm, 1, 1, &neighbors[LEFT], &neighbors[RIGHT]);
    start_x = _start_x;
    inner_end_x = _inner_end_x;
    start_y = _start_y;
    inner_start_y = _inner_start_y;
    inner_end_y = _inner_end_y;

    // Halo exchange uses wave pattern to communicate
    int height = inner_end_y-inner_start_y;	// The vertical halo in rows
    int width = halo_x;	// The number of columns of the matrix
    // Allocating pinned memory for the buffers
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &left_real_receive, height*width*sizeof(float), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &left_real_send, height*width*sizeof(float), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &right_real_receive, height*width*sizeof(float), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &right_real_send, height*width*sizeof(float), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &left_imag_receive, height*width*sizeof(float), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &left_imag_send, height*width*sizeof(float), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &right_imag_receive, height*width*sizeof(float), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &right_imag_send, height*width*sizeof(float), cudaHostAllocDefault));

    height = halo_y;	// The vertical halo in rows
    width = tile_width;	// The number of columns of the matrix
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &bottom_real_receive, height*width*sizeof(float), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &bottom_real_send, height*width*sizeof(float), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &top_real_receive, height*width*sizeof(float), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &top_real_send, height*width*sizeof(float), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &bottom_imag_receive, height*width*sizeof(float), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &bottom_imag_send, height*width*sizeof(float), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &top_imag_receive, height*width*sizeof(float), cudaHostAllocDefault));
    CUDA_SAFE_CALL(cudaHostAlloc( (void **) &top_imag_send, height*width*sizeof(float), cudaHostAllocDefault));
}

void CC2Kernel::start_halo_exchange() {

}

void CC2Kernel::finish_halo_exchange() {
    MPI_Request req[8];
    MPI_Status statuses[8];
    int offset=0;

    // Halo copy: LEFT/RIGHT
    int height = inner_end_y-inner_start_y;	// The vertical halo in rows
    int width = halo_x;	// The number of columns of the matrix
    int stride = tile_width;	// The combined width of the matrix with the halo
    offset=(inner_start_y-start_y)*tile_width+inner_end_x-halo_x-start_x;
    CUDA_SAFE_CALL(cudaMemcpy2DAsync(right_real_send, width * sizeof(float), &(pdev_real[sense][offset]), stride * sizeof(float), width * sizeof(float), height, cudaMemcpyDeviceToHost, stream1));
    CUDA_SAFE_CALL(cudaMemcpy2DAsync(right_imag_send, width * sizeof(float), &(pdev_imag[sense][offset]), stride * sizeof(float), width * sizeof(float), height, cudaMemcpyDeviceToHost, stream1));
    offset=(inner_start_y-start_y)*tile_width+halo_x;
    CUDA_SAFE_CALL(cudaMemcpy2DAsync(left_real_send, width * sizeof(float), &(pdev_real[sense][offset]), stride * sizeof(float), width * sizeof(float), height, cudaMemcpyDeviceToHost, stream1));
    CUDA_SAFE_CALL(cudaMemcpy2DAsync(left_imag_send, width * sizeof(float), &(pdev_imag[sense][offset]), stride * sizeof(float), width * sizeof(float), height, cudaMemcpyDeviceToHost, stream1));

    // Halo copy: UP/DOWN
    height = halo_y;	// The vertical halo in rows
    width = tile_width;	// The number of columns of the matrix
    stride = tile_width;	// The combined width of the matrix with the halo

    offset=(inner_end_y-halo_y-start_y)*tile_width;
    CUDA_SAFE_CALL(cudaMemcpy2DAsync(bottom_real_send, width * sizeof(float), &(pdev_real[sense][offset]), stride * sizeof(float), width * sizeof(float), height, cudaMemcpyDeviceToHost, stream1));
    CUDA_SAFE_CALL(cudaMemcpy2DAsync(bottom_imag_send, width * sizeof(float), &(pdev_imag[sense][offset]), stride * sizeof(float), width * sizeof(float), height, cudaMemcpyDeviceToHost, stream1));
    offset=halo_y*tile_width;
    CUDA_SAFE_CALL(cudaMemcpy2DAsync(top_real_send, width * sizeof(float), &(pdev_real[sense][offset]), stride * sizeof(float), width * sizeof(float), height, cudaMemcpyDeviceToHost, stream1));
    CUDA_SAFE_CALL(cudaMemcpy2DAsync(top_imag_send, width * sizeof(float), &(pdev_imag[sense][offset]), stride * sizeof(float), width * sizeof(float), height, cudaMemcpyDeviceToHost, stream1));
    
    cudaStreamSynchronize(stream1);


    // Halo exchange: LEFT/RIGHT
    height = inner_end_y-inner_start_y;	// The vertical halo in rows
    width = halo_x;	// The number of columns of the matrix
    stride = tile_width;	// The combined width of the matrix with the halo

    MPI_Irecv(left_real_receive, height*width, MPI_FLOAT, neighbors[LEFT], 1, cartcomm, req);
    MPI_Irecv(left_imag_receive, height*width, MPI_FLOAT, neighbors[LEFT], 2, cartcomm, req+1);
    MPI_Irecv(right_real_receive, height*width, MPI_FLOAT, neighbors[RIGHT], 3, cartcomm, req+2);
    MPI_Irecv(right_imag_receive, height*width, MPI_FLOAT, neighbors[RIGHT], 4, cartcomm, req+3);

    offset=(inner_start_y-start_y)*tile_width+inner_end_x-halo_x-start_x;
    MPI_Isend(right_real_send, height*width, MPI_FLOAT, neighbors[RIGHT], 1, cartcomm,req+4);
    MPI_Isend(right_imag_send, height*width, MPI_FLOAT, neighbors[RIGHT], 2, cartcomm,req+5);

    offset=(inner_start_y-start_y)*tile_width+halo_x;
    MPI_Isend(left_real_send, height*width, MPI_FLOAT, neighbors[LEFT], 3, cartcomm,req+6);
    MPI_Isend(left_imag_send, height*width, MPI_FLOAT, neighbors[LEFT], 4, cartcomm,req+7);

    MPI_Waitall(8, req, statuses);

    // Halo exchange: UP/DOWN
    height = halo_y;	// The vertical halo in rows
    width = tile_width;	// The number of columns of the matrix
    stride = tile_width;	// The combined width of the matrix with the halo

    MPI_Irecv(top_real_receive, height*width, MPI_FLOAT, neighbors[UP], 1, cartcomm, req);
    MPI_Irecv(top_imag_receive, height*width, MPI_FLOAT, neighbors[UP], 2, cartcomm, req+1);
    MPI_Irecv(bottom_real_receive, height*width, MPI_FLOAT, neighbors[DOWN], 3, cartcomm, req+2);
    MPI_Irecv(bottom_imag_receive, height*width, MPI_FLOAT, neighbors[DOWN], 4, cartcomm, req+3);

    offset=(inner_end_y-halo_y-start_y)*tile_width;
    MPI_Isend(bottom_real_send, height*width, MPI_FLOAT, neighbors[DOWN], 1, cartcomm,req+4);
    MPI_Isend(bottom_imag_send, height*width, MPI_FLOAT, neighbors[DOWN], 2, cartcomm,req+5);

    offset=halo_y*tile_width;
    MPI_Isend(top_real_send, height*width, MPI_FLOAT, neighbors[UP], 3, cartcomm,req+6);
    MPI_Isend(top_imag_send, height*width, MPI_FLOAT, neighbors[UP], 4, cartcomm,req+7);

    MPI_Waitall(8, req, statuses);

    // Copy back the halos to the GPU memory

    height = inner_end_y-inner_start_y;	// The vertical halo in rows
    width = halo_x;	// The number of columns of the matrix
    stride = tile_width;	// The combined width of the matrix with the halo

    offset = (inner_start_y-start_y)*tile_width;
    if (neighbors[LEFT]>=0) {
        CUDA_SAFE_CALL(cudaMemcpy2DAsync(&(pdev_real[sense][offset]), stride * sizeof(float), left_real_receive, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice, stream1));
        CUDA_SAFE_CALL(cudaMemcpy2DAsync(&(pdev_imag[sense][offset]), stride * sizeof(float), left_imag_receive, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice, stream1));
    }
    offset = (inner_start_y-start_y)*tile_width+inner_end_x-start_x;
    if (neighbors[RIGHT]>=0) {
        CUDA_SAFE_CALL(cudaMemcpy2DAsync(&(pdev_real[sense][offset]), stride * sizeof(float), right_real_receive, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice, stream1));
        CUDA_SAFE_CALL(cudaMemcpy2DAsync(&(pdev_imag[sense][offset]), stride * sizeof(float), right_imag_receive, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice, stream1));
    }

    height = halo_y;	// The vertical halo in rows
    width = tile_width;	// The number of columns of the matrix
    stride = tile_width;	// The combined width of the matrix with the halo

    offset = 0;
    if (neighbors[UP]>=0) {
        CUDA_SAFE_CALL(cudaMemcpy2DAsync(&(pdev_real[sense][offset]), stride * sizeof(float), top_real_receive, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice, stream1));
        CUDA_SAFE_CALL(cudaMemcpy2DAsync(&(pdev_imag[sense][offset]), stride * sizeof(float), top_imag_receive, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice, stream1));
    }

    offset = (inner_end_y-start_y)*tile_width;
    if (neighbors[DOWN]>=0) {
        CUDA_SAFE_CALL(cudaMemcpy2DAsync(&(pdev_real[sense][offset]), stride * sizeof(float), bottom_real_receive, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice, stream1));
        CUDA_SAFE_CALL(cudaMemcpy2DAsync(&(pdev_imag[sense][offset]), stride * sizeof(float), bottom_imag_receive, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice, stream1));
    }
}

