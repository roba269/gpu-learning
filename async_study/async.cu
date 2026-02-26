#include <cassert>
#include <iostream>
#include <cuda_runtime.h>

__global__ void ref_kernel(float* input, float *output, int N) {
    __shared__ float smem[32];
    float sum = .0;
    for (int i = threadIdx.x ; i < N ; i += 32) {
        smem[threadIdx.x] = input[i];
        __syncthreads();
        sum += smem[threadIdx.x];
        __syncthreads();
    }
    output[threadIdx.x] = sum;
}

void verify(float *input, float *output, int N) {
    for (int i = 0 ; i < 32 ; ++i) {
        float tmp = 0;
        for (int j = i ; j < N ; j += 32) {
            tmp += input[j];
        }
        assert(fabs(tmp - output[i]) < 1e-6);
    }
}

int main() {
    const int N = 1<<20;
    float *d_input, *d_output, *h_input, *h_output;
    cudaMallocHost(&h_input, sizeof(float) * N);
    for (int i = 0 ; i < N ; ++i) {
        h_input[i] = 1.0 * rand() / RAND_MAX;
    }
    cudaMalloc(&d_input, sizeof(float) * N);
    cudaMemcpy(d_input, h_input, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMallocHost(&h_output, sizeof(float) * 32);
    cudaMalloc(&d_output, sizeof(float) * 32);
    ref_kernel<<<1, 32>>>(d_input, d_output, N);    // launch only 1 thread block which contains 32 thread (i.e. 1 warp)
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, sizeof(float) * 32, cudaMemcpyDeviceToHost);
    verify(h_input, h_output, N);
    return 0;
}