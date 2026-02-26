#include <cassert>
#include <iostream>
#include <chrono>

#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>

const int WARP_SIZE = 32;

__global__ void ref_kernel(float *gA, float *gB, float *gC, int M, int N, int K) {
    __shared__ float sA[WARP_SIZE], sB[WARP_SIZE];
    float acc[WARP_SIZE] = {0};
    for (int phase = 0 ; phase < K ; ++phase) {
        // collectively load one column of A and one row of B from gmem to smem
        sA[threadIdx.x] = gA[threadIdx.x * K + phase];
        sB[threadIdx.x] = gB[phase * N + threadIdx.x];
        __syncthreads();
        // each thread is responsible for one row of final result
        for (int i = 0 ; i < WARP_SIZE ; ++i)
            acc[i] += sA[threadIdx.x] * sB[i];
        __syncthreads();
    }
    for (int i = 0 ; i < WARP_SIZE ; ++i)
        gC[threadIdx.x * N + i] = acc[i];
}

__global__ void pipeline_kernel(float *gA, float *gB, float *gC, int M, int N, int K) {
    constexpr auto scope = cuda::thread_scope_block;
    __shared__ cuda::pipeline_shared_state<scope, 2> shared_state;
    auto group = cooperative_groups::this_thread_block();
    auto pipe = cuda::make_pipeline(group, &shared_state);
    
    __shared__ float sA[2][WARP_SIZE], sB[2][WARP_SIZE];
    float acc[WARP_SIZE] = {0};
    pipe.producer_acquire();
    cuda::memcpy_async(&sA[0][threadIdx.x], &gA[threadIdx.x * K], sizeof(float), pipe);
    cuda::memcpy_async(&sB[0][threadIdx.x], &gB[threadIdx.x], sizeof(float), pipe);
    pipe.producer_commit();

    for (int produce_phase = 1 ; produce_phase < K ; ++produce_phase) {
        pipe.producer_acquire();
        cuda::memcpy_async(&sA[produce_phase % 2][threadIdx.x], &gA[threadIdx.x * K + produce_phase], sizeof(float), pipe);
        cuda::memcpy_async(&sB[produce_phase % 2][threadIdx.x], &gB[produce_phase * N + threadIdx.x], sizeof(float), pipe);
        pipe.producer_commit();
        pipe.consumer_wait();
        for (int i = 0 ; i < WARP_SIZE ; ++i)
            acc[i] += sA[(produce_phase-1)%2][threadIdx.x] * sB[(produce_phase-1)%2][i];
        pipe.consumer_release();
    }

    pipe.consumer_wait();
    for (int i = 0 ; i < WARP_SIZE ; ++i)
        acc[i] += sA[(K-1)%2][threadIdx.x] * sB[(K-1)%2][i];    
    pipe.consumer_release();

    for (int i = 0 ; i < WARP_SIZE ; ++i)
        gC[threadIdx.x * N + i] = acc[i];
}

void verify(float *a, float *b, float *c, int M, int N, int K) {
    for (int i = 0 ; i < M ; ++i)
        for (int j = 0 ; j < N ; ++j) {
            float tmp = .0;
            for (int k = 0 ; k < K ; ++k) {
                tmp += a[i*K+k] * b[k*N+j];
            }
            assert(fabs(c[i*N+j] - tmp) / max(c[i*N+j], tmp) < 1e-6);
        }
}

const int WARMUP_COUNTS = 2;
const int RUN_COUNTS = 10;
void benchmark_ref_kernel(float *d_A, float *d_B, float *d_C, float *h_A, float *h_B, float *h_C, int M, int N, int K) {
    double total_dur = 0;
    for (int i = 0 ; i < WARMUP_COUNTS + RUN_COUNTS ; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        ref_kernel<<<1,32>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        if (i >= WARMUP_COUNTS) total_dur += duration.count();
        if (i == 0) {
            cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
            verify(h_A, h_B, h_C, M, N, K);
        }
    }
    std::cout << "Ref kernel avg time:" << total_dur / RUN_COUNTS << " ms" << std::endl;
}

void benchmark_pipeline_kernel(float *d_A, float *d_B, float *d_C, float *h_A, float *h_B, float *h_C, int M, int N, int K) {
    double total_dur = 0;
    for (int i = 0 ; i < WARMUP_COUNTS + RUN_COUNTS ; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        pipeline_kernel<<<1,32>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        if (i >= WARMUP_COUNTS) total_dur += duration.count();
        if (i == 0) {
            cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
            verify(h_A, h_B, h_C, M, N, K);
        }
    }
    std::cout << "Pipeline kernel avg time:" << total_dur / RUN_COUNTS << " ms" << std::endl;
}

int main() {
    srand(2026);
    const int M = 32, N = 32, K = 65536;
    float *h_A, *h_B, *h_C;

    cudaMallocHost(&h_A, M*K*sizeof(float));
    for (int i = 0 ; i < M ; ++i)
        for (int j = 0 ; j < K ; ++j)
            h_A[i*K+j] = 1.0 * rand() / RAND_MAX;
    cudaMallocHost(&h_B, K*N*sizeof(float));
    for (int i = 0 ; i < K ; ++i)
        for (int j = 0 ; j < N ; ++j)
            h_B[i*N+j] = 1.0 * rand() / RAND_MAX;
    cudaMallocHost(&h_C, M*N*sizeof(float));

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M*K*sizeof(float));
    cudaMemcpy(d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_B, K*N*sizeof(float));
    cudaMemcpy(d_B, h_B, K*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_C, M*N*sizeof(float));

    benchmark_ref_kernel(d_A, d_B, d_C, h_A, h_B, h_C, M, N, K);
    benchmark_pipeline_kernel(d_A, d_B, d_C, h_A, h_B, h_C, M, N, K);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(h_A);
    cudaFree(h_B);
    cudaFree(h_C);
    return 0;
}