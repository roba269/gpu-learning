#include <cuda.h>
#include <cstdio>
#include <chrono>
#include <iostream>
#include <cassert>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

const int BLOCKSIZE = 32;

#define cdiv(a,b) (((a)+(b)-1)/(b))

// multiply of two matrics (M*K) x (K*N), most naive impl
__global__ void matmulKernel(float *a, float *b, float *c, int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    float val = 0.0;
    for (int i = 0 ; i < K ; ++i) {
        val += a[row*K+i] * b[i*N+col];
    }
    c[row*N+col] = val;
}

// switch row/col from the naive impl, to utilize memory coalescing
__global__ void matmulKernel_coal(float *a, float *b, float *c, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    float val = 0.0;
    for (int i = 0 ; i < K ; ++i) {
        val += a[row*K+i] * b[i*N+col];
    }
    c[row*N+col] = val;
}

// same as above, but with 1D thread block
template<const int BLOCK_SIZE_M, const int BLOCK_SIZE_N>
__global__ void matmulKernel_coal_v2(float *a, float *b, float *c, int M, int N, int K) {
    int row_idx_in_block = threadIdx.x / BLOCK_SIZE_N;
    int col_idx_in_block = threadIdx.x % BLOCK_SIZE_N;
    int row = blockIdx.x * BLOCK_SIZE_M + row_idx_in_block;
    int col = blockIdx.y * BLOCK_SIZE_N + col_idx_in_block;
    float val = 0.0;
    for (int i = 0 ; i < K ; ++i) {
        val += a[row*K+i] * b[i*N+col];
    }
    c[row*N+col] = val;
}

// naive block tile implmentation, assuming TILESIZE == blockDim.x == blockDim.y
template<const int TILESIZE>
__global__ void matmulKernel_blocktile(float *a, float *b, float *c, int M, int N, int K) {
    assert(K % TILESIZE == 0);
    int n_phase = K / TILESIZE;
    __shared__ float tile_a[TILESIZE][TILESIZE], tile_b[TILESIZE][TILESIZE];
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int tile_col = threadIdx.x;
    const int tile_row = threadIdx.y;
    float val = 0.0;
    for (int phase_idx = 0 ; phase_idx < n_phase ; ++phase_idx) {
        // read memory
        tile_a[tile_row][tile_col] = a[row * K + (phase_idx * TILESIZE + tile_col)]; // a[row][phase_idx*TILESIZE+tile_col] 
        tile_b[tile_row][tile_col] = b[(phase_idx * TILESIZE + tile_row) * N + col]; // b[phase_idx*TILESIZE+tile_row][col]
        __syncthreads();
        // compute
        for (int k = 0 ; k < TILESIZE ; ++k) {
            val += tile_a[tile_row][k] * tile_b[k][tile_col];
        }
        __syncthreads();
    }
    c[row*N+col] = val;
}

// each thread process multiple output cells 
template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void matmulKernel_blocktile_threadtile(float *a, float *b, float *c, int M, int N, int K) {
    //assert(K % BK == 0);
    //static_assert(BM % TM == 0 && BN % TN == 0);
    const int linear_block_idx = blockIdx.y * gridDim.x + blockIdx.x;
    const int linear_thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    const int num_threads_per_block = blockDim.x * blockDim.y;

    __shared__ float tile_a[BM][BK], tile_b[BK][BN];
    float reg_a[TM] = {0}, reg_b[TN] = {0}, reg_c[TM][TN] = {0};

    const int n_phase = K / BK;

    for (int phase = 0 ; phase < n_phase ; phase++) {
        // read A tiles
        // we have num_thread_per_block threads, to read BM * BK cells collectively.
        // And to utilize memory coalescing, consective threads should read consecutive columns.
        // So, each thread should read (BM * BK / blockDim.x) rows
        assert(BM * BK % num_threads_per_block == 0);
        int read_cells = BM * BK / num_threads_per_block;
        for (int load_phase = 0 ; load_phase < read_cells ; ++load_phase) {
            int linear_idx_in_tile = load_phase * num_threads_per_block + linear_thread_idx;
            int tile_row = linear_idx_in_tile / BK;
            int tile_col = linear_idx_in_tile % BK;
            int A_row = BM * blockIdx.y + tile_row;
            int A_col = phase * BK + tile_col;
            tile_a[tile_row][tile_col] = a[A_row * K + A_col];
        }
        
        // read B tiles
        assert(BK * BN % num_threads_per_block == 0);
        read_cells = BK * BN / num_threads_per_block;
        for (int load_phase = 0 ; load_phase < read_cells ; ++load_phase) {
            int linear_idx_in_tile = load_phase * num_threads_per_block + linear_thread_idx;
            int tile_row = linear_idx_in_tile / BN;
            int tile_col = linear_idx_in_tile % BN;
            int B_row = phase * BK + tile_row;
            int B_col = BN * blockIdx.x + tile_col;
            tile_b[tile_row][tile_col] = b[B_row * N + B_col];
        }
        __syncthreads();

        for (int k = 0 ; k < BK ; ++k) {
            for (int i = 0 ; i < TM ; ++i) reg_a[i] = tile_a[threadIdx.y * TM + i][k];
            for (int j = 0 ; j < TN ; ++j) reg_b[j] = tile_b[k][threadIdx.x * TN + j];
            for (int i = 0 ; i < TM ; ++i)
                for (int j = 0 ; j < TN ; ++j)
                    reg_c[i][j] += reg_a[i] * reg_b[j];
        }

        __syncthreads();
    }
    // write out
    for (int i = 0 ; i < TM ; ++i)
        for (int j = 0 ; j < TN ; ++j) {
            int row = (blockIdx.y * blockDim.y + threadIdx.y) * TM + i;
            int col = (blockIdx.x * blockDim.x + threadIdx.x) * TN + j;
            c[row * N + col] = reg_c[i][j];
        }
}

template<const int BM, const int BN, const int BK, const int WM, const int WN, const int TM, const int TN>
__global__ void matmulKernel_blocktile_threadtile_warptile(float *a, float *b, float *c, int M, int N, int K) {

}

bool verify(float *h_golden, float *d_test, int M, int N) {
    float *h_test = (float*)malloc(M*N*sizeof(float));
    cudaMemcpy(h_test, d_test, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0 ; i < M * N ; ++i) {
        if (fabs(h_golden[i] - h_test[i]) / max(h_golden[i], h_test[i]) > 1e-4) {
            printf("diff for location %d, golden: %lf, test: %lf\n", i, h_golden[i], h_test[i]);
            free(h_test);
            return false;
        }
    }
    free(h_test);
    return true;
}

void runBenchmark(int M, int N, int K, size_t kernel_id, float *h_A, float *h_B, float *h_C, int run_counts = 10) {
    auto kernels = std::vector{
        matmulKernel,
        matmulKernel_coal,
        matmulKernel_coal_v2<32,32>,
        matmulKernel_blocktile<32>,
        matmulKernel_blocktile_threadtile<128,128,16,8,8>,
    };
    auto grid_dims = std::vector{
        dim3(cdiv(M, 32), cdiv(N, 32)), 
        dim3(cdiv(M, 32), cdiv(N, 32)),
        dim3(cdiv(M, 32), cdiv(N, 32)),
        dim3(cdiv(M, 32), cdiv(N, 32)),
        dim3(cdiv(M, 128), cdiv(N, 128)),
    };
    auto block_dims = std::vector{
        dim3(32, 32),
        dim3(32, 32),
        dim3(32*32),
        dim3(32, 32),
        dim3(16, 16),
    };
    auto kernel_names = std::vector{
        "matmulKernel", "matmulKernel_coal", "matmulKernel_coal_v2", "matmulKernel_blocktile", "matmulKernel_blocktile_threadtile"
    };

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M*K*sizeof(float));
    cudaMemcpy(d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_B, K*N*sizeof(float));
    cudaMemcpy(d_B, h_B, K*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_C, M*N*sizeof(float));

    cublasHandle_t handle;
    if (cublasCreate(&handle)) {
        printf("failed to create cublas handle");
        exit(1);
    }

    // warmup and verify
    if (kernel_id < 100) {
        kernels[kernel_id]<<<grid_dims[kernel_id], block_dims[kernel_id]>>>(d_A, d_B, d_C, M, N, K);
    } else {
        float alpha = 1.0, beta = 1.0;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    }
    cudaDeviceSynchronize();
    assert(verify(h_C, d_C, M, N));

    double total_dur = 0;
    for (int i = 0 ; i < run_counts ; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        if (kernel_id < 100) {
            kernels[kernel_id]<<<grid_dims[kernel_id], block_dims[kernel_id]>>>(d_A, d_B, d_C, M, N, K);
        } else {
            float alpha = 1.0, beta = 1.0;
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        total_dur += duration.count();
    }
    std::string kernel_name;
    if (kernel_id < 100) kernel_name = kernel_names[kernel_id];
    else kernel_name = "cublas";
    std::cout << "Kernel " << kernel_name << " takes: " << total_dur / run_counts << " microseconds" << std::endl;

    cudaFree(d_C);
    cudaFree(d_B);
    cudaFree(d_A);
}

void run(int M, int N, int K) {
    srand(2025);
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(M*K*sizeof(float));
    for (int i = 0 ; i < M ; ++i)
        for (int j = 0 ; j < K ; ++j)
            h_A[i*K+j] = 1.0 * rand() / RAND_MAX;
    h_B = (float*)malloc(K*N*sizeof(float));
    for (int i = 0 ; i < K ; ++i)
        for (int j = 0 ; j < N ; ++j)
            h_B[i*N+j] = 1.0 * rand() / RAND_MAX;

    printf("Start computing golden result\n");
    h_C = (float*)malloc(M*N*sizeof(float));
    for (int i = 0 ; i < M ; ++i)
        for (int j = 0 ; j < N ; ++j) {
            float val = 0;
            for (int k = 0 ; k < K ; ++k) {
                val += h_A[i*K+k] * h_B[k*N+j];
            }
            h_C[i*N+j] = val;
        }
    printf("Finish computing golden result\n");

    for (int i = 0 ; i < 10 ; ++i) {
        for (int j = 0 ; j < 10 ; ++j)
            printf(" %.6lf", h_C[i*N+j]);
        printf("\n");
    }

    runBenchmark(M, N, K, 0, h_A, h_B, h_C);
    runBenchmark(M, N, K, 1, h_A, h_B, h_C);
    runBenchmark(M, N, K, 2, h_A, h_B, h_C);
    runBenchmark(M, N, K, 3, h_A, h_B, h_C);
    runBenchmark(M, N, K, 4, h_A, h_B, h_C);
    runBenchmark(M, N, K, 100, h_A, h_B, h_C);

    /*
    {
        float *d_C;
        cudaMalloc((void**)&d_C, M*N*sizeof(float));
        auto start = std::chrono::high_resolution_clock::now();
        matmulKernel<<<dim3((M+BLOCKSIZE-1)/BLOCKSIZE, (N+BLOCKSIZE-1)/BLOCKSIZE), dim3(BLOCKSIZE, BLOCKSIZE)>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "NaiveKernel Time taken: " << duration.count() << " microseconds" << std::endl;
        assert(verify(h_C, d_C, M, N));
        cudaFree(d_C);
    }
    
    {
        float *d_C;
        cudaMalloc((void**)&d_C, M*N*sizeof(float));
        auto start = std::chrono::high_resolution_clock::now();
        matmulKernel_coal<<<dim3((M+BLOCKSIZE-1)/BLOCKSIZE, (N+BLOCKSIZE-1)/BLOCKSIZE), dim3(BLOCKSIZE, BLOCKSIZE)>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "NaiveKernel+Coalescing Time taken: " << duration.count() << " microseconds" << std::endl;
        assert(verify(h_C, d_C, M, N));
        cudaFree(d_C);
    }
    
    {
        float *d_C;
        cudaMalloc((void**)&d_C, M*N*sizeof(float));
        auto start = std::chrono::high_resolution_clock::now();
        matmulKernel_blocktile<<<dim3((M+BLOCKSIZE-1)/BLOCKSIZE, (N+BLOCKSIZE-1)/BLOCKSIZE), dim3(BLOCKSIZE, BLOCKSIZE)>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Tiled Kernel Time taken: " << duration.count() << " microseconds" << std::endl;
        assert(verify(h_C, d_C, M, N));
        cudaFree(d_C);
    }
    cudaFree(d_B);
    cudaFree(d_A);
    */
    free(h_C);
    free(h_B);
    free(h_A);
}

int main() {
    /*
    for (int i = 0 ; i < N ; ++i)
        h_a[i] = i;
    int *d_a;
    cudaMalloc((void**)&d_a, N*sizeof(int));
    cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice);
    myKernel<<<(N+BLOCKSIZE-1)/BLOCKSIZE, BLOCKSIZE>>>(d_a, N);
    cudaMemcpy(h_a, d_a, N*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a);

    for (int i = 0 ; i < 10 ; ++i)
        printf("a[%d]:%d\n", i, h_a[i]);
    */
    run(2048, 2048, 2048);
    return 0;
}
