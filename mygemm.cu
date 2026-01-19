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

// matrix A: read a BM * BK tile from GMEM to SMEM, for phase phase_idx
// matrix B: read a BK * BN tile from GMEM to SMEM, for phase phase_idx
template<const int BM, const int BN, const int BK, const int TM, const int TN>
__device__ void load_tiles_gmem_to_smem(float *ga, float *gb, float *sa, float *sb, int phase_idx, int N, int K) {
    const int threads_per_block = blockDim.x * blockDim.y;
    const int linear_thread_idx_in_block = threadIdx.y * blockDim.x + threadIdx.x;

    // (row_in_A, col_in_A) is the top-left of the source A matrix
    const int row_in_A = BM * blockIdx.y;
    const int col_in_A = BK * phase_idx;
    assert(BM * BK % threads_per_block == 0);
    const int cells_per_thread_A = BM * BK / threads_per_block;
    for (int load_phase = 0 ; load_phase < cells_per_thread_A ; ++load_phase) {
        int linear_cell_idx_in_tile = load_phase * threads_per_block + linear_thread_idx_in_block;
        int row_in_tile = linear_cell_idx_in_tile / BK;
        int col_in_tile = linear_cell_idx_in_tile % BK;
        int row_global = row_in_A + row_in_tile;
        int col_global = col_in_A + col_in_tile;
        sa[row_in_tile * BK + col_in_tile] = ga[row_global * K + col_global];
    }

    // (row_in_B, col_in_B) is the top-left of the source B matrix
    const int row_in_B = BK * phase_idx;
    const int col_in_B = BN * blockIdx.x;
    assert(BK * BN % threads_per_block == 0);
    const int cells_per_thread_B = BK * BN / threads_per_block;
    for (int load_phase = 0 ; load_phase < cells_per_thread_B ; ++load_phase) {
        int linear_cell_idx_in_tile = load_phase * threads_per_block + linear_thread_idx_in_block;
        int row_in_tile = linear_cell_idx_in_tile / BN;
        int col_in_tile = linear_cell_idx_in_tile % BN;
        int row_global = row_in_B + row_in_tile;
        int col_global = col_in_B + col_in_tile;
        sb[row_in_tile * BN + col_in_tile] = gb[row_global * N + col_global];
    }
}

// each thread process a tile of output cells (TM, TN)
template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void matmulKernel_blocktile_threadtile(float *a, float *b, float *c, int M, int N, int K) {
    assert(K % BK == 0);
    static_assert(BM % TM == 0 && BN % TN == 0);
    assert(blockDim.x * blockDim.y * TM * TN == BM * BN);
    __shared__ float tile_a[BM][BK], tile_b[BK][BN];
    float reg_a[TM] = {0}, reg_b[TN] = {0}, reg_c[TM][TN] = {0};

    const int n_phase = K / BK;
    for (int phase = 0 ; phase < n_phase ; phase++) {
        load_tiles_gmem_to_smem<BM, BN, BK, TM, TN>(a, b, (float*)tile_a, (float*)tile_b, phase, N, K);
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

// each warp process (WM * WN) output cells, while each thread process (TM * TN) cells.
// since one warp contains 32 threads, there should be (WM * WN) / (TM * TN) == 32. 
template<const int BM, const int BN, const int BK, const int WM, const int WN, const int TM, const int TN>
__global__ void matmulKernel_blocktile_threadtile_warptile(float *a, float *b, float *c, int M, int N, int K) {
    static_assert(BM % WM == 0 && BN % WN == 0);
    static_assert(WM % TM == 0 && WN % TN == 0);
    assert(blockDim.x * blockDim.y * TM * TN == BM * BN);
    static_assert((WM/TM) * (WN/TN) == 32); // one warp-tile is handled by one warp (i.e. 32 threads) exactly
    assert(K % BK == 0);

    __shared__ float tile_a[BM][BK+1], tile_b[BK][BN+1];
    float reg_a[TM] = {0}, reg_b[TN] = {0}, reg_c[TM][TN] = {0};

    const int linear_thread_id_in_block_tile = threadIdx.y * blockDim.x + threadIdx.x;
    //assert(linear_thread_id_in_block_tile < blockDim.x * blockDim.y);
    const int warp_id_in_block_tile = linear_thread_id_in_block_tile / 32;

    constexpr int NUM_WARPTILE_ROWS_IN_BLOCK = BM / WM;
    constexpr int NUM_WARPTILE_COLS_IN_BLOCK = BN / WN;
    const int warp_row_idx_in_block = warp_id_in_block_tile / NUM_WARPTILE_COLS_IN_BLOCK;
    const int warp_col_idx_in_block = warp_id_in_block_tile % NUM_WARPTILE_COLS_IN_BLOCK;
    //assert(warp_row_idx_in_block < 2);  // TODO: remove
    //assert(warp_col_idx_in_block < 4);  // TODO: remove

    const int lane_id = linear_thread_id_in_block_tile % 32;
    constexpr int NUM_THREADTILE_ROWS_IN_WARPTILE = WM / TM;
    constexpr int NUM_THREADTILE_COLS_IN_WARPTILE = WN / TN;

    const int threadtile_row_idx_in_warptile = lane_id / NUM_THREADTILE_COLS_IN_WARPTILE;
    const int threadtile_col_idx_in_warptile = lane_id % NUM_THREADTILE_COLS_IN_WARPTILE;
    //assert(threadtile_row_idx_in_warptile < 8);
    //assert(threadtile_col_idx_in_warptile < 4);

    const int threadtile_row_idx_in_block = warp_row_idx_in_block * NUM_THREADTILE_ROWS_IN_WARPTILE + threadtile_row_idx_in_warptile;
    const int threadtile_col_idx_in_block = warp_col_idx_in_block * NUM_THREADTILE_COLS_IN_WARPTILE + threadtile_col_idx_in_warptile;
    //assert(threadtile_row_idx_in_block < 16);
    //assert(threadtile_col_idx_in_block < 16);

    const int block_row = BM * blockIdx.y;
    const int block_col = BN * blockIdx.x;
    const int warptile_row = block_row + warp_row_idx_in_block * WM;
    const int warptile_col = block_col + warp_col_idx_in_block * WN;
    const int threadtile_row = warptile_row + threadtile_row_idx_in_warptile * TM;
    const int threadtile_col = warptile_col + threadtile_col_idx_in_warptile * TN;

    const int n_phase = K / BK;
    for (int phase_idx = 0 ; phase_idx < n_phase ; ++phase_idx) {
        load_tiles_gmem_to_smem<BM, BN, BK, TM, TN>(a, b, (float*)tile_a, (float*)tile_b, phase_idx, N, K);
        __syncthreads();
        for (int k = 0 ; k < BK ; ++k) {
            for (int i = 0 ; i < TM ; ++i) {
                //assert(threadtile_row_idx_in_block * TM + i < BM);
                reg_a[i] = tile_a[threadtile_row_idx_in_block * TM + i][k];
            }
            for (int j = 0 ; j < TN ; ++j) {
                //assert(threadtile_col_idx_in_block * TN + j < BN);
                reg_b[j] = tile_b[k][threadtile_col_idx_in_block * TN + j];
            }
            for (int i = 0 ; i < TM ; ++i)
                for (int j = 0 ; j < TN ; ++j)
                    reg_c[i][j] += reg_a[i] * reg_b[j];
        }
        __syncthreads();
    }
    // write out
    for (int i = 0 ; i < TM ; ++i)
        for (int j = 0 ; j < TN ; ++j) {
            int row = threadtile_row + i;
            int col = threadtile_col + j;  
            c[row * N + col] = reg_c[i][j];
        }    
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
        matmulKernel_blocktile_threadtile_warptile<128,128,16,32,32,8,4>,
    };
    auto grid_dims = std::vector{
        dim3(cdiv(M, 32), cdiv(N, 32)), 
        dim3(cdiv(M, 32), cdiv(N, 32)),
        dim3(cdiv(M, 32), cdiv(N, 32)),
        dim3(cdiv(M, 32), cdiv(N, 32)),
        dim3(cdiv(M, 128), cdiv(N, 128)),
        dim3(cdiv(M, 128), cdiv(N, 128)),
    };
    auto block_dims = std::vector{
        dim3(32, 32),
        dim3(32, 32),
        dim3(32*32),
        dim3(32, 32),
        dim3(16, 16),
        dim3(16, 32),
    };
    auto kernel_names = std::vector{
        "matmulKernel", "matmulKernel_coal", "matmulKernel_coal_v2", "matmulKernel_blocktile",
        "matmulKernel_blocktile_threadtile", "matmulKernel_blocktile_threadtile_warptile"
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

    //runBenchmark(M, N, K, 0, h_A, h_B, h_C);
    //runBenchmark(M, N, K, 1, h_A, h_B, h_C);
    //runBenchmark(M, N, K, 2, h_A, h_B, h_C);
    //runBenchmark(M, N, K, 3, h_A, h_B, h_C);
    //runBenchmark(M, N, K, 4, h_A, h_B, h_C);
    runBenchmark(M, N, K, 5, h_A, h_B, h_C);
    //runBenchmark(M, N, K, 100, h_A, h_B, h_C);

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
