#pragma once

// matrix A: read a BM * BK tile from GMEM to SMEM, for phase phase_idx
// matrix B: read a BK * BN tile from GMEM to SMEM, for phase phase_idx
// row_stride_sa and row_stride_sb can be used to avoid bank-conflict. For example, can be use sa[BM][BK+1] instead of sa[BM][BK].
template<const int BM, const int BN, const int BK>
__device__ void load_tiles_gmem_to_smem(float *ga, float *gb, float *sa, int row_stride_sa, float *sb, int row_stride_sb, int phase_idx, int N, int K) {
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
        sa[row_in_tile * row_stride_sa + col_in_tile] = ga[row_global * K + col_global];
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
        sb[row_in_tile * row_stride_sb + col_in_tile] = gb[row_global * N + col_global];
    }
}

// similar as load_tiles_gmem_to_smem but use vectorized loading
template<const int BM, const int BN, const int BK>
__device__ void load_tiles_gmem_to_smem_vectorize(float *ga, float *gb, float *sa, float *sb, int phase_idx, int N, int K) {
    static_assert(BK % 4 == 0);
    static_assert(BN % 4 == 0);
    //const int threads_per_block = blockDim.x * blockDim.y;
    //const int linear_thread_idx_in_block = threadIdx.x;

    // (row_in_A, col_in_A) is the top-left of the source A matrix
    const int row_in_A = BM * blockIdx.y;
    const int col_in_A = BK * phase_idx;
    assert(BM * BK % blockDim.x == 0);
    const int cells_per_thread_A = BM * BK / blockDim.x;
    assert(cells_per_thread_A % 4 == 0);
    for (int load_phase = 0 ; load_phase < cells_per_thread_A / 4 ; ++load_phase) {
        int linear_cell_idx_in_tile = (load_phase * blockDim.x + threadIdx.x) * 4;
        int row_in_tile = linear_cell_idx_in_tile / BK;
        int col_in_tile = linear_cell_idx_in_tile % BK;
        int row_global = row_in_A + row_in_tile;
        int col_global = col_in_A + col_in_tile;

        // transpose A
        float4 tmp = *reinterpret_cast<float4*>(&ga[row_global * K + col_global]);
        sa[(col_in_tile + 0) * (BM+4) + row_in_tile] = tmp.x;
        sa[(col_in_tile + 1) * (BM+4) + row_in_tile] = tmp.y;
        sa[(col_in_tile + 2) * (BM+4) + row_in_tile] = tmp.z;
        sa[(col_in_tile + 3) * (BM+4) + row_in_tile] = tmp.w;
    }

    // (row_in_B, col_in_B) is the top-left of the source B matrix
    const int row_in_B = BK * phase_idx;
    const int col_in_B = BN * blockIdx.x;
    assert(BK * BN % blockDim.x == 0);
    const int cells_per_thread_B = BK * BN / blockDim.x;
    assert(cells_per_thread_B % 4 == 0);
    for (int load_phase = 0 ; load_phase < cells_per_thread_B / 4 ; ++load_phase) {
        int linear_cell_idx_in_tile = (load_phase * blockDim.x + threadIdx.x) * 4;
        int row_in_tile = linear_cell_idx_in_tile / BN;
        int col_in_tile = linear_cell_idx_in_tile % BN;
        int row_global = row_in_B + row_in_tile;
        int col_global = col_in_B + col_in_tile;
        *reinterpret_cast<float4*>(&sb[row_in_tile * (BN+4) + col_in_tile]) =
            *reinterpret_cast<float4*>(&gb[row_global * N + col_global]);
    }
}
