#pragma once

#include "utils.cuh"

// Each warp process (WM * WN) output cells, which contains (WITERM * WITERN) sub-warptiles.
// Each sub-warptile has size (WSUBM * WSUBN), contains 32 (TM * TN) cells.
template<const int BM, const int BN, const int BK, const int WM, const int WN,
    const int WSUBM, const int WSUBN, const int TM, const int TN>
__global__ void matmulKernel_blocktile_threadtile_warptile(float *a, float *b, float *c, int M, int N, int K) {
    assert(K % BK == 0);
    static_assert(BM % WM == 0 && BN % WN == 0);
    static_assert(WM % WSUBM == 0 && WN % WSUBN == 0);
    static_assert(WSUBM % TM == 0 && WSUBN % TN == 0);
    constexpr int WITERM = WM / WSUBM;
    constexpr int WITERN = WN / WSUBN;
    static_assert((WSUBM * WSUBN) / (TM * TN) == 32);

    const int thread_linear_idx = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = thread_linear_idx / 32;
    constexpr int NUM_WARPTILE_COLS = BN / WN;
    const int warp_row_idx = warp_id / NUM_WARPTILE_COLS;
    const int warp_col_idx = warp_id % NUM_WARPTILE_COLS;

    const int lane_id = thread_linear_idx % 32;
    const int thread_row_in_warp = lane_id / (WSUBN / TN);
    const int thread_col_in_warp = lane_id % (WSUBN / TN);

    __shared__ float sa[BM*BK], sb[BK*BN];
    float reg_a[WITERM*TM], reg_b[WITERN*TN], reg_c[WITERM * TM * WITERN * TN] = {0};

    const int n_phase = K / BK;
    for (int phase_idx = 0 ; phase_idx < n_phase ; ++phase_idx) {
        load_tiles_gmem_to_smem<BM, BN, BK>(a, b, (float*)sa, BK, (float*)sb, BN, phase_idx, N, K);
        __syncthreads();

        for (int k = 0 ; k < BK ; ++k) {
            for (int sub_row_idx = 0 ; sub_row_idx < WITERM ; ++sub_row_idx) {
                for (int thread_row_idx = 0 ; thread_row_idx < TM ; ++thread_row_idx) {
                    int row_in_sa = warp_row_idx * WM + sub_row_idx * WSUBM + thread_row_in_warp * TM + thread_row_idx;
                    reg_a[sub_row_idx * TM + thread_row_idx] = sa[row_in_sa * BK + k];
                }
            }

            for (int sub_col_idx = 0 ; sub_col_idx < WITERN ; ++sub_col_idx) {
                for (int thread_col_idx = 0 ; thread_col_idx < TN ; ++thread_col_idx) {
                    int col_in_sb = warp_col_idx * WN + sub_col_idx * WSUBN + thread_col_in_warp * TN + thread_col_idx;
                    reg_b[sub_col_idx * TN + thread_col_idx] = sb[k * BN + col_in_sb];
                }
            }
            
            for (int sub_row_idx = 0 ; sub_row_idx < WITERM ; ++sub_row_idx)
                for (int thread_row_idx = 0 ; thread_row_idx < TM ; thread_row_idx++) {
                    int tmp_row = sub_row_idx * TM + thread_row_idx;
                    for (int sub_col_idx = 0 ; sub_col_idx < WITERN ; ++sub_col_idx)
                        for (int thread_col_idx = 0 ; thread_col_idx < TN ; thread_col_idx++) {
                            int tmp_col = sub_col_idx * TN + thread_col_idx;
                            reg_c[tmp_row * WITERN * TN + tmp_col] += reg_a[tmp_row] * reg_b[tmp_col];
                        }
                }
        }
        __syncthreads();
    }
    // write to gmem
    for (int sub_row_idx = 0 ; sub_row_idx < WITERM ; ++sub_row_idx)
        for (int thread_row_idx = 0 ; thread_row_idx < TM ; thread_row_idx++) {
            int tmp_row = sub_row_idx * TM + thread_row_idx;
            for (int sub_col_idx = 0 ; sub_col_idx < WITERN ; ++sub_col_idx)
                for (int thread_col_idx = 0 ; thread_col_idx < TN ; thread_col_idx++) {
                    int tmp_col = sub_col_idx * TN + thread_col_idx;
                    int row_in_gc = blockIdx.y * BM + warp_row_idx * WM + sub_row_idx * WSUBM + thread_row_in_warp * TM + thread_row_idx;
                    int col_in_gc = blockIdx.x * BN + warp_col_idx * WN + sub_col_idx * WSUBN + thread_col_in_warp * TN + thread_col_idx;
                    c[row_in_gc * N + col_in_gc] = reg_c[tmp_row * WITERN * TN + tmp_col];
                }
        }
}