// K1 RowStats lse_dg_kcal on the resident path (docs/wiki/gpu/duplex-cuda.md §6.1,
// docs/wiki/thermo/duplex-mfe.md §8.6). Integer only: a window reduction over the
// chunk's looked-up engine keys and a gather of each work item's term from the
// host-built table of W(D). The device evaluates no exponential and performs no
// floating-point operation (AGENTS §3); the row sums are pf_resident.cuh's.
#pragma once
#include "hamming.cuh"

// The K1 engine key of an empty structure space (duplex-mfe.md §7, QK9); INT32_MAX is
// the engine's mask. Neither is ever looked up: its term is exactly 0 (LK2).
#define SB_K1_LSE_SENTINEL 2147483646

__device__ __forceinline__ bool sb_k1_lse_looked_up(int key, int has_cutoff, sb_i64 cutoff) {
    return key < SB_K1_LSE_SENTINEL && !(has_cutoff && (sb_i64)key > cutoff);
}

// bounds[0] and bounds[1]: the smallest and largest key of the chunk that is looked
// up (not the sentinel and, with a cutoff, key <= cutoff). The host initialises them to
// (INT32_MAX, INT32_MIN); a chunk without such an item leaves them unchanged.
extern "C" __global__ void sb_k1_lse_window(const int* keys, sb_u64 n, int has_cutoff,
                                            sb_i64 cutoff, int* bounds) {
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < n; t += step) {
        const int key = keys[t];
        if (!sb_k1_lse_looked_up(key, has_cutoff, cutoff)) continue;
        atomicMin(&bounds[0], key);
        atomicMax(&bounds[1], key);
    }
}

// For every work item t: the int64 key (sign-extended) and the term record (M, e),
// value M * 2^(e - 52), interleaved as U64[n, 2]: (0, 0) for an item that is not looked
// up, else table entry key - lo. An item outside the table sets status 1.
extern "C" __global__ void sb_k1_lse_gather(const int* keys, sb_u64 n, int has_cutoff,
                                            sb_i64 cutoff, const sb_u64* table, sb_i64 lo,
                                            sb_u64 len, sb_i64* keys64, sb_u64* terms,
                                            unsigned* status) {
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < n; t += step) {
        const int key = keys[t];
        sb_u64 m = 0, e = 0;
        if (sb_k1_lse_looked_up(key, has_cutoff, cutoff)) {
            const sb_i64 at = (sb_i64)key - lo;
            if (at < 0 || (sb_u64)at >= len) {
                atomicMax(status, 1u);
            } else {
                m = table[2 * (sb_u64)at];
                e = table[2 * (sb_u64)at + 1];
            }
        }
        keys64[t] = (sb_i64)key;
        terms[2 * t] = m;
        terms[2 * t + 1] = e;
    }
}
