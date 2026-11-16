// The generic unary data path's device sinks (tm-cuda.md §5.2): Hits compaction of
// per-sequence columns by a `passed` byte column. Written from that page only.
//
// G1's Hits method: per-block pass counts (warp ballot and popc in fixed in-block
// order), an exclusive scan of the block counts on the host, then an ordered write at
// the block offset plus the in-block rank. Output is born in sequence-index order for
// any block or chunk size. Integers only.
#pragma once
#include "device_rows.cuh"

typedef unsigned char sb_u8;
typedef long long sb_i64;
typedef unsigned long long sb_u64;

#define SB_UNARY_MAX_COLS 8

// One uploaded sequence table: concatenated raw bytes and int64 offsets (rows + 1).
struct SbUnaryTable {
    sb_u64 data;  // const u8*
    sb_u64 off;   // const i64*
    sb_u64 view;  // optional const SbDeviceRows*, zero for host packed rows
};

// Column descriptors for the Hits write: element c of a passing row t is copied from
// src[c] + t·width[c] to dst[c] + pos·width[c]. Widths are 4 or 8 bytes. Mirrors
// `SbUnaryCols` in src/unary.rs.
__device__ __forceinline__ SbSequence sb_unary_row(const SbUnaryTable& table, sb_u64 row) {
    return table.view ? sb_device_row(*(const SbDeviceRows*)table.view, row)
                      : sb_packed_row(table.data, table.off, row);
}

struct SbUnaryCols {
    sb_u64 count;
    sb_u64 src[SB_UNARY_MAX_COLS];
    sb_u64 dst[SB_UNARY_MAX_COLS];
    sb_u64 width[SB_UNARY_MAX_COLS];
};

// Hits step 1: pass count of each block's contiguous tile of blockDim.x sequences.
extern "C" __global__ void sb_unary_hits_count(const sb_u8* passed, sb_u64 n,
                                               sb_u64* block_counts) {
    __shared__ unsigned warp_counts[32];
    const sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x;
    const bool ok = t < n && passed[t] != 0;
    const unsigned ballot = __ballot_sync(0xffffffffu, ok);
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_counts[warp] = (unsigned)__popc(ballot);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        sb_u64 total = 0;
        for (unsigned w = 0; w < blockDim.x / 32u; ++w) {
            total += warp_counts[w];
        }
        block_counts[blockIdx.x] = total;
    }
}

// Hits step 3: a passing row t writes its columns and i = k0 + t at the block's
// exclusive offset plus its rank among the block's passing rows (warp, then lane).
extern "C" __global__ void sb_unary_hits_write(const sb_u8* passed, sb_u64 n, sb_u64 k0,
                                               const sb_u64* block_offsets, SbUnaryCols cols,
                                               sb_i64* out_i) {
    __shared__ unsigned warp_base[32];
    const sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x;
    const bool ok = t < n && passed[t] != 0;
    const unsigned ballot = __ballot_sync(0xffffffffu, ok);
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_base[warp] = (unsigned)__popc(ballot);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned running = 0;
        for (unsigned w = 0; w < blockDim.x / 32u; ++w) {
            const unsigned c = warp_base[w];
            warp_base[w] = running;
            running += c;
        }
    }
    __syncthreads();
    if (ok) {
        const unsigned below = ballot & ((1u << lane) - 1u);
        const sb_u64 pos = block_offsets[blockIdx.x] + warp_base[warp] + (unsigned)__popc(below);
        out_i[pos] = (sb_i64)(k0 + t);
        for (sb_u64 c = 0; c < cols.count && c < SB_UNARY_MAX_COLS; ++c) {
            if (cols.width[c] == 8) {
                ((sb_u64*)cols.dst[c])[pos] = ((const sb_u64*)cols.src[c])[t];
            } else {
                ((unsigned*)cols.dst[c])[pos] = ((const unsigned*)cols.src[c])[t];
            }
        }
    }
}
