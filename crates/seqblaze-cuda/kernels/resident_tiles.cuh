// Current-tile geometry and once-only indexing of explicit occurrences.
#pragma once
#include "hamming.cuh"

struct SbTileRect {
    sb_u64 row, col, rows, cols, domain, self_pairs;
};
static_assert(sizeof(SbTileRect) == 48, "tile rectangle ABI");

extern "C" __global__ void sb_tile_seed(
    sb_u64 left, sb_u64 right, sb_u64 first, sb_u64 n, sb_u64 height, sb_u64 width,
    sb_u64 out_i, sb_u64 out_j, sb_u64 tile_rows, sb_u64 tile_cols) {
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 at = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; at < n; at += step) {
        const sb_i64 i = ((const sb_i64*)left)[first + at];
        const sb_i64 j = ((const sb_i64*)right)[first + at];
        ((sb_i64*)out_i)[at] = i;
        ((sb_i64*)out_j)[at] = j;
        ((sb_u64*)tile_rows)[at] = (sb_u64)i / height;
        ((sb_u64*)tile_cols)[at] = (sb_u64)j / width;
    }
}
extern "C" __global__ void sb_tile_order(
    sb_u64 permutation, sb_u64 n, sb_u64 left, sb_u64 right, sb_u64 rows, sb_u64 cols,
    sb_u64 out_i, sb_u64 out_j, sb_u64 out_row, sb_u64 out_col,
    sb_u64 first_flags, sb_u64 duplicate_count) {
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 at = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; at < n; at += step) {
        const sb_u64 source = ((const sb_u64*)permutation)[at];
        const sb_i64 i = ((const sb_i64*)left)[source], j = ((const sb_i64*)right)[source];
        ((sb_i64*)out_i)[at] = i;
        ((sb_i64*)out_j)[at] = j;
        ((sb_u64*)out_row)[at] = ((const sb_u64*)rows)[source];
        ((sb_u64*)out_col)[at] = ((const sb_u64*)cols)[source];
        bool first = true;
        if (at) {
            const sb_u64 previous = ((const sb_u64*)permutation)[at - 1];
            first = i != ((const sb_i64*)left)[previous] || j != ((const sb_i64*)right)[previous];
        }
        ((sb_u64*)first_flags)[at] = first;
        if (!first) atomicAdd((sb_u64*)duplicate_count, 1ull);
    }
}
__device__ __forceinline__ bool sb_tile_before(
    sb_u64 row, sb_u64 col, sb_u64 wanted_row, sb_u64 wanted_col) {
    return row < wanted_row || (row == wanted_row && col < wanted_col);
}
extern "C" __global__ void sb_tile_bounds(
    sb_u64 row_address, sb_u64 col_address, sb_u64 n,
    sb_u64 row, sb_u64 col, sb_u64 output) {
    if (threadIdx.x || blockIdx.x) return;
    const sb_u64* rows = (const sb_u64*)row_address;
    const sb_u64* cols = (const sb_u64*)col_address;
    sb_u64 lo = 0, hi = n;
    while (lo < hi) {
        const sb_u64 mid = lo + (hi - lo) / 2;
        if (sb_tile_before(rows[mid], cols[mid], row, col)) lo = mid + 1;
        else hi = mid;
    }
    ((sb_u64*)output)[0] = lo;
    hi = n;
    while (lo < hi) {
        const sb_u64 mid = lo + (hi - lo) / 2;
        if (!sb_tile_before(row, col, rows[mid], cols[mid])) lo = mid + 1;
        else hi = mid;
    }
    ((sb_u64*)output)[1] = lo;
}
__device__ __forceinline__ bool sb_tile_contains(const SbTileRect& tile, sb_u64 i, sb_u64 j) {
    return i >= tile.row && j >= tile.col && i - tile.row < tile.rows && j - tile.col < tile.cols;
}
extern "C" __global__ void sb_tile_classify(
    SbTileRect tile, sb_u64 first, sb_u64 n, sb_u64 left, sb_u64 right,
    sb_u64 flags, sb_u64 summary) {
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 at = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; at < n; at += step) {
        const sb_u64 position = first + at;
        const sb_u64 i = tile.domain == 0 ? (tile.row > tile.col ? tile.row : tile.col) + position
                                         : tile.row + position / tile.cols;
        const sb_u64 j = tile.domain == 0 ? i : tile.col + position % tile.cols;
        bool keep = true;
        if (tile.domain == 2) {
            keep = !((i == j && !tile.self_pairs) || (i > j && sb_tile_contains(tile, j, i)));
        } else if (tile.domain == 3) {
            keep = i < j || (i == j && tile.self_pairs);
        } else if (tile.domain == 4) {
            keep = i != j || tile.self_pairs;
        }
        ((sb_i64*)left)[at] = (sb_i64)i;
        ((sb_i64*)right)[at] = (sb_i64)j;
        ((sb_u64*)flags)[at] = keep;
        if (keep && tile.domain == 2 && i > j) atomicAdd((sb_u64*)summary + 1, 1ull);
    }
}
extern "C" __global__ void sb_tile_count(sb_u64 prefix, sb_u64 n, sb_u64 summary) {
    if (!threadIdx.x && !blockIdx.x) ((sb_u64*)summary)[0] = ((const sb_u64*)prefix)[n];
}
extern "C" __global__ void sb_tile_compact(
    sb_u64 left, sb_u64 right, sb_u64 flags, sb_u64 prefix, sb_u64 n,
    sb_u64 out_i, sb_u64 out_j) {
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 at = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; at < n; at += step) {
        if (!((const sb_u64*)flags)[at]) continue;
        const sb_u64 destination = ((const sb_u64*)prefix)[at];
        ((sb_i64*)out_i)[destination] = ((const sb_i64*)left)[at];
        ((sb_i64*)out_j)[destination] = ((const sb_i64*)right)[at];
    }
}
extern "C" __global__ void sb_tile_copy_flags(sb_u64 source, sb_u64 first, sb_u64 n, sb_u64 flags) {
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 at = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; at < n; at += step)
        ((sb_u64*)flags)[at] = ((const sb_u64*)source)[first + at];
}
extern "C" __global__ void sb_tile_gather(
    sb_u64 source, sb_u64 width, sb_u64 first, sb_u64 n,
    sb_u64 flags, sb_u64 prefix, sb_u64 output) {
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 at = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; at < n; at += step) {
        if (!((const sb_u64*)flags)[at]) continue;
        const sb_u64 destination = ((const sb_u64*)prefix)[at];
        for (sb_u64 byte = 0; byte < width; ++byte)
            ((sb_u8*)output)[destination * width + byte] = ((const sb_u8*)source)[(first + at) * width + byte];
    }
}
