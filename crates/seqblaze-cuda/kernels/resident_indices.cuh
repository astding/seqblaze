// Device-native integer index normalization, validation and explicit-pair sort.
// Only fixed-size error/count records are consumed by the host. The source
// sequence tables and the pair-index payload never leave the device.
#pragma once
#include "hamming.cuh"

#define SB_INDEX_THREADS 256
#define SB_INDEX_NONE (~0ull)

struct SbIndexView {
    sb_u64 address, n;
    sb_i64 stride;
    sb_u64 width, is_signed;
};
struct SbIndexPair {
    sb_i64 i, j;
};
static_assert(sizeof(SbIndexView) == 40, "index view ABI");
static_assert(offsetof(SbIndexView, stride) == 16, "index stride ABI");
static_assert(sizeof(SbIndexPair) == 16, "index pair ABI");

extern "C" __global__ void sb_index_normalize(
    SbIndexView input, sb_u64 output_address, sb_u64 first_bad_address) {
    sb_u64* output = (sb_u64*)output_address;
    sb_u64* first_bad = (sb_u64*)first_bad_address;
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 at = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x;
         at < input.n; at += step) {
        // Unsigned address arithmetic handles a checked negative stride without
        // signed multiplication overflow. Byte reads also allow unaligned views.
        const sb_u8* bytes = (const sb_u8*)(input.address + at * (sb_u64)input.stride);
        sb_u64 bits = 0;
        for (sb_u64 byte = 0; byte < input.width; ++byte)
            bits |= (sb_u64)bytes[byte] << (8 * byte);
        if (input.is_signed && input.width < 8 &&
            (bits & (1ull << (8 * input.width - 1))))
            bits |= SB_INDEX_NONE << (8 * input.width);
        if (!input.is_signed && input.width == 8 && (bits >> 63)) {
            atomicMin(first_bad, at);
            bits = 0;
        }
        output[at] = bits;
    }
}

extern "C" __global__ void sb_index_validate(
    sb_u64 left_address, sb_u64 right_address, sb_u64 n,
    sb_u64 rows, sb_u64 columns, sb_u64 summary_address) {
    const sb_i64* left = (const sb_i64*)left_address;
    const sb_i64* right = (const sb_i64*)right_address;
    sb_u64* summary = (sb_u64*)summary_address;
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 at = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; at < n; at += step) {
        const sb_i64 i = left[at], j = right[at];
        if (i < 0 || j < 0 || (sb_u64)i >= rows || (sb_u64)j >= columns)
            atomicMin(summary, at);
    }
}
// The minimum is complete at this kernel boundary, so no later occurrence can
// overwrite the first occurrence's values through an atomic winner race.
extern "C" __global__ void sb_index_error(
    sb_u64 left_address, sb_u64 right_address, sb_u64 summary_address) {
    if (blockIdx.x || threadIdx.x) return;
    sb_u64* summary = (sb_u64*)summary_address;
    const sb_u64 at = summary[0];
    if (at == SB_INDEX_NONE) return;
    summary[1] = ((const sb_u64*)left_address)[at];
    summary[2] = ((const sb_u64*)right_address)[at];
}
extern "C" __global__ void sb_index_pack(
    sb_u64 left_address, sb_u64 right_address, sb_u64 n, SbIndexPair* output) {
    const sb_i64* left = (const sb_i64*)left_address;
    const sb_i64* right = (const sb_i64*)right_address;
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 at = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; at < n; at += step)
        output[at] = {left[at], right[at]};
}

__device__ __forceinline__ unsigned sb_index_radix_digit(
    sb_u64 input, sb_u64 at, sb_u64 field, sb_u64 shift,
    sb_u64 key_address, sb_u64 is_signed, sb_u64 record_words) {
    sb_u64 bits;
    if (record_words == 2) {
        const SbIndexPair pair = ((const SbIndexPair*)input)[at];
        bits = (sb_u64)(field == 0 ? pair.j : pair.i);
    } else {
        const sb_u64 index = ((const sb_u64*)input)[at];
        bits = ((const sb_u64*)key_address)[index];
    }
    if (is_signed) bits ^= 1ull << 63;
    return (unsigned)((bits >> shift) & 255ull);
}
// The same stable byte pass handles pair records and indirect U64 permutations.
// A bucket-major prefix includes global bucket bases and all preceding blocks.
__device__ __forceinline__ void sb_index_hist_body(
    sb_u64 input, sb_u64 n, sb_u64 field, sb_u64 shift,
    sb_u64 key_address, sb_u64 is_signed, sb_u64 record_words,
    sb_u64* histogram, unsigned counts[8][256]) {
    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (unsigned w = 0; w < 8; ++w) counts[w][threadIdx.x] = 0;
    __syncthreads();
    const sb_u64 at = (sb_u64)blockIdx.x * SB_INDEX_THREADS + threadIdx.x;
    const unsigned active = __ballot_sync(0xffffffffu, at < n);
    if (at < n) {
        const unsigned digit = sb_index_radix_digit(
            input, at, field, shift, key_address, is_signed, record_words);
        const unsigned equal = __match_any_sync(active, digit);
        if (lane == (unsigned)(__ffs((int)equal) - 1)) counts[warp][digit] = __popc(equal);
    }
    __syncthreads();
    unsigned total = 0;
    for (unsigned w = 0; w < 8; ++w) total += counts[w][threadIdx.x];
    histogram[(sb_u64)threadIdx.x * gridDim.x + blockIdx.x] = total;
}
__device__ __forceinline__ void sb_index_scatter_body(
    sb_u64 input, sb_u64 n, sb_u64 field, sb_u64 shift,
    sb_u64 key_address, sb_u64 is_signed, sb_u64 record_words,
    const sb_u64* prefix, sb_u64 output, unsigned counts[8][256]) {
    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (unsigned w = 0; w < 8; ++w) counts[w][threadIdx.x] = 0;
    __syncthreads();
    const sb_u64 at = (sb_u64)blockIdx.x * SB_INDEX_THREADS + threadIdx.x;
    const unsigned active = __ballot_sync(0xffffffffu, at < n);
    unsigned digit = 0, equal = 0;
    if (at < n) {
        digit = sb_index_radix_digit(
            input, at, field, shift, key_address, is_signed, record_words);
        equal = __match_any_sync(active, digit);
        if (lane == (unsigned)(__ffs((int)equal) - 1)) counts[warp][digit] = __popc(equal);
    }
    __syncthreads();
    if (at < n) {
        unsigned before = __popc(equal & ((1u << lane) - 1u));
        for (unsigned w = 0; w < warp; ++w) before += counts[w][digit];
        const sb_u64 destination = prefix[(sb_u64)digit * gridDim.x + blockIdx.x] + before;
        if (record_words == 2) ((SbIndexPair*)output)[destination] = ((const SbIndexPair*)input)[at];
        else ((sb_u64*)output)[destination] = ((const sb_u64*)input)[at];
    }
}
extern "C" __global__ void sb_index_radix_hist(
    const SbIndexPair* input, sb_u64 n, sb_u64 field, sb_u64 shift, sb_u64* histogram) {
    __shared__ unsigned counts[8][256];
    sb_index_hist_body((sb_u64)input, n, field, shift, 0, 1, 2, histogram, counts);
}
extern "C" __global__ void sb_index_radix_scatter(
    const SbIndexPair* input, sb_u64 n, sb_u64 field, sb_u64 shift,
    const sb_u64* prefix, SbIndexPair* output) {
    __shared__ unsigned counts[8][256];
    sb_index_scatter_body((sb_u64)input, n, field, shift, 0, 1, 2, prefix, (sb_u64)output, counts);
}
extern "C" __global__ void sb_index_order_init(sb_u64 n, sb_u64 output) {
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 at = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; at < n; at += step)
        ((sb_u64*)output)[at] = at;
}
extern "C" __global__ void sb_index_order_hist(
    sb_u64 input, sb_u64 n, sb_u64 key_address, sb_u64 is_signed,
    sb_u64 shift, sb_u64* histogram) {
    __shared__ unsigned counts[8][256];
    sb_index_hist_body(input, n, 0, shift, key_address, is_signed, 1, histogram, counts);
}
extern "C" __global__ void sb_index_order_scatter(
    sb_u64 input, sb_u64 n, sb_u64 key_address, sb_u64 is_signed,
    sb_u64 shift, const sb_u64* prefix, sb_u64 output) {
    __shared__ unsigned counts[8][256];
    sb_index_scatter_body(input, n, 0, shift, key_address, is_signed, 1, prefix, output, counts);
}
extern "C" __global__ void sb_index_unique(
    const SbIndexPair* input, sb_u64 n, sb_u64* flags) {
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 at = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; at < n; at += step)
        flags[at] = at == 0 || input[at].i != input[at - 1].i || input[at].j != input[at - 1].j;
}
extern "C" __global__ void sb_index_compact(
    const SbIndexPair* input, const sb_u64* flags, const sb_u64* prefix,
    sb_u64 n, sb_u64 left_address, sb_u64 right_address) {
    sb_i64* left = (sb_i64*)left_address;
    sb_i64* right = (sb_i64*)right_address;
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 at = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; at < n; at += step) {
        if (!flags[at]) continue;
        const sb_u64 destination = prefix[at];
        left[destination] = input[at].i;
        if (right_address) right[destination] = input[at].j;
    }
}
