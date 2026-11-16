// Integer-only, device-resident sink transformations. No score or input payload
// is downloaded. Bounded chunks feed stable radix ordering and streaming rows.
#pragma once
#include "hamming.cuh"

#define SB_RS_COLS 32
#define SB_RS_THREADS 256
#define SB_RS_NONE (~0ull)

struct SbResidentEntry {
    sb_u64 i, j;
    sb_i64 key;
    sb_u64 work_id, source;
};
static_assert(sizeof(SbResidentEntry) == 40, "resident entry ABI");
struct SbResidentCols {
    sb_u64 count;
    sb_u64 src[SB_RS_COLS], dst[SB_RS_COLS], width[SB_RS_COLS], mask[SB_RS_COLS];
};
static_assert(sizeof(SbResidentCols) == 1032, "resident column ABI");

__device__ __forceinline__ sb_i64 sb_rs_key(sb_u64 address, sb_u64 width, sb_u64 at) {
    return width == 4 ? (sb_i64)((const int*)address)[at] : ((const sb_i64*)address)[at];
}
__device__ __forceinline__ bool sb_rs_pass(sb_i64 key, sb_u64 has_cutoff, sb_i64 cutoff) {
    return !has_cutoff || key <= cutoff;
}
// Little-endian words of `width` bytes. Naturally aligned 4- and 8-byte words
// move in one access, which writes the same bytes as the byte loops.
__device__ __forceinline__ void sb_rs_word(sb_u64 address, sb_u64 width, sb_u64 at, sb_u64 bits) {
    sb_u8* out = (sb_u8*)address + at * width;
    const sb_u64 place = address + at * width;
    if (width == 4 && (place & 3ull) == 0) {
        const unsigned value = (unsigned)bits;
        memcpy(__builtin_assume_aligned(out, 4), &value, sizeof(value));
        return;
    }
    if (width == 8 && (place & 7ull) == 0) {
        memcpy(__builtin_assume_aligned(out, 8), &bits, sizeof(bits));
        return;
    }
    for (sb_u64 byte = 0; byte < width; ++byte) out[byte] = (sb_u8)(bits >> (8 * byte));
}
__device__ __forceinline__ void sb_rs_copy_word(sb_u64 dst, sb_u64 src, sb_u64 width,
                                               sb_u64 to, sb_u64 from) {
    sb_u8* output = (sb_u8*)dst + to * width;
    const sb_u8* input = (const sb_u8*)src + from * width;
    if (width == 4 && (((sb_u64)output | (sb_u64)input) & 3ull) == 0) {
        unsigned value;
        memcpy(&value, __builtin_assume_aligned(input, 4), sizeof(value));
        memcpy(__builtin_assume_aligned(output, 4), &value, sizeof(value));
        return;
    }
    if (width == 8 && (((sb_u64)output | (sb_u64)input) & 7ull) == 0) {
        sb_u64 value;
        memcpy(&value, __builtin_assume_aligned(input, 8), sizeof(value));
        memcpy(__builtin_assume_aligned(output, 8), &value, sizeof(value));
        return;
    }
    for (sb_u64 byte = 0; byte < width; ++byte) output[byte] = input[byte];
}

// All potential slots are initialized, including rejected and diagonal mirror
// slots; the inclusive prefix then produces a compact, stable sequence.
extern "C" __global__ void sb_resident_build(
    SbSpace space, sb_u64 keys, sb_u64 key_width, sb_u64 has_cutoff, sb_i64 cutoff,
    const sb_i64* thresholds, const sb_u8* active, sb_u64 mirror,
    sb_u64 keep_failures, SbResidentEntry* output, sb_u64* flags,
    const sb_u8* native_passed) {
    const sb_u64 stride = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < space.n; t += stride) {
        sb_u64 i, j;
        sb_decode(space, t, i, j);
        const sb_i64 key = sb_rs_key(keys, key_width, t);
        const bool passed = native_passed ? native_passed[t] != 0 : sb_rs_pass(key, has_cutoff, cutoff);
        const sb_u64 at = t * (mirror ? 2ull : 1ull);
        output[at] = {i, j, key, space.k0 + t, t};
        flags[at] = keep_failures || (passed && (!thresholds || (active[i] && key <= thresholds[i])));
        if (mirror) {
            output[at + 1] = {j, i, key, space.k0 + t, t};
            flags[at + 1] = i != j && passed && (!thresholds || (active[j] && key <= thresholds[j]));
        }
    }
}
extern "C" __global__ void sb_resident_compact(
    const SbResidentEntry* input, const sb_u64* flags, const sb_u64* prefix,
    sb_u64 n, sb_u64 source_base, SbResidentEntry* output) {
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 at = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; at < n; at += step) {
        if (!flags[at]) continue;
        const sb_u64 dst = prefix[at];
        SbResidentEntry entry = input[at];
        entry.source = source_base + dst;
        output[dst] = entry;
    }
}

__device__ __forceinline__ sb_u64 sb_rs_sort_word(const SbResidentEntry& value, sb_u64 field) {
    if (field == 0) return value.j;
    if (field == 1) return (sb_u64)value.key ^ (1ull << 63);
    if (field == 3) return (sb_u64)(unsigned int)(int)value.key ^ (1ull << 31);
    return value.i;
}
// One stable byte pass. Warp match ballots give each item its exact in-block
// position; the one scan of bucket-major histograms includes bucket bases.
extern "C" __global__ void sb_resident_radix_hist(
    const SbResidentEntry* input, sb_u64 n, sb_u64 field, sb_u64 shift, sb_u64* histogram) {
    __shared__ unsigned counts[8][256];
    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (unsigned w = 0; w < 8; ++w) counts[w][threadIdx.x] = 0;
    __syncthreads();
    const sb_u64 at = (sb_u64)blockIdx.x * SB_RS_THREADS + threadIdx.x;
    const unsigned active = __ballot_sync(0xffffffffu, at < n);
    if (at < n) {
        const unsigned digit = (unsigned)((sb_rs_sort_word(input[at], field) >> shift) & 255ull);
        const unsigned equal = __match_any_sync(active, digit);
        if (lane == (unsigned)(__ffs((int)equal) - 1)) counts[warp][digit] = __popc(equal);
    }
    __syncthreads();
    unsigned total = 0;
    for (unsigned w = 0; w < 8; ++w) total += counts[w][threadIdx.x];
    histogram[(sb_u64)threadIdx.x * gridDim.x + blockIdx.x] = total;
}
extern "C" __global__ void sb_resident_radix_scatter(
    const SbResidentEntry* input, sb_u64 n, sb_u64 field, sb_u64 shift,
    const sb_u64* prefix, SbResidentEntry* output) {
    __shared__ unsigned counts[8][256];
    const unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (unsigned w = 0; w < 8; ++w) counts[w][threadIdx.x] = 0;
    __syncthreads();
    const sb_u64 at = (sb_u64)blockIdx.x * SB_RS_THREADS + threadIdx.x;
    const unsigned active = __ballot_sync(0xffffffffu, at < n);
    unsigned digit = 0, equal = 0;
    if (at < n) {
        digit = (unsigned)((sb_rs_sort_word(input[at], field) >> shift) & 255ull);
        equal = __match_any_sync(active, digit);
        if (lane == (unsigned)(__ffs((int)equal) - 1)) counts[warp][digit] = __popc(equal);
    }
    __syncthreads();
    if (at < n) {
        unsigned before = __popc(equal & ((1u << lane) - 1u));
        for (unsigned w = 0; w < warp; ++w) before += counts[w][digit];
        output[prefix[(sb_u64)digit * gridDim.x + blockIdx.x] + before] = input[at];
    }
}

extern "C" __global__ void sb_resident_fields(
    const SbResidentEntry* entries, sb_u64 n, SbResidentCols columns,
    sb_u64 from_chunk, sb_u64 chunk_start, sb_u64 destination_start,
    sb_u64 has_cutoff, sb_i64 cutoff) {
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 at = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; at < n; at += step) {
        const SbResidentEntry entry = entries[at];
        const sb_u64 src = from_chunk ? entry.work_id - chunk_start : entry.source;
        const bool pass = sb_rs_pass(entry.key, has_cutoff, cutoff);
        for (sb_u64 col = 0; col < columns.count; ++col) {
            if (pass) sb_rs_copy_word(columns.dst[col], columns.src[col], columns.width[col], destination_start + at, src);
            else sb_rs_word(columns.dst[col], columns.width[col], destination_start + at, columns.mask[col]);
        }
    }
}
extern "C" __global__ void sb_resident_unpack(
    const SbResidentEntry* entries, sb_u64 n, sb_u64 destination_start,
    sb_u64 ids, sb_u64 rows, sb_u64 partners, sb_u64 keys, sb_u64 key_width,
    sb_u64 passed, sb_u64 has_cutoff, sb_i64 cutoff, sb_i64 mask) {
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 at = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; at < n; at += step) {
        const SbResidentEntry e = entries[at];
        const sb_u64 dst = destination_start + at;
        const bool pass = sb_rs_pass(e.key, has_cutoff, cutoff);
        if (ids) ((sb_u64*)ids)[dst] = e.work_id;
        if (rows) ((sb_i64*)rows)[dst] = (sb_i64)e.i;
        if (partners) ((sb_i64*)partners)[dst] = (sb_i64)e.j;
        if (keys) sb_rs_word(keys, key_width, dst, (sb_u64)(pass ? e.key : mask));
        if (passed) ((sb_u8*)passed)[dst] = pass;
    }
}
extern "C" __global__ void sb_resident_fill(sb_u64 address, sb_u64 n, sb_u64 width, sb_u64 bits) {
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 at = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; at < n; at += step)
        sb_rs_word(address, width, at, bits);
}
extern "C" __global__ void sb_resident_dense(
    SbSpace space, sb_u64 keys, sb_u64 key_width, SbResidentCols columns,
    sb_u64 dst_keys, sb_u64 passed, sb_u64 row0, sb_u64 col0,
    sb_u64 rows, sb_u64 cols, sb_u64 mirror) {
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < space.n; t += step) {
        sb_u64 i, j;
        sb_decode(space, t, i, j);
        const sb_i64 key = sb_rs_key(keys, key_width, t);
        for (unsigned side = 0; side < (mirror && i != j ? 2u : 1u); ++side) {
            const sb_u64 r = side ? j : i, c = side ? i : j;
            if (r < row0 || c < col0 || r - row0 >= rows || c - col0 >= cols) continue;
            const sb_u64 at = (r - row0) * cols + c - col0;
            sb_rs_word(dst_keys, key_width, at, (sb_u64)key);
            ((sb_u8*)passed)[at] = 1;
            for (sb_u64 col = 0; col < columns.count; ++col)
                sb_rs_copy_word(columns.dst[col], columns.src[col], columns.width[col], at, t);
        }
    }
}

// Row reductions combine a warp's contributions per contiguous run of equal
// rows before the integer atomics (levenshtein-cuda.md §14). A lane joins its
// lower neighbour's run only when both name the same row, so a run never mixes
// rows; each lane's contribution reaches exactly one run head, which applies
// the run's integer min and count. Integer atomicMin/atomicAdd commute, so the
// row arrays are the same integers as with one atomic per pair. The loops run
// warp-uniformly over the chunk (blockDim.x is a multiple of 32); lanes past
// the chunk end and failing pairs contribute the neutral value and count 0.
template <class V>
__device__ __forceinline__ void sb_rs_row_reduce(sb_u64 row, V value, sb_u64 count, V* low,
                                                 sb_u64* sum) {
    const unsigned lane = threadIdx.x & 31u;
    const sb_u64 below = __shfl_up_sync(0xffffffffu, row, 1);
    const bool head = lane == 0 || below != row;
    const unsigned heads = __ballot_sync(0xffffffffu, head);
    const unsigned run = (unsigned)__popc(heads & (0xffffffffu >> (31u - lane)));
#pragma unroll
    for (unsigned d = 1; d < 32; d <<= 1) {
        const V v = __shfl_down_sync(0xffffffffu, value, d);
        const sb_u64 c = __shfl_down_sync(0xffffffffu, count, d);
        const unsigned r = __shfl_down_sync(0xffffffffu, run, d);
        if (lane + d < 32u && r == run) {
            value = v < value ? v : value;
            count += c;
        }
    }
    if (head && count != 0) {
        atomicMin(&low[row], value);
        atomicAdd(&sum[row], count);
    }
}
extern "C" __global__ void sb_resident_row_min(
    SbSpace space, sb_u64 keys, sb_u64 key_width, sb_u64 has_cutoff, sb_i64 cutoff,
    sb_u64 mirror, sb_i64* best, sb_u64* counts) {
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    const sb_u64 lane = threadIdx.x & 31u;
    for (sb_u64 base = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x - lane; base < space.n;
         base += step) {
        const sb_u64 t = base + lane;
        sb_u64 i = ~0ull, j = ~0ull;
        sb_i64 key = 0x7fffffffffffffffll;
        bool pass = false;
        if (t < space.n) {
            key = sb_rs_key(keys, key_width, t);
            pass = sb_rs_pass(key, has_cutoff, cutoff);
            sb_decode(space, t, i, j);
        }
        const sb_i64 value = pass ? key : 0x7fffffffffffffffll;
        sb_rs_row_reduce<sb_i64>(i, value, pass ? 1ull : 0ull, best, counts);
        if (mirror) {
            const bool other = pass && i != j;
            sb_rs_row_reduce<sb_i64>(other ? j : ~0ull, other ? key : 0x7fffffffffffffffll,
                                     other ? 1ull : 0ull, best, counts);
        }
    }
}
extern "C" __global__ void sb_resident_row_arg(
    SbSpace space, sb_u64 keys, sb_u64 key_width, sb_u64 has_cutoff, sb_i64 cutoff,
    sb_u64 mirror, const sb_i64* best, sb_u64* ties, sb_u64* arg) {
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    const sb_u64 lane = threadIdx.x & 31u;
    for (sb_u64 base = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x - lane; base < space.n;
         base += step) {
        const sb_u64 t = base + lane;
        sb_u64 i = ~0ull, j = ~0ull;
        bool tie = false, mirrored = false;
        if (t < space.n) {
            const sb_i64 key = sb_rs_key(keys, key_width, t);
            sb_decode(space, t, i, j);
            if (sb_rs_pass(key, has_cutoff, cutoff)) {
                tie = key == best[i];
                mirrored = mirror && i != j && key == best[j];
            }
        }
        // The smallest tied partner: an unsigned min over partner indices.
        sb_rs_row_reduce<sb_u64>(i, tie ? j : ~0ull, tie ? 1ull : 0ull, arg, ties);
        if (mirror) {
            sb_rs_row_reduce<sb_u64>(mirrored ? j : ~0ull, mirrored ? i : ~0ull,
                                     mirrored ? 1ull : 0ull, arg, ties);
        }
    }
}
extern "C" __global__ void sb_resident_row_finish(
    const sb_i64* best, const sb_u64* counts, const sb_u64* ties, const sb_u64* arg,
    sb_u64 n, sb_u64 best_out, sb_u64 key_width, int* ties_out, sb_i64* arg_out,
    sb_i64* count_out, sb_u64* error) {
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 row = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; row < n; row += step) {
        if (ties[row] > 2147483647ull) atomicOr(error, 1ull);
        if (count_out && counts[row] > 9223372036854775807ull) atomicOr(error, 2ull);
        sb_rs_word(best_out, key_width, row, (sb_u64)best[row]);
        ties_out[row] = (int)ties[row];
        arg_out[row] = arg[row] == SB_RS_NONE ? -1ll : (sb_i64)arg[row];
        if (count_out) count_out[row] = (sb_i64)counts[row];
    }
}

__device__ __forceinline__ sb_u64 sb_rs_row_begin(const SbResidentEntry* entries, sb_u64 n, sb_u64 row) {
    sb_u64 lo = 0, hi = n;
    while (lo < hi) { const sb_u64 mid = lo + (hi - lo) / 2; if (entries[mid].i < row) lo = mid + 1; else hi = mid; }
    return lo;
}
extern "C" __global__ void sb_resident_topk_finish(
    const SbResidentEntry* entries, sb_u64 n, sb_u64 rows, sb_u64 k,
    int* ranks, sb_i64* offsets, sb_u8* ties, sb_u64* tie_rows, sb_u64* error) {
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 row = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; row <= rows; row += step) {
        const sb_u64 first = sb_rs_row_begin(entries, n, row);
        offsets[row] = (sb_i64)first;
        if (row < rows) {
            const sb_u64 count = sb_rs_row_begin(entries, n, row + 1) - first;
            ties[row] = count > k;
            if (count > k) atomicAdd(tie_rows, 1ull);
        }
    }
    for (sb_u64 at = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; at < n; at += step) {
        const SbResidentEntry e = entries[at];
        const sb_u64 first = sb_rs_row_begin(entries, n, e.i);
        sb_u64 lo = first, hi = at;
        while (lo < hi) { const sb_u64 mid = lo + (hi - lo) / 2; if (entries[mid].key < e.key) lo = mid + 1; else hi = mid; }
        const sb_u64 rank = lo - first + 1;
        if (rank > 2147483647ull) atomicOr(error, 4ull);
        ranks[at] = (int)rank;
    }
}

#include "resident_topk.cuh"
