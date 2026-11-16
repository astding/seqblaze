// Hamming kernel family for the seqblaze fatbin (milestone G1), plus the shared
// pair decoding and device sinks that levenshtein.cuh reuses.
// Spec: docs/wiki/gpu/cuda-backend.md and levenshtein-cuda.md. Written from those pages only.
//
// Integers only: no floating point, no float atomics. Results never depend on
// grid, block or piece size: each piece's count is exact, multi-piece pairs
// combine with integer atomicAdd, Hits output positions come from per-block
// counts plus an in-block rank, and RowStats uses integer atomicMin/atomicAdd
// in two sweeps separated by a kernel boundary.
#pragma once
#include "device_rows.cuh"
#include <string.h>

typedef unsigned char sb_u8;
typedef long long sb_i64;
typedef unsigned long long sb_u64;

#define SB_INT32_MAX 2147483647

// Pair-space kinds. Must match `Kind` in src/session.rs.
#define SB_ZIP 0ull   // item q -> (q, q)
#define SB_LIST 1ull  // item q -> (li[q], lj[q])
#define SB_RECT 2ull  // item q -> (r0 + q / width, c0 + q % width)
#define SB_TRI 3ull   // item q -> row segment search (within upper triangle)
#define SB_BOTH 4ull  // item q -> (q / width, r + [skip && r >= i]), r = q % width (oriented within, both)

// The structs below mirror #[repr(C)] structs in src/session.rs. Every field
// is 8 bytes wide so the layouts agree without padding rules; pointers travel
// as 64-bit device addresses.
struct SbTables {
    sb_u64 a_data;  // const u8*: concatenated raw bytes of table a
    sb_u64 a_off;   // const i64*: n_a + 1 offsets
    sb_u64 b_data;
    sb_u64 b_off;
    sb_u64 a_view;  // optional const SbDeviceRows*, zero for host-uploaded packed rows
    sb_u64 b_view;
};

__device__ __forceinline__ SbSequence sb_a_row(const SbTables& table, sb_u64 row) {
    return table.a_view ? sb_device_row(*(const SbDeviceRows*)table.a_view, row)
                        : sb_packed_row(table.a_data, table.a_off, row);
}
__device__ __forceinline__ SbSequence sb_b_row(const SbTables& table, sb_u64 row) {
    return table.b_view ? sb_device_row(*(const SbDeviceRows*)table.b_view, row)
                        : sb_packed_row(table.b_data, table.b_off, row);
}

static_assert(sizeof(SbTables) == 48, "SbTables ABI size");
static_assert(offsetof(SbTables, a_view) == 32 && offsetof(SbTables, b_view) == 40, "SbTables view offsets");

struct SbSpace {
    sb_u64 kind;
    sb_u64 k0;  // item number (work ID or request position) of chunk item 0
    sb_u64 n;   // items in this chunk
    sb_u64 r0;
    sb_u64 c0;
    sb_u64 width;
    sb_u64 li;     // const i64*, indexed by item number
    sb_u64 lj;     // const i64*
    sb_u64 seg_k;  // const u64*: first item number of each row segment
    sb_u64 seg_i;  // const i64*: row of each segment
    sb_u64 seg_j;  // const i64*: first column of each segment
    sb_u64 nseg;
    sb_u64 skip;   // SB_BOTH: 1 when the diagonal is excluded (no self pairs)
};

struct SbPieces {
    sb_u64 piece_bytes;  // P >= 1
    sb_u64 per_pair;     // prefix == 0: pieces per pair (uniform length, or 1 for a chunk
                         // of single-piece pairs)
    sb_u64 prefix;       // const u64* piece prefix over the chunk (n + 1), or 0
    sb_u64 total;        // pieces in this chunk
};

struct SbRows {
    sb_u64 mirror;  // also contribute (row j, partner i) when i != j
    sb_u64 first;   // const u8* first-occurrence mask by work ID, or 0
    sb_u64 best;    // int*, initialised to INT32_MAX
    sb_u64 count;   // u64*, initialised to 0
    sb_u64 nbest;   // u64*, initialised to 0
    sb_u64 arg;     // u64*, initialised to UINT64_MAX
};

__device__ __forceinline__ void sb_decode(const SbSpace& s, sb_u64 t, sb_u64& i, sb_u64& j) {
    const sb_u64 q = s.k0 + t;
    if (s.kind == SB_ZIP) {
        i = q;
        j = q;
    } else if (s.kind == SB_LIST) {
        i = (sb_u64)((const sb_i64*)s.li)[q];
        j = (sb_u64)((const sb_i64*)s.lj)[q];
    } else if (s.kind == SB_RECT || s.kind == SB_BOTH) {
        // Quotient and remainder by the row width; 32-bit division gives the
        // same integers when both operands fit 32 bits.
        sb_u64 row, r;
        if (((q | s.width) >> 32) == 0) {
            row = (unsigned)q / (unsigned)s.width;
            r = (unsigned)q % (unsigned)s.width;
        } else {
            row = q / s.width;
            r = q % s.width;
        }
        if (s.kind == SB_RECT) {
            i = s.r0 + row;
            j = s.c0 + r;
        } else {
            // Row width R = N - skip; j skips the diagonal (WorkSource::row_of, PairIter).
            i = row;
            j = r + ((s.skip != 0 && r >= i) ? 1ull : 0ull);
        }
    } else {
        // Last segment whose first item number is <= q.
        const sb_u64* seg_k = (const sb_u64*)s.seg_k;
        sb_u64 lo = 0, hi = s.nseg;
        while (hi - lo > 1) {
            const sb_u64 mid = lo + (hi - lo) / 2;
            if (seg_k[mid] <= q) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        i = (sb_u64)((const sb_i64*)s.seg_i)[lo];
        j = (sb_u64)((const sb_i64*)s.seg_j)[lo] + (q - seg_k[lo]);
    }
}

template <class T>
__device__ __forceinline__ bool sb_pass(T d, int has_cutoff, T cutoff) {
    return !has_cutoff || d <= cutoff;
}

// Piece p of a chunk: its chunk item t, its number within the pair and the
// pair's piece count (uniform per pair, or from the chunk's piece prefix).
__device__ __forceinline__ void sb_hamming_piece(const SbSpace& s, const SbPieces& pc, sb_u64 p,
                                                 sb_u64& t, sb_u64& piece, sb_u64& npieces) {
    if (pc.prefix == 0) {
        // One piece per pair needs no division.
        t = pc.per_pair == 1 ? p : p / pc.per_pair;
        piece = p - t * pc.per_pair;
        npieces = pc.per_pair;
    } else {
        // Last chunk item whose piece prefix is <= p (empty pairs own no piece).
        const sb_u64* prefix = (const sb_u64*)pc.prefix;
        sb_u64 lo = 0, hi = s.n;
        while (hi - lo > 1) {
            const sb_u64 mid = lo + (hi - lo) / 2;
            if (prefix[mid] <= p) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        t = lo;
        piece = p - prefix[t];
        npieces = prefix[t + 1] - prefix[t];
    }
}

// The two row views of piece `piece` of item t, from byte begin = piece * P,
// and its byte count: min(P, length - begin).
__device__ __forceinline__ sb_u64 sb_hamming_span(const SbTables& tb, const SbSpace& s,
                                                  const SbPieces& pc, sb_u64 t, sb_u64 piece,
                                                  SbSequence& pa, SbSequence& pb) {
    sb_u64 i, j;
    sb_decode(s, t, i, j);
    const SbSequence ra = sb_a_row(tb, i);
    const sb_u64 len = (sb_u64)ra.length;
    const sb_u64 begin = piece * pc.piece_bytes;
    const sb_u64 end = begin + pc.piece_bytes < len ? begin + pc.piece_bytes : len;
    pa = ra + begin;
    pb = sb_b_row(tb, j) + begin;
    return end - begin;
}

// Distance stage. One thread counts one piece of at most P bytes; `dist` was
// zeroed by the host. A single-piece pair stores its count; otherwise pieces
// add with integer atomicAdd, so the order of pieces is irrelevant. A
// grid-stride loop: any grid gives the same keys.
__device__ __forceinline__ void sb_hamming_pieces_body(const SbTables& tb, const SbSpace& s,
                                                       const SbPieces& pc, int* dist) {
    const sb_u64 stride = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 p = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; p < pc.total; p += stride) {
        sb_u64 t, piece, npieces;
        sb_hamming_piece(s, pc, p, t, piece, npieces);
        SbSequence pa, pb;
        const sb_u64 width = sb_hamming_span(tb, s, pc, t, piece, pa, pb);
        int count = 0;
        sb_u64 x = 0;
        // Bulk words require contiguous logical bytes in both row views.
        // Arbitrary positive, negative and zero strides retain scalar indexing.
        if (pa.stride == 1 && pb.stride == 1 &&
            ((pa.address | pb.address) & 3ull) == 0) {
            const sb_u8* left_data = (const sb_u8*)pa.address;
            const sb_u8* right_data = (const sb_u8*)pb.address;
            for (; width - x >= 4; x += 4) {
                unsigned left, right;
                // Copy object representations instead of aliasing byte storage
                // through an unsigned pointer. Each copy stays inside the piece.
                memcpy(&left, __builtin_assume_aligned(left_data + x, 4), sizeof(left));
                memcpy(&right, __builtin_assume_aligned(right_data + x, 4), sizeof(right));
                const unsigned z = left ^ right;
                // The low seven bits plus 127 never carry between bytes. A
                // byte's high bit is set here exactly when that byte is nonzero.
                const unsigned nonzero = (((z & 0x7f7f7f7fu) + 0x7f7f7f7fu) | z) & 0x80808080u;
                count += __popc(nonzero);
            }
        }
        for (; x < width; ++x) {
            count += pa[x] != pb[x] ? 1 : 0;
        }
        if (npieces == 1) {
            dist[t] = count;
        } else {
            atomicAdd(&dist[t], count);
        }
    }
}

// Group pieces (levenshtein-cuda.md §15): `group` lanes of one warp (a power
// of two, at most 32; launched only with whole warps) count one piece
// together. Member m compares the piece's bytes m, m + group, m + 2·group, …,
// so a group's loads are consecutive bytes of each row. The piece's count is
// the number of the group's bits in the warp's inequality ballots: the same
// count as the per-thread body, stored or added the same way.
__device__ __forceinline__ void sb_hamming_group_body(const SbTables& tb, const SbSpace& s,
                                                      const SbPieces& pc, int* dist,
                                                      unsigned group) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned member = lane & (group - 1u);
    const unsigned mine = (group == 32u ? 0xffffffffu : ((1u << group) - 1u)) << (lane - member);
    const sb_u64 per_warp = 32u / group;
    const sb_u64 warp = ((sb_u64)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const sb_u64 stride = (((sb_u64)gridDim.x * blockDim.x) >> 5) * per_warp;
    // `first` is uniform across the warp, so every lane runs every ballot.
    for (sb_u64 first = warp * per_warp; first < pc.total; first += stride) {
        const sb_u64 p = first + lane / group;
        sb_u64 t = 0, npieces = 1, width = 0;
        SbSequence pa = {0, 1, 0}, pb = {0, 1, 0};
        if (p < pc.total) {
            sb_u64 piece;
            sb_hamming_piece(s, pc, p, t, piece, npieces);
            width = sb_hamming_span(tb, s, pc, t, piece, pa, pb);
        }
        // The warp's longest piece in steps (a piece is at most P <= 2**31 - 1 bytes).
        unsigned steps = (unsigned)((width + group - 1u) / group);
#pragma unroll
        for (unsigned d = 16; d != 0; d >>= 1) {
            const unsigned other = __shfl_xor_sync(0xffffffffu, steps, d);
            steps = other > steps ? other : steps;
        }
        int count = 0;
        for (unsigned k = 0; k < steps; ++k) {
            const sb_u64 x = (sb_u64)k * group + member;
            const bool differ = x < width && pa[x] != pb[x];
            count += __popc(__ballot_sync(0xffffffffu, differ) & mine);
        }
        if (p < pc.total && member == 0) {
            if (npieces == 1) {
                dist[t] = count;
            } else {
                atomicAdd(&dist[t], count);
            }
        }
    }
}
extern "C" __global__ void sb_hamming_pieces(SbTables tb, SbSpace s, SbPieces pc, int* dist) {
    sb_hamming_pieces_body(tb, s, pc, dist);
}

// Each: passed = d <= c (always without a cutoff); a failing d becomes INT32_MAX,
// and so do its numeric detail fields when present (b_end INT64_MAX, n_ends
// INT32_MAX; levenshtein only, null pointers otherwise).
extern "C" __global__ void sb_each(int* dist, sb_u8* passed, sb_u64 n, int has_cutoff, int cutoff,
                                   sb_i64* b_end, int* n_ends) {
    const sb_u64 stride = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride) {
        const int d = dist[t];
        const bool ok = sb_pass(d, has_cutoff, cutoff);
        dist[t] = ok ? d : SB_INT32_MAX;
        if (passed != 0) {
            passed[t] = ok ? 1 : 0;
        }
        if (!ok && b_end != 0) {
            b_end[t] = 9223372036854775807ll;
        }
        if (!ok && n_ends != 0) {
            n_ends[t] = SB_INT32_MAX;
        }
    }
}

// Hits step 1: pass count of each block's contiguous tile of blockDim.x items. The
// bodies of the four sink kernels below are templates on the key type T: `int` (the G1
// names) and `long long` (the `_i64` names; tm-pairs-cuda.md §7); the int instances are
// the G1 kernels unchanged.
template <class T>
__device__ __forceinline__ void sb_hits_count_body(const T* dist, sb_u64 n, int has_cutoff,
                                                   T cutoff, sb_u64* block_counts) {
    __shared__ unsigned warp_counts[32];
    const sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x;
    const bool ok = t < n && sb_pass(dist[t], has_cutoff, cutoff);
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

extern "C" __global__ void sb_hits_count(const int* dist, sb_u64 n, int has_cutoff, int cutoff,
                                         sb_u64* block_counts) {
    sb_hits_count_body<int>(dist, n, has_cutoff, cutoff, block_counts);
}

extern "C" __global__ void sb_hits_count_i64(const sb_i64* dist, sb_u64 n, int has_cutoff,
                                             sb_i64 cutoff, sb_u64* block_counts) {
    sb_hits_count_body<sb_i64>(dist, n, has_cutoff, cutoff, block_counts);
}

// Hits step 3: write (i, j, d) at the block's exclusive offset plus the item's
// rank among passing items of the block (warp order, then lane order), and the
// numeric detail fields at the same position when present (levenshtein only).
template <class T>
__device__ __forceinline__ void sb_hits_write_body(const T* dist, sb_u64 n, int has_cutoff,
                                                   T cutoff, const sb_u64* block_offsets,
                                                   const SbSpace& s, sb_i64* out_i, sb_i64* out_j,
                                                   T* out_d, const sb_i64* b_end,
                                                   const int* n_ends, sb_i64* out_e, int* out_n) {
    __shared__ unsigned warp_base[32];
    const sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x;
    const bool ok = t < n && sb_pass(dist[t], has_cutoff, cutoff);
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
        sb_u64 i, j;
        sb_decode(s, t, i, j);
        out_i[pos] = (sb_i64)i;
        out_j[pos] = (sb_i64)j;
        out_d[pos] = dist[t];
        if (b_end != 0) {
            out_e[pos] = b_end[t];
        }
        if (n_ends != 0) {
            out_n[pos] = n_ends[t];
        }
    }
}

extern "C" __global__ void sb_hits_write(const int* dist, sb_u64 n, int has_cutoff, int cutoff,
                                         const sb_u64* block_offsets, SbSpace s, sb_i64* out_i,
                                         sb_i64* out_j, int* out_d, const sb_i64* b_end,
                                         const int* n_ends, sb_i64* out_e, int* out_n) {
    sb_hits_write_body<int>(dist, n, has_cutoff, cutoff, block_offsets, s, out_i, out_j, out_d,
                            b_end, n_ends, out_e, out_n);
}

extern "C" __global__ void sb_hits_write_i64(const sb_i64* dist, sb_u64 n, int has_cutoff,
                                             sb_i64 cutoff, const sb_u64* block_offsets, SbSpace s,
                                             sb_i64* out_i, sb_i64* out_j, sb_i64* out_d) {
    sb_hits_write_body<sb_i64>(dist, n, has_cutoff, cutoff, block_offsets, s, out_i, out_j, out_d,
                               0, 0, 0, 0);
}

__device__ __forceinline__ bool sb_participates(const SbRows& r, const SbSpace& s, sb_u64 t) {
    // A requested within diagonal is an ordinary partner of row i (A11, D87).
    if (r.first != 0 && ((const sb_u8*)r.first)[s.k0 + t] == 0) {
        return false;
    }
    return true;
}

// RowStats sweep 1: best distance and participating-partner count per row. `best` holds
// T (int: initialised to INT32_MAX; long long: to INT64_MAX, by sb_fill_i64); the
// long long instance uses the signed 64-bit atomicMin.
template <class T>
__device__ __forceinline__ void sb_rowstats_min_body(const T* dist, const SbSpace& s,
                                                     int has_cutoff, T cutoff, const SbRows& r) {
    const sb_u64 stride = (sb_u64)gridDim.x * blockDim.x;
    T* best = (T*)r.best;
    sb_u64* count = (sb_u64*)r.count;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < s.n; t += stride) {
        const T d = dist[t];
        if (!sb_pass(d, has_cutoff, cutoff)) {
            continue;
        }
        sb_u64 i, j;
        sb_decode(s, t, i, j);
        if (!sb_participates(r, s, t)) {
            continue;
        }
        atomicMin(&best[i], d);
        atomicAdd(&count[i], 1ull);
        if (r.mirror && i != j) {
            atomicMin(&best[j], d);
            atomicAdd(&count[j], 1ull);
        }
    }
}

extern "C" __global__ void sb_rowstats_min(const int* dist, SbSpace s, int has_cutoff, int cutoff,
                                           SbRows r) {
    sb_rowstats_min_body<int>(dist, s, has_cutoff, cutoff, r);
}

extern "C" __global__ void sb_rowstats_min_i64(const sb_i64* dist, SbSpace s, int has_cutoff,
                                               sb_i64 cutoff, SbRows r) {
    sb_rowstats_min_body<sb_i64>(dist, s, has_cutoff, cutoff, r);
}

// RowStats sweep 2 (distances recomputed): ties at the final best and the
// smallest partner among them.
template <class T>
__device__ __forceinline__ void sb_rowstats_arg_body(const T* dist, const SbSpace& s,
                                                     int has_cutoff, T cutoff, const SbRows& r) {
    const sb_u64 stride = (sb_u64)gridDim.x * blockDim.x;
    const T* best = (const T*)r.best;
    sb_u64* nbest = (sb_u64*)r.nbest;
    sb_u64* arg = (sb_u64*)r.arg;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < s.n; t += stride) {
        const T d = dist[t];
        if (!sb_pass(d, has_cutoff, cutoff)) {
            continue;
        }
        sb_u64 i, j;
        sb_decode(s, t, i, j);
        if (!sb_participates(r, s, t)) {
            continue;
        }
        if (d == best[i]) {
            atomicAdd(&nbest[i], 1ull);
            atomicMin(&arg[i], j);
        }
        if (r.mirror && i != j && d == best[j]) {
            atomicAdd(&nbest[j], 1ull);
            atomicMin(&arg[j], i);
        }
    }
}

extern "C" __global__ void sb_rowstats_arg(const int* dist, SbSpace s, int has_cutoff, int cutoff,
                                           SbRows r) {
    sb_rowstats_arg_body<int>(dist, s, has_cutoff, cutoff, r);
}

extern "C" __global__ void sb_rowstats_arg_i64(const sb_i64* dist, SbSpace s, int has_cutoff,
                                               sb_i64 cutoff, SbRows r) {
    sb_rowstats_arg_body<sb_i64>(dist, s, has_cutoff, cutoff, r);
}

// Set `n` int64 values to `value` (RowStats' best starts at exactly INT64_MAX, Score::MASK,
// which one cuMemsetD32 value cannot write).
extern "C" __global__ void sb_fill_i64(sb_i64* out, sb_u64 n, sb_i64 value) {
    const sb_u64 stride = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride) {
        out[t] = value;
    }
}
