// Levenshtein kernel family for the seqblaze fatbin.
// Spec: docs/wiki/gpu/levenshtein-cuda.md §4-§5 (device algorithm) and
// docs/wiki/edit/levenshtein.md §2, §6 (recurrence, word step). Written from
// those pages only.
//
// Integers only: wrapping u64 bit operations and integer counters, no floating
// point and no atomics. Every value is an exact integer of (a, b, mode): dist,
// b_end = min(ends) and n_ends = |ends|. The thread/warp mapping, grid and
// chunk sizes only change where and when the same word steps run; the word step
// for word w at column j reads its own Pv/Mv after column j - 1, Eq for b_j and
// the hout of word w - 1 at column j, exactly as the CPU kernel.
#pragma once
#include "hamming.cuh"

// Modes; must match `MODE_*` in src/lev.rs.
#define SB_LEV_GLOBAL 0ull
#define SB_LEV_PREFIX 1ull
#define SB_LEV_INFIX 2ull

// Tiers of a pair (§5.3-§5.5).
#define SB_TIER_THREAD 0
#define SB_TIER_WARP 1
#define SB_TIER_HOST 2

// Mirrors `SbLev` in src/lev.rs; every field is 8 bytes wide.
struct SbLev {
    sb_u64 mode;
    sb_u64 thread_words;  // largest W in the thread tier (0: warp tier for every pair)
    sb_u64 device_words;  // largest W on the device
    sb_u64 pair_steps;    // largest serial depth on the device
    sb_u64 sigma;         // raw tables: codes 0..sigma-1; the absent code is sigma
    sb_u64 map;           // raw tables: const unsigned short[256], byte -> code
    sb_u64 slots;         // raw tables: u64 Peq slots, all zero between pairs
    sb_u64 slot_words;    // raw tables: u64 words per thread (thread tier) or warp (warp tier)
    sb_u64 b_end;         // sb_i64* per chunk item, or 0
    sb_u64 n_ends;        // int* per chunk item, or 0
};

// The tier of a pair with |a| = m and |b| = n. One predicate decides the tier
// kernels' skips and the routed list. W = ceil(m / 64); depth is n * W in the
// thread tier and (n + 31) * q, q = ceil(W / 32), in the warp tier; a pair is
// routed to the host when W > device_words or depth > pair_steps (strictly).
__device__ __forceinline__ int sb_lev_tier(const SbLev& p, sb_u64 m, sb_u64 n) {
    const sb_u64 w = (m + 63) / 64;
    if (w > p.device_words) {
        return SB_TIER_HOST;
    }
    if (p.thread_words > 0 && w <= p.thread_words) {
        return n * w > p.pair_steps ? SB_TIER_HOST : SB_TIER_THREAD;
    }
    const sb_u64 q = (w + 31) / 32;
    return (n + 31) * q > p.pair_steps ? SB_TIER_HOST : SB_TIER_WARP;
}

__device__ __forceinline__ void sb_lev_lengths(const SbTables& tb, sb_u64 i, sb_u64 j, sb_u64& m,
                                               sb_u64& n) {
    m = (sb_u64)sb_a_row(tb, i).length;
    n = (sb_u64)sb_b_row(tb, j).length;
}

__device__ __forceinline__ void sb_lev_store(const SbLev& p, int* dist, sb_u64 t, int d,
                                             sb_i64 b_end, sb_u64 n_ends) {
    dist[t] = d;
    if (p.b_end != 0) {
        ((sb_i64*)p.b_end)[t] = b_end;
    }
    if (p.n_ends != 0) {
        ((int*)p.n_ends)[t] = (int)n_ends;
    }
}

// m = 0 (levenshtein §2 table): global gives n; prefix 0, b_end 0, n_ends 1;
// infix 0, b_end 0, n_ends n + 1.
__device__ __forceinline__ void sb_lev_empty_a(const SbLev& p, int* dist, sb_u64 t, sb_u64 n) {
    if (p.mode == SB_LEV_GLOBAL) {
        sb_lev_store(p, dist, t, (int)n, (sb_i64)n, 1);
    } else if (p.mode == SB_LEV_PREFIX) {
        sb_lev_store(p, dist, t, 0, 0, 1);
    } else {
        sb_lev_store(p, dist, t, 0, 0, n + 1);
    }
}

// Four-code map of a clean table a: A, C, G, T -> 0..3, every other byte -> 4
// (absent: Eq = 0, exact because no a_i has that value).
__device__ __forceinline__ int sb_acgt(unsigned x) {
    return x == 'A' ? 0 : x == 'C' ? 1 : x == 'G' ? 2 : x == 'T' ? 3 : 4;
}

// One Myers/Hyyro word step (levenshtein §6). Returns hout and the horizontal
// +1/-1 masks before the shift; bit k of them is the delta at the word's row k + 1.
__device__ __forceinline__ int sb_lev_step(sb_u64& pv, sb_u64& mv, sb_u64 eq, int hin, sb_u64& ph_out,
                                           sb_u64& mh_out) {
    const sb_u64 p = pv;
    const sb_u64 m = mv;
    const sb_u64 xv = eq | m;
    if (hin < 0) {
        eq |= 1ull;
    }
    const sb_u64 xh = (((eq & p) + p) ^ p) | eq;
    sb_u64 ph = m | ~(xh | p);
    sb_u64 mh = p & xh;
    ph_out = ph;
    mh_out = mh;
    const int hout = (int)(ph >> 63) - (int)(mh >> 63);
    ph = (ph << 1) | (hin > 0 ? 1ull : 0ull);
    mh = (mh << 1) | (hin < 0 ? 1ull : 0ull);
    pv = mh | ~(xv | ph);
    mv = ph & xv;
    return hout;
}

__device__ __forceinline__ void sb_lev_load_map(const SbLev& p, unsigned short* map) {
    for (unsigned k = threadIdx.x; k < 256u; k += blockDim.x) {
        map[k] = ((const unsigned short*)p.map)[k];
    }
    __syncthreads();
}

// Thread tier (§5.3): one thread per pair, W <= thread_words <= 4, grid-stride
// over the chunk's items; Pv, Mv, score and the counters in registers, words in
// increasing order within a column. Clean tables keep Peq in registers; raw
// tables use the thread's slot of sigma * W words, Peq[c] word w at c * W + w.
template <bool RAW>
__device__ __forceinline__ void sb_lev_thread_body(const SbTables& tb, const SbSpace& s,
                                                   const SbLev& p, int* dist) {
    __shared__ unsigned short map[256];
    if (RAW) {
        sb_lev_load_map(p, map);
    }
    const sb_u64 stride = (sb_u64)gridDim.x * blockDim.x;
    const sb_u64 tid = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x;
    sb_u64* slot = RAW ? (sb_u64*)p.slots + tid * p.slot_words : 0;
    const int top = p.mode == SB_LEV_INFIX ? 0 : 1;
    const bool ends = p.mode != SB_LEV_GLOBAL;
    for (sb_u64 t = tid; t < s.n; t += stride) {
        sb_u64 i, j, m, n;
        sb_decode(s, t, i, j);
        sb_lev_lengths(tb, i, j, m, n);
        if (sb_lev_tier(p, m, n) != SB_TIER_THREAD) {
            continue;
        }
        if (m == 0) {
            sb_lev_empty_a(p, dist, t, n);
            continue;
        }
        const SbSequence pa = sb_a_row(tb, i);
        const SbSequence pb = sb_b_row(tb, j);
        const int words = (int)((m + 63) / 64);
        sb_u64 pv[4], mv[4], e0[4], e1[4], e2[4], e3[4];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            pv[w] = ~0ull;
            mv[w] = 0;
            e0[w] = 0;
            e1[w] = 0;
            e2[w] = 0;
            e3[w] = 0;
        }
        // Peq: word w, bit k is [a at row 64w + k + 1 has code c].
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            if (w < words) {
                const sb_u64 lo = 64ull * w;
                const sb_u64 hi = lo + 64 < m ? lo + 64 : m;
                for (sb_u64 r = lo; r < hi; ++r) {
                    const sb_u64 bit = 1ull << (r - lo);
                    if (RAW) {
                        slot[(sb_u64)map[pa[r]] * words + w] |= bit;
                    } else {
                        const int c = sb_acgt(pa[r]);
                        e0[w] |= c == 0 ? bit : 0ull;
                        e1[w] |= c == 1 ? bit : 0ull;
                        e2[w] |= c == 2 ? bit : 0ull;
                        e3[w] |= c == 3 ? bit : 0ull;
                    }
                }
            }
        }
        int score = (int)m;
        int best = (int)m;
        sb_i64 b_end = 0;
        sb_u64 n_ends = 1;
        const unsigned bit = (unsigned)((m - 1) & 63);
        for (sb_u64 col = 0; col < n; ++col) {
            const unsigned x = pb[col];
            unsigned code = 0;
            if (RAW) {
                code = map[x];
            } else {
                code = (unsigned)sb_acgt(x);
            }
            const sb_u64* row = RAW ? slot + (sb_u64)code * words : 0;
            const bool present = RAW ? code < p.sigma : code < 4u;
            int hin = top;
            sb_u64 ph = 0, mh = 0;
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                if (w < words) {
                    sb_u64 eq = 0;
                    if (RAW) {
                        eq = present ? row[w] : 0ull;
                    } else {
                        eq = code == 0 ? e0[w] : eq;
                        eq = code == 1 ? e1[w] : eq;
                        eq = code == 2 ? e2[w] : eq;
                        eq = code == 3 ? e3[w] : eq;
                    }
                    hin = sb_lev_step(pv[w], mv[w], eq, hin, ph, mh);
                }
            }
            score += (int)((ph >> bit) & 1ull) - (int)((mh >> bit) & 1ull);
            if (ends) {
                if (score < best) {
                    best = score;
                    b_end = (sb_i64)col + 1;
                    n_ends = 1;
                } else if (score == best) {
                    ++n_ends;
                }
            }
        }
        if (RAW) {
            // Clear exactly the words this pair set: slots are zero between pairs.
            for (sb_u64 r = 0; r < m; ++r) {
                slot[(sb_u64)map[pa[r]] * words + r / 64] = 0;
            }
        }
        sb_lev_store(p, dist, t, ends ? best : score, b_end, n_ends);
    }
}

extern "C" __global__ void sb_lev_thread(SbTables tb, SbSpace s, SbLev p, int* dist) {
    sb_lev_thread_body<false>(tb, s, p, dist);
}

extern "C" __global__ void sb_lev_thread_raw(SbTables tb, SbSpace s, SbLev p, int* dist) {
    sb_lev_thread_body<true>(tb, s, p, dist);
}

// Warp tier (§5.4): one warp per pair, thread_words < W <= device_words. With
// q = ceil(W / 32), lane l owns words [l q, min(l q + q, W)); L is the last lane
// that owns a word. At step t = 1..n + L lane l processes column j = t - l when
// 1 <= j <= n: its first word's hin is the boundary (lane 0) or the carry lane
// l - 1 produced for column j at step t - 1 (__shfl_up_sync); it steps its q
// words in order and passes its last hout on. Only lane L updates score and
// ends, in column order. This kernel serves pairs with ceil(W / 32) in
// (Q / 2, Q] (Q = 1: W <= 32, including m = 0 when thread_words is 0). Control
// flow outside the column body is warp-uniform; each lane touches only the Peq
// words it owns.
template <int Q, bool RAW>
__device__ __forceinline__ void sb_lev_warp_body(const SbTables& tb, const SbSpace& s,
                                                 const SbLev& p, int* dist) {
    __shared__ unsigned short map[256];
    if (RAW) {
        sb_lev_load_map(p, map);
    }
    const int lane = (int)(threadIdx.x & 31u);
    const sb_u64 per_block = blockDim.x / 32u;
    const sb_u64 warp = (sb_u64)blockIdx.x * per_block + (threadIdx.x >> 5);
    const sb_u64 n_warps = (sb_u64)gridDim.x * per_block;
    sb_u64* slot = RAW ? (sb_u64*)p.slots + warp * p.slot_words : 0;
    const int top = p.mode == SB_LEV_INFIX ? 0 : 1;
    const bool ends = p.mode != SB_LEV_GLOBAL;
    for (sb_u64 t = warp; t < s.n; t += n_warps) {
        sb_u64 i, j, m, n;
        sb_decode(s, t, i, j);
        sb_lev_lengths(tb, i, j, m, n);
        if (sb_lev_tier(p, m, n) != SB_TIER_WARP) {
            continue;
        }
        const sb_u64 w_pair = (m + 63) / 64;
        if (w_pair > 32ull * Q || (Q > 1 && w_pair <= 16ull * Q)) {
            continue;  // another warp kernel's pair
        }
        if (m == 0) {
            if (lane == 0) {
                sb_lev_empty_a(p, dist, t, n);
            }
            continue;
        }
        const SbSequence pa = sb_a_row(tb, i);
        const SbSequence pb = sb_b_row(tb, j);
        const int words = (int)w_pair;
        const int q = (words + 31) / 32;
        const int last = (words + q - 1) / q - 1;
        const int w0 = lane * q;
        const int own = lane <= last ? min(q, words - w0) : 0;
        sb_u64 pv[Q], mv[Q], e0[Q], e1[Q], e2[Q], e3[Q];
#pragma unroll
        for (int x = 0; x < Q; ++x) {
            pv[x] = ~0ull;
            mv[x] = 0;
            e0[x] = 0;
            e1[x] = 0;
            e2[x] = 0;
            e3[x] = 0;
        }
#pragma unroll
        for (int x = 0; x < Q; ++x) {
            if (x < own) {
                const sb_u64 w = (sb_u64)(w0 + x);
                const sb_u64 lo = 64ull * w;
                const sb_u64 hi = lo + 64 < m ? lo + 64 : m;
                for (sb_u64 r = lo; r < hi; ++r) {
                    const sb_u64 bit = 1ull << (r - lo);
                    if (RAW) {
                        slot[(sb_u64)map[pa[r]] * words + w] |= bit;
                    } else {
                        const int c = sb_acgt(pa[r]);
                        e0[x] |= c == 0 ? bit : 0ull;
                        e1[x] |= c == 1 ? bit : 0ull;
                        e2[x] |= c == 2 ? bit : 0ull;
                        e3[x] |= c == 3 ? bit : 0ull;
                    }
                }
            }
        }
        int score = (int)m;
        int best = (int)m;
        sb_i64 b_end = 0;
        sb_u64 n_ends = 1;
        const unsigned bit = (unsigned)((m - 1) & 63);
        int carry = 0;
        const sb_u64 steps = n + (sb_u64)last;
        for (sb_u64 step = 1; step <= steps; ++step) {
            const int incoming = __shfl_up_sync(0xffffffffu, carry, 1);
            const sb_i64 col = (sb_i64)step - lane;  // 1-based column of this lane
            if (own > 0 && col >= 1 && col <= (sb_i64)n) {
                const unsigned xb = pb[col - 1];
                unsigned code = 0;
                if (RAW) {
                    code = map[xb];
                } else {
                    code = (unsigned)sb_acgt(xb);
                }
                const sb_u64* row = RAW ? slot + (sb_u64)code * words + w0 : 0;
                const bool present = RAW ? code < p.sigma : code < 4u;
                int hin = lane == 0 ? top : incoming;
                sb_u64 ph = 0, mh = 0;
#pragma unroll
                for (int x = 0; x < Q; ++x) {
                    if (x < own) {
                        sb_u64 eq = 0;
                        if (RAW) {
                            eq = present ? row[x] : 0ull;
                        } else {
                            eq = code == 0 ? e0[x] : eq;
                            eq = code == 1 ? e1[x] : eq;
                            eq = code == 2 ? e2[x] : eq;
                            eq = code == 3 ? e3[x] : eq;
                        }
                        hin = sb_lev_step(pv[x], mv[x], eq, hin, ph, mh);
                    }
                }
                carry = hin;
                if (lane == last) {
                    score += (int)((ph >> bit) & 1ull) - (int)((mh >> bit) & 1ull);
                    if (ends) {
                        if (score < best) {
                            best = score;
                            b_end = col;
                            n_ends = 1;
                        } else if (score == best) {
                            ++n_ends;
                        }
                    }
                }
            }
        }
        if (lane == last) {
            sb_lev_store(p, dist, t, ends ? best : score, b_end, n_ends);
        }
        if (RAW) {
            for (int x = 0; x < own; ++x) {
                const sb_u64 w = (sb_u64)(w0 + x);
                const sb_u64 lo = 64ull * w;
                const sb_u64 hi = lo + 64 < m ? lo + 64 : m;
                for (sb_u64 r = lo; r < hi; ++r) {
                    slot[(sb_u64)map[pa[r]] * words + w] = 0;
                }
            }
        }
        // The next pair's words may belong to other lanes.
        __syncwarp();
    }
}

#define SB_LEV_WARP(Q)                                                                        \
    extern "C" __global__ void sb_lev_warp##Q(SbTables tb, SbSpace s, SbLev p, int* dist) {     \
        sb_lev_warp_body<Q, false>(tb, s, p, dist);                                           \
    }                                                                                         \
    extern "C" __global__ void sb_lev_warp##Q##_raw(SbTables tb, SbSpace s, SbLev p, int* dist) { \
        sb_lev_warp_body<Q, true>(tb, s, p, dist);                                            \
    }
SB_LEV_WARP(1)
SB_LEV_WARP(2)
SB_LEV_WARP(4)
SB_LEV_WARP(8)

__device__ __forceinline__ bool sb_lev_routed(const SbTables& tb, const SbSpace& s, const SbLev& p,
                                              sb_u64 t) {
    if (t >= s.n) {
        return false;
    }
    sb_u64 i, j, m, n;
    sb_decode(s, t, i, j);
    sb_lev_lengths(tb, i, j, m, n);
    return sb_lev_tier(p, m, n) == SB_TIER_HOST;
}

// Routing step 1 (§5.5, the form of G1's Hits): routed count of each block's
// contiguous tile of blockDim.x items.
extern "C" __global__ void sb_lev_route_count(SbTables tb, SbSpace s, SbLev p, sb_u64* block_counts) {
    __shared__ unsigned warp_counts[32];
    const sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x;
    const bool routed = sb_lev_routed(tb, s, p, t);
    const unsigned ballot = __ballot_sync(0xffffffffu, routed);
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

// Routing step 3: chunk positions of the routed items, in item order, at the
// block's exclusive offset plus the in-block rank.
extern "C" __global__ void sb_lev_route_write(SbTables tb, SbSpace s, SbLev p,
                                              const sb_u64* block_offsets, sb_u64* out) {
    __shared__ unsigned warp_base[32];
    const sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x;
    const bool routed = sb_lev_routed(tb, s, p, t);
    const unsigned ballot = __ballot_sync(0xffffffffu, routed);
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
    if (routed) {
        const unsigned below = ballot & ((1u << lane) - 1u);
        out[block_offsets[blockIdx.x] + warp_base[warp] + (unsigned)__popc(below)] = t;
    }
}

// Host results of routed items, written into the chunk's buffers before any
// sink kernel runs.
extern "C" __global__ void sb_lev_scatter(const sb_u64* pos, const int* d, const sb_i64* e,
                                          const int* c, sb_u64 count, int* dist, sb_i64* b_end,
                                          int* n_ends) {
    const sb_u64 stride = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 k = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; k < count; k += stride) {
        const sb_u64 t = pos[k];
        dist[t] = d[k];
        if (b_end != 0) {
            b_end[t] = e[k];
        }
        if (n_ends != 0) {
            n_ends[t] = c[k];
        }
    }
}
