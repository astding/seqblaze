// The C1 bound of K1 native `duplex` (model "mfe") in the resident K1 stage
// (docs/wiki/thermo/duplex-mfe-bound.md §6.2, §6.5, §6.6, §8): per work item the pass
// threshold θ, the outcome (open, certified, pruned, or skipped by the TopK replay), the
// key λ of a resolved item, the DP skip flag and integer outcome counts; the probe of the
// weak-bound decision; and the multiplicities of canonical explicit pairs. Written from
// that page only.
//
// Integers only. The bound itself is `sb_bnd_fill` of duplex_bound.cuh, so every (LB, χ)
// equals the CPU's; counts are integer atomics, so their sums do not depend on order.
#pragma once
#include "duplex.cuh"
#include "duplex_bound.cuh"

#define SB_RB_OPEN 0u
#define SB_RB_CERTIFIED 1u
#define SB_RB_PRUNED 2u
#define SB_RB_SKIPPED 3u
// Staged letter codes of a and b ahead of the bound's bands in each thread's scratch:
// 400 bytes = 50 words.
#define SB_RB_CODE_WORDS 50ull

// Mirrors `SbResidentBound` in src/resident_bound.rs; every field is 8 bytes wide.
struct SbResidentBound {
    sb_u64 mode;        // 0: no threshold (certify only), 1: scalar θ, 2: TopK replay limits
    sb_i64 theta;       // mode 1
    sb_u64 thresholds;  // mode 2: const sb_i64* per row
    sb_u64 active;      // mode 2: const sb_u8* per row
    sb_u64 mirror;      // mode 2: a mirrored within item's limit is the max over its rows
    sb_u64 use;         // the pass's weak-bound decision: 1 evaluate the bound, 0 do not
    sb_u64 fields;      // dh/ds/tm requested: certified items keep the DP for their fields
    sb_u64 keys;        // int*: the chunk's keys
    sb_u64 dh;          // u64*: the chunk's field bits, or 0 outside the detail pass
    sb_u64 ds;
    sb_u64 tm;
    sb_u64 skip;        // u8*: the chunk's DP skip flags (SbDup::skip)
    sb_u64 mult;        // const sb_u64*: occurrences of work ID k0 + t, or 0 (once each)
    sb_u64 counts;      // u64[4]: certified, pruned, full, skipped
    sb_u64 scratch;     // sb_i64*: `stride` words per thread
    sb_u64 stride;
    sb_u64 count;       // 1: add the outcome counts; 0: a repeated sweep adds nothing
    sb_u64 words;       // int*: the chunk's outcome words (λ, or INT32_MAX for open), or 0
    sb_u64 words_mode;  // 0 none, 1 record (sweep 1), 2 restore (sweep 2, no evaluation)
};

// The outcome word of an open item (§8 *Per-item reuse*, revision 4): INT32_MAX, outside
// the clamped key range [INT32_MIN, SB_BND_SENTINEL = INT32_MAX - 1], so no resolved
// item's word equals it.
#define SB_RB_WORD_OPEN 2147483647

// θ of item (i, j) (§6.2): false when the TopK replay skips it (no row limit).
__device__ __forceinline__ bool sb_rb_theta(const SbResidentBound& r, sb_u64 i, sb_u64 j,
                                            bool& has, sb_i64& theta) {
    has = false;
    theta = 0;
    if (r.mode == 0) {
        return true;
    }
    if (r.mode == 1) {
        has = true;
        theta = r.theta;
        return true;
    }
    const sb_u8* active = (const sb_u8*)r.active;
    const sb_i64* limits = (const sb_i64*)r.thresholds;
    const bool ai = active[i] != 0;
    const bool aj = r.mirror != 0 && active[j] != 0;
    if (!ai && !aj) {
        return false;
    }
    theta = ai ? limits[i] : limits[j];
    if (ai && aj && limits[j] > theta) {
        theta = limits[j];
    }
    has = true;
    return true;
}

// The outcome code of item (i, j) against θ (§1) and its λ (the sentinel key for +∞),
// as `sb_dup_bound` computes it from (LB, χ).
__device__ __forceinline__ unsigned sb_rb_outcome(const SbDupBound& p, const SbTables& tb,
                                                  sb_u64 i, sb_u64 j, sb_u64 n, sb_u64 m,
                                                  bool has, sb_i64 theta, sb_i64* scratch,
                                                  sb_i64& lambda) {
    bool finite = false;
    sb_i64 lb = 0;
    bool chi = true;
    if (n >= 2 && m >= 2) {
        sb_u8* ca = (sb_u8*)scratch;
        sb_u8* cb = ca + SB_DUP_MAX_LENGTH;
        const SbSequence pa = sb_a_row(tb, i);
        const SbSequence pb = sb_b_row(tb, j);
        for (sb_u64 x = 0; x < n; ++x) ca[x] = sb_dup_code(pa[x]);
        for (sb_u64 x = 0; x < m; ++x) cb[x] = sb_dup_code(pb[x]);
        const sb_i64 best = sb_bnd_fill(p, ca, n, cb, m, scratch + SB_RB_CODE_WORDS);
        if (sb_bnd_fin(best)) {
            finite = true;
            lb = best >> 1;
            chi = (best & 1) == 0;
        }
    }
    return sb_bnd_code(p, finite, lb, chi, has, theta, lambda);
}

// One thread per item of the chunk, grid-striding, each with its own scratch slot.
// Every item gets its skip flag; a skipped (resolved) item also gets its key and NaN
// fields, which a certified item's DP would otherwise overwrite. A repeated sweep
// (RowStats sweep 2, §8 *Sweeps*) restores each outcome from its word without evaluating
// the bound: a resolved word λ is pruned iff a threshold exists and λ > θ, else
// certified, which is the outcome sweep 1 found (the same θ; λ is the key it wrote).
// The host launches it only for a pass that uses the bound or for the TopK replay
// (`use` may be 0 only in the replay); any other pass runs the DP without skip flags and
// counts on the host (§8 *Per chunk*).
extern "C" __global__ void sb_resident_dup_bound(SbTables tb, SbSpace s, SbDupBound p,
                                                 SbResidentBound r) {
    const sb_u64 threads = (sb_u64)gridDim.x * blockDim.x;
    const sb_u64 self = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x;
    sb_i64* scratch = (sb_i64*)r.scratch + self * r.stride;
    unsigned long long* counts = (unsigned long long*)r.counts;
    for (sb_u64 t = self; t < s.n; t += threads) {
        sb_u64 i, j, n, m;
        sb_decode(s, t, i, j);
        sb_dup_lengths(tb, i, j, n, m);
        const sb_u64 mult = r.mult != 0 ? ((const sb_u64*)r.mult)[s.k0 + t] : 1ull;
        bool has;
        sb_i64 theta;
        sb_i64 lambda = SB_BND_SENTINEL;
        unsigned code = SB_RB_SKIPPED;
        // In the replay a canonical explicit item stands for its first occurrence; the
        // other occurrences are skipped (host sinks, TopK eligibility).
        sb_u64 duplicates = 0;
        if (sb_rb_theta(r, i, j, has, theta)) {
            if (r.mode == 2) {
                duplicates = mult - 1;
            }
            if (r.words_mode == 2) {
                const int word = ((const int*)r.words)[t];
                if (word == SB_RB_WORD_OPEN) {
                    code = SB_RB_OPEN;
                } else {
                    lambda = (sb_i64)word;
                    code = (has && lambda > theta) ? SB_RB_PRUNED : SB_RB_CERTIFIED;
                }
            } else {
                code = r.use != 0
                           ? sb_rb_outcome(p, tb, i, j, n, m, has, theta, scratch, lambda)
                           : SB_RB_OPEN;
            }
        }
        const bool skipped = code == SB_RB_SKIPPED || code == SB_RB_PRUNED ||
                             (code == SB_RB_CERTIFIED && r.fields == 0);
        ((sb_u8*)r.skip)[t] = skipped ? 1 : 0;
        int word = SB_RB_WORD_OPEN;
        if (code != SB_RB_OPEN) {
            const sb_i64 low = -2147483648ll;
            const sb_i64 key = lambda < low ? low : (lambda > SB_BND_SENTINEL ? SB_BND_SENTINEL : lambda);
            ((int*)r.keys)[t] = (int)key;
            if (code != SB_RB_SKIPPED) {
                word = (int)key;
            }
        }
        if (r.words_mode == 1) {
            ((int*)r.words)[t] = word;
        }
        // The detail columns exist only in the detail pass (duplex-cuda.md §15.4 step 2);
        // the score passes pass no field addresses.
        if (skipped && r.dh != 0) {
            ((sb_u64*)r.dh)[t] = SB_F64_QNAN;
            ((sb_u64*)r.ds)[t] = SB_F64_QNAN;
            ((sb_u64*)r.tm)[t] = SB_F64_QNAN;
        }
        if (r.count == 0) {
            continue;
        }
        const sb_u64 once = r.mode == 2 ? 1ull : mult;
        if (code == SB_RB_CERTIFIED) {
            atomicAdd(counts + 0, (unsigned long long)once);
        } else if (code == SB_RB_PRUNED) {
            atomicAdd(counts + 1, (unsigned long long)once);
        } else if (code == SB_RB_OPEN) {
            atomicAdd(counts + 2, (unsigned long long)once);
        } else {
            duplicates = mult;
        }
        if (duplicates != 0) {
            atomicAdd(counts + 3, (unsigned long long)duplicates);
        }
    }
}

// The probe of the weak-bound decision (§6.6): per probe item (work ID ids[t] of the
// original-order space `s`), its strand lengths and its outcome code (skipped when the
// replay has no limit for it or `dup[t]` marks a later occurrence). out: 3 words per item.
extern "C" __global__ void sb_resident_dup_probe(SbTables tb, SbSpace s, SbDupBound p,
                                                 SbResidentBound r, const sb_u64* ids,
                                                 sb_u64 count, const unsigned* dup,
                                                 sb_u64* out) {
    const sb_u64 threads = (sb_u64)gridDim.x * blockDim.x;
    const sb_u64 self = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x;
    sb_i64* scratch = (sb_i64*)r.scratch + self * r.stride;
    for (sb_u64 t = self; t < count; t += threads) {
        sb_u64 i, j, n, m;
        sb_decode(s, ids[t], i, j);
        sb_dup_lengths(tb, i, j, n, m);
        bool has;
        sb_i64 theta;
        sb_i64 lambda = SB_BND_SENTINEL;
        unsigned code = SB_RB_SKIPPED;
        if ((dup == 0 || dup[t] == 0) && sb_rb_theta(r, i, j, has, theta)) {
            code = sb_rb_outcome(p, tb, i, j, n, m, has, theta, scratch, lambda);
        }
        out[3 * t] = n;
        out[3 * t + 1] = m;
        out[3 * t + 2] = code;
    }
}

// §6.6.2 step 2: per probe item (work ID ids[t] of the original-order space `s`), its
// strand lengths and whether the pass skips it (the replay has no limit for it, or
// `dup[t]` marks a later occurrence). No bound is evaluated. out: 3 words per item.
extern "C" __global__ void sb_resident_probe_lengths(SbTables tb, SbSpace s, SbResidentBound r,
                                                     const sb_u64* ids, sb_u64 count,
                                                     const unsigned* dup, sb_u64* out) {
    const sb_u64 threads = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < count; t += threads) {
        sb_u64 i, j, n, m;
        sb_decode(s, ids[t], i, j);
        sb_dup_lengths(tb, i, j, n, m);
        bool has;
        sb_i64 theta;
        const bool live = (dup == 0 || dup[t] == 0) && sb_rb_theta(r, i, j, has, theta);
        out[3 * t] = n;
        out[3 * t + 1] = m;
        out[3 * t + 2] = live ? 0ull : 1ull;
    }
}

// dup[t] = 1 iff some work ID q < ids[t] of the original-order explicit space `s`
// (n items) has the same pair: ids[t] is not a first occurrence (TopK replay eligibility).
// count <= 64.
extern "C" __global__ void sb_resident_probe_dups(SbSpace s, sb_u64 n, const sb_u64* ids,
                                                  sb_u64 count, unsigned* dup) {
    __shared__ sb_u64 pi[64], pj[64], pk[64];
    for (sb_u64 x = threadIdx.x; x < count && x < 64; x += blockDim.x) {
        pk[x] = ids[x];
        sb_decode(s, ids[x], pi[x], pj[x]);
    }
    __syncthreads();
    const sb_u64 stride = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 q = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; q < n; q += stride) {
        sb_u64 i, j;
        sb_decode(s, q, i, j);
        for (sb_u64 x = 0; x < count && x < 64; ++x) {
            if (q < pk[x] && i == pi[x] && j == pj[x]) {
                atomicOr(dup + x, 1u);
            }
        }
    }
}

// mult[u] = number of occurrences in the original-order explicit space `s` (n items) of
// canonical pair u of (ci, cj) (nu pairs, ascending lexicographic). mult starts at zero.
extern "C" __global__ void sb_resident_pair_mult(SbSpace s, sb_u64 n, const sb_i64* ci,
                                                 const sb_i64* cj, sb_u64 nu, sb_u64* mult) {
    const sb_u64 stride = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 q = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; q < n; q += stride) {
        sb_u64 i, j;
        sb_decode(s, q, i, j);
        const sb_i64 x = (sb_i64)i, y = (sb_i64)j;
        sb_u64 lo = 0, hi = nu;
        while (lo < hi) {
            const sb_u64 mid = lo + (hi - lo) / 2;
            if (ci[mid] < x || (ci[mid] == x && cj[mid] < y)) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if (lo < nu && ci[lo] == x && cj[lo] == y) {
            atomicAdd((unsigned long long*)(mult + lo), 1ull);
        }
    }
}

// x[0 .. n) = 0.
extern "C" __global__ void sb_resident_bound_zero(sb_u64* x, sb_u64 n) {
    const sb_u64 stride = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 q = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; q < n; q += stride) {
        x[q] = 0;
    }
}
