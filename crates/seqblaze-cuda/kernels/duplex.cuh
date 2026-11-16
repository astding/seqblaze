// K1 native `duplex` (model "mfe") on the device: the integer DP of one pair per
// sub-warp group, its tier predicate and routing, and the detail pass with the
// shared soft-binary64 finalize. Spec: docs/wiki/gpu/duplex-cuda.md §4-§6 and
// docs/wiki/thermo/duplex-mfe.md §5. Written from those pages only.
//
// Integers only. A key is D (int32 with +inf = 2^29, or int64 with +inf = 2^60) in
// the score pass and (D, H, G, Q) as four int64 in the detail pass; every sum has at
// most four terms and is normalised where §5.2 puts it, so it never overflows and
// +inf never wins. Every cell value is an exact integer of (a, b, pack), and
// lexicographic min is associative and commutative, so the column-to-lane
// partition, the reduction order, the group size, the tier and the grid only
// change which lane evaluates which candidate.
#pragma once
#include "finalize.cuh"
#include "hamming.cuh"

// Tiers of a pair (§5.6, §5.4); must match `DupTier` in src/plan.rs.
#define SB_DUP_CLOSED 0
#define SB_DUP_SHARED 1
#define SB_DUP_GLOBAL 2
#define SB_DUP_HOST 3

#define SB_DUP_SENTINEL_KEY 2147483646  // INT32_MAX - 1: K* = +inf
#define SB_DUP_MAX_LENGTH 200
// Staged bytes per group: the codes of a at 0 and of b at 200 (q = 400); with STATES also
// the run lengths ρ_a at 400 and ρ_b at 600 (q = 800). The call's q is `SbDup::codes`
// (nndb2024-dna.md §5.4.4).
#define SB_DUP_RHO 400
#define SB_DUP_NONE 4                   // the empty neighbour in END_L and END_R
#define SB_DUP_C2 0x4000000000000000ull // 2.0

// Mirrors `SbDup` in src/dup.rs; every field is 8 bytes wide. Table offsets index
// the concatenated cell keys (INIT at 0, INIT + SYM at 1).
struct SbDup {
    sb_u64 score;  // const int* (narrow) or const sb_i64* (wide): D of every cell
    sb_u64 full;   // const sb_i64*: (D, H, G, Q) of every cell, or 0 (score passes)
    sb_u64 lam;
    sb_u64 end_l;
    sb_u64 end_r;
    sb_u64 stack;
    sb_u64 bulge;
    sb_u64 i11;
    sb_u64 i12;
    sb_u64 i21;
    sb_u64 i22;
    sb_u64 gen;
    sb_u64 loop_out;
    sb_u64 loop_in;
    sb_u64 s;       // bits of the binary64 s
    sb_u64 kappa4;  // bits of κ₄
    sb_u64 kappa1;  // bits of κ₁
    sb_u64 group_lanes;  // 0: g by m
    sb_u64 band_bytes;
    sb_u64 pair_cells;   // largest depth of a device pair, candidate evaluations
    sb_u64 s_max;        // opt-in shared memory per block
    sb_u64 s_block;      // static shared memory per block
    sb_u64 tier;         // per launch: SB_DUP_SHARED or SB_DUP_GLOBAL
    sb_u64 stride;       // per launch: dynamic shared bytes per group
    sb_u64 slots;        // per launch (global tier): u8* of the band slots
    sb_u64 slot;         // per launch (global tier): bytes per slot
    sb_u64 dh;           // detail pass: u64* dh_kcal bits per item, or 0
    sb_u64 ds;           // u64* ds_cal_per_k bits
    sb_u64 tm;           // u64* tm_c bits
    sb_u64 poison;       // testing: byte written over each band before its pair; 256: none
    // Pack format 2 (nndb2024-dna.md §5.4.4): N_P, N_C, the u8 CLS map over (Λ + 1)², the
    // start of STATES in the key arrays, whether the pack has STATES, and q.
    sb_u64 n_p;
    sb_u64 n_c;
    sb_u64 cls;
    sb_u64 states;
    sb_u64 has_states;
    sb_u64 codes;
    // C1 bounded resident stage (duplex-mfe-bound.md §8): const u8* per item of the launch's
    // space, nonzero where the bound resolved the item and wrote its key and fields, so the
    // closed-form, DP and routing kernels skip it; 0: no item is skipped.
    sb_u64 skip;
#ifdef SB_TESTING
    sb_u64 resident_stats; // u64[2][3]: fast/continuation pairs, fills, bytes; 0: unobserved
#endif
};

#ifdef SB_TESTING
// Only resident launches supply a counter buffer. Count in the executing group,
// never from a host routing estimate or the allocation-time poison accounting.
__device__ __forceinline__ void sb_resident_poison_count(
    sb_u64 address, int route, int metric, sb_u64 value) {
    if (address != 0) {
        atomicAdd((unsigned long long*)address + route * 3 + metric,
                  (unsigned long long)value);
    }
}
#endif

// g of a pair whose b has length m (§5.3).
__device__ __forceinline__ sb_u64 sb_dup_group(const SbDup& p, sb_u64 m) {
    if (p.group_lanes != 0) {
        return p.group_lanes;
    }
    return m <= 16 ? 4ull : (m <= 32 ? 8ull : (m <= 64 ? 16ull : 32ull));
}

// c(n, m, Λ) of §5.6 for n, m >= 2: the terms are m - 1 until Λ - u falls below
// m - 2, then Λ + 1 - u.
__device__ __forceinline__ sb_u64 sb_dup_c(sb_u64 n, sb_u64 m, sb_u64 lam) {
    const sb_u64 top = lam < n - 2 ? lam : n - 2;
    const sb_u64 m2 = m - 2;
    const sb_u64 u0 = lam > m2 ? lam - m2 : 0ull;
    const sb_u64 flat = top + 1 < u0 ? top + 1 : u0;
    sb_u64 total = flat * (m2 + 1);
    if (top >= u0) {
        const sb_u64 count = top - u0 + 1;
        total += count * (lam + 1) - (top + u0) * count / 2;
    }
    return total;
}

// The tier of a pair in a pass whose band keys have `w` bytes; the host twin is
// `dup_tier` in src/plan.rs.
__device__ __forceinline__ int sb_dup_tier(const SbDup& p, sb_u64 w, sb_u64 n, sb_u64 m) {
    if (n < 2 || m < 2) {
        return SB_DUP_CLOSED;
    }
    const sb_u64 g = sb_dup_group(p, m);
    const sb_u64 depth = n * ((m + g - 1) / g) * sb_dup_c(n, m, p.lam);
    if (depth > p.pair_cells) {
        return SB_DUP_HOST;
    }
    const sb_u64 rows = n < p.lam + 2 ? n : p.lam + 2;
    const sb_u64 band = rows * m * w;
    const bool fits = (32ull / g) * (band + p.codes) + p.s_block <= p.s_max;
    return band <= p.band_bytes && fits ? SB_DUP_SHARED : SB_DUP_GLOBAL;
}

__device__ __forceinline__ void sb_dup_lengths(const SbTables& tb, sb_u64 i, sb_u64 j, sb_u64& n,
                                               sb_u64& m) {
    n = (sb_u64)sb_a_row(tb, i).length;
    m = (sb_u64)sb_b_row(tb, j).length;
}

// Whether the bounded stage resolved item t of the launch's space (SbDup::skip).
__device__ __forceinline__ bool sb_dup_skipped(const SbDup& p, sb_u64 t) {
    return p.skip != 0 && ((const sb_u8*)p.skip)[t] != 0;
}

// Total code map, as the CPU's: C 1, G 2, T 3, every other byte 0 (§4).
__device__ __forceinline__ sb_u8 sb_dup_code(sb_u8 b) {
    return b == 'C' ? 1 : (b == 'G' ? 2 : (b == 'T' ? 3 : 0));
}

// The pair type PT_dev(x, y; N_P) of nndb2024-dna.md §5.4.3 for codes x, y in 0..3: x for
// a Watson-Crick opposition (x + y = 3); with N_P = 6 also 4 for (G, T) and 5 for (T, G),
// the two oppositions with x + y = 5; otherwise -1 (no pair).
__device__ __forceinline__ int sb_dup_pt(int x, int y, sb_u64 np) {
    if (x + y == 3) {
        return x;
    }
    return np == 6 && x + y == 5 ? x + 2 : -1;
}

// The pair index of a predecessor (k, l): its type, or 0 when (k, l) is no pair. Such a
// candidate holds A(k, l) = +inf and normalises to +inf whatever cell it reads (§5.4.3).
__device__ __forceinline__ int sb_dup_pk(int x, int y, sb_u64 np) {
    const int t = sb_dup_pt(x, y, np);
    return t < 0 ? 0 : t;
}

// The run lengths ρ(x) of a staged strand of length n into `rho` (one byte each, 1..200;
// §5.4.4): the length of the maximal run of equal codes containing x. Lanes `me` of `g`.
__device__ __forceinline__ void sb_dup_runs(const sb_u8* s, int n, sb_u8* rho, int me, int g) {
    for (int x = me; x < n; x += g) {
        const sb_u8 c = s[x];
        int lo = x;
        int hi = x;
        while (lo > 0 && s[lo - 1] == c) {
            --lo;
        }
        while (hi + 1 < n && s[hi + 1] == c) {
            ++hi;
        }
        rho[x] = (sb_u8)(hi - lo + 1);
    }
}

// ------------------------------------------------------------------ key types
//
// Each key type gives +inf, the threshold normalisation N, the sum, the order, a
// width-g shuffle and the table load; the recurrence below is written once.

struct SbDupNarrow {
    typedef int T;
    static constexpr sb_u64 BYTES = 4;
    __device__ static __forceinline__ T inf() { return 1 << 29; }
    __device__ static __forceinline__ bool finite(T x) { return x < (1 << 28); }
    __device__ static __forceinline__ T norm(T x) { return finite(x) ? x : inf(); }
    __device__ static __forceinline__ T plus(T x, T y) { return x + y; }
    __device__ static __forceinline__ bool less(T x, T y) { return x < y; }
    __device__ static __forceinline__ T load(const SbDup& p, sb_u64 at) {
        return __ldg((const int*)p.score + at);
    }
    __device__ static __forceinline__ T shfl(unsigned mask, T x, int lane_mask, int width) {
        return __shfl_xor_sync(mask, x, lane_mask, width);
    }
    __device__ static __forceinline__ sb_i64 d(T x) { return x; }
};

struct SbDupWide {
    typedef sb_i64 T;
    static constexpr sb_u64 BYTES = 8;
    __device__ static __forceinline__ T inf() { return 1ll << 60; }
    __device__ static __forceinline__ bool finite(T x) { return x < (1ll << 59); }
    __device__ static __forceinline__ T norm(T x) { return finite(x) ? x : inf(); }
    __device__ static __forceinline__ T plus(T x, T y) { return x + y; }
    __device__ static __forceinline__ bool less(T x, T y) { return x < y; }
    __device__ static __forceinline__ T load(const SbDup& p, sb_u64 at) {
        return __ldg((const sb_i64*)p.score + at);
    }
    __device__ static __forceinline__ T shfl(unsigned mask, T x, int lane_mask, int width) {
        return __shfl_xor_sync(mask, x, lane_mask, width);
    }
    __device__ static __forceinline__ sb_i64 d(T x) { return x; }
};

struct SbDupKey {
    sb_i64 d, h, g, q;
};

struct SbDupFull {
    typedef SbDupKey T;
    static constexpr sb_u64 BYTES = 32;
    __device__ static __forceinline__ T inf() {
        T x;
        x.d = 1ll << 60;
        x.h = 0;
        x.g = 0;
        x.q = 0;
        return x;
    }
    __device__ static __forceinline__ bool finite(const T& x) { return x.d < (1ll << 59); }
    __device__ static __forceinline__ T norm(const T& x) { return finite(x) ? x : inf(); }
    __device__ static __forceinline__ T plus(const T& x, const T& y) {
        T r;
        r.d = x.d + y.d;
        r.h = x.h + y.h;
        r.g = x.g + y.g;
        r.q = x.q + y.q;
        return r;
    }
    // Lexicographic: D, then H, G, Q.
    __device__ static __forceinline__ bool less(const T& x, const T& y) {
        if (x.d != y.d) {
            return x.d < y.d;
        }
        if (x.h != y.h) {
            return x.h < y.h;
        }
        if (x.g != y.g) {
            return x.g < y.g;
        }
        return x.q < y.q;
    }
    __device__ static __forceinline__ T load(const SbDup& p, sb_u64 at) {
        const sb_i64* cell = (const sb_i64*)p.full + 4 * at;
        T x;
        x.d = __ldg(cell);
        x.h = __ldg(cell + 1);
        x.g = __ldg(cell + 2);
        x.q = __ldg(cell + 3);
        return x;
    }
    __device__ static __forceinline__ T shfl(unsigned mask, const T& x, int lane_mask, int width) {
        T r;
        r.d = __shfl_xor_sync(mask, x.d, lane_mask, width);
        r.h = __shfl_xor_sync(mask, x.h, lane_mask, width);
        r.g = __shfl_xor_sync(mask, x.g, lane_mask, width);
        r.q = __shfl_xor_sync(mask, x.q, lane_mask, width);
        return r;
    }
    __device__ static __forceinline__ sb_i64 d(const T& x) { return x.d; }
};

template <class K>
__device__ __forceinline__ typename K::T sb_dup_min(const typename K::T& x,
                                                    const typename K::T& y) {
    return K::less(y, x) ? y : x;
}

// ------------------------------------------------------------------ one cell
//
// A(i, j) of §5.1 for a pair cell: Start, the candidates over the band rows k < i,
// N(C) and the lexmin; the end term N(C) + END_R folds into `best`. `ca`, `cb` are
// the staged codes (ρ_a at ca + 400 and ρ_b at ca + 600 when the pack has STATES), `band`
// the ring of R rows of m keys. Tables are indexed by the pair types P_kl, P_ij and the
// loop class c = CLS[u][v] (nndb2024-dna.md §5.4.4).
template <class K>
__device__ __forceinline__ typename K::T sb_dup_cell(const SbDup& p, const sb_u8* ca,
                                                     const sb_u8* cb, int n, int m, int rows,
                                                     const typename K::T* band, int i, int j,
                                                     const typename K::T& start,
                                                     typename K::T& best) {
    typedef typename K::T T;
    const int lam = (int)p.lam;
    const sb_u64 np = p.n_p;
    const int pi = sb_dup_pt(ca[i], cb[j], np);  // a pair cell: pi >= 0
    T c = K::inf();
    if (i >= 1 && j + 1 < m) {
        // IN_c(i, j) of each class c < N_C, at ((c·4 + a_{i-1})·4 + b_{j+1})·N_P + P_ij.
        const sb_u64 in_at = p.loop_in + (sb_u64)((ca[i - 1] * 4 + cb[j + 1]) * (int)np + pi);
        const T in0 = K::load(p, in_at);
        T in1 = in0;
        T in2 = in0;
        if (p.n_c > 1) {
            in1 = K::load(p, in_at + 16 * np);
            in2 = K::load(p, in_at + 32 * np);
        }
        // STATES of a top bulge(1) (bulged base a_{i-1}) and a bottom one (b_{j+1}).
        T st_top = K::inf();
        T st_bottom = K::inf();
        if (p.has_states != 0) {
            const sb_u8* rho = ca + SB_DUP_RHO;
            st_top = K::load(p, p.states + (sb_u64)(rho[i - 1] - 1));
            st_bottom = K::load(p, p.states + (sb_u64)(rho[SB_DUP_MAX_LENGTH + j + 1] - 1));
        }
        const sb_u8* cls = (const sb_u8*)p.cls;
        const int umax = lam < i - 1 ? lam : i - 1;
        for (int u = 0; u <= umax; ++u) {
            const int k = i - 1 - u;
            const int ak = ca[k];
            const T* row = band + (k % rows) * m;
            const int vmax = lam - u < m - 2 - j ? lam - u : m - 2 - j;
            // v = 0: a stack or a top bulge.
            {
                const int pk = sb_dup_pk(ak, cb[j + 1], np);
                const sb_u64 at0 =
                    u == 0 ? p.stack + (sb_u64)(pk * (int)np + pi)
                           : p.bulge + (sb_u64)(((u - 1) * (int)np + pk) * (int)np + pi);
                T cand = K::plus(row[j + 1], K::load(p, at0));
                if (u == 1 && p.has_states != 0) {
                    cand = K::plus(cand, st_top);  // three terms (§5.2 F2)
                }
                c = sb_dup_min<K>(c, cand);
            }
            for (int v = 1; v <= vmax; ++v) {
                const int l = j + 1 + v;
                T cand;
                if (u == 0) {
                    const int pk = sb_dup_pk(ak, cb[l], np);
                    const sb_u64 at =
                        p.bulge + (sb_u64)(((lam + v - 1) * (int)np + pk) * (int)np + pi);
                    cand = K::plus(row[l], K::load(p, at));
                    if (v == 1 && p.has_states != 0) {
                        cand = K::plus(cand, st_bottom);
                    }
                } else if (u <= 2 && v <= 2) {
                    const int pk = sb_dup_pk(ak, cb[l], np);
                    const int p1 = ca[k + 1];
                    const int q1 = cb[l - 1];
                    sb_u64 at;
                    if (u == 1 && v == 1) {
                        at = p.i11 + (sb_u64)(((pk * 4 + p1) * 4 + q1) * (int)np + pi);
                    } else if (u == 1) {
                        const int q2 = cb[l - 2];
                        at = p.i12 + (sb_u64)((((pk * 4 + p1) * 4 + q1) * 4 + q2) * (int)np + pi);
                    } else if (v == 1) {
                        const int p2 = ca[k + 2];
                        at = p.i21 + (sb_u64)((((pk * 4 + p1) * 4 + p2) * 4 + q1) * (int)np + pi);
                    } else {
                        const int p2 = ca[k + 2];
                        const int q2 = cb[l - 2];
                        at = p.i22 +
                             (sb_u64)(((((pk * 4 + p1) * 4 + p2) * 4 + q1) * 4 + q2) * (int)np +
                                      pi);
                    }
                    cand = K::plus(row[l], K::load(p, at));
                } else {
                    // Generic: A(k, l) + LOOP_OUT + GEN + LOOP_IN of the shape's class c,
                    // four terms (§5.2 F2).
                    const int pk = sb_dup_pk(ak, cb[l], np);
                    const int shape = u * (lam + 1) + v;
                    const int cl = (int)__ldg(cls + shape);
                    const T out = K::load(
                        p, p.loop_out +
                               (sb_u64)(((cl * (int)np + pk) * 4 + ca[k + 1]) * 4 + cb[l - 1]));
                    const T gen = K::load(p, p.gen + (sb_u64)shape);
                    const T inner = cl == 0 ? in0 : (cl == 1 ? in1 : in2);
                    cand = K::plus(K::plus(K::plus(row[l], out), gen), inner);
                }
                c = sb_dup_min<K>(c, cand);
            }
        }
    }
    c = K::norm(c);
    const int alpha = i >= 1 ? ca[i - 1] : SB_DUP_NONE;
    const int beta = j + 1 < m ? cb[j + 1] : SB_DUP_NONE;
    const T first = K::norm(
        K::plus(start, K::load(p, p.end_l + (sb_u64)((alpha * (int)np + pi) * 5 + beta))));
    const int alpha2 = i + 1 < n ? ca[i + 1] : SB_DUP_NONE;
    const int beta2 = j >= 1 ? cb[j - 1] : SB_DUP_NONE;
    const T end = K::plus(c, K::load(p, p.end_r + (sb_u64)((pi * 5 + alpha2) * 5 + beta2)));
    best = sb_dup_min<K>(best, end);
    return K::less(c, first) ? c : first;
}

// The detail pass's fields (§5.5): canonical NaN for K* = +inf, else the shared
// finalize with P = Q / 2 (exact) and κ₁ for identical strands. Score passes store
// no field.
template <class T>
__device__ __forceinline__ void sb_dup_fields(const SbDup&, sb_u64, const T&, bool) {}

template <>
__device__ __forceinline__ void sb_dup_fields<SbDupKey>(const SbDup& p, sb_u64 t,
                                                        const SbDupKey& key, bool identical) {
    if (p.dh == 0) {
        return;
    }
    sb_u64 dh = SB_F64_QNAN, ds = SB_F64_QNAN, tm = SB_F64_QNAN;
    if (SbDupFull::finite(key)) {
        const sb_u64 half = sb_f64_div(sb_f64_from_i64(key.q), SB_DUP_C2);
        sb_nn_finalize(key.h, key.g, half, p.s, identical ? p.kappa1 : p.kappa4, dh, ds, tm);
    }
    ((sb_u64*)p.dh)[t] = sb_f64_canon(dh);
    ((sb_u64*)p.ds)[t] = sb_f64_canon(ds);
    ((sb_u64*)p.tm)[t] = sb_f64_canon(tm);
}

// ------------------------------------------------------------------ kernels

// Closed-form pairs (§5.6 item 1): key INT32_MAX - 1 and, in a detail pass, the
// canonical NaN fields. Written before the DP kernels; never routed.
extern "C" __global__ void sb_dup_closed(SbTables tb, SbSpace s, SbDup p, int* dist) {
    const sb_u64 stride = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < s.n; t += stride) {
        sb_u64 i, j, n, m;
        sb_decode(s, t, i, j);
        sb_dup_lengths(tb, i, j, n, m);
        if ((n >= 2 && m >= 2) || sb_dup_skipped(p, t)) {
            continue;
        }
        dist[t] = SB_DUP_SENTINEL_KEY;
        if (p.dh != 0) {
            ((sb_u64*)p.dh)[t] = SB_F64_QNAN;
            ((sb_u64*)p.ds)[t] = SB_F64_QNAN;
            ((sb_u64*)p.tm)[t] = SB_F64_QNAN;
        }
    }
}

// The DP of one variant (G, key type K, tier p.tier): one group of G lanes per pair,
// grid-striding over the chunk's items and skipping the pairs of other variants
// (§5.3). Dynamic shared memory: per group `p.stride` bytes, the q = `p.codes` staged
// bytes (the codes of a and b, and with STATES their run lengths) and, in the shared
// tier, the band ring.
template <int G, class K>
__device__ __forceinline__ void sb_dup_body(const SbTables& tb, const SbSpace& s, const SbDup& p,
                                            int* dist) {
    typedef typename K::T T;
    extern __shared__ __align__(16) sb_u8 sb_dup_smem[];
    const unsigned lane = threadIdx.x & 31u;
    const int me = (int)(threadIdx.x & (unsigned)(G - 1));
    const unsigned mask =
        G == 32 ? 0xffffffffu : (((1u << G) - 1u) << (lane & ~(unsigned)(G - 1)));
    const sb_u64 per_block = blockDim.x / G;
    const sb_u64 local = threadIdx.x / G;
    const sb_u64 group = (sb_u64)blockIdx.x * per_block + local;
    const sb_u64 groups = (sb_u64)gridDim.x * per_block;
    sb_u8* mine = sb_dup_smem + local * p.stride;
    sb_u8* ca = mine;
    sb_u8* cb = mine + SB_DUP_MAX_LENGTH;
    T* band = p.tier == SB_DUP_SHARED ? (T*)(mine + p.codes)
                                      : (T*)((sb_u8*)p.slots + group * p.slot);
    for (sb_u64 t = group; t < s.n; t += groups) {
        sb_u64 i, j, n64, m64;
        sb_decode(s, t, i, j);
        sb_dup_lengths(tb, i, j, n64, m64);
        if (sb_dup_group(p, m64) != (sb_u64)G || sb_dup_tier(p, K::BYTES, n64, m64) != (int)p.tier ||
            sb_dup_skipped(p, t)) {
            continue;  // another variant's, the closed-form kernel's, the host's or the bound's
        }
        const int n = (int)n64;
        const int m = (int)m64;
        const int rows = n < (int)p.lam + 2 ? n : (int)p.lam + 2;
        const SbSequence pa = sb_a_row(tb, i);
        const SbSequence pb = sb_b_row(tb, j);
        for (int x = me; x < n; x += G) {
            ca[x] = sb_dup_code(pa[x]);
        }
        for (int x = me; x < m; x += G) {
            cb[x] = sb_dup_code(pb[x]);
        }
#ifdef SB_TESTING
        if (me == 0) sb_resident_poison_count(p.resident_stats, 0, 0, 1);
        if (p.poison < 256) {
            // Testing hook (§8 item 8, QKG11): the pair's whole band is filled with
            // the poison byte; every byte is overwritten before any read. Release
            // builds compile no fill.
            sb_u8* bytes = (sb_u8*)band;
            const int total = rows * m * (int)K::BYTES;
            for (int x = me; x < total; x += G) {
                bytes[x] = (sb_u8)p.poison;
            }
        }
#endif
        __syncwarp(mask);
#ifdef SB_TESTING
        if (me == 0 && p.poison < 256) {
            sb_resident_poison_count(p.resident_stats, 0, 1, 1);
            sb_resident_poison_count(p.resident_stats, 0, 2,
                                     (sb_u64)rows * m * K::BYTES);
        }
#endif
        if (p.has_states != 0) {
            // The ρ prologue (§5.4.4): run lengths after the codes, read by the rows below.
            sb_dup_runs(ca, n, ca + SB_DUP_RHO, me, G);
            sb_dup_runs(cb, m, ca + SB_DUP_RHO + SB_DUP_MAX_LENGTH, me, G);
            __syncwarp(mask);
        }
        // identical: a = b; sym: identical and a palindrome (a_k + a_{n-1-k} = 3).
        bool same = n == m;
        bool pal = true;
        for (int x = me; x < n; x += G) {
            same = same && x < m && ca[x] == cb[x];
            pal = pal && ca[x] + ca[n - 1 - x] == 3;
        }
        const bool identical = __all_sync(mask, same) != 0;
        const bool sym = identical && __all_sync(mask, pal) != 0;
        const T start = K::load(p, sym ? 1ull : 0ull);
        T best = K::inf();
        for (int r = 0; r < n; ++r) {
            T* slot = band + (r % rows) * m;
            const int ar = ca[r];
            for (int c = me; c < m; c += G) {
                T value = K::inf();
                if (sb_dup_pt(ar, cb[c], p.n_p) >= 0) {
                    value = sb_dup_cell<K>(p, ca, cb, n, m, rows, band, r, c, start, best);
                }
                slot[c] = value;
            }
            __syncwarp(mask);
        }
#pragma unroll
        for (int offset = G / 2; offset > 0; offset >>= 1) {
            best = sb_dup_min<K>(best, K::shfl(mask, best, offset, G));
        }
        const T key = K::norm(best);
        if (me == 0) {
            const bool finite = K::finite(key);
            dist[t] = finite ? (int)K::d(key) : SB_DUP_SENTINEL_KEY;
            sb_dup_fields(p, t, key, identical);
        }
        // The next pair's staging and band writes follow every read of this one.
        __syncwarp(mask);
    }
}

#define SB_DUP_KERNEL(NAME, G, K)                                                            \
    extern "C" __global__ void sb_dup_##NAME##_g##G(SbTables tb, SbSpace s, SbDup p,          \
                                                     int* dist) {                              \
        sb_dup_body<G, K>(tb, s, p, dist);                                                     \
    }
#define SB_DUP_KERNELS(NAME, K)   \
    SB_DUP_KERNEL(NAME, 4, K)     \
    SB_DUP_KERNEL(NAME, 8, K)     \
    SB_DUP_KERNEL(NAME, 16, K)    \
    SB_DUP_KERNEL(NAME, 32, K)
SB_DUP_KERNELS(narrow, SbDupNarrow)
SB_DUP_KERNELS(wide, SbDupWide)
SB_DUP_KERNELS(detail, SbDupFull)

__device__ __forceinline__ bool sb_dup_routed(const SbTables& tb, const SbSpace& s, const SbDup& p,
                                              sb_u64 t) {
    if (t >= s.n) {
        return false;
    }
    sb_u64 i, j, n, m;
    sb_decode(s, t, i, j);
    sb_dup_lengths(tb, i, j, n, m);
    // The width does not enter the routing predicate; a bound-resolved item is never routed.
    return sb_dup_tier(p, 4, n, m) == SB_DUP_HOST && !sb_dup_skipped(p, t);
}

// Routing step 1 (§5.6, the form of G1's Hits): routed count of each block's
// contiguous tile of blockDim.x items.
extern "C" __global__ void sb_dup_route_count(SbTables tb, SbSpace s, SbDup p,
                                              sb_u64* block_counts) {
    __shared__ unsigned warp_counts[32];
    const sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x;
    const bool routed = sb_dup_routed(tb, s, p, t);
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
extern "C" __global__ void sb_dup_route_write(SbTables tb, SbSpace s, SbDup p,
                                              const sb_u64* block_offsets, sb_u64* out) {
    __shared__ unsigned warp_base[32];
    const sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x;
    const bool routed = sb_dup_routed(tb, s, p, t);
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

// Host results of routed pairs into the chunk's buffers: keys, and in a detail pass
// the three field bit patterns (`f`: three per entry).
extern "C" __global__ void sb_dup_scatter(const sb_u64* pos, const int* d, const sb_u64* f,
                                          sb_u64 count, int* dist, sb_u64* dh, sb_u64* ds,
                                          sb_u64* tm) {
    const sb_u64 stride = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 k = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; k < count; k += stride) {
        const sb_u64 t = pos[k];
        dist[t] = d[k];
        if (f != 0 && dh != 0) {
            dh[t] = f[3 * k];
            ds[t] = f[3 * k + 1];
            tm[t] = f[3 * k + 2];
        }
    }
}
