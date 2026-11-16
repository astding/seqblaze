// The C1 admissible bound of K1 native `duplex` (model "mfe") on the device: the relaxed
// integer recurrence of docs/wiki/thermo/duplex-mfe-bound.md §3.5 with flag packing
// (§4.6), one thread per pair. Written from that page only.
//
// Integers only: a packed value 2v + (1 - flag) in int64 with +inf = 2^60, normalised at
// 2^59 (§4.6: W < 2^56). Integer min is exact, associative and commutative, so the
// device's (LB, chi, outcome) equal the CPU's byte for byte (§8 "Byte identity").
#pragma once
#include "hamming.cuh"

#define SB_BND_INF (1ll << 60)
#define SB_BND_LIMIT (1ll << 59)
#define SB_BND_NO_PAIR 255
#define SB_BND_NONE 4
#define SB_BND_U_E 4
#define SB_BND_SENTINEL 2147483646ll
// Per relaxed class: class, plus, minus, n_lower, 5 x (depth, start, gamma), n_upper,
// 4 x (start, reach, gamma); 32 words (mirrors `bound_words` in src/dupbound.rs).
#define SB_BND_CLASS_WORDS 32

// Mirrors `SbDupBound` in src/dupbound.rs; every field is 8 bytes wide.
struct SbDupBound {
    sb_u64 keys;      // const sb_i64*: D of every cell, +inf = 2^60
    sb_u64 cls;       // const sb_u8*: CLS over (lam + 1)^2
    sb_u64 relaxed;   // const sb_i64*: SB_BND_CLASS_WORDS per relaxed class
    sb_u64 n_relaxed;
    sb_u64 lam;
    sb_u64 n_p;
    sb_u64 n_c;
    sb_u64 has_states;
    sb_u64 off[14];   // table starts: INIT, INIT_SYM, END_L, END_R, STACK, BULGE, I11, I12,
                      // I21, I22, GEN, LOOP_OUT, LOOP_IN, STATES
    sb_u64 codes;     // const sb_u8*: letter codes of every strand
    sb_u64 a_start;   // const sb_u64*: start of item t's a in `codes`
    sb_u64 a_len;     // const sb_u64*
    sb_u64 b_start;
    sb_u64 b_len;
    sb_u64 n_items;
    sb_u64 scratch;   // sb_i64*: `stride` words per thread
    sb_u64 stride;
    sb_u64 has_theta;
    sb_i64 theta;
    sb_u64 lb_plus_one;
    sb_u64 out_lb;    // sb_i64*
    sb_u64 out_chi;   // sb_u8*
    sb_u64 out_code;  // sb_u8*
};

__device__ __forceinline__ bool sb_bnd_fin(sb_i64 x) { return x < SB_BND_LIMIT; }
__device__ __forceinline__ sb_i64 sb_bnd_min(sb_i64 x, sb_i64 y) { return x < y ? x : y; }

__device__ __forceinline__ unsigned sb_bnd_pair(const SbDupBound& p, unsigned x, unsigned y) {
    if (x + y == 3u) {
        return x;  // A-T 0, C-G 1, G-C 2, T-A 3
    }
    if (p.n_p == 6) {
        if (x == 2u && y == 3u) return 4u;  // G-T
        if (x == 3u && y == 2u) return 5u;  // T-G
    }
    return SB_BND_NO_PAIR;
}

// Y of §3.4 as a plain value: RA(k, l) + OUT_c(k, l), +inf unless RA is finite,
// k + 1 < n, l >= 1 and OUT_c is finite.
__device__ __forceinline__ sb_i64 sb_bnd_y(const SbDupBound& p, const sb_i64* K, const sb_u8* a,
                                           const sb_u8* b, sb_u64 n, sb_i64 ra, sb_u64 k, sb_u64 l,
                                           sb_u64 cl) {
    if (!sb_bnd_fin(ra) || k + 1 >= n || l == 0) {
        return SB_BND_INF;
    }
    const sb_u64 pk = sb_bnd_pair(p, a[k], b[l]);
    const sb_i64 cost = K[p.off[11] + ((cl * p.n_p + pk) * 4 + a[k + 1]) * 4 + b[l - 1]];
    if (!sb_bnd_fin(cost)) {
        return SB_BND_INF;
    }
    return (ra >> 1) + cost;
}

// The packed minimum LB of one pair (n, m >= 2); +inf if empty.
__device__ sb_i64 sb_bnd_fill(const SbDupBound& p, const sb_u8* a, sb_u64 n, const sb_u8* b,
                              sb_u64 m, sb_i64* scratch) {
    const sb_i64* K = (const sb_i64*)p.keys;
    const sb_u8* CLS = (const sb_u8*)p.cls;
    const sb_i64* R = (const sb_i64*)p.relaxed;
    const sb_u64 lam = p.lam, w = lam + 1, np = p.n_p, nc = p.n_c, nr = p.n_relaxed;
    const sb_u64 rows = n < lam + 2 ? n : lam + 2;
    const sb_u64 cells = rows * m;
    sb_i64* ra_band = scratch;
    sb_i64* out_band = ra_band + cells;
    sb_i64* f_band = out_band + cells * nc;
    sb_i64* g_band = f_band + cells * nr;
    for (sb_u64 x = 0; x < cells * (1 + nc + 2 * nr); ++x) {
        scratch[x] = SB_BND_INF;
    }
    bool identical = n == m;
    for (sb_u64 k = 0; identical && k < n; ++k) {
        identical = a[k] == b[k];
    }
    bool palindrome = identical;
    for (sb_u64 k = 0; palindrome && k < n; ++k) {
        palindrome = a[k] + a[n - 1 - k] == 3;
    }
    const sb_i64 head = K[palindrome ? 1 : 0];
    sb_i64 best = SB_BND_INF;
    for (sb_u64 i = 0; i < n; ++i) {
        const sb_u64 cur = i % rows;
        const sb_u64 prev_slot = cur == 0 ? rows - 1 : cur - 1;
        const sb_u64 base = cur * m;
        for (sb_u64 j = 0; j < m; ++j) {
            const unsigned pi = sb_bnd_pair(p, a[i], b[j]);
            if (pi == SB_BND_NO_PAIR) {
                ra_band[base + j] = SB_BND_INF;
                for (sb_u64 cl = 0; cl < nc; ++cl) out_band[(base + j) * nc + cl] = SB_BND_INF;
                continue;
            }
            sb_i64 c = SB_BND_INF;
            if (i >= 1 && j + 1 < m) {
                sb_i64 inner[3] = {SB_BND_INF, SB_BND_INF, SB_BND_INF};
                for (sb_u64 cl = 0; cl < nc; ++cl) {
                    inner[cl] = K[p.off[12] + ((cl * 4 + a[i - 1]) * 4 + b[j + 1]) * np + pi];
                }
                const sb_u64 umax = lam < i - 1 ? lam : i - 1;
                const sb_u64 room = m - 2 - j;
                sb_u64 slot = prev_slot;
                for (sb_u64 u = 0; u <= umax; ++u) {
                    const sb_u64 k = i - 1 - u;
                    const sb_u64 row = slot * m;
                    const sb_i64 prev0 = ra_band[row + j + 1];
                    if (sb_bnd_fin(prev0)) {
                        const sb_u64 pk = sb_bnd_pair(p, a[k], b[j + 1]);
                        sb_i64 cost = u == 0 ? K[p.off[4] + pk * np + pi]
                                             : K[p.off[5] + ((u - 1) * np + pk) * np + pi];
                        if (u == 1 && p.has_states && sb_bnd_fin(cost)) {
                            // ρ of a[k + 1]: the run of identical letters containing it.
                            sb_u64 lo = k + 1, hi = k + 1;
                            while (lo > 0 && a[lo - 1] == a[k + 1]) --lo;
                            while (hi + 1 < n && a[hi + 1] == a[k + 1]) ++hi;
                            cost += K[p.off[13] + (hi - lo)];
                        }
                        if (sb_bnd_fin(cost)) {
                            c = sb_bnd_min(c, prev0 + 2 * cost);
                        }
                    }
                    sb_u64 vmax = 0;
                    if (u == 0) {
                        vmax = lam < room ? lam : room;
                    } else if (u <= SB_BND_U_E) {
                        vmax = SB_BND_U_E < lam - u ? SB_BND_U_E : lam - u;
                        vmax = vmax < room ? vmax : room;
                    }
                    for (sb_u64 v = 1; v <= vmax; ++v) {
                        const sb_u64 l = j + 1 + v;
                        sb_i64 cand;
                        if (u == 0) {
                            const sb_i64 prev = ra_band[row + l];
                            if (!sb_bnd_fin(prev)) continue;
                            const sb_u64 pk = sb_bnd_pair(p, a[k], b[l]);
                            sb_i64 cost = K[p.off[5] + ((lam + v - 1) * np + pk) * np + pi];
                            if (v == 1 && p.has_states && sb_bnd_fin(cost)) {
                                sb_u64 lo = l - 1, hi = l - 1;
                                while (lo > 0 && b[lo - 1] == b[l - 1]) --lo;
                                while (hi + 1 < m && b[hi + 1] == b[l - 1]) ++hi;
                                cost += K[p.off[13] + (hi - lo)];
                            }
                            if (!sb_bnd_fin(cost)) continue;
                            cand = prev + 2 * cost;
                        } else if (u <= 2 && v <= 2) {
                            const sb_i64 prev = ra_band[row + l];
                            if (!sb_bnd_fin(prev)) continue;
                            const sb_u64 pk = sb_bnd_pair(p, a[k], b[l]);
                            const sb_u64 p1 = a[k + 1], q1 = b[l - 1];
                            sb_i64 cost;
                            if (u == 1 && v == 1) {
                                cost = K[p.off[6] + ((pk * 4 + p1) * 4 + q1) * np + pi];
                            } else if (u == 1) {
                                cost = K[p.off[7] + (((pk * 4 + p1) * 4 + q1) * 4 + b[l - 2]) * np + pi];
                            } else if (v == 1) {
                                cost = K[p.off[8] + (((pk * 4 + p1) * 4 + a[k + 2]) * 4 + q1) * np + pi];
                            } else {
                                cost = K[p.off[9] +
                                         ((((pk * 4 + p1) * 4 + a[k + 2]) * 4 + q1) * 4 + b[l - 2]) * np +
                                         pi];
                            }
                            if (!sb_bnd_fin(cost)) continue;
                            cand = prev + 2 * cost;
                        } else {
                            const sb_u64 cl = CLS[u * w + v];
                            const sb_i64 prev = out_band[(row + l) * nc + cl];
                            const sb_i64 rest = K[p.off[10] + u * w + v];
                            if (!sb_bnd_fin(prev) || !sb_bnd_fin(rest) || !sb_bnd_fin(inner[cl])) continue;
                            cand = prev + 2 * (rest + inner[cl]);
                        }
                        c = sb_bnd_min(c, cand);
                    }
                    slot = slot == 0 ? rows - 1 : slot - 1;
                }
                // Relaxed edges: the region queries of every relaxed class (§3.4).
                const sb_i64 iq = (sb_i64)(i - 1), jq = (sb_i64)(j + 1);
                for (sb_u64 t = 0; t < nr; ++t) {
                    const sb_i64* rc = R + t * SB_BND_CLASS_WORDS;
                    const sb_i64 x = inner[rc[0]];
                    if (!sb_bnd_fin(x)) continue;
                    const sb_i64 plus = rc[1], minus = rc[2];
                    const sb_i64* f = f_band + t * cells;
                    const sb_i64* g = g_band + t * cells;
                    const sb_i64 lower_off = x + plus * iq - minus * jq;
                    for (sb_i64 r = 0; r < rc[3]; ++r) {
                        const sb_u64 depth = (sb_u64)rc[4 + 3 * r], start = (sb_u64)rc[5 + 3 * r];
                        const sb_i64 gamma = rc[6 + 3 * r];
                        const sb_u64 col = j + 1 + start;
                        if (depth > i - 1 || col >= m) continue;
                        const sb_i64 value = f[((i - 1 - depth) % rows) * m + col];
                        if (sb_bnd_fin(value)) c = sb_bnd_min(c, 2 * (value + gamma + lower_off) + 1);
                    }
                    const sb_i64 upper_off = x + minus * iq - plus * jq;
                    for (sb_i64 r = 0; r < rc[19]; ++r) {
                        const sb_u64 start = (sb_u64)rc[20 + 3 * r], reach = (sb_u64)rc[21 + 3 * r];
                        const sb_i64 gamma = rc[22 + 3 * r];
                        const sb_u64 col = j + 1 + reach;
                        if (start > i - 1 || col >= m) continue;
                        const sb_i64 value = g[((i - 1 - start) % rows) * m + col];
                        if (sb_bnd_fin(value)) c = sb_bnd_min(c, 2 * (value + gamma + upper_off) + 1);
                    }
                }
            }
            if (!sb_bnd_fin(c)) c = SB_BND_INF;
            const sb_u64 alpha = i >= 1 ? a[i - 1] : SB_BND_NONE;
            const sb_u64 beta = j + 1 < m ? b[j + 1] : SB_BND_NONE;
            const sb_i64 left = K[p.off[2] + (alpha * np + pi) * 5 + beta];
            const sb_i64 first =
                sb_bnd_fin(head) && sb_bnd_fin(left) ? 2 * (head + left) : SB_BND_INF;
            const sb_i64 value = sb_bnd_min(c, first);
            ra_band[base + j] = value;
            if (sb_bnd_fin(value) && i + 1 < n && j >= 1) {
                const sb_u64 tail = (sb_u64)a[i + 1] * 4 + b[j - 1];
                for (sb_u64 cl = 0; cl < nc; ++cl) {
                    const sb_i64 cost = K[p.off[11] + (cl * np + pi) * 16 + tail];
                    out_band[(base + j) * nc + cl] = sb_bnd_fin(cost) ? value + 2 * cost : SB_BND_INF;
                }
            } else {
                for (sb_u64 cl = 0; cl < nc; ++cl) out_band[(base + j) * nc + cl] = SB_BND_INF;
            }
            if (sb_bnd_fin(c)) {
                const sb_u64 ra_ = i + 1 < n ? a[i + 1] : SB_BND_NONE;
                const sb_u64 rb_ = j >= 1 ? b[j - 1] : SB_BND_NONE;
                const sb_i64 right = K[p.off[3] + ((sb_u64)pi * 5 + ra_) * 5 + rb_];
                if (sb_bnd_fin(right)) best = sb_bnd_min(best, c + 2 * right);
            }
        }
        // Row i is final: F_c(i, ·) and G_c(i, ·) of every relaxed class (§3.4).
        const sb_i64 ii = (sb_i64)i;
        for (sb_u64 t = 0; t < nr; ++t) {
            const sb_i64* rc = R + t * SB_BND_CLASS_WORDS;
            const sb_u64 cl = (sb_u64)rc[0];
            const sb_i64 plus = rc[1], minus = rc[2];
            sb_i64* f = f_band + t * cells;
            sb_i64* g = g_band + t * cells;
            for (sb_u64 l = 0; l < m; ++l) {
                const sb_i64 y = sb_bnd_y(p, K, a, b, n, ra_band[base + l], i, l, cl);
                sb_i64 fv = sb_bnd_fin(y) ? y - plus * ii + minus * (sb_i64)l : SB_BND_INF;
                if (i >= 1) {
                    fv = sb_bnd_min(fv, f[prev_slot * m + l]);
                    if (l + 1 < m) fv = sb_bnd_min(fv, f[prev_slot * m + l + 1]);
                }
                f[cur * m + l] = fv;
            }
            for (sb_u64 l = m; l-- > 0;) {
                const sb_i64 y = sb_bnd_y(p, K, a, b, n, ra_band[base + l], i, l, cl);
                sb_i64 gv = sb_bnd_fin(y) ? y - minus * ii + plus * (sb_i64)l : SB_BND_INF;
                if (l + 1 < m) {
                    gv = sb_bnd_min(gv, g[cur * m + l + 1]);
                    if (i >= 1) gv = sb_bnd_min(gv, g[prev_slot * m + l + 1]);
                }
                g[cur * m + l] = gv;
            }
        }
    }
    return best;
}

// The outcome code of (LB, χ) against θ (§1; 0 open, 1 certified, 2 pruned) and
// λ = min(LB, the sentinel key) (the sentinel for +∞), after the `lb_plus_one` testing
// fault. No exact engine key exceeds the empty-space sentinel, so a finite LB above it (int64
// width) is clamped before the comparison and never certified, as in the CPU's outcome.
// Shared by the testing kernel below and the resident stage (resident_bound.cuh).
__device__ __forceinline__ unsigned sb_bnd_code(const SbDupBound& p, bool finite, sb_i64 lb,
                                                bool chi, bool has_theta, sb_i64 theta,
                                                sb_i64& lambda) {
    if (finite && p.lb_plus_one) lb += 1;
    const bool above = finite && lb > SB_BND_SENTINEL;
    lambda = (finite && !above) ? lb : SB_BND_SENTINEL;
    if (has_theta && lambda > theta) {
        return 2u;
    }
    if (chi && !above && lambda >= -2147483648ll) {
        return 1u;  // a certified λ is an int32 engine key (as the CPU's outcome)
    }
    return 0u;
}

// One thread per item; grid-stride over items, each with its own scratch slot. Writes
// LB (INT64_MAX for +inf), chi and the outcome code (0 open, 1 certified, 2 pruned).
extern "C" __global__ void sb_dup_bound(SbDupBound p) {
    const sb_u64 threads = (sb_u64)gridDim.x * blockDim.x;
    const sb_u64 self = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x;
    sb_i64* scratch = (sb_i64*)p.scratch + self * p.stride;
    const sb_u8* codes = (const sb_u8*)p.codes;
    for (sb_u64 t = self; t < p.n_items; t += threads) {
        const sb_u64 n = ((const sb_u64*)p.a_len)[t], m = ((const sb_u64*)p.b_len)[t];
        const sb_u8* a = codes + ((const sb_u64*)p.a_start)[t];
        const sb_u8* b = codes + ((const sb_u64*)p.b_start)[t];
        bool finite = false;
        sb_i64 lb = 0;
        bool chi = true;
        if (n >= 2 && m >= 2) {
            const sb_i64 best = sb_bnd_fill(p, a, n, b, m, scratch);
            if (sb_bnd_fin(best)) {
                finite = true;
                lb = best >> 1;
                chi = (best & 1) == 0;
            }
        }
        sb_i64 lambda;
        const unsigned char code =
            (unsigned char)sb_bnd_code(p, finite, lb, chi, p.has_theta != 0, p.theta, lambda);
        if (finite && p.lb_plus_one) lb += 1;
        ((sb_i64*)p.out_lb)[t] = finite ? lb : 0x7fffffffffffffffll;
        ((sb_u8*)p.out_chi)[t] = chi ? 1 : 0;
        ((sb_u8*)p.out_code)[t] = code;
    }
}
