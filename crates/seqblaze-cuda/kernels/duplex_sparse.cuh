// Resident fast entries of K1 `duplex(model="mfe")` and K3 `duplex(model="pf")` with row
// compaction and sparse pair predecessors. Spec: docs/wiki/gpu/duplex-cuda.md §15.4 step 3
// and docs/wiki/gpu/duplex-pf-cuda.md §19.4 steps 2-3. Written from those pages and from
// this repository's own duplex.cuh and duplex_pf.cuh, whose recurrence, key types, tables
// and per-candidate arithmetic are reused unchanged.
//
// One whole warp per pair (g = 32), rows in order with one __syncwarp per row, as the dense
// bodies. Two things differ, both only in which candidates a lane visits:
//  - row compaction: in row i the lanes take only the pair cells (i, l), l in L_{a_i}, the
//    k-th of them by lane k mod 32 (any partition of a row's cells, §5.3 / §6.2);
//    non-pair cells are neither evaluated nor written;
//  - sparse predecessors: for each u a lane walks only the predecessor columns l of row k
//    that pair with a_k, ascending in l (so ascending v); these are exactly the
//    predecessors that can be finite (K1) or nonzero (K3).
// The skipped predecessors are the non-pair cells, whose A is +inf (K1) and whose ZA and ZO
// are zero (K3); every slot that is read is a pair cell written for this pair before its
// row barrier (§15.4 step 3, §19.4 step 2). These kernels exist only on the resident fast
// path; the Session runner and the continuation keep the dense helpers.
#pragma once
#include "duplex.cuh"
#include "duplex_pf.cuh"

// The list area of one group, after its q staged bytes (§15.4 step 3): for each top letter
// x the ascending columns L_x of b (or of the canonical bottom strand y) with pt(x, b_l)
// >= 0 at x·200, the counts nxt_x[j] = |{l in L_x : l < j}| for j = 0 .. m at 800 + x·201,
// and c_x at 1604 + x. 1,608 bytes, 1,616 reserved.
#define SB_SPARSE_LISTS 1616
#define SB_SPARSE_NXT 800
#define SB_SPARSE_CNT 1604

// Builds the lists of the bottom strand `y` of length m in `lists`, all 32 lanes of the
// warp (one ballot per letter per 32 columns). The caller's __syncwarp publishes them.
__device__ __forceinline__ void sb_sparse_lists(const sb_u8* y, int m, sb_u8* lists, int me,
                                                sb_u64 np) {
    const unsigned below = (1u << me) - 1u;
    int base0 = 0, base1 = 0, base2 = 0, base3 = 0;
    for (int l0 = 0; l0 < m; l0 += 32) {
        const int l = l0 + me;
        const bool in = l < m;
        const int yl = in ? (int)y[l] : 0;
#pragma unroll
        for (int x = 0; x < 4; ++x) {
            const bool hit = in && sb_dup_pt(x, yl, np) >= 0;
            const unsigned ballot = __ballot_sync(0xffffffffu, hit);
            const int base = x == 0 ? base0 : (x == 1 ? base1 : (x == 2 ? base2 : base3));
            const int rank = base + __popc(ballot & below);
            if (in) {
                lists[SB_SPARSE_NXT + x * 201 + l] = (sb_u8)rank;
            }
            if (hit) {
                lists[x * 200 + rank] = (sb_u8)l;
            }
            const int total = base + __popc(ballot);
            if (x == 0) base0 = total;
            if (x == 1) base1 = total;
            if (x == 2) base2 = total;
            if (x == 3) base3 = total;
        }
    }
    if (me < 4) {
        const int total = me == 0 ? base0 : (me == 1 ? base1 : (me == 2 ? base2 : base3));
        lists[SB_SPARSE_NXT + me * 201 + m] = (sb_u8)total;
        lists[SB_SPARSE_CNT + me] = (sb_u8)total;
    }
}

// ------------------------------------------------------------------ K1

// A(i, j) of §5.1 for a pair cell, as `sb_dup_cell`, visiting only pair predecessors.
// `cur` is the ring slot of row i (rows are written in order to slots 0, 1, .., R − 1, 0, ..).
template <class K>
__device__ __forceinline__ typename K::T sb_dup_sparse_cell(const SbDup& p, const sb_u8* ca,
                                                            const sb_u8* cb, const sb_u8* lists,
                                                            int n, int m, int rows,
                                                            const typename K::T* band, int cur,
                                                            int i, int j,
                                                            const typename K::T& start,
                                                            typename K::T& best) {
    typedef typename K::T T;
    const int lam = (int)p.lam;
    const int np = (int)p.n_p;
    const int pi = sb_dup_pt(ca[i], cb[j], p.n_p);  // a pair cell: pi >= 0
    T c = K::inf();
    if (i >= 1 && j + 1 < m) {
        const unsigned in_at = (unsigned)p.loop_in + (unsigned)((ca[i - 1] * 4 + cb[j + 1]) * np + pi);
        const T in0 = K::load(p, in_at);
        T in1 = in0;
        T in2 = in0;
        if (p.n_c > 1) {
            in1 = K::load(p, in_at + 16u * (unsigned)np);
            in2 = K::load(p, in_at + 32u * (unsigned)np);
        }
        T st_top = K::inf();
        T st_bottom = K::inf();
        if (p.has_states != 0) {
            const sb_u8* rho = ca + SB_DUP_RHO;
            st_top = K::load(p, p.states + (sb_u64)(rho[i - 1] - 1));
            st_bottom = K::load(p, p.states + (sb_u64)(rho[SB_DUP_MAX_LENGTH + j + 1] - 1));
        }
        const sb_u8* cls = (const sb_u8*)p.cls;
        const unsigned o_stack = (unsigned)p.stack;
        const unsigned o_bulge = (unsigned)p.bulge;
        const unsigned o_gen = (unsigned)p.gen;
        const unsigned o_out = (unsigned)p.loop_out;
        const int umax = lam < i - 1 ? lam : i - 1;
        // Ring slot of row k = i − 1 − u: one below the previous, wrapped by one addition.
        int slot = cur - 1;
        if (slot < 0) {
            slot += rows;
        }
        for (int u = 0; u <= umax; ++u) {
            const int k = i - 1 - u;
            const int ak = ca[k];
            const T* row = band + slot * m;
            const int vmax = lam - u < m - 2 - j ? lam - u : m - 2 - j;
            const int lmax = j + 1 + vmax;
            const sb_u8* lst = lists + ak * 200;
            const int cn = lists[SB_SPARSE_CNT + ak];
            for (int t = lists[SB_SPARSE_NXT + ak * 201 + j + 1]; t < cn; ++t) {
                const int l = lst[t];
                if (l > lmax) {
                    break;
                }
                const int v = l - j - 1;
                const int pk = sb_dup_pk(ak, cb[l], p.n_p);
                T cand;
                if (v == 0) {
                    // A stack (u = 0) or a top bulge of size u.
                    const unsigned at0 = u == 0 ? o_stack + (unsigned)(pk * np + pi)
                                                : o_bulge + (unsigned)(((u - 1) * np + pk) * np + pi);
                    cand = K::plus(row[l], K::load(p, at0));
                    if (u == 1 && p.has_states != 0) {
                        cand = K::plus(cand, st_top);  // three terms (§5.2 F2)
                    }
                } else if (u == 0) {
                    // A bottom bulge of size v.
                    cand = K::plus(row[l],
                                   K::load(p, o_bulge + (unsigned)(((lam + v - 1) * np + pk) * np + pi)));
                    if (v == 1 && p.has_states != 0) {
                        cand = K::plus(cand, st_bottom);
                    }
                } else if (u <= 2 && v <= 2) {
                    const int p1 = ca[k + 1];
                    const int q1 = cb[l - 1];
                    unsigned at;
                    if (u == 1 && v == 1) {
                        at = (unsigned)p.i11 + (unsigned)(((pk * 4 + p1) * 4 + q1) * np + pi);
                    } else if (u == 1) {
                        const int q2 = cb[l - 2];
                        at = (unsigned)p.i12 +
                             (unsigned)((((pk * 4 + p1) * 4 + q1) * 4 + q2) * np + pi);
                    } else if (v == 1) {
                        const int p2 = ca[k + 2];
                        at = (unsigned)p.i21 +
                             (unsigned)((((pk * 4 + p1) * 4 + p2) * 4 + q1) * np + pi);
                    } else {
                        const int p2 = ca[k + 2];
                        const int q2 = cb[l - 2];
                        at = (unsigned)p.i22 +
                             (unsigned)(((((pk * 4 + p1) * 4 + p2) * 4 + q1) * 4 + q2) * np + pi);
                    }
                    cand = K::plus(row[l], K::load(p, at));
                } else {
                    // Generic: A(k, l) + LOOP_OUT + GEN + LOOP_IN of the shape's class c,
                    // four terms (§5.2 F2).
                    const int shape = u * (lam + 1) + v;
                    const int cl = (int)__ldg(cls + shape);
                    const T out = K::load(
                        p, o_out + (unsigned)(((cl * np + pk) * 4 + ca[k + 1]) * 4 + cb[l - 1]));
                    const T gen = K::load(p, o_gen + (unsigned)shape);
                    const T inner = cl == 0 ? in0 : (cl == 1 ? in1 : in2);
                    cand = K::plus(K::plus(K::plus(row[l], out), gen), inner);
                }
                c = sb_dup_min<K>(c, cand);
            }
            slot -= 1;
            if (slot < 0) {
                slot += rows;
            }
        }
    }
    c = K::norm(c);
    const int alpha = i >= 1 ? ca[i - 1] : SB_DUP_NONE;
    const int beta = j + 1 < m ? cb[j + 1] : SB_DUP_NONE;
    const T first =
        K::norm(K::plus(start, K::load(p, p.end_l + (sb_u64)((alpha * np + pi) * 5 + beta))));
    const int alpha2 = i + 1 < n ? ca[i + 1] : SB_DUP_NONE;
    const int beta2 = j >= 1 ? cb[j - 1] : SB_DUP_NONE;
    const T end = K::plus(c, K::load(p, p.end_r + (sb_u64)((pi * 5 + alpha2) * 5 + beta2)));
    best = sb_dup_min<K>(best, end);
    return K::less(c, first) ? c : first;
}

// The resident fast DP of key type K at g = 32 in the global tier: one warp per pair,
// grid-striding over the launch's items and skipping the closed-form and continued pairs
// (the unchanged tier predicate) and the items the C1 bounded stage resolved
// (SbDup::skip, duplex-mfe-bound.md §8). Dynamic shared memory per warp: `p.stride` bytes, the
// q = `p.codes` staged bytes, then the list area.
template <class K>
__device__ __forceinline__ void sb_dup_sparse_body(const SbTables& tb, const SbSpace& s,
                                                   const SbDup& p, int* dist) {
    typedef typename K::T T;
    extern __shared__ __align__(16) sb_u8 sb_dup_sparse_smem[];
    const int me = (int)(threadIdx.x & 31u);
    const unsigned mask = 0xffffffffu;
    const sb_u64 per_block = blockDim.x / 32;
    const sb_u64 local = threadIdx.x / 32;
    const sb_u64 group = (sb_u64)blockIdx.x * per_block + local;
    const sb_u64 groups = (sb_u64)gridDim.x * per_block;
    sb_u8* mine = sb_dup_sparse_smem + local * p.stride;
    sb_u8* ca = mine;
    sb_u8* cb = mine + SB_DUP_MAX_LENGTH;
    sb_u8* lists = mine + p.codes;
    T* band = (T*)((sb_u8*)p.slots + group * p.slot);
    for (sb_u64 t = group; t < s.n; t += groups) {
        sb_u64 i, j, n64, m64;
        sb_decode(s, t, i, j);
        sb_dup_lengths(tb, i, j, n64, m64);
        if (sb_dup_group(p, m64) != 32ull ||
            sb_dup_tier(p, K::BYTES, n64, m64) != SB_DUP_GLOBAL || sb_dup_skipped(p, t)) {
            continue;  // the closed-form kernel's, the continuation's or the bound's pair
        }
        const int n = (int)n64;
        const int m = (int)m64;
        const int rows = n < (int)p.lam + 2 ? n : (int)p.lam + 2;
        const SbSequence pa = sb_a_row(tb, i);
        const SbSequence pb = sb_b_row(tb, j);
        for (int x = me; x < n; x += 32) {
            ca[x] = sb_dup_code(pa[x]);
        }
        for (int x = me; x < m; x += 32) {
            cb[x] = sb_dup_code(pb[x]);
        }
#ifdef SB_TESTING
        if (me == 0) sb_resident_poison_count(p.resident_stats, 0, 0, 1);
        if (p.poison < 256) {
            // Testing hook (§8 item 8): the pair's whole band is filled with the poison
            // byte; a read of a slot not written for this pair would show.
            sb_u8* bytes = (sb_u8*)band;
            const int total = rows * m * (int)K::BYTES;
            for (int x = me; x < total; x += 32) {
                bytes[x] = (sb_u8)p.poison;
            }
        }
#endif
        __syncwarp(mask);
#ifdef SB_TESTING
        if (me == 0 && p.poison < 256) {
            sb_resident_poison_count(p.resident_stats, 0, 1, 1);
            sb_resident_poison_count(p.resident_stats, 0, 2, (sb_u64)rows * m * K::BYTES);
        }
#endif
        if (p.has_states != 0) {
            // The ρ prologue (nndb2024-dna.md §5.4.4), read by the rows below.
            sb_dup_runs(ca, n, ca + SB_DUP_RHO, me, 32);
            sb_dup_runs(cb, m, ca + SB_DUP_RHO + SB_DUP_MAX_LENGTH, me, 32);
        }
        sb_sparse_lists(cb, m, lists, me, p.n_p);
        __syncwarp(mask);
        // identical: a = b; sym: identical and a palindrome (a_k + a_{n-1-k} = 3).
        bool same = n == m;
        bool pal = true;
        for (int x = me; x < n; x += 32) {
            same = same && x < m && ca[x] == cb[x];
            pal = pal && ca[x] + ca[n - 1 - x] == 3;
        }
        const bool identical = __all_sync(mask, same) != 0;
        const bool sym = identical && __all_sync(mask, pal) != 0;
        const T start = K::load(p, sym ? 1ull : 0ull);
        T best = K::inf();
        int cur = 0;
        for (int r = 0; r < n; ++r) {
            T* slot = band + cur * m;
            const int ar = ca[r];
            const sb_u8* row_list = lists + ar * 200;
            const int count = lists[SB_SPARSE_CNT + ar];
            for (int k = me; k < count; k += 32) {
                const int c = row_list[k];
                slot[c] = sb_dup_sparse_cell<K>(p, ca, cb, lists, n, m, rows, band, cur, r, c,
                                                start, best);
            }
            // Row r's values are written before any lane reads them, and the slot of row
            // r + 1 − R is overwritten only after every read of row r (§5.3).
            __syncwarp(mask);
            cur += 1;
            if (cur == rows) {
                cur = 0;
            }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            best = sb_dup_min<K>(best, K::shfl(mask, best, offset, 32));
        }
        const T key = K::norm(best);
        if (me == 0) {
            const bool finite = K::finite(key);
            dist[t] = finite ? (int)K::d(key) : SB_DUP_SENTINEL_KEY;
            sb_dup_fields(p, t, key, identical);
        }
        // The next pair's staging, lists and band writes follow every read of this one.
        __syncwarp(mask);
    }
}

// No __launch_bounds__ on the score entries: with it, ptxas spilled (narrow 64 registers
// with 16 B of spill stores); without, they use 64 and 72 registers and no spill.
extern "C" __global__ void sb_dup_res_narrow(SbTables tb, SbSpace s, SbDup p, int* dist) {
    sb_dup_sparse_body<SbDupNarrow>(tb, s, p, dist);
}
extern "C" __global__ void sb_dup_res_wide(SbTables tb, SbSpace s, SbDup p, int* dist) {
    sb_dup_sparse_body<SbDupWide>(tb, s, p, dist);
}
// At most 128 registers, so that 16 warps fit per SM (§15.2, the fields="all" target); a
// compile attribute that changes no result.
extern "C" __global__ void __launch_bounds__(128, 4)
    sb_dup_res_detail(SbTables tb, SbSpace s, SbDup p, int* dist) {
    sb_dup_sparse_body<SbDupFull>(tb, s, p, dist);
}

// ------------------------------------------------------------------ K3

// The tier of a pair for the resident sparse K3 entries (§19.4 step 3): `sb_pf_tier` with
// the list area counted in the fits term, (32 >> lg)·(band + 400 + L) + s_block <= s_max.
__device__ __forceinline__ int sb_pf_sparse_tier(const SbPf& p, sb_u64 n, sb_u64 m) {
    if (n < 2 || m < 2) {
        return SB_PF_CLOSED;
    }
    const sb_u64 g = sb_pf_group(p, n, m);
    const sb_u64 lg = sb_pf_log2(g);
    const sb_u64 cols_a = n * ((m + g - 1) >> lg);
    const sb_u64 cols_b = m * ((n + g - 1) >> lg);
    const sb_u64 depth = sb_dup_c(n, m, p.lam) * (cols_a > cols_b ? cols_a : cols_b);
    if (depth > p.pair_cells) {
        return SB_PF_HOST;
    }
    const sb_u64 b1 = sb_pf_band(p, n, m);
    const sb_u64 b2 = sb_pf_band(p, m, n);
    const sb_u64 band = b1 > b2 ? b1 : b2;
    const bool fits =
        (32ull >> lg) * (band + SB_PF_CODES + SB_SPARSE_LISTS) + p.s_block <= p.s_max;
    return band <= p.band_bytes && fits ? SB_PF_SHARED : SB_PF_GLOBAL;
}

// ZA, the N_C values ZO_c and the end term of one pair cell (i, j) of §6.1, as
// `sb_pf_cell`, visiting only pair predecessors in order O (u ascending, then l and so v
// ascending); zero predecessors are skipped as there.
__device__ __forceinline__ void sb_pf_sparse_cell(const SbPf& p, const SbF53* w, const sb_u8* x,
                                                  const sb_u8* y, const sb_u8* lists, int n,
                                                  int m, int rows, const SbF53* za,
                                                  const SbF53* zo, int cur, int i, int j,
                                                  int sigma, SbF53& za_out, SbF53* zo_out,
                                                  SbF53& end_out) {
    const int lam = (int)p.lam;
    const sb_u64 np = p.n_p;
    const int nc = (int)p.n_c;
    const int pi = sb_dup_pt(x[i], y[j], np);  // a pair cell: pi >= 0
    SbF53 special = sb_f53_zero();
    if (i >= 1 && j + 1 < m) {
        SbF53 g0 = sb_f53_zero();
        SbF53 g1 = sb_f53_zero();
        SbF53 g2 = sb_f53_zero();
        const sb_u8* cls = (const sb_u8*)p.cls;
        const int umax = lam < i - 1 ? lam : i - 1;
        // Ring slot of row k = i − 1 − u: one below the previous, wrapped by one addition.
        int slot = cur - 1;
        if (slot < 0) {
            slot += rows;
        }
        for (int u = 0; u <= umax; ++u) {
            const int k = i - 1 - u;
            const int xk = x[k];
            const SbF53* za_row = za + slot * m;
            const SbF53* zo_row = zo + slot * m * nc;
            const int vmax = lam - u < m - 2 - j ? lam - u : m - 2 - j;
            const int lmax = j + 1 + vmax;
            const sb_u8* lst = lists + xk * 200;
            const int cn = lists[SB_SPARSE_CNT + xk];
            for (int t = lists[SB_SPARSE_NXT + xk * 201 + j + 1]; t < cn; ++t) {
                const int l = lst[t];
                if (l > lmax) {
                    break;
                }
                const int v = l - j - 1;
                const int pk = sb_dup_pk(xk, y[l], np);
                if (v == 0) {
                    // A stack or a top bulge of size u.
                    const SbF53 first = za_row[l];
                    if (first.m != 0ull) {
                        const sb_u64 at =
                            u == 0 ? p.stack + (sb_u64)(pk * (int)np + pi)
                                   : p.bulge + (sb_u64)(((u - 1) * (int)np + pk) * (int)np + pi);
                        special = sb_f53_add(special, sb_f53_mul(first, sb_pf_w(w, at)));
                    }
                } else if (u == 0) {
                    // A bottom bulge of size v.
                    const SbF53 prev = za_row[l];
                    if (prev.m != 0ull) {
                        const sb_u64 at =
                            p.bulge + (sb_u64)(((lam + v - 1) * (int)np + pk) * (int)np + pi);
                        special = sb_f53_add(special, sb_f53_mul(prev, sb_pf_w(w, at)));
                    }
                } else if (u <= 2 && v <= 2) {
                    // The interior loops 1x1, 1x2, 2x1 and 2x2.
                    const SbF53 prev = za_row[l];
                    if (prev.m != 0ull) {
                        const int p1 = x[k + 1];
                        const int q1 = y[l - 1];
                        sb_u64 at;
                        if (u == 1 && v == 1) {
                            at = p.i11 + (sb_u64)(((pk * 4 + p1) * 4 + q1) * (int)np + pi);
                        } else if (u == 1) {
                            const int q2 = y[l - 2];
                            at = p.i12 +
                                 (sb_u64)((((pk * 4 + p1) * 4 + q1) * 4 + q2) * (int)np + pi);
                        } else if (v == 1) {
                            const int p2 = x[k + 2];
                            at = p.i21 +
                                 (sb_u64)((((pk * 4 + p1) * 4 + p2) * 4 + q1) * (int)np + pi);
                        } else {
                            const int p2 = x[k + 2];
                            const int q2 = y[l - 2];
                            at = p.i22 + (sb_u64)(((((pk * 4 + p1) * 4 + p2) * 4 + q1) * 4 + q2) *
                                                      (int)np +
                                                  pi);
                        }
                        special = sb_f53_add(special, sb_f53_mul(prev, sb_pf_w(w, at)));
                    }
                } else {
                    // Generic: ZO_c(k, l) (*) WGEN[u][v] into ZG_c, c = CLS[u][v].
                    const int shape = u * (lam + 1) + v;
                    const int cl = (int)__ldg(cls + shape);
                    const SbF53 prev = zo_row[l * nc + cl];
                    if (prev.m != 0ull) {
                        const SbF53 term = sb_f53_mul(prev, sb_pf_w(w, p.gen + (sb_u64)shape));
                        if (cl == 0) {
                            g0 = sb_f53_add(g0, term);
                        } else if (cl == 1) {
                            g1 = sb_f53_add(g1, term);
                        } else {
                            g2 = sb_f53_add(g2, term);
                        }
                    }
                }
            }
            slot -= 1;
            if (slot < 0) {
                slot += rows;
            }
        }
        // The WIN terms are the last terms of ZC, classes in grammar order, every class.
        const sb_u64 at_in = p.loop_in + (sb_u64)((x[i - 1] * 4 + y[j + 1]) * (int)np + pi);
        special = sb_f53_add(special, sb_f53_mul(g0, sb_pf_w(w, at_in)));
        if (nc > 1) {
            special = sb_f53_add(special, sb_f53_mul(g1, sb_pf_w(w, at_in + 16 * np)));
            special = sb_f53_add(special, sb_f53_mul(g2, sb_pf_w(w, at_in + 32 * np)));
        }
    }
    const int alpha = i >= 1 ? x[i - 1] : SB_PF_NONE;
    const int beta = j + 1 < m ? y[j + 1] : SB_PF_NONE;
    const sb_u64 at_start =
        p.start + (sb_u64)sigma * 25 * np + (sb_u64)((alpha * (int)np + pi) * 5 + beta);
    za_out = sb_f53_add(sb_pf_w(w, at_start), special);
    if (i + 1 < n && j >= 1) {
        const int tail = x[i + 1] * 4 + y[j - 1];
        for (int c = 0; c < nc; ++c) {
            const sb_u64 at_out = p.loop_out + (sb_u64)((c * (int)np + pi) * 16 + tail);
            zo_out[c] = sb_f53_mul(za_out, sb_pf_w(w, at_out));
        }
    } else {
        for (int c = 0; c < nc; ++c) {
            zo_out[c] = sb_f53_zero();
        }
    }
    if (special.m != 0ull) {
        const int alpha2 = i + 1 < n ? x[i + 1] : SB_PF_NONE;
        const int beta2 = j >= 1 ? y[j - 1] : SB_PF_NONE;
        const sb_u64 at_end = p.end + (sb_u64)((pi * 5 + alpha2) * 5 + beta2);
        end_out = sb_f53_mul(special, sb_pf_w(w, at_end));
    } else {
        end_out = sb_f53_zero();
    }
}

// The resident fast K3 DP at g = 32, tier p.tier (global: 128-thread blocks, band in the
// group's global slot; shared: one-warp blocks, band after the lists). Dynamic shared
// memory per warp: `p.stride` bytes, the codes (400), the list area, then in the shared
// tier the band. The band: ZA of R rows of m′ values, ZO of R rows of m′ cells of N_C
// values, then the m′ column sums E(j), R = min(n′, Λ + 2).
__device__ __forceinline__ void sb_pf_sparse_body(const SbTables& tb, const SbSpace& s,
                                                  const SbPf& p) {
    extern __shared__ __align__(16) sb_u8 sb_pf_sparse_smem[];
    const int me = (int)(threadIdx.x & 31u);
    const unsigned mask = 0xffffffffu;
    const sb_u64 per_block = blockDim.x / 32;
    const sb_u64 local = threadIdx.x / 32;
    const sb_u64 group = (sb_u64)blockIdx.x * per_block + local;
    const sb_u64 groups = (sb_u64)gridDim.x * per_block;
    sb_u8* mine = sb_pf_sparse_smem + local * p.stride;
    sb_u8* ca = mine;
    sb_u8* cb = mine + SB_PF_MAX_LENGTH;
    sb_u8* lists = mine + SB_PF_CODES;
    sb_u8* band = p.tier == SB_PF_SHARED ? mine + SB_PF_CODES + SB_SPARSE_LISTS
                                         : (sb_u8*)p.slots + group * p.slot;
    const SbF53* w = (const SbF53*)p.w;
    SbF53* out = (SbF53*)p.out;
    for (sb_u64 t = group; t < s.n; t += groups) {
        sb_u64 i, j, n64, m64;
        sb_decode(s, t, i, j);
        sb_dup_lengths(tb, i, j, n64, m64);
        if (n64 < 2 || m64 < 2 || sb_pf_group(p, n64, m64) != 32ull ||
            sb_pf_sparse_tier(p, n64, m64) != (int)p.tier || sb_pf_skipped(p, t)) {
            continue;  // the closed-form kernel's, the continuation's or the C2 bound's pair
        }
        const int na = (int)n64;
        const int nb = (int)m64;
        const SbSequence pa = sb_a_row(tb, i);
        const SbSequence pb = sb_b_row(tb, j);
        for (int q = me; q < na; q += 32) {
            ca[q] = sb_dup_code(pa[q]);
        }
        for (int q = me; q < nb; q += 32) {
            cb[q] = sb_dup_code(pb[q]);
        }
        __syncwarp(mask);
        // Canonical orientation (§6.2 step 2), uniform in the warp.
        const int shorter = na < nb ? na : nb;
        int diff = shorter;
        for (int q = 0; q < shorter; ++q) {
            if (ca[q] != cb[q]) {
                diff = q;
                break;
            }
        }
        const bool swap = diff == shorter ? na > nb : ca[diff] > cb[diff];
        const sb_u8* x = swap ? cb : ca;
        const sb_u8* y = swap ? ca : cb;
        const int n = swap ? nb : na;
        const int m = swap ? na : nb;
        // sigma: identical strands and a palindrome (K1's rule, independent of orientation).
        int sigma = 0;
        if (diff == shorter && na == nb) {
            sigma = 1;
            for (int q = 0; q < n; ++q) {
                if (x[q] + x[n - 1 - q] != 3) {
                    sigma = 0;
                    break;
                }
            }
        }
        const int rows = n < (int)p.lam + 2 ? n : (int)p.lam + 2;
        const int nc = (int)p.n_c;
        SbF53* za = (SbF53*)band;
        SbF53* zo = za + rows * m;
        SbF53* col = zo + rows * m * nc;
#ifdef SB_TESTING
        if (me == 0) sb_resident_poison_count(p.resident_stats, 0, 0, 1);
        if (p.poison < 256) {
            // Testing hook (§8 item 8): the pair's whole band (ZA, ZO and E) is filled with
            // the poison byte; every byte read is written for this pair first.
            const int total = ((1 + nc) * rows + 1) * m * (int)SB_PF_VALUE;
            for (int q = me; q < total; q += 32) {
                band[q] = (sb_u8)p.poison;
            }
            __syncwarp(mask);
            if (me == 0) {
                sb_resident_poison_count(p.resident_stats, 0, 1, 1);
                sb_resident_poison_count(p.resident_stats, 0, 2, (sb_u64)total);
            }
        }
#endif
        sb_sparse_lists(y, m, lists, me, p.n_p);
        // E(j) = 0 for every column before row 0.
        for (int c = me; c < m; c += 32) {
            col[c] = sb_f53_zero();
        }
        __syncwarp(mask);
        int cur = 0;
        for (int r = 0; r < n; ++r) {
            SbF53* za_row = za + cur * m;
            SbF53* zo_row = zo + cur * m * nc;
            const int xr = x[r];
            const sb_u8* row_list = lists + xr * 200;
            const int count = lists[SB_SPARSE_CNT + xr];
            for (int k = me; k < count; k += 32) {
                const int c = row_list[k];
                SbF53 za_value;
                SbF53 end_term;
                sb_pf_sparse_cell(p, w, x, y, lists, n, m, rows, za, zo, cur, r, c, sigma,
                                  za_value, zo_row + c * nc, end_term);
                // E(j) in ascending row order: only the lane holding (r, j) updates it,
                // after row r − 1's barrier.
                if (end_term.m != 0ull) {
                    col[c] = sb_f53_add(col[c], end_term);
                }
                za_row[c] = za_value;
            }
            __syncwarp(mask);
            cur += 1;
            if (cur == rows) {
                cur = 0;
            }
        }
        if (me == 0) {
            SbF53 total = sb_f53_zero();
            for (int c = 0; c < m; ++c) {
                total = sb_f53_add(total, col[c]);
            }
            out[t] = total;
        }
        // The next pair's staging, lists and band writes follow every read of this one.
        __syncwarp(mask);
    }
}

extern "C" __global__ void sb_pf_res(SbTables tb, SbSpace s, SbPf p) {
    sb_pf_sparse_body(tb, s, p);
}
