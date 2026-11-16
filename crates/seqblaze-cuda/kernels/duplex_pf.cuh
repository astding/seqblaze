// K3 native `duplex(model="pf")` on the device: Z-hat, the F53 partition function of one
// pair per sub-warp group over K1's structure space, its tier predicate and routing.
// Spec: docs/wiki/gpu/duplex-pf-cuda.md §4-§6 (the recurrence restated in §6.1 from
// docs/wiki/thermo/duplex-pf.md §6). Written from those pages only.
//
// Integers only (f53.cuh). Every fold is sequential in one lane in duplex-pf.md's fixed
// orders: ZC(i, j) over its predecessors in order O (u ascending, then v ascending; the
// special terms into one sum, the generic terms into ZG, ZG (*) WIN last), E(j) over
// rows in ascending order by the lane that owns column j, and Z-hat over the columns in
// ascending order by lane 0. No fold is split across lanes, so the column-to-lane
// partition, the group size, the tier, the ring size and the grid change only which lane
// evaluates which cell, never the operations applied to any value (§2).
#pragma once
#include "duplex.cuh"
#include "f53.cuh"
#include "hamming.cuh"

// Tiers of a pair (§6.4); must match `DupTier` in src/plan.rs.
#define SB_PF_CLOSED 0
#define SB_PF_SHARED 1
#define SB_PF_GLOBAL 2
#define SB_PF_HOST 3

#define SB_PF_CODES 400ull  // staged codes of a and b per group (q of §6.3)
#define SB_PF_MAX_LENGTH 200
#define SB_PF_NONE 4        // the empty neighbour in START and END
#define SB_PF_VALUE 16ull   // bytes of one F53 value (M, e)

// Mirrors `SbPf` in src/pf.rs; every field is 8 bytes wide. Table offsets index the
// concatenated weight entries (`w`) of the K3 device pack (§4).
struct SbPf {
    sb_u64 w;            // const SbF53*: every weight entry
    sb_u64 lam;
    sb_u64 start;        // START, sigma = 0 (sigma = 1 follows at +25·N_P)
    sb_u64 end;
    sb_u64 stack;
    sb_u64 bulge;
    sb_u64 i11;
    sb_u64 i12;
    sb_u64 i21;
    sb_u64 i22;
    sb_u64 gen;
    sb_u64 loop_out;
    sb_u64 loop_in;
    sb_u64 group_lanes;  // 0: g by max(n, m)
    sb_u64 band_bytes;
    sb_u64 pair_cells;   // the effective cap min(pair_cells, CAP_pf), candidate evaluations
    sb_u64 s_max;        // opt-in shared memory per block
    sb_u64 s_block;      // static shared memory per block
    sb_u64 tier;         // per launch: SB_PF_SHARED or SB_PF_GLOBAL
    sb_u64 stride;       // per launch: dynamic shared bytes per group
    sb_u64 slots;        // per launch (global tier): u8* of the band slots
    sb_u64 slot;         // per launch (global tier): bytes per slot
    sb_u64 out;          // SbF53*: Z-hat of every item of the chunk
    sb_u64 poison;       // testing: byte written over each band before its pair; 256: none
    // Pack format 2 (nndb2024-dna.md §5.4.5): N_P, N_C and the u8 CLS map over (Λ + 1)².
    sb_u64 n_p;
    sb_u64 n_c;
    sb_u64 cls;
    // C2 bounded resident stage (duplex-pf-bound.md §8): u8* skip flags of the launch's
    // items, or 0. A skipped item's Z-hat is (0, 0) from the closed-form kernel; the DP
    // and routing kernels skip it.
    sb_u64 skip;
#ifdef SB_TESTING
    sb_u64 resident_stats; // u64[2][3], as SbDup; 0 outside resident launches
#endif
};

// Whether the bounded stage resolved item t of the launch's space (SbPf::skip).
__device__ __forceinline__ bool sb_pf_skipped(const SbPf& p, sb_u64 t) {
    return p.skip != 0 && ((const sb_u8*)p.skip)[t] != 0;
}

// log2 of a group size (4, 8, 16 or 32): a shift count, so no runtime division.
__device__ __forceinline__ sb_u64 sb_pf_log2(sb_u64 g) { return (sb_u64)(__ffsll((long long)g) - 1); }

// g of a pair (§6.2): `group_lanes` if set, else by max(n, m).
__device__ __forceinline__ sb_u64 sb_pf_group(const SbPf& p, sb_u64 n, sb_u64 m) {
    if (p.group_lanes != 0) {
        return p.group_lanes;
    }
    const sb_u64 l = n > m ? n : m;
    return l <= 16 ? 4ull : (l <= 32 ? 8ull : (l <= 64 ? 16ull : 32ull));
}

// Band bytes of one orientation with `rows` rows and `cols` columns (§6.3,
// nndb2024-dna.md §5.4.5): ((1 + N_C)·R + 1)·cols·16 with R = min(rows, Λ + 2): the ZA ring,
// the ZO ring of N_C values per cell, then E.
__device__ __forceinline__ sb_u64 sb_pf_band(const SbPf& p, sb_u64 rows, sb_u64 cols) {
    const sb_u64 r = rows < p.lam + 2 ? rows : p.lam + 2;
    return ((1 + p.n_c) * r + 1) * cols * SB_PF_VALUE;
}

// The tier of a pair (§6.4); the host twin is `pf_tier` in src/plan.rs.
__device__ __forceinline__ int sb_pf_tier(const SbPf& p, sb_u64 n, sb_u64 m) {
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
    const bool fits = (32ull >> lg) * (band + SB_PF_CODES) + p.s_block <= p.s_max;
    return band <= p.band_bytes && fits ? SB_PF_SHARED : SB_PF_GLOBAL;
}

__device__ __forceinline__ SbF53 sb_pf_w(const SbF53* w, sb_u64 at) {
    SbF53 out;
    out.m = __ldg(&w[at].m);
    out.e = __ldg(&w[at].e);
    return out;
}

// ZA, the N_C values ZO_c and the end term of one pair cell (i, j) of §6.1 on pack format 2
// (nndb2024-dna.md §5.4.5). `x`, `y` are the canonical codes (n x m), `za` the ring of
// `rows` rows of m values, `zo` the ring of `rows` rows of m cells of N_C values (class c of
// (slot, column) at (slot·m + column)·N_C + c), `cur` the ring slot of row i; `zo_out`
// receives the cell's N_C values. Predecessors in order O; zero predecessors are skipped as
// on the CPU (QKP11); the generic terms of each class fold into their own ZG_c, and the
// terms ZG_c (*) WIN_c come last, c = 0 .. N_C - 1, a zero ZG_c included.
__device__ __forceinline__ void sb_pf_cell(const SbPf& p, const SbF53* w, const sb_u8* x,
                                           const sb_u8* y, int n, int m, int rows,
                                           const SbF53* za, const SbF53* zo, int cur, int i,
                                           int j, int sigma, SbF53& za_out, SbF53* zo_out,
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
        // Ring slot of row k = i - 1 - u: one below the previous, wrapped by one addition.
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
            // v = 0: a stack or a top bulge of size u. A nonzero ZA(k, l) is a pair.
            const SbF53 first = za_row[j + 1];
            if (first.m != 0ull) {
                const int pk = sb_dup_pk(xk, y[j + 1], np);
                const sb_u64 at =
                    u == 0 ? p.stack + (sb_u64)(pk * (int)np + pi)
                           : p.bulge + (sb_u64)(((u - 1) * (int)np + pk) * (int)np + pi);
                special = sb_f53_add(special, sb_f53_mul(first, sb_pf_w(w, at)));
            }
            for (int v = 1; v <= vmax; ++v) {
                const int l = j + 1 + v;
                if (u == 0) {
                    // A bottom bulge of size v.
                    const SbF53 prev = za_row[l];
                    if (prev.m != 0ull) {
                        const int pk = sb_dup_pk(xk, y[l], np);
                        const sb_u64 at =
                            p.bulge + (sb_u64)(((lam + v - 1) * (int)np + pk) * (int)np + pi);
                        special = sb_f53_add(special, sb_f53_mul(prev, sb_pf_w(w, at)));
                    }
                } else if (u <= 2 && v <= 2) {
                    // The interior loops 1x1, 1x2, 2x1 and 2x2.
                    const SbF53 prev = za_row[l];
                    if (prev.m != 0ull) {
                        const int pk = sb_dup_pk(xk, y[l], np);
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
                    // Generic: ZO_c(k, l) (*) WGEN[u][v] into ZG_c, c = CLS[u][v]; ZO formed
                    // first (one association).
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

// ------------------------------------------------------------------ kernels

// Closed-form pairs (§6.4 item 1) and items the C2 bound resolved: Z-hat = (0, 0).
// Written before the DP kernels; never routed.
extern "C" __global__ void sb_pf_closed(SbTables tb, SbSpace s, SbPf p) {
    SbF53* out = (SbF53*)p.out;
    const sb_u64 stride = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < s.n; t += stride) {
        sb_u64 i, j, n, m;
        sb_decode(s, t, i, j);
        sb_dup_lengths(tb, i, j, n, m);
        if (n < 2 || m < 2 || sb_pf_skipped(p, t)) {
            out[t] = sb_f53_zero();
        }
    }
}

// The DP of one variant (G, tier p.tier): one group of G lanes per pair, grid-striding
// over the chunk's items and skipping the pairs of other variants (§6.2). Dynamic shared
// memory: per group `p.stride` bytes, the codes of a and b (400 bytes) and, in the shared
// tier, the band. The band of a pair: ZA of R rows of m' values, ZO of R rows of m' cells of
// N_C values, then the m' column sums E(j), R = min(n', Λ + 2) (§6.3, nndb2024-dna.md
// §5.4.5).
template <int G>
__device__ __forceinline__ void sb_pf_body(const SbTables& tb, const SbSpace& s, const SbPf& p) {
    extern __shared__ __align__(16) sb_u8 sb_pf_smem[];
    const unsigned lane = threadIdx.x & 31u;
    const int me = (int)(threadIdx.x & (unsigned)(G - 1));
    const unsigned mask =
        G == 32 ? 0xffffffffu : (((1u << G) - 1u) << (lane & ~(unsigned)(G - 1)));
    const sb_u64 per_block = blockDim.x / G;
    const sb_u64 local = threadIdx.x / G;
    const sb_u64 group = (sb_u64)blockIdx.x * per_block + local;
    const sb_u64 groups = (sb_u64)gridDim.x * per_block;
    sb_u8* mine = sb_pf_smem + local * p.stride;
    sb_u8* ca = mine;
    sb_u8* cb = mine + SB_PF_MAX_LENGTH;
    sb_u8* band = p.tier == SB_PF_SHARED ? mine + SB_PF_CODES
                                         : (sb_u8*)p.slots + group * p.slot;
    const SbF53* w = (const SbF53*)p.w;
    SbF53* out = (SbF53*)p.out;
    for (sb_u64 t = group; t < s.n; t += groups) {
        sb_u64 i, j, n64, m64;
        sb_decode(s, t, i, j);
        sb_dup_lengths(tb, i, j, n64, m64);
        if (n64 < 2 || m64 < 2 || sb_pf_group(p, n64, m64) != (sb_u64)G ||
            sb_pf_tier(p, n64, m64) != (int)p.tier || sb_pf_skipped(p, t)) {
            continue;  // the closed-form kernel's, another variant's or the host's pair
        }
        const int na = (int)n64;
        const int nb = (int)m64;
        const SbSequence pa = sb_a_row(tb, i);
        const SbSequence pb = sb_b_row(tb, j);
        for (int q = me; q < na; q += G) {
            ca[q] = sb_dup_code(pa[q]);
        }
        for (int q = me; q < nb; q += G) {
            cb[q] = sb_dup_code(pb[q]);
        }
        __syncwarp(mask);
        // Canonical orientation (§6.2 step 2): every lane finds the first differing
        // position from the staged codes, so the decision is uniform in the group.
        // a <= b iff no position differs and n <= m, or a_p < b_p (a proper prefix is
        // smaller); the codes keep the byte order of validated input.
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
            // Testing hook (§8 item 8, QKP13): the pair's whole band (ZA, ZO and E) is
            // filled with the poison byte; every byte read is written for this pair first.
            // Release builds compile no fill.
            const int total = ((1 + nc) * rows + 1) * m * (int)SB_PF_VALUE;
            for (int q = me; q < total; q += G) {
                band[q] = (sb_u8)p.poison;
            }
            __syncwarp(mask);
            if (me == 0) {
                sb_resident_poison_count(p.resident_stats, 0, 1, 1);
                sb_resident_poison_count(p.resident_stats, 0, 2, (sb_u64)total);
            }
        }
#endif
        for (int c = me; c < m; c += G) {
            col[c] = sb_f53_zero();
        }
        int cur = 0;
        for (int r = 0; r < n; ++r) {
            SbF53* za_row = za + cur * m;
            SbF53* zo_row = zo + cur * m * nc;
            const int xr = x[r];
            for (int c = me; c < m; c += G) {
                SbF53 za_value = sb_f53_zero();
                if (sb_dup_pt(xr, y[c], p.n_p) >= 0) {
                    SbF53 end_term;
                    sb_pf_cell(p, w, x, y, n, m, rows, za, zo, cur, r, c, sigma, za_value,
                               zo_row + c * nc, end_term);
                    if (end_term.m != 0ull) {
                        col[c] = sb_f53_add(col[c], end_term);
                    }
                } else {
                    for (int q = 0; q < nc; ++q) {
                        zo_row[c * nc + q] = sb_f53_zero();
                    }
                }
                za_row[c] = za_value;
            }
            // Row r's values are written before any lane reads them, and the slot of
            // row r + 1 - R is overwritten only after every read of row r (§6.2).
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
        // The next pair's staging and band writes follow every read of this one.
        __syncwarp(mask);
    }
}

#define SB_PF_KERNEL(G)                                                                   \
    extern "C" __global__ void sb_pf_g##G(SbTables tb, SbSpace s, SbPf p) {               \
        sb_pf_body<G>(tb, s, p);                                                          \
    }
SB_PF_KERNEL(4)
SB_PF_KERNEL(8)
SB_PF_KERNEL(16)
SB_PF_KERNEL(32)

__device__ __forceinline__ bool sb_pf_routed(const SbTables& tb, const SbSpace& s, const SbPf& p,
                                             sb_u64 t) {
    if (t >= s.n) {
        return false;
    }
    sb_u64 i, j, n, m;
    sb_decode(s, t, i, j);
    sb_dup_lengths(tb, i, j, n, m);
    return sb_pf_tier(p, n, m) == SB_PF_HOST && !sb_pf_skipped(p, t);
}

// Routing step 1 (§6.4, the form of G1's Hits): routed count of each block's contiguous
// tile of blockDim.x items.
extern "C" __global__ void sb_pf_route_count(SbTables tb, SbSpace s, SbPf p,
                                             sb_u64* block_counts) {
    __shared__ unsigned warp_counts[32];
    const sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x;
    const bool routed = sb_pf_routed(tb, s, p, t);
    const unsigned ballot = __ballot_sync(0xffffffffu, routed);
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_counts[warp] = (unsigned)__popc(ballot);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        sb_u64 total = 0;
        for (unsigned v = 0; v < blockDim.x / 32u; ++v) {
            total += warp_counts[v];
        }
        block_counts[blockIdx.x] = total;
    }
}

// Routing step 3: chunk positions of the routed items, in item order, at the block's
// exclusive offset plus the in-block rank.
extern "C" __global__ void sb_pf_route_write(SbTables tb, SbSpace s, SbPf p,
                                             const sb_u64* block_offsets, sb_u64* out) {
    __shared__ unsigned warp_base[32];
    const sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x;
    const bool routed = sb_pf_routed(tb, s, p, t);
    const unsigned ballot = __ballot_sync(0xffffffffu, routed);
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_base[warp] = (unsigned)__popc(ballot);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned running = 0;
        for (unsigned v = 0; v < blockDim.x / 32u; ++v) {
            const unsigned c = warp_base[v];
            warp_base[v] = running;
            running += c;
        }
    }
    __syncthreads();
    if (routed) {
        const unsigned below = ballot & ((1u << lane) - 1u);
        out[block_offsets[blockIdx.x] + warp_base[warp] + (unsigned)__popc(below)] = t;
    }
}
