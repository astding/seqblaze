// Align kernel family for the seqblaze fatbin: the engine key of affine-gap
// alignment (native local and global Gotoh, compat water and needle).
// Spec: docs/wiki/gpu/align-cuda.md §4-§5 (score definitions, integer range,
// carry form, tiers) and docs/wiki/align/gotoh-align.md §2, §5.5, §5.6. Written
// from those pages only.
//
// Integers only: int32 cell arithmetic with -infinity stored as SB_ALN_NEG =
// -3 * 2^29 (§5.2), no floating point and no atomics. Every value is an exact
// integer of (a, b, scoring, mode): the key is -score_fx, the largest element of
// a finite set of exact integers, and integer max is associative and
// commutative, so the thread/warp mapping, strip height, grid and chunk sizes
// only change which unit evaluates which cell. The device implements no tie
// rule; every detail field comes from the host.
#pragma once
#include "hamming.cuh"

// Kinds; must match `KIND_*` in src/aln.rs.
#define SB_ALN_LOCAL 0
#define SB_ALN_GLOBAL 1
#define SB_ALN_WATER 2
#define SB_ALN_NEEDLE 3

// Tiers of a pair (§5.6); must match `AlnTier` in src/plan.rs.
#define SB_ALN_CLOSED 0
#define SB_ALN_THREAD 1
#define SB_ALN_WARP 2
#define SB_ALN_HOST 3

#define SB_ALN_NEG (-1610612736)            // -3 * 2^29: stored -infinity (§5.2)
#define SB_ALN_SENTINEL_KEY 2147483646      // water's -FLT_MAX sentinel (INT32_MAX - 1)
#define SB_ALN_THREAD_ROWS 32               // compiled limit of thread_rows
#define SB_ALN_SHARED_CLASSES 64            // class tables up to 64 x 64 go to shared memory

// Mirrors `SbAln` in src/aln.rs; every field is 8 bytes wide.
struct SbAln {
    sb_u64 kind;
    sb_u64 end_gaps;     // global, needle: 1 when end gaps are penalised
    sb_u64 go;           // fixed point, [0, 2^30]
    sb_u64 ge;
    sb_u64 mm;           // 1: s = match if the bytes are equal, else mismatch
    sb_i64 match;
    sb_i64 mismatch;
    sb_u64 n_classes;    // c: s(x, y) = table[cls[x] * c + cls[y]]
    sb_u64 cls;          // const u8[256]
    sb_u64 table;        // const int[c * c]
    sb_u64 thread_rows;  // largest m in the thread tier (0: none)
    sb_u64 lane_rows;    // largest rows per warp lane: 1, 2, 4 or 8
    sb_u64 pair_cells;   // largest serial depth of a device pair, in cell steps
    sb_u64 slots;        // int*: per warp, 2 * slot_cols int32 (multi-strip pairs)
    sb_u64 slot_cols;    // columns (n + 1) of one warp's slot
};

// Closed form below this length on either side: native 1, water 2, needle 0.
__device__ __forceinline__ sb_u64 sb_aln_closed_below(const SbAln& p) {
    return p.kind == SB_ALN_WATER ? 2ull : (p.kind == SB_ALN_NEEDLE ? 0ull : 1ull);
}

// q = min(lane_rows, Q(m)), Q(m) the smallest of 1, 2, 4, 8 with 32Q >= m.
__device__ __forceinline__ sb_u64 sb_aln_q(const SbAln& p, sb_u64 m) {
    const sb_u64 big = m <= 32 ? 1ull : m <= 64 ? 2ull : m <= 128 ? 4ull : 8ull;
    return big < p.lane_rows ? big : p.lane_rows;
}

// The tier predicate (§5.6, QAG3): closed form; else the thread tier when
// thread_rows > 0, m <= thread_rows and m n <= pair_cells; else the warp tier
// when P (n + 31) q <= pair_cells, P = ceil(m / 32q); else the host. The host
// twin is `aln_tier` in src/plan.rs.
__device__ __forceinline__ int sb_aln_tier(const SbAln& p, sb_u64 m, sb_u64 n) {
    const sb_u64 low = sb_aln_closed_below(p);
    if (m < low || n < low) {
        return SB_ALN_CLOSED;
    }
    if (p.thread_rows > 0 && m <= p.thread_rows && m * n <= p.pair_cells) {
        return SB_ALN_THREAD;
    }
    const sb_u64 q = sb_aln_q(p, m);
    const sb_u64 strips = (m + 32 * q - 1) / (32 * q);
    return strips * q * (n + 31) <= p.pair_cells ? SB_ALN_WARP : SB_ALN_HOST;
}

// lambda(L) = go + (L - 1) ge for L >= 1 with end gaps, else 0 (§5.1). Under
// the value-range preflight it lies in [0, 2^30] for every L <= max(m, n).
__device__ __forceinline__ int sb_aln_lam(const SbAln& p, sb_u64 len) {
    if (!p.end_gaps || len == 0) {
        return 0;
    }
    return (int)((sb_i64)p.go + (sb_i64)(len - 1) * (sb_i64)p.ge);
}

__device__ __forceinline__ int sb_aln_max(int x, int y) {
    return x > y ? x : y;
}

// Substitution scores (§4, QAG5): byte equality for match/mismatch, else the
// class table, staged into shared memory when c <= 64.
struct SbAlnSub {
    const int* tab;
    const sb_u8* cls;
    int c;
    int mm;
    int match;
    int mismatch;
};

// Every thread of the block calls this before its first decode.
__device__ __forceinline__ SbAlnSub sb_aln_sub(const SbAln& p, int* s_tab, sb_u8* s_cls) {
    SbAlnSub sub;
    sub.c = (int)p.n_classes;
    sub.mm = p.mm != 0;
    sub.match = (int)p.match;
    sub.mismatch = (int)p.mismatch;
    sub.tab = (const int*)p.table;
    sub.cls = (const sb_u8*)p.cls;
    if (!sub.mm) {
        for (unsigned k = threadIdx.x; k < 256u; k += blockDim.x) {
            s_cls[k] = sub.cls[k];
        }
        if (p.n_classes <= SB_ALN_SHARED_CLASSES) {
            const unsigned cells = (unsigned)(p.n_classes * p.n_classes);
            for (unsigned k = threadIdx.x; k < cells; k += blockDim.x) {
                s_tab[k] = sub.tab[k];
            }
            sub.tab = s_tab;
        }
        sub.cls = s_cls;
    }
    __syncthreads();
    return sub;
}

// Carry form (§5.3). Native: the vertical carry of row i - 1 at column j is
// (x1, x2) = (U, P) = (max(V, Q), P); the horizontal carry of row i at column
// j - 1 is (h1, h2) = (W, Q) = (max(V, P), Q); dg = D(i - 1, j - 1) =
// max(U, P)(i - 1, j - 1), and max(0, U, P) in local mode. Compat: (x1, x2) =
// (H, F) of row i - 1, (h1, h2) = (H, E) of row i, dg = H(i - 1, j - 1).
// One cell updates (h1, h2) and dg in place and turns (x1, x2) into the
// vertical carry for row i + 1; `fold` gets V (native), R (water) or M
// (needle) of the cell.
template <int KIND>
__device__ __forceinline__ void sb_aln_cell(int s, int go, int ge, int& x1, int& x2, int& h1,
                                            int& h2, int& dg, int& fold) {
    if (KIND == SB_ALN_LOCAL || KIND == SB_ALN_GLOBAL) {
        const int v = s + dg;
        int pp = sb_aln_max(x1 - go, x2 - ge);
        int qq = sb_aln_max(h1 - go, h2 - ge);
        if (KIND == SB_ALN_LOCAL) {
            // -infinity occurs at P(1, .) and Q(., 1) in local mode only (§5.2 R3).
            pp = sb_aln_max(pp, SB_ALN_NEG);
            qq = sb_aln_max(qq, SB_ALN_NEG);
            dg = sb_aln_max(0, sb_aln_max(x1, x2));
        } else {
            dg = sb_aln_max(x1, x2);
        }
        h1 = sb_aln_max(v, pp);
        h2 = qq;
        x1 = sb_aln_max(v, qq);
        x2 = pp;
        fold = v;
    } else {
        const int e = sb_aln_max(h1 - go, h2 - ge);
        const int f = sb_aln_max(x1 - go, x2 - ge);
        const int mm = dg + s;
        const int r = sb_aln_max(mm, sb_aln_max(e, f));
        const int h = KIND == SB_ALN_WATER ? sb_aln_max(r, 0) : r;
        dg = x1;
        h1 = h;
        h2 = e;
        x1 = h;
        x2 = f;
        fold = KIND == SB_ALN_WATER ? r : mm;
    }
}

// The fold's start: the closed candidates of §5.1 (local 0; global and needle
// -lambda(m) - lambda(n), the all-gap alignment and needle's H(m, 0) and
// H(0, n) candidates; water none).
template <int KIND>
__device__ __forceinline__ int sb_aln_best0(const SbAln& p, sb_u64 m, sb_u64 n) {
    if (KIND == SB_ALN_LOCAL) {
        return 0;
    }
    if (KIND == SB_ALN_WATER) {
        return -2147483647 - 1;
    }
    return -sb_aln_lam(p, m) - sb_aln_lam(p, n);
}

// Row 0's vertical carry at column j >= 1 (§5.3 boundary table).
template <int KIND>
__device__ __forceinline__ void sb_aln_top(const SbAln& p, sb_u64 j, int& x1, int& x2) {
    x2 = SB_ALN_NEG;
    if (KIND == SB_ALN_LOCAL) {
        x1 = SB_ALN_NEG;
    } else if (KIND == SB_ALN_WATER) {
        x1 = 0;
    } else {
        x1 = -sb_aln_lam(p, j);
    }
}

// Row r >= 1 before column 1: its horizontal carry at column 0 and the
// diagonal input D(r - 1, 0) (H(r - 1, 0) in compat mode).
template <int KIND>
__device__ __forceinline__ void sb_aln_left(const SbAln& p, sb_u64 r, int& h1, int& h2, int& dg) {
    h2 = SB_ALN_NEG;
    if (KIND == SB_ALN_LOCAL) {
        h1 = SB_ALN_NEG;
        dg = 0;
    } else if (KIND == SB_ALN_WATER) {
        h1 = 0;
        dg = 0;
    } else {
        h1 = -sb_aln_lam(p, r);
        dg = -sb_aln_lam(p, r - 1);
    }
}

__device__ __forceinline__ void sb_aln_lengths(const SbTables& tb, sb_u64 i, sb_u64 j, sb_u64& m,
                                               sb_u64& n) {
    m = (sb_u64)sb_a_row(tb, i).length;
    n = (sb_u64)sb_b_row(tb, j).length;
}

// Closed-form pairs (§5.1): native local 0; native global -lambda(max(m, n));
// water's sentinel. Written by one kernel per chunk, so no tier kernel owns
// them and they are never routed.
extern "C" __global__ void sb_aln_closed(SbTables tb, SbSpace s, SbAln p, int* dist) {
    const sb_u64 stride = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < s.n; t += stride) {
        sb_u64 i, j, m, n;
        sb_decode(s, t, i, j);
        sb_aln_lengths(tb, i, j, m, n);
        if (sb_aln_tier(p, m, n) != SB_ALN_CLOSED) {
            continue;
        }
        int key = 0;
        if (p.kind == SB_ALN_WATER) {
            key = SB_ALN_SENTINEL_KEY;
        } else if (p.kind == SB_ALN_GLOBAL) {
            key = sb_aln_lam(p, m > n ? m : n);
        }
        dist[t] = key;
    }
}

// Substitution lookup with the mm / class-table choice fixed at compile time:
// both tiers resolve it once per launch, outside the pair loop. Row code of a
// byte of a: the class times c, or the byte; column code of a byte of b: the
// class, or the byte.
template <bool MM>
__device__ __forceinline__ int sb_aln_row_code_t(const SbAlnSub& sub, unsigned x) {
    return MM ? (int)x : (int)sub.cls[x] * sub.c;
}
template <bool MM>
__device__ __forceinline__ int sb_aln_col_code_t(const SbAlnSub& sub, unsigned y) {
    return MM ? (int)y : (int)sub.cls[y];
}
template <bool MM>
__device__ __forceinline__ int sb_aln_s_t(const SbAlnSub& sub, int row, int col) {
    return MM ? (row == col ? sub.match : sub.mismatch) : sub.tab[row + col];
}

// One cell of a thread-tier row sweep: update row x's carries and fold the
// per-cell candidates of local (max V) and water (R at i, j >= 2). The
// end-gap candidates of global and needle are folded by the caller: row m's
// W (global) or M (needle) at columns j < n is returned in `tail`, and column
// n is a separate sweep (`LAST`), so lambda is evaluated once per column or
// once per row of column n, never per cell.
template <int KIND, bool MM, bool LAST>
__device__ __forceinline__ void sb_aln_thread_cell(const SbAln& p, const SbAlnSub& sub, int code,
                                                   int cb, int go, int ge, int x, int rows,
                                                   sb_u64 col, sb_u64 m, sb_u64 n, int& x1,
                                                   int& x2, int& h1, int& h2, int& dg, int& tail,
                                                   int& best) {
    int fold;
    sb_aln_cell<KIND>(sb_aln_s_t<MM>(sub, code, cb), go, ge, x1, x2, h1, h2, dg, fold);
    if (KIND == SB_ALN_LOCAL) {
        best = sb_aln_max(best, fold);
    } else if (KIND == SB_ALN_WATER) {
        if (x >= 1 && col >= 2) {
            best = sb_aln_max(best, fold);
        }
    } else if (!LAST) {
        if (x + 1 == rows) {
            tail = KIND == SB_ALN_GLOBAL ? h1 : fold;
        }
    } else if (KIND == SB_ALN_GLOBAL) {
        // V(m, n); T_I(m - r) = U(r, n) - lambda(m - r) for r < m.
        best = sb_aln_max(best, x + 1 == rows ? fold : x1 - sb_aln_lam(p, m - (sb_u64)x - 1));
    } else {
        // M(m, n); M(i, n) - trail(m - i) for i < m when n >= 2.
        if (x + 1 == rows) {
            best = sb_aln_max(best, fold);
        } else if (n >= 2) {
            best = sb_aln_max(best, fold - sb_aln_lam(p, m - (sb_u64)x - 1));
        }
    }
}

// The score of one thread-tier pair (m <= SB_ALN_THREAD_ROWS). The
// horizontal carries, diagonals and row codes of every row stay in registers;
// the thread sweeps j = 1..n and, within a column, rows in increasing order,
// passing the vertical carry down.
template <int KIND, bool MM>
__device__ __forceinline__ int sb_aln_thread_pair(const SbAln& p, const SbAlnSub& sub,
                                                  SbSequence pa, SbSequence pb, sb_u64 m,
                                                  sb_u64 n) {
    const int go = (int)p.go;
    const int ge = (int)p.ge;
    const int rows = (int)m;
    int code[SB_ALN_THREAD_ROWS], h1[SB_ALN_THREAD_ROWS], h2[SB_ALN_THREAD_ROWS],
        dg[SB_ALN_THREAD_ROWS];
#pragma unroll
    for (int x = 0; x < SB_ALN_THREAD_ROWS; ++x) {
        if (x >= rows) {
            break;
        }
        code[x] = sb_aln_row_code_t<MM>(sub, pa[x]);
        sb_aln_left<KIND>(p, (sb_u64)x + 1, h1[x], h2[x], dg[x]);
    }
    int best = sb_aln_best0<KIND>(p, m, n);
    // Columns 1..n-1: row m's T_D(n - j) = W(m, j) - lambda(n - j) (global),
    // or M(m, j) - trail(n - j) when m >= 2 (needle).
    const bool tails = KIND == SB_ALN_GLOBAL || (KIND == SB_ALN_NEEDLE && m >= 2);
    for (sb_u64 col = 1; col < n; ++col) {
        const int cb = sb_aln_col_code_t<MM>(sub, pb[col - 1]);
        int x1, x2;
        sb_aln_top<KIND>(p, col, x1, x2);
        int tail = 0;
#pragma unroll
        for (int x = 0; x < SB_ALN_THREAD_ROWS; ++x) {
            if (x >= rows) {
                break;
            }
            sb_aln_thread_cell<KIND, MM, false>(p, sub, code[x], cb, go, ge, x, rows, col, m, n,
                                                x1, x2, h1[x], h2[x], dg[x], tail, best);
        }
        if (tails && rows >= 1) {
            best = sb_aln_max(best, tail - sb_aln_lam(p, n - col));
        }
    }
    if (n >= 1) {
        const int cb = sb_aln_col_code_t<MM>(sub, pb[n - 1]);
        int x1, x2;
        sb_aln_top<KIND>(p, n, x1, x2);
        int tail = 0;
#pragma unroll
        for (int x = 0; x < SB_ALN_THREAD_ROWS; ++x) {
            if (x >= rows) {
                break;
            }
            sb_aln_thread_cell<KIND, MM, true>(p, sub, code[x], cb, go, ge, x, rows, n, m, n, x1,
                                               x2, h1[x], h2[x], dg[x], tail, best);
        }
    }
    return best;
}

// Thread tier (§5.4): one thread per pair with m <= thread_rows <= 32,
// grid-stride over the chunk's items. The mm / class-table choice is uniform
// over the launch and taken once, outside the pair loop.
template <int KIND, bool MM>
__device__ __forceinline__ void sb_aln_thread_loop(const SbTables& tb, const SbSpace& s,
                                                   const SbAln& p, const SbAlnSub& sub,
                                                   int* dist) {
    const sb_u64 stride = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < s.n; t += stride) {
        sb_u64 i, j, m, n;
        sb_decode(s, t, i, j);
        sb_aln_lengths(tb, i, j, m, n);
        if (sb_aln_tier(p, m, n) != SB_ALN_THREAD) {
            continue;
        }
        const SbSequence pa = sb_a_row(tb, i);
        const SbSequence pb = sb_b_row(tb, j);
        dist[t] = -sb_aln_thread_pair<KIND, MM>(p, sub, pa, pb, m, n);
    }
}

template <int KIND>
__device__ __forceinline__ void sb_aln_thread_body(const SbTables& tb, const SbSpace& s,
                                                   const SbAln& p, int* dist) {
    __shared__ int s_tab[SB_ALN_SHARED_CLASSES * SB_ALN_SHARED_CLASSES];
    __shared__ sb_u8 s_cls[256];
    const SbAlnSub sub = sb_aln_sub(p, s_tab, s_cls);
    if (sub.mm) {
        sb_aln_thread_loop<KIND, true>(tb, s, p, sub, dist);
    } else {
        sb_aln_thread_loop<KIND, false>(tb, s, p, sub, dist);
    }
}

#define SB_ALN_DEF_THREAD(NAME, KIND)                                                        \
    extern "C" __global__ void sb_aln_thread_##NAME(SbTables tb, SbSpace s, SbAln p, int* dist) { \
        sb_aln_thread_body<KIND>(tb, s, p, dist);                                            \
    }
SB_ALN_DEF_THREAD(local, SB_ALN_LOCAL)
SB_ALN_DEF_THREAD(global, SB_ALN_GLOBAL)
SB_ALN_DEF_THREAD(water, SB_ALN_WATER)
SB_ALN_DEF_THREAD(needle, SB_ALN_NEEDLE)

// One cell of a warp-tier row sweep, the warp twin of `sb_aln_thread_cell`:
// update row `row` (0-based, row + 1 = r) at column `col` and fold the per-cell
// candidates of local (max V) and water (R at i, j >= 2). The end-gap
// candidates of global and needle are folded by the caller, as in the thread
// tier: row m's W (global) or M (needle) at columns j < n is returned in
// `tail`, and column n is a separate sweep (`LAST`), so lambda is evaluated
// once per column (by the lane that owns row m) or once per row of column n,
// never per cell.
template <int KIND, bool MM, bool LAST>
__device__ __forceinline__ void sb_aln_warp_cell(const SbAln& p, const SbAlnSub& sub, int code,
                                                 int cb, int go, int ge, sb_u64 row, sb_u64 col,
                                                 sb_u64 m, sb_u64 n, int& x1, int& x2, int& h1,
                                                 int& h2, int& dg, int& tail, int& best) {
    int fold;
    sb_aln_cell<KIND>(sb_aln_s_t<MM>(sub, code, cb), go, ge, x1, x2, h1, h2, dg, fold);
    if (KIND == SB_ALN_LOCAL) {
        best = sb_aln_max(best, fold);
    } else if (KIND == SB_ALN_WATER) {
        if (row >= 1 && col >= 2) {
            best = sb_aln_max(best, fold);
        }
    } else if (!LAST) {
        if (row + 1 == m) {
            tail = KIND == SB_ALN_GLOBAL ? h1 : fold;
        }
    } else if (KIND == SB_ALN_GLOBAL) {
        // V(m, n); T_I(m - r) = U(r, n) - lambda(m - r) for r < m.
        best = sb_aln_max(best, row + 1 == m ? fold : x1 - sb_aln_lam(p, m - row - 1));
    } else {
        // M(m, n); M(i, n) - trail(m - i) for i < m when n >= 2.
        if (row + 1 == m) {
            best = sb_aln_max(best, fold);
        } else if (n >= 2) {
            best = sb_aln_max(best, fold - sb_aln_lam(p, m - row - 1));
        }
    }
}

// Warp tier (§5.5): one warp per pair whose q = min(lane_rows, Q(m)) equals Q,
// as a systolic pipeline over strips of 32Q rows. In strip s lane l owns rows
// s 32Q + l Q + [0, Q) (within the strip), L_s lanes own a row, and at step
// t = 1..n + L_s - 1 lane l processes column j = t - l when 1 <= j <= n. Its
// first row's vertical carry is lane l - 1's last-row carry for column j from
// step t - 1 (__shfl_up_sync), or for lane 0 row 0's closed form (strip 0) or
// the boundary row strip s - 1 left in the warp's slot. A strip that is not
// the last writes its last row's vertical carry, column by column, into the
// slot; lane 0 of the same strip read that column at an earlier step, and the
// value it read feeds the shuffle every later step waits on. __syncwarp
// separates strips and pairs. Each lane folds its own cells; a warp max
// reduction gives the score. Control flow outside the cell body is
// warp-uniform except the column-n branch, which holds no shuffle. The
// substitution form (MM) is a template parameter and the end-gap candidates
// are folded outside the cell, as in the thread tier (align-cuda.md §13).
template <int KIND, int Q, bool MM>
__device__ __forceinline__ void sb_aln_warp_loop(const SbTables& tb, const SbSpace& s,
                                                 const SbAln& p, const SbAlnSub& sub,
                                                 int* dist) {
    const int lane = (int)(threadIdx.x & 31u);
    const sb_u64 per_block = blockDim.x / 32u;
    const sb_u64 warp = (sb_u64)blockIdx.x * per_block + (threadIdx.x >> 5);
    const sb_u64 n_warps = (sb_u64)gridDim.x * per_block;
    int* slot = p.slots != 0 ? (int*)p.slots + warp * 2 * p.slot_cols : 0;
    const int go = (int)p.go;
    const int ge = (int)p.ge;
    const sb_u64 height = 32ull * Q;
    for (sb_u64 t = warp; t < s.n; t += n_warps) {
        sb_u64 i, j, m, n;
        sb_decode(s, t, i, j);
        sb_aln_lengths(tb, i, j, m, n);
        if (sb_aln_tier(p, m, n) != SB_ALN_WARP || sb_aln_q(p, m) != (sb_u64)Q) {
            continue;  // another kernel's pair
        }
        const SbSequence pa = sb_a_row(tb, i);
        const SbSequence pb = sb_b_row(tb, j);
        const sb_u64 strips = (m + height - 1) / height;
        // Row m's T_D(n - j) = W(m, j) - lambda(n - j) (global), or
        // M(m, j) - trail(n - j) when m >= 2 (needle), for columns j < n.
        const bool tails = KIND == SB_ALN_GLOBAL || (KIND == SB_ALN_NEEDLE && m >= 2);
        int best = sb_aln_best0<KIND>(p, m, n);
        for (sb_u64 strip = 0; strip < strips; ++strip) {
            const sb_u64 row0 = strip * height;  // rows row0 + 1 .. are this strip's
            const sb_u64 in_strip = m - row0 < height ? m - row0 : height;
            const int lanes = (int)((in_strip + Q - 1) / Q);
            const sb_i64 left = (sb_i64)in_strip - (sb_i64)lane * Q;
            const int own = left <= 0 ? 0 : (left < Q ? (int)left : Q);
            const sb_u64 first = row0 + (sb_u64)lane * Q;  // 0-based index of my first row
            const bool last_strip = strip + 1 == strips;
            // Whether this lane owns row m (last strip only).
            const bool has_m = own > 0 && first < m && m <= first + (sb_u64)own;
            int code[Q], h1[Q], h2[Q], dg[Q];
#pragma unroll
            for (int x = 0; x < Q; ++x) {
                code[x] = 0;
                h1[x] = 0;
                h2[x] = 0;
                dg[x] = 0;
                if (x < own) {
                    code[x] = sb_aln_row_code_t<MM>(sub, pa[first + x]);
                    sb_aln_left<KIND>(p, first + x + 1, h1[x], h2[x], dg[x]);
                }
            }
            int o1 = 0, o2 = 0;
            const sb_u64 steps = n + (sb_u64)lanes - 1;
            for (sb_u64 step = 1; step <= steps; ++step) {
                const int in1 = __shfl_up_sync(0xffffffffu, o1, 1);
                const int in2 = __shfl_up_sync(0xffffffffu, o2, 1);
                const sb_i64 col = (sb_i64)step - lane;
                if (own > 0 && col >= 1 && col <= (sb_i64)n) {
                    int x1 = in1, x2 = in2;
                    if (lane == 0) {
                        if (strip == 0) {
                            sb_aln_top<KIND>(p, (sb_u64)col, x1, x2);
                        } else {
                            x1 = slot[2 * col];
                            x2 = slot[2 * col + 1];
                        }
                    }
                    const int cb = sb_aln_col_code_t<MM>(sub, pb[col - 1]);
                    int tail = 0;
                    if (col < (sb_i64)n) {
#pragma unroll
                        for (int x = 0; x < Q; ++x) {
                            if (x < own) {
                                sb_aln_warp_cell<KIND, MM, false>(
                                    p, sub, code[x], cb, go, ge, first + x, (sb_u64)col, m, n, x1,
                                    x2, h1[x], h2[x], dg[x], tail, best);
                            }
                        }
                        if (tails && has_m) {
                            best = sb_aln_max(best, tail - sb_aln_lam(p, n - (sb_u64)col));
                        }
                    } else {
#pragma unroll
                        for (int x = 0; x < Q; ++x) {
                            if (x < own) {
                                sb_aln_warp_cell<KIND, MM, true>(
                                    p, sub, code[x], cb, go, ge, first + x, (sb_u64)col, m, n, x1,
                                    x2, h1[x], h2[x], dg[x], tail, best);
                            }
                        }
                    }
                    o1 = x1;
                    o2 = x2;
                    if (!last_strip && lane == lanes - 1) {
                        slot[2 * col] = o1;
                        slot[2 * col + 1] = o2;
                    }
                }
            }
            // The slot's writes reach the next strip's lane 0.
            __syncwarp();
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            best = sb_aln_max(best, __shfl_xor_sync(0xffffffffu, best, offset));
        }
        if (lane == 0) {
            dist[t] = -best;
        }
        __syncwarp();
    }
}

template <int KIND, int Q>
__device__ __forceinline__ void sb_aln_warp_body(const SbTables& tb, const SbSpace& s,
                                                 const SbAln& p, int* dist) {
    __shared__ int s_tab[SB_ALN_SHARED_CLASSES * SB_ALN_SHARED_CLASSES];
    __shared__ sb_u8 s_cls[256];
    const SbAlnSub sub = sb_aln_sub(p, s_tab, s_cls);
    if (sub.mm) {
        sb_aln_warp_loop<KIND, Q, true>(tb, s, p, sub, dist);
    } else {
        sb_aln_warp_loop<KIND, Q, false>(tb, s, p, sub, dist);
    }
}

#define SB_ALN_WARP_KIND(NAME, KIND, Q)                                                      \
    extern "C" __global__ void sb_aln_warp##Q##_##NAME(SbTables tb, SbSpace s, SbAln p,        \
                                                       int* dist) {                          \
        sb_aln_warp_body<KIND, Q>(tb, s, p, dist);                                           \
    }
#define SB_ALN_DEF_WARP(Q)                          \
    SB_ALN_WARP_KIND(local, SB_ALN_LOCAL, Q)        \
    SB_ALN_WARP_KIND(global, SB_ALN_GLOBAL, Q)      \
    SB_ALN_WARP_KIND(water, SB_ALN_WATER, Q)        \
    SB_ALN_WARP_KIND(needle, SB_ALN_NEEDLE, Q)
SB_ALN_DEF_WARP(1)
SB_ALN_DEF_WARP(2)
SB_ALN_DEF_WARP(4)
SB_ALN_DEF_WARP(8)

__device__ __forceinline__ bool sb_aln_routed(const SbTables& tb, const SbSpace& s, const SbAln& p,
                                              sb_u64 t) {
    if (t >= s.n) {
        return false;
    }
    sb_u64 i, j, m, n;
    sb_decode(s, t, i, j);
    sb_aln_lengths(tb, i, j, m, n);
    return sb_aln_tier(p, m, n) == SB_ALN_HOST;
}

// Routing step 1 (§5.6, the form of G1's Hits): routed count of each block's
// contiguous tile of blockDim.x items.
extern "C" __global__ void sb_aln_route_count(SbTables tb, SbSpace s, SbAln p, sb_u64* block_counts) {
    __shared__ unsigned warp_counts[32];
    const sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x;
    const bool routed = sb_aln_routed(tb, s, p, t);
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
extern "C" __global__ void sb_aln_route_write(SbTables tb, SbSpace s, SbAln p,
                                              const sb_u64* block_offsets, sb_u64* out) {
    __shared__ unsigned warp_base[32];
    const sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x;
    const bool routed = sb_aln_routed(tb, s, p, t);
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
