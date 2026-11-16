// Resident T2/K2 stages. Reuse seqblaze's unchanged primer3 cell, exterior,
// backtrace and software-arithmetic helpers. No input or pair result is read
// by the host. The old HOST tier retains its full table across bounded launches.
#pragma once
#include "primer3.cuh"

struct SbResidentP3Plan {
    sb_u64 buckets;  // u8[n], one of 0..4
    sb_u64 stats;    // u64[5][5]: count,slot,depth,fill_groups,selection_cells
    sb_u64 cap;
    sb_u64 group_lanes;
    sb_u64 unary;
    // primer3-cuda.md §22.4 step 3: byte c (c = 0 … 3) is the base g of width class c
    // (w <= 16, <= 32, <= 64, > 64) chosen by item count; 0 leaves the width rule.
    sb_u64 classes;
};
struct SbResidentP3Resume {
    sb_u64 items;
    sb_u64 first;
    sb_u64 n;
    sb_u64 state;
    sb_u64 steps;
};
struct SbResidentP3State {
    sb_u64 l1, l2, both, u, z, scan, bu, bz, score;
};
static_assert(sizeof(SbResidentP3Plan) == 48, "resident P3 plan ABI");
static_assert(sizeof(SbResidentP3Resume) == 40, "resident P3 continuation ABI");
static_assert(sizeof(SbResidentP3State) == 72, "resident P3 state ABI");

// The existing scalar planner's integer formulas (plan.rs, primer3-cuda.md
// section 6.6). No arithmetic result depends on the bucket or group size.
__device__ __forceinline__ sb_u64 sb_rp3_min(sb_u64 a, sb_u64 b) { return a < b ? a : b; }
__device__ __forceinline__ sb_u64 sb_rp3_ceil(sb_u64 n, sb_u64 d) { return (n + d - 1) / d; }
__device__ __forceinline__ sb_u64 sb_rp3_candidates(sb_u64 n, sb_u64 m) {
    if (n < 2 || m < 2) return 0;
    sb_u64 c = 0;
    for (sb_u64 a = 0; a <= sb_rp3_min(30, n - 2); ++a)
        c += sb_rp3_min(30 - a, m - 2) + 1;
    return c;
}
__device__ __forceinline__ sb_u64 sb_rp3_depth(sb_u64 n, sb_u64 m, sb_u64 g, bool unary) {
    const sb_u64 c = sb_rp3_candidates(n, m);
    if (!unary) return n * (sb_rp3_ceil(m, g) * (c + 1) + m) + sb_rp3_min(n, m) * (c + 2);
    const sb_u64 width = n > 4 ? n - 4 : 1;
    return n * (sb_rp3_ceil(width, g) * (c + 2) + 4 * n)
        + sb_rp3_ceil(n, 2) * 2 * (c + 2) + 6 * n * sb_rp3_ceil(n, 5);
}
extern "C" __global__ void sb_resident_p3_plan(
    SbTables tb, SbSpace s, SbP3 p, SbResidentP3Plan plan) {
    const sb_u64 stride = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < s.n; t += stride) {
        sb_u64 i, j;
        if (plan.unary) { i = s.k0 + t; j = i; } else sb_decode(s, t, i, j);
        sb_u64 n = sb_a_row(tb, i).length;
        sb_u64 m = plan.unary ? n : (sb_u64)sb_b_row(tb, j).length;
        if (!plan.unary && p.swap) { const sb_u64 q = n; n = m; m = q; }
        const sb_u64 width = plan.unary ? (n > 4 ? n - 4 : 1) : m;
        const unsigned klass = width <= 16 ? 0u : width <= 32 ? 1u : width <= 64 ? 2u : 3u;
        const sb_u64 by_width = 4ull << klass;
        const sb_u64 by_count = (plan.classes >> (8u * klass)) & 0xffull;
        const sb_u64 base = plan.group_lanes ? plan.group_lanes
            : by_count > by_width ? by_count : by_width;
        unsigned bucket = 4;
        sb_u64 depth = sb_rp3_depth(n, m, 32, plan.unary != 0);
        for (unsigned v = 0; v < 4; ++v) {
            const sb_u64 g = 4ull << v;
            const sb_u64 d = sb_rp3_depth(n, m, g, plan.unary != 0);
            if (g >= base && d <= plan.cap) { bucket = v; depth = d; break; }
        }
        if (plan.unary && bucket == 4) bucket = 3; // T2's existing never-route rule.
        const sb_u64 bytes = plan.unary ? SB_T2_CELLS + (n + 1) * (n + 1) * sizeof(SbHs) + 64
            : sb_rp3_ceil(n + m + 4, 16) * 16 + (n + 1) * (m + 1) * sizeof(SbHs);
        ((sb_u8*)plan.buckets)[t] = (sb_u8)bucket;
        sb_u64* stats = (sb_u64*)plan.stats + 5 * bucket;
        atomicAdd(stats, 1ull);
        atomicMax(stats + 1, bytes);
        atomicMax(stats + 2, depth);
        atomicMax(stats + 3, n * sb_rp3_ceil(m, 32));
        atomicMax(stats + 4, p.end1 ? m : n * m);
    }
}
extern "C" __global__ void sb_resident_p3_flags(
    const sb_u8* buckets, sb_u64 n, sb_u64 bucket, sb_u64* flags) {
    const sb_u64 stride = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride)
        flags[t] = buckets[t] == bucket ? 1ull : 0ull;
}
extern "C" __global__ void sb_resident_p3_positions(
    const sb_u64* flags, const sb_u64* prefix, sb_u64 n, sb_u64* items) {
    const sb_u64 stride = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride)
        if (flags[t]) items[prefix[t]] = t;
}

// Each invocation owns a complete group and a disjoint global slot. SbSequence
// performs the device layout/stride access, before any codes are interpreted.
template <int G>
__device__ __forceinline__ SbK2 sb_rp3_initialize(
    const SbTables& tb, const SbSpace& s, const SbP3& p, sb_u64 t,
    sb_u8* slot, int me, unsigned mask) {
    sb_u64 i, j;
    sb_decode(s, t, i, j);
    SbSequence a = sb_a_row(tb, i), b = sb_b_row(tb, j);
    if (p.swap) { const SbSequence q = a; a = b; b = q; }
    const int l1 = (int)a.length, l2 = (int)b.length;
    const sb_u64 head = sb_rp3_ceil((sb_u64)l1 + l2 + 4, 16) * 16;
    sb_p3_poison(p, slot, me, G, head + (sb_u64)(l1 + 1) * (l2 + 1) * sizeof(SbHs));
    __syncwarp(mask);
    sb_u8* x = slot;
    sb_u8* y = slot + l1 + 2;
    for (int k = me; k <= l1 + 1; k += G)
        x[k] = (k == 0 || k == l1 + 1) ? (sb_u8)4 : sb_p3_code(a[k - 1]);
    for (int k = me; k <= l2 + 1; k += G)
        y[k] = (k == 0 || k == l2 + 1) ? (sb_u8)4 : sb_p3_code(b[l2 - k]);
    __syncwarp(mask);
    const bool both = sb_k2_self_compl(x, l1, me, G, mask)
        && sb_k2_self_compl(y, l2, me, G, mask);
    SbK2 f;
    f.p = &p; f.x = x; f.y = y;
    f.ends = (const sb_u64*)p.ends + (both ? 1250u : 0u);
    f.cell = (SbHs*)(slot + head);
    f.q = both ? p.c.q1 : p.c.q4;
    f.theta_omega = both ? p.c.theta_omega1 : p.c.theta_omega4;
    f.l1 = l1; f.l2 = l2; f.stride = l2 + 1;
    return f;
}
__device__ __forceinline__ SbK2 sb_rp3_restore(
    const SbP3& p, sb_u8* slot, const SbResidentP3State& state) {
    SbK2 f;
    f.p = &p; f.l1 = (int)state.l1; f.l2 = (int)state.l2;
    f.x = slot; f.y = slot + f.l1 + 2;
    f.cell = (SbHs*)(slot + sb_rp3_ceil(state.l1 + state.l2 + 4, 16) * 16);
    f.ends = (const sb_u64*)p.ends + (state.both ? 1250u : 0u);
    f.q = state.both ? p.c.q1 : p.c.q4;
    f.theta_omega = state.both ? p.c.theta_omega1 : p.c.theta_omega4;
    f.stride = f.l2 + 1;
    return f;
}
__device__ __forceinline__ void sb_rp3_choose(
    const SbK2& f, sb_u64 position, sb_u64& score, sb_u64& bu, sb_u64& bz) {
    const int u = (int)(position / (sb_u64)f.l2) + 1;
    const int z = (int)(position % (sb_u64)f.l2) + 1;
    const SbHs c = sb_k2_c(f, u, z);
    if (sb_f64_is_finite(c.h)) {
        const sb_u64 candidate = sb_k2_gamma(*f.p, c, sb_k2_right(f, u, z));
        if (sb_f64_lt(candidate, score)) { score = candidate; bu = u; bz = z; }
    }
}
__device__ __forceinline__ void sb_rp3_report(
    const SbK2& f, sb_u64 t, int bu, int bz) {
    const SbP3& p = *f.p;
    bool found = false;
    sb_u64 dg = 0, dh = 0, ds = 0, tm = 0;
    if (sb_p3_wc(f.x[bu], f.y[bz])) {
        found = true;
        const sb_i64 n_bp = sb_k2_backtrace(f, bu, bz);
        const sb_u64 nu = sb_f64_from_i64(n_bp - 1);
        const SbHs c = sb_k2_c(f, bu, bz), r = sb_k2_right(f, bu, bz);
        const sb_u64 h = sb_p3_add(sb_p3_add(c.h, r.h), p.c.init_h);
        const sb_u64 entropy = sb_p3_add(sb_p3_add(sb_p3_add(c.s, r.s), p.c.init_s),
                                        sb_p3_mul(nu, p.c.sigma));
        const sb_u64 energy = sb_p3_sub(h, sb_p3_mul(p.c.t_u, entropy));
        const sb_u64 temperature = sb_p3_sub(sb_p3_div(h, sb_p3_add(entropy, f.q)), p.c.k0);
        dg = sb_p3_div(energy, p.c.kcal_div);
        dh = sb_p3_div(h, p.c.kcal_div); ds = entropy; tm = temperature;
    }
    // The existing K2 engine key is evaluated from the same unaltered dg bits.
    ((sb_i64*)p.key)[t] = sb_p3_key(dg);
    ((int*)p.found)[t] = found ? 1 : 0;
    ((sb_u64*)p.dg)[t] = sb_f64_canon(dg);
    ((sb_u64*)p.dh)[t] = sb_f64_canon(dh);
    ((sb_u64*)p.ds)[t] = sb_f64_canon(ds);
    ((sb_u64*)p.tm)[t] = sb_f64_canon(tm);
}

// ------------------------------------------------------------------ resident fast kernels
//
// primer3-cuda.md §22 (lane dpgpu). Everything below up to the continuation kernels is
// compiled only into the resident fast kernels sb_resident_p3_dimer_g* and
// sb_resident_p3_hairpin_g* (§22.3 item 3). The Session kernels (sb_p3_*_g*, through
// sb_k2_item and sb_t2_item) and the K2 continuation (through sb_k2_cell) keep the
// helpers of primer3.cuh unchanged.

// The cell-invariant inner-pair entries of one K2 cell (§22.4 step 1 (h)): MM and M1
// at the inner key (y[z], y[z - 1], x[u], x[u - 1]) and the terminal A·T term at
// (x[u], y[z]), each read once per cell instead of once per candidate.
struct SbRp3Inner {
    SbHs mm;
    SbHs m1;
    SbHs at;
};

// sb_k2_loop_candidate with the inner entries of the cell and fl(α1 · d) from the block's
// table `asym` (asym[d] = mul(α1, from_i64(d)), the same routine on the same operand
// bits). Every other operation, operand and grouping is the original's.
__device__ __noinline__ SbHs sb_rp3_k2_candidate(const SbK2& f, int u, int z, int alpha,
                                                  int beta, SbHs pred, const SbRp3Inner& in,
                                                  const sb_u64* asym) {
    const SbP3& p = *f.p;
    const sb_u8* x = f.x;
    const sb_u8* y = f.y;
    const int up = u - alpha - 1;
    const int zp = z - beta - 1;
    const int ell = alpha + beta;
    SbHs value;
    if (alpha == 0 || beta == 0) {
        if (ell == 1) {
            const SbHs inc =
                sb_p3_sum(sb_p3_entry(p, SB_P3_LOOP_BU, 1u),
                          sb_p3_entry(p, SB_P3_NN, sb_p3_idx4(x[up], x[u], y[zp], y[z])));
            if (sb_f64_gt(inc.h, SB_P3_ZERO) || sb_f64_gt(inc.s, SB_P3_ZERO)) {
                return sb_p3_omega(p);
            }
            return sb_p3_sum(inc, pred);
        }
        value = sb_p3_sum(sb_p3_sum(sb_p3_sum(sb_p3_entry(p, SB_P3_LOOP_BU, (unsigned)ell),
                                              sb_p3_at(p, x[up], y[zp])),
                                    in.at),
                          pred);
    } else if (alpha == 1 && beta == 1) {
        const SbHs outer =
            sb_p3_entry(p, SB_P3_MM, sb_p3_idx4(x[up], x[up + 1], y[zp], y[zp + 1]));
        value = sb_p3_sum(sb_p3_sum(outer, in.mm), pred);
    } else {
        const SbHs outer =
            sb_p3_entry(p, SB_P3_M1, sb_p3_idx4(x[up], x[up + 1], y[zp], y[zp + 1]));
        const int diff = alpha > beta ? alpha - beta : beta - alpha;
        const SbHs lp = sb_p3_entry(p, SB_P3_LOOP_IN, (unsigned)ell);
        value.h = sb_p3_add(
            sb_f64_add_zero(sb_p3_add(sb_p3_add(lp.h, outer.h), in.m1.h)), pred.h);
        value.s = sb_p3_add(
            sb_p3_add(sb_p3_add(sb_p3_add(lp.s, outer.s), in.m1.s), asym[diff]), pred.s);
    }
    if (!sb_f64_is_finite(value.h) ||
        (sb_f64_gt(value.h, SB_P3_ZERO) && sb_f64_gt(value.s, SB_P3_ZERO))) {
        return sb_p3_omega(p);
    }
    return value;
}

// One round of the K2 fill (§22.4 step 1 (c), (d)): every lane of the group calls this,
// `has` false for a lane without a cell in the round. A lane with a cell runs stages 1
// and 2 exactly as sb_k2_cell, then walks the stage-3 sequence (ℓ ascending, then α) to
// its next candidate whose predecessor has a finite ΔH (the original skip, R3); the
// group votes, and the live lanes evaluate their candidates together under the
// original strict test and clamp. A lane's own sequence of candidates, tests and
// operations is sb_k2_cell's.
template <int G>
__device__ __forceinline__ SbHs sb_rp3_k2_cell(const SbK2& f, int u, int z, bool has,
                                               unsigned mask, const sb_u64* asym) {
    const SbP3& p = *f.p;
    SbHs v = sb_p3_omega(p);
    SbHs r = v;
    sb_u64 gv = 0;
    int amax = 0, bmax = 0, lmax = 0;
    bool live = false;
    if (has) {
        r = sb_k2_right(f, u, z);
        // Stage 1: initiation.
        v = sb_k2_left(f, u, z);
        if (u > 1 && z > 1) {
            // Stage 2: the stack, else Θ_Ω of the pair's Q.
            const SbHs pred = sb_k2_c(f, u - 1, z - 1);
            const SbHs nn =
                sb_p3_entry(p, SB_P3_NN, sb_p3_idx4(f.x[u - 1], f.x[u], f.y[z - 1], f.y[z]));
            SbHs stacked;
            sb_u64 theta;
            if (sb_f64_is_finite(pred.h) && sb_f64_is_finite(nn.h)) {
                stacked = sb_p3_sum(pred, nn);
                theta = sb_k2_theta_r(f, stacked, r);
            } else {
                stacked = sb_p3_omega(p);
                theta = f.theta_omega;
            }
            if (sb_f64_gt(theta, sb_k2_theta_r(f, v, r))) {
                v = stacked;
            }
            v = sb_p3_clamp(p, v);
            gv = sb_k2_g_r(p, v, r);
            amax = u - 2;
            bmax = z - 2;
            const int lsum = amax + bmax;
            lmax = lsum < SB_P3_MAX_LOOP ? lsum : SB_P3_MAX_LOOP;
            live = lmax >= 1;
        }
    }
    SbRp3Inner in;
    if (live) {
        const unsigned key = sb_p3_idx4(f.y[z], f.y[z - 1], f.x[u], f.x[u - 1]);
        in.mm = sb_p3_entry(p, SB_P3_MM, key);
        in.m1 = sb_p3_entry(p, SB_P3_M1, key);
        in.at = sb_p3_at(p, f.x[u], f.y[z]);
    }
    // Stage 3: ℓ ascending, then α, strict on G_R, against the current value.
    int ell = 1;
    int alpha = 1 > bmax ? 1 - bmax : 0;
    SbHs pc = v;
    for (;;) {
        while (live) {
            const int hi = ell < amax ? ell : amax;
            if (alpha > hi) {
                if (++ell > lmax) {
                    live = false;
                    break;
                }
                alpha = ell > bmax ? ell - bmax : 0;
                continue;
            }
            pc = sb_k2_c(f, u - alpha - 1, z - (ell - alpha) - 1);
            if (sb_f64_is_finite(pc.h)) {
                break;
            }
            ++alpha;
        }
        if (__ballot_sync(mask, live) == 0u) {
            break;
        }
        if (live) {
            const SbHs x = sb_rp3_k2_candidate(f, u, z, alpha, ell - alpha, pc, in, asym);
            if (sb_f64_is_finite(x.h)) {
                const sb_u64 gx = sb_k2_g_r(p, x, r);
                if (sb_f64_lt(gx, gv)) {
                    // cl(x) is F or x itself; G_R(x) is gx (same routine, same bits).
                    const bool fresh = sb_f64_lt(x.s, p.c.s_cutoff);
                    v = fresh ? sb_p3_hs(p.c.fresh_h, p.c.fresh_s) : x;
                    gv = fresh ? sb_k2_g_r(p, v, r) : gx;
                }
            }
            ++alpha;
        }
    }
    return v;
}

// One K2 pair of the resident fast kernels (§22.4 step 1): rows in order; Ω first, then
// the k-th pairing cell of the row to lane k mod G in round ⌊k/G⌋; Γ of each finite
// cell into the lane's first strict minimum (§6.4 forms (d) and (a)); after the last
// row a width-G butterfly; lane 0 runs the unchanged backtrace and report.
template <int G>
__device__ __forceinline__ void sb_rp3_dimer_item(const SbTables& tb, const SbSpace& s,
                                                  const SbP3& p, sb_u64 t, sb_u8* slot, int me,
                                                  unsigned mask, const sb_u64* asym) {
    const SbK2 f = sb_rp3_initialize<G>(tb, s, p, t, slot, me, mask);
    const SbHs omega = sb_p3_omega(p);
    sb_u64 best = SB_F64_INF;
    sb_u64 index = 0;
    int bu = 1, bz = 1;
    const int first = p.end1 ? f.l1 : 1;
    for (int u = 1; u <= f.l1; ++u) {
        for (int z = 1 + me; z <= f.l2; z += G) {
            if (!sb_p3_wc(f.x[u], f.y[z])) f.cell[u * f.stride + z] = omega;
        }
        int scan = 1;  // this lane's scan position in the row
        int k = 0;     // pairing cells before `scan`
        for (;;) {
            int mine = 0;
            // Lane-local rank scan. Measured: this loop shape keeps the groups of one
            // warp evaluating together at g < 32 (2.5x against a single-exit form).
            while (scan <= f.l2) {
                if (sb_p3_wc(f.x[u], f.y[scan])) {
                    if (k++ % G == me) {
                        mine = scan;
                        ++scan;
                        break;
                    }
                }
                ++scan;
            }
            if (__ballot_sync(mask, mine != 0) == 0u) break;
            const SbHs c = sb_rp3_k2_cell<G>(f, u, mine != 0 ? mine : 1, mine != 0, mask, asym);
            if (mine != 0) {
                f.cell[u * f.stride + mine] = c;
                if (u >= first && sb_f64_is_finite(c.h)) {
                    const sb_u64 gamma = sb_k2_gamma(p, c, sb_k2_right(f, u, mine));
                    if (sb_f64_lt(gamma, best)) {
                        best = gamma;
                        index = (sb_u64)(u - 1) * (sb_u64)f.l2 + (sb_u64)(mine - 1);
                        bu = u;
                        bz = mine;
                    }
                }
            }
        }
        __syncwarp(mask);
    }
    // §6.4 (a): B replaces A iff lt(Γ_B, Γ_A), or eq(Γ_B, Γ_A) and B is earlier.
    for (int off = G / 2; off >= 1; off >>= 1) {
        const sb_u64 ob = __shfl_xor_sync(mask, best, off, G);
        const sb_u64 oi = __shfl_xor_sync(mask, index, off, G);
        const int ou = __shfl_xor_sync(mask, bu, off, G);
        const int oz = __shfl_xor_sync(mask, bz, off, G);
        if (sb_f64_lt(ob, best) || (sb_f64_eq(ob, best) && oi < index)) {
            best = ob;
            index = oi;
            bu = ou;
            bz = oz;
        }
    }
    if (me == 0) {
        // No cell below +∞: the sequential scan keeps its start (1, 1).
        if (best == SB_F64_INF) {
            bu = 1;
            bz = 1;
        }
        sb_rp3_report(f, t, bu, bz);
    }
    // The next item's writes follow every read of this one.
    __syncwarp(mask);
}

template <int G>
__device__ __forceinline__ void sb_rp3_dimer(SbTables tb, SbSpace s, const SbP3& p) {
    // §22.4 step 1 (h): fl(α1 · d) for d = 0 … 30, once per block, the same routine on the
    // same operand bits as the original's per-candidate product.
    __shared__ sb_u64 asym[SB_P3_MAX_LOOP + 1];
    for (int d = (int)threadIdx.x; d <= SB_P3_MAX_LOOP; d += (int)blockDim.x)
        asym[d] = sb_p3_mul(p.c.alpha1, sb_f64_from_i64(d));
    __syncthreads();
    const int me = (int)threadIdx.x % G;
    const unsigned shift = (threadIdx.x & 31u) / G * G;
    const unsigned mask = (0xffffffffu >> (32 - G)) << shift;
    const sb_u64 group = ((sb_u64)blockIdx.x * blockDim.x + threadIdx.x) / G;
    const sb_u64 groups = (sb_u64)gridDim.x * blockDim.x / G;
    sb_u8* slot = (sb_u8*)p.slots + group * p.slot;
    for (sb_u64 k = group; k < p.count; k += groups)
        sb_rp3_dimer_item<G>(tb, s, p, ((const sb_u64*)p.items)[k], slot, me, mask, asym);
}

// One round of the T2 fill (§22.4 step 2): every lane of the group calls this, `has`
// false for a lane without a cell in the round. A lane with a cell runs stage 1 as
// sb_t2_cell; in stage 2 it advances through (n = 1 … n_max, λ5 = 0 … n) to its next
// Watson–Crick inner pair (the original `continue`), the group votes, and the live lanes
// evaluate their loop candidates together under the original test (finite ΔH, gt on
// Θ, the 1×1 margin) and clamp; stage 3 follows per lane. A lane's own sequence of
// candidates, tests and operations is sb_t2_cell's.
template <int G>
__device__ __forceinline__ SbHs sb_rp3_t2_cell(const SbT2& f, int y, int z, bool has,
                                               unsigned mask) {
    const SbP3& p = *f.p;
    SbHs v = sb_p3_hs(p.c.fresh_h, p.c.fresh_s);
    sb_u64 tv = 0;
    int n_max = 0;
    bool live = false;
    if (has) {
        // Stage 1: the stack on (y + 1, z − 1).
        const SbHs stack = sb_p3_sum(sb_t2_c(f, y + 1, z - 1),
                                     sb_p3_entry(p, SB_P3_NN, sb_t2_stack_key(f, y, z)));
        if (sb_f64_gt(sb_p3_theta(p, stack), sb_p3_theta(p, v))) {
            v = sb_p3_clamp(p, stack);
        }
        tv = sb_p3_theta(p, v);
        const int span = z - y;
        if (span >= SB_P3_MIN_SPAN + 3) {
            const int top = span - SB_P3_MIN_SPAN - 2;
            n_max = top < SB_P3_MAX_LOOP ? top : SB_P3_MAX_LOOP;
            live = n_max >= 1;
        }
    }
    // Stage 2: n = λ5 + λ3 ascending, then λ5; strict; the 1×1 margin.
    int n = 1, l5 = 0;
    for (;;) {
        while (live) {
            if (l5 > n) {
                if (++n > n_max) {
                    live = false;
                    break;
                }
                l5 = 0;
                continue;
            }
            if (sb_p3_wc(f.b[y + 1 + l5], f.b[z - 1 - (n - l5)])) {
                break;
            }
            ++l5;
        }
        if (__ballot_sync(mask, live) == 0u) {
            break;
        }
        if (live) {
            const int l3 = n - l5;
            const SbHs x = sb_t2_loop_candidate(f, y, z, y + 1 + l5, z - 1 - l3);
            if (sb_f64_is_finite(x.h)) {
                const sb_u64 tx = sb_p3_theta(p, x);
                const bool one_by_one = l5 == 1 && l3 == 1;
                if (sb_f64_gt(tx, tv) &&
                    (!one_by_one || sb_f64_ge(sb_p3_sub(tx, tv), p.c.margin))) {
                    v = sb_p3_clamp(p, x);
                    tv = sb_p3_theta(p, v);
                }
            }
            ++l5;
        }
    }
    if (!has) {
        return v;
    }
    // Stage 3: the hairpin loop against v on G_E, ties to the hairpin loop.
    const SbHs hp = sb_t2_hairpin(f, y, z, v);
    const SbHs e = sb_p3_end((const sb_u64*)p.ends,
                             sb_p3_idx4(f.b[y], f.b[y + 1], f.b[z], f.b[z + 1]));
    const sb_u64 g_v =
        sb_p3_sub(sb_p3_add(v.h, e.h), sb_p3_mul(p.c.t37, sb_p3_add(v.s, e.s)));
    const sb_u64 g_hp =
        sb_p3_sub(sb_p3_add(hp.h, e.h), sb_p3_mul(p.c.t37, sb_p3_add(hp.s, e.s)));
    const SbHs chosen = sb_f64_lt(g_v, g_hp) ? v : hp;
    return sb_p3_clamp(p, chosen);
}

// The T2 exterior selection with κ spread over the group (§22.4 step 2, §6.4 (b)). For
// each ρ and option, every lane keeps the first strict maximum (gt from −∞) of Θ over its
// own κ = me, me + G, …; a width-G butterfly takes the other candidate iff it has one and
// (this lane has none, or its Θ is gt, or eq with the smaller κ). Every lane then applies
// the sequential option rule (ge, the later option wins) and the acceptance test; lane 0
// writes P(ρ). The Θ of an option's candidate is the one its lane computed (same routine,
// same bits).
template <int G>
__device__ __forceinline__ void sb_rp3_t2_exterior(const SbT2& f, int me, unsigned mask) {
    const SbP3& p = *f.p;
    for (int rho = 2; rho <= f.len; ++rho) {
        const SbHs carry = f.pre[rho - 1];
        bool have = false;
        SbHs winner = carry;
        sb_u64 best = sb_p3_theta(p, carry);
        for (int opt = 0; opt < 4; ++opt) {
            const int kmax = sb_t2_kappa_max(opt, rho);
            bool mine = false;
            sb_u64 theta = SB_P3_NEG_INF;
            int kappa_best = 0x7fffffff;
            SbHs x_best = sb_p3_hs(SB_P3_ZERO, SB_P3_ZERO);
            for (int kappa = me; kappa <= kmax; kappa += G) {
                const SbHs pk = f.pre[kappa];
                const SbHs pre = sb_f64_ge(sb_p3_theta(p, pk), SB_P3_NEG_ZERO)
                                     ? pk
                                     : sb_p3_hs(SB_P3_ZERO, SB_P3_ZERO);
                const SbHs x = sb_t2_helix(f, opt, kappa, rho, pre);
                const bool valid = sb_f64_is_finite(x.h) && sb_f64_le(x.h, SB_P3_ZERO) &&
                                   sb_f64_le(x.s, SB_P3_ZERO) && sb_f64_gt(x.s, p.c.s_cutoff);
                if (valid) {
                    const sb_u64 tx = sb_p3_theta(p, x);
                    if (sb_f64_gt(tx, theta)) {
                        theta = tx;
                        x_best = x;
                        kappa_best = kappa;
                        mine = true;
                    }
                }
            }
            for (int off = G / 2; off >= 1; off >>= 1) {
                const int oh = __shfl_xor_sync(mask, (int)mine, off, G);
                const sb_u64 ot = __shfl_xor_sync(mask, theta, off, G);
                const int ok = __shfl_xor_sync(mask, kappa_best, off, G);
                const sb_u64 oxh = __shfl_xor_sync(mask, x_best.h, off, G);
                const sb_u64 oxs = __shfl_xor_sync(mask, x_best.s, off, G);
                if (oh != 0 && (!mine || sb_f64_gt(ot, theta) ||
                                (sb_f64_eq(ot, theta) && ok < kappa_best))) {
                    mine = true;
                    theta = ot;
                    kappa_best = ok;
                    x_best = sb_p3_hs(oxh, oxs);
                }
            }
            if (mine && sb_f64_ge(theta, best)) {
                best = theta;
                winner = x_best;
                have = true;
            }
        }
        const bool accept =
            have && sb_f64_lt(sb_p3_sub(winner.h, sb_p3_mul(p.c.t_u, winner.s)), SB_P3_ZERO);
        if (me == 0) {
            f.pre[rho] = accept ? winner : carry;
        }
        // P(ρ) is final before any lane reads it at a later ρ.
        __syncwarp(mask);
    }
}

// T2's lane-0 tail after the exterior (§6.2 steps 3-5): found, the backtrace, the
// reported values and the stored row t, as in sb_t2_item (a copy, so that the Session
// kernels' code stays as it is, §22.3 item 3).
__device__ __forceinline__ void sb_rp3_t2_finish(const SbT2& f, const SbP3& p, int len, sb_u64 t) {
    const SbHs pl = f.pre[len];
    bool found = false;
    sb_u64 dg = 0, dh = 0, ds = 0, tm = 0;
    if (sb_f64_is_finite(pl.h)) {
        found = true;
        unsigned long long paired = 0;
        sb_t2_backtrace(f, paired);
        // Positions 1 … L − 1 (the CPU counts paired[1..L]).
        const unsigned long long counted = paired & (((1ull << len) - 1ull) & ~1ull);
        const sb_i64 n_pair = (sb_i64)__popcll(counted);
        const sb_i64 m = n_pair / 2 - 1;
        const sb_u64 salt = sb_p3_mul(sb_f64_from_i64(m), p.c.sigma);
        const sb_u64 s = sb_p3_add(pl.s, salt);
        const sb_u64 h = pl.h;
        const sb_u64 g = sb_p3_sub(h, sb_p3_mul(p.c.t_u, s));
        // tm: +0 at an entropy of exactly zero (either sign), else the formula.
        const sb_u64 temp = sb_f64_eq(s, SB_P3_ZERO)
                                ? SB_P3_ZERO
                                : sb_p3_sub(sb_p3_div(h, s), p.c.k0);
        dg = sb_p3_div(g, p.c.kcal_div);
        dh = sb_p3_div(h, p.c.kcal_div);
        ds = s;
        tm = temp;
    }
    const bool passed = p.has_cutoff == 0 || sb_f64_le(dg, p.cutoff);
    const bool masked = p.mask_each != 0 && !passed;
    if (p.passed != 0) {
        ((sb_u8*)p.passed)[t] = passed ? 1 : 0;
    }
    if (p.found != 0) {
        ((int*)p.found)[t] = masked ? SB_P3_INT32_MAX : (found ? 1 : 0);
    }
    if (p.dg != 0) {
        ((sb_u64*)p.dg)[t] = masked ? SB_F64_QNAN : sb_f64_canon(dg);
    }
    if (p.dh != 0) {
        ((sb_u64*)p.dh)[t] = masked ? SB_F64_QNAN : sb_f64_canon(dh);
    }
    if (p.ds != 0) {
        ((sb_u64*)p.ds)[t] = masked ? SB_F64_QNAN : sb_f64_canon(ds);
    }
    if (p.tm != 0) {
        ((sb_u64*)p.tm)[t] = masked ? SB_F64_QNAN : sb_f64_canon(tm);
    }
}

// One T2 item of the resident fast kernels (§22.4 step 2): sb_t2_item's slot layout,
// poison and codes; the compacted fill; the κ-parallel exterior; lane 0's unchanged
// backtrace and report (sb_rp3_t2_finish).
template <int G>
__device__ __forceinline__ void sb_rp3_t2_item(const SbP3& p, const sb_u8* seq, int len,
                                               sb_u8* slot, sb_u64 t, int me, unsigned mask) {
    sb_u8* codes = slot + SB_T2_CODES;
    SbT2 f;
    f.p = &p;
    f.b = codes;
    f.pre = (SbHs*)(slot + SB_T2_PREFIX);
    f.cell = (SbHs*)(slot + SB_T2_CELLS);
    f.len = len;
    f.stride = len + 1;
    sb_p3_poison(p, slot, me, G, SB_T2_CELLS + (sb_u64)(len + 1) * (len + 1) * sizeof(SbHs));
    __syncwarp(mask);
    for (int x = me; x <= len + 1; x += G) {
        codes[x] = (x == 0 || x == len + 1) ? (sb_u8)4 : sb_p3_code(seq[x - 1]);
    }
    __syncwarp(mask);
    const SbHs omega = sb_p3_omega(p);
    // Fill: columns in order; Ω first, then the k-th admissible cell of the column (y
    // ascending) to lane k mod G in round ⌊k/G⌋; every y < z written.
    for (int z = 1; z <= len; ++z) {
        for (int y = 1 + me; y < z; y += G) {
            if (!(z >= y + SB_P3_MIN_SPAN && sb_p3_wc(codes[y], codes[z]))) {
                f.cell[y * f.stride + z] = omega;
            }
        }
        int scan = 1;  // this lane's scan position in the column
        int k = 0;     // admissible cells before `scan`
        for (;;) {
            int mine = 0;
            // Lane-local rank scan; the loop shape of the K2 fill (measured, see there).
            while (scan < z) {
                if (z >= scan + SB_P3_MIN_SPAN && sb_p3_wc(codes[scan], codes[z])) {
                    if (k++ % G == me) {
                        mine = scan;
                        ++scan;
                        break;
                    }
                }
                ++scan;
            }
            if (__ballot_sync(mask, mine != 0) == 0u) break;
            const SbHs c = sb_rp3_t2_cell<G>(f, mine != 0 ? mine : 1, z, mine != 0, mask);
            if (mine != 0) f.cell[mine * f.stride + z] = c;
        }
        __syncwarp(mask);
    }
    if (me == 0) {
        f.pre[0] = omega;
        f.pre[1] = omega;
    }
    __syncwarp(mask);
    sb_rp3_t2_exterior<G>(f, me, mask);
    if (me == 0) sb_rp3_t2_finish(f, p, len, t);
    // The next item's writes follow every read of this one.
    __syncwarp(mask);
}

template <int G>
__device__ __forceinline__ void sb_rp3_hairpin(SbTables tb, SbSpace s, const SbP3& p) {
    const int me = (int)threadIdx.x % G;
    const unsigned shift = (threadIdx.x & 31u) / G * G;
    const unsigned mask = (0xffffffffu >> (32 - G)) << shift;
    const sb_u64 group = ((sb_u64)blockIdx.x * blockDim.x + threadIdx.x) / G;
    const sb_u64 groups = (sb_u64)gridDim.x * blockDim.x / G;
    sb_u8* slot = (sb_u8*)p.slots + group * p.slot;
    for (sb_u64 k = group; k < p.count; k += groups) {
        const sb_u64 t = ((const sb_u64*)p.items)[k];
        const SbSequence a = sb_a_row(tb, s.k0 + t);
        sb_u8* raw = slot + p.slot - 64;
        for (int z = me; z < (int)a.length; z += G) raw[z] = a[z];
        __syncwarp(mask);
        sb_rp3_t2_item<G>(p, raw, (int)a.length, slot, t, me, mask);
    }
}
#define SB_RP3_FAST(G) \
extern "C" __global__ void sb_resident_p3_dimer_g##G(SbTables tb, SbSpace s, const __grid_constant__ SbP3 p) { sb_rp3_dimer<G>(tb,s,p); } \
extern "C" __global__ void sb_resident_p3_hairpin_g##G(SbTables tb, SbSpace s, const __grid_constant__ SbP3 p) { sb_rp3_hairpin<G>(tb,s,p); }
SB_RP3_FAST(4)
SB_RP3_FAST(8)
SB_RP3_FAST(16)
SB_RP3_FAST(32)

extern "C" __global__ void sb_resident_p3_init(
    SbTables tb, SbSpace s, SbP3 p, SbResidentP3Resume r) {
    if ((sb_u64)blockIdx.x >= r.n) return;
    const int me = (int)threadIdx.x;
    const sb_u64 t = ((const sb_u64*)r.items)[r.first + blockIdx.x];
    sb_u8* slot = (sb_u8*)p.slots + (sb_u64)blockIdx.x * p.slot;
    const SbK2 f = sb_rp3_initialize<32>(tb, s, p, t, slot, me, 0xffffffffu);
    if (me == 0) {
        SbResidentP3State& state = ((SbResidentP3State*)r.state)[blockIdx.x];
        state.l1 = f.l1; state.l2 = f.l2;
        state.both = f.ends != (const sb_u64*)p.ends;
        state.u = 1; state.z = 1;
        state.scan = p.end1 ? (sb_u64)(f.l1 - 1) * f.l2 : 0;
        state.bu = 1; state.bz = 1; state.score = SB_F64_INF;
    }
}
extern "C" __global__ void sb_resident_p3_fill(SbP3 p, SbResidentP3Resume r) {
    if ((sb_u64)blockIdx.x >= r.n) return;
    const int me = (int)threadIdx.x;
    SbResidentP3State& state = ((SbResidentP3State*)r.state)[blockIdx.x];
    const SbK2 f = sb_rp3_restore(p, (sb_u8*)p.slots + (sb_u64)blockIdx.x * p.slot, state);
    int u = (int)state.u, z = (int)state.z;
    const SbHs omega = sb_p3_omega(p);
    // One candidate list is never split. A 32-group launch costs at most
    // 32*(497+1) per lane, independently of either legal strand length.
    for (sb_u64 q = 0; q < r.steps && u <= f.l1; ++q) {
        const int col = z + me;
        if (col <= f.l2)
            f.cell[u * f.stride + col] = sb_p3_wc(f.x[u], f.y[col]) ? sb_k2_cell(f,u,col) : omega;
        __syncwarp(0xffffffffu);
        z += 32;
        if (z > f.l2) { ++u; z = 1; }
    }
    if (me == 0) { state.u = u; state.z = z; }
}
extern "C" __global__ void sb_resident_p3_select(SbP3 p, SbResidentP3Resume r) {
    if ((sb_u64)blockIdx.x >= r.n || threadIdx.x != 0) return;
    SbResidentP3State& state = ((SbResidentP3State*)r.state)[blockIdx.x];
    const SbK2 f = sb_rp3_restore(p, (sb_u8*)p.slots + (sb_u64)blockIdx.x * p.slot, state);
    const sb_u64 end = sb_rp3_min((sb_u64)f.l1 * f.l2, state.scan + r.steps);
    for (sb_u64 q = state.scan; q < end; ++q)
        sb_rp3_choose(f, q, state.score, state.bu, state.bz);
    state.scan = end;
}
extern "C" __global__ void sb_resident_p3_finish(SbP3 p, SbResidentP3Resume r) {
    if ((sb_u64)blockIdx.x >= r.n || threadIdx.x != 0) return;
    const SbResidentP3State& state = ((const SbResidentP3State*)r.state)[blockIdx.x];
    const SbK2 f = sb_rp3_restore(p, (sb_u8*)p.slots + (sb_u64)blockIdx.x * p.slot, state);
    const sb_u64 t = ((const sb_u64*)r.items)[r.first + blockIdx.x];
    // A legal K2 pair has min(lengths)<=60, so this unchanged backtrace has
    // at most 60 steps, each with at most 496 loop candidates.
    sb_rp3_report(f, t, (int)state.bu, (int)state.bz);
}
extern "C" __global__ void sb_resident_p3_columns(
    SbP3 p, sb_u64 n, sb_u8* found, sb_u64 unary) {
    const sb_u64 stride = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride) {
        found[t] = ((const int*)p.found)[t] != 0 ? 1 : 0;
        if (unary) ((sb_i64*)p.key)[t] = sb_p3_key(((const sb_u64*)p.dg)[t]);
    }
}
