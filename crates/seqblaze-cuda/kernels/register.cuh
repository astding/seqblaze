// `tm` on a Pairs input on the device: the register pairing of two equal-length
// strands (one thread per pair), its integer key (D, H, G, Q) from K1's per-call
// full cell keys, the shared soft-binary64 finalize and the int64 selecting key.
// Spec: docs/wiki/gpu/tm-pairs-cuda.md §3-§6 (and thermo/tm.md §13). Written from
// those pages only.
//
// Integers only: the key is an exact int64 sum (every partial sum far inside int64,
// so the order of the terms is irrelevant), float64 values are u64 bit patterns and
// every arithmetic step on them is a softf64.cuh routine. No transcendental, no
// floating-point type, no atomics but the integer defect slot.
#pragma once
#include "duplex.cuh"
#include "finalize.cuh"
#include "hamming.cuh"

#define SB_REG_INT32_MAX 2147483647
#define SB_REG_NAN_KEY 9223372036854775806ll  // 2^63 - 2: a NaN tm_c
#define SB_REG_SIGN 0x8000000000000000ull
#define SB_REG_INF_BITS 0x7FF0000000000000ull

// Mirrors `SbReg` in src/reg.rs; every field is 8 bytes wide. Table offsets index the
// concatenated full cell keys (duplex-cuda.md §4: INIT at 0, INIT + SYM at 1).
struct SbReg {
    sb_u64 full;  // const sb_i64*: (D, H, G, Q) of every cell, ⊥ as (2^60, 0, 0, 0)
    sb_u64 lam;
    sb_u64 init;
    sb_u64 init_sym;
    sb_u64 end_l;
    sb_u64 end_r;
    sb_u64 stack;
    sb_u64 i11;
    sb_u64 i22;
    sb_u64 gen;
    sb_u64 loop_out;
    sb_u64 loop_in;
    sb_u64 s;       // bits of the binary64 s
    sb_u64 kappa4;  // bits of κ₄
    sb_u64 kappa1;  // bits of κ₁
    sb_u64 key;     // sb_i64* per item: the selecting key (always written)
    sb_u64 tm_c;    // u64* per item, or 0 (fields mode writes it)
    sb_u64 dg_kcal;
    sb_u64 dg_dcal;  // int* per item, or 0
    sb_u64 dh_kcal;
    sb_u64 ds_cal_per_k;
    sb_u64 passed;      // u8* per item, or 0 (Each with a cutoff)
    sb_u64 has_cutoff;  // 1: passed = key <= cutoff
    sb_u64 cutoff;      // int64 key of the cutoff, as bits
    sb_u64 mask;        // 1: a failing row stores NaN and INT32_MAX (Each)
    sb_u64 defect;      // u64*: the smallest item number that read a ⊥ cell
    // Pack format 2 (nndb2024-dna.md §5.4.6): N_P and the u8 CLS map over (Λ + 1)².
    sb_u64 n_p;
    sb_u64 cls;
};

__device__ __forceinline__ SbDupKey sb_reg_cell(const SbReg& p, sb_u64 at) {
    const sb_i64* cell = (const sb_i64*)p.full + 4 * at;
    SbDupKey x;
    x.d = __ldg(cell);
    x.h = __ldg(cell + 1);
    x.g = __ldg(cell + 2);
    x.q = __ldg(cell + 3);
    return x;
}

// The key of a canonical tm_c (tm-pairs-cuda.md §3.5): NAN_KEY for a NaN, +m for a set
// sign bit, -m otherwise, with m the magnitude bits; -0.0 and +0.0 give 0.
__device__ __forceinline__ sb_i64 sb_reg_key(sb_u64 bits) {
    const sb_u64 m = bits & ~SB_REG_SIGN;
    if (m > SB_REG_INF_BITS) {
        return SB_REG_NAN_KEY;
    }
    return (bits & SB_REG_SIGN) != 0 ? (sb_i64)m : -(sb_i64)m;
}

// Stage kernel: one thread per item t in [0, n), grid-stride. §3.3 register key, §3.4
// fields, §3.5 selecting key and cutoff, §6 Each masks. Every requested output of every
// item is written.
extern "C" __global__ void sb_reg(SbTables tb, SbSpace s, SbReg p) {
    const sb_u64 stride = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < s.n; t += stride) {
        sb_u64 i, j;
        sb_decode(s, t, i, j);
        const SbSequence pa = sb_a_row(tb, i);
        const SbSequence pb = sb_b_row(tb, j);
        // Equal lengths after validation; the smaller one keeps every read in range.
        const sb_i64 na = pa.length;
        const sb_i64 nb = pb.length;
        const sb_i64 n = na < nb ? na : nb;
        // identical: equal letter codes; sym: identical and a palindrome.
        bool identical = na == nb;
        for (sb_i64 r = 0; identical && r < n; ++r) {
            identical = sb_dup_code(pa[r]) == sb_dup_code(pb[r]);
        }
        bool sym = identical;
        for (sb_i64 r = 0; sym && r < n; ++r) {
            sym = sb_dup_code(pa[r]) + sb_dup_code(pa[n - 1 - r]) == 3;
        }
        bool bottom = false;
        SbDupKey key = sb_reg_cell(p, sym ? p.init_sym : p.init);
        bottom = bottom || !SbDupFull::finite(key);
        // The pair columns r (PT_dev(a_r, b_{n-1-r}) of nndb2024-dna.md §5.4.3), left to
        // right, with their pair types; the step between consecutive ones, or +∞ at a gap
        // g with 2g > Λ. Tables are indexed by the pair types and CLS (§5.4.6).
        const sb_u64 np = p.n_p;
        sb_i64 first = -1, last = -1;
        sb_u64 p_first = 0, p_last = 0;
        bool inf = false;
        for (sb_i64 r = 0; r < n; ++r) {
            const int pr = sb_dup_pt(sb_dup_code(pa[r]), sb_dup_code(pb[n - 1 - r]), np);
            if (pr < 0) {
                continue;
            }
            const sb_u64 x = (sb_u64)pr;
            if (first < 0) {
                first = r;
                p_first = x;
            } else {
                const sb_i64 g = r - last - 1;
                if (2 * (sb_u64)g > p.lam) {
                    inf = true;
                    break;
                }
                const sb_u64 xp = p_last;
                SbDupKey step;
                if (g == 0) {
                    step = sb_reg_cell(p, p.stack + xp * np + x);
                } else {
                    const sb_u64 a1 = sb_dup_code(pa[last + 1]);
                    const sb_u64 b1 = sb_dup_code(pb[n - 2 - last]);
                    if (g == 1) {
                        step = sb_reg_cell(p, p.i11 + ((xp * 4 + a1) * 4 + b1) * np + x);
                    } else if (g == 2) {
                        const sb_u64 a2 = sb_dup_code(pa[last + 2]);
                        const sb_u64 b2 = sb_dup_code(pb[n - 3 - last]);
                        step = sb_reg_cell(
                            p, p.i22 + ((((xp * 4 + a1) * 4 + a2) * 4 + b1) * 4 + b2) * np + x);
                    } else {
                        const sb_u64 lam1 = p.lam + 1;
                        const sb_u64 shape = (sb_u64)g * lam1 + (sb_u64)g;
                        const sb_u64 c = __ldg((const sb_u8*)p.cls + shape);
                        const sb_u64 in_a = sb_dup_code(pa[r - 1]);
                        const sb_u64 in_b = sb_dup_code(pb[n - r]);
                        step = SbDupFull::plus(
                            SbDupFull::plus(
                                sb_reg_cell(p, p.gen + shape),
                                sb_reg_cell(p, p.loop_out + ((c * np + xp) * 4 + a1) * 4 + b1)),
                            sb_reg_cell(p, p.loop_in + ((c * 4 + in_a) * 4 + in_b) * np + x));
                    }
                }
                // A ⊥ part is flagged and not added, so no sum can leave int64.
                if (SbDupFull::finite(step)) {
                    key = SbDupFull::plus(key, step);
                } else {
                    bottom = true;
                }
            }
            last = r;
            p_last = x;
        }
        if (!inf && first >= 0 && first < last) {
            // END_L at the first column, END_R at the last, by pair type; 4 is the empty
            // neighbour.
            const sb_u64 alpha = first >= 1 ? sb_dup_code(pa[first - 1]) : SB_DUP_NONE;
            const sb_u64 beta = first >= 1 ? sb_dup_code(pb[n - first]) : SB_DUP_NONE;
            const SbDupKey left = sb_reg_cell(p, p.end_l + (alpha * np + p_first) * 5 + beta);
            if (SbDupFull::finite(left)) {
                key = SbDupFull::plus(key, left);
            } else {
                bottom = true;
            }
            const sb_u64 alpha2 = last + 1 < n ? sb_dup_code(pa[last + 1]) : SB_DUP_NONE;
            const sb_u64 beta2 = last <= n - 2 ? sb_dup_code(pb[n - 2 - last]) : SB_DUP_NONE;
            const SbDupKey right = sb_reg_cell(p, p.end_r + (p_last * 5 + alpha2) * 5 + beta2);
            if (SbDupFull::finite(right)) {
                key = SbDupFull::plus(key, right);
            } else {
                bottom = true;
            }
        } else {
            inf = true;  // k < 2 (QK8), or a gap above the loop limit
        }
        if (bottom) {
            // Unreachable by rule F (tm.md §13.1); the host recomputes this item on the CPU
            // kernel, which raises its internal error (QTP6).
            atomicMin((sb_u64*)p.defect, s.k0 + t);
        }
        // §3.4 fields: the sentinel for +∞, else the shared F1, F3-F10 (F2, `dg_kcal`, is
        // computed only where it is stored, below: the key needs `tm` alone).
        int dg_dcal;
        sb_u64 dh = SB_F64_QNAN, ds = SB_F64_QNAN, tm = SB_F64_QNAN;
        if (inf) {
            dg_dcal = SB_REG_INT32_MAX;
        } else {
            dg_dcal = (int)key.d;
            const sb_u64 half = sb_f64_div(sb_f64_from_i64(key.q), SB_DUP_C2);  // P = Q/2
            sb_u64 dh_raw, ds_raw, tm_raw;
            sb_nn_finalize(key.h, key.g, half, p.s, identical ? p.kappa1 : p.kappa4, dh_raw,
                           ds_raw, tm_raw);
            dh = sb_f64_canon(dh_raw);
            ds = sb_f64_canon(ds_raw);
            tm = sb_f64_canon(tm_raw);
        }
        // §3.5 the selecting key and the cutoff, integer only; §6 Each masks.
        const sb_i64 q = sb_reg_key(tm);
        const bool passed = p.has_cutoff == 0 || q <= (sb_i64)p.cutoff;
        const bool masked = p.mask != 0 && !passed;
        ((sb_i64*)p.key)[t] = q;
        if (p.passed != 0) {
            ((sb_u8*)p.passed)[t] = passed ? 1 : 0;
        }
        if (p.tm_c != 0) {
            ((sb_u64*)p.tm_c)[t] = masked ? SB_F64_QNAN : tm;
        }
        if (p.dg_kcal != 0) {
            // F2 on D, or on the sentinel's INT32_MAX for +∞; never in key mode.
            ((sb_u64*)p.dg_kcal)[t] =
                masked ? SB_F64_QNAN
                       : sb_f64_canon(sb_f64_div(
                             sb_f64_from_i64(inf ? (sb_i64)SB_REG_INT32_MAX : key.d), SB_TM_C100));
        }
        if (p.dg_dcal != 0) {
            ((int*)p.dg_dcal)[t] = masked ? SB_REG_INT32_MAX : dg_dcal;
        }
        if (p.dh_kcal != 0) {
            ((sb_u64*)p.dh_kcal)[t] = masked ? SB_F64_QNAN : dh;
        }
        if (p.ds_cal_per_k != 0) {
            ((sb_u64*)p.ds_cal_per_k)[t] = masked ? SB_F64_QNAN : ds;
        }
    }
}
