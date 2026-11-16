// T1 native `tm` on the device: per-sequence integer sums and the F1-F10 finalize in
// soft-binary64 (D55). Spec: docs/wiki/gpu/tm-cuda.md §3 (and thermo/tm.md §4-§6).
// Written from those pages only.
//
// Integers only: float64 values travel and are stored as u64 bit patterns, and every
// arithmetic step is a softf64.cuh routine. No transcendental, no floating-point type.
#pragma once

#include "finalize.cuh"
#include "softf64.cuh"
#include "unary.cuh"

// One (h, g, d) entry: ΔH and ΔG37 (dcal/mol) and G_e(T) (dcal/mol) of the call's table.
struct SbTmStep {
    sb_i64 h;
    sb_i64 g;
    sb_i64 d;
};

// The per-call pack (tm-cuda.md §3.1), 480 bytes, passed by value. Mirrors `TmPack`
// in src/lib.rs; every field is 8 bytes wide.
struct SbTmPack {
    SbTmStep step[16];  // index 4·code(X) + code(Y) for the top-strand step XY
    SbTmStep init;
    SbTmStep terminal_at;
    SbTmStep symmetry;
    sb_u64 s;       // bits of the binary64 s
    sb_u64 kappa4;  // bits of κ₄
    sb_u64 kappa1;  // bits of κ₁
};

// Output columns of one chunk, addressed from 0; 0 means "not stored". Float64
// columns are u64 bit patterns, dg_dcal int32, passed one byte per sequence.
struct SbTmOut {
    sb_u64 tm_c;
    sb_u64 dg_kcal;
    sb_u64 dg_dcal;
    sb_u64 dh_kcal;
    sb_u64 ds_cal_per_k;
    sb_u64 passed;
};

#define SB_TM_INT32_MAX 2147483647

// Total code map (A and every other byte 0, C 1, G 2, T 3), as the CPU's.
__device__ __forceinline__ unsigned sb_tm_code(sb_u8 b) {
    return b == 'C' ? 1u : (b == 'G' ? 2u : (b == 'T' ? 3u : 0u));
}

// Stage kernel: one thread per sequence i = k0 + t, t in [0, n), grid-stride.
// §3.2 sums, §3.3 finalize, §3.4 cutoff and Each masks.
extern "C" __global__ void sb_tm(SbUnaryTable tb, sb_u64 k0, sb_u64 n, SbTmPack pack, SbTmOut out,
                                 int has_cutoff, sb_u64 cutoff_bits, int mask_each) {
    // The per-step index differs across lanes. Cache the small table in shared
    // memory once per block so those lookups do not repeatedly read divergent
    // addresses in the kernel parameter space. Every thread reaches the barrier,
    // including the inactive tail threads of the last block.
    __shared__ SbTmStep steps[16];
    for (unsigned x = threadIdx.x; x < 16; x += blockDim.x) {
        steps[x] = pack.step[x];
    }
    __syncthreads();
    const sb_u64 stride = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride) {
        const sb_u64 i = k0 + t;
        const SbSequence seq = sb_unary_row(tb, i);
        const sb_i64 len = seq.length;  // 2 <= len <= 200 after validation
        // §3.2 integer sums, int64, exact.
        const unsigned first = len > 0 ? sb_tm_code(seq[0]) : 0u;
        unsigned previous = first;
        sb_i64 h = 0, g = 0, d = 0;
        for (sb_i64 x = 1; x < len; ++x) {
            const unsigned code = sb_tm_code(seq[x]);
            const SbTmStep& e = steps[(previous << 2) | code];
            h += e.h;
            g += e.g;
            d += e.d;
            previous = code;
        }
        h += pack.init.h;
        g += pack.init.g;
        d += pack.init.d;
        if (first == 0u || first == 3u) {
            h += pack.terminal_at.h;
            g += pack.terminal_at.g;
            d += pack.terminal_at.d;
        }
        if (previous == 0u || previous == 3u) {
            h += pack.terminal_at.h;
            g += pack.terminal_at.g;
            d += pack.terminal_at.d;
        }
        bool sc = len > 0 && (len & 1) == 0;
        for (sb_i64 x = 0; sc && x < len / 2; ++x) {
            sc = sb_tm_code(seq[x]) + sb_tm_code(seq[len - 1 - x]) == 3u;
        }
        if (sc) {
            h += pack.symmetry.h;
            g += pack.symmetry.g;
            d += pack.symmetry.d;
        }
        const sb_i64 p = len - 1;
        const sb_u64 kappa = sc ? pack.kappa1 : pack.kappa4;
        // §3.3 F1-F10: one soft-binary64 operation per step, in this order; F1 and
        // F3-F10 in the shared routine (finalize.cuh) with P = from_i64(L - 1).
        const sb_u64 dg_kcal = sb_f64_div(sb_f64_from_i64(d), SB_TM_C100);  // F2
        sb_u64 dh_kcal, ds, tm_raw;
        sb_nn_finalize(h, g, sb_f64_from_i64(p), pack.s, kappa, dh_kcal, ds, tm_raw);
        const sb_u64 tm_c = sb_f64_canon(tm_raw);
        // §3.4 cutoff (IEEE >= on the canonical bits) and the CPU's masks.
        const bool passed = !has_cutoff || sb_f64_ge(tm_c, cutoff_bits);
        const bool masked = mask_each && !passed;
        if (out.passed != 0) {
            ((sb_u8*)out.passed)[t] = passed ? 1 : 0;
        }
        if (out.tm_c != 0) {
            ((sb_u64*)out.tm_c)[t] = masked ? SB_F64_QNAN : tm_c;
        }
        if (out.dg_kcal != 0) {
            ((sb_u64*)out.dg_kcal)[t] = masked ? SB_F64_QNAN : sb_f64_canon(dg_kcal);
        }
        if (out.dg_dcal != 0) {
            ((int*)out.dg_dcal)[t] = masked ? SB_TM_INT32_MAX : (int)d;
        }
        if (out.dh_kcal != 0) {
            ((sb_u64*)out.dh_kcal)[t] = masked ? SB_F64_QNAN : sb_f64_canon(dh_kcal);
        }
        if (out.ds_cal_per_k != 0) {
            ((sb_u64*)out.ds_cal_per_k)[t] = masked ? SB_F64_QNAN : sb_f64_canon(ds);
        }
    }
}
