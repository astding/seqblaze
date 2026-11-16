// The shared soft-binary64 finalize of the nearest-neighbour fields (D55): F1 and
// F3-F10 of docs/wiki/gpu/tm-cuda.md §3.3, used by `tm` (tm.cuh, P = L - 1) and by
// K1 `duplex` (duplex.cuh, P = Q / 2; docs/wiki/gpu/duplex-cuda.md §5.5, QKG10).
// Written from those pages only. Values are u64 bit patterns; every step is one
// softf64.cuh routine, in this order, nothing fused or reordered.
#pragma once

#include "softf64.cuh"

#define SB_TM_C100 0x4059000000000000ull   // 100.0
#define SB_TM_C6203 0x40B83B0000000000ull  // 6203.0
#define SB_TM_K0 0x4071126666666666ull     // 0x1.1126666666666p+8, nearest 273.15

// From the integer sums H and G (dcal/mol), the binary64 P, s and κ: dh_kcal (F1),
// ds_cal_per_k (F6) and tm_c (F10), not canonicalised.
__device__ __forceinline__ void sb_nn_finalize(sb_i64 h, sb_i64 g, sb_u64 p, sb_u64 s,
                                               sb_u64 kappa, sb_u64& dh_kcal, sb_u64& ds_cal_per_k,
                                               sb_u64& tm_c) {
    dh_kcal = sb_f64_div(sb_f64_from_i64(h), SB_TM_C100);  // F1
    const sb_u64 x1 = sb_f64_from_i64(200 * (h - g));      // F3, exact
    const sb_u64 x2 = sb_f64_div(x1, SB_TM_C6203);         // F4
    const sb_u64 x3 = sb_f64_mul(p, s);                    // F5
    ds_cal_per_k = sb_f64_add(x2, x3);                     // F6
    const sb_u64 x4 = sb_f64_add(ds_cal_per_k, kappa);     // F7
    const sb_u64 x5 = sb_f64_from_i64(10 * h);             // F8, exact
    const sb_u64 x6 = sb_f64_div(x5, x4);                  // F9
    tm_c = sb_f64_sub(x6, SB_TM_K0);                       // F10
}
