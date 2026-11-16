// T1 `tm` with the Biopython and primer3 compatibility presets on the device: the
// per-sequence steps of each CPU preset replayed operation by operation in
// soft-binary64 (D55). Spec: docs/wiki/gpu/tm-compat-cuda.md §3-§5 (the formulas of
// thermo/tm-biopython.md §5 and thermo/tm-primer3.md §5). Written from those pages only.
//
// Integers only: every binary64 travels and is stored as its u64 bit pattern, and every
// arithmetic step is one softf64.cuh routine call, in the CPU's order. Every binary64
// constant comes from the pack (QTC3): this header holds no model constant and no
// floating-point literal. No transcendental, no floating-point type.
#pragma once

#include "softf64.cuh"
#include "tm.cuh"
#include "unary.cuh"

// One duplex step of the Biopython pack: (H[k], S[k]) as binary64 bits.
struct SbTmBiopythonStep {
    sb_u64 h;
    sb_u64 s;
};

// The Biopython per-call pack (§3.3), 328 bytes, by value. Mirrors `TmBiopythonPack`
// in src/lib.rs.
struct SbTmBiopythonPack {
    SbTmBiopythonStep step[16];  // index 4·code(X) + code(Y) for the top-strand step XY
    sb_u64 h_at;                 // H_A
    sb_u64 s_at;                 // S_A
    sb_u64 h_gc;                 // H_G
    sb_u64 s_gc;                 // S_G
    sb_u64 salt;                 // c_s
    sb_u64 ln_mon;               // λ_m
    sb_u64 rho;                  // ρ
    sb_u64 kcal_to_cal;          // κ
    sb_u64 k0;                   // K₀
};
static_assert(sizeof(SbTmBiopythonPack) == 328, "SbTmBiopythonPack mirrors TmBiopythonPack");

// One duplex step of the primer3 pack: integer tallies (τ_H, τ_S).
struct SbTmPrimer3Step {
    sb_i64 h;
    sb_i64 s;
};

// The primer3 per-call pack (§3.4), 432 bytes, by value. Mirrors `TmPrimer3Pack` in
// src/lib.rs.
struct SbTmPrimer3Pack {
    SbTmPrimer3Step step[16];  // index 4·code(X) + code(Y) for the top-strand step XY
    SbTmPrimer3Step end_at;    // e_A
    SbTmPrimer3Step end_gc;    // e_G
    SbTmPrimer3Step symmetry;  // e_Y
    sb_i64 max_nn_length;      // M
    sb_u64 salt;               // c_s
    sb_u64 ln_mu;              // λ
    sb_u64 gamma4;             // γ₄
    sb_u64 gamma1;             // γ₁
    sb_u64 dmso_t;             // δ_D
    sb_u64 phi_formamide;      // φ_F
    sb_u64 base;               // B
    sb_u64 sig10;              // Q
    sb_u64 tally_h;            // κ_H
    sb_u64 tally_s;            // κ_S
    sb_u64 formamide_gc;       // a_F
    sb_u64 formamide_c;        // b_F
    sb_u64 long_gc;            // a_G
    sb_u64 long_len;           // a_L
    sb_u64 k0;                 // K₀
};
static_assert(sizeof(SbTmPrimer3Pack) == 432, "SbTmPrimer3Pack mirrors TmPrimer3Pack");

// §4.3: the cutoff (IEEE >= on the canonical bits), the Each mask and the stores of
// slot 0 (tm_c) and `passed`; the presets have no other column.
__device__ __forceinline__ void sb_tm_compat_store(const SbTmOut& out, sb_u64 t, sb_u64 tm_c,
                                                   int has_cutoff, sb_u64 cutoff_bits,
                                                   int mask_each) {
    const bool passed = !has_cutoff || sb_f64_ge(tm_c, cutoff_bits);
    const bool masked = mask_each && !passed;
    if (out.passed != 0) {
        ((sb_u8*)out.passed)[t] = passed ? 1 : 0;
    }
    if (out.tm_c != 0) {
        ((sb_u64*)out.tm_c)[t] = masked ? SB_F64_QNAN : tm_c;
    }
}

// Stage kernel, Biopython preset (§4.1): one thread per sequence i = k0 + t, t in
// [0, n), grid-stride. S3, S1, S4, S5, S6 (5' to 3'), F1-F6, then §4.3.
extern "C" __global__ void sb_tm_biopython(SbUnaryTable tb, sb_u64 k0, sb_u64 n,
                                           SbTmBiopythonPack pack, SbTmOut out, int has_cutoff,
                                           sb_u64 cutoff_bits, int mask_each) {
    const sb_u64 stride = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride) {
        const sb_u64 i = k0 + t;
        const SbSequence seq = sb_unary_row(tb, i);
        const sb_i64 len = seq.length;  // 2 <= len <= 200 after validation
        const unsigned first = len > 0 ? sb_tm_code(seq[0]) : 0u;
        const unsigned last = len > 0 ? sb_tm_code(seq[len - 1]) : 0u;
        // S3: A (0) and T (3) ends.
        const sb_i64 n_at = (sb_i64)(first == 0u || first == 3u) + (sb_i64)(last == 0u || last == 3u);
        const sb_i64 n_gc = 2 - n_at;
        const sb_u64 at = sb_f64_from_i64(n_at);
        const sb_u64 gc = sb_f64_from_i64(n_gc);
        // S1: +0; S4, S5 add to it (never assign: +0 + -0 is +0).
        sb_u64 h = 0, sig = 0;
        h = sb_f64_add(h, sb_f64_mul(pack.h_at, at));    // S4
        sig = sb_f64_add(sig, sb_f64_mul(pack.s_at, at));
        h = sb_f64_add(h, sb_f64_mul(pack.h_gc, gc));    // S5
        sig = sb_f64_add(sig, sb_f64_mul(pack.s_gc, gc));
        // S6: the stacks 5' to 3', in this order.
        unsigned previous = first;
        for (sb_i64 x = 1; x < len; ++x) {
            const unsigned code = sb_tm_code(seq[x]);
            const SbTmBiopythonStep& e = pack.step[(previous << 2) | code];
            h = sb_f64_add(h, e.h);
            sig = sb_f64_add(sig, e.s);
            previous = code;
        }
        const sb_u64 c0 = sb_f64_mul(pack.salt, sb_f64_from_i64(len - 1));  // F1
        const sb_u64 c = sb_f64_mul(c0, pack.ln_mon);
        sig = sb_f64_add(sig, c);                                           // F2
        const sb_u64 a = sb_f64_mul(pack.kcal_to_cal, h);                   // F3
        const sb_u64 den = sb_f64_add(sig, pack.rho);                       // F4
        const sb_u64 q = sb_f64_div(a, den);                                // F5
        const sb_u64 tm_c = sb_f64_canon(sb_f64_sub(q, pack.k0));           // F6
        sb_tm_compat_store(out, t, tm_c, has_cutoff, cutoff_bits, mask_each);
    }
}

// F9 and L1: ε = RN(RN(RN(RN(a_F · nGC) / L) − b_F) · φ_F).
__device__ __forceinline__ sb_u64 sb_tm_primer3_formamide(const SbTmPrimer3Pack& pack,
                                                          sb_u64 n_gc, sb_u64 l) {
    const sb_u64 a = sb_f64_mul(pack.formamide_gc, n_gc);
    const sb_u64 b = sb_f64_div(a, l);
    const sb_u64 c = sb_f64_sub(b, pack.formamide_c);
    return sb_f64_mul(c, pack.phi_formamide);
}

// Stage kernel, primer3 preset (§4.2): one thread per sequence, grid-stride. T4 and the
// integer tallies T1-T2 in one pass, then the long path L1-L6 (L > M) or T3 and F1-F9,
// then §4.3.
extern "C" __global__ void sb_tm_primer3(SbUnaryTable tb, sb_u64 k0, sb_u64 n,
                                         SbTmPrimer3Pack pack, SbTmOut out, int has_cutoff,
                                         sb_u64 cutoff_bits, int mask_each) {
    const sb_u64 stride = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride) {
        const sb_u64 i = k0 + t;
        const SbSequence seq = sb_unary_row(tb, i);
        const sb_i64 len = seq.length;  // 2 <= len <= 200 after validation
        // T4 (C = 1, G = 2) and the step tallies, exact int64 (order-free).
        const unsigned first = len > 0 ? sb_tm_code(seq[0]) : 0u;
        sb_i64 n_gc = (first == 1u || first == 2u) ? 1 : 0;
        sb_i64 th = 0, ts = 0;
        unsigned previous = first;
        for (sb_i64 x = 1; x < len; ++x) {
            const unsigned code = sb_tm_code(seq[x]);
            n_gc += (code == 1u || code == 2u) ? 1 : 0;
            const SbTmPrimer3Step& e = pack.step[(previous << 2) | code];
            th += e.h;
            ts += e.s;
            previous = code;
        }
        const sb_u64 l = sb_f64_from_i64(len);
        const sb_u64 gcount = sb_f64_from_i64(n_gc);
        const sb_u64 eps = sb_tm_primer3_formamide(pack, gcount, l);  // F9 / L1
        sb_u64 tm_c;
        if (len > pack.max_nn_length) {
            // Long path: no tallies, no strand concentration, no self-complementarity.
            const sb_u64 t1 = sb_f64_add(pack.base, eps);                           // L2
            const sb_u64 t2 = sb_f64_add(t1, pack.sig10);                           // L3
            const sb_u64 g = sb_f64_mul(pack.long_gc, sb_f64_div(gcount, l));       // L4
            const sb_u64 t3 = sb_f64_add(t2, g);                                    // L5
            tm_c = sb_f64_canon(sb_f64_sub(t3, sb_f64_div(pack.long_len, l)));      // L6
        } else {
            // T1, T2: the two termini (A/T or G/C), exact.
            const SbTmPrimer3Step& e1 = (first == 0u || first == 3u) ? pack.end_at : pack.end_gc;
            const SbTmPrimer3Step& e2 =
                (previous == 0u || previous == 3u) ? pack.end_at : pack.end_gc;
            th += e1.h + e2.h;
            ts += e1.s + e2.s;
            // T3: s = revcomp(s) for even L (base k pairs with base L + 1 − k).
            bool sc = len > 0 && (len & 1) == 0;
            for (sb_i64 x = 0; sc && x < len / 2; ++x) {
                sc = sb_tm_code(seq[x]) + sb_tm_code(seq[len - 1 - x]) == 3u;
            }
            if (sc) {
                th += pack.symmetry.h;
                ts += pack.symmetry.s;
            }
            const sb_u64 dh = sb_f64_mul(sb_f64_from_i64(th), pack.tally_h);       // F1
            const sb_u64 ds = sb_f64_mul(sb_f64_from_i64(ts), pack.tally_s);       // F2
            const sb_u64 s0 = sb_f64_mul(pack.salt, sb_f64_from_i64(len - 1));     // F3
            const sb_u64 s = sb_f64_mul(s0, pack.ln_mu);
            const sb_u64 ds_salt = sb_f64_add(ds, s);                              // F4
            const sb_u64 den = sb_f64_add(ds_salt, sc ? pack.gamma1 : pack.gamma4);  // F5
            const sb_u64 tk = sb_f64_div(dh, den);                                 // F6
            const sb_u64 tc = sb_f64_sub(tk, pack.k0);                             // F7
            const sb_u64 td = sb_f64_sub(tc, pack.dmso_t);                         // F8
            tm_c = sb_f64_canon(sb_f64_add(td, eps));                              // F9
        }
        sb_tm_compat_store(out, t, tm_c, has_cutoff, cutoff_bits, mask_each);
    }
}
