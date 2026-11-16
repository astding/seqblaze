// T2 `hairpin` and K2 `duplex` with the primer3 preset on the device: the binary64
// replay of the CPU kernels (seqblaze-thermo hairpin.rs and dimer.rs) in soft-binary64
// (D55), one sub-warp group per item, the whole cell table of the item in a global slot.
// Spec: docs/wiki/gpu/primer3-cuda.md (§2 why the bytes cannot differ, §4 data path,
// §5 soft operations, §6 device algorithm), hairpin-primer3.md §5 and dimer-primer3.md
// §6, the shared alignment page §2. Written from those pages and seqblaze's own CPU
// kernels only. Derivation route: a behaviour specification written from a reading of
// primer3-py 2.3.1's source (no code copied; D94), confirmed against the unmodified tool.
//
// Integers only: every binary64 value is a u64 bit pattern and every arithmetic step one
// softf64.cuh routine on the same operand bits, grouped as the CPU groups it; nothing is
// fused or reordered. Every comparison is an IEEE quiet comparison (false with a NaN,
// +0 equal to -0). Every constant comes from the host pack (QPC-6). A cell is evaluated by
// one lane (R1); the cells of one T2 column or K2 row are spread over the lanes, with one
// barrier per column or row (R4); selection, the T2 exterior, both backtraces and the
// reported values run on lane 0 in the CPU's order (QPC-4).
#pragma once

#include "hamming.cuh"
#include "softf64.cuh"
#include "unary.cuh"

// Word offsets of the device tables; must match `device_layout` in
// seqblaze-thermo src/p3files.rs. (ΔH, ΔS) of entry k of a two-word table at
// table + 2k; a bonus at table + 3k as (ΔH, ΔS, presence bits).
#define SB_P3_NN 0u
#define SB_P3_MM 1250u
#define SB_P3_M1 2500u
#define SB_P3_M2 3750u
#define SB_P3_D3 5000u
#define SB_P3_D5 5250u
#define SB_P3_LOOP_IN 5500u
#define SB_P3_LOOP_BU 5562u
#define SB_P3_LOOP_HP 5624u
#define SB_P3_B3 5686u
#define SB_P3_B4 8758u

#define SB_P3_MAX_LOOP 30
#define SB_P3_LOOP_ROWS 30
#define SB_P3_MIN_SPAN 4
#define SB_P3_NO_POISON 256ull
#define SB_P3_INT32_MAX 2147483647

// T2 slot layout (bytes): the codes b[0 .. L+1], the prefix values P(0 .. L), the cells
// C(y, z) at y·(L+1) + z. Must match src/p3.rs.
#define SB_T2_CODES 0ull
#define SB_T2_PREFIX 64ull
#define SB_T2_CELLS 1056ull

// Bit patterns of every binary64 scalar the replay reads; mirrors `P3Scalars` in
// src/lib.rs (and `P3DeviceScalars` in seqblaze-thermo).
struct SbP3Scalars {
    sb_u64 t37, phi, fresh_h, fresh_s, s_cutoff, tolerance, margin, alpha1, at_h, at_s;
    sb_u64 omega_h, omega_s, k0, kcal_div, init_h, init_s, shift;
    sb_u64 sigma, t_u, q4, q1, theta_omega4, theta_omega1;
};

// One launch's parameters; mirrors `SbP3` in src/p3.rs. Every field is 8 bytes.
struct SbP3 {
    sb_u64 tables;  // const u64*: the handle's tables (SB_P3_* layout)
    sb_u64 ends;    // const u64*: end terms (T2: one table; K2: at Q4, then at Q1)
    SbP3Scalars c;
    sb_u64 items;  // const unsigned*: chunk positions of this launch's items
    sb_u64 count;  // entries of `items`
    sb_u64 slots;  // u8*: one slot of `slot` bytes per group
    sb_u64 slot;
    sb_u64 poison;  // testing: byte written over the slot before each item; 256: none
    // Outputs by chunk position; 0 means "not stored".
    sb_u64 key;    // K2: sb_i64* engine key
    sb_u64 found;  // int*: 0 / 1 (T2 masks: INT32_MAX)
    sb_u64 dg;     // T2: u64* dg_kcal bits
    sb_u64 dh;     // u64* dh_kcal bits
    sb_u64 ds;     // u64* ds_cal_per_k bits
    sb_u64 tm;     // u64* tm_c bits
    sb_u64 passed;      // T2: u8* (with a cutoff)
    sb_u64 has_cutoff;  // T2
    sb_u64 cutoff;      // T2: cutoff bits
    sb_u64 mask_each;   // T2: Each with a cutoff masks failing rows
    sb_u64 swap;   // K2: END2 swaps the operands
    sb_u64 end1;   // K2: the selection scans the last row only
};

struct SbHs {
    sb_u64 h;
    sb_u64 s;
};

// ------------------------------------------------------------------ soft operations
//
// Out-of-line wrappers keep the replay's code size bounded; each is the softf64.cuh
// routine itself, so the result bits are the same.

__device__ __noinline__ sb_u64 sb_p3_add(sb_u64 a, sb_u64 b) { return sb_f64_add(a, b); }
__device__ __noinline__ sb_u64 sb_p3_mul(sb_u64 a, sb_u64 b) { return sb_f64_mul(a, b); }
__device__ __noinline__ sb_u64 sb_p3_div(sb_u64 a, sb_u64 b) { return sb_f64_div(a, b); }
__device__ __forceinline__ sb_u64 sb_p3_sub(sb_u64 a, sb_u64 b) {
    return sb_p3_add(a, b ^ SB_F64_SIGN);
}

#define SB_P3_ZERO 0ull                    // +0
#define SB_P3_NEG_ZERO 0x8000000000000000ull  // -0
#define SB_P3_NEG_INF 0xFFF0000000000000ull   // -inf

__device__ __forceinline__ SbHs sb_p3_hs(sb_u64 h, sb_u64 s) {
    SbHs v;
    v.h = h;
    v.s = s;
    return v;
}

__device__ __forceinline__ unsigned sb_p3_idx4(unsigned a, unsigned b, unsigned c, unsigned d) {
    return ((a * 5u + b) * 5u + c) * 5u + d;
}
__device__ __forceinline__ unsigned sb_p3_idx3(unsigned a, unsigned c, unsigned d) {
    return (a * 5u + c) * 5u + d;
}
__device__ __forceinline__ bool sb_p3_wc(unsigned a, unsigned b) {
    return a < 4u && b < 4u && a + b == 3u;
}
// Letter code of an upper-case ACGT byte, 4 for any other (the CPU's map).
__device__ __forceinline__ sb_u8 sb_p3_code(sb_u8 byte) {
    return byte == 'A' ? 0 : (byte == 'C' ? 1 : (byte == 'G' ? 2 : (byte == 'T' ? 3 : 4)));
}

__device__ __forceinline__ SbHs sb_p3_entry(const SbP3& p, unsigned table, unsigned k) {
    const sb_u64* w = (const sb_u64*)p.tables + table + 2u * k;
    return sb_p3_hs(__ldg(w), __ldg(w + 1));
}

__device__ __forceinline__ SbHs sb_p3_end(const sb_u64* ends, unsigned k) {
    return sb_p3_hs(__ldg(ends + 2u * k), __ldg(ends + 2u * k + 1u));
}

__device__ __forceinline__ SbHs sb_p3_omega(const SbP3& p) {
    return sb_p3_hs(p.c.omega_h, p.c.omega_s);
}

// The per-component sum, one rounding per component.
__device__ __forceinline__ SbHs sb_p3_sum(const SbHs& a, const SbHs& b) {
    return sb_p3_hs(sb_p3_add(a.h, b.h), sb_p3_add(a.s, b.s));
}

// Θ(h, s) = fl(fl(h + 0) / fl(fl(s − φ) + 0)), every addition performed.
__device__ __forceinline__ sb_u64 sb_p3_theta(const SbP3& p, const SbHs& v) {
    const sb_u64 num = sb_f64_add_zero(v.h);
    const sb_u64 den = sb_f64_add_zero(sb_p3_sub(v.s, p.c.phi));
    return sb_p3_div(num, den);
}

// G(h, s) = fl(h − fl(T37 · s)).
__device__ __forceinline__ sb_u64 sb_p3_g37(const SbP3& p, const SbHs& v) {
    return sb_p3_sub(v.h, sb_p3_mul(p.c.t37, v.s));
}

// cl(v): F if s < the entropy cutoff, else v.
__device__ __forceinline__ SbHs sb_p3_clamp(const SbP3& p, const SbHs& v) {
    return sb_f64_lt(v.s, p.c.s_cutoff) ? sb_p3_hs(p.c.fresh_h, p.c.fresh_s) : v;
}

// x ≈ y: every component finite and each differs by less than the tolerance.
__device__ __forceinline__ bool sb_p3_approx(const SbP3& p, const SbHs& x, const SbHs& y) {
    return sb_f64_is_finite(x.h) && sb_f64_is_finite(y.h) && sb_f64_is_finite(x.s) &&
           sb_f64_is_finite(y.s) && sb_f64_lt(sb_f64_abs(sb_p3_sub(x.h, y.h)), p.c.tolerance) &&
           sb_f64_lt(sb_f64_abs(sb_p3_sub(x.s, y.s)), p.c.tolerance);
}

// The terminal A·T term A(p, q): (ΔH, ΔS) for A·T and T·A, (0, φ) otherwise.
__device__ __forceinline__ SbHs sb_p3_at(const SbP3& p, unsigned a, unsigned b) {
    if ((a == 0u && b == 3u) || (a == 3u && b == 0u)) {
        return sb_p3_hs(p.c.at_h, p.c.at_s);
    }
    return sb_p3_hs(SB_P3_ZERO, p.c.phi);
}

// The group's slot, filled with the poison byte in testing builds (QPC-16).
__device__ __forceinline__ void sb_p3_poison(const SbP3& p, sb_u8* slot, int me, int g,
                                             sb_u64 bytes) {
#ifdef SB_TESTING
    if (p.poison < SB_P3_NO_POISON) {
        for (sb_u64 x = (sb_u64)me; x < bytes; x += (sb_u64)g) {
            slot[x] = (sb_u8)p.poison;
        }
    }
#else
    (void)p;
    (void)slot;
    (void)me;
    (void)g;
    (void)bytes;
#endif
}

// ------------------------------------------------------------------ T2 (hairpin §5)

struct SbT2 {
    const SbP3* p;
    const sb_u8* b;  // codes, b[0] and b[L + 1] are 4
    SbHs* cell;      // C(y, z) at y * stride + z
    SbHs* pre;       // P(0 .. L)
    int len;
    int stride;
};

__device__ __forceinline__ SbHs sb_t2_c(const SbT2& f, int y, int z) {
    return f.cell[y * f.stride + z];
}

__device__ __forceinline__ unsigned sb_t2_stack_key(const SbT2& f, int y, int z) {
    return sb_p3_idx4(f.b[y], f.b[y + 1], f.b[z], f.b[z - 1]);
}

__device__ __forceinline__ unsigned sb_t2_inner_key(const SbT2& f, int yp, int zp) {
    return sb_p3_idx4(f.b[zp], f.b[zp + 1], f.b[yp], f.b[yp - 1]);
}

// The loop term ℓ of §5.2 (b) for the closing pair (y, z) and inner pair (y′, z′).
__device__ __noinline__ SbHs sb_t2_loop_term(const SbT2& f, int y, int z, int yp, int zp) {
    const SbP3& p = *f.p;
    const int l5 = yp - y - 1;
    const int l3 = z - zp - 1;
    const int n = l5 + l3;
    if (l5 == 0 || l3 == 0) {
        if (n == 1) {
            const SbHs nn = sb_p3_entry(p, SB_P3_NN, sb_p3_idx4(f.b[y], f.b[yp], f.b[z], f.b[zp]));
            return sb_p3_sum(sb_p3_entry(p, SB_P3_LOOP_BU, 1u), nn);
        }
        const SbHs outer = sb_p3_at(p, f.b[y], f.b[z]);
        const SbHs inner = sb_p3_at(p, f.b[yp], f.b[zp]);
        return sb_p3_sum(sb_p3_sum(sb_p3_entry(p, SB_P3_LOOP_BU, (unsigned)n), outer), inner);
    }
    if (l5 == 1 && l3 == 1) {
        return sb_p3_sum(sb_p3_entry(p, SB_P3_MM, sb_t2_stack_key(f, y, z)),
                         sb_p3_entry(p, SB_P3_MM, sb_t2_inner_key(f, yp, zp)));
    }
    // V1a primary: internal-loop closures use M1.
    const SbHs outer = sb_p3_entry(p, SB_P3_M1, sb_t2_stack_key(f, y, z));
    const SbHs inner = sb_p3_entry(p, SB_P3_M1, sb_t2_inner_key(f, yp, zp));
    const int asym = l5 > l3 ? l5 - l3 : l3 - l5;
    const SbHs body =
        sb_p3_sum(sb_p3_sum(sb_p3_entry(p, SB_P3_LOOP_IN, (unsigned)n), outer), inner);
    // The asymmetry term (+0, fl(alpha1 · asym)) added per component.
    return sb_p3_hs(sb_f64_add_zero(body.h),
                    sb_p3_add(body.s, sb_p3_mul(p.c.alpha1, sb_f64_from_i64(asym))));
}

// X = ℓ + C(y′, z′); Ω if its ΔH is not finite.
__device__ __forceinline__ SbHs sb_t2_loop_candidate(const SbT2& f, int y, int z, int yp, int zp) {
    const SbHs x = sb_p3_sum(sb_t2_loop_term(f, y, z, yp, zp), sb_t2_c(f, yp, zp));
    return sb_f64_is_finite(x.h) ? x : sb_p3_omega(*f.p);
}

// The hairpin-loop bonus of the n-nucleotide loop closed at y (n = 3 or 4): the key is
// the n + 2 codes from y; absent when a code is 4.
__device__ __forceinline__ void sb_t2_bonus(const SbT2& f, int y, int n, SbHs& x) {
    const SbP3& p = *f.p;
    unsigned index = 0;
    for (int k = 0; k < n + 2; ++k) {
        const unsigned code = f.b[y + k];
        if (code > 3u) {
            return;
        }
        index = index * 4u + code;
    }
    const sb_u64* w = (const sb_u64*)p.tables + (n == 3 ? SB_P3_B3 : SB_P3_B4) + 3u * index;
    const sb_u64 flags = __ldg(w + 2);
    if ((flags & 1ull) != 0) {
        x.h = sb_p3_add(x.h, __ldg(w));
    }
    if ((flags & 2ull) != 0) {
        x.s = sb_p3_add(x.s, __ldg(w + 1));
    }
}

// X_hp of §5.2 (c) closed by (y, z), validity rule against v.
__device__ __noinline__ SbHs sb_t2_hairpin(const SbT2& f, int y, int z, SbHs v) {
    const SbP3& p = *f.p;
    const int n = z - y - 1;
    SbHs x;
    if (n == 3) {
        x = sb_p3_sum(sb_p3_entry(p, SB_P3_LOOP_HP, 3u), sb_p3_at(p, f.b[y], f.b[z]));
        sb_t2_bonus(f, y, 3, x);
    } else if (n == 4) {
        x = sb_p3_sum(sb_p3_entry(p, SB_P3_LOOP_HP, 4u),
                      sb_p3_entry(p, SB_P3_M2, sb_t2_stack_key(f, y, z)));
        sb_t2_bonus(f, y, 4, x);
    } else {
        const unsigned row = n < SB_P3_LOOP_ROWS ? (unsigned)n : (unsigned)SB_P3_LOOP_ROWS;
        x = sb_p3_sum(sb_p3_entry(p, SB_P3_LOOP_HP, row),
                      sb_p3_entry(p, SB_P3_M2, sb_t2_stack_key(f, y, z)));
    }
    const bool positive = sb_f64_gt(x.h, SB_P3_ZERO) && sb_f64_gt(x.s, SB_P3_ZERO);
    if (!sb_f64_is_finite(x.h) ||
        (positive && (sb_f64_le(v.h, SB_P3_ZERO) || sb_f64_le(v.s, SB_P3_ZERO)))) {
        return sb_p3_omega(p);
    }
    return x;
}

// The cell of the admissible pair (y, z): stages 1–3 of §5.3.
__device__ __noinline__ SbHs sb_t2_cell(const SbT2& f, int y, int z) {
    const SbP3& p = *f.p;
    // Stage 1: the stack on (y + 1, z − 1).
    SbHs v = sb_p3_hs(p.c.fresh_h, p.c.fresh_s);
    const SbHs stack =
        sb_p3_sum(sb_t2_c(f, y + 1, z - 1), sb_p3_entry(p, SB_P3_NN, sb_t2_stack_key(f, y, z)));
    if (sb_f64_gt(sb_p3_theta(p, stack), sb_p3_theta(p, v))) {
        v = sb_p3_clamp(p, stack);
    }
    // Stage 2: internal loops and bulges, n = λ5 + λ3 ascending, then λ5; strict; the
    // 1×1 margin.
    sb_u64 tv = sb_p3_theta(p, v);
    const int span = z - y;
    if (span >= SB_P3_MIN_SPAN + 3) {
        const int top = span - SB_P3_MIN_SPAN - 2;
        const int n_max = top < SB_P3_MAX_LOOP ? top : SB_P3_MAX_LOOP;
        for (int n = 1; n <= n_max; ++n) {
            for (int l5 = 0; l5 <= n; ++l5) {
                const int l3 = n - l5;
                const int yp = y + 1 + l5;
                const int zp = z - 1 - l3;
                if (!sb_p3_wc(f.b[yp], f.b[zp])) {
                    continue;
                }
                const SbHs x = sb_t2_loop_candidate(f, y, z, yp, zp);
                if (!sb_f64_is_finite(x.h)) {
                    continue;
                }
                const sb_u64 tx = sb_p3_theta(p, x);
                const bool one_by_one = l5 == 1 && l3 == 1;
                if (sb_f64_gt(tx, tv) &&
                    (!one_by_one || sb_f64_ge(sb_p3_sub(tx, tv), p.c.margin))) {
                    v = sb_p3_clamp(p, x);
                    tv = sb_p3_theta(p, v);
                }
            }
        }
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

// The exterior options (§5.5): 0 blunt, 1 the 5′ dangle, 2 the 3′ dangle, 3 both sides.
__device__ __forceinline__ int sb_t2_kappa_max(int opt, int rho) {
    const int offset = opt == 0 ? 5 : (opt == 3 ? 7 : 6);
    return rho - offset;  // negative: the option has no candidate at ρ
}

__device__ __forceinline__ void sb_t2_closing(int opt, int kappa, int rho, int& u, int& w) {
    u = (opt == 1 || opt == 3) ? kappa + 2 : kappa + 1;
    w = (opt == 2 || opt == 3) ? rho - 1 : rho;
}

// The helix candidate of an option with prefix value `pre`: blunt fl(fl(P + A) + C), the
// others fl(fl(fl(P + A) + D) + C), per component.
__device__ __noinline__ SbHs sb_t2_helix(const SbT2& f, int opt, int kappa, int rho, SbHs pre) {
    const SbP3& p = *f.p;
    const sb_u8* b = f.b;
    int u, w;
    sb_t2_closing(opt, kappa, rho, u, w);
    const SbHs a = sb_p3_at(p, b[u], b[w]);
    const SbHs c = sb_t2_c(f, u, w);
    const SbHs with_a = sb_p3_sum(pre, a);
    if (opt == 0) {
        return sb_p3_sum(with_a, c);
    }
    SbHs d;
    if (opt == 1) {
        d = sb_p3_entry(p, SB_P3_D5, sb_p3_idx3(b[rho], b[kappa + 2], b[kappa + 1]));
    } else if (opt == 2) {
        d = sb_p3_entry(p, SB_P3_D3, sb_p3_idx3(b[rho - 1], b[kappa + 1], b[rho]));
    } else {
        // V1b primary: the exterior both-sides option uses M2.
        d = sb_p3_entry(p, SB_P3_M2, sb_p3_idx4(b[rho - 1], b[rho], b[kappa + 2], b[kappa + 1]));
    }
    return sb_p3_sum(sb_p3_sum(with_a, d), c);
}

// An option's best valid candidate at ρ: κ ascending, strictly larger Θ. False if none.
__device__ __noinline__ bool sb_t2_option_best(const SbT2& f, int opt, int rho, SbHs& best) {
    const SbP3& p = *f.p;
    const int kmax = sb_t2_kappa_max(opt, rho);
    bool have = false;
    sb_u64 best_theta = SB_P3_NEG_INF;
    for (int kappa = 0; kappa <= kmax; ++kappa) {
        const SbHs pk = f.pre[kappa];
        const SbHs pre =
            sb_f64_ge(sb_p3_theta(p, pk), SB_P3_NEG_ZERO) ? pk : sb_p3_hs(SB_P3_ZERO, SB_P3_ZERO);
        const SbHs x = sb_t2_helix(f, opt, kappa, rho, pre);
        const bool valid = sb_f64_is_finite(x.h) && sb_f64_le(x.h, SB_P3_ZERO) &&
                           sb_f64_le(x.s, SB_P3_ZERO) && sb_f64_gt(x.s, p.c.s_cutoff);
        if (valid) {
            const sb_u64 tx = sb_p3_theta(p, x);
            if (sb_f64_gt(tx, best_theta)) {
                best_theta = tx;
                best = x;
                have = true;
            }
        }
    }
    return have;
}

// The exterior selection (§5.5): P(ρ) for ρ = 2 … L.
__device__ __noinline__ void sb_t2_exterior(const SbT2& f) {
    const SbP3& p = *f.p;
    for (int rho = 2; rho <= f.len; ++rho) {
        const SbHs carry = f.pre[rho - 1];
        bool have = false;
        SbHs winner = carry;
        sb_u64 best = sb_p3_theta(p, carry);
        for (int opt = 0; opt < 4; ++opt) {
            SbHs candidate;
            if (sb_t2_option_best(f, opt, rho, candidate)) {
                const sb_u64 t = sb_p3_theta(p, candidate);
                if (sb_f64_ge(t, best)) {
                    best = t;
                    winner = candidate;
                    have = true;
                }
            }
        }
        const bool accept =
            have && sb_f64_lt(sb_p3_sub(winner.h, sb_p3_mul(p.c.t_u, winner.s)), SB_P3_ZERO);
        f.pre[rho] = accept ? winner : carry;
    }
}

// Record (y, z) and follow its decomposition (§5.6 "inside a pair"); positions marked in
// `paired` (bit k for position k).
__device__ __noinline__ void sb_t2_trace_pair(const SbT2& f, int y, int z,
                                              unsigned long long& paired) {
    const SbP3& p = *f.p;
    for (;;) {
        paired |= (1ull << y) | (1ull << z);
        const SbHs c = sb_t2_c(f, y, z);
        // Row 1: the stack.
        const SbHs stack = sb_p3_sum(sb_t2_c(f, y + 1, z - 1),
                                     sb_p3_entry(p, SB_P3_NN, sb_t2_stack_key(f, y, z)));
        if (sb_p3_approx(p, c, stack)) {
            y += 1;
            z -= 1;
            continue;
        }
        // Row 2: the hairpin loop, validity against the stored cell.
        if (sb_p3_approx(p, c, sb_t2_hairpin(f, y, z, c))) {
            return;
        }
        // Row 3 (V4b primary): the last candidate with Θ ≥ Θ(cell) must match.
        const sb_u64 tc = sb_p3_theta(p, c);
        const int span = z - y;
        if (span < SB_P3_MIN_SPAN + 3) {
            return;
        }
        const int top = span - SB_P3_MIN_SPAN - 2;
        const int n_max = top < SB_P3_MAX_LOOP ? top : SB_P3_MAX_LOOP;
        bool have_last = false;
        SbHs last;
        for (int n = 1; n <= n_max; ++n) {
            for (int l5 = 0; l5 <= n; ++l5) {
                const int yp = y + 1 + l5;
                const int zp = z - 1 - (n - l5);
                if (!sb_p3_wc(f.b[yp], f.b[zp])) {
                    continue;
                }
                const SbHs x = sb_t2_loop_candidate(f, y, z, yp, zp);
                if (sb_f64_is_finite(x.h) && sb_f64_ge(sb_p3_theta(p, x), tc)) {
                    last = x;
                    have_last = true;
                }
            }
        }
        if (!(have_last && sb_p3_approx(p, c, last))) {
            return;
        }
        int next_y = -1, next_z = -1;
        for (int n = 1; n <= n_max && next_y < 0; ++n) {
            for (int l5 = 0; l5 <= n; ++l5) {
                const int yp = y + 1 + l5;
                const int zp = z - 1 - (n - l5);
                if (!sb_p3_wc(f.b[yp], f.b[zp])) {
                    continue;
                }
                const SbHs x = sb_p3_sum(sb_t2_loop_term(f, y, z, yp, zp), sb_t2_c(f, yp, zp));
                if (sb_p3_approx(p, c, x)) {
                    next_y = yp;
                    next_z = zp;
                    break;
                }
            }
        }
        if (next_y < 0) {
            return;
        }
        y = next_y;
        z = next_z;
    }
}

// The exterior backtrace (§5.6), V4a primary.
__device__ __noinline__ void sb_t2_backtrace(const SbT2& f, unsigned long long& paired) {
    const SbP3& p = *f.p;
    int rho = f.len;
    for (;;) {
        while (rho > 0 && sb_p3_approx(p, f.pre[rho], f.pre[rho - 1])) {
            rho -= 1;
        }
        if (rho == 0) {
            break;
        }
        const SbHs target = f.pre[rho];
        int opt = 0;
        for (; opt < 4; ++opt) {
            SbHs best;
            if (sb_t2_option_best(f, opt, rho, best) && sb_p3_approx(p, target, best)) {
                break;
            }
        }
        if (opt == 4) {
            break;
        }
        const int kmax = sb_t2_kappa_max(opt, rho);
        bool matched = false;
        int next = -1;
        for (int kappa = 0; kappa <= kmax; ++kappa) {
            const SbHs without = sb_t2_helix(f, opt, kappa, rho, sb_p3_hs(SB_P3_ZERO, SB_P3_ZERO));
            int found = 0;  // 1: without the prefix, 2: with P(κ)
            if (sb_p3_approx(p, without, target)) {
                found = 1;
            } else if (sb_p3_approx(p, sb_t2_helix(f, opt, kappa, rho, f.pre[kappa]), target)) {
                found = 2;
            }
            if (found != 0) {
                int u, w;
                sb_t2_closing(opt, kappa, rho, u, w);
                sb_t2_trace_pair(f, u, w, paired);
                next = found == 2 ? kappa : -1;
                matched = true;
                break;
            }
        }
        if (matched && next >= 0) {
            rho = next;
        } else {
            break;
        }
    }
}

// One T2 item of G lanes: the fill, then on lane 0 the exterior, the backtrace and the
// fields of row t (§5.7, §6 of the preset page; §6.2 of this kernel's page).
template <int G>
__device__ __forceinline__ void sb_t2_item(const SbP3& p, const sb_u8* seq, int len, sb_u8* slot,
                                           sb_u64 t, int me, unsigned mask) {
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
    // Fill: columns in order; the cells of a column over the lanes; every y < z written.
    for (int z = 1; z <= len; ++z) {
        for (int y = 1 + me; y < z; y += G) {
            SbHs value = omega;
            if (z >= y + SB_P3_MIN_SPAN && sb_p3_wc(codes[y], codes[z])) {
                value = sb_t2_cell(f, y, z);
            }
            f.cell[y * f.stride + z] = value;
        }
        __syncwarp(mask);
    }
    if (me == 0) {
        f.pre[0] = omega;
        f.pre[1] = omega;
        sb_t2_exterior(f);
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
    // The next item's writes follow every read of this one.
    __syncwarp(mask);
}

// The group geometry of a launch: G lanes per group, g-aligned in the warp.
#define SB_P3_GROUP(G)                                                                       \
    const unsigned lane = threadIdx.x & 31u;                                                 \
    const int me = (int)(threadIdx.x & (unsigned)((G) - 1));                                 \
    const unsigned mask =                                                                    \
        (G) == 32 ? 0xffffffffu : ((((1u << (G)) - 1u)) << (lane & ~(unsigned)((G) - 1)));   \
    const sb_u64 per_block = blockDim.x / (G);                                               \
    const sb_u64 group = (sb_u64)blockIdx.x * per_block + threadIdx.x / (G);                 \
    const sb_u64 groups = (sb_u64)gridDim.x * per_block;                                     \
    sb_u8* slot = (sb_u8*)p.slots + group * p.slot;

// T2 stage of one group size: the groups stride over this launch's item list; item t is
// sequence k0 + t of the uploaded table.
template <int G>
__device__ __forceinline__ void sb_t2_body(const SbUnaryTable& tb, sb_u64 k0, const SbP3& p) {
    SB_P3_GROUP(G)
    const sb_u8* data = (const sb_u8*)tb.data;
    const sb_i64* off = (const sb_i64*)tb.off;
    const unsigned* items = (const unsigned*)p.items;
    for (sb_u64 k = group; k < p.count; k += groups) {
        const sb_u64 t = items[k];
        const sb_u64 i = k0 + t;
        const int len = (int)(off[i + 1] - off[i]);
        sb_t2_item<G>(p, data + off[i], len, slot, t, me, mask);
    }
}

#define SB_T2_KERNEL(G)                                                                      \
    extern "C" __global__ void sb_p3_hairpin_g##G(SbUnaryTable tb, sb_u64 k0, SbP3 p) {      \
        sb_t2_body<G>(tb, k0, p);                                                            \
    }
SB_T2_KERNEL(4)
SB_T2_KERNEL(8)
SB_T2_KERNEL(16)
SB_T2_KERNEL(32)

// ------------------------------------------------------------------ K2 (dimer §6)

struct SbK2 {
    const SbP3* p;
    const sb_u8* x;  // first strand codes, x[0] and x[L1 + 1] are 4
    const sb_u8* y;  // second strand reversed (not complemented), y[0] and y[L2 + 1] are 4
    const sb_u64* ends;  // the end-term table of the pair's Q
    SbHs* cell;      // C(u, z) at u * stride + z
    sb_u64 q;
    sb_u64 theta_omega;
    int l1;
    int l2;
    int stride;
};

__device__ __forceinline__ SbHs sb_k2_c(const SbK2& f, int u, int z) {
    return f.cell[u * f.stride + z];
}

// R(u, z) and Lt(u, z) (§6.2).
__device__ __forceinline__ SbHs sb_k2_right(const SbK2& f, int u, int z) {
    return sb_p3_end(f.ends, sb_p3_idx4(f.x[u], f.y[z], f.x[u + 1], f.y[z + 1]));
}
__device__ __forceinline__ SbHs sb_k2_left(const SbK2& f, int u, int z) {
    return sb_p3_end(f.ends, sb_p3_idx4(f.y[z], f.x[u], f.y[z - 1], f.x[u - 1]));
}

// Θ_R(v) = fl(fl(fl(v_H + 200) + R_H) / fl(fl(fl(v_S + (−5.7)) + R_S) + Q)).
__device__ __forceinline__ sb_u64 sb_k2_theta_r(const SbK2& f, const SbHs& v, const SbHs& r) {
    const SbP3& p = *f.p;
    const sb_u64 num = sb_p3_add(sb_p3_add(v.h, p.c.init_h), r.h);
    const sb_u64 den = sb_p3_add(sb_p3_add(sb_p3_add(v.s, p.c.init_s), r.s), f.q);
    return sb_p3_div(num, den);
}

// G_R(v) = fl(fl(v_H + R_H) − fl(T37 · fl(v_S + R_S))).
__device__ __forceinline__ sb_u64 sb_k2_g_r(const SbP3& p, const SbHs& v, const SbHs& r) {
    return sb_p3_sub(sb_p3_add(v.h, r.h), sb_p3_mul(p.c.t37, sb_p3_add(v.s, r.s)));
}

// Γ_R(v), the selection score.
__device__ __forceinline__ sb_u64 sb_k2_gamma(const SbP3& p, const SbHs& v, const SbHs& r) {
    const sb_u64 h = sb_p3_add(sb_p3_add(v.h, sb_p3_add(r.h, p.c.shift)), p.c.init_h);
    const sb_u64 s = sb_p3_add(sb_p3_add(v.s, sb_p3_add(r.s, p.c.shift)), p.c.init_s);
    return sb_p3_sub(h, sb_p3_mul(p.c.t37, s));
}

// The loop candidate of stage 3 from the predecessor (u − α − 1, z − β − 1) holding
// `pred` (not Ω); guards included, no clamp.
__device__ __noinline__ SbHs sb_k2_loop_candidate(const SbK2& f, int u, int z, int alpha,
                                                  int beta, SbHs pred) {
    const SbP3& p = *f.p;
    const sb_u8* x = f.x;
    const sb_u8* y = f.y;
    const int up = u - alpha - 1;
    const int zp = z - beta - 1;
    const int ell = alpha + beta;
    SbHs value;
    if (alpha == 0 || beta == 0) {
        if (ell == 1) {
            // Bulge of one: the stack across the bulge, guarded on the increment.
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
                                    sb_p3_at(p, x[u], y[z])),
                          pred);
    } else if (alpha == 1 && beta == 1) {
        const SbHs outer =
            sb_p3_entry(p, SB_P3_MM, sb_p3_idx4(x[up], x[up + 1], y[zp], y[zp + 1]));
        const SbHs inner = sb_p3_entry(p, SB_P3_MM, sb_p3_idx4(y[z], y[z - 1], x[u], x[u - 1]));
        value = sb_p3_sum(sb_p3_sum(outer, inner), pred);
    } else {
        // V1a primary: internal-loop closures use M1.
        const SbHs outer =
            sb_p3_entry(p, SB_P3_M1, sb_p3_idx4(x[up], x[up + 1], y[zp], y[zp + 1]));
        const SbHs inner = sb_p3_entry(p, SB_P3_M1, sb_p3_idx4(y[z], y[z - 1], x[u], x[u - 1]));
        const int diff = alpha > beta ? alpha - beta : beta - alpha;
        const sb_u64 asym = sb_p3_mul(p.c.alpha1, sb_f64_from_i64(diff));
        const SbHs lp = sb_p3_entry(p, SB_P3_LOOP_IN, (unsigned)ell);
        value.h = sb_p3_add(
            sb_f64_add_zero(sb_p3_add(sb_p3_add(lp.h, outer.h), inner.h)), pred.h);
        value.s = sb_p3_add(sb_p3_add(sb_p3_add(sb_p3_add(lp.s, outer.s), inner.s), asym), pred.s);
    }
    if (!sb_f64_is_finite(value.h) ||
        (sb_f64_gt(value.h, SB_P3_ZERO) && sb_f64_gt(value.s, SB_P3_ZERO))) {
        return sb_p3_omega(p);
    }
    return value;
}

// C(u, z) of an admissible cell: stages 1–3 of §6.3.
__device__ __noinline__ SbHs sb_k2_cell(const SbK2& f, int u, int z) {
    const SbP3& p = *f.p;
    const SbHs r = sb_k2_right(f, u, z);
    // Stage 1: initiation.
    SbHs v = sb_k2_left(f, u, z);
    if (u > 1 && z > 1) {
        // Stage 2: the stack, available iff the diagonal predecessor's value is not Ω and
        // the NN entry is finite; else Θ_Ω of the pair's Q.
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
        // Stage 3: loops on G_R, strict, against the current value; ℓ ascending, then α.
        sb_u64 gv = sb_k2_g_r(p, v, r);
        const int amax = u - 2;
        const int bmax = z - 2;
        const int lsum = amax + bmax;
        const int lmax = lsum < SB_P3_MAX_LOOP ? lsum : SB_P3_MAX_LOOP;
        for (int ell = 1; ell <= lmax; ++ell) {
            const int lo = ell > bmax ? ell - bmax : 0;
            const int hi = ell < amax ? ell : amax;
            for (int alpha = lo; alpha <= hi; ++alpha) {
                const int beta = ell - alpha;
                const SbHs pc = sb_k2_c(f, u - alpha - 1, z - beta - 1);
                if (!sb_f64_is_finite(pc.h)) {
                    continue;
                }
                const SbHs x = sb_k2_loop_candidate(f, u, z, alpha, beta, pc);
                if (sb_f64_is_finite(x.h)) {
                    const sb_u64 gx = sb_k2_g_r(p, x, r);
                    if (sb_f64_lt(gx, gv)) {
                        v = sb_p3_clamp(p, x);
                        gv = sb_k2_g_r(p, v, r);
                    }
                }
            }
        }
    }
    return v;
}

// The backtrace from (u, z) (§6.5): the number of cells on the path, the reported cell
// included.
__device__ __noinline__ sb_i64 sb_k2_backtrace(const SbK2& f, int u, int z) {
    const SbP3& p = *f.p;
    sb_i64 n_bp = 0;
    for (;;) {
        n_bp += 1;
        const SbHs c = sb_k2_c(f, u, z);
        // Row 1: the duplex starts here.
        if (sb_p3_approx(p, c, sb_k2_left(f, u, z))) {
            return n_bp;
        }
        // Row 2: the stack.
        if (u > 1 && z > 1) {
            const SbHs pred = sb_k2_c(f, u - 1, z - 1);
            if (sb_f64_is_finite(pred.h)) {
                const SbHs nn =
                    sb_p3_entry(p, SB_P3_NN, sb_p3_idx4(f.x[u - 1], f.x[u], f.y[z - 1], f.y[z]));
                if (sb_p3_approx(p, c, sb_p3_sum(pred, nn))) {
                    u -= 1;
                    z -= 1;
                    continue;
                }
            }
        }
        // Row 3: the first loop predecessor in stage-3 order whose candidate matches.
        if (u < 2 || z < 2) {
            return n_bp;
        }
        const int amax = u - 2;
        const int bmax = z - 2;
        const int lsum = amax + bmax;
        const int lmax = lsum < SB_P3_MAX_LOOP ? lsum : SB_P3_MAX_LOOP;
        int next_u = -1, next_z = -1;
        for (int ell = 1; ell <= lmax && next_u < 0; ++ell) {
            const int lo = ell > bmax ? ell - bmax : 0;
            const int hi = ell < amax ? ell : amax;
            for (int alpha = lo; alpha <= hi; ++alpha) {
                const int beta = ell - alpha;
                const SbHs pc = sb_k2_c(f, u - alpha - 1, z - beta - 1);
                if (!sb_f64_is_finite(pc.h)) {
                    continue;
                }
                if (sb_p3_approx(p, c, sb_k2_loop_candidate(f, u, z, alpha, beta, pc))) {
                    next_u = u - alpha - 1;
                    next_z = z - beta - 1;
                    break;
                }
            }
        }
        if (next_u < 0) {
            return n_bp;  // Row 4: stop.
        }
        u = next_u;
        z = next_z;
    }
}

// The K3 bit key of a binary64 value: the bits if the sign is clear, else the negated
// magnitude bits (−0 maps to 0).
__device__ __forceinline__ sb_i64 sb_p3_key(sb_u64 bits) {
    return (bits >> 63) == 0 ? (sb_i64)bits : -(sb_i64)(bits & 0x7FFFFFFFFFFFFFFFull);
}

// Whether the coded strand s[1 .. len] is self-complementary: even length and every base
// of the first half pairs with its mirror base (the lanes share the comparisons).
__device__ __forceinline__ bool sb_k2_self_compl(const sb_u8* s, int len, int me, int g,
                                                 unsigned mask) {
    bool ok = len % 2 == 0;
    for (int i = 1 + me; ok && i <= len / 2; i += g) {
        ok = sb_p3_wc(s[i], s[len + 1 - i]);
    }
    return __all_sync(mask, ok) != 0;
}

// One K2 pair of G lanes: operands a (length la) and b (lb) as given; END2 swaps them.
template <int G>
__device__ __forceinline__ void sb_k2_item(const SbP3& p, const sb_u8* a, int la, const sb_u8* b,
                                           int lb, sb_u8* slot, sb_u64 t, int me,
                                           unsigned mask) {
    if (p.swap != 0) {
        const sb_u8* ts = a;
        a = b;
        b = ts;
        const int tl = la;
        la = lb;
        lb = tl;
    }
    const int l1 = la;
    const int l2 = lb;
    const sb_u64 head = ((sb_u64)(l1 + l2 + 4) + 15ull) / 16ull * 16ull;
    sb_p3_poison(p, slot, me, G, head + (sb_u64)(l1 + 1) * (l2 + 1) * sizeof(SbHs));
    __syncwarp(mask);
    sb_u8* x = slot;
    sb_u8* y = slot + (l1 + 2);
    for (int k = me; k <= l1 + 1; k += G) {
        x[k] = (k == 0 || k == l1 + 1) ? (sb_u8)4 : sb_p3_code(a[k - 1]);
    }
    for (int k = me; k <= l2 + 1; k += G) {
        y[k] = (k == 0 || k == l2 + 1) ? (sb_u8)4 : sb_p3_code(b[l2 - k]);
    }
    __syncwarp(mask);
    const bool both =
        sb_k2_self_compl(x, l1, me, G, mask) && sb_k2_self_compl(y, l2, me, G, mask);
    SbK2 f;
    f.p = &p;
    f.x = x;
    f.y = y;
    f.ends = (const sb_u64*)p.ends + (both ? 1250u : 0u);
    f.cell = (SbHs*)(slot + head);
    f.q = both ? p.c.q1 : p.c.q4;
    f.theta_omega = both ? p.c.theta_omega1 : p.c.theta_omega4;
    f.l1 = l1;
    f.l2 = l2;
    f.stride = l2 + 1;
    const SbHs omega = sb_p3_omega(p);
    // Fill: rows in order. Cells that cannot pair are Ω. The k-th pairing cell of a row
    // (k from 0) goes to lane k mod G, and each lane finds its next one before the
    // evaluation, so the lanes of a warp evaluate pairing cells together.
    for (int u = 1; u <= l1; ++u) {
        for (int z = 1 + me; z <= l2; z += G) {
            if (!sb_p3_wc(x[u], y[z])) {
                f.cell[u * f.stride + z] = omega;
            }
        }
        int z = 1;
        int k = 0;  // pairing cells before z
        for (;;) {
            while (z <= l2) {
                if (sb_p3_wc(x[u], y[z]) && k++ % G == me) {
                    break;
                }
                ++z;
            }
            if (z > l2) {
                break;
            }
            f.cell[u * f.stride + z] = sb_k2_cell(f, u, z);
            ++z;
        }
        __syncwarp(mask);
    }
    if (me == 0) {
        // Selection (§6.4): the first strict minimum of Γ over the scanned cells (every
        // cell, or the last row for END1), from (1, 1) with +∞.
        int bu = 1, bz = 1;
        sb_u64 score = SB_F64_INF;
        for (int u = p.end1 != 0 ? l1 : 1; u <= l1; ++u) {
            for (int z = 1; z <= l2; ++z) {
                const SbHs c = sb_k2_c(f, u, z);
                if (!sb_f64_is_finite(c.h)) {
                    continue;
                }
                const sb_u64 g = sb_k2_gamma(p, c, sb_k2_right(f, u, z));
                if (sb_f64_lt(g, score)) {
                    score = g;
                    bu = u;
                    bz = z;
                }
            }
        }
        bool found = false;
        sb_u64 dg = 0, dh = 0, ds = 0, tm = 0;
        if (sb_p3_wc(x[bu], y[bz])) {
            found = true;
            const sb_i64 n_bp = sb_k2_backtrace(f, bu, bz);
            const sb_u64 nu = sb_f64_from_i64(n_bp - 1);
            const SbHs c = sb_k2_c(f, bu, bz);
            const SbHs r = sb_k2_right(f, bu, bz);
            const sb_u64 h = sb_p3_add(sb_p3_add(c.h, r.h), p.c.init_h);
            const sb_u64 s = sb_p3_add(sb_p3_add(sb_p3_add(c.s, r.s), p.c.init_s),
                                       sb_p3_mul(nu, p.c.sigma));
            const sb_u64 g = sb_p3_sub(h, sb_p3_mul(p.c.t_u, s));
            const sb_u64 temp = sb_p3_sub(sb_p3_div(h, sb_p3_add(s, f.q)), p.c.k0);
            dg = sb_p3_div(g, p.c.kcal_div);
            dh = sb_p3_div(h, p.c.kcal_div);
            ds = s;
            tm = temp;
        }
        ((sb_i64*)p.key)[t] = sb_p3_key(dg);
        if (p.found != 0) {
            ((int*)p.found)[t] = found ? 1 : 0;
            ((sb_u64*)p.dh)[t] = sb_f64_canon(dh);
            ((sb_u64*)p.ds)[t] = sb_f64_canon(ds);
            ((sb_u64*)p.tm)[t] = sb_f64_canon(tm);
        }
    }
    __syncwarp(mask);
}

// K2 stage of one group size: the groups stride over this launch's item list; item t of
// the chunk is the pair sb_decode(s, t).
template <int G>
__device__ __forceinline__ void sb_k2_body(const SbTables& tb, const SbSpace& s, const SbP3& p) {
    SB_P3_GROUP(G)
    const sb_i64* a_off = (const sb_i64*)tb.a_off;
    const sb_i64* b_off = (const sb_i64*)tb.b_off;
    const unsigned* items = (const unsigned*)p.items;
    for (sb_u64 k = group; k < p.count; k += groups) {
        const sb_u64 t = items[k];
        sb_u64 i, j;
        sb_decode(s, t, i, j);
        const sb_u8* a = (const sb_u8*)tb.a_data + a_off[i];
        const sb_u8* b = (const sb_u8*)tb.b_data + b_off[j];
        const int la = (int)(a_off[i + 1] - a_off[i]);
        const int lb = (int)(b_off[j + 1] - b_off[j]);
        sb_k2_item<G>(p, a, la, b, lb, slot, t, me, mask);
    }
}

#define SB_K2_KERNEL(G)                                                                      \
    extern "C" __global__ void sb_p3_dimer_g##G(SbTables tb, SbSpace s, SbP3 p) {            \
        sb_k2_body<G>(tb, s, p);                                                             \
    }
SB_K2_KERNEL(4)
SB_K2_KERNEL(8)
SB_K2_KERNEL(16)
SB_K2_KERNEL(32)

// ------------------------------------------------------------------ int64 key sinks
//
// The int32 sinks of hamming.cuh on int64 keys (primer3-cuda.md §7): Each, Hits and
// RowStats. Integers and integer atomics only.

__device__ __forceinline__ bool sb_pass64(sb_i64 d, int has_cutoff, sb_i64 cutoff) {
    return !has_cutoff || d <= cutoff;
}

// Each: passed = d <= c; a failing d becomes INT64_MAX.
extern "C" __global__ void sb_each64(sb_i64* dist, sb_u8* passed, sb_u64 n, int has_cutoff,
                                     sb_i64 cutoff) {
    const sb_u64 stride = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride) {
        const sb_i64 d = dist[t];
        const bool ok = sb_pass64(d, has_cutoff, cutoff);
        dist[t] = ok ? d : 9223372036854775807ll;
        if (passed != 0) {
            passed[t] = ok ? 1 : 0;
        }
    }
}

extern "C" __global__ void sb_hits_count64(const sb_i64* dist, sb_u64 n, int has_cutoff,
                                           sb_i64 cutoff, sb_u64* block_counts) {
    __shared__ unsigned warp_counts[32];
    const sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x;
    const bool ok = t < n && sb_pass64(dist[t], has_cutoff, cutoff);
    const unsigned ballot = __ballot_sync(0xffffffffu, ok);
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

extern "C" __global__ void sb_hits_write64(const sb_i64* dist, sb_u64 n, int has_cutoff,
                                           sb_i64 cutoff, const sb_u64* block_offsets, SbSpace s,
                                           sb_i64* out_i, sb_i64* out_j, sb_i64* out_d) {
    __shared__ unsigned warp_base[32];
    const sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x;
    const bool ok = t < n && sb_pass64(dist[t], has_cutoff, cutoff);
    const unsigned ballot = __ballot_sync(0xffffffffu, ok);
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
    if (ok) {
        const unsigned below = ballot & ((1u << lane) - 1u);
        const sb_u64 pos = block_offsets[blockIdx.x] + warp_base[warp] + (unsigned)__popc(below);
        sb_u64 i, j;
        sb_decode(s, t, i, j);
        out_i[pos] = (sb_i64)i;
        out_j[pos] = (sb_i64)j;
        out_d[pos] = dist[t];
    }
}

// RowStats sweep 1 on int64 keys: best key and participating-partner count per row.
extern "C" __global__ void sb_rowstats_min64(const sb_i64* dist, SbSpace s, int has_cutoff,
                                             sb_i64 cutoff, SbRows r) {
    const sb_u64 stride = (sb_u64)gridDim.x * blockDim.x;
    long long* best = (long long*)r.best;
    sb_u64* count = (sb_u64*)r.count;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < s.n; t += stride) {
        const sb_i64 d = dist[t];
        if (!sb_pass64(d, has_cutoff, cutoff)) {
            continue;
        }
        sb_u64 i, j;
        sb_decode(s, t, i, j);
        if (!sb_participates(r, s, t)) {
            continue;
        }
        atomicMin(&best[i], (long long)d);
        atomicAdd(&count[i], 1ull);
        if (r.mirror && i != j) {
            atomicMin(&best[j], (long long)d);
            atomicAdd(&count[j], 1ull);
        }
    }
}

// RowStats sweep 2 on int64 keys: ties at the final best and the smallest partner.
extern "C" __global__ void sb_rowstats_arg64(const sb_i64* dist, SbSpace s, int has_cutoff,
                                             sb_i64 cutoff, SbRows r) {
    const sb_u64 stride = (sb_u64)gridDim.x * blockDim.x;
    const sb_i64* best = (const sb_i64*)r.best;
    sb_u64* nbest = (sb_u64*)r.nbest;
    sb_u64* arg = (sb_u64*)r.arg;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < s.n; t += stride) {
        const sb_i64 d = dist[t];
        if (!sb_pass64(d, has_cutoff, cutoff)) {
            continue;
        }
        sb_u64 i, j;
        sb_decode(s, t, i, j);
        if (!sb_participates(r, s, t)) {
            continue;
        }
        if (d == best[i]) {
            atomicAdd(&nbest[i], 1ull);
            atomicMin(&arg[i], j);
        }
        if (r.mirror && i != j && d == best[j]) {
            atomicAdd(&nbest[j], 1ull);
            atomicMin(&arg[j], i);
        }
    }
}

// Host results of routed K2 pairs into the chunk's buffers: keys and, in a detail
// pass, found and the three field bit patterns (`f`: four words per entry, found first).
extern "C" __global__ void sb_p3_scatter(const sb_u64* pos, const sb_i64* d, const sb_u64* f,
                                         sb_u64 count, sb_i64* dist, int* found, sb_u64* dh,
                                         sb_u64* ds, sb_u64* tm) {
    const sb_u64 stride = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 k = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; k < count; k += stride) {
        const sb_u64 t = pos[k];
        dist[t] = d[k];
        if (f != 0 && found != 0) {
            found[t] = (int)f[4 * k];
            dh[t] = f[4 * k + 1];
            ds[t] = f[4 * k + 2];
            tm[t] = f[4 * k + 3];
        }
    }
}
