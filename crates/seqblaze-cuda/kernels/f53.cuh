// F53 on the device: the software float of K3 `duplex(model="pf")` with an integer
// significand. Spec: docs/wiki/gpu/duplex-pf-cuda.md §5 (and duplex-pf.md §3).
// Written from those pages only.
//
// A value is (M, e) with value M * 2^(e - 52): zero is exactly (0, 0); every other value
// has 2^52 <= M < 2^53. Only non-negative values occur. Each operation returns the exact
// result rounded once to 53 significant bits, nearest, ties to even, which is the CPU's
// binary64-significand form bit for bit (duplex-pf.md §3.3). Integers only: 64-bit
// shifts, compares and adds and the integer intrinsic __umul64hi; no floating-point type,
// no inline PTX, no runtime integer division or remainder. The per-call range checks of
// duplex-pf.md §5.4 run on the host, so |e| < 2^60 + 403 for every operand and exponent
// sums and differences never overflow int64; the device checks nothing and raises nothing.
#pragma once
#include "hamming.cuh"

struct SbF53 {
    sb_u64 m;
    sb_i64 e;
};

#define SB_F53_HIDDEN (1ull << 52)
#define SB_F53_TOP (1ull << 53)
// Gaps of at least this many binades leave the larger operand unchanged (§5.3).
#define SB_F53_SHORTCUT 54

__device__ __forceinline__ SbF53 sb_f53_make(sb_u64 m, sb_i64 e) {
    SbF53 out;
    out.m = m;
    out.e = e;
    return out;
}

__device__ __forceinline__ SbF53 sb_f53_zero() { return sb_f53_make(0ull, 0ll); }

// Round to nearest, ties to even (§5.1): the candidate `mp` in [2^52, 2^53), the
// remainder `r` below its last bit and the half-way value `h` of that remainder.
__device__ __forceinline__ SbF53 sb_f53_round(sb_u64 mp, sb_u64 r, sb_u64 h, sb_i64 e) {
    if (r > h || (r == h && (mp & 1ull) != 0ull)) {
        mp += 1ull;
        if (mp == SB_F53_TOP) {
            mp = SB_F53_HIDDEN;
            e += 1;
        }
    }
    return sb_f53_make(mp, e);
}

// x (*) y = RN53(x * y) (§5.2): the exact 106-bit product from its two 64-bit halves.
__device__ __forceinline__ SbF53 sb_f53_mul(const SbF53& x, const SbF53& y) {
    if (x.m == 0ull || y.m == 0ull) {
        return sb_f53_zero();
    }
    const sb_u64 hi = __umul64hi(x.m, y.m);
    const sb_u64 lo = x.m * y.m;
    const sb_i64 e = x.e + y.e;
    if (hi >= (1ull << 41)) {
        // P >= 2^105: M' = floor(P / 2^53), r = P mod 2^53.
        return sb_f53_round((hi << 11) | (lo >> 53), lo & (SB_F53_TOP - 1ull), SB_F53_HIDDEN,
                            e + 1);
    }
    // P < 2^105: M' = floor(P / 2^52), r = P mod 2^52.
    return sb_f53_round((hi << 12) | (lo >> 52), lo & (SB_F53_HIDDEN - 1ull), SB_F53_HIDDEN >> 1,
                        e);
}

// x (+) y = RN53(x + y) (§5.3): the larger exponent first; a gap of 54 or more returns
// the larger operand before any shift; otherwise both significands scaled by 2^10, the
// smaller one shifted right with a sticky bit, one 64-bit sum, then the rounding rule.
__device__ __forceinline__ SbF53 sb_f53_add(const SbF53& p, const SbF53& q) {
    if (p.m == 0ull) {
        return q;
    }
    if (q.m == 0ull) {
        return p;
    }
    const bool order = p.e >= q.e;
    const SbF53 x = order ? p : q;
    const SbF53 y = order ? q : p;
    const sb_u64 d = (sb_u64)(x.e - y.e);
    if (d >= SB_F53_SHORTCUT) {
        return x;
    }
    const sb_u64 a = x.m << 10;
    const sb_u64 v = y.m << 10;
    // jam(v, d): floor(v / 2^d) with bit 0 set when a nonzero bit was shifted out.
    const sb_u64 lost = v & ((1ull << d) - 1ull);
    const sb_u64 b = (v >> d) | (lost != 0ull ? 1ull : 0ull);
    const sb_u64 s = a + b;
    if (s >= (1ull << 63)) {
        return sb_f53_round(s >> 11, s & 2047ull, 1024ull, x.e + 1);
    }
    return sb_f53_round(s >> 10, s & 1023ull, 512ull, x.e);
}

#define SB_F53_OP_ADD 0
#define SB_F53_OP_MUL 1

// Testing kernel (duplex-pf-cuda.md §5.5, §8 item 7): one operation per element on
// canonical operands (the host checks them); reads and writes only its arguments.
extern "C" __global__ void sb_f53_eval(int op, const sb_u64* xm, const sb_i64* xe,
                                       const sb_u64* ym, const sb_i64* ye, sb_u64* om,
                                       sb_i64* oe, sb_u64 n) {
    const sb_u64 stride = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride) {
        const SbF53 x = sb_f53_make(xm[t], xe[t]);
        const SbF53 y = sb_f53_make(ym[t], ye[t]);
        const SbF53 r = op == SB_F53_OP_MUL ? sb_f53_mul(x, y) : sb_f53_add(x, y);
        om[t] = r.m;
        oe[t] = r.e;
    }
}
