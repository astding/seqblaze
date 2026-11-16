// Soft-binary64: the shared IEEE 754 binary64 routine (D55) on integer bit patterns.
// Spec: docs/wiki/gpu/tm-cuda.md §4. Written from that page and IEEE Std 754-2019 only.
//
// Every operation returns the binary64 result for roundTiesToEven with gradual
// underflow and no flush-to-zero. (P1) every NaN result is the canonical quiet NaN
// 0x7FF8000000000000; no exception flags are kept. Values are `sb_u64` bit patterns:
// no floating-point type or instruction appears in this header (build.rs rejects any
// `.f64` in the module's PTX, §4.7).
#pragma once

typedef long long sb_i64;
typedef unsigned long long sb_u64;

#define SB_F64_SIGN 0x8000000000000000ull
#define SB_F64_INF 0x7FF0000000000000ull
#define SB_F64_QNAN 0x7FF8000000000000ull
#define SB_F64_FRAC 0x000FFFFFFFFFFFFFull
#define SB_F64_HIDDEN 0x0010000000000000ull

__device__ __forceinline__ unsigned sb_f64_exp(sb_u64 x) { return (unsigned)((x >> 52) & 0x7FFu); }
__device__ __forceinline__ bool sb_f64_is_nan(sb_u64 x) {
    return sb_f64_exp(x) == 0x7FFu && (x & SB_F64_FRAC) != 0;
}
__device__ __forceinline__ bool sb_f64_is_inf(sb_u64 x) {
    return sb_f64_exp(x) == 0x7FFu && (x & SB_F64_FRAC) == 0;
}
__device__ __forceinline__ bool sb_f64_is_zero(sb_u64 x) { return (x & ~SB_F64_SIGN) == 0; }

// clz of a non-zero value (callers never pass 0).
__device__ __forceinline__ int sb_clz64(sb_u64 x) { return __clzll((long long)x); }

// §4.2 unpacking of a finite non-zero x: value = (-1)^sign * m * 2^e.
__device__ __forceinline__ void sb_f64_unpack(sb_u64 x, sb_u64& m, sb_i64& e) {
    const unsigned be = sb_f64_exp(x);
    if (be == 0) {
        m = x & SB_F64_FRAC;
        e = -1074;
    } else {
        m = SB_F64_HIDDEN | (x & SB_F64_FRAC);
        e = (sb_i64)be - 1075;
    }
}

// §4.2 normalised significand: afterwards 2^52 <= m < 2^53.
__device__ __forceinline__ void sb_f64_normalise(sb_u64& m, sb_i64& e) {
    if (m < SB_F64_HIDDEN) {
        const int z = sb_clz64(m) - 11;
        m <<= z;
        e -= z;
    }
}

// §4.2 shift right with jamming: every dropped one-bit is collected into bit 0.
// Shifts by 64 or more are handled without a C++ shift (undefined behaviour).
__device__ __forceinline__ sb_u64 sb_jam(sb_u64 x, sb_i64 d) {
    if (d <= 0) {
        return x;
    }
    if (d >= 64) {
        return x != 0 ? 1ull : 0ull;
    }
    const sb_u64 lost = x & ((1ull << d) - 1ull);
    return (x >> d) | (lost != 0 ? 1ull : 0ull);
}

// §4.3: the binary64 nearest (ties to even) to (-1)^sigma * M * 2^k, 2^63 <= M < 2^64.
__device__ __forceinline__ sb_u64 sb_round_pack(sb_u64 sigma, sb_i64 k, sb_u64 m) {
    const sb_u64 s = sigma << 63;
    sb_i64 e = k + 1086;
    if (e >= 2047) {
        return s | SB_F64_INF;
    }
    if (e < 1) {
        m = sb_jam(m, 1 - e);
        e = 0;
    }
    sb_u64 sig = m >> 11;
    const sb_u64 r = m & 0x7FFull;
    if (r > 0x400ull || (r == 0x400ull && (sig & 1ull) != 0)) {
        sig += 1;
    }
    if (e == 0) {
        return s | sig;  // sig <= 2^52: 2^52 is the smallest normal, 0 a signed zero
    }
    if (sig == (1ull << 53)) {
        sig = SB_F64_HIDDEN;
        e += 1;
        if (e == 2047) {
            return s | SB_F64_INF;
        }
    }
    return s | ((sb_u64)e << 52) | (sig - SB_F64_HIDDEN);
}

// True iff x is normal (biased exponent 1 ... 2046): finite, non-zero, not subnormal.
__device__ __forceinline__ bool sb_f64_is_normal(sb_u64 x) { return sb_f64_exp(x) - 1u < 0x7FEu; }

// §4.4 add of two finite non-zero operands, unpacked: signs, significands and exponents.
__device__ __forceinline__ sb_u64 sb_f64_add_finite(sb_u64 sa, sb_u64 ma, sb_i64 ea, sb_u64 sb,
                                                    sb_u64 mb, sb_i64 eb) {
    ma <<= 10;
    mb <<= 10;
    if (ea < eb || (ea == eb && ma < mb)) {
        const sb_u64 ts = sa, tm = ma;
        const sb_i64 te = ea;
        sa = sb;
        ma = mb;
        ea = eb;
        sb = ts;
        mb = tm;
        eb = te;
    }
    mb = sb_jam(mb, ea - eb);
    sb_u64 total;
    if (sa == sb) {
        total = ma + mb;
    } else {
        total = ma - mb;
        if (total == 0) {
            return 0;
        }
    }
    const int z = sb_clz64(total);
    return sb_round_pack(sa, ea - 10 - z, total << z);
}

// §4.4 add. Two normal operands (the common case) skip the special cases, which
// cannot apply to them, and unpack directly; the arithmetic is the same.
__device__ __forceinline__ sb_u64 sb_f64_add(sb_u64 a, sb_u64 b) {
    if (sb_f64_is_normal(a) && sb_f64_is_normal(b)) {
        return sb_f64_add_finite(a >> 63, SB_F64_HIDDEN | (a & SB_F64_FRAC),
                                 (sb_i64)sb_f64_exp(a) - 1075, b >> 63,
                                 SB_F64_HIDDEN | (b & SB_F64_FRAC), (sb_i64)sb_f64_exp(b) - 1075);
    }
    if (sb_f64_is_nan(a) || sb_f64_is_nan(b)) {
        return SB_F64_QNAN;
    }
    const bool ia = sb_f64_is_inf(a), ib = sb_f64_is_inf(b);
    if (ia && ib) {
        return ((a ^ b) & SB_F64_SIGN) == 0 ? a : SB_F64_QNAN;
    }
    if (ia) {
        return a;
    }
    if (ib) {
        return b;
    }
    const bool za = sb_f64_is_zero(a), zb = sb_f64_is_zero(b);
    if (za && zb) {
        return (a & b & SB_F64_SIGN);  // -0 only if both are -0
    }
    if (za) {
        return b;
    }
    if (zb) {
        return a;
    }
    sb_u64 ma, mb;
    sb_i64 ea, eb;
    sb_f64_unpack(a, ma, ea);
    sb_f64_unpack(b, mb, eb);
    return sb_f64_add_finite(a >> 63, ma, ea, b >> 63, mb, eb);
}

// add(x, +0) without the arithmetic: x itself, except that -0 gives +0
// (roundTiesToEven) and a NaN gives the canonical NaN (P1). Bit-identical to
// sb_f64_add(x, 0) for every x.
__device__ __forceinline__ sb_u64 sb_f64_add_zero(sb_u64 x) {
    return sb_f64_is_nan(x) ? SB_F64_QNAN : (x == SB_F64_SIGN ? 0ull : x);
}

__device__ __forceinline__ sb_u64 sb_f64_sub(sb_u64 a, sb_u64 b) {
    return sb_f64_add(a, b ^ SB_F64_SIGN);
}

// §4.4 mul of normalised significands (2^52 <= m < 2^53): the 128-bit product, rounded.
__device__ __forceinline__ sb_u64 sb_f64_mul_finite(sb_u64 s, sb_u64 ma, sb_i64 ea, sb_u64 mb,
                                                    sb_i64 eb) {
    const sb_u64 lo = ma * mb;
    const sb_u64 hi = __umul64hi(ma, mb);  // 2^40 <= hi < 2^42
    const int z = sb_clz64(hi);
    sb_u64 m = z == 0 ? hi : ((hi << z) | (lo >> (64 - z)));
    const sb_u64 rest = z == 0 ? lo : (lo << z);
    if (rest != 0) {
        m |= 1ull;
    }
    return sb_round_pack(s >> 63, ea + eb + 64 - z, m);
}

// §4.4 mul. Two normal operands skip the special cases and unpack directly, as in add.
__device__ __forceinline__ sb_u64 sb_f64_mul(sb_u64 a, sb_u64 b) {
    const sb_u64 s = (a ^ b) & SB_F64_SIGN;
    if (sb_f64_is_normal(a) && sb_f64_is_normal(b)) {
        return sb_f64_mul_finite(s, SB_F64_HIDDEN | (a & SB_F64_FRAC), (sb_i64)sb_f64_exp(a) - 1075,
                                 SB_F64_HIDDEN | (b & SB_F64_FRAC), (sb_i64)sb_f64_exp(b) - 1075);
    }
    if (sb_f64_is_nan(a) || sb_f64_is_nan(b)) {
        return SB_F64_QNAN;
    }
    const bool ia = sb_f64_is_inf(a), ib = sb_f64_is_inf(b);
    const bool za = sb_f64_is_zero(a), zb = sb_f64_is_zero(b);
    if ((ia && zb) || (za && ib)) {
        return SB_F64_QNAN;
    }
    if (ia || ib) {
        return s | SB_F64_INF;
    }
    if (za || zb) {
        return s;
    }
    sb_u64 ma, mb;
    sb_i64 ea, eb;
    sb_f64_unpack(a, ma, ea);
    sb_f64_unpack(b, mb, eb);
    sb_f64_normalise(ma, ea);
    sb_f64_normalise(mb, eb);
    return sb_f64_mul_finite(s, ma, ea, mb, eb);
}

// Exact division by a small compile-time integer. With a = m * 2^e and
// 2^52 <= m < 2^53, form a normalised 64-bit quotient and jam its remainder.
// Two constant integer divisions replace the restoring loop; sb_round_pack
// still performs the one binary64 rounding (including gradual underflow).
template <unsigned Divisor>
__device__ __forceinline__ sb_u64 sb_f64_div_small(sb_u64 sigma, sb_u64 m, sb_i64 e) {
    static_assert(Divisor >= 2 && Divisor < 65536, "small integer divisor");
    const sb_u64 scaled = m << 11;
    const sb_u64 q = scaled / Divisor;
    const unsigned r = (unsigned)(scaled - q * Divisor);
    const int z = sb_clz64(q);  // 1 <= z <= 16
    const unsigned tail = r << z;  // < 2^32
    const unsigned extra = tail / Divisor;
    sb_u64 significand = (q << z) | extra;
    if (tail - extra * Divisor != 0) {
        significand |= 1ull;
    }
    return sb_round_pack(sigma, e - 11 - z, significand);
}

// Reciprocal division (P2's alternative): T. Möller and T. Granlund, "Improved division
// by invariant integers", IEEE Trans. Comput. 60(2):165-175, 2011, Algorithms 2 and 4.
// v0 of Algorithm 2: floor((2^19 - 3 * 2^8) / d9) for d9 = 256 ... 511.
__device__ const unsigned short sb_rcp_v0[256] = {
    2045, 2037, 2029, 2021, 2013, 2005, 1998, 1990, 1983, 1975, 1968, 1960,
    1953, 1946, 1938, 1931, 1924, 1917, 1910, 1903, 1896, 1889, 1883, 1876,
    1869, 1863, 1856, 1849, 1843, 1836, 1830, 1824, 1817, 1811, 1805, 1799,
    1792, 1786, 1780, 1774, 1768, 1762, 1756, 1750, 1745, 1739, 1733, 1727,
    1722, 1716, 1710, 1705, 1699, 1694, 1688, 1683, 1677, 1672, 1667, 1661,
    1656, 1651, 1646, 1641, 1636, 1630, 1625, 1620, 1615, 1610, 1605, 1600,
    1596, 1591, 1586, 1581, 1576, 1572, 1567, 1562, 1558, 1553, 1548, 1544,
    1539, 1535, 1530, 1526, 1521, 1517, 1513, 1508, 1504, 1500, 1495, 1491,
    1487, 1483, 1478, 1474, 1470, 1466, 1462, 1458, 1454, 1450, 1446, 1442,
    1438, 1434, 1430, 1426, 1422, 1418, 1414, 1411, 1407, 1403, 1399, 1396,
    1392, 1388, 1384, 1381, 1377, 1374, 1370, 1366, 1363, 1359, 1356, 1352,
    1349, 1345, 1342, 1338, 1335, 1332, 1328, 1325, 1322, 1318, 1315, 1312,
    1308, 1305, 1302, 1299, 1295, 1292, 1289, 1286, 1283, 1280, 1276, 1273,
    1270, 1267, 1264, 1261, 1258, 1255, 1252, 1249, 1246, 1243, 1240, 1237,
    1234, 1231, 1228, 1226, 1223, 1220, 1217, 1214, 1211, 1209, 1206, 1203,
    1200, 1197, 1195, 1192, 1189, 1187, 1184, 1181, 1179, 1176, 1173, 1171,
    1168, 1165, 1163, 1160, 1158, 1155, 1153, 1150, 1148, 1145, 1143, 1140,
    1138, 1135, 1133, 1130, 1128, 1125, 1123, 1121, 1118, 1116, 1113, 1111,
    1109, 1106, 1104, 1102, 1099, 1097, 1095, 1092, 1090, 1088, 1086, 1083,
    1081, 1079, 1077, 1074, 1072, 1070, 1068, 1066, 1064, 1061, 1059, 1057,
    1055, 1053, 1051, 1049, 1047, 1044, 1042, 1040, 1038, 1036, 1034, 1032,
    1030, 1028, 1026, 1024,
};

// floor((2^128 - 1) / d) - 2^64 for a normalised divisor (2^63 <= d), Algorithm 2.
__device__ __forceinline__ sb_u64 sb_reciprocal(sb_u64 d) {
    const sb_u64 d0 = d & 1ull;
    const sb_u64 d40 = (d >> 24) + 1ull;
    const sb_u64 d63 = (d >> 1) + d0;
    const sb_u64 v0 = __ldg(&sb_rcp_v0[(d >> 55) - 256u]);
    const sb_u64 v1 = (v0 << 11) - ((v0 * v0 * d40) >> 40) - 1ull;
    const sb_u64 v2 = (v1 << 13) + ((v1 * ((1ull << 60) - v1 * d40)) >> 47);
    const sb_u64 e = ((v2 >> 1) & (0ull - d0)) - v2 * d63;
    const sb_u64 v3 = (__umul64hi(v2, e) >> 1) + (v2 << 31);
    // v3 - floor((v3 + 2^64 + 1) d / 2^64), all modulo 2^64.
    const sb_u64 lo = v3 * d;
    const sb_u64 hi = __umul64hi(v3, d) + (lo + d < lo ? 1ull : 0ull);
    return v3 - hi - d;
}

// floor((u1 2^64 + u0) / d) with its remainder in r, for a normalised d, u1 < d and
// v = sb_reciprocal(d), Algorithm 4.
__device__ __forceinline__ sb_u64 sb_udiv_preinv(sb_u64 u1, sb_u64 u0, sb_u64 d, sb_u64 v,
                                                 sb_u64& r) {
    sb_u64 q0 = v * u1;
    sb_u64 q1 = __umul64hi(v, u1);
    const sb_u64 s0 = q0 + u0;
    q1 += u1 + (s0 < q0 ? 1ull : 0ull) + 1ull;
    q0 = s0;
    r = u0 - q1 * d;
    if (r > q0) {
        q1 -= 1ull;
        r += d;
    }
    if (r >= d) {
        q1 += 1ull;
        r -= d;
    }
    return q1;
}

// §4.4 div of normalised significands. The 64-bit quotient floor(ma 2^63 / mb) with a
// jammed remainder, which the 64-step restoring division (P2) also gives, comes from the
// reciprocal division above (2026-11-05; equal to the restoring loop on 2 x 10^7 random
// and edge significand pairs, and its two algorithms on 2 x 10^7 divisors and quotients).
__device__ __forceinline__ sb_u64 sb_f64_div(sb_u64 a, sb_u64 b) {
    const sb_u64 s = (a ^ b) & SB_F64_SIGN;
    if (sb_f64_is_nan(a) || sb_f64_is_nan(b)) {
        return SB_F64_QNAN;
    }
    const bool ia = sb_f64_is_inf(a), ib = sb_f64_is_inf(b);
    if (ia && ib) {
        return SB_F64_QNAN;
    }
    if (ia) {
        return s | SB_F64_INF;
    }
    if (ib) {
        return s;
    }
    const bool za = sb_f64_is_zero(a), zb = sb_f64_is_zero(b);
    if (za && zb) {
        return SB_F64_QNAN;
    }
    if (zb) {
        return s | SB_F64_INF;
    }
    if (za) {
        return s;
    }
    sb_u64 ma, mb;
    sb_i64 ea, eb;
    sb_f64_unpack(a, ma, ea);
    sb_f64_normalise(ma, ea);
    // F1/F2 and F4 of the thermodynamic finalize. Signs and exceptional
    // operands were handled above; the finite quotient keeps their IEEE bits.
    const sb_u64 divisor = b & ~SB_F64_SIGN;
    if (divisor == 0x4059000000000000ull) {  // 100
        return sb_f64_div_small<100>(s >> 63, ma, ea);
    }
    if (divisor == 0x40B83B0000000000ull) {  // 6203
        return sb_f64_div_small<6203>(s >> 63, ma, ea);
    }
    sb_f64_unpack(b, mb, eb);
    sb_f64_normalise(mb, eb);
    if (ma < mb) {
        ma <<= 1;
        ea -= 1;
    }
    // mb <= ma < 2 mb: with d = mb 2^11 the numerator ma 2^74 has high word ma 2^10 < d.
    const sb_u64 d = mb << 11;
    sb_u64 r;
    sb_u64 q = sb_udiv_preinv(ma << 10, 0ull, d, sb_reciprocal(d), r);
    if (r != 0) {
        q |= 1ull;
    }
    return sb_round_pack(s >> 63, ea - eb - 63, q);
}

// §4.4 from_i64: exact for |n| <= 2^53, correctly rounded beyond.
__device__ __forceinline__ sb_u64 sb_f64_from_i64(sb_i64 n) {
    if (n == 0) {
        return 0;
    }
    const sb_u64 sigma = n < 0 ? 1ull : 0ull;
    // |n| without signed overflow: -2^63 gives 2^63.
    const sb_u64 u = n < 0 ? (sb_u64)(-(n + 1)) + 1ull : (sb_u64)n;
    const int z = sb_clz64(u);
    return sb_round_pack(sigma, -(sb_i64)z, u << z);
}

// §4.4 ge: the IEEE >= predicate (NaN never, -0 == +0).
__device__ __forceinline__ bool sb_f64_ge(sb_u64 a, sb_u64 b) {
    if (sb_f64_is_nan(a) || sb_f64_is_nan(b)) {
        return false;
    }
    if (sb_f64_is_zero(a) && sb_f64_is_zero(b)) {
        return true;
    }
    const sb_u64 ka = (a >> 63) != 0 ? ~a : (a | SB_F64_SIGN);
    const sb_u64 kb = (b >> 63) != 0 ? ~b : (b | SB_F64_SIGN);
    return ka >= kb;
}

// IEEE 754-2019 quiet comparisons and class operations of primer3-cuda.md §5, on the
// order key of `ge`: the complement of a negative x, x with bit 63 set otherwise.
__device__ __forceinline__ sb_u64 sb_f64_key(sb_u64 x) {
    return (x >> 63) != 0 ? ~x : (x | SB_F64_SIGN);
}

// lt: false if either is NaN or both are zeros (either sign); else key(a) < key(b).
__device__ __forceinline__ bool sb_f64_lt(sb_u64 a, sb_u64 b) {
    if (sb_f64_is_nan(a) || sb_f64_is_nan(b)) {
        return false;
    }
    if (sb_f64_is_zero(a) && sb_f64_is_zero(b)) {
        return false;
    }
    return sb_f64_key(a) < sb_f64_key(b);
}

// le: false if either is NaN; true if both are zeros; else key(a) <= key(b).
__device__ __forceinline__ bool sb_f64_le(sb_u64 a, sb_u64 b) {
    if (sb_f64_is_nan(a) || sb_f64_is_nan(b)) {
        return false;
    }
    if (sb_f64_is_zero(a) && sb_f64_is_zero(b)) {
        return true;
    }
    return sb_f64_key(a) <= sb_f64_key(b);
}

__device__ __forceinline__ bool sb_f64_gt(sb_u64 a, sb_u64 b) { return sb_f64_lt(b, a); }

// eq: false if either is NaN; true if both are zeros; else equal bits.
__device__ __forceinline__ bool sb_f64_eq(sb_u64 a, sb_u64 b) {
    if (sb_f64_is_nan(a) || sb_f64_is_nan(b)) {
        return false;
    }
    if (sb_f64_is_zero(a) && sb_f64_is_zero(b)) {
        return true;
    }
    return a == b;
}

// is_finite: the exponent field is not 2047.
__device__ __forceinline__ bool sb_f64_is_finite(sb_u64 x) { return sb_f64_exp(x) != 0x7FFu; }

// abs: the sign bit cleared (exact); a NaN gives the canonical NaN (P1).
__device__ __forceinline__ sb_u64 sb_f64_abs(sb_u64 x) {
    return sb_f64_is_nan(x) ? SB_F64_QNAN : (x & ~SB_F64_SIGN);
}

// §4.4 canon: NaN -> 0x7FF8000000000000, anything else unchanged.
__device__ __forceinline__ sb_u64 sb_f64_canon(sb_u64 x) {
    return sb_f64_is_nan(x) ? SB_F64_QNAN : x;
}

// Operation codes of the testing kernel; must match `F64Op` in src/unary.rs.
#define SB_F64_OP_ADD 0
#define SB_F64_OP_SUB 1
#define SB_F64_OP_MUL 2
#define SB_F64_OP_DIV 3
#define SB_F64_OP_GE 4
#define SB_F64_OP_FROM_I64 5
#define SB_F64_OP_LT 6
#define SB_F64_OP_LE 7
#define SB_F64_OP_GT 8
#define SB_F64_OP_EQ 9
#define SB_F64_OP_IS_FINITE 10
#define SB_F64_OP_ABS 11

// Testing kernel (P5, tm-cuda.md §10 item 11; primer3-cuda.md §5): out[t] = op(a[t], b[t]);
// the comparisons and `is_finite` write 0 or 1; `from_i64` reads a[t] as int64 and the
// unary ops (`from_i64`, `is_finite`, `abs`) ignore b. In every fatbin; only testing
// builds expose it to Python.
extern "C" __global__ void sb_f64_eval(int op, const sb_u64* a, const sb_u64* b, sb_u64* out,
                                       sb_u64 n) {
    const sb_u64 stride = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride) {
        const sb_u64 x = a[t];
        sb_u64 r;
        if (op == SB_F64_OP_FROM_I64) {
            r = sb_f64_from_i64((sb_i64)x);
        } else if (op == SB_F64_OP_IS_FINITE) {
            r = sb_f64_is_finite(x) ? 1ull : 0ull;
        } else if (op == SB_F64_OP_ABS) {
            r = sb_f64_abs(x);
        } else {
            const sb_u64 y = b[t];
            if (op == SB_F64_OP_ADD) {
                r = sb_f64_add(x, y);
            } else if (op == SB_F64_OP_SUB) {
                r = sb_f64_sub(x, y);
            } else if (op == SB_F64_OP_MUL) {
                r = sb_f64_mul(x, y);
            } else if (op == SB_F64_OP_DIV) {
                r = sb_f64_div(x, y);
            } else if (op == SB_F64_OP_LT) {
                r = sb_f64_lt(x, y) ? 1ull : 0ull;
            } else if (op == SB_F64_OP_LE) {
                r = sb_f64_le(x, y) ? 1ull : 0ull;
            } else if (op == SB_F64_OP_GT) {
                r = sb_f64_gt(x, y) ? 1ull : 0ull;
            } else if (op == SB_F64_OP_EQ) {
                r = sb_f64_eq(x, y) ? 1ull : 0ull;
            } else {
                r = sb_f64_ge(x, y) ? 1ull : 0ull;
            }
        }
        out[t] = r;
    }
}
