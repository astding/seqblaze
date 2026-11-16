// Resident K3 finalize and one-pass exact dense integer row sums.
// Written from duplex-pf.md sections 7/9.4 and the existing core ExactSum.
// Host logarithms are the finite table; device work is integer soft-binary64.
#pragma once
#include "f53.cuh"
#include "softf64.cuh"

struct SbPfFinalize {
    sb_u64 table[128];
    sb_u64 l2, c_t, kappa, b3, b5, b7, loop_max;
};
struct SbPfTerms {
    sb_u64 mantissa, exponent, stride_bytes, count;
};
// One mutable dense integer N*2^beta per row. No fixed exponent anchor or
// exponent-indexed banks. Capacity is a fallible allocation limit, not precision.
struct SbPfExactRow {
    sb_i64 beta;
    sb_u64 used, capacity, offset, partners, needed, status, head;
};
#define SB_PF_ROW_MAX_LIMBS (1ull << 18)
#define SB_PF_ROW_MAX_BITS (64ull * SB_PF_ROW_MAX_LIMBS)
#define SB_PF_ROW_END (~0ull)
#define SB_PF_BAD_TERM 1u
#define SB_PF_NEED_CAPACITY 2u
#define SB_PF_WIDTH 3u

static_assert(sizeof(SbPfFinalize) == 1080, "resident PF finalize ABI");
static_assert(sizeof(SbPfTerms) == 32, "resident PF terms ABI");
static_assert(sizeof(SbPfExactRow) == 64, "resident PF exact row ABI");
#define SB_PF_ABI_OFFSET(type, field, byte) static_assert(offsetof(type, field) == byte, "resident PF " #type "." #field)
SB_PF_ABI_OFFSET(SbPfFinalize, l2, 1024); SB_PF_ABI_OFFSET(SbPfFinalize, loop_max, 1072);
SB_PF_ABI_OFFSET(SbPfTerms, mantissa, 0); SB_PF_ABI_OFFSET(SbPfTerms, exponent, 8);
SB_PF_ABI_OFFSET(SbPfTerms, stride_bytes, 16); SB_PF_ABI_OFFSET(SbPfTerms, count, 24);
SB_PF_ABI_OFFSET(SbPfExactRow, beta, 0); SB_PF_ABI_OFFSET(SbPfExactRow, used, 8);
SB_PF_ABI_OFFSET(SbPfExactRow, capacity, 16); SB_PF_ABI_OFFSET(SbPfExactRow, offset, 24);
SB_PF_ABI_OFFSET(SbPfExactRow, partners, 32); SB_PF_ABI_OFFSET(SbPfExactRow, needed, 40);
SB_PF_ABI_OFFSET(SbPfExactRow, status, 48); SB_PF_ABI_OFFSET(SbPfExactRow, head, 56);
#undef SB_PF_ABI_OFFSET

__device__ __forceinline__ SbF53 sb_pf_resident_term(SbPfTerms input, sb_u64 t) {
    return {*reinterpret_cast<const sb_u64*>(input.mantissa + t * input.stride_bytes),
            *reinterpret_cast<const sb_i64*>(input.exponent + t * input.stride_bytes)};
}
__device__ __forceinline__ bool sb_pf_resident_canonical(SbF53 z) {
    return (z.m == 0 && z.e == 0) ||
           (z.m >= SB_F53_HIDDEN && z.m < SB_F53_TOP &&
            z.e > -((1ll << 60) + 403) && z.e < ((1ll << 60) + 403));
}
// Exact G1 index/c_i/mu construction from M. Every remaining statement is one
// RN operation in the CPU's G2-G8 order; no FMA, ln, exp or device transcendental.
// g(mu, e) of a nonzero z (G1-G8; the finalize multiplies it by c_T).
__device__ __forceinline__ sb_u64 sb_pf_resident_g(SbF53 z, const SbPfFinalize& p) {
    const unsigned i = (unsigned)((z.m - SB_F53_HIDDEN) >> 45);
    const sb_u64 mu = 0x3FF0000000000000ull | (z.m - SB_F53_HIDDEN);
    const sb_u64 c = 0x3FF0000000000000ull | ((sb_u64)i << 45);
    const sb_u64 f = sb_f64_sub(mu, c);
    const sb_u64 d = sb_f64_add(mu, c);
    const sb_u64 s = sb_f64_div(f, d);
    const sb_u64 s2 = sb_f64_mul(s, s);
    const sb_u64 p5 = sb_f64_add(p.b5, sb_f64_mul(s2, p.b7));
    const sb_u64 p3 = sb_f64_add(p.b3, sb_f64_mul(s2, p5));
    const sb_u64 p1 = sb_f64_add(0x4000000000000000ull, sb_f64_mul(s2, p3));
    const sb_u64 q = sb_f64_mul(s, p1);
    const sb_u64 y = sb_f64_add(p.table[i], q);
    return z.e == 0 ? y : sb_f64_add(sb_f64_mul(sb_f64_from_i64(z.e), p.l2), y);
}
__device__ __forceinline__ sb_u64 sb_pf_resident_dg(SbF53 z, const SbPfFinalize& p) {
    if (z.m == 0) return SB_F64_INF;
    const sb_u64 dg = sb_f64_mul(p.c_t, sb_pf_resident_g(z, p));
    return sb_f64_is_zero(dg) ? 0ull : dg;
}
__device__ __forceinline__ sb_i64 sb_pf_resident_key(sb_u64 bits) {
    return bits & SB_F64_SIGN ? -(sb_i64)(bits & ~SB_F64_SIGN) : (sb_i64)bits;
}
__device__ __forceinline__ sb_u64 sb_pf_resident_ru(sb_u64 bits) {
    return bits == 0 || bits == SB_F64_INF ? bits : bits + 1;
}
__device__ __forceinline__ sb_u64 sb_pf_resident_error(sb_u64 dg, sb_u64 n, sb_u64 m,
                                                      const SbPfFinalize& p) {
    if (dg == SB_F64_INF) return 0;
    const sb_u64 l = n < m ? n : m;
    const sb_u64 triangle = (p.loop_max + 1) * (p.loop_max + 2) / 2;
    const sb_u64 cz = l < 2 ? 0 : (l - 1) * (triangle + 6) + n + m + 1;
    const sb_u64 a = sb_pf_resident_ru(sb_f64_mul(p.kappa, sb_f64_from_i64((sb_i64)(cz + 12))));
    const sb_u64 b = sb_pf_resident_ru(sb_f64_mul(0x401D000000000000ull, dg & ~SB_F64_SIGN));
    const sb_u64 eps = sb_pf_resident_ru(sb_f64_add(a, b));
    return sb_f64_mul(eps, 0x3CA0000000000000ull); // exact multiplication by 2^-53
}
extern "C" __global__ void sb_pf_resident_finalize(SbPfFinalize p, SbPfTerms terms,
                                                    SbTables tables, SbSpace space,
                                                    sb_i64* keys, sb_u64* dg, sb_u64* error,
                                                    unsigned* status) {
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < terms.count; t += step) {
        const SbF53 z = sb_pf_resident_term(terms, t);
        if (!sb_pf_resident_canonical(z)) { atomicMax(status, SB_PF_BAD_TERM); continue; }
        sb_u64 i, j;
        sb_decode(space, t, i, j);
        const sb_u64 value = sb_pf_resident_dg(z, p);
        keys[t] = sb_pf_resident_key(value);
        dg[t] = value;
        error[t] = sb_pf_resident_error(value, (sb_u64)sb_a_row(tables, i).length,
                                      (sb_u64)sb_b_row(tables, j).length, p);
    }
}

extern "C" __global__ void sb_pf_rows_clear(SbPfExactRow* rows, sb_u64 count) {
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 r = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; r < count; r += step)
        rows[r].head = SB_PF_ROW_END;
}
// Build chunk-sized linked lists, not a pair-space payload. Link order may vary;
// exact positive integer addition is order-free. Canonical input is already
// distinct, or first[t] supplies the producer's first-occurrence mask.
extern "C" __global__ void sb_pf_rows_link(SbSpace space, SbPfTerms terms, const sb_i64* keys,
                                            const sb_u8* first, int mirror, int has_cutoff,
                                            sb_i64 cutoff, SbPfExactRow* rows, sb_u64 count,
                                            sb_u64* next, unsigned* status) {
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < terms.count; t += step) {
        if ((first != nullptr && first[t] == 0) || (has_cutoff && keys[t] > cutoff)) continue;
        sb_u64 i, j;
        sb_decode(space, t, i, j);
        if (i >= count || (mirror && j >= count)) { atomicMax(status, SB_PF_BAD_TERM); continue; }
        next[2*t] = atomicExch(&rows[i].head, 2*t);
        if (mirror && i != j) next[2*t+1] = atomicExch(&rows[j].head, 2*t+1);
    }
}
__device__ __forceinline__ sb_u64 sb_pf_row_bits(const SbPfExactRow& row, const sb_u64* words) {
    return row.used == 0 ? 0 : 64 * (row.used - 1) + 64 - (sb_u64)sb_clz64(words[row.used - 1]);
}
__device__ __forceinline__ void sb_pf_row_fail(SbPfExactRow& row, unsigned code,
                                              sb_u64 need, unsigned* status) {
    row.status = code;
    row.needed = need;
    atomicMax(status, code);
}
// Same in-place descending lowering as ExactSum::lower; no second copy of N.
__device__ __forceinline__ void sb_pf_row_lower(SbPfExactRow& row, sb_u64* words, sb_u64 shift) {
    if (shift == 0) return;
    const sb_u64 word = shift >> 6, bits = shift & 63, old = row.used;
    const sb_u64 length = old + word + 1;
    for (sb_u64 j = length; j-- > word;) {
        const sb_u64 k = j - word;
        const sb_u64 high = k < old ? words[k] : 0;
        const sb_u64 low = k == 0 || k-1 >= old ? 0 : words[k-1];
        words[j] = bits == 0 ? high : (high << bits) | (low >> (64-bits));
    }
    for (sb_u64 k = 0; k < word; ++k) words[k] = 0;
    row.used = length;
    while (row.used && words[row.used-1] == 0) --row.used;
    row.beta -= (sb_i64)shift;
}
__device__ __forceinline__ void sb_pf_row_add_word(SbPfExactRow& row, sb_u64* words,
                                                  sb_u64 at, sb_u64 value) {
    while (value != 0) {
        while (row.used <= at) words[row.used++] = 0;
        const sb_u64 old = words[at];
        words[at] = old + value;
        value = words[at] < old ? 1ull : 0ull;
        ++at;
    }
}
extern "C" __global__ void sb_pf_rows_add(SbPfTerms terms, SbPfExactRow* rows, sb_u64 count,
                                          const sb_u64* next, sb_u64* arena, unsigned* status) {
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 r = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; r < count; r += step) {
        SbPfExactRow& row = rows[r];
        if (row.status != 0) continue;
        sb_u64* words = arena + row.offset;
        for (sb_u64 node = row.head; node != SB_PF_ROW_END; node = next[node]) {
            if (row.partners == SB_PF_ROW_END) { sb_pf_row_fail(row, SB_PF_BAD_TERM, 0, status); break; }
            ++row.partners;
            const SbF53 z = sb_pf_resident_term(terms, node >> 1);
            if (!sb_pf_resident_canonical(z)) { sb_pf_row_fail(row, SB_PF_BAD_TERM, 0, status); break; }
            if (z.m == 0) continue;
            const sb_i64 e = z.e - 52;
            if (row.used == 0) {
                if (row.capacity == 0) { sb_pf_row_fail(row, SB_PF_NEED_CAPACITY, 1, status); break; }
                row.beta = e; row.used = 1; words[0] = z.m; continue;
            }
            const sb_i64 base = row.beta < e ? row.beta : e;
            const sb_u64 lower = (sb_u64)(row.beta - base);
            const sb_u64 shift = (sb_u64)(e - base);
            const sb_u64 a_bits = sb_pf_row_bits(row, words) + lower;
            const sb_u64 b_bits = 53 + shift;
            const sb_u64 longer = a_bits > b_bits ? a_bits : b_bits;
            if (longer > SB_PF_ROW_MAX_BITS) { sb_pf_row_fail(row, SB_PF_WIDTH, 0, status); break; }
            const sb_u64 need = ((longer + 63) >> 6) + 1;
            if (need > row.capacity) { sb_pf_row_fail(row, SB_PF_NEED_CAPACITY, need, status); break; }
            sb_pf_row_lower(row, words, lower);
            const sb_u64 word = shift >> 6, bits = shift & 63;
            sb_pf_row_add_word(row, words, word, z.m << bits);
            if (bits != 0) sb_pf_row_add_word(row, words, word+1, z.m >> (64-bits));
            if (row.used > SB_PF_ROW_MAX_LIMBS) { sb_pf_row_fail(row, SB_PF_WIDTH, 0, status); break; }
        }
    }
}
__device__ __forceinline__ SbF53 sb_pf_row_round53(const SbPfExactRow& row, const sb_u64* words) {
    const sb_u64 bits = sb_pf_row_bits(row, words);
    if (bits <= 53) return {words[0] << (53-bits), row.beta - (sb_i64)(53-bits) + 52};
    const sb_u64 shift = bits - 53, word = shift >> 6, tail = shift & 63;
    sb_u64 m = words[word] >> tail;
    if (tail != 0 && word+1 < row.used) m |= words[word+1] << (64-tail);
    const sb_u64 below = shift - 1, hword = below >> 6, hbit = below & 63;
    const bool half = ((words[hword] >> hbit) & 1) != 0;
    bool sticky = hbit != 0 && (words[hword] & ((1ull << hbit)-1)) != 0;
    for (sb_u64 k = 0; k < hword; ++k) sticky = sticky || words[k] != 0;
    sb_i64 e = row.beta + (sb_i64)shift + 52;
    if (half && (sticky || (m & 1))) {
        ++m;
        if (m == SB_F53_TOP) { m = SB_F53_HIDDEN; ++e; }
    }
    return {m, e};
}
extern "C" __global__ void sb_pf_rows_finish(SbPfFinalize p, const SbPfExactRow* rows,
                                             sb_u64 count, const sb_u64* arena,
                                             sb_u64* result, unsigned* status) {
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 r = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; r < count; r += step) {
        const SbPfExactRow& row = rows[r];
        if (row.status) { atomicMax(status, (unsigned)row.status); continue; }
        result[r] = row.partners == 0 ? SB_F64_QNAN : row.used == 0 ? SB_F64_INF :
                    sb_pf_resident_dg(sb_pf_row_round53(row, arena + row.offset), p);
    }
}

// Grow only from a true one-pass capacity request, then reset for producer
// replay. No exponent-span prepass, anchored window, or per-pair host channel.
// Exactly one control scalar (total words) is downloaded by the host allocator.
extern "C" __global__ void sb_pf_rows_grow(SbPfExactRow* rows, sb_u64 count,
                                           sb_u64* total, unsigned* status) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    sb_u64 offset = 0;
    *status = 0;
    for (sb_u64 r = 0; r < count; ++r) {
        SbPfExactRow& row = rows[r];
        if (row.status == SB_PF_WIDTH || row.status == SB_PF_BAD_TERM) {
            *status = (unsigned)row.status; return;
        }
        const sb_u64 capacity = row.needed > row.capacity ? row.needed : row.capacity;
        if (capacity > SB_PF_ROW_MAX_LIMBS + 2 || offset > (~0ull)-capacity) {
            *status = SB_PF_BAD_TERM; return;
        }
        row.beta = 0; row.used = 0; row.capacity = capacity; row.offset = offset;
        row.partners = 0; row.needed = 0; row.status = 0; row.head = SB_PF_ROW_END;
        offset += capacity;
    }
    *total = offset;
}
