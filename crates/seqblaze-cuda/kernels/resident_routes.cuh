// Native K1/K3 continuation for device-resident inputs. The original cell
// helpers are shared unchanged: K1 keeps (D,H,G,Q), K3 keeps ZA/ZO and E(j).
// One launch advances exactly one row; stream ordering separates rows. There
// is no host sequence access, host result patch, or whole-pair watchdog loop.
#pragma once
#include "duplex.cuh"
#include "duplex_pf.cuh"

struct SbResidentResume {
    sb_u64 positions;  // chunk-relative work IDs compacted on the device
    sb_u64 first;      // first compacted work ID of this bounded batch
    sb_u64 n;          // at most eight live continuation slots
    sb_u64 codes;      // u8[n][q]: K1 q = SbDup::codes (400, or 800 with ρ), K3 400
    sb_u64 band;       // independently retained global ring slots
    sb_u64 slot;       // bytes per ring slot (16-byte aligned)
    sb_u64 meta;       // u64[n][6]: n,m,R,(K1 identical / K3 cur),sigma,swap
    sb_u64 best;       // K1: SbDupKey[n][32], K3: unused
    sb_u64 row;        // this launch's one ascending DP row
};
static_assert(sizeof(SbResidentResume) == 72, "resident continuation ABI");

extern "C" __global__ void sb_resident_route_flags_dup(
    SbTables tb, SbSpace s, SbDup p, sb_u64* flags) {
    const sb_u64 stride = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < s.n; t += stride) {
        flags[t] = sb_dup_routed(tb, s, p, t) ? 1ull : 0ull;
    }
}
extern "C" __global__ void sb_resident_route_flags_pf(
    SbTables tb, SbSpace s, SbPf p, sb_u64* flags) {
    const sb_u64 stride = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < s.n; t += stride) {
        flags[t] = sb_pf_routed(tb, s, p, t) ? 1ull : 0ull;
    }
}
extern "C" __global__ void sb_resident_route_positions(
    const sb_u64* flags, const sb_u64* prefix, sb_u64 n, sb_u64* positions) {
    const sb_u64 stride = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < n; t += stride) {
        if (flags[t] != 0) positions[prefix[t]] = t;
    }
}

// A launch has one warp per slot, blockDim=32, gridDim=batch size. Every live
// warp reaches each warp barrier; inactive slots do not exist in the launch.
__device__ __forceinline__ sb_u64 sb_resident_work(const SbResidentResume& r) {
    return ((const sb_u64*)r.positions)[r.first + blockIdx.x];
}

extern "C" __global__ void sb_resident_dup_init(
    SbTables tb, SbSpace s, SbDup p, SbResidentResume r) {
    const int lane = (int)threadIdx.x;
    const sb_u64 slot = blockIdx.x;
    if (slot >= r.n) return;
#ifdef SB_TESTING
    if (lane == 0) sb_resident_poison_count(p.resident_stats, 1, 0, 1);
#endif
    sb_u64 i, j, n64, m64;
    sb_decode(s, sb_resident_work(r), i, j);
    sb_dup_lengths(tb, i, j, n64, m64);
    const int n = (int)n64, m = (int)m64;
#ifdef SB_TESTING
    if (p.poison < 256) {
        const sb_u64 rows = n64 < p.lam + 2 ? n64 : p.lam + 2;
        const sb_u64 total = rows * m64 * SbDupFull::BYTES;
        sb_u8* band = (sb_u8*)r.band + slot * r.slot;
        for (sb_u64 q = (sb_u64)lane; q < total; q += 32) band[q] = (sb_u8)p.poison;
        __syncwarp(0xffffffffu);
        if (lane == 0) {
            sb_resident_poison_count(p.resident_stats, 1, 1, 1);
            sb_resident_poison_count(p.resident_stats, 1, 2, total);
        }
    }
#endif
    sb_u8* ca = (sb_u8*)r.codes + slot * p.codes;
    sb_u8* cb = ca + SB_DUP_MAX_LENGTH;
    const SbSequence a = sb_a_row(tb, i), b = sb_b_row(tb, j);
    for (int q = lane; q < n; q += 32) ca[q] = sb_dup_code(a[q]);
    for (int q = lane; q < m; q += 32) cb[q] = sb_dup_code(b[q]);
    __syncwarp(0xffffffffu);
    if (p.has_states != 0) {
        // ρ in the slot's staged area, kept across the one-row launches (§5.4.4).
        sb_dup_runs(ca, n, ca + SB_DUP_RHO, lane, 32);
        sb_dup_runs(cb, m, ca + SB_DUP_RHO + SB_DUP_MAX_LENGTH, lane, 32);
        __syncwarp(0xffffffffu);
    }
    bool same = n == m, palindrome = true;
    for (int q = lane; q < n; q += 32) {
        same = same && q < m && ca[q] == cb[q];
        palindrome = palindrome && ca[q] + ca[n - 1 - q] == 3;
    }
    const bool identical = __all_sync(0xffffffffu, same) != 0;
    const bool sym = identical && __all_sync(0xffffffffu, palindrome) != 0;
    if (lane == 0) {
        sb_u64* meta = (sb_u64*)r.meta + slot * 6;
        meta[0] = n64; meta[1] = m64;
        meta[2] = n64 < p.lam + 2 ? n64 : p.lam + 2;
        meta[3] = identical; meta[4] = sym; meta[5] = 0;
    }
    ((SbDupKey*)r.best)[slot * 32 + lane] = SbDupFull::inf();
}

extern "C" __global__ void sb_resident_dup_step(SbDup p, SbResidentResume r) {
    const int lane = (int)threadIdx.x;
    const sb_u64 slot = blockIdx.x;
    if (slot >= r.n) return;
    const sb_u64* meta = (const sb_u64*)r.meta + slot * 6;
    const int n = (int)meta[0], m = (int)meta[1], rows = (int)meta[2];
    const int row = (int)r.row;
    if (row >= n) return;
    const sb_u8* ca = (const sb_u8*)r.codes + slot * p.codes;
    const sb_u8* cb = ca + SB_DUP_MAX_LENGTH;
    SbDupKey* band = (SbDupKey*)((sb_u8*)r.band + slot * r.slot);
    SbDupKey best = ((SbDupKey*)r.best)[slot * 32 + lane];
    const SbDupKey start = SbDupFull::load(p, meta[4] ? 1ull : 0ull);
    SbDupKey* destination = band + (row % rows) * m;
    for (int col = lane; col < m; col += 32) {
        SbDupKey value = SbDupFull::inf();
        if (sb_dup_pt(ca[row], cb[col], p.n_p) >= 0) {
            value = sb_dup_cell<SbDupFull>(p, ca, cb, n, m, rows, band, row, col, start, best);
        }
        destination[col] = value;
    }
    __syncwarp(0xffffffffu);
    ((SbDupKey*)r.best)[slot * 32 + lane] = best;
}

extern "C" __global__ void sb_resident_dup_finish(
    SbDup p, SbResidentResume r, int* keys) {
    const int lane = (int)threadIdx.x;
    const sb_u64 slot = blockIdx.x;
    if (slot >= r.n) return;
    SbDupKey best = ((const SbDupKey*)r.best)[slot * 32 + lane];
    for (int distance = 16; distance != 0; distance >>= 1) {
        best = sb_dup_min<SbDupFull>(best,
            SbDupFull::shfl(0xffffffffu, best, distance, 32));
    }
    if (lane == 0) {
        const sb_u64 t = sb_resident_work(r);
        const SbDupKey key = SbDupFull::norm(best);
        keys[t] = SbDupFull::finite(key) ? (int)key.d : SB_DUP_SENTINEL_KEY;
        const bool identical = ((const sb_u64*)r.meta)[slot * 6 + 3] != 0;
        sb_dup_fields(p, t, key, identical);
    }
}

extern "C" __global__ void sb_resident_pf_init(
    SbTables tb, SbSpace s, SbPf p, SbResidentResume r) {
    const int lane = (int)threadIdx.x;
    const sb_u64 slot = blockIdx.x;
    if (slot >= r.n) return;
#ifdef SB_TESTING
    if (lane == 0) sb_resident_poison_count(p.resident_stats, 1, 0, 1);
#endif
    sb_u64 i, j, na64, nb64;
    sb_decode(s, sb_resident_work(r), i, j);
    sb_dup_lengths(tb, i, j, na64, nb64);
    const int na = (int)na64, nb = (int)nb64;
    sb_u8* ca = (sb_u8*)r.codes + slot * SB_PF_CODES;
    sb_u8* cb = ca + SB_PF_MAX_LENGTH;
    const SbSequence a = sb_a_row(tb, i), b = sb_b_row(tb, j);
    for (int q = lane; q < na; q += 32) ca[q] = sb_dup_code(a[q]);
    for (int q = lane; q < nb; q += 32) cb[q] = sb_dup_code(b[q]);
    __syncwarp(0xffffffffu);
    const int shorter = na < nb ? na : nb;
    int diff = shorter;
    for (int q = 0; q < shorter; ++q) {
        if (ca[q] != cb[q]) { diff = q; break; }
    }
    const bool swap = diff == shorter ? na > nb : ca[diff] > cb[diff];
    const sb_u8* x = swap ? cb : ca;
    const int n = swap ? nb : na, m = swap ? na : nb;
    const int rows = n < (int)p.lam + 2 ? n : (int)p.lam + 2;
#ifdef SB_TESTING
    if (p.poison < 256) {
        // Fill the ZA and ZO rings and E before E's normal zero initialization.
        const sb_u64 total = ((1 + p.n_c) * (sb_u64)rows + 1) * (sb_u64)m * SB_PF_VALUE;
        sb_u8* band = (sb_u8*)r.band + slot * r.slot;
        for (sb_u64 q = (sb_u64)lane; q < total; q += 32) band[q] = (sb_u8)p.poison;
        __syncwarp(0xffffffffu);
        if (lane == 0) {
            sb_resident_poison_count(p.resident_stats, 1, 1, 1);
            sb_resident_poison_count(p.resident_stats, 1, 2, total);
        }
    }
#endif
    bool palindrome = true;
    for (int q = lane; q < n; q += 32) {
        palindrome = palindrome && x[q] + x[n - 1 - q] == 3;
    }
    const bool pal = __all_sync(0xffffffffu, palindrome) != 0;
    if (lane == 0) {
        sb_u64* meta = (sb_u64*)r.meta + slot * 6;
        meta[0] = (sb_u64)n; meta[1] = (sb_u64)m; meta[2] = (sb_u64)rows;
        meta[3] = 0;  // ring counter; advance without runtime division/remainder
        meta[4] = diff == shorter && na == nb && pal; meta[5] = swap;
    }
    SbF53* band = (SbF53*)((sb_u8*)r.band + slot * r.slot);
    SbF53* col = band + (1 + (int)p.n_c) * rows * m;
    for (int c = lane; c < m; c += 32) col[c] = sb_f53_zero();
}

extern "C" __global__ void sb_resident_pf_step(SbPf p, SbResidentResume r) {
    const int lane = (int)threadIdx.x;
    const sb_u64 slot = blockIdx.x;
    if (slot >= r.n) return;
    sb_u64* meta = (sb_u64*)r.meta + slot * 6;
    const int n = (int)meta[0], m = (int)meta[1], rows = (int)meta[2];
    const int row = (int)r.row;
    if (row >= n) return;
    const sb_u8* ca = (const sb_u8*)r.codes + slot * SB_PF_CODES;
    const sb_u8* cb = ca + SB_PF_MAX_LENGTH;
    const sb_u8* x = meta[5] ? cb : ca;
    const sb_u8* y = meta[5] ? ca : cb;
    const int nc = (int)p.n_c;
    SbF53* za = (SbF53*)((sb_u8*)r.band + slot * r.slot);
    SbF53* zo = za + rows * m;
    SbF53* col = zo + rows * m * nc;
    const int cur = (int)meta[3];
    for (int c = lane; c < m; c += 32) {
        SbF53 za_value = sb_f53_zero();
        SbF53* zo_cell = zo + (cur * m + c) * nc;
        if (sb_dup_pt(x[row], y[c], p.n_p) >= 0) {
            SbF53 term;
            sb_pf_cell(p, (const SbF53*)p.w, x, y, n, m, rows, za, zo,
                cur, row, c, (int)meta[4], za_value, zo_cell, term);
            if (term.m != 0) col[c] = sb_f53_add(col[c], term);
        } else {
            for (int q = 0; q < nc; ++q) zo_cell[q] = sb_f53_zero();
        }
        za[cur * m + c] = za_value;
    }
    __syncwarp(0xffffffffu);
    if (lane == 0) meta[3] = cur + 1 == rows ? 0ull : (sb_u64)(cur + 1);
}

extern "C" __global__ void sb_resident_pf_finish(SbPf p, SbResidentResume r) {
    const sb_u64 slot = blockIdx.x;
    if (slot >= r.n || threadIdx.x != 0) return;
    const sb_u64* meta = (const sb_u64*)r.meta + slot * 6;
    const int m = (int)meta[1], rows = (int)meta[2];
    const SbF53* band = (const SbF53*)((const sb_u8*)r.band + slot * r.slot);
    const SbF53* col = band + (1 + (int)p.n_c) * rows * m;
    SbF53 total = sb_f53_zero();
    for (int c = 0; c < m; ++c) total = sb_f53_add(total, col[c]);
    ((SbF53*)p.out)[sb_resident_work(r)] = total;
}
