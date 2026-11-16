// The C2 bound of K3 `duplex(model="pf")` in the resident K3 stage
// (docs/wiki/thermo/duplex-pf-bound.md §6.2, §6.5, §6.6, §8): per work item the pass
// threshold θ, the outcome (open, certified, pruned, or skipped by the TopK replay), the
// DP skip flag, the int64 outcome word and integer outcome counts; and the probe of the
// weak-bound decision. Written from that page only.
//
// Integers only. A resolved item is skipped by K3's DP; the closed-form kernel writes its
// Z-hat = (0, 0), so its key is key(+inf): exact for a certified item, and above θ (as
// the item's own key is) for a pruned one. Counts are integer atomics.
#pragma once
#include "duplex_pf_bound.cuh"
#include "resident_bound.cuh"

// The outcome word of an open item: INT64_MAX, never a λ (λ <= key(+inf) < INT64_MAX).
#define SB_RPB_WORD_OPEN 0x7FFFFFFFFFFFFFFFll

// The outcome code of item (i, j) of the resident tables against θ, and its λ.
__device__ __forceinline__ unsigned sb_rpb_outcome(const SbPfBound& p, const SbPfFinalize& f,
                                                   const SbPfbClass* rel, const SbTables& tb,
                                                   sb_u64 i, sb_u64 j, sb_u64 n, sb_u64 m,
                                                   bool has, sb_i64 theta, SbF53* scratch,
                                                   sb_i64& lambda) {
    SbF53 z = sb_f53_zero();
    if (n >= 2 && m >= 2) {
        sb_u8* ca = (sb_u8*)scratch;
        sb_u8* cb = ca + SB_PF_MAX_LENGTH;
        const SbSequence pa = sb_a_row(tb, i);
        const SbSequence pb = sb_b_row(tb, j);
        for (sb_u64 x = 0; x < n; ++x) ca[x] = sb_dup_code(pa[x]);
        for (sb_u64 x = 0; x < m; ++x) cb[x] = sb_dup_code(pb[x]);
        z = sb_pfb_pair(p, rel, ca, (int)n, cb, (int)m, scratch + SB_PFB_CODE_VALUES);
    }
    return sb_pfb_outcome(z, sb_pfb_lower(z, f), has, theta, lambda);
}

// One thread per item of the chunk, grid-striding, each with its own scratch slot. Every
// item gets its skip flag (resolved and replay-skipped items are skipped). A repeated
// sweep (RowStats sweep 2) restores each outcome from its word without evaluating the
// bound: a resolved word λ is pruned iff a threshold exists and λ > θ, else certified.
// `r.keys`, `r.dh`, `r.ds`, `r.tm` and `r.fields` are unused (K3 skips every resolved
// item; its closed-form kernel writes the term).
extern "C" __global__ void sb_resident_pf_bound(SbTables tb, SbSpace s, SbPfBound p,
                                                SbPfFinalize f, SbResidentBound r) {
    SbPfbClass rel[SB_PFB_MAX_CLASSES];
    sb_pfb_load(p, rel);
    const sb_u64 threads = (sb_u64)gridDim.x * blockDim.x;
    const sb_u64 self = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x;
    SbF53* scratch = (SbF53*)r.scratch + self * r.stride;
    unsigned long long* counts = (unsigned long long*)r.counts;
    for (sb_u64 t = self; t < s.n; t += threads) {
        sb_u64 i, j, n, m;
        sb_decode(s, t, i, j);
        sb_dup_lengths(tb, i, j, n, m);
        const sb_u64 mult = r.mult != 0 ? ((const sb_u64*)r.mult)[s.k0 + t] : 1ull;
        bool has;
        sb_i64 theta;
        sb_i64 lambda = 0;
        unsigned code = SB_RB_SKIPPED;
        sb_u64 duplicates = 0;
        if (sb_rb_theta(r, i, j, has, theta)) {
            if (r.mode == 2) {
                duplicates = mult - 1;
            }
            if (r.words_mode == 2) {
                const sb_i64 word = ((const sb_i64*)r.words)[t];
                if (word == SB_RPB_WORD_OPEN) {
                    code = SB_RB_OPEN;
                } else {
                    lambda = word;
                    code = (has && lambda > theta) ? SB_RB_PRUNED : SB_RB_CERTIFIED;
                }
            } else {
                code = r.use != 0 ? sb_rpb_outcome(p, f, rel, tb, i, j, n, m, has, theta,
                                                   scratch, lambda)
                                  : SB_RB_OPEN;
            }
        }
        ((sb_u8*)r.skip)[t] = code == SB_RB_OPEN ? 0 : 1;
        if (r.words_mode == 1) {
            ((sb_i64*)r.words)[t] =
                (code == SB_RB_OPEN || code == SB_RB_SKIPPED) ? SB_RPB_WORD_OPEN : lambda;
        }
        if (r.count == 0) {
            continue;
        }
        const sb_u64 once = r.mode == 2 ? 1ull : mult;
        if (code == SB_RB_CERTIFIED) {
            atomicAdd(counts + 0, (unsigned long long)once);
        } else if (code == SB_RB_PRUNED) {
            atomicAdd(counts + 1, (unsigned long long)once);
        } else if (code == SB_RB_OPEN) {
            atomicAdd(counts + 2, (unsigned long long)once);
        } else {
            duplicates = mult;
        }
        if (duplicates != 0) {
            atomicAdd(counts + 3, (unsigned long long)duplicates);
        }
    }
}

// The probe of the weak-bound decision (§6.6): per probe item (work ID ids[t] of the
// original-order space `s`), its strand lengths and its outcome code (skipped when the
// replay has no limit for it or `dup[t]` marks a later occurrence). out: 3 words per item.
extern "C" __global__ void sb_resident_pf_probe(SbTables tb, SbSpace s, SbPfBound p,
                                                SbPfFinalize f, SbResidentBound r,
                                                const sb_u64* ids, sb_u64 count,
                                                const unsigned* dup, sb_u64* out) {
    SbPfbClass rel[SB_PFB_MAX_CLASSES];
    sb_pfb_load(p, rel);
    const sb_u64 threads = (sb_u64)gridDim.x * blockDim.x;
    const sb_u64 self = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x;
    SbF53* scratch = (SbF53*)r.scratch + self * r.stride;
    for (sb_u64 t = self; t < count; t += threads) {
        sb_u64 i, j, n, m;
        sb_decode(s, ids[t], i, j);
        sb_dup_lengths(tb, i, j, n, m);
        bool has;
        sb_i64 theta;
        sb_i64 lambda = 0;
        unsigned code = SB_RB_SKIPPED;
        if ((dup == 0 || dup[t] == 0) && sb_rb_theta(r, i, j, has, theta)) {
            code = sb_rpb_outcome(p, f, rel, tb, i, j, n, m, has, theta, scratch, lambda);
        }
        out[3 * t] = n;
        out[3 * t + 1] = m;
        out[3 * t + 2] = code;
    }
}
