// Device-only routing around the existing edit/alignment score tiers.
// All pair classification and compaction happens on the device. Only seven
// counts and seven pairs of maximum lengths are host allocation controls.
#pragma once
#include "levenshtein.cuh"
#include "align.cuh"

struct SbResidentEditPlan {
    sb_u64 family;
    SbLev lev;
    SbAln aln;
};
static_assert(sizeof(SbResidentEditPlan) == 26 * 8, "resident edit plan ABI");
struct SbResidentEditRoutes {
    sb_u64 bins, cursors, i, j, positions;
    sb_u64 starts[8];
};
static_assert(sizeof(SbResidentEditRoutes) == 13 * 8, "resident edit routes ABI");

// Buckets: closed, thread, warp Q=1/2/4/8, device continuation. This uses
// division before multiplication: a legal all-zero scoring pack may have
// extremely long zero-stride rows, and routing must not wrap their work size.
__device__ __forceinline__ unsigned sb_resident_edit_bucket(const SbResidentEditPlan& p,
                                                              sb_u64 m, sb_u64 n) {
    if (p.family == 0) {
        const SbLev& q = p.lev;
        const sb_u64 w = m / 64 + (m % 64 != 0 ? 1 : 0);
        if (w > q.device_words) return 6;
        if (q.thread_words && w <= q.thread_words)
            return w && n > q.pair_steps / w ? 6 : 1;
        const sb_u64 lanes = w / 32 + (w % 32 != 0 ? 1 : 0);
        if (lanes && (n > ~0ull - 31 || n + 31 > q.pair_steps / lanes)) return 6;
        return lanes <= 1 ? 2 : lanes <= 2 ? 3 : lanes <= 4 ? 4 : 5;
    }
    const SbAln& q = p.aln;
    const sb_u64 low = sb_aln_closed_below(q);
    if (m < low || n < low) return 0;
    // Needle's empty-side closed form belongs to the general continuation,
    // not the old nonempty warp recurrence.
    if (!m || !n) return 6;
    if (q.thread_rows && m <= q.thread_rows && m <= q.pair_cells / n) return 1;
    const sb_u64 lanes = sb_aln_q(q, m), height = 32 * lanes;
    const sb_u64 strips = m / height + (m % height != 0 ? 1 : 0);
    if (n > ~0ull - 31 || strips > q.pair_cells / lanes / (n + 31)) return 6;
    return lanes <= 1 ? 2 : lanes <= 2 ? 3 : lanes <= 4 ? 4 : 5;
}

extern "C" __global__ void sb_resident_edit_classify(SbTables tables, SbSpace work,
                                                      SbResidentEditPlan plan,
                                                      sb_u64 bins, sb_u64 control) {
    sb_u64* summary = (sb_u64*)control;
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < work.n; t += step) {
        sb_u64 i, j; sb_decode(work, t, i, j);
        if (!tables.a_view || !tables.b_view ||
            i >= ((const SbDeviceRows*)tables.a_view)->rows ||
            j >= ((const SbDeviceRows*)tables.b_view)->rows) {
            atomicExch(summary + 21, 1ull); continue;
        }
        const SbSequence a = sb_a_row(tables, i), b = sb_b_row(tables, j);
        if (a.length < 0 || b.length < 0) { atomicExch(summary + 21, 1ull); continue; }
        const sb_u64 m = (sb_u64)a.length, n = (sb_u64)b.length;
        const unsigned bucket = sb_resident_edit_bucket(plan, m, n);
        ((sb_u8*)bins)[t] = (sb_u8)bucket;
        atomicAdd(summary + bucket, 1ull);
        atomicMax(summary + 7 + bucket, m);
        atomicMax(summary + 14 + bucket, n);
    }
}

extern "C" __global__ void sb_resident_edit_compact(SbSpace work, SbResidentEditRoutes out) {
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < work.n; t += step) {
        const unsigned bucket = ((const sb_u8*)out.bins)[t];
        const sb_u64 at = out.starts[bucket] + atomicAdd((sb_u64*)out.cursors + bucket, 1ull);
        sb_u64 i, j; sb_decode(work, t, i, j);
        ((sb_i64*)out.i)[at] = (sb_i64)i;
        ((sb_i64*)out.j)[at] = (sb_i64)j;
        ((sb_u64*)out.positions)[at] = t;
    }
}

extern "C" __global__ void sb_resident_edit_scatter(sb_u64 positions, sb_u64 values,
                                                      sb_u64 output, sb_u64 n) {
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; t < n; t += step)
        ((int*)output)[((const sb_u64*)positions)[t]] = ((const int*)values)[t];
}
