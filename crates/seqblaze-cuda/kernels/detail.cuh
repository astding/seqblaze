// Resident Levenshtein and affine-gap alignment detail replay.
// Source: seqblaze wiki edit/levenshtein §§2,4,6; align/gotoh-align §§2,4,5,7,
// and the repository's own MIT/Apache implementation. No reference source is used.
// All row reads, DP states, traceback, run formatting and outputs stay on device.
// A bounded continuation kernel advances every selected pair by a work quantum;
// no pair is routed to host, regardless of the score-stage tier limits.
#pragma once
#include "levenshtein.cuh"

#define SB_DETAIL_NEG (-2305843009213693952ll)
#define SB_DETAIL_DONE 99ull
#define SB_DETAIL_STATE_WORDS 80ull
#define SB_DETAIL_LEV_ENDS 16ull
#define SB_DETAIL_LEV_CIGAR 32ull
#define SB_DETAIL_ALN_CIGAR 256ull
#define SB_DETAIL_ALN_A 512ull
#define SB_DETAIL_ALN_B 1024ull
#define SB_DETAIL_ALN_N_ENDS 2048ull

struct SbDetailConfig {
    sb_u64 family;             // 0: unit-cost edit, 1: affine alignment
    sb_u64 kind;               // edit: global/prefix/infix; align: local/global/water/needle
    sb_u64 fields;
    sb_u64 go, ge, end_gaps, mm;
    sb_i64 matched, mismatched;
    sb_u64 n_classes, class_map, substitution, identity_map, display_map;
    sb_u64 keys, passed, key_output, output_base;
};
static_assert(sizeof(SbDetailConfig) == 18 * 8, "detail config ABI");

// Numeric slots: edit b_end/n_ends/b_start/n_starts; align coords, stats,
// then n_ends (slot 8, int64; native local only).
// Count/offset channels: cigar bytes, ends int64 elements, aligned columns.
struct SbDetailOutput {
    sb_u64 numeric[9];
    sb_u64 counts[3];
    sb_u64 offsets[3];
    sb_u64 payload[4];          // cigar, ends, aligned_a, aligned_b
    sb_u64 validity;
    sb_u64 writing;             // 0: count/numeric pass; 1: exact-sized payload pass
};
static_assert(sizeof(SbDetailOutput) == 21 * 8, "detail output ABI");

struct SbDetailState {
    sb_u64 m, n, valid, phase, after_init, reverse, words, peq_words;
    sb_u64 scratch_at, trace_at, ops_at, pv_at, mv_at, rows_at, col_at, bottom_at;
    sb_u64 i, j, k, init_at;
    sb_u64 a_start, a_end, b_start, b_end, n_ends, n_starts;
    sb_i64 score, best, current, rmax, diag0, diag1, diag2, left0, left1, left2;
    sb_u64 best_i, best_j, last_end, reverse_count;
    sb_u64 ti, tj, ts, op_len, emit_at, emit_i, emit_j, run, run_op, cigar_len;
    sb_u64 identity, similarity, gaps, pending, pending_op, walk_stage, extra, error;
    sb_u64 local_ends;          // native local: cells with V = best > 0 (align n_ends)
    sb_u64 reserved[21];
};
static_assert(sizeof(SbDetailState) == SB_DETAIL_STATE_WORDS * 8, "detail state ABI");

__device__ __forceinline__ bool sb_detail_cigar(const SbDetailConfig& c) {
    return (c.fields & (c.family == 0 ? SB_DETAIL_LEV_CIGAR : SB_DETAIL_ALN_CIGAR)) != 0;
}
__device__ __forceinline__ bool sb_detail_align_trace(const SbDetailConfig& c) {
    return (c.fields & (16ull | 32ull | 64ull | 128ull | SB_DETAIL_ALN_CIGAR |
                        SB_DETAIL_ALN_A | SB_DETAIL_ALN_B)) != 0 ||
           ((c.kind == 0 || c.kind == 2) && (c.fields & 15ull) != 0);
}
// Native local n_ends needs the forward fill only: no traceback, no cell cap.
__device__ __forceinline__ bool sb_detail_align_count(const SbDetailConfig& c) {
    return c.kind == 0 && (c.fields & SB_DETAIL_ALN_N_ENDS) != 0;
}
__device__ __forceinline__ bool sb_detail_align_fill(const SbDetailConfig& c) {
    return sb_detail_align_trace(c) || sb_detail_align_count(c) || c.key_output;
}
__device__ __forceinline__ sb_i64 sb_detail_max(sb_i64 a, sb_i64 b) { return a > b ? a : b; }
__device__ __forceinline__ sb_i64 sb_detail_max3(sb_i64 a, sb_i64 b, sb_i64 d) {
    return sb_detail_max(sb_detail_max(a, b), d);
}
__device__ __forceinline__ sb_i64 sb_detail_gap(const SbDetailConfig& c, sb_u64 n) {
    return n == 0 || c.end_gaps == 0 ? 0 : (sb_i64)c.go + (sb_i64)(n - 1) * (sb_i64)c.ge;
}
__device__ __forceinline__ sb_i64 sb_detail_sub(const SbDetailConfig& c, unsigned a, unsigned b) {
    if (c.mm) return a == b ? c.matched : c.mismatched;
    const unsigned char* map = (const unsigned char*)c.class_map;
    return ((const int*)c.substitution)[(sb_u64)map[a] * c.n_classes + map[b]];
}
__device__ __forceinline__ bool sb_detail_identity(const SbDetailConfig& c, unsigned a, unsigned b) {
    const sb_u8* map = (const sb_u8*)c.identity_map;
    return map[a] == map[b];
}
__device__ __forceinline__ unsigned sb_detail_digits(sb_u64 n) {
    unsigned digits = 1;
    while (n >= 10) { n /= 10; ++digits; }
    return digits;
}
__device__ __forceinline__ void sb_detail_run(sb_u8* out, sb_u64 at, sb_u64 n, sb_u8 op) {
    const unsigned digits = sb_detail_digits(n);
    out[at + digits] = op;
    for (unsigned k = digits; k != 0; --k) { out[at + k - 1] = (sb_u8)('0' + n % 10); n /= 10; }
}
__device__ __forceinline__ bool sb_detail_add(sb_u64& a, sb_u64 b) {
    if (b > ~0ull - a) return false;
    a += b; return true;
}
__device__ __forceinline__ bool sb_detail_mul(sb_u64 a, sb_u64 b, sb_u64& out) {
    if (a != 0 && b > ~0ull / a) return false;
    out = a * b; return true;
}

// At most the caller's bounded entry chunk is planned. Only the four aggregate
// scalars are copied to host; lengths and sequence content remain device-side.
extern "C" __global__ void sb_detail_plan(SbTables tables, SbSpace entries,
                                            SbDetailConfig c, SbDetailState* states,
                                            sb_u64* aggregate) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    sb_u64 scratch = 0, trace = 0, ops = 0, error = 0;
    for (sb_u64 t = 0; t < entries.n; ++t) {
        SbDetailState s = {};
        s.valid = !c.passed || ((const sb_u8*)c.passed)[entries.k0 + t - c.output_base] != 0;
        if (s.valid) {
            sb_u64 i, j; sb_decode(entries, t, i, j);
            if (!tables.a_view || !tables.b_view ||
                i >= ((const SbDeviceRows*)tables.a_view)->rows ||
                j >= ((const SbDeviceRows*)tables.b_view)->rows) {
                s.valid = 0; s.error = 3; error = 3;
            } else {
                s.m = (sb_u64)sb_a_row(tables, i).length;
                s.n = (sb_u64)sb_b_row(tables, j).length;
            }
        }
        s.scratch_at = scratch; s.trace_at = trace; s.ops_at = ops;
        if (s.valid && c.family == 0) {
            s.words = s.m / 64 + (s.m % 64 != 0 ? 1 : 0);
            // Small/medium pairs cache every byte's exact match masks. Very
            // long rows generate one 64-byte mask at a time, bounding memory
            // independently of the raw alphabet's 256 possible symbols.
            s.peq_words = s.words <= 256 ? 256 * s.words : 0;
            s.pv_at = scratch;
            if (!sb_detail_add(s.pv_at, s.peq_words)) error = 1;
            s.mv_at = s.pv_at;
            if (!sb_detail_add(s.mv_at, s.words) ||
                !sb_detail_add(scratch, s.peq_words) ||
                !sb_detail_add(scratch, s.words) || !sb_detail_add(scratch, s.words)) error = 1;
            if (sb_detail_cigar(c)) {
                sb_u64 twice = 0, window = s.n;
                if (c.kind != SB_LEV_GLOBAL) {
                    if (!sb_detail_mul(2, s.m, twice)) error = 1;
                    window = s.n < twice ? s.n : twice;
                }
                sb_u64 bytes = 0, cap = 0, word_bytes = 0, columns = window, moves = s.m;
                if (!sb_detail_add(columns, 1) || !sb_detail_mul(16, s.words, word_bytes) ||
                    !sb_detail_mul(word_bytes, columns, cap) || cap > (1ull << 27) ||
                    !sb_detail_mul(word_bytes, window, bytes) ||
                    !sb_detail_add(trace, bytes) || !sb_detail_add(moves, window) ||
                    !sb_detail_add(ops, moves)) error = 1;
            }
        } else if (s.valid && c.family == 1 && sb_detail_align_fill(c) && s.m && s.n) {
            sb_u64 rows = s.m, columns = s.n, cells = 0;
            if (!sb_detail_add(rows, 1) || !sb_detail_add(columns, 1)) error = 1;
            if (sb_detail_align_trace(c) &&
                (!sb_detail_mul(rows, columns, cells) || cells > (1ull << 26))) error = 1;
            sb_u64 row_words = 0, column_words = 0, total_words = 0;
            if (!sb_detail_mul(3, columns, row_words) ||
                !sb_detail_mul(3, rows, column_words)) error = 1;
            s.rows_at = scratch; s.col_at = scratch;
            if (!sb_detail_add(s.col_at, row_words)) error = 1;
            s.bottom_at = s.col_at;
            if (!sb_detail_add(s.bottom_at, column_words)) error = 1;
            total_words = row_words;
            if (!sb_detail_add(total_words, column_words) ||
                !sb_detail_add(total_words, row_words) ||
                !sb_detail_add(scratch, total_words)) error = 1;
            if (sb_detail_align_trace(c)) {
                sb_u64 bytes = 0, moves = s.m;
                if (!sb_detail_mul(s.m, s.n, bytes) || !sb_detail_add(trace, bytes) ||
                    !sb_detail_add(moves, s.n) || !sb_detail_add(ops, moves)) error = 1;
            }
        } else if (s.valid && c.family == 1 && sb_detail_align_trace(c)) {
            sb_u64 moves = s.m;
            if (!sb_detail_add(moves, s.n) || !sb_detail_add(ops, moves)) error = 1;
        }
        states[t] = s;
    }
    aggregate[0] = scratch; aggregate[1] = trace; aggregate[2] = ops; aggregate[3] = error;
}

__device__ __forceinline__ sb_u64 sb_detail_eq(const SbDetailState& s, const SbSequence& a,
                                               const sb_u64* scratch, unsigned byte, sb_u64 word) {
    if (s.peq_words) return scratch[s.scratch_at + (sb_u64)byte * s.words + word];
    sb_u64 mask = 0;
    const sb_u64 begin = word * 64;
    const sb_u64 end = begin + 64 < s.m ? begin + 64 : s.m;
    for (sb_u64 row = begin; row < end; ++row)
        if (a[s.reverse ? s.m - 1 - row : row] == byte) mask |= 1ull << (row - begin);
    return mask;
}

__device__ __forceinline__ void sb_detail_lev_begin(SbDetailState& s, sb_u64 next, bool reverse) {
    s.reverse = reverse ? 1 : 0; s.after_init = next; s.phase = 1; s.init_at = 0;
    s.i = s.j = s.k = 0; s.current = (sb_i64)s.m;
    s.best = (sb_i64)s.m; s.last_end = 0; s.reverse_count = 1;
}

__device__ __forceinline__ void sb_detail_complete(SbDetailState& s, sb_u64 row,
                                                   const SbDetailConfig& c, const SbDetailOutput& o) {
    if (!o.writing) {
        if (o.validity) ((sb_u8*)o.validity)[row] = s.valid ? 1 : 0;
        if (c.family == 0) {
            const sb_u64 v[4] = {s.b_end, s.n_ends, s.b_start, s.n_starts};
            for (int k = 0; k < 4; ++k) if (o.numeric[k]) {
                if (k == 1 || k == 3) ((int*)o.numeric[k])[row] = s.valid ? (int)v[k] : SB_INT32_MAX;
                else ((sb_i64*)o.numeric[k])[row] = s.valid ? (sb_i64)v[k] : 9223372036854775807ll;
            }
        } else {
            const sb_u64 v[9] = {s.a_start, s.a_end, s.b_start, s.b_end,
                                 s.op_len, s.identity, s.similarity, s.gaps, s.local_ends};
            for (int k = 0; k < 9; ++k) if (o.numeric[k]) {
                if (k >= 4 && k < 8) ((int*)o.numeric[k])[row] = s.valid ? (int)v[k] : SB_INT32_MAX;
                else ((sb_i64*)o.numeric[k])[row] = s.valid ? (sb_i64)v[k] : 9223372036854775807ll;
            }
        }
        if (o.counts[0]) ((sb_u64*)o.counts[0])[row] = s.valid ? s.cigar_len : 0;
        if (o.counts[1]) ((sb_u64*)o.counts[1])[row] = s.valid ? s.n_ends : 0;
        if (o.counts[2]) ((sb_u64*)o.counts[2])[row] = s.valid ? s.op_len : 0;
    }
    s.phase = SB_DETAIL_DONE;
}

// The second pass checks every payload write against the first pass's
// immutable row offsets. A changed payload length during replay cannot
// turn its mismatch into a write outside the allocated output span.
__device__ __forceinline__ bool sb_detail_room(SbDetailState& s, sb_u64 row,
                                                unsigned channel, sb_u64 at, sb_u64 n,
                                                const SbDetailOutput& o) {
    const sb_i64* offsets = (const sb_i64*)o.offsets[channel];
    const sb_u64 start = (sb_u64)offsets[row], end = (sb_u64)offsets[row + 1];
    if (end < start || at > end - start || n > end - start - at) {
        s.error = 3; s.phase = SB_DETAIL_DONE; return false;
    }
    return true;
}
__device__ __forceinline__ void sb_detail_emit_end(SbDetailState& s, sb_u64 row, sb_u64 end,
                                                   const SbDetailOutput& o) {
    if (o.writing && o.payload[1]) {
        if (!sb_detail_room(s, row, 1, s.extra, 1, o)) return;
        const sb_u64 start = (sb_u64)((const sb_i64*)o.offsets[1])[row];
        ((sb_i64*)o.payload[1])[start + s.extra] = (sb_i64)end;
    }
    ++s.extra;
}

// Returns after one word/cell/move so the outer work quantum also bounds long
// rows, raw match-mask construction, traceback and string formatting.
__device__ __forceinline__ void sb_detail_lev_step(SbDetailState& s, const SbSequence& a,
                                                   const SbSequence& b, sb_u64 row,
                                                   const SbDetailConfig& c, const SbDetailOutput& o,
                                                   sb_u64* scratch, sb_u8* codes, sb_u8* moves) {
    const sb_i64 target = c.keys ? ((const int*)c.keys)[row] : -1;
    if (s.phase == 0) {
        s.b_end = c.kind == SB_LEV_GLOBAL ? s.n : 0; s.n_ends = 1; s.n_starts = 1;
        s.a_end = s.m;
        sb_detail_lev_begin(s, 2, false);
        return;
    }
    if (s.phase == 1) {
        if (s.init_at < s.peq_words) scratch[s.scratch_at + s.init_at++] = 0;
        else if (s.init_at < s.peq_words + s.words) {
            const sb_u64 w = s.init_at++ - s.peq_words;
            scratch[s.pv_at + w] = ~0ull; scratch[s.mv_at + w] = 0;
        } else if (s.peq_words && s.i < s.m) {
            const sb_u64 bit = s.reverse ? s.m - 1 - s.i : s.i;
            scratch[s.scratch_at + (sb_u64)a[s.i] * s.words + bit / 64] |= 1ull << (bit & 63);
            ++s.i;
        } else {
            s.i = 0; s.phase = s.after_init; s.extra = 0;
            if (s.phase == 2 && c.kind != SB_LEV_GLOBAL && (sb_i64)s.m == target)
                sb_detail_emit_end(s, row, 0, o);
        }
        return;
    }
    if (s.phase == 2 || s.phase == 4 || s.phase == 6) {
        const bool forward = s.phase == 2, reverse = s.phase == 4;
        sb_u64 width = forward ? s.n : reverse ? s.b_end : s.b_end - s.b_start;
        if (reverse) {
            // Phase 4 has a verified nonnegative target; both terms fit int32.
            const sb_u64 max_window = s.m + (sb_u64)target;
            width = width < max_window ? width : max_window;
        }
        if (s.j < width) {
            if (s.k == 0) s.left0 = (!forward || c.kind != SB_LEV_INFIX) ? 1 : 0;
            const sb_u64 at = reverse ? s.b_end - 1 - s.j : forward ? s.j : s.b_start + s.j;
            if (s.words != 0) {
                sb_u64 ph = 0, mh = 0;
                int hin = sb_lev_step(scratch[s.pv_at + s.k], scratch[s.mv_at + s.k],
                                     sb_detail_eq(s, a, scratch, b[at], s.k), (int)s.left0, ph, mh);
                s.left0 = hin;
                if (s.phase == 6) {
                    sb_u64* trace = (sb_u64*)(codes + s.trace_at);
                    trace[2 * (s.j * s.words + s.k)] = scratch[s.pv_at + s.k];
                    trace[2 * (s.j * s.words + s.k) + 1] = ph;
                }
                ++s.k;
                if (s.k != s.words) return;
                const unsigned bit = (unsigned)((s.m - 1) & 63);
                s.current += (sb_i64)((ph >> bit) & 1ull) - (sb_i64)((mh >> bit) & 1ull);
            } else s.current = forward && c.kind == SB_LEV_INFIX ? 0 : (sb_i64)s.j + 1;
            s.k = 0; ++s.j;
            if (forward && c.kind != SB_LEV_GLOBAL) {
                if (s.current < s.best) { s.best = s.current; s.b_end = s.j; s.n_ends = 1; }
                else if (s.current == s.best) ++s.n_ends;
                if (s.current == target) sb_detail_emit_end(s, row, s.j, o);
            } else if (reverse) {
                if (s.current < s.best) { s.best = s.current; s.last_end = s.j; s.reverse_count = 1; }
                else if (s.current == s.best) { s.last_end = s.j; ++s.reverse_count; }
            }
            return;
        }
        if (forward) {
            const sb_i64 actual = c.kind == SB_LEV_GLOBAL ? s.current : s.best;
            if (c.keys && actual != target) { s.error = 2; s.phase = SB_DETAIL_DONE; return; }
            if (c.key_output) ((int*)c.key_output)[row] = (int)actual;
            if (c.kind == SB_LEV_GLOBAL) { s.b_end = s.n; s.n_ends = 1; }
            if (c.kind == SB_LEV_INFIX && (c.fields & (4ull | 8ull | SB_DETAIL_LEV_CIGAR))) {
                sb_detail_lev_begin(s, 4, true); return;
            }
        } else if (reverse) {
            if (s.best != target) { s.error = 2; s.phase = SB_DETAIL_DONE; return; }
            s.b_start = s.b_end - s.last_end; s.n_starts = s.reverse_count;
        } else {
            s.ti = s.m; s.tj = width; s.phase = 7; return;
        }
        if (sb_detail_cigar(c)) { sb_detail_lev_begin(s, 6, false); return; }
        sb_detail_complete(s, row, c, o); return;
    }
    if (s.phase == 7) {
        if (s.ti || s.tj) {
            sb_u8 op;
            if (!s.ti) op = 'D';
            else if (!s.tj) op = 'I';
            else {
                const sb_u64* trace = (const sb_u64*)(codes + s.trace_at);
                const sb_u64 at = 2 * ((s.tj - 1) * s.words + (s.ti - 1) / 64);
                const unsigned bit = (unsigned)((s.ti - 1) & 63);
                op = ((trace[at] >> bit) & 1ull) ? 'I' :
                     ((trace[at + 1] >> bit) & 1ull) ? 'D' :
                     a[s.ti - 1] == b[s.b_start + s.tj - 1] ? '=' : 'X';
            }
            if (op == 'I') --s.ti; else if (op == 'D') --s.tj; else { --s.ti; --s.tj; }
            moves[s.ops_at + s.op_len++] = op; return;
        }
        s.phase = 30; s.emit_at = 0; s.emit_i = 0; s.emit_j = s.b_start;
    }
}

// Native code state numbers exactly describe rule E: START, P, Q, V.
__device__ __forceinline__ unsigned sb_detail_first3(sb_i64 top, sb_i64 first, sb_i64 second) {
    return top == first ? 1u : top == second ? 2u : 3u;
}

__device__ __forceinline__ void sb_detail_align_begin(SbDetailState& s) {
    s.phase = 11; s.init_at = 0; s.i = 1; s.j = 1;
    s.best = 0; s.rmax = SB_DETAIL_NEG; s.local_ends = 0;
}

__device__ __forceinline__ void sb_detail_align_step(SbDetailState& s, const SbSequence& a,
                                                     const SbSequence& b, sb_u64 row,
                                                     const SbDetailConfig& c, const SbDetailOutput& o,
                                                     sb_u64* scratch, sb_u8* codes, sb_u8* moves) {
    sb_i64* all = (sb_i64*)scratch;
    sb_i64* r0 = all + s.rows_at;
    sb_i64* r1 = r0 + s.n + 1;
    sb_i64* r2 = r1 + s.n + 1;
    sb_i64* col = all + s.col_at;
    sb_i64* bottom = all + s.bottom_at;
    const bool local = c.kind == 0, water = c.kind == 2, needle = c.kind == 3;
    if (s.phase == 0) {
        s.a_end = s.m; s.b_end = s.n; s.local_ends = 0;
        if (!sb_detail_align_fill(c)) { sb_detail_complete(s, row, c, o); return; }
        if (!s.m || !s.n) {
            const sb_i64 key = water ? (sb_i64)SB_INT32_MAX - 1 :
                local ? 0 : sb_detail_gap(c, s.m > s.n ? s.m : s.n);
            if (c.key_output) ((int*)c.key_output)[row] = (int)key;
            if (c.keys && key != ((const int*)c.keys)[row]) { s.error = 2; s.phase = SB_DETAIL_DONE; return; }
            if (!sb_detail_align_trace(c)) { sb_detail_complete(s, row, c, o); return; }
            if (local || water) s.a_end = s.b_end = 0;
            else { s.pending = s.m ? s.m : s.n; s.pending_op = s.m ? 'I' : 'D'; }
            s.walk_stage = 3; s.phase = 20; return;
        }
        // Needle's R-m1 display is I followed by all D, independently of DP.
        if (needle && !c.end_gaps && s.m == 1 && s.n >= 2 && c.ge == 0 &&
            sb_detail_sub(c, a[0], b[s.n - 1]) <= 0) {
            if (c.key_output) ((int*)c.key_output)[row] = 0;
            if (c.keys && ((const int*)c.keys)[row] != 0) { s.error = 2; s.phase = SB_DETAIL_DONE; return; }
            if (!sb_detail_align_trace(c)) { sb_detail_complete(s, row, c, o); return; }
            s.pending = s.n; s.pending_op = 'D'; s.walk_stage = 4; s.phase = 20; return;
        }
        sb_detail_align_begin(s); return;
    }
    if (s.phase == 11) {
        const sb_u64 count = 6 * (s.n + 1) + 3 * (s.m + 1);
        if (s.init_at < count) { all[s.rows_at + s.init_at++] = SB_DETAIL_NEG; return; }
        s.init_at = 0; s.phase = 12; return;
    }
    if (s.phase == 12) {
        if (s.init_at <= s.n) {
            const sb_u64 j = s.init_at++;
            if (water || needle) { r0[j] = water ? 0 : -sb_detail_gap(c, j); r1[j] = SB_DETAIL_NEG; }
            else if (!local && j) r2[j] = -sb_detail_gap(c, j);
            return;
        }
        if (!water && !needle) { col[0] = r0[s.n]; col[1] = r2[s.n]; }
        s.phase = 13; s.j = 0; return;
    }
    if (s.phase == 13) {
        if (s.i <= s.m) {
            if (s.j == 0) {
                s.diag0 = r0[0]; s.diag1 = r1[0]; s.diag2 = r2[0];
                r0[0] = water ? 0 : needle ? -sb_detail_gap(c, s.i) : SB_DETAIL_NEG;
                r1[0] = (!local && !water && !needle) ? -sb_detail_gap(c, s.i) : SB_DETAIL_NEG;
                r2[0] = SB_DETAIL_NEG;
                s.left0 = r0[0]; s.left1 = water || needle ? SB_DETAIL_NEG : r1[0]; s.left2 = r2[0];
                s.j = 1; return;
            }
            if (s.j <= s.n) {
                const sb_u64 j = s.j;
                const sb_i64 u0 = r0[j], u1 = r1[j], u2 = r2[j];
                const sb_i64 sub = sb_detail_sub(c, a[s.i - 1], b[j - 1]);
                sb_i64 v, p, q; unsigned code;
                if (!water && !needle) {
                    const sb_i64 top = sb_detail_max3(s.diag0, s.diag1, s.diag2);
                    const bool start = local ? top <= 0 : s.i == 1 && j == 1;
                    v = sub + (start ? 0 : top);
                    const sb_i64 pe = u1 - (sb_i64)c.ge, qo = u2 - (sb_i64)c.go, vo = u0 - (sb_i64)c.go;
                    p = sb_detail_max3(pe, qo, vo);
                    const sb_i64 po = s.left1 - (sb_i64)c.go, qe = s.left2 - (sb_i64)c.ge;
                    q = sb_detail_max3(po, qe, s.left0 - (sb_i64)c.go);
                    code = (start ? 0 : sb_detail_first3(top, s.diag1, s.diag2)) |
                           (sb_detail_first3(p, pe, qo) << 2) | (sb_detail_first3(q, po, qe) << 4);
                    if (local) {
                        if (v > s.best) { s.best = v; s.best_i = s.i; s.best_j = j; s.local_ends = 1; }
                        else if (v == s.best && s.best > 0) ++s.local_ends;
                    }
                } else {
                    const sb_i64 eo = s.left0 - (sb_i64)c.go, fo = u0 - (sb_i64)c.go;
                    const sb_i64 en = sb_detail_max(eo, s.left1 - (sb_i64)c.ge);
                    const sb_i64 fv = sb_detail_max(fo, u1 - (sb_i64)c.ge);
                    const sb_i64 mm = s.diag0 + sub;
                    const sb_i64 rr = sb_detail_max3(mm, en, fv);
                    v = water ? sb_detail_max(rr, 0) : rr; p = fv; q = en;
                    if (water) {
                        code = v == 0 ? 0 : mm == v ? 1 : en == v ? 2 : 3;
                        code |= (eo == en ? 4u : 0u) | (fo == fv ? 8u : 0u);
                        if (s.i >= 2 && j >= 2) s.rmax = sb_detail_max(s.rmax, rr);
                        if (v > s.best) { s.best = v; s.best_i = s.i; s.best_j = j; }
                    } else {
                        code = mm == v ? 0 : en == v ? 1 : 2;
                        const sb_i64 xe = s.i >= 2 || c.end_gaps ? (sb_i64)c.ge : 0;
                        const sb_i64 xf = j >= 2 || c.end_gaps ? (sb_i64)c.ge : 0;
                        code |= j >= 2 && (s.left1 == s.left0 || s.left1 - xe == en) ? 4u : 0u;
                        code |= s.i >= 2 && (u1 == u0 || u1 - xf == fv) ? 8u : 0u;
                        if (s.i == s.m) { bottom[3 * j] = mm; bottom[3 * j + 1] = en; bottom[3 * j + 2] = fv; }
                        if (j == s.n) { col[3 * s.i] = mm; col[3 * s.i + 1] = en; col[3 * s.i + 2] = fv; }
                    }
                }
                if (sb_detail_align_trace(c))
                    codes[s.trace_at + (s.i - 1) * s.n + j - 1] = (sb_u8)code;
                s.diag0 = u0; s.diag1 = u1; s.diag2 = u2;
                r0[j] = v; r1[j] = p; r2[j] = q;
                s.left0 = v; s.left1 = water || needle ? q : p; s.left2 = q;
                ++s.j; return;
            }
            if (!water && !needle) { col[3 * s.i] = r0[s.n]; col[3 * s.i + 1] = r2[s.n]; }
            ++s.i; s.j = 0; return;
        }
        if (local || water) {
            const sb_i64 key = water && (s.m < 2 || s.n < 2) ? (sb_i64)SB_INT32_MAX - 1 :
                -(water ? s.rmax : s.best);
            if (c.keys && key != ((const int*)c.keys)[row]) { s.error = 2; s.phase = SB_DETAIL_DONE; return; }
            if (c.key_output) ((int*)c.key_output)[row] = (int)key;
            if (!sb_detail_align_trace(c)) { sb_detail_complete(s, row, c, o); return; }
            if (!s.best) {
                s.a_start = s.b_start = 0;
                s.a_end = s.b_end = water ? 1 : 0;
                if (water) { moves[s.ops_at] = 'M'; s.op_len = 1; }
                s.phase = 30;
            } else {
                s.a_end = s.best_i; s.b_end = s.best_j;
                s.ti = s.best_i; s.tj = s.best_j; s.ts = water ? 0 : 3;
                s.walk_stage = 0; s.phase = 20;
            }
            return;
        }
        s.score = needle ? bottom[3 * s.n] : r0[s.n];
        s.ti = s.m; s.tj = s.n; s.k = 0;
        s.phase = 14; return;
    }
    if (s.phase == 14) {
        if (needle) {
            if (s.k < s.n) {
                const sb_u64 j = s.k++;
                if (s.m >= 2) {
                    const sb_i64 value = (j == 0 ? -sb_detail_gap(c, s.m) : bottom[3 * j]) - sb_detail_gap(c, s.n - j);
                    if (value > s.score) { s.score = value; s.ti = s.m; s.tj = j; }
                }
                return;
            }
            if (s.k < s.n + s.m) {
                const sb_u64 i = s.k++ - s.n;
                if (s.n >= 2) {
                    const sb_i64 value = (i == 0 ? -sb_detail_gap(c, s.n) : col[3 * i]) - sb_detail_gap(c, s.m - i);
                    if (value > s.score) { s.score = value; s.ti = i; s.tj = s.n; }
                }
                return;
            }
            if (s.m == 1 && s.n == 1) {
                const sb_i64 value = -2 * sb_detail_gap(c, 1);
                if (value > s.score) { s.score = value; s.ti = 1; s.tj = 0; }
            }
            if (c.keys && -s.score != ((const int*)c.keys)[row]) { s.error = 2; s.phase = SB_DETAIL_DONE; return; }
            if (c.key_output) ((int*)c.key_output)[row] = (int)-s.score;
            if (!sb_detail_align_trace(c)) { sb_detail_complete(s, row, c, o); return; }
            s.ts = 0;
            if (s.ti && s.tj) {
                const sb_i64* endpoint = s.ti == s.m ? bottom + 3 * s.tj : col + 3 * s.ti;
                sb_i64 best = endpoint[0];
                if (s.ti != s.m && endpoint[1] > best) { best = endpoint[1]; s.ts = 1; }
                if (s.tj != s.n && endpoint[2] > best) s.ts = 2;
            }
            s.pending = s.ti == s.m ? s.n - s.tj : s.m - s.ti;
            s.pending_op = s.ti == s.m ? 'D' : 'I'; s.walk_stage = 0; s.phase = 20; return;
        }
        if (s.k < s.m + s.n) {
            const sb_u64 k = ++s.k;
            const sb_i64 value = k <= s.m ?
                sb_detail_max(col[3 * (s.m - k)], col[3 * (s.m - k) + 1]) - sb_detail_gap(c, k) :
                sb_detail_max(r0[s.n - (k - s.m)], r1[s.n - (k - s.m)]) - sb_detail_gap(c, k - s.m);
            s.score = sb_detail_max(s.score, value); return;
        }
        if (c.keys && -s.score != ((const int*)c.keys)[row]) { s.error = 2; s.phase = SB_DETAIL_DONE; return; }
        if (c.key_output) ((int*)c.key_output)[row] = (int)-s.score;
        if (!sb_detail_align_trace(c)) { sb_detail_complete(s, row, c, o); return; }
        s.k = s.m; s.phase = 15; return;
    }
    if (s.phase == 15) {
        if (s.k) {
            const sb_u64 len = s.k--;
            const sb_i64 target = s.score + sb_detail_gap(c, len);
            const sb_i64* v = col + 3 * (s.m - len);
            if (sb_detail_max(v[0], v[1]) == target) {
                s.ti = s.m - len; s.tj = s.n; s.ts = v[1] == target ? 2 : 3;
                s.pending = len; s.pending_op = 'I'; s.walk_stage = 0; s.phase = 20;
            }
            return;
        }
        s.k = s.n; s.phase = 16; return;
    }
    if (s.phase == 16) {
        if (s.k) {
            const sb_u64 len = s.k--;
            const sb_i64 target = s.score + sb_detail_gap(c, len);
            if (sb_detail_max(r0[s.n - len], r1[s.n - len]) == target) {
                s.extra = len; s.k = 1; s.phase = 17;
            }
            return;
        }
        s.ti = s.m; s.tj = s.n; s.ts = 3; s.walk_stage = 0; s.phase = 20; return;
    }
    if (s.phase == 17) {
        const sb_u64 len = s.k++;
        const sb_i64 target = s.score + sb_detail_gap(c, len);
        if (r1[s.n - len] == target || len == s.extra) {
            s.ti = s.m; s.tj = s.n - len; s.ts = r1[s.n - len] == target ? 1 : 3;
            s.pending = len; s.pending_op = 'D'; s.walk_stage = 0; s.phase = 20;
        }
        return;
    }
    if (s.phase == 20) {
        if (s.pending) { moves[s.ops_at + s.op_len++] = (sb_u8)s.pending_op; --s.pending; return; }
        if (s.walk_stage == 4) { s.pending = 1; s.pending_op = 'I'; s.walk_stage = 3; return; }
        if (s.walk_stage == 1) { s.pending = s.ti; s.pending_op = 'I'; s.ti = 0; s.walk_stage = 2; return; }
        if (s.walk_stage == 2) { s.pending = s.tj; s.pending_op = 'D'; s.tj = 0; s.walk_stage = 3; return; }
        if (s.walk_stage == 3) { s.phase = 30; return; }
        if (!needle && !water) {
            if (s.ts == 3) {
                const unsigned code = codes[s.trace_at + (s.ti - 1) * s.n + s.tj - 1] & 3;
                moves[s.ops_at + s.op_len++] = 'M'; --s.ti; --s.tj;
                if (!code) {
                    if (local) { s.a_start = s.ti; s.b_start = s.tj; s.walk_stage = 3; }
                    else s.walk_stage = 1;
                } else s.ts = code;
            } else if (s.ts == 1) {
                if (!s.tj) { s.walk_stage = 1; return; }
                const unsigned code = codes[s.trace_at + (s.ti - 1) * s.n + s.tj - 1] >> 2 & 3;
                moves[s.ops_at + s.op_len++] = 'I'; --s.ti; s.ts = code;
            } else {
                if (!s.ti) { s.walk_stage = 1; return; }
                const unsigned code = codes[s.trace_at + (s.ti - 1) * s.n + s.tj - 1] >> 4 & 3;
                moves[s.ops_at + s.op_len++] = 'D'; --s.tj; s.ts = code;
            }
            return;
        }
        if (water) {
            if (s.ts == 0 && (!s.ti || !s.tj)) {
                s.a_start = s.ti; s.b_start = s.tj; s.walk_stage = 3; return;
            }
            const unsigned code = codes[s.trace_at + (s.ti - 1) * s.n + s.tj - 1];
            if (s.ts == 0) {
                if (!(code & 3)) { s.a_start = s.ti; s.b_start = s.tj; s.walk_stage = 3; }
                else if ((code & 3) == 1) { moves[s.ops_at + s.op_len++] = 'M'; --s.ti; --s.tj; }
                else s.ts = (code & 3) == 2 ? 1 : 2;
            } else if (s.ts == 1) {
                moves[s.ops_at + s.op_len++] = 'D'; --s.tj; s.ts = code & 4 ? 0 : 1;
            } else { moves[s.ops_at + s.op_len++] = 'I'; --s.ti; s.ts = code & 8 ? 0 : 2; }
            return;
        }
        if (!s.ti || !s.tj) { s.walk_stage = 1; return; }
        if (s.ts == 0) {
            moves[s.ops_at + s.op_len++] = 'M'; --s.ti; --s.tj;
            if (!s.ti || !s.tj) s.walk_stage = 1; else s.ts = 3;
        } else {
            const unsigned code = codes[s.trace_at + (s.ti - 1) * s.n + s.tj - 1];
            if (s.ts == 3) s.ts = code & 3;
            else if (s.ts == 1) {
                moves[s.ops_at + s.op_len++] = 'D'; --s.tj;
                if (!(code & 4)) { if (!s.tj) s.walk_stage = 1; else s.ts = 3; }
            } else {
                moves[s.ops_at + s.op_len++] = 'I'; --s.ti;
                if (!(code & 8)) { if (!s.ti) s.walk_stage = 1; else s.ts = 3; }
            }
        }
    }
}

__device__ __forceinline__ void sb_detail_emit(SbDetailState& s, const SbSequence& a,
                                               const SbSequence& b, sb_u64 row,
                                               const SbDetailConfig& c, const SbDetailOutput& o,
                                               const sb_u8* moves) {
    if (!s.emit_at) { s.emit_i = s.a_start; s.emit_j = s.b_start; }
    if (s.emit_at < s.op_len) {
        const sb_u8 op = moves[s.ops_at + s.op_len - 1 - s.emit_at];
        const bool diagonal = op != 'I' && op != 'D';
        sb_u8 symbol = op;
        if (diagonal && c.family) symbol = sb_detail_identity(c, a[s.emit_i], b[s.emit_j]) ? '=' : 'X';
        if (c.family) {
            if (diagonal) {
                s.identity += symbol == '=' ? 1 : 0;
                s.similarity += sb_detail_sub(c, a[s.emit_i], b[s.emit_j]) > 0 ? 1 : 0;
            } else ++s.gaps;
            if (o.writing && o.offsets[2]) {
                if (!sb_detail_room(s, row, 2, s.emit_at, 1, o)) return;
                const sb_u64 at = ((const sb_i64*)o.offsets[2])[row] + s.emit_at;
                const sb_u8* map = (const sb_u8*)c.display_map;
                if (o.payload[2]) ((sb_u8*)o.payload[2])[at] = op == 'D' ? '-' : map[a[s.emit_i]];
                if (o.payload[3]) ((sb_u8*)o.payload[3])[at] = op == 'I' ? '-' : map[b[s.emit_j]];
            }
        }
        if (s.run && s.run_op != symbol) {
            if (o.writing && o.payload[0]) {
                if (!sb_detail_room(s, row, 0, s.cigar_len, sb_detail_digits(s.run) + 1, o)) return;
                sb_detail_run((sb_u8*)o.payload[0], ((const sb_i64*)o.offsets[0])[row] + s.cigar_len,
                              s.run, (sb_u8)s.run_op);
            }
            s.cigar_len += sb_detail_digits(s.run) + 1; s.run = 0;
        }
        s.run_op = symbol; ++s.run;
        if (op != 'D') ++s.emit_i;
        if (op != 'I') ++s.emit_j;
        ++s.emit_at; return;
    }
    if (s.run) {
        if (o.writing && o.payload[0]) {
            if (!sb_detail_room(s, row, 0, s.cigar_len, sb_detail_digits(s.run) + 1, o)) return;
            sb_detail_run((sb_u8*)o.payload[0], ((const sb_i64*)o.offsets[0])[row] + s.cigar_len,
                          s.run, (sb_u8)s.run_op);
        }
        s.cigar_len += sb_detail_digits(s.run) + 1; s.run = 0;
    }
    sb_detail_complete(s, row, c, o);
}

extern "C" __global__ void sb_detail_continue(SbTables tables, SbSpace entries,
                                                SbDetailConfig c, SbDetailOutput output,
                                                SbDetailState* states, sb_u64* scratch,
                                                sb_u8* codes, sb_u8* moves, sb_u64 quantum) {
    const sb_u64 t = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= entries.n) return;
    SbDetailState s = states[t];
    const sb_u64 row = entries.k0 + t - c.output_base;
    if (!s.valid) {
        sb_detail_complete(s, row, c, output); states[t] = s; return;
    }
    sb_u64 i, j; sb_decode(entries, t, i, j);
    const SbSequence a = sb_a_row(tables, i), b = sb_b_row(tables, j);
    for (sb_u64 step = 0; step < quantum && s.phase != SB_DETAIL_DONE; ++step) {
        if (s.phase == 30) sb_detail_emit(s, a, b, row, c, output, moves);
        else if (c.family == 0) sb_detail_lev_step(s, a, b, row, c, output, scratch, codes, moves);
        else sb_detail_align_step(s, a, b, row, c, output, scratch, codes, moves);
    }
    states[t] = s;
}

extern "C" __global__ void sb_detail_progress(const SbDetailState* states, sb_u64 n,
                                               SbDetailOutput output, sb_u64* aggregate) {
    if (blockIdx.x || threadIdx.x) return;
    sb_u64 unfinished = 0, error = 0, entry = 0, totals[3] = {}, overflow = 0;
    for (sb_u64 t = 0; t < n; ++t) {
        const SbDetailState& s = states[t];
        if (s.phase != SB_DETAIL_DONE) ++unfinished;
        if (!error && s.error) { error = s.error; entry = t; }
        if (s.valid) {
            const sb_u64 counts[3] = {s.cigar_len, s.n_ends, s.op_len};
            for (unsigned k = 0; k < 3; ++k)
                if (output.counts[k] && !sb_detail_add(totals[k], counts[k])) overflow = 1;
        }
    }
    aggregate[0] = unfinished; aggregate[1] = error; aggregate[2] = entry;
    for (unsigned k = 0; k < 3; ++k) aggregate[3 + k] = totals[k];
    aggregate[6] = overflow;
}
