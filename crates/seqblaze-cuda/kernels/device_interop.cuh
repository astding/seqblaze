// Shared device-input validation and integer prefix scan. Validation reports
// contain control facts only; sequence bytes and per-row metadata stay on GPU.
#pragma once
#include "device_rows.cuh"

#define SB_INPUT_CHUNK 65536ull
#define SB_INPUT_NONE (~0ull)
#define SB_INPUT_REDUCE_THREADS 128

struct SbInputFact { sb_u64 length, bad_position, byte, kind; };
struct SbInputSummary {
    sb_u64 rows, valid, total, min_length, min_row, max_length, max_row;
    sb_u64 bad_count, bad_row, bad_position, bad_byte, bad_rows[5];
    sb_u64 span_row, length_row, length_value, length_bound;
};
struct SbInputLegal { sb_u64 word[4]; };
static_assert(sizeof(SbInputFact) == 32, "device input fact ABI");
static_assert(sizeof(SbInputSummary) == 160, "device input summary ABI");

// Metadata only: validate a row's span before any byte dereference.
__device__ __forceinline__ SbInputFact sb_input_fact(SbDeviceRows table, sb_u64 row) {
        SbInputFact fact = {0, SB_INPUT_NONE, 0, 0};
        if (!sb_row_valid(table, row)) {
            fact.kind = 1;  // null: no sequence or offset payload is read
            fact.bad_position = 0;
        } else if (table.layout == 0) {
            const sb_u64 at = table.offsets + row * (sb_u64)table.offset_stride;
            const sb_i64 first = sb_row_integer(at, table.offset_width);
            const sb_i64 last = sb_row_integer(at + (sb_u64)table.offset_stride, table.offset_width);
            if (first < 0 || last < first || (sb_u64)last > table.columns) {fact.kind = 2;fact.length=(sb_u64)first;fact.byte=(sb_u64)last;}
            else fact.length = (sb_u64)(last - first);
        } else if (table.layout == 1) {
            const sb_i64 length = sb_row_integer(table.lengths + row * (sb_u64)table.length_stride, table.length_width);
            if (length < 0 || (sb_u64)length > table.columns) {fact.kind = 2;fact.length=(sb_u64)length;}
            else fact.length = (sb_u64)length;
        } else if (table.layout == 2) {
            fact.length = (sb_u64)sb_device_row(table, row).length;
        } else fact.kind = 2;
    return fact;
}
extern "C" __global__ void sb_input_init(SbDeviceRows table, SbInputFact* facts,
                                          sb_u64* chunk_counts, sb_u64* valid_lengths) {
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 row = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; row < table.rows; row += step) {
        const SbInputFact fact = sb_input_fact(table, row);
        facts[row] = fact;
        chunk_counts[row] = fact.kind == 0 ? (fact.length + SB_INPUT_CHUNK - 1) / SB_INPUT_CHUNK : 0;
        valid_lengths[row] = fact.kind == 0 ? fact.length : 0;
    }
}

// Inclusive block scan into output[i+1]; output[0]=0. The host recursively
// scans block totals and adds their exclusive prefix, without downloading them.
extern "C" __global__ void sb_input_scan_u64(const sb_u64* input, sb_u64* output,
                                              sb_u64* totals, sb_u64 n) {
    __shared__ sb_u64 value[256];
    const unsigned lane = threadIdx.x;
    const sb_u64 at = (sb_u64)blockIdx.x * 256 + lane;
    value[lane] = at < n ? input[at] : 0;
    __syncthreads();
    for (unsigned delta = 1; delta < 256; delta <<= 1) {
        const sb_u64 previous = lane >= delta ? value[lane - delta] : 0;
        __syncthreads();
        value[lane] += previous;
        __syncthreads();
    }
    if (at < n) output[at + 1] = value[lane];
    if (lane == 255) totals[blockIdx.x] = value[lane];
    if (blockIdx.x == 0 && lane == 0) output[0] = 0;
}
extern "C" __global__ void sb_input_scan_add(sb_u64* output, const sb_u64* blocks, sb_u64 n) {
    const sb_u64 at = (sb_u64)blockIdx.x * 256 + threadIdx.x;
    if (at < n) output[at + 1] += blocks[blockIdx.x];
}

// One bounded (<=64KiB) logical sequence chunk per block. Arbitrarily long
// rows never turn validation into one unbounded thread or one unbounded block.
// The chunk count is the device-resident end of the prefix, so the host sizes
// the grid with an upper bound and never reads the count; any grid gives the
// same facts (integer atomicMin).
extern "C" __global__ void sb_input_alphabet(SbDeviceRows table, SbInputFact* facts,
                                               const sb_u64* prefix, SbInputLegal legal) {
    __shared__ sb_u64 first[128];
    const sb_u64 chunks = prefix[table.rows];
    for (sb_u64 chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x) {
        sb_u64 lo = 0, hi = table.rows;
        while (hi - lo > 1) {
            const sb_u64 mid = lo + (hi - lo) / 2;
            if (prefix[mid] <= chunk) lo = mid; else hi = mid;
        }
        const sb_u64 begin = (chunk - prefix[lo]) * SB_INPUT_CHUNK;
        const sb_u64 end = begin + SB_INPUT_CHUNK < facts[lo].length ? begin + SB_INPUT_CHUNK : facts[lo].length;
        const SbSequence sequence = sb_device_row(table, lo);
        sb_u64 bad = SB_INPUT_NONE;
        for (sb_u64 position = begin + threadIdx.x; position < end; position += 128) {
            const unsigned byte = sequence[(sb_i64)position];
            if (((legal.word[byte >> 6] >> (byte & 63)) & 1ull) == 0) {
                bad = position;
                break;
            }
        }
        first[threadIdx.x] = bad;
        __syncthreads();
        for (unsigned delta = 64; delta != 0; delta >>= 1) {
            if (threadIdx.x < delta && first[threadIdx.x + delta] < first[threadIdx.x])
                first[threadIdx.x] = first[threadIdx.x + delta];
            __syncthreads();
        }
        if (threadIdx.x == 0 && first[0] != SB_INPUT_NONE)
            atomicMin(&facts[lo].bad_position, first[0]);
        __syncthreads();
    }
}

extern "C" __global__ void sb_input_alphabet_rows(SbDeviceRows table, SbInputFact* facts, SbInputLegal legal) {
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 row = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; row < table.rows; row += step) {
        if (facts[row].kind != 0 || facts[row].length > 64) continue;
        const SbSequence sequence = sb_device_row(table, row);
        for (sb_u64 at = 0; at < facts[row].length; ++at) {
            const unsigned byte = sequence[(sb_i64)at];
            if (((legal.word[byte >> 6] >> (byte & 63)) & 1ull) == 0) {
                atomicMin(&facts[row].bad_position, at); break;
            }
        }
    }
    const unsigned lane = threadIdx.x & 31;
    const sb_u64 warps = step / 32;
    for (sb_u64 row = ((sb_u64)blockIdx.x * blockDim.x + threadIdx.x) / 32;
         row < table.rows; row += warps) {
        if (facts[row].kind != 0 || facts[row].length <= 64) continue;
        const SbSequence sequence = sb_device_row(table, row);
        sb_u64 first = SB_INPUT_NONE;
        for (sb_u64 at = lane; at < facts[row].length; at += 32) {
            const unsigned byte = sequence[(sb_i64)at];
            if (((legal.word[byte >> 6] >> (byte & 63)) & 1ull) == 0) { first = at; break; }
        }
        for (unsigned delta = 16; delta; delta >>= 1) {
            const sb_u64 other = __shfl_down_sync(0xffffffffu, first, delta);
            if (other < first) first = other;
        }
        if (lane == 0 && first != SB_INPUT_NONE) atomicMin(&facts[row].bad_position, first);
    }
}

__device__ __forceinline__ SbInputSummary sb_input_empty() {
    SbInputSummary value = {};
    value.min_length = value.min_row = value.max_row = SB_INPUT_NONE;
    value.bad_row = value.bad_position = value.span_row = value.length_row = SB_INPUT_NONE;
    for (int k = 0; k < 5; ++k) value.bad_rows[k] = SB_INPUT_NONE;
    return value;
}
__device__ __forceinline__ SbInputSummary sb_input_merge(SbInputSummary a, const SbInputSummary& b) {
    a.rows += b.rows; a.valid += b.valid; a.total += b.total; a.bad_count += b.bad_count;
    if (b.min_length < a.min_length || (b.min_length == a.min_length && b.min_row < a.min_row)) {
        a.min_length = b.min_length; a.min_row = b.min_row;
    }
    if (b.max_length > a.max_length || (b.max_length == a.max_length && b.max_row < a.max_row)) {
        a.max_length = b.max_length; a.max_row = b.max_row;
    }
    if (b.bad_row < a.bad_row) {
        a.bad_row = b.bad_row; a.bad_position = b.bad_position; a.bad_byte = b.bad_byte;
    }
    if (b.span_row < a.span_row) a.span_row = b.span_row;
    if (b.length_row < a.length_row) {
        a.length_row = b.length_row; a.length_value = b.length_value; a.length_bound = b.length_bound;
    }
    sb_u64 smallest[5];
    int left = 0, right = 0;
    for (int k = 0; k < 5; ++k) {
        if (left < 5 && (right == 5 || a.bad_rows[left] < b.bad_rows[right])) smallest[k] = a.bad_rows[left++];
        else smallest[k] = b.bad_rows[right++];
    }
    for (int k = 0; k < 5; ++k) a.bad_rows[k] = smallest[k];
    return a;
}
__device__ __forceinline__ SbInputSummary sb_input_row_summary(SbDeviceRows table, const SbInputFact& fact,
                                                                sb_u64 row, sb_u64 minimum,
                                                                sb_u64 maximum, bool alphabet) {
    SbInputSummary value = sb_input_empty();
    value.rows = 1;
    if (fact.kind == 2) { value.span_row = row; return value; }
    if (fact.kind == 0) {
        value.valid = 1; value.total = fact.length;
        value.min_length = value.max_length = fact.length;
        value.min_row = value.max_row = row;
        if (fact.length < minimum || fact.length > maximum) {
            value.length_row = row; value.length_value = fact.length;
            value.length_bound = fact.length < minimum ? minimum : maximum;
        }
    }
    if (fact.kind == 1 || (alphabet && fact.bad_position != SB_INPUT_NONE)) {
        value.bad_count = 1; value.bad_row = row; value.bad_rows[0] = row;
        value.bad_position = fact.kind == 1 ? SB_INPUT_NONE : fact.bad_position;
        if (fact.kind == 0) value.bad_byte = sb_device_row(table, row)[(sb_i64)fact.bad_position];
    }
    return value;
}

// The two reductions use identical exact associative integer summaries. The
// second pass has <=1024 summaries; no sequence payload returns to the host.
extern "C" __global__ void sb_input_reduce_rows(SbDeviceRows table, const SbInputFact* facts,
                                                  SbInputSummary* out, sb_u64 minimum,
                                                  sb_u64 maximum, unsigned alphabet) {
    __shared__ SbInputSummary partial[SB_INPUT_REDUCE_THREADS];
    SbInputSummary mine = sb_input_empty();
    const sb_u64 step = (sb_u64)gridDim.x * SB_INPUT_REDUCE_THREADS;
    for (sb_u64 row = (sb_u64)blockIdx.x * SB_INPUT_REDUCE_THREADS + threadIdx.x; row < table.rows; row += step)
        mine = sb_input_merge(mine, sb_input_row_summary(table, facts[row], row, minimum, maximum, alphabet != 0));
    partial[threadIdx.x] = mine;
    __syncthreads();
    for (unsigned delta = SB_INPUT_REDUCE_THREADS / 2; delta != 0; delta >>= 1) {
        if (threadIdx.x < delta) partial[threadIdx.x] = sb_input_merge(partial[threadIdx.x], partial[threadIdx.x + delta]);
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = partial[0];
}
extern "C" __global__ void sb_input_reduce_summaries(const SbInputSummary* input, SbInputSummary* out, sb_u64 n) {
    __shared__ SbInputSummary partial[SB_INPUT_REDUCE_THREADS];
    SbInputSummary mine = sb_input_empty();
    for (sb_u64 row = threadIdx.x; row < n; row += SB_INPUT_REDUCE_THREADS)
        mine = sb_input_merge(mine, input[row]);
    partial[threadIdx.x] = mine;
    __syncthreads();
    for (unsigned delta = SB_INPUT_REDUCE_THREADS / 2; delta != 0; delta >>= 1) {
        if (threadIdx.x < delta) partial[threadIdx.x] = sb_input_merge(partial[threadIdx.x], partial[threadIdx.x + delta]);
        __syncthreads();
    }
    if (threadIdx.x == 0) out[0] = partial[0];
}

// Explicit repro snapshot only: create one owned logical packed table on the
// device before any host serialization. Null and empty rows stay distinct.
extern "C" __global__ void sb_input_snapshot_lengths(const SbInputFact* facts,sb_u64 n,
                                                        sb_u64* lengths,sb_u8* validity) {
    const sb_u64 step=(sb_u64)gridDim.x*blockDim.x;
    for(sb_u64 row=(sb_u64)blockIdx.x*blockDim.x+threadIdx.x;row<n;row+=step) {
        lengths[row]=facts[row].kind==0?facts[row].length:0;
        validity[row]=facts[row].kind==1?0:1;
    }
}
extern "C" __global__ void sb_input_snapshot_bytes(SbDeviceRows table,const SbInputFact* facts,
                                                      const sb_u64* chunks,
                                                      const sb_u64* offsets,sb_u8* output) {
    const sb_u64 count=chunks[table.rows];
    for(sb_u64 chunk=blockIdx.x;chunk<count;chunk+=gridDim.x) {
        sb_u64 lo=0,hi=table.rows;
        while(hi-lo>1) { const sb_u64 mid=lo+(hi-lo)/2;if(chunks[mid]<=chunk)lo=mid;else hi=mid; }
        const sb_u64 begin=(chunk-chunks[lo])*SB_INPUT_CHUNK;
        const sb_u64 end=begin+SB_INPUT_CHUNK<facts[lo].length?begin+SB_INPUT_CHUNK:facts[lo].length;
        const SbSequence sequence=sb_device_row(table,lo);
        for(sb_u64 position=begin+threadIdx.x;position<end;position+=blockDim.x)
            output[offsets[lo]+position]=sequence[(sb_i64)position];
    }
}
