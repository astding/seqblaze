// SHA-256 arithmetic/constants: NIST FIPS 180-4 (2015), sections 4.1.2,
// 4.2.2, 5.1.1, 5.3.3, 6.2.2; public US-government specification:
// https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf
// BLAKE3 hash mode (compression, chunks, tree, 32-byte output): J. O'Connor,
// J.-P. Aumasson, S. Neves, Z. Wilcox-O'Hearn, "BLAKE3: one function, fast
// everywhere" (2020), as restated in docs/wiki/engine/result-provenance.md.
// Original implementation of seqblaze's versioned, typed Merkle input codecs:
// merkle-blake3-v1 for sequence tables (D128) and merkle-sha256-v1 for indices.
// Independent <=64KiB stream leaves and ordered trees keep every input parallel.
#pragma once
#include "device_interop.cuh"

struct SbDigest { sb_u8 byte[32]; };
static_assert(sizeof(SbDigest) == 32, "device digest ABI");
__device__ __constant__ unsigned sb_sha256_round[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u,
};
__device__ __forceinline__ unsigned sb_sha_ror(unsigned x, unsigned n) { return (x >> n) | (x << (32 - n)); }
struct SbSha256 {
    unsigned state[8], used;
    sb_u64 bytes;
    sb_u8 block[64];
    __device__ SbSha256() : state{0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
                                0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u}, used(0), bytes(0) {}
    __device__ void compress() {
        unsigned word[16];
        for (unsigned k = 0; k < 16; ++k) {
            const unsigned at = 4 * k;
            word[k] = ((unsigned)block[at] << 24) | ((unsigned)block[at + 1] << 16) |
                      ((unsigned)block[at + 2] << 8) | block[at + 3];
        }
        unsigned a=state[0], b=state[1], c=state[2], d=state[3];
        unsigned e=state[4], f=state[5], g=state[6], h=state[7];
        for (unsigned k = 0; k < 64; ++k) {
            const unsigned slot = k & 15;
            if (k >= 16) {
                const unsigned x = word[(k + 1) & 15], y = word[(k + 14) & 15];
                const unsigned small0 = sb_sha_ror(x,7) ^ sb_sha_ror(x,18) ^ (x >> 3);
                const unsigned small1 = sb_sha_ror(y,17) ^ sb_sha_ror(y,19) ^ (y >> 10);
                word[slot] += small0 + word[(k + 9) & 15] + small1;
            }
            const unsigned choice = (e & f) ^ (~e & g);
            const unsigned majority = (a & b) ^ (a & c) ^ (b & c);
            const unsigned upper0 = sb_sha_ror(a,2) ^ sb_sha_ror(a,13) ^ sb_sha_ror(a,22);
            const unsigned upper1 = sb_sha_ror(e,6) ^ sb_sha_ror(e,11) ^ sb_sha_ror(e,25);
            const unsigned first = h + upper1 + choice + sb_sha256_round[k] + word[slot];
            const unsigned second = upper0 + majority;
            h=g; g=f; f=e; e=d+first; d=c; c=b; b=a; a=first+second;
        }
        state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
        state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
    }
    __device__ void put(sb_u8 byte) {
        block[used++] = byte; ++bytes;
        if (used == 64) { compress(); used = 0; }
    }
    __device__ void le16(unsigned value) { put((sb_u8)value); put((sb_u8)(value >> 8)); }
    __device__ void le64(sb_u64 value) { for (unsigned k=0;k<8;++k) put((sb_u8)(value >> (8*k))); }
    __device__ void digest(const SbDigest& value) { for (unsigned k=0;k<32;++k) put(value.byte[k]); }
    __device__ void frame(const char* domain, sb_u64 payload) {
        const char magic[] = "SEQBLAZE";
        for (unsigned k=0;k<9;++k) put((sb_u8)magic[k]);
        le16(1);
        unsigned length=0; while (domain[length] != 0) ++length;
        le16(length);
        for (unsigned k=0;k<length;++k) put((sb_u8)domain[k]);
        le64(payload);
    }
    __device__ SbDigest finish() {
        const sb_u64 bit_length = bytes * 8;
        put(0x80);
        while (used != 56) put(0);
        for (int k=7;k>=0;--k) put((sb_u8)(bit_length >> (8*k)));
        SbDigest result;
        for (unsigned k=0;k<8;++k) for (unsigned byte=0;byte<4;++byte)
            result.byte[4*k+byte]=(sb_u8)(state[k] >> (24-8*byte));
        return result;
    }
};
__device__ __forceinline__ SbDigest sb_digest_empty() {
    SbSha256 hash; hash.frame("logical-empty",0); return hash.finish();
}
__device__ __forceinline__ SbDigest sb_digest_node(sb_u64 left_count,const SbDigest& left,
                                                   sb_u64 right_count,const SbDigest& right) {
    SbSha256 hash; hash.frame("logical-node",80);
    hash.le64(left_count); hash.digest(left); hash.le64(right_count); hash.digest(right);
    return hash.finish();
}
// ---- BLAKE3, plain hash mode, 32-byte output (integer arithmetic only) ----
#define SB_B3_CHUNK_START 1u
#define SB_B3_CHUNK_END 2u
#define SB_B3_PARENT 4u
#define SB_B3_ROOT 8u
__device__ __constant__ unsigned sb_b3_iv[8] = {
    0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u,
};
// A chaining value: eight 32-bit words.
struct SbCv { unsigned word[8]; };
static_assert(sizeof(SbCv) == 32, "device chaining value ABI");
__device__ __forceinline__ unsigned sb_b3_ror(unsigned x, unsigned n) { return (x >> n) | (x << (32 - n)); }
__device__ __forceinline__ void sb_b3_g(unsigned& a, unsigned& b, unsigned& c, unsigned& d,
                                        unsigned x, unsigned y) {
    a = a + b + x; d = sb_b3_ror(d ^ a, 16); c = c + d; b = sb_b3_ror(b ^ c, 12);
    a = a + b + y; d = sb_b3_ror(d ^ a, 8);  c = c + d; b = sb_b3_ror(b ^ c, 7);
}
// F(h, m, t, b, d); the output chaining value h'_i = v_i ^ v_(i+8) replaces h.
__device__ __forceinline__ void sb_b3_compress(unsigned h[8], const unsigned block[16], sb_u64 counter,
                                               unsigned length, unsigned flags) {
    unsigned v[16] = {h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7],
                      sb_b3_iv[0], sb_b3_iv[1], sb_b3_iv[2], sb_b3_iv[3],
                      (unsigned)counter, (unsigned)(counter >> 32), length, flags};
    unsigned m[16];
#pragma unroll
    for (unsigned k = 0; k < 16; ++k) m[k] = block[k];
#pragma unroll
    for (unsigned round = 0; round < 7; ++round) {
        sb_b3_g(v[0], v[4], v[8],  v[12], m[0],  m[1]);
        sb_b3_g(v[1], v[5], v[9],  v[13], m[2],  m[3]);
        sb_b3_g(v[2], v[6], v[10], v[14], m[4],  m[5]);
        sb_b3_g(v[3], v[7], v[11], v[15], m[6],  m[7]);
        sb_b3_g(v[0], v[5], v[10], v[15], m[8],  m[9]);
        sb_b3_g(v[1], v[6], v[11], v[12], m[10], m[11]);
        sb_b3_g(v[2], v[7], v[8],  v[13], m[12], m[13]);
        sb_b3_g(v[3], v[4], v[9],  v[14], m[14], m[15]);
        if (round < 6) {  // m'_i = m_sigma(i), sigma = (2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8)
            const unsigned p[16] = {m[2], m[6], m[3], m[10], m[7], m[0], m[4], m[13],
                                    m[1], m[11], m[12], m[5], m[9], m[14], m[15], m[8]};
#pragma unroll
            for (unsigned k = 0; k < 16; ++k) m[k] = p[k];
        }
    }
#pragma unroll
    for (unsigned k = 0; k < 8; ++k) h[k] = v[k] ^ v[k + 8];
}
// One chunk of `length` bytes (0..1024) of a BLAKE3 input, numbered `counter`,
// whose bytes `source.next()` yields in order. `root`: the input is this chunk.
template <class Source>
__device__ void sb_b3_chunk(Source& source, sb_u64 counter, unsigned length, bool root, unsigned h[8]) {
#pragma unroll
    for (unsigned k = 0; k < 8; ++k) h[k] = sb_b3_iv[k];
    const unsigned blocks = length == 0 ? 1 : (length + 63) / 64;
    for (unsigned block = 0; block < blocks; ++block) {
        const unsigned size = block + 1 < blocks ? 64 : length - 64 * block;
        unsigned m[16];
#pragma unroll
        for (unsigned w = 0; w < 16; ++w) {  // little-endian words, zero padded
            unsigned word = 0;
#pragma unroll
            for (unsigned b = 0; b < 4; ++b)
                if (4 * w + b < size) word |= (unsigned)source.next() << (8 * b);
            m[w] = word;
        }
        unsigned flags = block == 0 ? SB_B3_CHUNK_START : 0;
        if (block + 1 == blocks) flags |= SB_B3_CHUNK_END | (root ? SB_B3_ROOT : 0);
        sb_b3_compress(h, m, counter, size, flags);
    }
}
__device__ __forceinline__ void sb_b3_parent(unsigned h[8], const SbCv& left, const SbCv& right, bool root) {
    unsigned m[16];
#pragma unroll
    for (unsigned k = 0; k < 8; ++k) { m[k] = left.word[k]; m[k + 8] = right.word[k]; h[k] = sb_b3_iv[k]; }
    sb_b3_compress(h, m, 0, 64, SB_B3_PARENT | (root ? SB_B3_ROOT : 0));
}
__device__ __forceinline__ SbDigest sb_b3_digest(const unsigned h[8]) {
    SbDigest result;
    for (unsigned k = 0; k < 8; ++k) for (unsigned byte = 0; byte < 4; ++byte)
        result.byte[4 * k + byte] = (sb_u8)(h[k] >> (8 * byte));
    return result;
}
// v1's frame head: magic, u16 version 1, u16 domain length, domain, u64 payload length.
__device__ __forceinline__ unsigned sb_frame_head(sb_u8* out, const char* domain, sb_u64 payload) {
    const char magic[] = "SEQBLAZE";
    unsigned at = 0;
    for (unsigned k = 0; k < 9; ++k) out[at++] = (sb_u8)magic[k];
    unsigned length = 0; while (domain[length] != 0) ++length;
    out[at++] = 1; out[at++] = 0; out[at++] = (sb_u8)length; out[at++] = (sb_u8)(length >> 8);
    for (unsigned k = 0; k < length; ++k) out[at++] = (sb_u8)domain[k];
    for (unsigned k = 0; k < 8; ++k) out[at++] = (sb_u8)(payload >> (8 * k));
    return at;
}
__device__ __forceinline__ unsigned sb_le64(sb_u8* out, sb_u64 value) {
    for (unsigned k = 0; k < 8; ++k) out[k] = (sb_u8)(value >> (8 * k));
    return 8;
}
struct SbBytesSource {
    const sb_u8* bytes;
    __device__ sb_u8 next() { return *bytes++; }
};
// B(domain, payload) for a frame of at most 1024 bytes: one chunk, the root.
__device__ __forceinline__ SbDigest sb_b3_small(const sb_u8* frame, unsigned length) {
    SbBytesSource source = {frame};
    unsigned h[8];
    sb_b3_chunk(source, 0, length, true, h);
    return sb_b3_digest(h);
}
__device__ __forceinline__ SbDigest sb_b3_empty() {
    sb_u8 frame[40];
    return sb_b3_small(frame, sb_frame_head(frame, "logical-empty", 0));
}
__device__ __forceinline__ SbDigest sb_b3_node(sb_u64 left_count, const SbDigest& left,
                                               sb_u64 right_count, const SbDigest& right) {
    sb_u8 frame[128];
    unsigned at = sb_frame_head(frame, "logical-node", 80);
    at += sb_le64(frame + at, left_count);
    for (unsigned k = 0; k < 32; ++k) frame[at++] = left.byte[k];
    at += sb_le64(frame + at, right_count);
    for (unsigned k = 0; k < 32; ++k) frame[at++] = right.byte[k];
    return sb_b3_small(frame, at);
}

// merkle-blake3-v1 (result-provenance.md, D128). Both streams have leaves of at
// most 65536 payload bytes: 8192 i64LE row lengths (-1 for a null row), or
// 65536 bytes of the concatenated valid rows. A leaf is B over a frame of
// head + payload bytes, head 40 ("row-lengths" with u64 value count) or 38
// ("row-bytes" with u64 byte count), so a leaf frame has at most 65 BLAKE3
// chunks. One thread hashes one chunk; sb_digest_leaf_roots folds them.
#define SB_B3_LEAF_CHUNKS 65ull
#define SB_LENGTHS_HEAD 40ull
#define SB_BYTES_HEAD 38ull
__device__ __forceinline__ sb_u64 sb_b3_frame_chunks(sb_u64 frame) { return frame == 0 ? 1 : (frame + 1023) / 1024; }
// Frame bytes of one length leaf from frame position `at`: head, then values.
struct SbLengthsSource {
    sb_u8 head[SB_LENGTHS_HEAD];
    unsigned head_at, head_end;
    const SbInputFact* facts;
    sb_u64 position;  // byte position in the length stream
    __device__ sb_u8 next() {
        if (head_at < head_end) return head[head_at++];
        const SbInputFact fact = facts[position >> 3];
        const sb_u64 value = fact.kind == 1 ? ~0ull : fact.length;  // i64LE: -1 for null
        return (sb_u8)(value >> (8 * (position++ & 7)));
    }
};
__device__ void sb_hash_row_lengths_chunk(const SbInputFact* facts, sb_u64 rows, sb_u64 leaf, sb_u64 chunk, SbCv* out) {
    const sb_u64 stream = rows * 8;
        const sb_u64 first = leaf * SB_INPUT_CHUNK;
        const sb_u64 length = stream - first < SB_INPUT_CHUNK ? stream - first : SB_INPUT_CHUNK;
        const sb_u64 frame = SB_LENGTHS_HEAD + length, chunks = sb_b3_frame_chunks(frame);
        if (chunk >= chunks) return;
        const sb_u64 at = chunk * 1024, size = frame - at < 1024 ? frame - at : 1024;
        SbLengthsSource source;
        source.facts = facts;
        source.head_at = 0; source.head_end = 0;
        if (chunk == 0) {
            const unsigned head = sb_frame_head(source.head, "row-lengths", 8 + length);
            source.head_end = head + sb_le64(source.head + head, length / 8);
            source.position = first;
        } else {
            source.position = first + at - SB_LENGTHS_HEAD;
        }
        unsigned h[8];
        sb_b3_chunk(source, chunk, (unsigned)size, chunks == 1, h);
    for (unsigned k = 0; k < 8; ++k) out->word[k] = h[k];
}
// Fused metadata init and length-leaf hash. Each hashing thread owns its
// whole row-length values; no other block reads a fact before it is written.
struct SbInitLengthsSource {
    SbLengthsSource framed;
    SbDeviceRows table;
    SbInputFact* facts;
    sb_u64* counts;
    sb_u64* lengths;
    sb_u64 value;
    __device__ sb_u8 next() {
        if (framed.head_at < framed.head_end) return framed.head[framed.head_at++];
        if ((framed.position & 7) == 0) {
            const sb_u64 row = framed.position >> 3;
            const SbInputFact fact = sb_input_fact(table, row);
            facts[row] = fact;
            counts[row] = fact.kind == 0 ? (fact.length + SB_INPUT_CHUNK - 1) / SB_INPUT_CHUNK : 0;
            lengths[row] = fact.kind == 0 ? fact.length : 0;
            value = fact.kind == 1 ? ~0ull : fact.length;
        }
        return (sb_u8)(value >> (8 * (framed.position++ & 7)));
    }
};
extern "C" __global__ void sb_input_init_digest(SbDeviceRows table, SbInputFact* facts,
    sb_u64* counts, sb_u64* lengths, SbCv* out) {
    const sb_u64 stream = table.rows * 8;
    const sb_u64 leaves = (stream + SB_INPUT_CHUNK - 1) / SB_INPUT_CHUNK;
    const sb_u64 step = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 item = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x;
         item < leaves * SB_B3_LEAF_CHUNKS; item += step) {
        const sb_u64 leaf = item / SB_B3_LEAF_CHUNKS, chunk = item % SB_B3_LEAF_CHUNKS;
        const sb_u64 first = leaf * SB_INPUT_CHUNK;
        const sb_u64 length = stream - first < SB_INPUT_CHUNK ? stream - first : SB_INPUT_CHUNK;
        const sb_u64 frame = SB_LENGTHS_HEAD + length, chunks = sb_b3_frame_chunks(frame);
        if (chunk >= chunks) continue;
        const sb_u64 at = chunk * 1024, size = frame - at < 1024 ? frame - at : 1024;
        SbInitLengthsSource source;
        source.table = table; source.facts = facts; source.counts = counts; source.lengths = lengths;
        source.framed.head_at = 0; source.framed.head_end = 0;
        if (chunk == 0) {
            const unsigned head = sb_frame_head(source.framed.head, "row-lengths", 8 + length);
            source.framed.head_end = head + sb_le64(source.framed.head + head, length / 8);
            source.framed.position = first;
        } else source.framed.position = first + at - SB_LENGTHS_HEAD;
        unsigned h[8];
        sb_b3_chunk(source, chunk, (unsigned)size, chunks == 1, h);
        for (unsigned k = 0; k < 8; ++k) out[item].word[k] = h[k];
    }
}

// One bounded block: span facts, alphabet and the exact summary. Its
// chunk counts remain unscanned until an explicit snapshot asks for them.
extern "C" __global__ void sb_input_small(SbDeviceRows table, SbInputFact* facts,
    sb_u64* counts, sb_u64* lengths, SbInputLegal legal, sb_u64 minimum,
    sb_u64 maximum, SbInputSummary* out, SbCv* length_hashes) {
    __shared__ SbInputSummary partial[SB_INPUT_REDUCE_THREADS];
    const sb_u64 row = threadIdx.x;
    SbInputSummary summary = sb_input_empty();
    if (row < table.rows) {
        SbInputFact fact = sb_input_fact(table, row);
        if (fact.kind == 0 && fact.length != 0) {
            const SbSequence sequence = sb_device_row(table, row);
            for (sb_u64 at = 0; at < fact.length; ++at) {
                const unsigned byte = sequence[(sb_i64)at];
                if (((legal.word[byte >> 6] >> (byte & 63)) & 1ull) == 0) { fact.bad_position = at; break; }
            }
        }
        facts[row] = fact;
        counts[row] = fact.kind == 0 ? (fact.length + SB_INPUT_CHUNK - 1) / SB_INPUT_CHUNK : 0;
        lengths[row] = fact.kind == 0 ? fact.length : 0;
        summary = sb_input_row_summary(table, fact, row, minimum, maximum, true);
    }
    partial[threadIdx.x] = summary;
    __syncthreads();
    if (length_hashes && threadIdx.x < sb_b3_frame_chunks(SB_LENGTHS_HEAD + table.rows * 8))
        sb_hash_row_lengths_chunk(facts, table.rows, 0, threadIdx.x, length_hashes + threadIdx.x);

    for (unsigned delta = SB_INPUT_REDUCE_THREADS / 2; delta; delta >>= 1) {
        if (threadIdx.x < delta) partial[threadIdx.x] = sb_input_merge(partial[threadIdx.x], partial[threadIdx.x + delta]);
        __syncthreads();
    }
    if (threadIdx.x == 0) out[0] = partial[0];
}

// Frame bytes of one byte leaf: head, then byte-stream positions. offsets is
// the exclusive prefix of valid lengths; null and empty rows own no positions
// and are never read.
struct SbRowBytesSource {
    sb_u8 head[SB_BYTES_HEAD];
    unsigned head_at, head_end;
    const SbDeviceRows* table;
    const sb_u64* offsets;
    sb_u64 row, position;
    SbSequence sequence;
    __device__ void seek(sb_u64 at) {  // the last row starting at or before `at`
        sb_u64 lo = 0, hi = table->rows;
        while (hi - lo > 1) { const sb_u64 mid = lo + (hi - lo) / 2; if (offsets[mid] <= at) lo = mid; else hi = mid; }
        row = lo; position = at;
        while (offsets[row + 1] <= position) ++row;
        sequence = sb_device_row(*table, row);
    }
    __device__ sb_u8 next() {
        if (head_at < head_end) return head[head_at++];
        if (offsets[row + 1] <= position) {
            do ++row; while (offsets[row + 1] <= position);
            sequence = sb_device_row(*table, row);
        }
        return sequence[(sb_i64)(position++ - offsets[row])];
    }
};
__device__ void sb_hash_row_bytes_chunk(SbDeviceRows table, const sb_u64* offsets, sb_u64 total, sb_u64 leaf, sb_u64 chunk, SbCv* out) {
        const sb_u64 first = leaf * SB_INPUT_CHUNK;
        const sb_u64 length = total - first < SB_INPUT_CHUNK ? total - first : SB_INPUT_CHUNK;
        const sb_u64 frame = SB_BYTES_HEAD + length, chunks = sb_b3_frame_chunks(frame);
        if (chunk >= chunks) return;
        const sb_u64 at = chunk * 1024, size = frame - at < 1024 ? frame - at : 1024;
        SbRowBytesSource source;
        source.table = &table; source.offsets = offsets;
        source.head_at = 0; source.head_end = 0;
        if (chunk == 0) {
            const unsigned head = sb_frame_head(source.head, "row-bytes", 8 + length);
            source.head_end = head + sb_le64(source.head + head, length);
            source.seek(first);  // length >= 1: this chunk reads stream bytes
        } else {
            source.seek(first + at - SB_BYTES_HEAD);
        }
        unsigned h[8];
        sb_b3_chunk(source, chunk, (unsigned)size, chunks == 1, h);
    for (unsigned k = 0; k < 8; ++k) out->word[k] = h[k];
}
extern "C" __global__ void sb_digest_row_bytes(SbDeviceRows table, const sb_u64* offsets, sb_u64 total, SbCv* out) {
    const sb_u64 leaves = (total + SB_INPUT_CHUNK - 1) / SB_INPUT_CHUNK;
    const sb_u64 items = leaves * SB_B3_LEAF_CHUNKS, step = (sb_u64)gridDim.x * blockDim.x;
    for (sb_u64 item = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x; item < items; item += step)
        sb_hash_row_bytes_chunk(table, offsets, total, item / SB_B3_LEAF_CHUNKS, item % SB_B3_LEAF_CHUNKS, out + item);
}
// Adjacent BLAKE3 parents are independent within a level; store each
// parent at its left child's index, then barrier before the next level.
__device__ void sb_fold_leaf(SbCv* node, sb_u64 count) {
    __syncthreads();
    for (sb_u64 width = 1; width < count; width *= 2) {
        const sb_u64 left = (sb_u64)threadIdx.x * 2 * width, right = left + width;
        if (right < count) {
            unsigned h[8];
            sb_b3_parent(h, node[left], node[right], 2 * width >= count);
            for (unsigned k = 0; k < 8; ++k) node[left].word[k] = h[k];
        }
        __syncthreads();
    }
}
extern "C" __global__ void sb_digest_leaf_roots(const SbCv* chunks, sb_u64 stream, sb_u64 head, SbDigest* out) {
    __shared__ SbCv node[SB_B3_LEAF_CHUNKS];
    const sb_u64 leaves = (stream + SB_INPUT_CHUNK - 1) / SB_INPUT_CHUNK;
    for (sb_u64 leaf = blockIdx.x; leaf < leaves; leaf += gridDim.x) {
        const sb_u64 remaining = stream - leaf * SB_INPUT_CHUNK;
        const sb_u64 length = remaining < SB_INPUT_CHUNK ? remaining : SB_INPUT_CHUNK;
        const sb_u64 count = sb_b3_frame_chunks(head + length);
        if (threadIdx.x < count) node[threadIdx.x] = chunks[leaf * SB_B3_LEAF_CHUNKS + threadIdx.x];
        sb_fold_leaf(node, count);
        if (threadIdx.x == 0) out[leaf] = sb_b3_digest(node[0].word);
        __syncthreads();
    }
}
// Fuse up to seven exact ordered-tree levels in a 128-thread block. A
// partial right subtree keeps its original leaf count in every node frame.
template<bool Table>
__device__ void sb_digest_tree_block(SbDigest* values, sb_u64 count, sb_u64 width, SbDigest* node) {
    const sb_u64 groups = (count + 128 * width - 1) / (128 * width);
    for (sb_u64 group = blockIdx.x; group < groups; group += gridDim.x) {
        const sb_u64 first = group * 128 * width;
        const sb_u64 at = first + (sb_u64)threadIdx.x * width;
        if (at < count) node[threadIdx.x] = values[at];
        __syncthreads();
        for (unsigned delta = 1; delta < 128; delta *= 2) {
            const unsigned left = threadIdx.x * 2 * delta, right = left + delta;
            const sb_u64 start = first + (sb_u64)right * width;
            if (right < 128 && start < count) {
                const sb_u64 size = (sb_u64)delta * width;
                const sb_u64 right_count = count - start < size ? count - start : size;
                node[left] = Table ? sb_b3_node(size, node[left], right_count, node[right])
                                   : sb_digest_node(size, node[left], right_count, node[right]);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) values[first] = node[0];
        __syncthreads();
    }
}
extern "C" __global__ void sb_digest_table_tree_level(SbDigest* values, sb_u64 count, sb_u64 width) {
    __shared__ SbDigest node[128];
    sb_digest_tree_block<true>(values, count, width, node);
}
extern "C" __global__ void sb_digest_tree_level(SbDigest* values, sb_u64 count, sb_u64 width) {
    __shared__ SbDigest node[128];
    sb_digest_tree_block<false>(values, count, width, node);
}
__device__ SbDigest sb_sequence_root(sb_u64 rows, sb_u64 total, SbDigest length_root, SbDigest byte_root) {
    sb_u8 frame[128];
    unsigned at = sb_frame_head(frame, "sequence-table-blake3", 80);
    at += sb_le64(frame + at, rows); at += sb_le64(frame + at, total);
    for (unsigned k = 0; k < 32; ++k) frame[at++] = length_root.byte[k];
    for (unsigned k = 0; k < 32; ++k) frame[at++] = byte_root.byte[k];
    return sb_b3_small(frame, at);
}
// Both logical streams fit one leaf. One block hashes their independent
// chunks, folds their roots, and writes the same domain-separated table root.
extern "C" __global__ void sb_digest_sequence_small(SbDeviceRows table, const SbInputFact* facts,
    const sb_u64* offsets, sb_u64 total, const SbCv* known_lengths, SbDigest* out) {
    __shared__ SbCv lengths[SB_B3_LEAF_CHUNKS], bytes[SB_B3_LEAF_CHUNKS];
    __shared__ SbDigest length_root, byte_root;
    const sb_u64 lc = sb_b3_frame_chunks(SB_LENGTHS_HEAD + table.rows * 8);
    const sb_u64 bc = sb_b3_frame_chunks(SB_BYTES_HEAD + total);
    if (table.rows != 0 && threadIdx.x < lc) {
        if (known_lengths) lengths[threadIdx.x] = known_lengths[threadIdx.x];
        else sb_hash_row_lengths_chunk(facts, table.rows, 0, threadIdx.x, lengths + threadIdx.x);
    }
    if (total != 0 && threadIdx.x < bc) sb_hash_row_bytes_chunk(table, offsets, total, 0, threadIdx.x, bytes + threadIdx.x);
    sb_fold_leaf(lengths, table.rows == 0 ? 0 : lc);
    sb_fold_leaf(bytes, total == 0 ? 0 : bc);
    if (threadIdx.x == 0) {
        length_root = table.rows == 0 ? sb_b3_empty() : sb_b3_digest(lengths[0].word);
        byte_root = total == 0 ? sb_b3_empty() : sb_b3_digest(bytes[0].word);
        out[0] = sb_sequence_root(table.rows, total, length_root, byte_root);
    }
}
extern "C" __global__ void sb_digest_sequence_table(const SbDigest* lengths,sb_u64 length_leaves,
                                                      const SbDigest* bytes,sb_u64 byte_leaves,
                                                      sb_u64 rows,sb_u64 total,SbDigest* out) {
    if(blockIdx.x!=0||threadIdx.x!=0) return;
    sb_u8 frame[128];
    unsigned at = sb_frame_head(frame, "sequence-table-blake3", 80);
    at += sb_le64(frame + at, rows);
    at += sb_le64(frame + at, total);
    const SbDigest length_root = length_leaves == 0 ? sb_b3_empty() : lengths[0];
    const SbDigest byte_root = byte_leaves == 0 ? sb_b3_empty() : bytes[0];
    for (unsigned k = 0; k < 32; ++k) frame[at++] = length_root.byte[k];
    for (unsigned k = 0; k < 32; ++k) frame[at++] = byte_root.byte[k];
    out[0] = sb_b3_small(frame, at);
}
extern "C" __global__ void sb_digest_index_chunks(const sb_i64* values,sb_u64 count,SbDigest* out) {
    const sb_u64 chunks=(count+8191)/8192,step=(sb_u64)gridDim.x*blockDim.x;
    for(sb_u64 chunk=(sb_u64)blockIdx.x*blockDim.x+threadIdx.x;chunk<chunks;chunk+=step) {
        const sb_u64 begin=chunk*8192,length=count-begin<8192?count-begin:8192;
        SbSha256 hash;hash.frame("index-chunk",8+8*length);hash.le64(length);
        for(sb_u64 k=0;k<length;++k) hash.le64((sb_u64)values[begin+k]);
        out[chunk]=hash.finish();
    }
}
extern "C" __global__ void sb_digest_pair_index(const SbDigest* chunks,sb_u64 count,SbDigest* out) {
    if(blockIdx.x!=0||threadIdx.x!=0) return;
    SbSha256 hash;hash.frame("pair-index",40);hash.le64(count);
    hash.digest(count==0?sb_digest_empty():chunks[0]);out[0]=hash.finish();
}
