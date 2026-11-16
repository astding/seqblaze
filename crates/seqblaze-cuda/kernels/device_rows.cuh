// Common logical row access for packed strings and strided device matrices.
// Device pointers are never dereferenced by the host. Arithmetic is integer-only.
#pragma once
#include <stddef.h>
#include <string.h>

typedef unsigned char sb_u8;
typedef long long sb_i64;
typedef unsigned long long sb_u64;

// Must match interop::DeviceRows. Input validation precedes every compute launch.
struct SbDeviceRows {
    sb_u64 data;
    sb_u64 rows;
    sb_u64 layout;
    sb_u64 offsets;
    sb_u64 offset_width;
    sb_i64 offset_stride;
    sb_i64 row_stride;
    sb_i64 column_stride;
    sb_u64 lengths;
    sb_u64 length_width;
    sb_i64 length_stride;
    sb_u64 validity;
    sb_u64 validity_offset;
    sb_u64 columns;
};

struct SbSequence {
    sb_u64 address;
    sb_i64 stride;
    sb_i64 length;
    __device__ __forceinline__ sb_u8 operator[](sb_i64 position) const {
        return *(const sb_u8*)(address + (sb_u64)position * (sb_u64)stride);
    }
    __device__ __forceinline__ SbSequence operator+(sb_u64 position) const {
        return {address + position * (sb_u64)stride, stride,
                length - (sb_i64)position};
    }
};

__device__ __forceinline__ sb_i64 sb_row_integer(sb_u64 address, sb_u64 width) {
    const sb_u8* bytes = (const sb_u8*)address;
    // A naturally aligned 4- or 8-byte integer is one load of the same
    // little-endian bytes the loop below assembles.
    if (width == 8 && (address & 7ull) == 0) {
        sb_i64 value;
        memcpy(&value, __builtin_assume_aligned(bytes, 8), sizeof(value));
        return value;
    }
    if (width == 4 && (address & 3ull) == 0) {
        int value;
        memcpy(&value, __builtin_assume_aligned(bytes, 4), sizeof(value));
        return (sb_i64)value;
    }
    sb_u64 bits = 0;
    for (sb_u64 k = 0; k < width; ++k) bits |= (sb_u64)bytes[k] << (8 * k);
    return width == 4 ? (sb_i64)(int)bits : (sb_i64)bits;
}

__device__ __forceinline__ bool sb_row_valid(const SbDeviceRows& table, sb_u64 row) {
    if (table.validity == 0) return true;
    const sb_u64 bit = table.validity_offset + row;
    return (((const sb_u8*)table.validity)[bit >> 3] >> (bit & 7)) & 1;
}

__device__ __forceinline__ SbSequence sb_device_row(const SbDeviceRows& table, sb_u64 row) {
    if (table.layout == 0) {
        const sb_u64 at = table.offsets + row * (sb_u64)table.offset_stride;
        const sb_i64 first = sb_row_integer(at, table.offset_width);
        const sb_i64 last = sb_row_integer(at + (sb_u64)table.offset_stride, table.offset_width);
        return {table.data + (sb_u64)first, 1, last - first};
    }
    const sb_u64 first = table.data + row * (sb_u64)table.row_stride;
    if (table.layout == 1) {
        const sb_u64 at = table.lengths + row * (sb_u64)table.length_stride;
        return {first, table.column_stride, sb_row_integer(at, table.length_width)};
    }
    // Fixed-width byte strings use NumPy S<n> trailing-zero semantics.
    sb_i64 length = (sb_i64)table.columns;
    SbSequence sequence = {first, table.column_stride, length};
    while (length > 0 && sequence[length - 1] == 0) --length;
    sequence.length = length;
    return sequence;
}

__device__ __forceinline__ SbSequence sb_packed_row(sb_u64 data, sb_u64 offsets, sb_u64 row) {
    const sb_i64* off = (const sb_i64*)offsets;
    return {data + (sb_u64)off[row], 1, off[row + 1] - off[row]};
}

static_assert(sizeof(SbDeviceRows) == 112, "DeviceRows ABI size");
static_assert(alignof(SbDeviceRows) == 8, "DeviceRows ABI alignment");
#define SB_ROW_OFFSET(field, bytes) static_assert(offsetof(SbDeviceRows, field) == bytes, "DeviceRows ABI " #field)
SB_ROW_OFFSET(data, 0); SB_ROW_OFFSET(rows, 8); SB_ROW_OFFSET(layout, 16);
SB_ROW_OFFSET(offsets, 24); SB_ROW_OFFSET(offset_width, 32); SB_ROW_OFFSET(offset_stride, 40);
SB_ROW_OFFSET(row_stride, 48); SB_ROW_OFFSET(column_stride, 56); SB_ROW_OFFSET(lengths, 64);
SB_ROW_OFFSET(length_width, 72); SB_ROW_OFFSET(length_stride, 80); SB_ROW_OFFSET(validity, 88);
SB_ROW_OFFSET(validity_offset, 96); SB_ROW_OFFSET(columns, 104);
#undef SB_ROW_OFFSET
