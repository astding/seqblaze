//! One bulk UTF-8 conversion of native fixed-width UCS4 rows into final storage
//! (D86 (3); text-nulls-fp-environment.md §2 E1, E4, E7). NumPy U storage and
//! padding: docs/wiki/engine/engine-layout.md.
use super::view::FixedBytes;
use crate::{batch::SeqBatch, validate::EngineError};

fn overflow() -> EngineError {
    EngineError::Overflow("NumPy Unicode converted storage exceeds platform/int64 limits".into())
}
fn codepoint(source: &FixedBytes, row: usize, position: usize, big_endian: bool) -> u32 {
    let offset = position * 4;
    let bytes = [
        source.read(row, offset),
        source.read(row, offset + 1),
        source.read(row, offset + 2),
        source.read(row, offset + 3),
    ];
    if big_endian {
        u32::from_be_bytes(bytes)
    } else {
        u32::from_le_bytes(bytes)
    }
}
/// The number of code points of `row` after removing its trailing U+0000 padding.
fn row_length(source: &FixedBytes, row: usize, big_endian: bool) -> usize {
    let mut length = source.width / 4;
    while length != 0 && codepoint(source, row, length - 1, big_endian) == 0 {
        length -= 1;
    }
    length
}
/// E4: a surrogate or a value above U+10FFFF has no UTF-8 encoding.
fn not_encodable(row: usize, position: usize, value: u32) -> EngineError {
    let what = if value > 0x10FFFF {
        "a value above U+10FFFF"
    } else {
        "a surrogate code point"
    };
    EngineError::Parameter(format!(
        "NumPy Unicode text at sequence {row}, code point position {position} is U+{value:04X}, \
         {what}, which has no UTF-8 encoding; pass valid text or the intended bytes"
    ))
}
/// Rows become the UTF-8 encoding of their code points: a first pass checks
/// every code point and sums the UTF-8 lengths (checked), then one fallible
/// reservation, then the encoding into the final buffer. The output never
/// exceeds the UCS-4 input size (at most 4 bytes per code point).
pub(super) fn convert(source: &FixedBytes, big_endian: bool) -> Result<SeqBatch, EngineError> {
    if source.width % 4 != 0 {
        return Err(EngineError::Parameter(
            "NumPy Unicode items must contain four-byte codepoints".into(),
        ));
    }
    let count = source.rows.checked_add(1).ok_or_else(overflow)?;
    count
        .checked_mul(std::mem::size_of::<i64>())
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(overflow)?;
    let mut offsets = Vec::new();
    crate::detail::try_reserve_exact(&mut offsets, count, "NumPy Unicode converted offsets")?;
    offsets.push(0i64);
    let mut total = 0usize;
    // Exact final byte capacity, without allocating a packed UCS4 copy.
    for row in 0..source.rows {
        let mut bytes = 0usize;
        for position in 0..row_length(source, row, big_endian) {
            let value = codepoint(source, row, position, big_endian);
            let width = char::from_u32(value)
                .ok_or_else(|| not_encodable(row, position, value))?
                .len_utf8();
            bytes += width; // at most 4 per code point: below the row's UCS4 width
        }
        total = total
            .checked_add(bytes)
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(overflow)?;
        offsets.push(i64::try_from(total).map_err(|_| overflow())?);
    }
    let mut data = Vec::new();
    crate::detail::try_reserve_exact(&mut data, total, "NumPy Unicode converted bytes")?;
    let mut buffer = [0u8; 4];
    let changed = |row: usize| {
        EngineError::Parameter(format!(
            "NumPy Unicode row {row} changed during conversion; do not modify inputs during a call"
        ))
    };
    for row in 0..source.rows {
        let end = offsets[row + 1] as usize;
        for position in 0..source.width / 4 {
            if data.len() == end {
                break;
            }
            let value = codepoint(source, row, position, big_endian);
            // Checked by the first pass; a concurrent mutation is a contract
            // violation and is reported, never encoded as replacement bytes.
            let character =
                char::from_u32(value).ok_or_else(|| not_encodable(row, position, value))?;
            let encoded = character.encode_utf8(&mut buffer).as_bytes();
            if data.len() + encoded.len() > end {
                return Err(changed(row));
            }
            data.extend_from_slice(encoded);
        }
        if data.len() != end {
            return Err(changed(row));
        }
    }
    SeqBatch::try_from_parts(data, offsets)
        .map_err(|message| EngineError::Parameter(message.into()))
}
