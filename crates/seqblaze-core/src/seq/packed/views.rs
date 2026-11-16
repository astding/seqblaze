//! Arrow view layout from the Apache-2.0 columnar/C Data specifications:
//! https://arrow.apache.org/docs/format/Columnar.html#variable-size-binary-view-layout
//! https://arrow.apache.org/docs/format/CDataInterface.html#binary-view-arrays
use super::{null_rows, overflow, span, ArrowDescriptor, RowSpan};
use crate::validate::EngineError;

/// Row spans and the global null rows of a view array. The slot of a null row
/// is neither decoded nor checked (N3): its span is empty and never read.
/// # Safety
/// The descriptor and owner retain valid immutable CPU view/data/validity buffers.
pub(super) unsafe fn prepare(
    descriptor: &ArrowDescriptor,
    base_index: usize,
) -> Result<(Vec<RowSpan>, Vec<usize>), EngineError> {
    let buffers = descriptor.view_buffers.as_ref().ok_or_else(|| {
        EngineError::Parameter("Arrow view requires variadic buffer metadata".into())
    })?;
    let end = descriptor
        .offset
        .checked_add(descriptor.rows)
        .ok_or_else(overflow)?;
    let bytes = end.checked_mul(16).ok_or_else(overflow)?;
    if descriptor.rows != 0 || descriptor.address != 0 {
        span(descriptor.address, bytes)?;
    }
    for buffer in buffers.iter() {
        span(buffer.address, buffer.size)?;
    }
    // SAFETY: the retained native bitmap covers the checked visible slice.
    let nulls = unsafe {
        null_rows(
            descriptor.validity_address,
            descriptor.offset,
            descriptor.rows,
            base_index,
        )
    }?;
    let mut rows = Vec::new();
    crate::detail::try_reserve_exact(&mut rows, descriptor.rows, "Arrow view row metadata")?;
    let mut next_null = nulls.iter().peekable();
    for row in 0..descriptor.rows {
        if next_null.next_if_eq(&&(base_index + row)).is_some() {
            rows.push(RowSpan { address: 0, len: 0 });
            continue;
        }
        let slot = descriptor.address + (descriptor.offset + row) * 16;
        // SAFETY: checked geometry covers this complete 16-byte native slot.
        let length = unsafe { (slot as *const i32).read_unaligned() };
        let index = base_index + row;
        if length < 0 {
            return Err(EngineError::Parameter(format!(
                "Arrow view length at sequence {index} must be nonnegative"
            )));
        }
        let len = length as usize;
        let address = if len <= 12 {
            slot + 4
        } else {
            // SAFETY: these signed i32 fields occupy +8 and +12 in this slot.
            let buffer_index = unsafe { ((slot + 8) as *const i32).read_unaligned() };
            let offset = unsafe { ((slot + 12) as *const i32).read_unaligned() };
            if buffer_index < 0 || buffer_index as usize >= buffers.len() {
                return Err(EngineError::Parameter(format!(
                    "Arrow view buffer index at sequence {index} is out of range"
                )));
            }
            if offset < 0 {
                return Err(EngineError::Parameter(format!(
                    "Arrow view offset at sequence {index} must be nonnegative"
                )));
            }
            let buffer = buffers[buffer_index as usize];
            let offset = offset as usize;
            let end = offset.checked_add(len).ok_or_else(overflow)?;
            if end > buffer.size {
                return Err(EngineError::Parameter(format!(
                    "Arrow view span at sequence {index} exceeds its data buffer size"
                )));
            }
            buffer.address.checked_add(offset).ok_or_else(overflow)?
        };
        rows.push(RowSpan { address, len });
    }
    Ok((rows, nulls))
}
