//! Immutable Arrow binary/string buffers with checked derived offset metadata.
use super::view::{load_byte, LengthDescriptor};
use crate::validate::EngineError;
use std::{any::Any, fmt, sync::Arc};
mod views;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArrowBuffer {
    pub address: usize,
    pub size: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArrowDescriptor {
    pub address: usize,
    pub rows: usize,
    pub offsets_address: usize,
    pub offsets_itemsize: usize,
    pub offset: usize,
    pub validity_address: usize,
    pub format: &'static str,
    /// The variadic data buffers of a view array, shared without a copy (a
    /// fixed-size `Arc` over the fallibly reserved list; F4 round 3).
    pub view_buffers: Option<Arc<Vec<ArrowBuffer>>>,
}

pub(super) struct RowSpan {
    pub(super) address: usize,
    pub(super) len: usize,
}
enum Rows {
    Offsets(Vec<usize>),
    Views(Vec<RowSpan>),
}

pub(super) struct PackedBytes {
    pub(super) descriptor: ArrowDescriptor,
    rows: Rows,
    /// Global row numbers (base index included) of the visible null rows,
    /// ascending. Their spans are empty: null slots are never read (N2).
    pub(super) nulls: Arc<Vec<usize>>,
    /// Global number of row 0 (the offset of this chunk in a stream).
    base_index: usize,
    _owner: Box<dyn Any + Send + Sync>,
}
impl fmt::Debug for PackedBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PackedBytes")
            .field("descriptor", &self.descriptor)
            .finish_non_exhaustive()
    }
}
fn span(address: usize, size: usize) -> Result<(), EngineError> {
    if size != 0
        && (address == 0 || size > isize::MAX as usize || address.checked_add(size).is_none())
    {
        return Err(EngineError::Overflow(
            "Arrow buffer address/span exceeds platform limits".into(),
        ));
    }
    Ok(())
}
fn overflow() -> EngineError {
    EngineError::Overflow("Arrow offset metadata exceeds platform limits".into())
}

/// Visible rows whose validity bit is 0, as `base_index + row`, ascending.
/// The bitmap is authoritative (`null_count` is not consulted); an absent
/// bitmap (address 0) means no nulls (Arrow Columnar Format, "Validity bitmaps").
///
/// # Safety
/// A nonzero `validity_address` must point to a retained bitmap covering bits
/// `offset..offset + rows`.
pub(super) unsafe fn null_rows(
    validity_address: usize,
    offset: usize,
    rows: usize,
    base_index: usize,
) -> Result<Vec<usize>, EngineError> {
    let mut nulls = Vec::new();
    if validity_address == 0 {
        return Ok(nulls);
    }
    let end = offset.checked_add(rows).ok_or_else(overflow)?;
    span(validity_address, end.div_ceil(8))?;
    for row in 0..rows {
        let bit = offset + row;
        // SAFETY: checked geometry; the CPU export retains this bitmap.
        let byte = unsafe { load_byte((validity_address as *const u8).wrapping_add(bit / 8)) };
        if byte & (1 << (bit % 8)) == 0 {
            crate::detail::try_reserve(&mut nulls, 1, "Arrow null row metadata")?;
            nulls.push(base_index + row);
        }
    }
    Ok(nulls)
}
impl PackedBytes {
    /// # Safety
    /// The CPU Arrow allocations must be valid, retained and immutable.
    pub(super) unsafe fn new(
        descriptor: ArrowDescriptor,
        owner: Box<dyn Any + Send + Sync>,
    ) -> Result<Self, EngineError> {
        // SAFETY: the caller supplies the same native allocation guarantees.
        unsafe { Self::prepare(descriptor, owner, 0) }
    }

    /// # Safety
    /// The exporter supplies valid CPU Arrow allocations and a retained owner.
    /// All buffers remain immutable while this imported table is retained.
    pub(super) unsafe fn prepare(
        descriptor: ArrowDescriptor,
        owner: Box<dyn Any + Send + Sync>,
        base_index: usize,
    ) -> Result<Self, EngineError> {
        if matches!(descriptor.format, "vu" | "vz") {
            // SAFETY: same retained immutable CPU export as the offset path.
            let (rows, nulls) = unsafe { views::prepare(&descriptor, base_index) }?;
            return Ok(Self {
                descriptor,
                rows: Rows::Views(rows),
                nulls: Arc::new(nulls),
                base_index,
                _owner: owner,
            });
        }
        let ArrowDescriptor {
            rows,
            offset,
            offsets_address,
            offsets_itemsize,
            validity_address,
            format,
            ..
        } = descriptor;
        let width = match format {
            "z" | "u" => 4,
            "Z" | "U" => 8,
            _ => {
                return Err(EngineError::Parameter(
                    "Arrow inputs require binary/large_binary/string/large_string".into(),
                ))
            }
        };
        if offsets_itemsize != width {
            return Err(EngineError::Parameter(
                "Arrow offsets width does not match schema".into(),
            ));
        }
        let end = offset.checked_add(rows).ok_or_else(overflow)?;
        let count = end.checked_add(1).ok_or_else(overflow)?;
        let offset_bytes = count.checked_mul(width).ok_or_else(overflow)?;
        // An empty export may omit buffers. There is then nothing to dereference.
        if rows != 0 || offsets_address != 0 {
            span(offsets_address, offset_bytes)?;
        }
        // SAFETY: the export retains the bitmap of the visible slice.
        let nulls = unsafe { null_rows(validity_address, offset, rows, base_index) }?;
        let mut offsets = Vec::new();
        crate::detail::try_reserve_exact(
            &mut offsets,
            rows.checked_add(1).ok_or_else(overflow)?,
            "Arrow derived offset metadata",
        )?;
        if rows == 0 && offsets_address == 0 {
            offsets.push(0);
        } else {
            let source = LengthDescriptor {
                address: offsets_address,
                rows: count,
                itemsize: width,
                stride: width as isize,
            };
            for row in 0..=rows {
                // SAFETY: contiguous native offsets geometry is checked above;
                // Arrow's exporter retains the complete declared allocation.
                let value = unsafe { source.read(offset + row) };
                if value < 0 || usize::try_from(value).is_err() {
                    return Err(EngineError::Parameter(format!(
                        "Arrow offset at position {row} must be nonnegative and platform-sized"
                    )));
                }
                let value = value as usize;
                if offsets.last().is_some_and(|&previous| value < previous) {
                    return Err(EngineError::Parameter(format!(
                        "Arrow offsets decrease at position {row}"
                    )));
                }
                offsets.push(value);
            }
        }
        // Arrow C Data supplies logical buffer lengths through offsets, and its
        // native exporter guarantees the underlying buffer covers this endpoint.
        span(descriptor.address, *offsets.last().unwrap())?;
        // Every offset, null slots included, was checked above (N3). A null
        // slot may span an undefined region of the data buffer (N2): the
        // offsets stay as they are and `start`/`seq_len` give a null row an
        // empty span, so the null record costs O(n_null) (N5).
        let rows = Rows::Offsets(offsets);
        Ok(Self {
            descriptor,
            rows,
            nulls: Arc::new(nulls),
            base_index,
            _owner: owner,
        })
    }
    pub(super) fn into_owner(self) -> Box<dyn Any + Send + Sync> {
        self._owner
    }
    /// Whether `row` of an offset layout is null (view layouts store null
    /// rows as empty spans). O(1) without nulls, O(log n_null) with them.
    fn null_offset_row(&self, row: usize) -> bool {
        !self.nulls.is_empty() && self.nulls.binary_search(&(self.base_index + row)).is_ok()
    }
    pub(super) fn seq_len(&self, row: usize) -> usize {
        match &self.rows {
            Rows::Offsets(_) if self.null_offset_row(row) => 0,
            Rows::Offsets(offsets) => offsets[row + 1] - offsets[row],
            Rows::Views(rows) => rows[row].len,
        }
    }
    pub(super) fn start(&self, row: usize) -> usize {
        match &self.rows {
            Rows::Offsets(_) if self.null_offset_row(row) => 0,
            Rows::Offsets(offsets) => self.descriptor.address.wrapping_add(offsets[row]),
            Rows::Views(rows) => rows[row].address,
        }
    }
}
