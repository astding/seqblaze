//! Device-resident array ownership and row descriptors shared by every kernel.
//!
//! The protocol adapters keep producer objects alive; a descriptor never turns
//! a device address into a host slice. Only explicit output export copies data.
use std::{any::Any, sync::Arc};

use seqblaze_core::validate::EngineError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceDType {
    U8,
    I8,
    I16,
    U16,
    U32,
    U64,
    Bool,
    I32,
    I64,
    F64,
}

impl DeviceDType {
    pub fn width(self) -> usize {
        match self {
            Self::U8 | Self::I8 | Self::Bool => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }
    pub fn typestr(self) -> &'static str {
        match self {
            Self::U8 => "|u1",
            Self::I8 => "|i1",
            Self::I16 => "<i2",
            Self::U16 => "<u2",
            Self::U32 => "<u4",
            Self::U64 => "<u8",
            Self::Bool => "|b1",
            Self::I32 => "<i4",
            Self::I64 => "<i8",
            Self::F64 => "<f8",
        }
    }
}

/// Logical layout supplied by CAI or DLPack. Strides are bytes, not elements.
/// `address` denotes logical element zero, including for negative strides.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceArrayDescriptor {
    pub address: u64,
    pub shape: Vec<usize>,
    pub strides: Vec<i64>,
    pub dtype: DeviceDType,
    pub readonly: bool,
    pub ordinal: usize,
    /// CAI v3 stream: None, legacy=1, per-thread=2, or an actual stream handle.
    pub producer_stream: Option<u64>,
}

impl DeviceArrayDescriptor {
    /// Validate integer metadata before any driver call or address use.
    pub fn validate(&self) -> Result<(), EngineError> {
        if self.shape.len() != self.strides.len() {
            return Err(EngineError::Parameter(
                "device shape and strides have different ranks".into(),
            ));
        }
        if self.producer_stream == Some(0) {
            return Err(EngineError::Parameter(
                "CUDA Array Interface stream 0 is ambiguous; use None, 1 or 2".into(),
            ));
        }
        let _ = self.elements()?;
        let _ = self.span()?;
        Ok(())
    }

    pub fn elements(&self) -> Result<usize, EngineError> {
        self.shape.iter().try_fold(1usize, |count, &dim| {
            count
                .checked_mul(dim)
                .filter(|&n| n <= isize::MAX as usize)
                .ok_or_else(|| {
                    EngineError::Parameter("device array shape exceeds platform limits".into())
                })
        })
    }

    /// The half-open initialized byte span required by this view. An empty
    /// view dereferences no pointer and has no required allocation span.
    pub fn span(&self) -> Result<Option<(u64, u64)>, EngineError> {
        if self.shape.contains(&0) {
            return Ok(None);
        }
        let bad = || {
            EngineError::Parameter(
                "device array address or stride span exceeds platform limits".into(),
            )
        };
        let (mut lo, mut hi) = (0i128, self.dtype.width() as i128);
        for (&dim, &stride) in self.shape.iter().zip(&self.strides) {
            let extent = (dim.saturating_sub(1) as i128)
                .checked_mul(i128::from(stride))
                .ok_or_else(bad)?;
            lo = lo.checked_add(extent.min(0)).ok_or_else(bad)?;
            hi = hi.checked_add(extent.max(0)).ok_or_else(bad)?;
        }
        let first = i128::from(self.address).checked_add(lo).ok_or_else(bad)?;
        let end = i128::from(self.address).checked_add(hi).ok_or_else(bad)?;
        if self.address == 0
            || first <= 0
            || end <= first
            || end > usize::MAX as i128
            || end - first > isize::MAX as i128
        {
            return Err(bad());
        }
        Ok(Some((first as u64, end as u64)))
    }

    pub fn is_contiguous(&self) -> bool {
        let mut stride = self.dtype.width() as i128;
        for (&dim, &actual) in self.shape.iter().zip(&self.strides).rev() {
            if dim > 1 && i128::from(actual) != stride {
                return false;
            }
            let Some(next) = stride.checked_mul(dim as i128) else {
                return false;
            };
            stride = next;
        }
        true
    }
}

/// A retained foreign allocation. Protocol producers promise that their
/// descriptor remains valid while this owner is retained. Mutable inputs are
/// scanned again for every call; no readonly flag is treated as a snapshot.
#[derive(Clone)]
pub struct DeviceArray {
    descriptor: DeviceArrayDescriptor,
    // Retain the allocation owner even when CUDA consumers are not compiled.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) owner: Arc<dyn Any + Send + Sync>,
}

impl DeviceArray {
    pub fn descriptor(&self) -> &DeviceArrayDescriptor {
        &self.descriptor
    }
    /// # Safety
    /// `owner` must retain all storage described by `descriptor`, and the
    /// producer must not resize/free it or mutate it during an active call.
    pub unsafe fn from_external(
        descriptor: DeviceArrayDescriptor,
        owner: Arc<dyn Any + Send + Sync>,
    ) -> Result<Self, EngineError> {
        descriptor.validate()?;
        Ok(Self { descriptor, owner })
    }
}

/// Kernel-side row ABI. It addresses packed offsets and strided matrices
/// directly; importing a matrix never concatenates its bytes into a new table.
/// All quantities are integer bit patterns, matching kernels/device_rows.cuh.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DeviceRows {
    pub data: u64,
    pub rows: u64,
    /// 0: offsets, 1: matrix plus lengths, 2: fixed-width byte strings.
    pub layout: u64,
    pub offsets: u64,
    pub offset_width: u64,
    pub offset_stride: i64,
    pub row_stride: i64,
    pub column_stride: i64,
    pub lengths: u64,
    pub length_width: u64,
    pub length_stride: i64,
    /// Optional Arrow little-bit-order validity bitmap; zero means all valid.
    pub validity: u64,
    pub validity_offset: u64,
    pub columns: u64,
}

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
mod digest;
#[cfg(feature = "cuda")]
pub mod execution_stream;
#[cfg(feature = "cuda")]
pub mod pf_finalize;
#[cfg(feature = "cuda")]
pub mod retirement;
#[cfg(feature = "cuda")]
mod snapshot;
#[cfg(feature = "cuda")]
mod validate;
#[cfg(feature = "cuda")]
pub(crate) use cuda::DeviceOutput;
#[cfg(feature = "cuda")]
pub use cuda::{device_pointer_ordinal, BoundDeviceRows, DeviceColumn, DeviceLease};
#[cfg(feature = "cuda")]
pub use digest::{
    device_column_index_digest, device_sequence_digest, enqueue_index_digest,
    enqueue_sequence_digest,
};
#[cfg(feature = "cuda")]
pub use snapshot::{device_input_snapshot, DeviceInputSnapshot};
#[cfg(feature = "cuda")]
pub(crate) use validate::{allocate, scan, Retained, PAIR_BAD, PAIR_FIRST, PAIR_OUT};
#[cfg(feature = "cuda")]
pub use validate::{
    device_null_summary, enqueue_device_rows, enqueue_null_summary, validate_device_rows,
    ControlBlock, ControlWords, DeviceInputFact, DeviceInputSummary, DigestSlot, NullSummary,
    SummarySlot, ValidatedDeviceRows,
};

const _: () = {
    assert!(std::mem::size_of::<DeviceRows>() == 112);
    assert!(std::mem::align_of::<DeviceRows>() == 8);
    assert!(std::mem::offset_of!(DeviceRows, data) == 0);
    assert!(std::mem::offset_of!(DeviceRows, rows) == 8);
    assert!(std::mem::offset_of!(DeviceRows, layout) == 16);
    assert!(std::mem::offset_of!(DeviceRows, offsets) == 24);
    assert!(std::mem::offset_of!(DeviceRows, offset_width) == 32);
    assert!(std::mem::offset_of!(DeviceRows, offset_stride) == 40);
    assert!(std::mem::offset_of!(DeviceRows, row_stride) == 48);
    assert!(std::mem::offset_of!(DeviceRows, column_stride) == 56);
    assert!(std::mem::offset_of!(DeviceRows, lengths) == 64);
    assert!(std::mem::offset_of!(DeviceRows, length_width) == 72);
    assert!(std::mem::offset_of!(DeviceRows, length_stride) == 80);
    assert!(std::mem::offset_of!(DeviceRows, validity) == 88);
    assert!(std::mem::offset_of!(DeviceRows, validity_offset) == 96);
    assert!(std::mem::offset_of!(DeviceRows, columns) == 104);
};

/// One zero-copy device sequence table, retaining every source buffer. Row
/// values (offset ordering, lengths, nulls and alphabet) are checked on device
/// for each call before any compute stage receives its bound descriptor.
#[derive(Clone)]
pub struct DeviceSequenceTable {
    pub(crate) rows: DeviceRows,
    // Retain source buffers even when CUDA consumers are not compiled.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) inputs: Vec<DeviceArray>,
    pub(crate) text: bool,
}

impl DeviceSequenceTable {
    pub fn rows(&self) -> DeviceRows {
        self.rows
    }
    pub fn is_text(&self) -> bool {
        self.text
    }
    pub fn matrix(data: DeviceArray, lengths: DeviceArray) -> Result<Self, EngineError> {
        let a = &data.descriptor;
        let l = &lengths.descriptor;
        a.validate()?;
        l.validate()?;
        if a.dtype != DeviceDType::U8 || a.shape.len() != 2 {
            return Err(EngineError::Parameter(
                "device sequence matrices require rank-two uint8 data".into(),
            ));
        }
        if !matches!(l.dtype, DeviceDType::I32 | DeviceDType::I64) || l.shape != [a.shape[0]] {
            return Err(EngineError::Parameter(
                "device matrix lengths require one signed int32/int64 value per row".into(),
            ));
        }
        if a.ordinal != l.ordinal {
            return Err(EngineError::Device(
                "matrix data and lengths belong to different CUDA devices".into(),
            ));
        }
        let rows = DeviceRows {
            data: a.address,
            rows: a.shape[0] as u64,
            layout: 1,
            row_stride: a.strides[0],
            column_stride: a.strides[1],
            lengths: l.address,
            length_width: l.dtype.width() as u64,
            length_stride: l.strides[0],
            columns: a.shape[1] as u64,
            ..DeviceRows::default()
        };
        Ok(Self {
            rows,
            inputs: vec![data, lengths],
            text: false,
        })
    }

    pub fn packed(
        data: DeviceArray,
        offsets: DeviceArray,
        validity: Option<DeviceArray>,
        rows: usize,
        row_offset: usize,
        text: bool,
    ) -> Result<Self, EngineError> {
        let a = &data.descriptor;
        let o = &offsets.descriptor;
        a.validate()?;
        o.validate()?;
        let need = row_offset
            .checked_add(rows)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| EngineError::Parameter("device strings row range overflows".into()))?;
        if a.dtype != DeviceDType::U8
            || a.shape.len() != 1
            || a.strides[0] != 1
            || o.shape.len() != 1
            || o.shape[0] < need
            || !matches!(o.dtype, DeviceDType::I32 | DeviceDType::I64)
        {
            return Err(EngineError::Parameter("device strings require contiguous bytes and int32/int64 offsets covering every row".into()));
        }
        if a.ordinal != o.ordinal {
            return Err(EngineError::Device(
                "string data and offsets belong to different CUDA devices".into(),
            ));
        }
        let mut view = DeviceRows {
            data: a.address,
            rows: rows as u64,
            layout: 0,
            offsets: (i128::from(o.address) + row_offset as i128 * i128::from(o.strides[0])) as u64,
            offset_width: o.dtype.width() as u64,
            offset_stride: o.strides[0],
            column_stride: 1,
            columns: a.shape[0] as u64,
            validity_offset: row_offset as u64,
            ..DeviceRows::default()
        };
        if let Some(validity) = validity.as_ref() {
            let v = &validity.descriptor;
            v.validate()?;
            let bits = v
                .elements()?
                .checked_mul(v.dtype.width())
                .and_then(|n| n.checked_mul(8))
                .ok_or_else(|| {
                    EngineError::Parameter("device validity capacity overflows".into())
                })?;
            if v.ordinal != a.ordinal {
                return Err(EngineError::Device(
                    "string validity belongs to a different CUDA device".into(),
                ));
            }
            if v.shape.len() != 1 || !v.is_contiguous() || bits < need - 1 {
                return Err(EngineError::Parameter(
                    "device validity bitmap does not cover every string row".into(),
                ));
            }
            view.validity = v.address;
        }
        let mut inputs = vec![data, offsets];
        inputs.extend(validity);
        Ok(Self {
            rows: view,
            inputs,
            text: text && rows != 0,
        })
    }
}
