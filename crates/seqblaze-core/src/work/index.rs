//! Retained native index buffers; signed validation precedes all pair traversal.
use crate::{seq::LengthDescriptor, validate::EngineError};
use std::{any::Any, fmt, sync::Arc};

enum Values {
    Owned(Vec<usize>),
    Borrowed {
        descriptor: LengthDescriptor,
        _owner: Box<dyn Any + Send + Sync>,
        copied: bool,
    },
}

#[derive(Clone)]
pub struct IndexArray {
    values: Arc<Values>,
}

pub struct IndexInfo {
    pub address: usize,
    pub len: usize,
    pub itemsize: usize,
    pub stride: isize,
    pub copied: bool,
}

impl fmt::Debug for IndexArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IndexArray")
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}
impl From<Vec<usize>> for IndexArray {
    fn from(values: Vec<usize>) -> Self {
        Self {
            values: Arc::new(Values::Owned(values)),
        }
    }
}

impl IndexArray {
    /// # Safety
    /// The native descriptor must cover allocated int32/int64 bytes retained by
    /// owner. No resize/reallocation while retained or mutation during execution.
    pub unsafe fn from_borrowed(
        descriptor: LengthDescriptor,
        owner: Box<dyn Any + Send + Sync>,
        copied: bool,
    ) -> Result<Self, EngineError> {
        descriptor.validate()?;
        Ok(Self {
            values: Arc::new(Values::Borrowed {
                descriptor,
                _owner: owner,
                copied,
            }),
        })
    }
    pub fn len(&self) -> usize {
        match &*self.values {
            Values::Owned(values) => values.len(),
            Values::Borrowed { descriptor, .. } => descriptor.rows,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub(super) fn value(&self, k: usize) -> i128 {
        assert!(k < self.len());
        match &*self.values {
            Values::Owned(values) => values[k] as i128,
            Values::Borrowed { descriptor, .. } => {
                // SAFETY: k is checked, the native descriptor was validated and
                // the Arc retains its export through this read.
                (unsafe { descriptor.read(k) }) as i128
            }
        }
    }
    pub fn get(&self, k: usize) -> usize {
        usize::try_from(self.value(k)).expect("index value preflight must succeed")
    }
    pub fn storage_info(&self) -> IndexInfo {
        match &*self.values {
            Values::Owned(values) => IndexInfo {
                address: values.as_ptr() as usize,
                len: values.len(),
                itemsize: std::mem::size_of::<usize>(),
                stride: std::mem::size_of::<usize>() as isize,
                copied: true,
            },
            Values::Borrowed {
                descriptor, copied, ..
            } => IndexInfo {
                address: descriptor.address,
                len: descriptor.rows,
                itemsize: descriptor.itemsize,
                stride: descriptor.stride,
                copied: *copied,
            },
        }
    }
}
