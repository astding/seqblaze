//! Immutable Arrow chunks flattened to direct row spans, never a payload copy.
use super::packed::{ArrowDescriptor, PackedBytes};
use crate::{detail::try_reserve_exact, validate::EngineError};
use std::{any::Any, fmt, sync::Arc};

struct RowSpan {
    address: usize,
    len: usize,
}
pub(super) struct SegmentedBytes {
    pub(super) format: &'static str,
    pub(super) descriptors: Vec<ArrowDescriptor>,
    rows: Vec<RowSpan>,
    /// Global row numbers of the null rows of every chunk, ascending.
    pub(super) nulls: Arc<Vec<usize>>,
    _owners: Vec<Box<dyn Any + Send + Sync>>,
}
impl fmt::Debug for SegmentedBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SegmentedBytes")
            .field("chunks", &self.descriptors.len())
            .field("rows", &self.rows.len())
            .finish_non_exhaustive()
    }
}
impl SegmentedBytes {
    pub(super) fn new(
        chunks: Vec<Arc<PackedBytes>>,
        format: &'static str,
    ) -> Result<Self, EngineError> {
        if !matches!(format, "z" | "Z" | "u" | "U" | "vu" | "vz")
            || chunks.iter().any(|chunk| chunk.descriptor.format != format)
        {
            return Err(EngineError::Parameter(
                "Arrow chunks must have the declared binary/string schema".into(),
            ));
        }
        let total = chunks.iter().try_fold(0usize, |n, chunk| {
            n.checked_add(chunk.descriptor.rows).ok_or_else(|| {
                EngineError::Overflow("Arrow chunk row count exceeds platform limit".into())
            })
        })?;
        const LABEL: &str = "Arrow chunk metadata";
        let mut rows = Vec::new();
        try_reserve_exact(&mut rows, total, LABEL)?;
        let mut descriptors = Vec::new();
        try_reserve_exact(&mut descriptors, chunks.len(), LABEL)?;
        let mut owners: Vec<Box<dyn Any + Send + Sync>> = Vec::new();
        try_reserve_exact(&mut owners, chunks.len(), LABEL)?;
        // Chunks were prepared with their global base index, in order.
        let n_nulls = chunks.iter().map(|chunk| chunk.nulls.len()).sum();
        let mut nulls = Vec::new();
        try_reserve_exact(&mut nulls, n_nulls, LABEL)?;
        for chunk in &chunks {
            nulls.extend_from_slice(&chunk.nulls);
        }
        for chunk in chunks {
            let descriptor = chunk.descriptor.clone();
            for row in 0..descriptor.rows {
                rows.push(RowSpan {
                    address: chunk.start(row),
                    len: chunk.seq_len(row),
                });
            }
            descriptors.push(descriptor);
            // Preserve only the exporter after preparing global row spans when
            // uniquely owned; do not keep a second per-row offsets cache.
            owners.push(match Arc::try_unwrap(chunk) {
                Ok(chunk) => chunk.into_owner(),
                Err(shared) => Box::new(shared),
            });
        }
        Ok(Self {
            format,
            descriptors,
            rows,
            nulls: Arc::new(nulls),
            _owners: owners,
        })
    }
    pub(super) fn len(&self) -> usize {
        self.rows.len()
    }
    pub(super) fn seq_len(&self, row: usize) -> usize {
        self.rows[row].len
    }
    /// Absolute native address and length of a prepared row span.
    #[inline]
    pub(super) fn span(&self, row: usize) -> (usize, usize) {
        let span = &self.rows[row];
        (span.address, span.len)
    }
}
