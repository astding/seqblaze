//! Shared device-resident producer, sink and detail interfaces. This is internal
//! Rust composition; it does not add a new Python call shape or output layout.
use crate::interop::{DeviceColumn, DeviceDType, DeviceLease};
use seqblaze_core::{sink::Sink, validate::EngineError};

/// Byte-for-byte mirror of kernels/hamming.cuh's SbSpace. Explicit indices
/// point at resident normalized int64 arrays; implicit shapes hold O(n) control
/// segments at most and never expand a Cartesian product into coordinate lists.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct ResidentSpace {
    pub kind: u64,
    pub k0: u64,
    pub n: u64,
    pub r0: u64,
    pub c0: u64,
    pub width: u64,
    pub li: u64,
    pub lj: u64,
    pub seg_k: u64,
    pub seg_i: u64,
    pub seg_j: u64,
    pub nseg: u64,
    pub skip: u64,
}
// SAFETY: thirteen repr(C) u64 fields, matching SbSpace exactly.
unsafe impl cudarc::driver::DeviceRepr for ResidentSpace {}
const _: () = {
    assert!(std::mem::size_of::<ResidentSpace>() == 104);
    assert!(std::mem::offset_of!(ResidentSpace, kind) == 0);
    assert!(std::mem::offset_of!(ResidentSpace, li) == 48);
    assert!(std::mem::offset_of!(ResidentSpace, seg_k) == 64);
    assert!(std::mem::offset_of!(ResidentSpace, skip) == 96);
};

impl ResidentSpace {
    pub(crate) fn abi(self) -> crate::session::SbSpace {
        crate::session::SbSpace {
            kind: self.kind,
            k0: self.k0,
            n: self.n,
            r0: self.r0,
            c0: self.c0,
            width: self.width,
            li: self.li,
            lj: self.lj,
            seg_k: self.seg_k,
            seg_i: self.seg_i,
            seg_j: self.seg_j,
            nseg: self.nseg,
            skip: self.skip,
        }
    }
}

pub struct ResidentWork {
    pub space: ResidentSpace,
    pub n_a: usize,
    pub n_b: usize,
    /// Original occurrence count. Non-Each explicit work is canonical unique;
    /// execution counts still describe every original occurrence, including ties.
    pub original_items: usize,
    pub mirror: bool,
    /// Strong immutable owners of generated index/segment control columns.
    pub owners: Vec<DeviceColumn>,
}

#[derive(Clone)]
pub struct ResidentFieldSpec {
    pub name: String,
    pub dtype: DeviceDType,
    /// Exact failure mask encoding; float64 masks are raw bits, never host FP.
    pub mask_bits: u64,
}
#[derive(Clone)]
pub struct ResidentField {
    pub name: String,
    pub values: DeviceColumn,
    pub mask_bits: u64,
}

pub struct ResidentChunk {
    pub space: ResidentSpace,
    /// Selecting score (I32/I64), not a rounded public float primary field.
    pub keys: DeviceColumn,
    pub fields: Vec<ResidentField>,
}

pub trait ResidentProducer {
    fn key_dtype(&self) -> DeviceDType;
    fn fields(&self) -> &[ResidentFieldSpec];
    fn next_chunk(&mut self, lease: &DeviceLease) -> Result<Option<ResidentChunk>, EngineError>;
    /// Optional native cutoff decisions for the latest chunk, in chunk order.
    /// The byte-wide mask uses the call's exact cutoff; sinks consume it on
    /// the same stream before requesting the next chunk.
    fn passed(&self) -> Option<&DeviceColumn> {
        None
    }
    /// Replay on the same already-validated input lease, with no new source
    /// import/hash or CPU fallback. TopK uses this for its complete ties pass.
    fn reset(&mut self) -> Result<(), EngineError>;
    /// The TopK replay's per-row limits (device addresses of the int64 thresholds and
    /// the u8 active flags; `mirror`: a mirrored within item's limit is the larger of
    /// its rows'), given before the replay's `reset`. Only a bounded producer reads
    /// them (duplex-mfe-bound.md §6.2).
    fn replay_limits(&mut self, _values: u64, _active: u64, _mirror: bool) {}
}

pub struct ResidentSinkConfig {
    pub sink: Sink,
    pub cutoff: Option<i64>,
    pub k: usize,
    pub mirror_symmetric: bool,
    /// For a streamed dense tile only: original row/column origin and local
    /// dimensions. None keeps the Result's original full pair-space shape.
    pub tile: Option<[usize; 4]>,
}

/// Selected entries for device detail replay. IDs address the producer's work
/// space: implicit original pair IDs, or the canonical unique explicit list.
/// Each retains original order/repeats and has passed=false for masked rows.
pub struct ResidentEntries {
    pub work_ids: DeviceColumn,
    pub i: DeviceColumn,
    pub j: DeviceColumn,
    pub keys: DeviceColumn,
    pub passed: Option<DeviceColumn>,
    pub rank: Option<DeviceColumn>,
}
pub struct ResidentSinkOutput {
    pub columns: Vec<ResidentField>,
    pub entries: Option<ResidentEntries>,
    pub shape: [usize; 2],
    pub tie_rows: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidentRaggedKind {
    Utf8,
    Binary,
    ListI64,
}
pub struct ResidentRaggedField {
    pub name: String,
    pub kind: ResidentRaggedKind,
    pub offsets: DeviceColumn,
    pub values: DeviceColumn,
    /// None means all-valid; an empty valid payload and a null row differ.
    pub validity: Option<DeviceColumn>,
}
pub struct ResidentDetails {
    pub numeric: Vec<ResidentField>,
    pub ragged: Vec<ResidentRaggedField>,
}
