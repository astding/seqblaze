//! Whole-device-input validation. Only one fixed-size control summary and the
//! chunk count cross to the host; bytes, lengths and per-row facts stay device-side.
use std::{
    mem::size_of,
    ops::{Deref, DerefMut},
    sync::Arc,
};

use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg};
use seqblaze_core::validate::EngineError;

use super::{BoundDeviceRows, DeviceLease};
use crate::{device::guard, session::failure_ordinal};

const NONE: u64 = u64::MAX;
const REDUCE_THREADS: u32 = 128;
const SCAN_THREADS: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct DeviceInputFact {
    pub length: u64,
    pub bad_position: u64,
    pub byte: u64,
    pub kind: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct DeviceInputSummary {
    pub rows: u64,
    pub valid: u64,
    pub total: u64,
    pub min_length: u64,
    pub min_row: u64,
    pub max_length: u64,
    pub max_row: u64,
    pub bad_count: u64,
    pub bad_row: u64,
    pub bad_position: u64,
    pub bad_byte: u64,
    pub bad_rows: [u64; 5],
    pub span_row: u64,
    pub length_row: u64,
    pub length_value: u64,
    pub length_bound: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Legal {
    word: [u64; 4],
}

// SAFETY: these repr(C) plain-integer structs exactly mirror device_interop.cuh.
unsafe impl DeviceRepr for DeviceInputFact {}
unsafe impl DeviceRepr for DeviceInputSummary {}
unsafe impl DeviceRepr for Legal {}
const _: () = {
    assert!(size_of::<DeviceInputFact>() == 32);
    assert!(size_of::<DeviceInputSummary>() == 160);
    assert!(size_of::<Legal>() == 32);
};

fn failed(lease: &DeviceLease, what: &str, error: cudarc::driver::DriverError) -> EngineError {
    failure_ordinal(lease.device(), what, error)
}

/// Retire raw-kernel metadata on its consumer stream. On a failed sync the
/// allocation stays alive rather than risking a queued GPU read/write after free.
pub struct Retained<T> {
    value: Option<T>,
    stream: Arc<CudaStream>,
    device: crate::Device,
}
impl<T> Retained<T> {
    pub(crate) fn new(lease: &DeviceLease, value: T) -> Self {
        Self::on_owned_stream(lease.stream(), lease.device(), value)
    }
    pub(crate) fn on_owned_stream(
        stream: &Arc<CudaStream>,
        device: &crate::Device,
        value: T,
    ) -> Self {
        Self {
            value: Some(value),
            stream: Arc::clone(stream),
            device: device.clone(),
        }
    }
    pub(crate) fn into_inner(mut self) -> T {
        self.value.take().expect("retained input metadata")
    }
}
impl<T> Deref for Retained<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.value.as_ref().expect("retained input metadata")
    }
}
impl<T> DerefMut for Retained<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.value.as_mut().expect("retained input metadata")
    }
}
impl<T> Drop for Retained<T> {
    fn drop(&mut self) {
        if self.value.is_some()
            && guard(|| {
                crate::interop::execution_stream::retire_in_drop(&self.stream)
                    .map_err(|e| failure_ordinal(&self.device, "retiring device input metadata", e))
            })
            .is_err()
        {
            if let Some(value) = self.value.take() {
                std::mem::forget(value);
            }
        }
    }
}

pub(crate) fn allocate<T: DeviceRepr>(
    lease: &DeviceLease,
    len: usize,
) -> Result<crate::arena::Scratch<T>, EngineError> {
    if len
        .checked_mul(size_of::<T>())
        .is_none_or(|bytes| bytes > isize::MAX as usize)
    {
        return Err(EngineError::Device(
            "device input validation allocation exceeds platform limits".into(),
        ));
    }
    // SAFETY: every element read by a later kernel/copy is initialized first.
    unsafe { lease.execution().scratch(len.max(1)) }
        .map_err(|e| failed(lease, "allocating device input validation", e))
}

fn grid(blocks: usize, threads: u32) -> Result<LaunchConfig, EngineError> {
    Ok(LaunchConfig {
        grid_dim: (
            u32::try_from(blocks.max(1)).map_err(|_| {
                EngineError::Device("device input validation grid exceeds CUDA limits".into())
            })?,
            1,
            1,
        ),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    })
}

pub(crate) fn scan(
    lease: &DeviceLease,
    input: &CudaSlice<u64>,
    n: usize,
) -> Result<crate::arena::Scratch<u64>, EngineError> {
    scan_with::<0>(lease, input, n, None)
}

/// The prefix scan; with a control block, its block-total scratch stays alive
/// until the block's read instead of retiring (and waiting) at once (S6).
/// Keep the caller's existing launch trace category through every recursion:
/// generic compute/snapshot (0), input validation (2), or input digest (3).
/// Sharing the scan kernel does not make functional counts validation work.
pub(crate) fn scan_with<const CATEGORY: usize>(
    lease: &DeviceLease,
    input: &CudaSlice<u64>,
    n: usize,
    keep: Option<&ControlBlock>,
) -> Result<crate::arena::Scratch<u64>, EngineError> {
    let output_len = n
        .checked_add(1)
        .ok_or_else(|| EngineError::Device("device input prefix length overflows".into()))?;
    let mut output = Retained::new(lease, allocate::<u64>(lease, output_len)?);
    if n == 0 {
        lease
            .execution()
            .fill(&mut **output, 1, 0)
            .map_err(|e| failed(lease, "initializing an empty device input prefix", e))?;
        return Ok(output.into_inner());
    }
    let blocks = n.div_ceil(SCAN_THREADS as usize);
    let mut totals = Retained::new(lease, allocate::<u64>(lease, blocks)?);
    let count = n as u64;
    let kernels = &lease.device().0.kernels;
    let mut launch = lease.execution().builder(&kernels.input_scan_u64, CATEGORY);
    launch
        .slice_arg(input)
        .slice_arg_mut(&mut *output)
        .slice_arg_mut(&mut *totals)
        .arg(&count);
    // SAFETY: launch dimensions match the fixed 256-element block scan and
    // all input/output allocations cover their complete logical ranges.
    unsafe { launch.launch(grid(blocks, SCAN_THREADS)?) }
        .map_err(|e| failed(lease, "scanning device input chunks", e))?;
    if blocks > 1 {
        let block_prefix =
            Retained::new(lease, scan_with::<CATEGORY>(lease, &totals, blocks, keep)?);
        let mut launch = lease.execution().builder(&kernels.input_scan_add, CATEGORY);
        launch
            .slice_arg_mut(&mut *output)
            .slice_arg(&*block_prefix)
            .arg(&count);
        // SAFETY: the recursive prefix has blocks+1 initialized entries.
        unsafe { launch.launch(grid(blocks, SCAN_THREADS)?) }
            .map_err(|e| failed(lease, "combining device input chunk prefixes", e))?;
        if let Some(block) = keep {
            block.keep(block_prefix);
        }
    }
    if let Some(block) = keep {
        block.keep(totals);
    }
    Ok(output.into_inner())
}

// ---- The per-lease control block (cuda-backend.md, per-call cost S6) ----
//
// One device allocation of fixed slots. Validation summaries, the pair-check
// control and the input digests are written into it on the device, and the
// host reads the whole block with one copy and one wait. Chunk and piece
// totals stay where their prefix scans leave them: the kernels that need them
// read them in place, so no slot holds them.

/// Words of one `DeviceInputSummary` slot.
const SUMMARY_WORDS: usize = 20;
/// The pair-check control: the violating pair record (5 words), then the
/// kernels' scratch, the first reference rows (2 words) and the first
/// violating item (1 word), initialized to `u64::MAX` by one memset.
pub(crate) const PAIR_OUT: usize = 4 * SUMMARY_WORDS;
pub(crate) const PAIR_FIRST: usize = PAIR_OUT + 5;
pub(crate) const PAIR_BAD: usize = PAIR_FIRST + 2;
const DIGESTS: usize = PAIR_BAD + 1;
/// Words of the control block.
pub(crate) const CONTROL_WORDS: usize = DIGESTS + 4 * 4;
const _: () = assert!(size_of::<DeviceInputSummary>() == 8 * SUMMARY_WORDS);

/// A validation summary slot of the control block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SummarySlot {
    A,
    B,
    NullA,
    NullB,
}
impl SummarySlot {
    fn word(self) -> usize {
        SUMMARY_WORDS
            * match self {
                Self::A => 0,
                Self::B => 1,
                Self::NullA => 2,
                Self::NullB => 3,
            }
    }
}
/// A digest slot of the control block: tables a and b, index columns i and j.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DigestSlot {
    A,
    B,
    I,
    J,
}
impl DigestSlot {
    fn word(self) -> usize {
        DIGESTS
            + 4 * match self {
                Self::A => 0,
                Self::B => 1,
                Self::I => 2,
                Self::J => 3,
            }
    }
}

/// Device scratch whose retirement waits until the control block's read.
pub(crate) enum Kept {
    Words(#[allow(dead_code)] Retained<crate::arena::Scratch<u64>>),
    Summaries(#[allow(dead_code)] Retained<crate::arena::Scratch<DeviceInputSummary>>),
    Digests(#[allow(dead_code)] Retained<crate::arena::Scratch<super::digest::Digest>>),
    Column(#[allow(dead_code)] Retained<super::DeviceColumn>),
}
impl From<Retained<crate::arena::Scratch<u64>>> for Kept {
    fn from(value: Retained<crate::arena::Scratch<u64>>) -> Self {
        Self::Words(value)
    }
}
impl From<Retained<crate::arena::Scratch<DeviceInputSummary>>> for Kept {
    fn from(value: Retained<crate::arena::Scratch<DeviceInputSummary>>) -> Self {
        Self::Summaries(value)
    }
}
impl From<Retained<crate::arena::Scratch<super::digest::Digest>>> for Kept {
    fn from(value: Retained<crate::arena::Scratch<super::digest::Digest>>) -> Self {
        Self::Digests(value)
    }
}
impl From<Retained<super::DeviceColumn>> for Kept {
    fn from(value: Retained<super::DeviceColumn>) -> Self {
        Self::Column(value)
    }
}

/// One lease's control block (S6). Every kernel that writes it runs on the
/// lease's borrowed stream before the single read. Scratch that those kernels
/// use stays in the block until the read: each guard keeps its host-proved
/// retirement, which then finds the stream idle.
pub struct ControlBlock {
    words: Retained<crate::arena::Scratch<u64>>,
    base: u64,
    kept: std::sync::Mutex<Vec<Kept>>,
}
impl ControlBlock {
    pub fn new(lease: &DeviceLease) -> Result<Self, EngineError> {
        guard(|| {
            let words = Retained::new(lease, allocate::<u64>(lease, CONTROL_WORDS)?);
            let base = crate::session::ptr(&words, lease.stream());
            Ok(Self {
                words,
                base,
                kept: std::sync::Mutex::new(Vec::new()),
            })
        })
    }
    /// Keep `scratch` until the read. A refused list slot retires it at once,
    /// as before S6 (no new error).
    pub(crate) fn keep(&self, scratch: impl Into<Kept>) {
        let scratch = scratch.into();
        let mut kept = self.kept.lock().unwrap_or_else(|p| p.into_inner());
        if kept.try_reserve(1).is_ok() {
            kept.push(scratch);
        } else {
            drop(kept);
            drop(scratch);
        }
    }
    /// The device address of word `word`.
    pub(crate) fn at(&self, word: usize) -> u64 {
        debug_assert!(word < CONTROL_WORDS);
        self.base + 8 * word as u64
    }
    pub(crate) fn summary_at(&self, slot: SummarySlot) -> u64 {
        self.at(slot.word())
    }
    pub(crate) fn digest_at(&self, slot: DigestSlot) -> u64 {
        self.at(slot.word())
    }
    /// Read the whole block: one copy and the lease's completion wait.
    pub fn read(&self, lease: &DeviceLease) -> Result<ControlWords, EngineError> {
        self.read_as(lease, "reading device validation control metadata")
    }
    /// [`Self::read`] naming its copy `what` in a driver failure.
    pub fn read_as(&self, lease: &DeviceLease, what: &str) -> Result<ControlWords, EngineError> {
        guard(|| {
            let mut host = [0u64; CONTROL_WORDS];
            lease
                .execution()
                .memcpy_dtoh(&**self.words, &mut host[..])
                .map_err(|e| failed(lease, what, e))?;
            lease.synchronize()?;
            // The stream is idle now: the kept scratch retires without waiting.
            let kept = std::mem::take(&mut *self.kept.lock().unwrap_or_else(|p| p.into_inner()));
            drop(kept);
            Ok(ControlWords(host))
        })
    }
}

/// The host image of a control block. Only slots written before the read
/// are meaningful; the caller knows which.
pub struct ControlWords([u64; CONTROL_WORDS]);
impl ControlWords {
    pub fn summary(&self, slot: SummarySlot) -> DeviceInputSummary {
        let at = slot.word();
        let words: [u64; SUMMARY_WORDS] = self.0[at..at + SUMMARY_WORDS]
            .try_into()
            .expect("fixed summary slot");
        // SAFETY: DeviceInputSummary is repr(C) with exactly twenty u64 words
        // (asserted above), the layout the device wrote.
        unsafe { std::mem::transmute::<[u64; SUMMARY_WORDS], DeviceInputSummary>(words) }
    }
    /// The pair-check control: the violating pair (k, i, j, la, lb), if any.
    pub fn pair(&self) -> Option<[u64; 5]> {
        let control: [u64; 5] = self.0[PAIR_OUT..PAIR_OUT + 5]
            .try_into()
            .expect("fixed pair slot");
        (control[0] != NONE).then_some(control)
    }
    /// The 32 digest bytes, in the device's byte order.
    pub fn digest(&self, slot: DigestSlot) -> [u8; 32] {
        let at = slot.word();
        let mut bytes = [0u8; 32];
        for (chunk, word) in bytes.chunks_exact_mut(8).zip(&self.0[at..at + 4]) {
            chunk.copy_from_slice(&word.to_ne_bytes());
        }
        bytes
    }
}

/// Validated row state retained for computation and logical digest. Callers
/// apply their kernel's error wording/order using the small exact summary.
pub struct ValidatedDeviceRows {
    pub bound: BoundDeviceRows,
    pub facts: Retained<crate::arena::Scratch<DeviceInputFact>>,
    /// Alphabet chunk prefixes, or unscanned counts for host-preflight rows
    /// (S13a). Only an explicit snapshot needs those counts scanned.
    pub chunks: Retained<crate::arena::Scratch<u64>>,
    /// Valid after settle, or immediately for host-preflight/empty tables.
    pub summary: DeviceInputSummary,
    slot: SummarySlot,
    pub(crate) lengths: Retained<crate::arena::Scratch<u64>>,
    pub(crate) chunks_scanned: bool,
    pub(crate) length_hashes: Option<Retained<crate::arena::Scratch<super::digest::Digest>>>,
}
impl ValidatedDeviceRows {
    /// Take this table's summary from the control block read after
    /// [`enqueue_device_rows`] (an empty table keeps its host summary).
    pub fn settle(&mut self, words: &ControlWords) {
        if self.bound.rows.rows != 0 && self.bound.host_summary.is_none() {
            self.summary = words.summary(self.slot);
            if self.summary.valid == 0 {
                self.summary.min_length = 0;
            }
        }
    }
    /// The host-known byte total of a host-origin table without null rows
    /// (per-call cost S6): the sum of its row lengths, known before any read.
    pub fn host_total(&self) -> Option<u64> {
        self.bound.host_total()
    }
    /// The table's row count (host-known, from its descriptor).
    pub fn row_count(&self) -> u64 {
        self.bound.rows.rows
    }
}

/// An upper bound of a table's alphabet chunks from its descriptor alone:
/// every row is at most `columns` bytes, and for packed strings the rows lie
/// within `columns` bytes (a host-origin table's bound is its host total).
fn chunk_grid(table: &crate::interop::DeviceRows) -> usize {
    const CHUNK: u64 = 65_536;
    let bound = if table.layout == 0 {
        table.rows.saturating_add(table.columns / CHUNK)
    } else {
        table.rows.saturating_mul(table.columns.div_ceil(CHUNK))
    };
    bound.clamp(1, 65_535) as usize
}

/// The alphabet chunk grid of a validated table (snapshot copies use it too).
pub(crate) fn alphabet_grid(rows: &ValidatedDeviceRows) -> usize {
    chunk_grid(&rows.bound.rows)
}

/// No CPU sequence scan or host data snapshot occurs. Chunking bounds the
/// alphabet stage even for the public multi-gigabyte edit-distance length cap.
/// One control read (no separate chunk-count read).
pub fn validate_device_rows(
    lease: &DeviceLease,
    bound: BoundDeviceRows,
    legal: [u64; 4],
    minimum: usize,
    maximum: usize,
) -> Result<ValidatedDeviceRows, EngineError> {
    guard(|| {
        let block = ControlBlock::new(lease)?;
        let mut rows = enqueue_device_rows(
            lease,
            bound,
            legal,
            minimum,
            maximum,
            &block,
            SummarySlot::A,
        )?;
        if rows.bound.rows.rows != 0 {
            let words = block.read(lease)?;
            rows.settle(&words);
        }
        Ok(rows)
    })
}

/// Enqueue one table's validation, writing its summary into `slot` of the
/// control block (S6). Its `summary` is valid after [`ValidatedDeviceRows::settle`]
/// with the block's read; an empty table's summary is known at once.
pub fn enqueue_device_rows(
    lease: &DeviceLease,
    bound: BoundDeviceRows,
    legal: [u64; 4],
    minimum: usize,
    maximum: usize,
    block: &ControlBlock,
    slot: SummarySlot,
) -> Result<ValidatedDeviceRows, EngineError> {
    guard(|| {
        bound.ensure_lease(lease)?;
        let n = usize::try_from(bound.rows.rows)
            .map_err(|_| EngineError::Parameter("device row count exceeds the platform".into()))?;
        let mut facts = Retained::new(lease, allocate::<DeviceInputFact>(lease, n)?);
        let mut counts = Retained::new(lease, allocate::<u64>(lease, n)?);
        let mut lengths = Retained::new(lease, allocate::<u64>(lease, n)?);
        if n == 0 {
            let chunks = Retained::new(lease, scan(lease, &counts, 0)?);
            block.keep(counts);
            let summary = DeviceInputSummary {
                min_length: 0,
                min_row: NONE,
                max_row: NONE,
                bad_row: NONE,
                bad_position: NONE,
                bad_rows: [NONE; 5],
                span_row: NONE,
                length_row: NONE,
                ..DeviceInputSummary::default()
            };
            return Ok(ValidatedDeviceRows {
                bound,
                facts,
                chunks,
                summary,
                slot,
                lengths,
                chunks_scanned: true,
                length_hashes: None,
            });
        }
        let kernels = &lease.device().0.kernels;
        let table = bound.rows;
        let blocks = n.div_ceil(REDUCE_THREADS as usize).clamp(1, 1024);
        let byte_bound = bound
            .host_total
            .unwrap_or_else(|| table.rows.saturating_mul(table.columns));
        let large_digest = n > 8192 || byte_bound > 65_536;
        let mut length_hashes = if large_digest {
            Some(super::digest::allocate::<super::digest::Digest>(
                lease,
                n.div_ceil(8192) * 65,
            )?)
        } else {
            None
        };
        if bound.host_summary.is_none() && n <= REDUCE_THREADS as usize && table.columns <= 65_536 {
            let legal = Legal { word: legal };
            let (minimum, maximum, output) =
                (minimum as u64, maximum as u64, block.summary_at(slot));
            let length_hashes_address = length_hashes
                .as_ref()
                .map_or(0, |hashes| crate::session::ptr(hashes, lease.stream()));
            let mut launch = lease.execution().validation_builder(&kernels.input_small);
            launch
                .arg(&table)
                .slice_arg_mut(&mut *facts)
                .slice_arg_mut(&mut *counts)
                .slice_arg_mut(&mut *lengths)
                .arg(&legal)
                .arg(&minimum)
                .arg(&maximum)
                .arg(&output)
                .arg(&length_hashes_address);
            // SAFETY: exactly one 128-thread block; checked metadata precedes
            // each bounded row byte scan, including invalid offsets/null rows.
            unsafe { launch.launch(grid(1, REDUCE_THREADS)?) }
                .map_err(|e| failed(lease, "validating device row spans", e))?;
            return Ok(ValidatedDeviceRows {
                bound,
                facts,
                chunks: counts,
                summary: DeviceInputSummary::default(),
                slot,
                lengths,
                chunks_scanned: false,
                length_hashes,
            });
        }
        let function = if large_digest {
            &kernels.input_init_digest
        } else {
            &kernels.input_init
        };
        let mut launch = if bound.host_summary.is_some() {
            lease.execution().launch_builder(function)
        } else {
            lease.execution().validation_builder(function)
        };
        launch
            .arg(&table)
            .slice_arg_mut(&mut *facts)
            .slice_arg_mut(&mut *counts)
            .slice_arg_mut(&mut *lengths);
        if let Some(hashes) = length_hashes.as_mut() {
            launch.slice_arg_mut(&mut **hashes);
        }
        // SAFETY: initialized descriptors are retained by bound; init reads
        // only checked metadata buffers and writes exactly n row records.
        unsafe {
            launch.launch(grid(
                if large_digest {
                    (n.div_ceil(8192) * 65)
                        .div_ceil(REDUCE_THREADS as usize)
                        .clamp(1, 1024)
                } else {
                    blocks
                },
                REDUCE_THREADS,
            )?)
        }
        .map_err(|e| failed(lease, "validating device row spans", e))?;
        if let Some(summary) = bound.host_summary {
            // S13a: host preflight already established alphabet, lengths and
            // pair rules. Only facts used by compute/digest are needed. Keep
            // the unscanned chunk counts; an explicit snapshot scans lazily.
            return Ok(ValidatedDeviceRows {
                bound,
                facts,
                chunks: counts,
                summary,
                slot,
                lengths,
                chunks_scanned: false,
                length_hashes,
            });
        }
        let short = table.columns <= 65_536;
        let mut scanned = None;
        if !short {
            scanned = Some(Retained::new(
                lease,
                scan_with::<2>(lease, &counts, n, Some(block))?,
            ));
        }

        if table.columns != 0 {
            let legal = Legal { word: legal };
            let function = if short {
                &kernels.input_alphabet_rows
            } else {
                &kernels.input_alphabet
            };
            let mut launch = lease.execution().validation_builder(function);
            launch.arg(&table).slice_arg_mut(&mut *facts);
            if let Some(chunks) = scanned.as_ref() {
                launch.slice_arg(&**chunks);
            }
            launch.arg(&legal);
            // SAFETY: each task owns a bounded valid row subspan, established
            // by init and the exact prefix, whose device-resident end is the
            // task count. Invalid/null rows have no tasks.
            unsafe {
                launch.launch(grid(
                    if short {
                        n.div_ceil(32).clamp(1, 65_535)
                    } else {
                        chunk_grid(&table)
                    },
                    REDUCE_THREADS,
                )?)
            }
            .map_err(|e| failed(lease, "scanning the device input alphabet", e))?;
        }
        let chunks = match scanned {
            Some(chunks) => {
                block.keep(counts);
                chunks
            }
            None => counts,
        };
        let mut summaries = Retained::new(lease, allocate::<DeviceInputSummary>(lease, blocks)?);
        let minimum = minimum as u64;
        let maximum = maximum as u64;
        let mut launch = lease
            .execution()
            .validation_builder(&kernels.input_reduce_rows);
        launch
            .arg(&table)
            .slice_arg(&*facts)
            .slice_arg_mut(&mut *summaries)
            .arg(&minimum)
            .arg(&maximum)
            .arg(&1u32);
        // SAFETY: all n facts are initialized, and there is one output per block.
        unsafe { launch.launch(grid(blocks, REDUCE_THREADS)?) }
            .map_err(|e| failed(lease, "reducing device input validation", e))?;
        let mut launch = lease
            .execution()
            .validation_builder(&kernels.input_reduce_summaries);
        let block_count = blocks as u64;
        let output = block.summary_at(slot);
        launch.slice_arg(&*summaries).arg(&output).arg(&block_count);
        // SAFETY: at most 1024 initialized summaries reduce to one fixed record
        // in the block's summary slot.
        unsafe { launch.launch(grid(1, REDUCE_THREADS)?) }
            .map_err(|e| failed(lease, "finalizing device input validation", e))?;
        block.keep(summaries);
        Ok(ValidatedDeviceRows {
            bound,
            facts,
            chunks,
            summary: DeviceInputSummary::default(),
            slot,
            lengths,
            chunks_scanned: !short,
            length_hashes,
        })
    })
}

/// A second fixed control summary with null rows only. The alphabet facts are
/// already device-resident; check_bases=False must still reject nulls while
/// counting illegal non-null rows for its warning, without another byte scan.
/// The summary goes to `slot` of the control block (S6). An empty table
/// enqueues nothing (None), and its null summary is its validation summary.
pub fn enqueue_null_summary(
    lease: &DeviceLease,
    rows: &ValidatedDeviceRows,
    block: &ControlBlock,
    slot: SummarySlot,
) -> Result<Option<NullSummary>, EngineError> {
    guard(|| {
        rows.bound.ensure_lease(lease)?;
        let n = rows.bound.rows.rows;
        if n == 0 {
            return Ok(None);
        }
        let blocks = (n.div_ceil(128) as usize).clamp(1, 1024);
        let mut partial = Retained::new(lease, allocate::<DeviceInputSummary>(lease, blocks)?);
        let kernels = &lease.device().0.kernels;
        let table = rows.bound.rows;
        let minimum = 0u64;
        let maximum = isize::MAX as u64;
        let alphabet = 0u32;
        let mut launch = lease
            .execution()
            .validation_builder(&kernels.input_reduce_rows);
        launch
            .arg(&table)
            .slice_arg(&*rows.facts)
            .slice_arg_mut(&mut *partial)
            .arg(&minimum)
            .arg(&maximum)
            .arg(&alphabet);
        // SAFETY: all n row facts already exist; this reads no sequence bytes.
        unsafe { launch.launch(grid(blocks, 128)?) }
            .map_err(|e| failed(lease, "reducing device null-row controls", e))?;
        let count = blocks as u64;
        let output = block.summary_at(slot);
        let mut launch = lease
            .execution()
            .validation_builder(&kernels.input_reduce_summaries);
        launch.slice_arg(&*partial).arg(&output).arg(&count);
        // SAFETY: complete initialized partial summaries reduce to one record
        // in the block's summary slot.
        unsafe { launch.launch(grid(1, 128)?) }
            .map_err(|e| failed(lease, "finishing device null-row controls", e))?;
        block.keep(partial);
        Ok(Some(NullSummary { slot }))
    })
}

/// A null summary enqueued into the control block.
pub struct NullSummary {
    slot: SummarySlot,
}
impl NullSummary {
    /// The null summary from the block's read.
    pub fn summary(&self, words: &ControlWords) -> DeviceInputSummary {
        let mut summary = words.summary(self.slot);
        if summary.valid == 0 {
            summary.min_length = 0;
        }
        summary
    }
}

/// The null summary on its own control read (device-only callers).
pub fn device_null_summary(
    lease: &DeviceLease,
    rows: &ValidatedDeviceRows,
) -> Result<DeviceInputSummary, EngineError> {
    guard(|| {
        let block = ControlBlock::new(lease)?;
        match enqueue_null_summary(lease, rows, &block, SummarySlot::NullA)? {
            None => Ok(rows.summary),
            Some(null) => {
                let words = block.read(lease)?;
                Ok(null.summary(&words))
            }
        }
    })
}
