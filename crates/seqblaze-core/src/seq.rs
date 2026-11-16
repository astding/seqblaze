//! Owned tables and retained zero-copy foreign byte descriptors.
use crate::{batch::SeqBatch, sched::Workers, validate::EngineError};
use std::{
    any::Any,
    ops::ControlFlow,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex, MutexGuard, OnceLock,
    },
};
mod acgt;
mod packed;
mod segmented;
use segmented::SegmentedBytes;
mod unicode;
mod view;
use packed::PackedBytes;
pub use packed::{ArrowBuffer, ArrowDescriptor};
use view::FixedBytes;
pub use view::{ByteView, LengthDescriptor, MatrixDescriptor, BLOCK};

/// Bin of each byte value: A, C, G, T map to 0..=3; every other byte maps to 4.
pub const ACGT_BIN: [u8; 256] = {
    let mut table = [4u8; 256];
    table[b'A' as usize] = 0;
    table[b'C' as usize] = 1;
    table[b'G' as usize] = 2;
    table[b'T' as usize] = 3;
    table
};

/// Per-row byte composition counts for composition lower bounds
/// (engine-layout §4). Each row must be at most `u32::MAX` bytes.
#[derive(Debug)]
pub enum Composition {
    /// Clean table (its alphabet scan found no non-ACGT byte): counts of A, C, G, T.
    /// Every other byte value has count zero.
    Acgt(Vec<[u32; 4]>),
    /// Raw-byte table: counts of all 256 byte values.
    Bytes(Vec<[u32; 256]>),
}

#[derive(Debug, Clone)]
pub struct AlphabetScan {
    pub first: Option<(usize, usize, u8)>,
    pub n_bad: usize,
    pub bad_indices: Vec<usize>,
}

impl AlphabetScan {
    fn empty() -> Self {
        Self {
            first: None,
            n_bad: 0,
            bad_indices: Vec::new(),
        }
    }

    fn merge(&mut self, mut other: Self) {
        if let Some(first) = other.first {
            self.first = Some(self.first.map_or(first, |current| current.min(first)));
        }
        // Ranges are disjoint, so the sum is bounded by the table's row count.
        self.n_bad += other.n_bad;
        self.bad_indices.append(&mut other.bad_indices);
        self.bad_indices.sort_unstable();
        self.bad_indices.dedup();
        self.bad_indices.truncate(5);
    }
}

#[derive(Debug)]
enum Storage {
    Owned(SeqBatch),
    /// An immutable batch shared with its producer (the FASTA/FASTQ readers'
    /// `seq` column, D92): zero-copy, cached like owned storage.
    Shared(Arc<SeqBatch>),
    Fixed(Arc<FixedBytes>),
    Packed(Arc<PackedBytes>),
    Segmented(Arc<SegmentedBytes>),
}

/// Legal-byte set of an exact (kernel-supplied) alphabet, as a 256-bit key.
pub type LegalSet = [u64; 4];

/// The 256-bit key of a legal-byte table.
pub fn legal_set(legal: &[bool; 256]) -> LegalSet {
    let mut set = [0u64; 4];
    for (byte, &ok) in legal.iter().enumerate() {
        if ok {
            set[byte / 64] |= 1 << (byte % 64);
        }
    }
    set
}

/// Arrow formats whose values are text (string, large_string, string_view).
fn is_text_format(format: &str) -> bool {
    matches!(format, "u" | "U" | "vu")
}

/// Source of [`SeqTable::id`]: every constructed table takes the next value.
static NEXT_TABLE_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
pub struct SeqTable {
    storage: Storage,
    /// Process-unique identity, never reused (see [`SeqTable::id`]).
    id: u64,
    /// Ascending row numbers of the null rows (D86 (1)), fixed at import. A
    /// null row has an empty span in storage: no byte of it is ever read.
    nulls: Arc<Vec<usize>>,
    /// Whether at least one row came from a text value (manifest `encoding`).
    text: bool,
    lengths: OnceLock<Result<Vec<usize>, EngineError>>,
    /// Optional parallel trim configured before validation, but allocated and
    /// run only at the original length-metadata preflight of this side.
    trim_workers: AtomicUsize,
    /// Inclusive row-length extrema, reused only with this table's facts.
    length_bounds: OnceLock<Option<(usize, usize)>>,
    scan: OnceLock<AlphabetScan>,
    /// `scan` was published by this call's stream pass
    /// ([`SeqTable::stream_certify`]) and is not yet reported: the next
    /// alphabet scan reports it as performed, as the scan it replaces would be.
    scan_primed: AtomicBool,
    scan_init: InitLock,
    /// Completed scans against exact kernel alphabets, one per legal set
    /// (immutable tables only reuse them; fresh NumPy views start empty).
    /// Locked only to look up or publish a scan, never across one.
    exact_scans: Mutex<Vec<(LegalSet, AlphabetScan)>>,
    exact_init: InitLock,
    composition: OnceLock<Composition>,
    composition_init: InitLock,
}

/// First-initialization lock of a table cache. Concurrent first users on other
/// threads wait for the running initialization instead of repeating it. The
/// thread that holds it never waits on it again: a nested call made by Python
/// code that a cancellation poll runs on that thread, while the outer call is
/// inside the scan (cancellation-progress.md §3.3), initializes on its own and
/// publishes through the same `OnceLock` (or cache) instead of deadlocking.
#[derive(Debug, Default)]
struct InitLock {
    mutex: Mutex<()>,
    /// The token of the thread that holds `mutex` (0: none).
    owner: AtomicUsize,
}

/// A held [`InitLock`], or `Nested` on the thread that already holds it.
enum InitGuard<'a> {
    Held {
        lock: &'a InitLock,
        _guard: MutexGuard<'a, ()>,
    },
    Nested,
}

impl Drop for InitGuard<'_> {
    fn drop(&mut self) {
        if let InitGuard::Held { lock, .. } = self {
            // Cleared before the mutex is released (fields drop after this).
            lock.owner.store(0, Ordering::Release);
        }
    }
}

thread_local! {
    static THREAD_TOKEN: u8 = const { 0 };
}

/// A nonzero token unique among live threads (the address of a thread-local).
fn thread_token() -> usize {
    THREAD_TOKEN.with(|token| token as *const u8 as usize)
}

impl InitLock {
    fn acquire(&self, poisoned: &str) -> Result<InitGuard<'_>, EngineError> {
        let me = thread_token();
        if self.owner.load(Ordering::Acquire) == me {
            return Ok(InitGuard::Nested);
        }
        let guard = self
            .mutex
            .lock()
            .map_err(|_| EngineError::Device(poisoned.into()))?;
        self.owner.store(me, Ordering::Release);
        Ok(InitGuard::Held {
            lock: self,
            _guard: guard,
        })
    }
}

pub struct ArrowChunk {
    pub descriptor: ArrowDescriptor,
    pub owner: Box<dyn Any + Send + Sync>,
}

pub struct ArrowChunksInfo {
    pub format: &'static str,
    pub chunks: Vec<ArrowDescriptor>,
}

pub struct StorageInfo {
    pub kind: &'static str,
    pub address: usize,
    pub rows: usize,
    pub itemsize: Option<usize>,
    pub stride: Option<isize>,
    pub matrix: Option<MatrixDescriptor>,
    pub arrow: Option<ArrowDescriptor>,
    pub arrow_chunks: Option<ArrowChunksInfo>,
}

/// Row access to one table with its storage kind resolved once
/// ([`SeqTable::rows`]). [`SeqTable::get`] resolves it on every call, which
/// for a borrowed NumPy view includes the per-call length cache lookup; pair
/// and row loops take a `TableRows` once and index through it.
#[derive(Clone, Copy)]
pub struct TableRows<'a> {
    source: RowSource<'a>,
}

#[derive(Clone, Copy)]
enum RowSource<'a> {
    Batch(&'a SeqBatch),
    /// Borrowed NumPy storage with the call's prepared effective lengths.
    Fixed(&'a FixedBytes, &'a [usize]),
    Packed(&'a PackedBytes),
    Segmented(&'a SegmentedBytes),
}

impl<'a> TableRows<'a> {
    /// Row `i`, as [`SeqTable::get`].
    #[inline(always)]
    pub fn get(self, i: usize) -> ByteView<'a> {
        match self.source {
            RowSource::Batch(batch) => ByteView::owned(batch.get(i)),
            RowSource::Fixed(source, lengths) => ByteView::fixed(source, i, lengths[i]),
            RowSource::Packed(source) => ByteView::packed(source, i),
            RowSource::Segmented(source) => ByteView::segmented(source, i),
        }
    }
    /// Length of row `i`, as [`SeqTable::seq_len`].
    #[inline]
    pub fn seq_len(self, i: usize) -> usize {
        match self.source {
            RowSource::Batch(batch) => batch.seq_len(i),
            RowSource::Fixed(_, lengths) => lengths[i],
            RowSource::Packed(source) => source.seq_len(i),
            RowSource::Segmented(source) => source.seq_len(i),
        }
    }
    /// A [`Stager`] for a borrowed packed `S<n>` view of 1 to [`STAGE_WIDTH`]
    /// bytes per row; None for every other storage (immutable rows are
    /// slices already; other layouts copy row by row).
    #[inline]
    pub fn stager(self) -> Option<Stager<'a>> {
        match self.source {
            RowSource::Fixed(source, lengths)
                if source.window().is_some() && (1..=STAGE_WIDTH).contains(&source.width) =>
            {
                Some(Stager { source, lengths })
            }
            _ => None,
        }
    }
    /// `visit(seq_len(i))` for every row `i` of the table, in row order, with
    /// the storage resolved once for the whole loop (a null row has length 0).
    #[inline]
    pub fn for_each_len(self, mut visit: impl FnMut(usize)) {
        match self.source {
            RowSource::Batch(batch) => batch
                .offsets()
                .windows(2)
                .for_each(|pair| visit((pair[1] - pair[0]) as usize)),
            RowSource::Fixed(_, lengths) => lengths.iter().for_each(|&len| visit(len)),
            RowSource::Packed(source) => {
                (0..source.descriptor.rows).for_each(|i| visit(source.seq_len(i)))
            }
            RowSource::Segmented(source) => {
                (0..source.len()).for_each(|i| visit(source.seq_len(i)))
            }
        }
    }
    /// Fill `out` with the concatenated bytes of rows `row`, `row + 1`, …,
    /// starting at byte `skip` of row `row` (at most its length): the byte
    /// stream of these rows, in which null and empty rows contribute nothing.
    /// The rows must hold at least `out.len()` bytes from there. Rows of a
    /// packed `S<n>` view are read a group at a time (one relaxed block read of
    /// up to 4,096 bytes, padding included); other storage row by row.
    pub fn copy_concatenated(self, mut row: usize, mut skip: usize, out: &mut [u8]) {
        let mut written = 0;
        if let RowSource::Fixed(source, lengths) = self.source {
            let width = source.width;
            if source.window().is_some() && (1..=SCAN_GROUP_BYTES / 2).contains(&width) {
                let mut buffer = [0u8; SCAN_GROUP_BYTES];
                while written < out.len() {
                    let count = (SCAN_GROUP_BYTES / width).min(source.rows - row);
                    assert!(count > 0, "the rows hold fewer bytes than requested");
                    let group = &mut buffer[..count * width];
                    source.copy_rows(row, group);
                    for (bytes, &len) in group.chunks_exact(width).zip(&lengths[row..row + count]) {
                        let take = (len - skip).min(out.len() - written);
                        out[written..written + take].copy_from_slice(&bytes[skip..skip + take]);
                        written += take;
                        skip = 0;
                    }
                    row += count;
                }
                return;
            }
        }
        while written < out.len() {
            let view = self.get(row);
            let take = (view.len() - skip).min(out.len() - written);
            view.copy_to(skip, &mut out[written..written + take]);
            written += take;
            skip = 0;
            row += 1;
        }
    }
}

/// Leaf size of the byte stream of [`SeqTable::stream_certify`]: the
/// 65,536-byte byte-stream leaves of the input identity (result-provenance.md).
pub const STREAM_LEAF: usize = 65_536;
/// Bytes of room before a leaf's bytes in the buffer that
/// [`SeqTable::stream_certify`] hands its visitor (for a frame head).
pub const STREAM_HEAD: usize = 64;

/// Group bounds of the grouped alphabet scan (`SeqTable::scan_range_with`).
const SCAN_GROUP_ROWS: usize = 1024;
const SCAN_GROUP_BYTES: usize = 4096;

/// Widest packed `S<n>` view whose rows a [`Stager`] copies whole.
pub const STAGE_WIDTH: usize = 1024;

/// Whole-row copies of a borrowed packed `S<n>` view of at most
/// [`STAGE_WIDTH`] bytes per row ([`TableRows::stager`]): rows that lie back to
/// back are read as one span with aligned relaxed loads (padding included),
/// and a row's sequence is the first `len(row)` bytes of its copy. Kernels then
/// read staged rows as slices ([`ByteView::from_slice`]), so a batch copies
/// each run of consecutive rows once instead of each row of each pair.
#[derive(Clone, Copy)]
pub struct Stager<'a> {
    source: &'a FixedBytes,
    lengths: &'a [usize],
}

impl Stager<'_> {
    /// Bytes per row of a copy: the view's width.
    #[inline]
    pub fn width(self) -> usize {
        self.source.width
    }
    /// The effective length of `row`, as [`TableRows::seq_len`].
    #[inline]
    pub fn len(self, row: usize) -> usize {
        self.lengths[row]
    }
    /// The number of rows of the view.
    #[inline]
    pub fn rows(self) -> usize {
        self.lengths.len()
    }
    /// Copy whole rows `first..first + out.len() / width` into `out`, whose
    /// length is a multiple of the width (relaxed loads, one span).
    #[inline]
    pub fn copy_rows(self, first: usize, out: &mut [u8]) {
        self.source.copy_rows(first, out);
    }
}

/// First `(position, byte)` of `bytes` that `illegal` flags, asked block by
/// block in [`BLOCK`]-byte blocks, as [`ByteView::for_each_block`] visits a row.
#[inline]
fn first_illegal<F: Fn(&[u8]) -> Option<usize>>(bytes: &[u8], illegal: &F) -> Option<(usize, u8)> {
    bytes.chunks(BLOCK).enumerate().find_map(|(index, block)| {
        illegal(block).map(|position| (index * BLOCK + position, block[position]))
    })
}

/// [`first_illegal`] of one row of any storage.
fn scan_row<F: Fn(&[u8]) -> Option<usize>>(view: ByteView<'_>, illegal: &F) -> Option<(usize, u8)> {
    let bad = view.for_each_block(|offset, block| match illegal(block) {
        Some(position) => ControlFlow::Break((offset + position, block[position])),
        None => ControlFlow::Continue(()),
    });
    match bad {
        ControlFlow::Break(bad) => Some(bad),
        ControlFlow::Continue(()) => None,
    }
}

impl SeqTable {
    fn from_storage(storage: Storage) -> Self {
        let nulls = match &storage {
            Storage::Packed(source) => Arc::clone(&source.nulls),
            Storage::Segmented(source) => Arc::clone(&source.nulls),
            _ => Arc::new(Vec::new()),
        };
        // At least one non-null row of a text container (§5): an empty or
        // all-null Arrow string array has no row that came from text.
        let text = match &storage {
            Storage::Packed(source) => {
                is_text_format(source.descriptor.format) && source.descriptor.rows > nulls.len()
            }
            Storage::Segmented(source) => {
                is_text_format(source.format) && source.len() > nulls.len()
            }
            _ => false,
        };
        Self {
            storage,
            id: NEXT_TABLE_ID.fetch_add(1, Ordering::Relaxed),
            nulls,
            text,
            lengths: OnceLock::new(),
            trim_workers: AtomicUsize::new(1),
            length_bounds: OnceLock::new(),
            scan: OnceLock::new(),
            scan_primed: AtomicBool::new(false),
            scan_init: InitLock::default(),
            exact_scans: Mutex::new(Vec::new()),
            exact_init: InitLock::default(),
            composition: OnceLock::new(),
            composition_init: InitLock::default(),
        }
    }
    pub fn new(batch: SeqBatch) -> Self {
        Self::from_storage(Storage::Owned(batch))
    }
    /// An owned table with recorded null rows (ascending, each an empty row of
    /// `batch`) and its text flag (D86 (1), (3); text-nulls-fp-environment.md §3, §5).
    pub fn with_nulls(batch: SeqBatch, nulls: Vec<usize>, text: bool) -> Result<Self, EngineError> {
        if nulls.windows(2).any(|pair| pair[0] >= pair[1])
            || nulls.last().is_some_and(|&row| row >= batch.len())
            || nulls.iter().any(|&row| batch.seq_len(row) != 0)
        {
            return Err(EngineError::Parameter(
                "null rows must be ascending, in range and empty".into(),
            ));
        }
        let mut table = Self::from_storage(Storage::Owned(batch));
        table.nulls = Arc::new(nulls);
        table.text = text;
        Ok(table)
    }
    /// Ascending row numbers of the null rows (empty when there are none).
    pub fn nulls(&self) -> &[usize] {
        &self.nulls
    }
    /// Whether row `i` is null. O(1) without nulls, else O(log n_null).
    #[inline]
    pub fn is_null(&self, i: usize) -> bool {
        !self.nulls.is_empty() && self.nulls.binary_search(&i).is_ok()
    }
    /// Whether at least one row came from a text value (`str`, NumPy `U<n>`,
    /// Arrow string/large_string/string_view): the manifest's `encoding: "utf-8"`.
    pub fn is_text(&self) -> bool {
        self.text
    }
    /// Share an immutable batch without copying it (storage kind `"seqbatch"`).
    pub fn from_shared(batch: Arc<SeqBatch>) -> Self {
        Self::from_storage(Storage::Shared(batch))
    }
    /// Identity of this table object, unique for the life of the process: no
    /// two constructed tables share it, even when one is dropped and the next
    /// sits at the same address. Equality checks only; it never reaches an
    /// output, an order or a schedule.
    pub fn id(&self) -> u64 {
        self.id
    }
    /// Retain a native fixed-width descriptor without copying any sequence bytes.
    /// # Safety
    /// `owner` must retain every described row's allocation; native descriptor
    /// metadata must be valid. Do not resize/reallocate while retained or mutate
    /// bytes during execution/active tile iteration. Call `fresh_view` for each
    /// new execution; the engine does this before using borrowed cache facts.
    pub unsafe fn from_fixed_bytes(
        address: usize,
        rows: usize,
        width: usize,
        stride: isize,
        owner: Box<dyn Any + Send + Sync>,
    ) -> Result<Self, EngineError> {
        // SAFETY: the caller supplies the native descriptor and retained owner.
        let source = unsafe { FixedBytes::new(address, rows, width, stride, owner) }?;
        Ok(Self::from_storage(Storage::Fixed(Arc::new(source))))
    }
    /// Convert fixed-width UCS4 (NumPy `U<n>`) rows once into final immutable
    /// storage, each row as the UTF-8 encoding of its code points (D86 (3)).
    /// A surrogate or a value above U+10FFFF has no UTF-8 encoding: a
    /// ParameterError naming the row and the code-point position (E4).
    /// # Safety
    /// The native descriptor must be valid and retained by owner. Producers must
    /// not resize/reallocate or mutate until this bulk conversion has finished.
    pub unsafe fn from_utf32_text(
        address: usize,
        rows: usize,
        itemsize: usize,
        stride: isize,
        big_endian: bool,
        owner: Box<dyn Any + Send + Sync>,
    ) -> Result<Self, EngineError> {
        // SAFETY: caller supplies the native descriptor and its live export.
        let source = unsafe { FixedBytes::new(address, rows, itemsize, stride, owner) }?;
        let batch = unicode::convert(&source, big_endian)?;
        let mut table = Self::new(batch);
        // U<n> has no null rows: text exactly when there is a row (§5).
        table.text = !table.is_empty();
        Ok(table)
    }
    /// Retain a uint8 matrix and its integer lengths without copying their buffers.
    /// # Safety
    /// Both native descriptors must be valid and kept alive by owner. Producers
    /// must not resize/reallocate while retained or mutate during active execution.
    pub unsafe fn from_matrix_bytes(
        descriptor: MatrixDescriptor,
        owner: Box<dyn Any + Send + Sync>,
    ) -> Result<Self, EngineError> {
        // SAFETY: the caller provides both native descriptors and retained owners.
        let source = unsafe { FixedBytes::matrix(descriptor, owner) }?;
        let table = Self::from_storage(Storage::Fixed(Arc::new(source)));
        table.validate_lengths()?;
        Ok(table)
    }
    /// Import immutable CPU Arrow buffers without copying payload or input buffers.
    /// # Safety
    /// The descriptor must describe valid native allocations retained by owner;
    /// all exported buffers stay immutable until the imported table is dropped.
    pub unsafe fn from_arrow_bytes(
        descriptor: ArrowDescriptor,
        owner: Box<dyn Any + Send + Sync>,
    ) -> Result<Self, EngineError> {
        // SAFETY: caller guarantees Arrow ABI validity, immutability and ownership.
        let source = unsafe { PackedBytes::new(descriptor, owner) }?;
        Ok(Self::from_storage(Storage::Packed(Arc::new(source))))
    }
    /// Preserve immutable exported chunks and prepare global native row spans.
    /// # Safety
    /// Each descriptor describes valid CPU allocations retained by its owner;
    /// every buffer remains immutable through the returned table's lifetime.
    pub unsafe fn from_arrow_chunks(
        chunks: Vec<ArrowChunk>,
        format: &'static str,
    ) -> Result<Self, EngineError> {
        chunks.iter().try_fold(0usize, |n, chunk| {
            n.checked_add(chunk.descriptor.rows).ok_or_else(|| {
                EngineError::Overflow("Arrow chunk row count exceeds platform limit".into())
            })
        })?;
        let mut sources = Vec::new();
        crate::detail::try_reserve_exact(&mut sources, chunks.len(), "Arrow chunks")?;
        let mut base_index = 0;
        for chunk in chunks {
            let rows = chunk.descriptor.rows;
            // SAFETY: the caller retains each valid immutable native export.
            let source =
                unsafe { PackedBytes::prepare(chunk.descriptor, chunk.owner, base_index) }?;
            base_index += rows;
            sources.push(Arc::new(source));
        }
        let source = SegmentedBytes::new(sources, format)?;
        Ok(Self::from_storage(Storage::Segmented(Arc::new(source))))
    }
    /// Configure an optional parallel trim of this call's NumPy `S<n>` view.
    /// This only records a worker count: no allocation, byte read or worker
    /// start precedes the original [`Self::validate_lengths`] preflight.
    /// That preflight reserves the same "numpy view lengths" vector, in the
    /// same side order, then trims it. A refused optional worker start falls
    /// back to an inline trim of that buffer, with no second reservation.
    /// Matrices retain their original explicit-length checks.
    pub fn prepare_lengths(&self, workers: Workers) -> Result<(), EngineError> {
        if let Storage::Fixed(source) = &self.storage {
            if source.padding_readable() && self.lengths.get().is_none() {
                self.trim_workers
                    .store(workers.count(source.rows, 8), Ordering::Relaxed);
            }
        }
        Ok(())
    }

    /// Allocate only when the base length preflight would: even a certified
    /// full-width stream keeps this reservation here, not in the stream pass.
    fn prepare_fixed_lengths(&self, source: &FixedBytes) -> Result<Vec<usize>, EngineError> {
        let full_width = self.length_bounds.get() == Some(&Some((source.width, source.width)));
        let workers = self.trim_workers.load(Ordering::Relaxed);
        if full_width || workers > 1 {
            source.lengths_with(workers, full_width)
        } else {
            source.lengths()
        }
    }
    pub fn validate_lengths(&self) -> Result<(), EngineError> {
        // Every fixed-width view prepares its lengths here: the matrix form
        // checks its explicit lengths, and both forms reserve the lengths
        // vector fallibly (host-allocation-failures §12.3 B5), so a refusal is
        // reported by this preflight and `seq_len` never meets a failed cache.
        if let Storage::Fixed(source) = &self.storage {
            self.lengths
                .get_or_init(|| self.prepare_fixed_lengths(source))
                .as_ref()
                .map_err(Clone::clone)?;
        }
        Ok(())
    }
    pub fn len(&self) -> usize {
        match &self.storage {
            Storage::Owned(batch) => batch.len(),
            Storage::Shared(batch) => batch.len(),
            Storage::Fixed(source) => source.rows,
            Storage::Packed(source) => source.descriptor.rows,
            Storage::Segmented(source) => source.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// The contiguous owned batch, when this table owns its bytes (device
    /// upload can then copy it without host staging).
    pub fn owned_batch(&self) -> Option<&SeqBatch> {
        match &self.storage {
            Storage::Owned(batch) => Some(batch),
            Storage::Shared(batch) => Some(batch),
            _ => None,
        }
    }
    /// Raw contiguous bytes of a per-call prepared, full-width NumPy view.
    ///
    /// Returns None until effective lengths have been prepared successfully, or
    /// when row geometry/padding requires staging. No shared Rust slice is made
    /// over the externally mutable allocation. A consumer may bulk-read the
    /// returned native span only while this table retains its owner, must finish
    /// the read synchronously before releasing that owner, and must observe the
    /// producer contract forbidding mutation during active execution. Use the
    /// call's fresh view: borrowed length facts are refreshed between calls.
    pub fn contiguous_fixed_span(&self) -> Option<(usize, usize)> {
        let Storage::Fixed(source) = &self.storage else {
            return None;
        };
        let (address, bytes) = source.contiguous_span()?;
        self.lengths.get()?.as_ref().ok()?;
        (self.length_bounds() == Some((source.width, source.width))).then_some((address, bytes))
    }

    /// Copy bytes `start..start + out.len()` of the span that
    /// [`Self::contiguous_fixed_span`] proves, with the relaxed loads that
    /// `ByteView::copy_to` uses for a borrowed row (aligned 8-byte words). The
    /// span holds every row's logical bytes in row order. Returns false, and
    /// copies nothing, when there is no such span or the range exceeds it. The
    /// same retention and no-mutation contract as `contiguous_fixed_span` holds.
    pub fn copy_fixed_span(&self, start: usize, out: &mut [u8]) -> bool {
        let Some((address, bytes)) = self.contiguous_fixed_span() else {
            return false;
        };
        if start.checked_add(out.len()).is_none_or(|end| end > bytes) {
            return false;
        }
        // SAFETY: contiguous_fixed_span proves that all `bytes` lie in the
        // table's retained allocation; the range was checked against it. No
        // Rust slice spans the foreign storage.
        unsafe { view::load_bytes((address as *const u8).wrapping_add(start), out) };
        true
    }

    pub fn get(&self, i: usize) -> ByteView<'_> {
        self.rows().get(i)
    }
    pub fn seq_len(&self, i: usize) -> usize {
        self.rows().seq_len(i)
    }
    /// Row access with the storage kind (and, for a borrowed NumPy view, its
    /// prepared lengths) resolved once, for loops over many rows. Same rows
    /// and lengths as [`Self::get`] and [`Self::seq_len`].
    #[inline]
    pub fn rows(&self) -> TableRows<'_> {
        let source = match &self.storage {
            Storage::Owned(batch) => RowSource::Batch(batch),
            Storage::Shared(batch) => RowSource::Batch(batch),
            Storage::Fixed(source) => RowSource::Fixed(source, self.fixed_lengths(source)),
            Storage::Packed(source) => RowSource::Packed(source),
            Storage::Segmented(source) => RowSource::Segmented(source),
        };
        TableRows { source }
    }
    fn fixed_lengths<'s>(&'s self, source: &FixedBytes) -> &'s [usize] {
        self.lengths
            .get_or_init(|| self.prepare_fixed_lengths(source))
            .as_ref()
            .expect("length metadata preflight must succeed")
    }
    /// Inclusive extrema after length metadata validation; None for no rows.
    /// Mutable input views get a fresh cache on every call, like alphabet facts.
    pub(crate) fn length_bounds(&self) -> Option<(usize, usize)> {
        *self.length_bounds.get_or_init(|| {
            if let Storage::Fixed(source) = &self.storage {
                let lengths = self.fixed_lengths(source);
                let (&first, rest) = lengths.split_first()?;
                return Some(
                    rest.iter()
                        .fold((first, first), |(lo, hi), &len| (lo.min(len), hi.max(len))),
                );
            }
            let rows = self.rows();
            let mut lengths = (0..self.len()).map(|i| rows.seq_len(i));
            let first = lengths.next()?;
            Some(lengths.fold((first, first), |(lo, hi), len| (lo.min(len), hi.max(len))))
        })
    }
    /// The length every row shares, if all rows have one length (from the
    /// cached extrema); no pair of two such tables with the same value differs.
    pub(crate) fn uniform_length(&self) -> Option<usize> {
        self.length_bounds()
            .and_then(|(lo, hi)| (lo == hi).then_some(lo))
    }

    /// Owned/immutable Arrow facts are reusable; mutable NumPy facts are per-call.
    pub fn fresh_view(&self) -> Option<Self> {
        match &self.storage {
            Storage::Owned(_) | Storage::Shared(_) | Storage::Packed(_) | Storage::Segmented(_) => {
                None
            }
            Storage::Fixed(source) => {
                let mut view = Self::from_storage(Storage::Fixed(Arc::clone(source)));
                view.nulls = Arc::clone(&self.nulls);
                view.text = self.text;
                Some(view)
            }
        }
    }
    pub fn for_call(self: &Arc<Self>) -> Arc<Self> {
        self.fresh_view()
            .map(Arc::new)
            .unwrap_or_else(|| Arc::clone(self))
    }
    pub fn same_input(&self, other: &Self) -> bool {
        if std::ptr::eq(self, other) {
            return true;
        }
        match (&self.storage, &other.storage) {
            (Storage::Shared(a), Storage::Shared(b)) => Arc::ptr_eq(a, b),
            (Storage::Fixed(a), Storage::Fixed(b)) => a.same_input(b),
            (Storage::Packed(a), Storage::Packed(b)) => a.descriptor == b.descriptor,
            (Storage::Segmented(a), Storage::Segmented(b)) => {
                a.format == b.format && a.descriptors == b.descriptors
            }
            (Storage::Packed(a), Storage::Segmented(b))
            | (Storage::Segmented(b), Storage::Packed(a)) => {
                b.descriptors.as_slice() == std::slice::from_ref(&a.descriptor)
            }
            _ => false,
        }
    }
    /// Constant-time producer category; does not construct chunk descriptors.
    pub fn storage_kind(&self) -> &'static str {
        match &self.storage {
            Storage::Owned(_) => "owned",
            Storage::Shared(_) => "seqbatch",
            Storage::Fixed(source) => {
                if source.matrix_info().is_some() {
                    "numpy_u8_matrix"
                } else {
                    "numpy_fixed"
                }
            }
            Storage::Packed(source) => {
                if source.descriptor.view_buffers.is_some() {
                    "arrow_view"
                } else {
                    "arrow"
                }
            }
            Storage::Segmented(_) => "arrow_chunked",
        }
    }

    pub fn storage_info(&self) -> StorageInfo {
        match &self.storage {
            Storage::Owned(batch) => StorageInfo {
                kind: "owned",
                address: batch.data().as_ptr() as usize,
                rows: batch.len(),
                itemsize: None,
                stride: None,
                matrix: None,
                arrow: None,
                arrow_chunks: None,
            },
            Storage::Shared(batch) => StorageInfo {
                kind: "seqbatch",
                address: batch.data().as_ptr() as usize,
                rows: batch.len(),
                itemsize: None,
                stride: None,
                matrix: None,
                arrow: None,
                arrow_chunks: None,
            },
            Storage::Fixed(source) => StorageInfo {
                kind: if source.matrix_info().is_some() {
                    "numpy_u8_matrix"
                } else {
                    "numpy_fixed"
                },
                address: source.address,
                rows: source.rows,
                itemsize: Some(source.width),
                stride: Some(source.stride),
                matrix: source.matrix_info(),
                arrow: None,
                arrow_chunks: None,
            },
            Storage::Packed(source) => StorageInfo {
                kind: self.storage_kind(),
                address: source.descriptor.address,
                rows: source.descriptor.rows,
                itemsize: None,
                stride: None,
                matrix: None,
                arrow: Some(source.descriptor.clone()),
                arrow_chunks: None,
            },
            Storage::Segmented(source) => StorageInfo {
                kind: "arrow_chunked",
                address: 0,
                rows: source.len(),
                itemsize: None,
                stride: None,
                matrix: None,
                arrow: None,
                arrow_chunks: Some(ArrowChunksInfo {
                    format: source.format,
                    chunks: source.descriptors.clone(),
                }),
            },
        }
    }

    /// Mean sequence length, used only to size worker counts. O(1) for owned
    /// and NumPy storage (fixed width is an upper bound), O(rows) of length
    /// metadata for Arrow; never reads sequence bytes.
    pub fn mean_len(&self) -> usize {
        let n = self.len();
        if n == 0 {
            return 0;
        }
        match &self.storage {
            Storage::Owned(batch) => batch.data().len() / n,
            Storage::Shared(batch) => batch.data().len() / n,
            Storage::Fixed(source) => source.width,
            Storage::Packed(_) | Storage::Segmented(_) => {
                let rows = self.rows();
                let total: u128 = (0..n).map(|i| rows.seq_len(i) as u128).sum();
                (total / n as u128) as usize
            }
        }
    }

    /// Scan complete row ranges under the resolved CPU worker limit.
    /// The bool reports whether this call performed the physical table scan.
    /// Only a completed scan is published; worker errors leave the cache empty.
    pub fn alphabet_scan(&self, threads: usize) -> Result<(&AlphabetScan, bool), EngineError> {
        self.alphabet_scan_with(Workers::exact(threads))
    }

    /// [`Self::alphabet_scan`] with an explicit scheduling grain.
    pub fn alphabet_scan_with(
        &self,
        workers: Workers,
    ) -> Result<(&AlphabetScan, bool), EngineError> {
        if workers.limit == 0 {
            return Err(EngineError::Parameter("threads must be positive".into()));
        }
        if let Some(scan) = self.scan.get() {
            return self.report_primed(scan, workers);
        }
        // Serialize only first initialization, including concurrent first users.
        // Cached readers take the fast path above without locking or workers.
        let _init = self
            .scan_init
            .acquire("alphabet validation cache initialization lock poisoned")?;
        if let Some(scan) = self.scan.get() {
            return self.report_primed(scan, workers);
        }
        let count = if self.len() < 2 {
            1
        } else {
            workers.count(self.len(), self.mean_len())
        };
        let scan = if count == 1 {
            let _unit = crate::cancel::inline_unit();
            self.scan_range(0, self.len())?
        } else {
            let parts = crate::sched::map_ranges(self.len(), count, &|start, end| {
                self.scan_range(start, end)
            })?;
            let mut scan = AlphabetScan::empty();
            for part in parts {
                scan.merge(part);
            }
            scan
        };
        Ok((self.scan.get_or_init(|| scan), true))
    }

    /// Whole-table scan against an exact alphabet: every byte outside `legal`
    /// is illegal. Same partition, merge and first-error order as the ACGT
    /// scan. Immutable tables cache one completed scan per legal set; the bool
    /// reports whether this call performed the scan.
    pub fn alphabet_scan_exact(
        &self,
        legal: &[bool; 256],
        workers: Workers,
    ) -> Result<(AlphabetScan, bool), EngineError> {
        if workers.limit == 0 {
            return Err(EngineError::Parameter("threads must be positive".into()));
        }
        let key = legal_set(legal);
        let cached = || -> Result<Option<AlphabetScan>, EngineError> {
            let cache = self.exact_scans.lock().map_err(|_| {
                EngineError::Device("alphabet validation cache lock poisoned".into())
            })?;
            Ok(cache
                .iter()
                .find(|(set, _)| *set == key)
                .map(|(_, scan)| scan.clone()))
        };
        if let Some(scan) = cached()? {
            return Ok((scan, false));
        }
        // Held through the scan: concurrent first users wait instead of
        // scanning twice. Workers never take it.
        let _init = self
            .exact_init
            .acquire("alphabet validation cache initialization lock poisoned")?;
        if let Some(scan) = cached()? {
            return Ok((scan, false));
        }
        let count = if self.len() < 2 {
            1
        } else {
            workers.count(self.len(), self.mean_len())
        };
        let illegal = |block: &[u8]| block.iter().position(|&byte| !legal[byte as usize]);
        let scan = if count == 1 {
            let _unit = crate::cancel::inline_unit();
            self.scan_range_with(0, self.len(), &illegal)?
        } else {
            let parts = crate::sched::map_ranges(self.len(), count, &|start, end| {
                self.scan_range_with(start, end, &illegal)
            })?;
            let mut scan = AlphabetScan::empty();
            for part in parts {
                scan.merge(part);
            }
            scan
        };
        let mut cache = self
            .exact_scans
            .lock()
            .map_err(|_| EngineError::Device("alphabet validation cache lock poisoned".into()))?;
        // A nested call on this thread may have published the same scan meanwhile.
        if !cache.iter().any(|(set, _)| *set == key) {
            cache.push((key, scan.clone()));
        }
        Ok((scan, true))
    }

    /// Whether the ACGT scan of this table is cached (a validation would not
    /// rescan). A scan the stream pass prepared counts as not yet cached
    /// until it is reported.
    pub fn scan_cached(&self) -> bool {
        self.scan.get().is_some() && !self.scan_primed.load(Ordering::Acquire)
    }

    /// A cached scan: reported as performed exactly once when the stream pass
    /// prepared it, else not performed. The report replays the checkpoints of
    /// the scan it replaces (one item per row, one ticker per unit of the
    /// scan's partition), on this thread.
    fn report_primed<'s>(
        &self,
        scan: &'s AlphabetScan,
        workers: Workers,
    ) -> Result<(&'s AlphabetScan, bool), EngineError> {
        if !self.scan_primed.swap(false, Ordering::AcqRel) {
            return Ok((scan, false));
        }
        let n = self.len();
        let count = if n < 2 {
            1
        } else {
            workers.count(n, self.mean_len()).min(n)
        };
        let _unit = crate::cancel::inline_unit();
        for unit in 0..count {
            let (start, end) = crate::sched::bounds(n, unit, count);
            let mut ticker = crate::cancel::Ticker::new(crate::cancel::Site::Validate);
            ticker.add(end - start)?;
            ticker.finish()?;
        }
        Ok((scan, true))
    }

    /// The number of [`STREAM_LEAF`]-byte leaves [`Self::stream_certify`]
    /// would visit, or None when it makes no pass (same conditions).
    pub fn stream_leaves(&self) -> Option<usize> {
        if self.scan.get().is_some() {
            return None;
        }
        let total = match &self.storage {
            Storage::Owned(_) | Storage::Shared(_) => {
                self.owned_batch().map_or(0, |batch| batch.data().len())
            }
            Storage::Fixed(source)
                if source.window().is_some()
                    && source.padding_readable()
                    && self.lengths.get().is_none() =>
            {
                source.rows * source.width
            }
            _ => 0,
        };
        (total > 0).then(|| total.div_ceil(STREAM_LEAF))
    }

    /// The pre-validation stream pass of a call's table whose byte stream
    /// (its rows' bytes in row order; result-provenance.md) is one contiguous
    /// span: an owned or shared batch, whose null rows are empty, or a packed
    /// `S<n>` view taken to have every row fill its width. `leaf(index, buffer)`
    /// receives each [`STREAM_LEAF`]-byte leaf of that stream at
    /// `buffer[STREAM_HEAD..]`, in parallel over leaves, and every byte is
    /// checked against the strict ACGT alphabet.
    ///
    /// Ok(true) when every byte is A, C, G or T. Then no `S<n>` row ends in NUL
    /// padding, so the view's lengths are its width, and no row holds an
    /// illegal byte: its full-width proof (not a length-vector allocation) and
    /// the ACGT scan are prepared from these facts. The original length
    /// preflight still reserves its vector, in the original side order, and
    /// fills it from the proof without rereading bytes. The next alphabet scan
    /// reports the scan as this call's (`report_primed`). Ok(false) otherwise,
    /// with nothing prepared;
    /// the leaves already visited are then not known to be this table's
    /// stream and must be discarded. A table with a cached scan or prepared
    /// lengths, an empty stream and every other layout give Ok(false) at once.
    /// So does a pass one of whose workers cannot start
    /// ([`crate::sched::try_map_ranges`]): the refused start reports nothing
    /// and the steps run as without the pass. The pass reads every byte once
    /// and decides nothing that validation reports: its order relative to the
    /// validation steps is not observable.
    pub fn stream_certify(
        &self,
        workers: Workers,
        leaf: &(dyn Fn(usize, &mut [u8]) + Sync),
    ) -> Result<bool, EngineError> {
        let Some(leaves) = self.stream_leaves() else {
            return Ok(false);
        };
        let (data, fixed) = match &self.storage {
            Storage::Fixed(source) => (None, Some(&**source)),
            _ => (self.owned_batch().map(SeqBatch::data), None),
        };
        // A packed view's span was checked against isize::MAX at import.
        let total = data.map_or_else(|| fixed.map_or(0, |s| s.rows * s.width), <[u8]>::len);
        let dirty = AtomicBool::new(false);
        let count = workers.count(leaves, STREAM_LEAF);
        let parts = crate::sched::try_map_ranges(leaves, count, &|first, last| {
            let mut buffer = [0u8; STREAM_HEAD + STREAM_LEAF];
            for index in first..last {
                if dirty.load(Ordering::Relaxed) {
                    return Ok(false);
                }
                crate::cancel::poll()?;
                let start = index * STREAM_LEAF;
                let len = STREAM_LEAF.min(total - start);
                let bytes = &mut buffer[STREAM_HEAD..STREAM_HEAD + len];
                match (data, fixed) {
                    (Some(data), _) => bytes.copy_from_slice(&data[start..start + len]),
                    (_, Some(source)) => source.copy_span(start, bytes),
                    _ => unreachable!("one source"),
                }
                if !acgt::all_acgt(bytes) {
                    dirty.store(true, Ordering::Relaxed);
                    return Ok(false);
                }
                leaf(index, &mut buffer[..STREAM_HEAD + len]);
            }
            Ok(true)
        })?;
        if parts.is_none_or(|parts| parts.iter().any(|&clean| !clean)) {
            return Ok(false);
        }
        let _init = self
            .scan_init
            .acquire("alphabet validation cache initialization lock poisoned")?;
        if let Some(source) = fixed {
            // A nonallocating proof only. Allocating either side's lengths
            // here could precede the other side's earlier validation error.
            let _ = self.length_bounds.set(Some((source.width, source.width)));
        }
        // A concurrent first user of a shared immutable table may have
        // published (and reported) the same clean scan meanwhile.
        if self.scan.set(AlphabetScan::empty()).is_ok() {
            self.scan_primed.store(true, Ordering::Release);
        }
        Ok(true)
    }

    /// Whether an exact-alphabet scan over `legal` is cached.
    pub fn exact_scan_cached(&self, legal: &[bool; 256]) -> bool {
        let key = legal_set(legal);
        self.exact_scans
            .lock()
            .is_ok_and(|cache| cache.iter().any(|(set, _)| *set == key))
    }

    fn scan_range(&self, start: usize, end: usize) -> Result<AlphabetScan, EngineError> {
        self.scan_range_with(start, end, &|block: &[u8]| {
            if acgt::all_acgt(block) {
                None
            } else {
                // The fast check only certifies clean blocks. Keep the scalar
                // search for the exact first offending byte of an invalid one.
                block.iter().position(|&byte| ACGT_BIN[byte as usize] == 4)
            }
        })
    }

    /// One unit's rows of a scan, with `validate` checkpoints (a cancelled scan
    /// returns `Cancelled` and is never cached).
    ///
    /// Rows stored back to back (owned batches; NumPy views with unit column
    /// stride and row stride = width) are checked a group of consecutive rows
    /// at a time. Every byte before the group's first illegal byte belongs to
    /// a clean row, and that byte is the first illegal byte of its own row, so
    /// the scan resumes after that row; the rows, first positions and counts
    /// are those of the row-by-row scan. Padded NumPy groups are checked row
    /// by row from one block read.
    fn scan_range_with<F: Fn(&[u8]) -> Option<usize>>(
        &self,
        start: usize,
        end: usize,
        illegal: &F,
    ) -> Result<AlphabetScan, EngineError> {
        let mut result = AlphabetScan::empty();
        let mut ticker = crate::cancel::Ticker::new(crate::cancel::Site::Validate);
        let mut record = |i: usize, (position, byte): (usize, u8)| {
            result.first.get_or_insert((i, position, byte));
            result.n_bad += 1;
            if result.bad_indices.len() < 5 {
                result.bad_indices.push(i);
            }
        };
        let rows = self.rows();
        let mut i = start;
        match rows.source {
            RowSource::Fixed(source, lengths)
                if source.packed_rows() && (1..=SCAN_GROUP_BYTES / 2).contains(&source.width) =>
            {
                let width = source.width;
                let mut buffer = [0u8; SCAN_GROUP_BYTES];
                // A legal byte, to blank the NUL padding of S<n> rows in the
                // group's copy ('A' when legal, as for every built-in alphabet).
                let filler = std::iter::once(b'A')
                    .chain(0..=u8::MAX)
                    .find(|&byte| illegal(&[byte]).is_none());
                while i < end {
                    let count = (SCAN_GROUP_BYTES / width).min(end - i);
                    let group_lengths = &lengths[i..i + count];
                    let full = group_lengths.iter().all(|&len| len == width);
                    let next = if full || (source.padding_readable() && filler.is_some()) {
                        let group = &mut buffer[..count * width];
                        source.copy_rows(i, group);
                        if !full {
                            // S<n> rows: NUL padding is read anyway to trim it.
                            // Blanked with a legal byte, every illegal byte of the
                            // copy lies in its row's sequence, so the group is
                            // checked as one span, as full rows are.
                            let filler = filler.expect("checked above");
                            for (row, &len) in group.chunks_exact_mut(width).zip(group_lengths) {
                                row[len..].fill(filler);
                            }
                        }
                        // The first illegal byte of the rest of the group is the
                        // first illegal byte of its own row: every byte before it
                        // belongs to clean rows. Resume after that row.
                        let mut from = 0;
                        while let Some((position, byte)) =
                            first_illegal(&group[from * width..], illegal)
                        {
                            let row = from + position / width;
                            record(i + row, (position % width, byte));
                            from = row + 1;
                            if from == count {
                                break;
                            }
                        }
                        i + count
                    } else if source.padding_readable() {
                        // S<n> rows under an alphabet with no legal byte.
                        let group = &mut buffer[..count * width];
                        source.copy_rows(i, group);
                        for (row, &len) in group_lengths.iter().enumerate() {
                            if let Some(bad) = first_illegal(&group[row * width..][..len], illegal)
                            {
                                record(i + row, bad);
                            }
                        }
                        i + count
                    } else {
                        // Matrix bytes past a row's length are never read.
                        for row in i..i + count {
                            if let Some(bad) = scan_row(rows.get(row), illegal) {
                                record(row, bad);
                            }
                        }
                        i + count
                    };
                    ticker.add(next - i)?;
                    i = next;
                }
            }
            RowSource::Batch(batch) => {
                let (offsets, data) = (batch.offsets(), batch.data());
                while i < end {
                    let first = offsets[i] as usize;
                    let mut stop = i + 1;
                    while stop < end
                        && stop - i < SCAN_GROUP_ROWS
                        && offsets[stop + 1] as usize - first <= SCAN_GROUP_BYTES
                    {
                        stop += 1;
                    }
                    let group = &data[first..offsets[stop] as usize];
                    let next = match first_illegal(group, illegal) {
                        None => stop,
                        Some((position, byte)) => {
                            // Rows that end at or before the illegal byte are clean.
                            let mut row = i;
                            while offsets[row + 1] as usize - first <= position {
                                row += 1;
                            }
                            record(row, (position - (offsets[row] as usize - first), byte));
                            row + 1
                        }
                    };
                    ticker.add(next - i)?;
                    i = next;
                }
            }
            _ => {
                for row in start..end {
                    if let Some(bad) = scan_row(rows.get(row), illegal) {
                        record(row, bad);
                    }
                    ticker.tick()?;
                }
            }
        }
        ticker.finish()?;
        Ok(result)
    }

    /// Prepared composition counts, if [`Self::prepare_composition`] succeeded.
    pub fn composition(&self) -> Option<&Composition> {
        self.composition.get()
    }

    /// Prepare (or reuse) per-row composition counts: ACGT counts for a clean
    /// table, 256-value counts only for a raw-byte table. Immutable tables keep
    /// them cached; borrowed NumPy views are fresh per call. Size arithmetic is
    /// checked (OverflowError) and storage is reserved fallibly (DeviceError);
    /// a failure publishes nothing and a later call may retry.
    pub fn prepare_composition(&self, workers: Workers) -> Result<&Composition, EngineError> {
        if workers.limit == 0 {
            return Err(EngineError::Parameter("threads must be positive".into()));
        }
        if let Some(composition) = self.composition.get() {
            return Ok(composition);
        }
        let _init = self
            .composition_init
            .acquire("composition cache initialization lock poisoned")?;
        if let Some(composition) = self.composition.get() {
            return Ok(composition);
        }
        // Validation has already scanned this exact (possibly fresh) view.
        let (scan, _) = self.alphabet_scan_with(workers)?;
        // Bound preparation of this table's rows (cancellation-progress.md §4.2).
        crate::cancel::stage(crate::cancel::Stage::Bound, self.len());
        let composition = if scan.n_bad == 0 {
            Composition::Acgt(self.row_counts(workers, |view| {
                let mut counts = [0u32; 5];
                let _ = view.for_each_block(|_, block| {
                    for &byte in block {
                        counts[ACGT_BIN[byte as usize] as usize] += 1;
                    }
                    ControlFlow::<()>::Continue(())
                });
                [counts[0], counts[1], counts[2], counts[3]]
            })?)
        } else {
            Composition::Bytes(self.row_counts(workers, |view| {
                let mut counts = [0u32; 256];
                let _ = view.for_each_block(|_, block| {
                    for &byte in block {
                        counts[byte as usize] += 1;
                    }
                    ControlFlow::<()>::Continue(())
                });
                counts
            })?)
        };
        Ok(self.composition.get_or_init(|| composition))
    }

    fn row_counts<const N: usize>(
        &self,
        workers: Workers,
        count: impl Fn(ByteView<'_>) -> [u32; N] + Sync,
    ) -> Result<Vec<[u32; N]>, EngineError> {
        let rows = self.len();
        let bytes = rows
            .checked_mul(std::mem::size_of::<[u32; N]>())
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or_else(|| {
                EngineError::Overflow("composition counts exceed platform allocation limit".into())
            })?;
        let mut counts = Vec::new();
        counts.try_reserve_exact(rows).map_err(|error| {
            EngineError::Device(format!(
                "unable to reserve {bytes} bytes of composition counts: {error}"
            ))
        })?;
        counts.resize(rows, [0u32; N]);
        let limit = workers.count(rows, self.mean_len());
        let table = self.rows();
        crate::sched::fill_chunks(&mut counts, limit, &|start, chunk: &mut [[u32; N]]| {
            let mut ticker = crate::cancel::Ticker::new(crate::cancel::Site::Bound);
            for (offset, slot) in chunk.iter_mut().enumerate() {
                let view = table.get(start + offset);
                if view.len() > u32::MAX as usize {
                    return Err(EngineError::Overflow(format!(
                        "sequence {} length {} exceeds the composition count limit {}",
                        start + offset,
                        view.len(),
                        u32::MAX
                    )));
                }
                *slot = count(view);
                ticker.tick()?;
            }
            ticker.finish()
        })?;
        Ok(counts)
    }
}
