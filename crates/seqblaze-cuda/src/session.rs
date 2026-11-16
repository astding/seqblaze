//! Per-call device state and the device stages (docs/wiki/gpu/cuda-backend.md,
//! "Data path and pair numbering", "Distance stage", "Sinks";
//! docs/wiki/gpu/levenshtein-cuda.md §4-§6).
//!
//! One stream and one set of buffers per call (or per tile iterator), freed on
//! drop. Every launch is enqueued on that stream, so stream order is the only
//! synchronisation needed; host buffers are read only after a synchronize.
//! The hamming distance stage is here; the levenshtein one is in `lev.rs`,
//! the align one in `aln.rs` and the duplex one in `dup.rs`. The int64-key
//! path (`tm` on Pairs: its stage and the int64 sinks) is in `reg.rs`.
use std::sync::Arc;

use cudarc::driver::{
    sys, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, DriverError, LaunchConfig,
    PushKernelArg,
};
use seqblaze_core::{
    cancel::{self, Site, Stage},
    detail::{
        try_reserve, try_reserve_exact, DetailColumns, DetailData, DetailRequest, CANONICAL_NAN,
    },
    exec::{self, ExecOptions},
    kernel::{BatchDistance, CellClass, DistanceVisit, PairDistanceKernel, Score},
    sched::Workers,
    seq::SeqTable,
    sink::{
        kept, matrix, topk, try_filled, Columns, Counts, RowState, RowStatsColumns, RunOutput, Sink,
    },
    validate::{EngineError, ValidationReport},
    work::{Orientation, PairKind, WorkSource},
};

use crate::{
    aln::Aln,
    device::{error_name, guard, Device},
    dup::Dup,
    launch::{self, Settings},
    lev::Lev,
    p3::P3,
    pf::{Pf, PfBound},
    plan::{Request, MAX_GRID},
    reg::Reg,
    staging::{stage_view, PARALLEL_STAGING_BYTES},
    DeviceCall, DeviceRun,
};

// Pair-space kinds; must match kernels/hamming.cuh.
pub(crate) const ZIP: u64 = 0;
pub(crate) const LIST: u64 = 1;
pub(crate) const RECT: u64 = 2;
pub(crate) const TRI: u64 = 3;
pub(crate) const BOTH: u64 = 4;

// #[repr(C)] mirrors of the structs in kernels/hamming.cuh; all fields are u64.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct SbTables {
    a_data: u64,
    a_off: u64,
    b_data: u64,
    b_off: u64,
    pub(crate) a_view: u64,
    pub(crate) b_view: u64,
}
impl SbTables {
    pub(crate) fn from_views(a_view: u64, b_view: u64) -> Self {
        Self {
            a_view,
            b_view,
            ..Self::default()
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct SbSpace {
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
    /// BOTH: 1 when the diagonal is excluded.
    pub skip: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct SbPieces {
    piece_bytes: u64,
    per_pair: u64,
    prefix: u64,
    total: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct SbRows {
    pub(crate) mirror: u64,
    pub(crate) first: u64,
    pub(crate) best: u64,
    pub(crate) count: u64,
    pub(crate) nbest: u64,
    pub(crate) arg: u64,
}

// SAFETY: plain #[repr(C)] structs of u64 fields with the same field order and
// widths as their CUDA counterparts; passed by value as kernel parameters.
unsafe impl DeviceRepr for SbTables {}
const _: () = {
    assert!(std::mem::size_of::<SbTables>() == 48);
    assert!(std::mem::offset_of!(SbTables, a_view) == 32);
    assert!(std::mem::offset_of!(SbTables, b_view) == 40);
};
unsafe impl DeviceRepr for SbSpace {}
unsafe impl DeviceRepr for SbPieces {}
unsafe impl DeviceRepr for SbRows {}

/// Allocate `len` elements on `stream`; every device allocation of this crate
/// goes through here. In a testing build with the poison byte set
/// (`set_cuda_poison`, align-cuda.md §13) the new allocation is filled with
/// that byte on the same stream, so stream order puts the fill before every
/// upload, memset and kernel; a read of a cell nobody wrote then sees the
/// poison instead of whatever the allocator returned.
///
/// # Safety
/// As `CudaStream::alloc`: the contents are unset (or poison) and must be
/// written before any read.
pub(crate) unsafe fn dev_alloc<T: DeviceRepr>(
    stream: &Arc<CudaStream>,
    len: usize,
) -> Result<CudaSlice<T>, DriverError> {
    // SAFETY: forwarded to the caller.
    let slice = unsafe { stream.alloc::<T>(len) }?;
    #[cfg(feature = "testing")]
    if let Some(byte) = launch::poison() {
        let bytes = len * std::mem::size_of::<T>();
        if bytes > 0 {
            stream.context().bind_to_thread()?;
            // SAFETY: `slice` is a live allocation of `bytes` bytes, used on
            // this stream only.
            unsafe { sys::cuMemsetD8Async(ptr(&slice, stream), byte, bytes, stream.cu_stream()) }
                .result()?;
            launch::count_poisoned(bytes as u64);
        }
    }
    Ok(slice)
}

/// A reusable device buffer, grown on demand; never allocates zero bytes.
pub(crate) struct Scratch<T> {
    slice: Option<CudaSlice<T>>,
}

impl<T: DeviceRepr> Scratch<T> {
    pub(crate) fn new() -> Self {
        Self { slice: None }
    }

    /// At least `len` elements; the bool reports a fresh allocation.
    pub(crate) fn ensure_fresh(
        &mut self,
        stream: &Arc<CudaStream>,
        len: usize,
    ) -> Result<(&mut CudaSlice<T>, bool), DriverError> {
        let fresh = self.slice.as_ref().is_none_or(|slice| slice.len() < len);
        if fresh {
            self.slice = None;
            // SAFETY: contents are written (upload, memset or kernel) before any read.
            self.slice = Some(unsafe { dev_alloc::<T>(stream, len.max(1)) }?);
        }
        Ok((self.slice.as_mut().expect("allocated above"), fresh))
    }

    pub(crate) fn ensure(
        &mut self,
        stream: &Arc<CudaStream>,
        len: usize,
    ) -> Result<&mut CudaSlice<T>, DriverError> {
        self.ensure_fresh(stream, len).map(|(slice, _)| slice)
    }

    pub(crate) fn get(&self) -> &CudaSlice<T> {
        self.slice
            .as_ref()
            .expect("scratch is allocated before use")
    }
}

pub(crate) fn ptr<T>(slice: &CudaSlice<T>, stream: &CudaStream) -> u64 {
    check_stream(slice.stream(), stream);
    slice.device_ptr(stream).0
}

/// Testing builds: accesses of a device slice through a stream other than its
/// allocation stream (cuda-backend.md, per-call cost S1 (b)); process-wide,
/// never reset. With cudarc's event tracking off such an access would lose
/// its ordering, so it must stay 0 (`cuda_stream_misuse_count`).
#[cfg(feature = "testing")]
static STREAM_MISUSE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The S1 (b) assertion: `used` must be the slice's own allocation stream
/// `own`. Counted in testing builds; nothing in other builds.
#[inline]
pub(crate) fn check_stream(own: &CudaStream, used: &CudaStream) {
    #[cfg(feature = "testing")]
    if own.cu_stream() != used.cu_stream() || own.context().cu_ctx() != used.context().cu_ctx() {
        STREAM_MISUSE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    #[cfg(not(feature = "testing"))]
    let _ = (own, used);
}

/// Testing builds: the S1 (b) violations counted so far.
#[cfg(feature = "testing")]
pub fn stream_misuse_count() -> u64 {
    STREAM_MISUSE.load(std::sync::atomic::Ordering::Relaxed)
}

/// cudarc's typed host-to-device copy on `stream`, with the S1 (b) check.
pub(crate) fn htod<T: DeviceRepr, D: DevicePtrMut<T>>(
    stream: &Arc<CudaStream>,
    host: &[T],
    target: &mut D,
) -> Result<(), DriverError> {
    check_stream(target.stream(), stream);
    stream.memcpy_htod(host, target)
}

/// cudarc's typed device-to-host copy on `stream`, with the S1 (b) check.
pub(crate) fn dtoh<T: DeviceRepr, S: DevicePtr<T>>(
    stream: &Arc<CudaStream>,
    source: &S,
    host: &mut [T],
) -> Result<(), DriverError> {
    check_stream(source.stream(), stream);
    stream.memcpy_dtoh(source, host)
}

/// A host staging column of `len` values in a fallibly reserved vector: a
/// refused host reservation raises DeviceError instead of aborting.
fn host_column<T>(
    len: usize,
    values: impl Iterator<Item = T>,
    what: &str,
) -> Result<Vec<T>, EngineError> {
    let mut column = Vec::new();
    try_reserve_exact(&mut column, len, what)?;
    column.extend(values.take(len));
    Ok(column)
}

/// One table on the device: concatenated raw bytes (all 256 values) and int64
/// offsets, whole, unreferenced rows included.
pub(crate) struct Table {
    pub(crate) data: CudaSlice<u8>,
    pub(crate) off: CudaSlice<i64>,
    host_off: Vec<i64>,
}

impl Table {
    pub(crate) fn len(&self, i: usize) -> u64 {
        (self.host_off[i + 1] - self.host_off[i]) as u64
    }
    /// Host copy of the int64 offsets (rows + 1 entries).
    pub(crate) fn offsets(&self) -> &[i64] {
        &self.host_off
    }
}

/// A staged chunk: distances of its `n` items sit in `dist[0..n]` (and, when
/// the stage computed them, the numeric detail fields in the levenshtein buffers).
pub(crate) struct Chunk {
    pub n: usize,
    pub space: SbSpace,
}

/// Host side of one call: the kernel (routed pairs, host detail fields) and the
/// validated per-call tables it reads; `S` is the kernel's key type.
pub(crate) struct Host<'h, S: Score = i32> {
    pub kernel: &'h dyn PairDistanceKernel<S>,
    pub a: &'h SeqTable,
    pub b: &'h SeqTable,
}

impl<S: Score> Clone for Host<'_, S> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S: Score> Copy for Host<'_, S> {}

pub(crate) struct Session {
    pub(crate) device: Device,
    pub(crate) settings: Settings,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) a: Table,
    /// None when b is the same input as a (uploaded once).
    pub(crate) b: Option<Table>,
    index: Option<(CudaSlice<i64>, CudaSlice<i64>)>,
    pub(crate) dist: Scratch<i32>,
    prefix: Scratch<u64>,
    seg_k: Scratch<u64>,
    seg_i: Scratch<i64>,
    seg_j: Scratch<i64>,
    list_i: Scratch<i64>,
    list_j: Scratch<i64>,
    host_prefix: Vec<u64>,
    host_dist: Vec<i32>,
    /// Levenshtein state (None for hamming and align).
    pub(crate) lev: Option<Lev>,
    /// Align state (None for hamming and levenshtein).
    pub(crate) aln: Option<Aln>,
    /// Duplex state (None for the other kernels).
    pub(crate) dup: Option<Dup>,
    /// K3 duplex pf state (None for the other kernels).
    pub(crate) pf: Option<Pf>,
    /// `tm` on Pairs state (None for the other kernels): the int64-key stage.
    pub(crate) reg: Option<Reg>,
    /// The int64 keys of a staged chunk (the `tm`-on-Pairs int64-key stages).
    pub(crate) dist64: Scratch<i64>,
    /// K2 primer3 state (int64 keys; None for the other kernels).
    pub(crate) p3: Option<P3>,
    /// Host workers for routed pairs and host detail fields.
    pub(crate) workers: Workers,
    /// Whether routed explicit pairs are distinct work items (Tiles on zip and
    /// index), so they count in `host_pairs`.
    pub(crate) count_pairs: bool,
}

impl Session {
    pub(crate) fn new(
        device: &Device,
        call: DeviceCall,
        (a, b): (&SeqTable, &SeqTable),
        workers: Workers,
        count_pairs: bool,
    ) -> Result<Self, EngineError> {
        let label = format!("cuda:{}", device.info().ordinal);
        let fail = |what: &str, error: DriverError| failure(&label, what, error);
        let stream = device
            .0
            .ctx
            .new_stream()
            .map_err(|error| fail("creating a stream", error))?;
        let settings = launch::settings();
        let shared = a.same_input(b);
        // A raw table a needs its exact byte set for the levenshtein code map
        // (levenshtein-cuda.md §4); a clean one takes the four-code path.
        let byte_set = matches!(call, DeviceCall::Levenshtein { .. })
            && a.alphabet_scan_with(workers)?.0.n_bad > 0;
        let (table_a, bytes_a) = upload_table(&stream, a, byte_set, workers, &fail)?;
        let table_b = if shared {
            None
        } else {
            Some(upload_table(&stream, b, false, workers, &fail)?.0)
        };
        let (mut lev, mut aln, mut dup, mut pf, mut reg, mut p3) =
            (None, None, None, None, None, None);
        match &call {
            DeviceCall::Hamming => {}
            DeviceCall::Levenshtein {
                mode,
                b_end,
                n_ends,
                ..
            } => {
                lev = Some(Lev::new(
                    (&stream, device),
                    &settings,
                    (*mode, (*b_end, *n_ends)),
                    table_b.as_ref().unwrap_or(&table_a),
                    bytes_a.as_ref(),
                    // A tile stream sweeps each pair once; its cache would only grow.
                    !count_pairs,
                )?)
            }
            // The scoring pack only: align scans no table (align-cuda.md §4).
            DeviceCall::Align(call) => {
                aln = Some(Aln::new(
                    (&stream, device),
                    &settings,
                    call,
                    table_b.as_ref().unwrap_or(&table_a),
                    !count_pairs,
                )?)
            }
            // The score keys; the full keys wait for a detail pass (duplex-cuda.md §4).
            DeviceCall::Duplex(call) => {
                dup = Some(Dup::new(
                    (&stream, device),
                    &settings,
                    call,
                    (&table_a, table_b.as_ref().unwrap_or(&table_a)),
                    !count_pairs,
                )?)
            }
            // The weights (duplex-pf-cuda.md §4).
            DeviceCall::DuplexPf(call) => {
                pf = Some(Pf::new(
                    (&stream, device),
                    &settings,
                    call,
                    (&table_a, table_b.as_ref().unwrap_or(&table_a)),
                    !count_pairs,
                )?)
            }
            // The full keys, uploaded once (tm-pairs-cuda.md §4).
            DeviceCall::TmPairs(call) => reg = Some(Reg::new((&stream, device), call)?),
            // The device copy of the handle's tables and the pack (primer3-cuda.md §4);
            // a tile stream keeps no routed cache.
            DeviceCall::DuplexPrimer3(call) => {
                p3 = Some(P3::new(&stream, device, call, !count_pairs)?)
            }
        }
        Ok(Self {
            device: device.clone(),
            settings,
            stream,
            a: table_a,
            b: table_b,
            index: None,
            dist: Scratch::new(),
            prefix: Scratch::new(),
            seg_k: Scratch::new(),
            seg_i: Scratch::new(),
            seg_j: Scratch::new(),
            list_i: Scratch::new(),
            list_j: Scratch::new(),
            host_prefix: Vec::new(),
            host_dist: Vec::new(),
            lev,
            aln,
            dup,
            pf,
            reg,
            dist64: Scratch::new(),
            p3,
            workers,
            count_pairs,
        })
    }

    pub(crate) fn fail(&self, what: &str, error: DriverError) -> EngineError {
        failure(&format!("cuda:{}", self.device.info().ordinal), what, error)
    }

    pub(crate) fn sync(&self) -> Result<(), EngineError> {
        self.stream
            .synchronize()
            .map_err(|error| self.fail("synchronization", error))
    }

    pub(crate) fn grid(&self, items: u64) -> LaunchConfig {
        let threads = u64::from(self.settings.threads_per_block);
        LaunchConfig {
            grid_dim: (items.div_ceil(threads).clamp(1, MAX_GRID) as u32, 1, 1),
            block_dim: (self.settings.threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// One block per tile of `threads_per_block` items (Hits, routing).
    pub(crate) fn tiles(&self, items: usize) -> (usize, LaunchConfig) {
        let threads = self.settings.threads_per_block as usize;
        let blocks = items.div_ceil(threads).max(1);
        let config = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (self.settings.threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };
        (blocks, config)
    }

    pub(crate) fn tables(&self) -> SbTables {
        let b = self.b.as_ref().unwrap_or(&self.a);
        SbTables {
            a_data: ptr(&self.a.data, &self.stream),
            a_off: ptr(&self.a.off, &self.stream),
            b_data: ptr(&b.data, &self.stream),
            b_off: ptr(&b.off, &self.stream),
            a_view: 0,
            b_view: 0,
        }
    }

    pub(crate) fn table_b(&self) -> &Table {
        self.b.as_ref().unwrap_or(&self.a)
    }

    /// cuMemsetD32Async / cuMemsetD8Async over `count` elements.
    pub(crate) fn fill(
        &self,
        address: u64,
        value: u32,
        count: usize,
        wide: bool,
    ) -> Result<(), EngineError> {
        if count == 0 {
            return Ok(());
        }
        self.device
            .0
            .ctx
            .bind_to_thread()
            .map_err(|error| self.fail("binding the context", error))?;
        let stream = self.stream.cu_stream();
        // SAFETY: `address` points at a live allocation of at least `count`
        // elements (4 bytes when `wide`, else 1) owned by this session.
        let result = unsafe {
            if wide {
                sys::cuMemsetD32Async(address, value, count, stream)
            } else {
                sys::cuMemsetD8Async(address, value as u8, count, stream)
            }
        };
        result.result().map_err(|error| self.fail("memset", error))
    }

    pub(crate) fn upload<T: DeviceRepr>(
        stream: &Arc<CudaStream>,
        scratch: &mut Scratch<T>,
        host: &[T],
    ) -> Result<u64, DriverError> {
        let slice = scratch.ensure(stream, host.len())?;
        if !host.is_empty() {
            crate::session::htod(stream, host, slice)?;
        }
        Ok(ptr(slice, stream))
    }

    pub(crate) fn download<T: DeviceRepr>(
        &self,
        source: &CudaSlice<T>,
        n: usize,
        target: &mut [T],
    ) -> Result<(), EngineError> {
        if n == 0 {
            return Ok(());
        }
        crate::session::dtoh(&self.stream, &source.slice(0..n), &mut target[..n])
            .map_err(|error| self.fail("device-to-host copy", error))
    }

    /// Index columns as int64, uploaded once per call.
    pub(crate) fn index_columns(&mut self, work: &WorkSource) -> Result<(u64, u64), EngineError> {
        if self.index.is_none() {
            let PairKind::Index { i, j } = &work.kind else {
                return Err(EngineError::Parameter(
                    "index columns need an index work source".into(),
                ));
            };
            // Fallible host columns: a refused reservation raises DeviceError
            // instead of aborting (primer3 on CUDA §9 step 6).
            let column = |array: &seqblaze_core::work::IndexArray| {
                host_column(
                    array.len(),
                    (0..array.len()).map(|k| array.get(k) as i64),
                    "index columns",
                )
            };
            let (host_i, host_j) = (column(i)?, column(j)?);
            let upload = |host: &[i64]| -> Result<CudaSlice<i64>, DriverError> {
                // SAFETY: fully written by the copy below.
                let mut slice = unsafe { dev_alloc::<i64>(&self.stream, host.len().max(1)) }?;
                if !host.is_empty() {
                    crate::session::htod(&self.stream, host, &mut slice)?;
                }
                Ok(slice)
            };
            let columns = upload(&host_i)
                .and_then(|dev_i| Ok((dev_i, upload(&host_j)?)))
                .map_err(|error| self.fail("uploading index columns", error))?;
            self.index = Some(columns);
        }
        let (dev_i, dev_j) = self.index.as_ref().expect("uploaded above");
        Ok((ptr(dev_i, &self.stream), ptr(dev_j, &self.stream)))
    }

    /// Upload a chunk's explicit pairs (addressed from 0) into `space`.
    pub(crate) fn upload_pairs(
        &mut self,
        pairs: &[(usize, usize)],
        space: &mut SbSpace,
    ) -> Result<(), EngineError> {
        space.k0 = 0;
        let host_i = host_column(
            pairs.len(),
            pairs.iter().map(|&(i, _)| i as i64),
            "tile pairs",
        )?;
        let host_j = host_column(
            pairs.len(),
            pairs.iter().map(|&(_, j)| j as i64),
            "tile pairs",
        )?;
        let stream = &self.stream;
        let uploaded = Self::upload(stream, &mut self.list_i, &host_i)
            .and_then(|li| Ok((li, Self::upload(stream, &mut self.list_j, &host_j)?)));
        (space.li, space.lj) =
            uploaded.map_err(|error| self.fail("uploading tile pairs", error))?;
        Ok(())
    }

    /// Upload the row segments of within work IDs `[pos, pos + n)` (TRI).
    pub(crate) fn upload_segments(
        &mut self,
        work: &WorkSource,
        pos: usize,
        n: usize,
        space: &mut SbSpace,
    ) -> Result<(), EngineError> {
        let segments = || work.row_segment_iter(pos, pos + n).expect("within work");
        let count = segments().count();
        let seg_k = host_column(count, segments().map(|s| s.2 as u64), "row segments")?;
        let seg_i = host_column(count, segments().map(|s| s.0 as i64), "row segments")?;
        let seg_j = host_column(count, segments().map(|s| s.1 as i64), "row segments")?;
        let stream = &self.stream;
        let uploaded = Self::upload(stream, &mut self.seg_k, &seg_k).and_then(|k| {
            Ok((
                k,
                Self::upload(stream, &mut self.seg_i, &seg_i)?,
                Self::upload(stream, &mut self.seg_j, &seg_j)?,
            ))
        });
        (space.seg_k, space.seg_i, space.seg_j) =
            uploaded.map_err(|error| self.fail("uploading row segments", error))?;
        space.nseg = count as u64;
        Ok(())
    }

    /// Host row of an item, for the per-pair length of variable-length shapes.
    pub(crate) fn item_row(request: &Request<'_>, item: usize) -> usize {
        match request {
            Request::Range { work, .. } => match &work.kind {
                PairKind::Index { i, .. } => i.get(item),
                _ => item,
            },
            Request::Pairs(pairs) => pairs[item].0,
            Request::Rect { .. } => unreachable!("rectangles have uniform length"),
        }
    }

    /// Stage the chunk starting at item `pos`: hamming, levenshtein, align or
    /// duplex. With `fields`, a levenshtein stage also computes the requested
    /// numeric detail fields, and a duplex stage is a detail pass.
    pub(crate) fn stage(
        &mut self,
        host: Host<'_>,
        request: &Request<'_>,
        pos: usize,
        fields: bool,
    ) -> Result<Chunk, EngineError> {
        if self.reg.is_some() || self.pf.is_some() || self.p3.is_some() {
            // An int64-key call never reaches the int32 stages (`tm` on Pairs and K2 with
            // the primer3 preset: `run_i64`, `bind_i64`; K3: `run_pf`, `bind_pf`).
            return Err(EngineError::Device(
                "internal error: an int64-key device call reached an int32 stage; no result \
                 was produced; use device='cpu'"
                    .into(),
            ));
        }
        if self.lev.is_some() {
            self.lev_stage(host, request, pos, fields)
        } else if self.aln.is_some() {
            self.aln_stage(host, request, pos)
        } else if self.dup.is_some() {
            self.dup_stage(host, request, pos, fields)
        } else {
            self.hamming_stage(request, pos)
        }
    }

    /// Plan the hamming chunk starting at item `pos`, upload its metadata, zero
    /// its distances and launch the distance stage. Chunks close at
    /// `pairs_per_chunk` pairs or `bytes_per_launch` compared bytes and hold
    /// at least one pair.
    fn hamming_stage(&mut self, request: &Request<'_>, pos: usize) -> Result<Chunk, EngineError> {
        let settings = self.settings;
        let piece = settings.piece_bytes;
        let cap = (request.end() - pos).min(settings.pairs_per_chunk as usize);
        let mut space = SbSpace {
            k0: pos as u64,
            ..SbSpace::default()
        };
        // Uniform length for the implicit shapes, else per-pair lengths.
        let uniform = match request {
            Request::Range { work, .. } => match &work.kind {
                PairKind::Zip => {
                    space.kind = ZIP;
                    None
                }
                PairKind::Index { .. } => {
                    space.kind = LIST;
                    (space.li, space.lj) = self.index_columns(work)?;
                    None
                }
                PairKind::Cross | PairKind::OneToMany => {
                    space.kind = RECT;
                    space.width = work.shape.1 as u64;
                    Some(self.a.len(0))
                }
                PairKind::Within { .. }
                | PairKind::WithinOriented {
                    orientation: Orientation::AsGiven,
                    ..
                } => {
                    space.kind = TRI;
                    Some(self.a.len(0))
                }
                PairKind::WithinOriented {
                    orientation: Orientation::Both,
                    self_pairs,
                } => {
                    space.kind = BOTH;
                    space.skip = u64::from(!*self_pairs);
                    space.width = (work.shape.0 - usize::from(!*self_pairs)) as u64;
                    Some(self.a.len(0))
                }
            },
            Request::Pairs(_) => {
                space.kind = LIST;
                None
            }
            Request::Rect { r0, c0, width, .. } => {
                space.kind = RECT;
                space.r0 = *r0 as u64;
                space.c0 = *c0 as u64;
                space.width = *width as u64;
                Some(self.a.len(*r0))
            }
        };
        let mut pieces = SbPieces {
            piece_bytes: piece,
            ..SbPieces::default()
        };
        let n = match uniform {
            Some(len) => {
                let by_bytes = settings
                    .bytes_per_launch
                    .checked_div(len)
                    .map_or(cap, |pairs| pairs.max(1) as usize);
                let n = cap.min(by_bytes);
                pieces.per_pair = len.div_ceil(piece);
                pieces.total = n as u64 * pieces.per_pair;
                n
            }
            None => {
                let mut prefix = std::mem::take(&mut self.host_prefix);
                prefix.clear();
                // Amortized growth through the §3 helper before each push (F4
                // round 3); the chunk cap is never reserved up front.
                try_reserve(&mut prefix, 1, "piece prefix")?;
                prefix.push(0u64);
                let (mut n, mut bytes, mut total, mut most) = (0usize, 0u64, 0u64, 0u64);
                while n < cap {
                    let len = self.a.len(Self::item_row(request, pos + n));
                    if n > 0 && bytes + len > settings.bytes_per_launch {
                        break;
                    }
                    let count = len.div_ceil(piece);
                    bytes += len;
                    total += count;
                    most = most.max(count);
                    try_reserve(&mut prefix, 1, "piece prefix")?;
                    prefix.push(total);
                    n += 1;
                }
                if most <= 1 {
                    // Every pair fits one piece: item t is piece t (an empty
                    // pair's piece stores 0), so no prefix is uploaded or searched.
                    pieces.per_pair = 1;
                    pieces.total = if total == 0 { 0 } else { n as u64 };
                } else {
                    pieces.total = total;
                    pieces.prefix = Self::upload(&self.stream, &mut self.prefix, &prefix)
                        .map_err(|error| self.fail("uploading the piece prefix", error))?;
                }
                self.host_prefix = prefix;
                n
            }
        };
        space.n = n as u64;
        if let Request::Pairs(pairs) = request {
            // Explicit pairs are uploaded per chunk and addressed from 0.
            self.upload_pairs(&pairs[pos..pos + n], &mut space)?;
        }
        if space.kind == TRI {
            let Request::Range { work, .. } = request else {
                unreachable!("only within ranges use row segments")
            };
            self.upload_segments(work, pos, n, &mut space)?;
        }
        let dist = self
            .dist
            .ensure(&self.stream, n)
            .map_err(|error| failure_ordinal(&self.device, "allocation", error))?;
        let dist = ptr(dist, &self.stream);
        self.fill(dist, 0, n, true)?;
        if pieces.total > 0 {
            let tables = self.tables();
            let config = self.grid(pieces.total);
            let mut launch = self.stream.launch_builder(&self.device.0.kernels.pieces);
            launch.arg(&tables).arg(&space).arg(&pieces).arg(&dist);
            // SAFETY: argument types and order match sb_hamming_pieces; every
            // address is a live allocation of this session sized as the kernel reads it.
            unsafe { launch.launch(config) }
                .map_err(|error| self.fail("launching sb_hamming_pieces", error))?;
        }
        Ok(Chunk { n, space })
    }

    /// Copy the staged chunk's distances to `host_dist[0..n]`.
    fn fetch(&mut self, n: usize) -> Result<(), EngineError> {
        let mut host = std::mem::take(&mut self.host_dist);
        let grow = n.saturating_sub(host.len());
        if let Err(error) = try_reserve(&mut host, grow, "device distance chunk") {
            self.host_dist = host;
            return Err(error);
        }
        host.resize(n, 0);
        let result = self
            .download(self.dist.get(), n, &mut host)
            .and_then(|()| self.sync());
        self.host_dist = host;
        result
    }

    /// Append the staged chunk's distances to `out`, copied straight from the
    /// device into place (no staging copy). Callers reserve `out` up front.
    pub(crate) fn fetch_into(&self, n: usize, out: &mut Vec<i32>) -> Result<(), EngineError> {
        let base = out.len();
        try_reserve(out, n, "device distance chunk")?;
        out.resize(base + n, 0);
        self.download(self.dist.get(), n, &mut out[base..])
            .and_then(|()| self.sync())
    }

    pub(crate) fn dist_ptr(&self) -> u64 {
        ptr(self.dist.get(), &self.stream)
    }

    /// Whether the device fills every requested detail field of the call.
    pub(crate) fn delivers(&self, request: DetailRequest) -> bool {
        self.lev.as_ref().is_some_and(|lev| lev.delivers(request))
            || self.dup.as_ref().is_some_and(|dup| dup.delivers(request))
    }

    /// Distinct work items routed to the host so far.
    pub(crate) fn host_pairs(&self) -> usize {
        self.lev.as_ref().map_or(0, |lev| lev.host_pairs)
            + self.aln.as_ref().map_or(0, |aln| aln.host_pairs)
            + self.dup.as_ref().map_or(0, |dup| dup.host_pairs)
            + self.pf.as_ref().map_or(0, |pf| pf.host_pairs)
            + self.p3.as_ref().map_or(0, |p3| p3.host_pairs)
    }

    /// Each: `passed = d <= c`, a failing d (and its numeric fields) becomes
    /// INT32_MAX; positional by work ID. Numeric detail fields come from the
    /// device; a heavy request computes every requested field on the host for
    /// the passing rows.
    fn each(
        &mut self,
        host: Host<'_>,
        work: &WorkSource,
        cutoff: Option<i32>,
    ) -> Result<Columns, EngineError> {
        let n_pairs = work.n_pairs;
        let request = host.kernel.detail_request();
        let device_fields = self.delivers(request);
        let mut dist = try_filled(n_pairs, 0i32, "device each key column")?;
        let mut passed = match cutoff {
            Some(_) => Some(try_filled(n_pairs, 0u8, "device each passed column")?),
            None => None,
        };
        let mut fields = if device_fields {
            Some(self.numeric_host(request, n_pairs)?)
        } else {
            None
        };
        let mut passed_dev = Scratch::<u8>::new();
        let (has_cutoff, limit) = (i32::from(cutoff.is_some()), cutoff.unwrap_or(0));
        let request_range = Request::Range { work, end: n_pairs };
        let mut pos = 0;
        cancel::stage(Stage::Pairs, n_pairs);
        while pos < n_pairs {
            let chunk = self.stage(host, &request_range, pos, device_fields)?;
            let n = chunk.n;
            let (b_end_ptr, n_ends_ptr) = self.numeric_ptrs(request, device_fields);
            if passed.is_some() {
                // Without a cutoff sb_each would leave every distance as it is.
                let passed_ptr = ptr(
                    passed_dev
                        .ensure(&self.stream, n)
                        .map_err(|error| self.fail("allocation", error))?,
                    &self.stream,
                );
                let dist_ptr = self.dist_ptr();
                let count = n as u64;
                let config = self.grid(count);
                let mut launch = self.stream.launch_builder(&self.device.0.kernels.each);
                launch
                    .arg(&dist_ptr)
                    .arg(&passed_ptr)
                    .arg(&count)
                    .arg(&has_cutoff)
                    .arg(&limit)
                    .arg(&b_end_ptr)
                    .arg(&n_ends_ptr);
                // SAFETY: matches sb_each; every buffer holds at least n elements.
                unsafe { launch.launch(config) }
                    .map_err(|error| self.fail("launching sb_each", error))?;
            }
            self.download(self.dist.get(), n, &mut dist[pos..pos + n])?;
            if let Some(passed) = passed.as_mut() {
                self.download(passed_dev.get(), n, &mut passed[pos..pos + n])?;
            }
            if let Some(fields) = fields.as_mut() {
                self.download_numeric(fields, pos, n)?;
            }
            pos += n;
            // A Session chunk boundary (cancellation-progress.md §3.5).
            cancel::boundary(Site::Pairs, n)?;
        }
        self.sync()?;
        if let (Some(fields), Some(passed)) = (fields.as_mut(), passed.as_deref()) {
            // Duplex: a failing row holds the canonical NaN in every float64
            // column (sb_each masks levenshtein's integer fields itself).
            fields.mask_floats(passed);
        }
        let details = match fields {
            Some(fields) => Some(self.numeric_columns(host, fields)),
            None if !request.is_empty() => Some(kept::each_details(
                host.kernel,
                work,
                (host.a, host.b),
                &dist,
                passed.as_deref(),
                self.workers,
            )?),
            None => None,
        };
        Ok(Columns {
            dist,
            passed,
            details,
            ..Columns::default()
        })
    }

    /// Hits: per-block counts, a host exclusive scan, then ordered writes, so
    /// output is born in work-ID order for any block or chunk size. Numeric
    /// detail fields are written beside each hit; a heavy request computes
    /// every requested field on the host over the final hit list.
    fn hits(
        &mut self,
        host: Host<'_>,
        work: &WorkSource,
        cutoff: Option<i32>,
    ) -> Result<Columns, EngineError> {
        let request = host.kernel.detail_request();
        // Duplex fields come from a detail pass over the final hit list (below).
        let device_fields = self.dup.is_none() && self.delivers(request);
        let (want_e, want_n) = self.lev.as_ref().map_or((false, false), |lev| {
            let (e, n) = lev.wanted(request);
            (e && device_fields, n && device_fields)
        });
        let (mut ii, mut jj, mut dd) = (Vec::<i64>::new(), Vec::<i64>::new(), Vec::<i32>::new());
        let (mut ee, mut cc) = (Vec::<i64>::new(), Vec::<i32>::new());
        let mut counts_dev = Scratch::<u64>::new();
        let mut offsets_dev = Scratch::<u64>::new();
        let mut out_i = Scratch::<i64>::new();
        let mut out_j = Scratch::<i64>::new();
        let mut out_d = Scratch::<i32>::new();
        let mut out_e = Scratch::<i64>::new();
        let mut out_n = Scratch::<i32>::new();
        let (has_cutoff, limit) = (i32::from(cutoff.is_some()), cutoff.unwrap_or(0));
        let request_range = Request::Range {
            work,
            end: work.n_pairs,
        };
        let mut pos = 0;
        cancel::stage(Stage::Pairs, work.n_pairs);
        while pos < work.n_pairs {
            let chunk = self.stage(host, &request_range, pos, device_fields)?;
            let (n, count) = (chunk.n, chunk.n as u64);
            let (blocks, config) = self.tiles(n);
            let dist_ptr = self.dist_ptr();
            let (b_end_ptr, n_ends_ptr) = self.numeric_ptrs(request, device_fields);
            let counts_ptr = ptr(
                counts_dev
                    .ensure(&self.stream, blocks)
                    .map_err(|error| self.fail("allocation", error))?,
                &self.stream,
            );
            let mut launch = self
                .stream
                .launch_builder(&self.device.0.kernels.hits_count);
            launch
                .arg(&dist_ptr)
                .arg(&count)
                .arg(&has_cutoff)
                .arg(&limit)
                .arg(&counts_ptr);
            // SAFETY: matches sb_hits_count; one count slot per block.
            unsafe { launch.launch(config) }
                .map_err(|error| self.fail("launching sb_hits_count", error))?;
            let mut block_counts = try_filled(blocks, 0u64, "hit block counts")?;
            let mut offsets = Vec::new();
            try_reserve_exact(&mut offsets, blocks, "hit block offsets")?;
            self.download(counts_dev.get(), blocks, &mut block_counts)?;
            self.sync()?;
            let mut total = 0u64;
            for value in block_counts {
                offsets.push(total);
                total += value;
            }
            if total > 0 {
                let hits = total as usize;
                let stream = &self.stream;
                let alloc = |error| failure_ordinal(&self.device, "allocation", error);
                let offsets_ptr =
                    Self::upload(stream, &mut offsets_dev, &offsets).map_err(alloc)?;
                let i_ptr = ptr(out_i.ensure(stream, hits).map_err(alloc)?, stream);
                let j_ptr = ptr(out_j.ensure(stream, hits).map_err(alloc)?, stream);
                let d_ptr = ptr(out_d.ensure(stream, hits).map_err(alloc)?, stream);
                let e_ptr = if want_e {
                    ptr(out_e.ensure(stream, hits).map_err(alloc)?, stream)
                } else {
                    0
                };
                let c_ptr = if want_n {
                    ptr(out_n.ensure(stream, hits).map_err(alloc)?, stream)
                } else {
                    0
                };
                let space = chunk.space;
                let mut launch = self
                    .stream
                    .launch_builder(&self.device.0.kernels.hits_write);
                launch
                    .arg(&dist_ptr)
                    .arg(&count)
                    .arg(&has_cutoff)
                    .arg(&limit)
                    .arg(&offsets_ptr)
                    .arg(&space)
                    .arg(&i_ptr)
                    .arg(&j_ptr)
                    .arg(&d_ptr)
                    .arg(&b_end_ptr)
                    .arg(&n_ends_ptr)
                    .arg(&e_ptr)
                    .arg(&c_ptr);
                // SAFETY: matches sb_hits_write; outputs hold `total` entries.
                unsafe { launch.launch(config) }
                    .map_err(|error| self.fail("launching sb_hits_write", error))?;
                let base = dd.len();
                // Every hit column is reserved before any is written
                // (host-allocation-failures §12.6 rule 8).
                try_reserve(&mut ii, hits, "device hits i column")?;
                try_reserve(&mut jj, hits, "device hits j column")?;
                try_reserve(&mut dd, hits, "device hits key column")?;
                if want_e {
                    try_reserve(&mut ee, hits, "device hits b_end column")?;
                }
                if want_n {
                    try_reserve(&mut cc, hits, "device hits n_ends column")?;
                }
                ii.resize(base + hits, 0);
                jj.resize(base + hits, 0);
                dd.resize(base + hits, 0);
                self.download(out_i.get(), hits, &mut ii[base..])?;
                self.download(out_j.get(), hits, &mut jj[base..])?;
                self.download(out_d.get(), hits, &mut dd[base..])?;
                if want_e {
                    ee.resize(base + hits, 0);
                    self.download(out_e.get(), hits, &mut ee[base..])?;
                }
                if want_n {
                    cc.resize(base + hits, 0);
                    self.download(out_n.get(), hits, &mut cc[base..])?;
                }
            }
            pos += n;
            // A Session chunk boundary (cancellation-progress.md §3.5).
            cancel::boundary(Site::Pairs, n)?;
        }
        self.sync()?;
        let mut details = device_fields.then(|| {
            let fields = Numeric {
                b_end: want_e.then_some(ee),
                n_ends: want_n.then_some(cc),
                floats: [None, None, None],
            };
            self.numeric_columns(host, fields)
        });
        let mut columns = if matches!(work.kind, PairKind::Index { .. }) {
            // Explicit lists go through the CPU's host sort and dedup, with the
            // detail gather permutation when the device delivered fields.
            let mut hits: Vec<(usize, usize, i32)> = Vec::new();
            try_reserve_exact(&mut hits, dd.len(), "device hit records")?;
            hits.extend(
                ii.into_iter()
                    .zip(jj)
                    .zip(dd)
                    .map(|((i, j), d)| (i as usize, j as usize, d)),
            );
            if let Some(rows) = details.take() {
                let (sorted, rows) = kept::sort_hits(hits, rows)?;
                hits = sorted;
                details = Some(rows);
            }
            Columns::from_hits(hits)?
        } else {
            // Implicit shapes: work-ID order is (i, j) order and unique.
            Columns {
                dist: dd,
                i: Some(ii),
                j: Some(jj),
                ..Columns::default()
            }
        };
        if details.is_none() && !request.is_empty() && self.dup.is_some() {
            details = Some(self.dup_hit_details(host, &columns)?);
        }
        if details.is_none() && !request.is_empty() {
            // Heavy request: every requested field over the final hit list.
            let (i, j) = (columns.i.as_deref(), columns.j.as_deref());
            details = Some(exec::entry_details(
                host.kernel,
                host.a,
                host.b,
                (i.unwrap_or(&[]), j.unwrap_or(&[]), &columns.dist),
                self.workers,
            )?);
        }
        columns.details = details;
        Ok(columns)
    }

    /// RowStats: sweep 1 takes atomicMin(best) and counts; sweep 2 counts ties
    /// at the final best and takes the smallest partner. A single staged chunk
    /// retains its distances for sweep 2; multiple chunks are recomputed.
    fn rowstats(
        &mut self,
        host: Host<'_>,
        work: &WorkSource,
        cutoff: Option<i32>,
    ) -> Result<RowStatsColumns, EngineError> {
        let rows = work.shape.0;
        let alloc = |error| failure_ordinal(&self.device, "allocation", error);
        // SAFETY (all four): initialised by the fills below before any kernel reads them.
        let best = unsafe { dev_alloc::<i32>(&self.stream, rows.max(1)) }.map_err(alloc)?;
        let count = unsafe { dev_alloc::<u64>(&self.stream, rows.max(1)) }.map_err(alloc)?;
        let nbest = unsafe { dev_alloc::<u64>(&self.stream, rows.max(1)) }.map_err(alloc)?;
        let arg = unsafe { dev_alloc::<u64>(&self.stream, rows.max(1)) }.map_err(alloc)?;
        let first = match work.first_occurrences()? {
            Some(mask) => {
                let host_mask: Vec<u8> = mask.into_iter().map(u8::from).collect();
                let mut scratch = Scratch::<u8>::new();
                Self::upload(&self.stream, &mut scratch, &host_mask).map_err(alloc)?;
                Some(scratch)
            }
            None => None,
        };
        let stream = &self.stream;
        let table = SbRows {
            mirror: u64::from(work.mirrors_within()),
            first: first.as_ref().map_or(0, |mask| ptr(mask.get(), stream)),
            best: ptr(&best, stream),
            count: ptr(&count, stream),
            nbest: ptr(&nbest, stream),
            arg: ptr(&arg, stream),
        };
        self.fill(table.best, i32::MAX as u32, rows, true)?;
        self.fill(table.count, 0, rows * 8, false)?;
        self.fill(table.nbest, 0, rows * 8, false)?;
        self.fill(table.arg, 0xFF, rows * 8, false)?;
        let (has_cutoff, limit) = (i32::from(cutoff.is_some()), cutoff.unwrap_or(0));
        let request = Request::Range {
            work,
            end: work.n_pairs,
        };
        let mut single_chunk = None;
        for sweep in [0, 1] {
            let mut pos = 0;
            // Each sweep is one stage row over every pair.
            cancel::stage(Stage::Pairs, work.n_pairs);
            while pos < work.n_pairs {
                let chunk = if sweep == 1 {
                    match single_chunk.take() {
                        Some(chunk) => chunk,
                        None => self.stage(host, &request, pos, false)?,
                    }
                } else {
                    self.stage(host, &request, pos, false)?
                };
                let dist_ptr = self.dist_ptr();
                let config = self.grid(chunk.n as u64);
                let (kernel, name) = if sweep == 0 {
                    (&self.device.0.kernels.rowstats_min, "sb_rowstats_min")
                } else {
                    (&self.device.0.kernels.rowstats_arg, "sb_rowstats_arg")
                };
                let mut launch = self.stream.launch_builder(kernel);
                launch
                    .arg(&dist_ptr)
                    .arg(&chunk.space)
                    .arg(&has_cutoff)
                    .arg(&limit)
                    .arg(&table);
                // SAFETY: matches the rowstats kernels; row arrays hold `rows` entries.
                unsafe { launch.launch(config) }
                    .map_err(|error| self.fail(&format!("launching {name}"), error))?;
                pos += chunk.n;
                // A Session chunk boundary (cancellation-progress.md §3.5).
                cancel::boundary(Site::Pairs, chunk.n)?;
                if sweep == 0 && chunk.n == work.n_pairs {
                    single_chunk = Some(chunk);
                }
            }
        }
        // Every download buffer is reserved before any is written
        // (host-allocation-failures §12.6 rule 8).
        let (mut host_best, mut host_count) = (Vec::<i32>::new(), Vec::<u64>::new());
        let (mut host_nbest, mut host_arg) = (Vec::<u64>::new(), Vec::<u64>::new());
        try_reserve_exact(&mut host_best, rows, "rowstats best download")?;
        try_reserve_exact(&mut host_count, rows, "rowstats count download")?;
        try_reserve_exact(&mut host_nbest, rows, "rowstats n_best download")?;
        try_reserve_exact(&mut host_arg, rows, "rowstats argbest download")?;
        host_best.resize(rows, 0);
        host_count.resize(rows, 0);
        host_nbest.resize(rows, 0);
        host_arg.resize(rows, 0);
        self.download(&best, rows, &mut host_best)?;
        self.download(&count, rows, &mut host_count)?;
        self.download(&nbest, rows, &mut host_nbest)?;
        self.download(&arg, rows, &mut host_arg)?;
        self.sync()?;
        let mut states = Vec::new();
        try_reserve_exact(&mut states, rows, "rowstats states")?;
        for row in 0..rows {
            states.push(if host_count[row] == 0 {
                RowState::default()
            } else {
                let partner = i64::try_from(host_arg[row]).map_err(|_| {
                    EngineError::Device(format!("device rowstats argbest is invalid in row {row}"))
                })?;
                RowState::from_summary(host_best[row], partner, host_nbest[row], host_count[row])?
            });
        }
        RowStatsColumns::from_states(states, cutoff.is_some())
    }

    /// Device addresses of the numeric field buffers a `fields` stage wrote
    /// (0 for a field not computed).
    fn numeric_ptrs(&self, request: DetailRequest, fields: bool) -> (u64, u64) {
        match (&self.lev, fields) {
            (Some(lev), true) => lev.field_ptrs(&self.stream, request),
            _ => (0, 0),
        }
    }

    /// Host vectors for the device's numeric fields of `rows` items.
    fn numeric_host(&self, request: DetailRequest, rows: usize) -> Result<Numeric, EngineError> {
        let (e, n) = self
            .lev
            .as_ref()
            .map_or((false, false), |lev| lev.wanted(request));
        let wanted = self
            .dup
            .as_ref()
            .map_or([false; 3], |dup| dup.wanted(request));
        let mut floats = [None, None, None];
        for (slot, want) in floats.iter_mut().zip(wanted) {
            if want {
                *slot = Some(try_filled(rows, 0u64, "device float field column")?);
            }
        }
        Ok(Numeric {
            b_end: if e {
                Some(try_filled(rows, 0i64, "device b_end column")?)
            } else {
                None
            },
            n_ends: if n {
                Some(try_filled(rows, 0i32, "device n_ends column")?)
            } else {
                None
            },
            floats,
        })
    }

    /// Copy the staged chunk's numeric fields into `fields[pos..pos + n]`.
    fn download_numeric(
        &self,
        fields: &mut Numeric,
        pos: usize,
        n: usize,
    ) -> Result<(), EngineError> {
        if let Some(dup) = self.dup.as_ref() {
            for (k, values) in fields.floats.iter_mut().enumerate() {
                if let Some(values) = values.as_mut() {
                    self.download(dup.field_buffer(k), n, &mut values[pos..pos + n])?;
                }
            }
            return Ok(());
        }
        let lev = self.lev.as_ref().expect("levenshtein stage");
        if let Some(values) = fields.b_end.as_mut() {
            self.download(lev.b_end_buffer(), n, &mut values[pos..pos + n])?;
        }
        if let Some(values) = fields.n_ends.as_mut() {
            self.download(lev.n_ends_buffer(), n, &mut values[pos..pos + n])?;
        }
        Ok(())
    }

    /// The staged chunk's numeric fields as detail rows (after a sync).
    pub(crate) fn fetch_details(
        &self,
        host: Host<'_>,
        n: usize,
    ) -> Result<DetailColumns, EngineError> {
        let mut fields = self.numeric_host(host.kernel.detail_request(), n)?;
        self.download_numeric(&mut fields, 0, n)?;
        self.sync()?;
        Ok(self.numeric_columns(host, fields))
    }

    /// Detail columns of the call from whole numeric vectors.
    fn numeric_columns(&self, host: Host<'_>, fields: Numeric) -> DetailColumns {
        let mut columns =
            DetailColumns::new(host.kernel.detail_fields(), host.kernel.detail_request());
        if let Some(dup) = self.dup.as_ref() {
            // Float64 columns from their bit patterns, nothing else in between.
            for (k, values) in fields.floats.into_iter().enumerate() {
                if let Some(values) = values {
                    let values = values.into_iter().map(f64::from_bits).collect();
                    columns.set_column(dup.field_index[k], DetailData::Float64(values));
                }
            }
            return columns;
        }
        let lev = self.lev.as_ref().expect("levenshtein stage");
        if let Some(values) = fields.b_end {
            columns.set_column(lev.b_end_field, DetailData::Int64(values));
        }
        if let Some(values) = fields.n_ends {
            columns.set_column(lev.n_ends_field, DetailData::Int32(values));
        }
        columns
    }
}

/// Host copies of the device's numeric detail fields: levenshtein's `b_end`
/// and `n_ends`, duplex's three float64 fields as bit patterns.
struct Numeric {
    b_end: Option<Vec<i64>>,
    n_ends: Option<Vec<i32>>,
    floats: [Option<Vec<u64>>; 3],
}

impl Numeric {
    /// The canonical NaN in every float64 field of a failing row.
    fn mask_floats(&mut self, passed: &[u8]) {
        for values in self.floats.iter_mut().flatten() {
            for (value, &ok) in values.iter_mut().zip(passed) {
                if ok == 0 {
                    *value = CANONICAL_NAN.to_bits();
                }
            }
        }
    }
}

/// A session bound to one call's kernel and tables: the batch-distance port.
pub(crate) struct Bound<'s> {
    session: &'s mut Session,
    host: Host<'s>,
}

impl BatchDistance for Bound<'_> {
    fn delivers_details(&self) -> bool {
        self.session.delivers(self.host.kernel.detail_request())
    }

    fn range(
        &mut self,
        work: &WorkSource,
        start: usize,
        end: usize,
        details: bool,
        visit: &mut DistanceVisit<'_>,
    ) -> Result<(), EngineError> {
        let request = Request::Range { work, end };
        let details = details && self.delivers_details();
        let mut pos = start;
        while pos < end {
            let chunk = self.session.stage(self.host, &request, pos, details)?;
            self.session.fetch(chunk.n)?;
            let rows = if details {
                Some(self.session.fetch_details(self.host, chunk.n)?)
            } else {
                None
            };
            visit(pos, &self.session.host_dist[..chunk.n], rows.as_ref())?;
            pos += chunk.n;
        }
        Ok(())
    }

    fn pairs(
        &mut self,
        pairs: &[(usize, usize)],
        out: &mut Vec<i32>,
        mut details: Option<&mut DetailColumns>,
    ) -> Result<(), EngineError> {
        let request = Request::Pairs(pairs);
        let fields = details.is_some() && self.delivers_details();
        try_reserve(out, pairs.len(), "device batch keys")?;
        let mut pos = 0;
        while pos < pairs.len() {
            let chunk = self.session.stage(self.host, &request, pos, fields)?;
            self.session.fetch_into(chunk.n, out)?;
            if let Some(rows) = details.as_deref_mut() {
                let mut chunk_rows = self.chunk_rows(fields, chunk.n)?;
                rows.append(&mut chunk_rows)?;
            }
            pos += chunk.n;
        }
        Ok(())
    }

    fn rect(
        &mut self,
        rows: (usize, usize),
        cols: (usize, usize),
        class: &CellClass<'_>,
        out: &mut Vec<i32>,
        mut details: Option<&mut DetailColumns>,
    ) -> Result<(), EngineError> {
        let width = cols.1 - cols.0;
        let cells = (rows.1 - rows.0) * width;
        let request = Request::Rect {
            r0: rows.0,
            c0: cols.0,
            width,
            end: cells,
            class: Some(class),
        };
        let fields = details.is_some() && self.delivers_details();
        try_reserve(out, cells, "device batch keys")?;
        let mut pos = 0;
        while pos < cells {
            let chunk = self.session.stage(self.host, &request, pos, fields)?;
            self.session.fetch_into(chunk.n, out)?;
            if let Some(target) = details.as_deref_mut() {
                let mut chunk_rows = self.chunk_rows(fields, chunk.n)?;
                target.append(&mut chunk_rows)?;
            }
            pos += chunk.n;
        }
        Ok(())
    }
}

impl Bound<'_> {
    /// Detail rows of the staged chunk: the device's numeric fields, or (a
    /// backend that cannot deliver the request was asked anyway) a refusal.
    fn chunk_rows(&self, fields: bool, n: usize) -> Result<DetailColumns, EngineError> {
        if !fields {
            return Err(EngineError::Parameter(
                "this device call cannot deliver the requested detail fields".into(),
            ));
        }
        self.session.fetch_details(self.host, n)
    }
}

pub(crate) fn failure(label: &str, what: &str, error: DriverError) -> EngineError {
    EngineError::Device(format!(
        "CUDA {} during {what} on {label}; no result was produced; retry, or use device='cpu'",
        error_name(error)
    ))
}

pub(crate) fn failure_ordinal(device: &Device, what: &str, error: DriverError) -> EngineError {
    failure(&format!("cuda:{}", device.info().ordinal), what, error)
}

/// Unary upload of a fully validated table. A contiguous, full-width NumPy
/// view needs no Rust staging buffer: the driver reads its retained raw span.
/// No Rust slice or reference to the mutable foreign bytes is constructed.
/// All other layouts use the ordinary packing path.
pub(crate) fn upload_unary_table(
    stream: &Arc<CudaStream>,
    table: &SeqTable,
    workers: Workers,
    fail: &dyn Fn(&str, DriverError) -> EngineError,
) -> Result<Table, EngineError> {
    let Some((address, bytes)) = table.contiguous_fixed_span() else {
        return upload_table(stream, table, false, workers, fail).map(|(table, _)| table);
    };
    let n = table.len();
    let mut host_off = Vec::new();
    try_reserve_exact(&mut host_off, n + 1, "upload offsets")?;
    let width = bytes / n.max(1);
    // The span proof bounds n * width by int64/platform limits.
    host_off.extend((0..=n).map(|i| (i * width) as i64));
    // SAFETY: the data allocation is initialized by the ordered copy below.
    let mut data = unsafe { dev_alloc::<u8>(stream, bytes.max(1)) }
        .map_err(|error| fail("allocation", error))?;
    if bytes != 0 {
        stream
            .context()
            .bind_to_thread()
            .map_err(|error| fail("binding the context", error))?;
        let copy = {
            check_stream(data.stream(), stream);
            let (dst, _record) = data.device_ptr_mut(stream);
            // SAFETY: contiguous_fixed_span proves that all `bytes` belong to
            // the table's retained allocation. The producer's existing contract
            // forbids resize or mutation during execution. The destination has
            // sufficient capacity; the stream orders its allocation and copy.
            // Synchronize below, including on copy failure, before this borrow
            // of the table ends. This never aliases foreign bytes as Rust data.
            unsafe {
                sys::cuMemcpyHtoDAsync_v2(
                    dst,
                    address as *const std::ffi::c_void,
                    bytes,
                    stream.cu_stream(),
                )
            }
            .result()
        };
        let complete = stream.synchronize();
        copy.map_err(|error| fail("uploading sequences", error))?;
        complete.map_err(|error| fail("synchronizing sequence upload", error))?;
    }
    // SAFETY: every offset is initialized by the copy below.
    let mut off = unsafe { dev_alloc::<i64>(stream, host_off.len()) }
        .map_err(|error| fail("allocation", error))?;
    crate::session::htod(stream, &host_off, &mut off)
        .map_err(|error| fail("uploading offsets", error))?;
    Ok(Table {
        data,
        off,
        host_off,
    })
}

/// Upload a table; with `byte_set`, also return which byte values it holds
/// (one O(bytes) host pass, for a raw levenshtein table a). An owned batch is
/// copied directly; a view goes through one host staging copy, filled in
/// contiguous row ranges by up to `workers.limit` threads (the bytes do not
/// depend on the split).
pub(crate) fn upload_table(
    stream: &Arc<CudaStream>,
    table: &SeqTable,
    byte_set: bool,
    workers: Workers,
    fail: &dyn Fn(&str, DriverError) -> EngineError,
) -> Result<(Table, Option<[bool; 256]>), EngineError> {
    let staged: Vec<u8>;
    let (data, host_off): (&[u8], Vec<i64>) = match table.owned_batch() {
        Some(batch) => {
            // A fallible copy: a refused reservation raises DeviceError, not an abort.
            let mut offsets = Vec::new();
            try_reserve_exact(&mut offsets, batch.offsets().len(), "sequence offsets")?;
            offsets.extend_from_slice(batch.offsets());
            (batch.data(), offsets)
        }
        None => {
            let (bytes, offsets) = stage_view(table, workers, PARALLEL_STAGING_BYTES)?;
            staged = bytes;
            (&staged[..], offsets)
        }
    };
    let set = byte_set.then(|| {
        // Offsets may start past 0 (a sliced batch); scan exactly the rows' bytes.
        let (lo, hi) = (host_off[0] as usize, host_off[host_off.len() - 1] as usize);
        let mut seen = [false; 256];
        for &byte in &data[lo..hi] {
            seen[byte as usize] = true;
        }
        seen
    });
    // SAFETY (both): fully written by the copies below; never cuMemAlloc(0).
    let mut dev_data = unsafe { dev_alloc::<u8>(stream, data.len().max(1)) }
        .map_err(|error| fail("allocation", error))?;
    if !data.is_empty() {
        crate::session::htod(stream, data, &mut dev_data)
            .map_err(|error| fail("uploading sequences", error))?;
    }
    let mut dev_off = unsafe { dev_alloc::<i64>(stream, host_off.len()) }
        .map_err(|error| fail("allocation", error))?;
    crate::session::htod(stream, &host_off, &mut dev_off)
        .map_err(|error| fail("uploading offsets", error))?;
    Ok((
        Table {
            data: dev_data,
            off: dev_off,
            host_off,
        },
        set,
    ))
}

pub(crate) fn run(
    device: &Device,
    kernel: &dyn PairDistanceKernel,
    call: DeviceCall,
    work: &WorkSource,
    (a, b): (&SeqTable, &SeqTable),
    options: &ExecOptions,
    validation: ValidationReport,
) -> Result<DeviceRun, EngineError> {
    guard(|| {
        let mut session = Session::new(device, call, (a, b), options.workers(), false)?;
        let host = Host { kernel, a, b };
        let output = match options.sink {
            Sink::TopK => {
                let mut port = Bound {
                    session: &mut session,
                    host,
                };
                topk::run_batch(kernel, (a, b), &mut port, work, options, validation)?
            }
            Sink::Matrix => {
                let mut port = Bound {
                    session: &mut session,
                    host,
                };
                matrix::run_batch(kernel, (a, b), &mut port, work, options, validation)?
            }
            sink => {
                let columns = match sink {
                    Sink::Each => session.each(host, work, options.cutoff)?,
                    Sink::Hits => session.hits(host, work, options.cutoff)?,
                    _ => Columns {
                        rowstats: Some(session.rowstats(host, work, options.cutoff)?),
                        ..Columns::default()
                    },
                };
                RunOutput {
                    columns,
                    shape: work.shape,
                    counts: Counts {
                        certified: 0,
                        pruned: 0,
                        full: work.n_pairs,
                    },
                    threads: options.threads,
                    n_illegal_seqs: validation.n_illegal_seqs,
                    validation_scans: validation.scans,
                    topk_replay: None,
                }
            }
        };
        Ok(DeviceRun {
            output,
            host_pairs: session.host_pairs(),
        })
    })
}

/// K3 on the device (duplex-pf-cuda.md §7): every sink through the engine core's batch
/// drivers over the 64-bit-key port.
pub(crate) fn run_pf(
    device: &Device,
    kernel: &dyn PairDistanceKernel<i64>,
    call: DeviceCall,
    work: &WorkSource,
    (a, b): (&SeqTable, &SeqTable),
    options: &ExecOptions<i64>,
    validation: ValidationReport,
) -> Result<DeviceRun<i64>, EngineError> {
    pf_call(&call)?;
    guard(|| {
        let mut session = Session::new(device, call, (a, b), options.workers(), false)?;
        let output = {
            let mut port = PfBound {
                session: &mut session,
                host: Host { kernel, a, b },
            };
            exec::run_batch(kernel, (a, b), &mut port, work, options, validation)?
        };
        Ok(DeviceRun {
            output,
            host_pairs: session.host_pairs(),
        })
    })
}

/// Ẑ of every work item (the testing hook `duplex_pf_zhat`).
pub(crate) fn pf_zhat(
    device: &Device,
    kernel: &dyn PairDistanceKernel<i64>,
    call: DeviceCall,
    work: &WorkSource,
    (a, b): (&SeqTable, &SeqTable),
    workers: Workers,
) -> Result<(Vec<u64>, Vec<i64>), EngineError> {
    pf_call(&call)?;
    guard(|| {
        let mut session = Session::new(device, call, (a, b), workers, false)?;
        let mut port = PfBound {
            session: &mut session,
            host: Host { kernel, a, b },
        };
        port.zhat(&Request::Range {
            work,
            end: work.n_pairs,
        })
    })
}

/// K3's 64-bit-key entries (`run_pf`, `pf_zhat`) serve K3 only; `tm` on Pairs has its own
/// (`reg::run`).
fn pf_call(call: &DeviceCall) -> Result<(), EngineError> {
    match call {
        DeviceCall::DuplexPf(_) => Ok(()),
        _ => Err(EngineError::Parameter(
            "internal error: the 64-bit-key CUDA entry serves duplex model='pf' only".into(),
        )),
    }
}

/// Device state of one Tiles iterator: tables uploaded once, freed on drop.
pub struct TileSession(pub(crate) Session);

impl TileSession {
    /// `workers` are the stream's host workers (routed pairs).
    pub fn new(
        device: &Device,
        call: DeviceCall,
        tables: (&SeqTable, &SeqTable),
        workers: Workers,
    ) -> Result<Self, EngineError> {
        guard(|| Session::new(device, call, tables, workers, true).map(Self))
    }

    /// The batch-distance port for one `next()`, bound to the call's kernel
    /// and the stream's validated tables.
    pub fn bind<'s>(
        &'s mut self,
        kernel: &'s dyn PairDistanceKernel,
        (a, b): (&'s SeqTable, &'s SeqTable),
    ) -> TileBatch<'s> {
        TileBatch(Bound {
            session: &mut self.0,
            host: Host { kernel, a, b },
        })
    }

    /// The 64-bit-key batch-distance port of a K3 iterator for one `next()`
    /// (duplex-pf-cuda.md §7.2).
    pub fn bind_pf<'s>(
        &'s mut self,
        kernel: &'s dyn PairDistanceKernel<i64>,
        (a, b): (&'s SeqTable, &'s SeqTable),
    ) -> PfTileBatch<'s> {
        PfTileBatch(PfBound {
            session: &mut self.0,
            host: Host { kernel, a, b },
        })
    }

    /// Distinct work items routed to the host by this iterator so far.
    pub fn host_pairs(&self) -> usize {
        self.0.host_pairs()
    }
}

/// A K3 tile session bound for one `next()`; every call runs under `guard`.
pub struct PfTileBatch<'s>(PfBound<'s>);

impl BatchDistance<i64> for PfTileBatch<'_> {
    fn delivers_details(&self) -> bool {
        self.0.delivers_details()
    }

    fn range(
        &mut self,
        work: &WorkSource,
        start: usize,
        end: usize,
        details: bool,
        visit: &mut DistanceVisit<'_, i64>,
    ) -> Result<(), EngineError> {
        guard(|| self.0.range(work, start, end, details, visit))
    }

    fn pairs(
        &mut self,
        pairs: &[(usize, usize)],
        out: &mut Vec<i64>,
        details: Option<&mut DetailColumns>,
    ) -> Result<(), EngineError> {
        guard(|| self.0.pairs(pairs, out, details))
    }

    fn rect(
        &mut self,
        rows: (usize, usize),
        cols: (usize, usize),
        class: &CellClass<'_>,
        out: &mut Vec<i64>,
        details: Option<&mut DetailColumns>,
    ) -> Result<(), EngineError> {
        guard(|| self.0.rect(rows, cols, class, out, details))
    }
}

/// A tile session bound for one `next()`; every call runs under `guard`.
pub struct TileBatch<'s>(Bound<'s>);

impl BatchDistance for TileBatch<'_> {
    fn delivers_details(&self) -> bool {
        self.0.delivers_details()
    }

    fn range(
        &mut self,
        work: &WorkSource,
        start: usize,
        end: usize,
        details: bool,
        visit: &mut DistanceVisit<'_>,
    ) -> Result<(), EngineError> {
        guard(|| self.0.range(work, start, end, details, visit))
    }

    fn pairs(
        &mut self,
        pairs: &[(usize, usize)],
        out: &mut Vec<i32>,
        details: Option<&mut DetailColumns>,
    ) -> Result<(), EngineError> {
        guard(|| self.0.pairs(pairs, out, details))
    }

    fn rect(
        &mut self,
        rows: (usize, usize),
        cols: (usize, usize),
        class: &CellClass<'_>,
        out: &mut Vec<i32>,
        details: Option<&mut DetailColumns>,
    ) -> Result<(), EngineError> {
        guard(|| self.0.rect(rows, cols, class, out, details))
    }
}
