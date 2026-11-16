//! Integer-only resident sink construction. Only allocation/prefix counts and
//! overflow flags return to the host; sequence payloads and result arrays do not.
use crate::{
    interop::{DeviceColumn, DeviceDType, DeviceLease, DeviceOutput},
    resident::{
        ResidentChunk, ResidentEntries, ResidentField, ResidentFieldSpec, ResidentProducer,
        ResidentSinkConfig, ResidentSinkOutput, ResidentWork,
    },
    resident_producer::NativeProducer,
    session::{failure_ordinal, ptr},
};
use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, DeviceRepr, DriverError, LaunchConfig, PushKernelArg,
};
use seqblaze_core::{
    cancel::{self, Stage},
    sink::Sink,
    validate::EngineError,
};
use std::{
    mem::size_of,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

const THREADS: u32 = 256;
const MAX_COLS: usize = 32;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Entry {
    i: u64,
    j: u64,
    key: i64,
    work_id: u64,
    source: u64,
}
// SAFETY: five eight-byte integer words, matching SbResidentEntry exactly.
unsafe impl DeviceRepr for Entry {}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Cols {
    count: u64,
    src: [u64; MAX_COLS],
    dst: [u64; MAX_COLS],
    width: [u64; MAX_COLS],
    mask: [u64; MAX_COLS],
}
// SAFETY: integer-only repr(C), matching SbResidentCols exactly.
unsafe impl DeviceRepr for Cols {}
const _: () = {
    assert!(size_of::<Entry>() == 40);
    assert!(size_of::<Cols>() == 1032);
};

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct TopKRow {
    capacity: u64,
    workspace: u64,
    base_size: u64,
    forest_mask: u64,
    final_size: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct TopKSegment {
    row: u64,
    first: u64,
    count: u64,
    carry: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct TopKView {
    source: u64,
    offset: u64,
    size: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct TopKTask {
    a: TopKView,
    b: TopKView,
    size: u64,
    target: u64,
    scratch: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct TopKJob {
    task: u64,
    first: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct TopKGrowth {
    old_row: TopKRow,
    capacity: u64,
    workspace: u64,
}
// SAFETY: these repr(C) records contain only initialized fixed-width integer
// fields, and their exact sizes are asserted against resident_topk.cuh below.
unsafe impl DeviceRepr for TopKRow {}
unsafe impl DeviceRepr for TopKSegment {}
unsafe impl DeviceRepr for TopKTask {}
unsafe impl DeviceRepr for TopKJob {}
unsafe impl DeviceRepr for TopKGrowth {}
const _: () = {
    assert!(size_of::<TopKRow>() == 40);
    assert!(size_of::<TopKSegment>() == 32);
    assert!(size_of::<TopKView>() == 24);
    assert!(size_of::<TopKTask>() == 72);
    assert!(size_of::<TopKJob>() == 16);
    assert!(size_of::<TopKGrowth>() == 56);
};

pub(crate) struct Kernels {
    build: CudaFunction,
    compact: CudaFunction,
    radix_hist: CudaFunction,
    radix_scatter: CudaFunction,
    fields: CudaFunction,
    unpack: CudaFunction,
    fill: CudaFunction,
    dense: CudaFunction,
    row_min: CudaFunction,
    row_arg: CudaFunction,
    row_finish: CudaFunction,
    topk_finish: CudaFunction,
    topk_init_empty: CudaFunction,
    topk_segments: CudaFunction,
    topk_growth_prepare: CudaFunction,
    topk_growth_copy_jobs: CudaFunction,
    topk_growth_commit: CudaFunction,
    topk_chunk_large: CudaFunction,
    topk_chunk_level: CudaFunction,
    topk_chunk_flush: CudaFunction,
    topk_final_begin: CudaFunction,
    topk_final_level: CudaFunction,
    topk_final_flush: CudaFunction,
    topk_merge_jobs: CudaFunction,
    topk_commit_jobs: CudaFunction,
    topk_thresholds: CudaFunction,
    fixed_block_supported: bool,
}
impl Kernels {
    pub(crate) fn load(module: &Arc<CudaModule>) -> Result<Self, DriverError> {
        let mut result = Self {
            build: module.load_function("sb_resident_build")?,
            compact: module.load_function("sb_resident_compact")?,
            radix_hist: module.load_function("sb_resident_radix_hist")?,
            radix_scatter: module.load_function("sb_resident_radix_scatter")?,
            fields: module.load_function("sb_resident_fields")?,
            unpack: module.load_function("sb_resident_unpack")?,
            fill: module.load_function("sb_resident_fill")?,
            dense: module.load_function("sb_resident_dense")?,
            row_min: module.load_function("sb_resident_row_min")?,
            row_arg: module.load_function("sb_resident_row_arg")?,
            row_finish: module.load_function("sb_resident_row_finish")?,
            topk_finish: module.load_function("sb_resident_topk_finish")?,
            topk_init_empty: module.load_function("sb_resident_topk_init_empty")?,
            topk_segments: module.load_function("sb_resident_topk_segments")?,
            topk_growth_prepare: module.load_function("sb_resident_topk_growth_prepare")?,
            topk_growth_copy_jobs: module.load_function("sb_resident_topk_growth_copy_jobs")?,
            topk_growth_commit: module.load_function("sb_resident_topk_growth_commit")?,
            topk_chunk_large: module.load_function("sb_resident_topk_chunk_large")?,
            topk_chunk_level: module.load_function("sb_resident_topk_chunk_level")?,
            topk_chunk_flush: module.load_function("sb_resident_topk_chunk_flush")?,
            topk_final_begin: module.load_function("sb_resident_topk_final_begin")?,
            topk_final_level: module.load_function("sb_resident_topk_final_level")?,
            topk_final_flush: module.load_function("sb_resident_topk_final_flush")?,
            topk_merge_jobs: module.load_function("sb_resident_topk_merge_jobs")?,
            topk_commit_jobs: module.load_function("sb_resident_topk_commit_jobs")?,
            topk_thresholds: module.load_function("sb_resident_topk_thresholds")?,
            fixed_block_supported: true,
        };
        for function in [
            &result.build,
            &result.compact,
            &result.radix_hist,
            &result.radix_scatter,
            &result.fields,
            &result.unpack,
            &result.fill,
            &result.dense,
            &result.row_min,
            &result.row_arg,
            &result.row_finish,
            &result.topk_finish,
            &result.topk_init_empty,
            &result.topk_segments,
            &result.topk_growth_prepare,
            &result.topk_growth_copy_jobs,
            &result.topk_growth_commit,
            &result.topk_chunk_large,
            &result.topk_chunk_level,
            &result.topk_chunk_flush,
            &result.topk_final_begin,
            &result.topk_final_level,
            &result.topk_final_flush,
            &result.topk_merge_jobs,
            &result.topk_commit_jobs,
            &result.topk_thresholds,
        ] {
            result.fixed_block_supported &= function.max_threads_per_block()? >= THREADS as i32;
        }
        Ok(result)
    }
}
fn fail(lease: &DeviceLease, operation: &str, error: DriverError) -> EngineError {
    failure_ordinal(lease.device(), operation, error)
}
fn parameter(text: &str) -> EngineError {
    EngineError::Parameter(text.into())
}
fn grid(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n.div_ceil(THREADS as usize).clamp(1, 65_535)) as u32, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}
fn exact_grid(n: usize) -> Result<LaunchConfig, EngineError> {
    let blocks = u32::try_from(n.div_ceil(THREADS as usize).max(1))
        .map_err(|_| parameter("resident radix chunk grid exceeds CUDA limits"))?;
    Ok(LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    })
}
fn alloc<T: DeviceRepr>(
    lease: &DeviceLease,
    n: usize,
) -> Result<crate::arena::Scratch<T>, EngineError> {
    n.max(1)
        .checked_mul(size_of::<T>())
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(|| parameter("resident sink allocation size exceeds platform limits"))?;
    // SAFETY: all logical elements are initialized by stream-ordered kernels
    // or copies before use; an empty allocation has no logical read.
    unsafe { lease.execution().scratch(n.max(1)) }
        .map_err(|e| fail(lease, "allocating resident sink state", e))
}
pub(crate) fn scan_u64(
    lease: &DeviceLease,
    input: &CudaSlice<u64>,
    n: usize,
) -> Result<crate::arena::Scratch<u64>, EngineError> {
    crate::interop::scan(lease, input, n)
}
fn control(lease: &DeviceLease, input: &CudaSlice<u64>, at: usize) -> Result<u64, EngineError> {
    let mut out = [0u64];
    lease
        .execution()
        .memcpy_dtoh(&input.slice(at..at + 1), &mut out[..])
        .map_err(|e| fail(lease, "reading resident sink control count", e))?;
    lease.synchronize()?;
    Ok(out[0])
}
fn fill(
    lease: &DeviceLease,
    kernels: &Kernels,
    address: u64,
    n: usize,
    width: usize,
    bits: u64,
) -> Result<(), EngineError> {
    let (n64, w) = (n as u64, width as u64);
    let mut launch = lease.execution().launch_builder(&kernels.fill);
    launch.arg(&address).arg(&n64).arg(&w).arg(&bits);
    // SAFETY: caller supplies an initialized output allocation of n*width
    // writable bytes on this stream; width is an integral scalar width.
    unsafe { launch.launch(grid(n)) }
        .map_err(|e| fail(lease, "initializing resident sink output", e))?;
    Ok(())
}
fn copy(lease: &DeviceLease, dst: u64, src: u64, bytes: usize) -> Result<(), EngineError> {
    if bytes == 0 {
        return Ok(());
    }
    // SAFETY: both disjoint live allocations cover bytes and belong to this
    // lease context; operations and destruction use the same consumer stream.
    unsafe { lease.execution().copy_device(dst, src, bytes) }
        .map_err(|e| fail(lease, "joining resident result chunks", e))
}
fn publish(output: DeviceOutput) -> Result<DeviceColumn, EngineError> {
    // SAFETY: every caller queues a complete initialization before publishing
    // and relinquishes the construction-only writable allocation here.
    unsafe { output.finish() }
}
fn named(name: &str, values: DeviceColumn, mask: u64) -> ResidentField {
    ResidentField {
        name: name.into(),
        values,
        mask_bits: mask,
    }
}
fn mask(dtype: DeviceDType) -> Result<i64, EngineError> {
    match dtype {
        DeviceDType::I32 => Ok(i32::MAX as i64),
        DeviceDType::I64 => Ok(i64::MAX),
        _ => Err(parameter(
            "resident selecting keys must be signed int32 or int64",
        )),
    }
}
fn columns(
    specs: &[ResidentFieldSpec],
    sources: &[ResidentField],
    destinations: &[DeviceOutput],
) -> Result<Cols, EngineError> {
    if specs.len() > MAX_COLS || specs.len() != sources.len() || specs.len() != destinations.len() {
        return Err(parameter("resident numeric column schema mismatch"));
    }
    let mut cols = Cols {
        count: specs.len() as u64,
        ..Cols::default()
    };
    for (at, ((spec, src), dst)) in specs.iter().zip(sources).zip(destinations).enumerate() {
        if spec.name != src.name || spec.dtype != src.values.dtype() {
            return Err(parameter("resident stage column name or dtype changed"));
        }
        cols.src[at] = src.values.address();
        cols.dst[at] = dst.address();
        cols.width[at] = spec.dtype.width() as u64;
        cols.mask[at] = spec.mask_bits;
    }
    Ok(cols)
}
fn allocate_fields(
    lease: &DeviceLease,
    specs: &[ResidentFieldSpec],
    shape: &[usize],
) -> Result<Vec<DeviceOutput>, EngineError> {
    let mut fields = Vec::new();
    fields.try_reserve_exact(specs.len()).map_err(|_| {
        EngineError::Device("unable to retain resident numeric output columns".into())
    })?;
    for spec in specs {
        fields.push(lease.output(spec.dtype, shape.to_vec())?);
    }
    Ok(fields)
}
struct Records {
    values: crate::arena::Scratch<Entry>,
    n: usize,
}
fn validate_chunk(
    lease: &DeviceLease,
    chunk: &ResidentChunk,
    dtype: DeviceDType,
) -> Result<usize, EngineError> {
    let n = usize::try_from(chunk.space.n)
        .map_err(|_| parameter("resident chunk length exceeds platform limits"))?;
    if chunk.keys.dtype() != dtype
        || chunk.keys.shape() != [n]
        || chunk.keys.ordinal() != lease.device().info().ordinal
    {
        return Err(parameter(
            "resident stage keys have incompatible dtype, shape or device",
        ));
    }
    for field in &chunk.fields {
        if field.values.shape() != [n] || field.values.ordinal() != lease.device().info().ordinal {
            return Err(parameter("resident stage field shape or device mismatch"));
        }
    }
    Ok(n)
}
#[allow(clippy::too_many_arguments)]
fn build(
    lease: &DeviceLease,
    kernels: &Kernels,
    chunk: &ResidentChunk,
    cutoff: Option<i64>,
    mirror: bool,
    keep_failures: bool,
    thresholds: u64,
    active: u64,
    base: usize,
    passed: Option<&DeviceColumn>,
) -> Result<Records, EngineError> {
    let n = validate_chunk(lease, chunk, chunk.keys.dtype())?;
    if let Some(passed) = passed {
        if passed.dtype() != DeviceDType::Bool
            || passed.shape() != [n]
            || passed.ordinal() != lease.device().info().ordinal
        {
            return Err(parameter(
                "resident cutoff mask has incompatible dtype, shape or device",
            ));
        }
    }
    let passed = passed.map_or(0, DeviceColumn::address);
    let slots = n
        .checked_mul(if mirror { 2 } else { 1 })
        .ok_or_else(|| parameter("resident mirror chunk size overflows"))?;
    let output = alloc::<Entry>(lease, slots)?;
    let flags = alloc::<u64>(lease, slots)?;
    let (keys, width, has, limit, m, all) = (
        chunk.keys.address(),
        chunk.keys.dtype().width() as u64,
        u64::from(cutoff.is_some()),
        cutoff.unwrap_or(0),
        u64::from(mirror),
        u64::from(keep_failures),
    );
    let (out, flag) = (ptr(&output, lease.stream()), ptr(&flags, lease.stream()));
    let mut launch = lease.execution().launch_builder(&kernels.build);
    launch
        .arg(&chunk.space)
        .arg(&keys)
        .arg(&width)
        .arg(&has)
        .arg(&limit)
        .arg(&thresholds)
        .arg(&active)
        .arg(&m)
        .arg(&all)
        .arg(&out)
        .arg(&flag)
        .arg(&passed);
    // SAFETY: the producer validates row coordinates; both potential mirror
    // slots and their flags are initialized for every chunk work item.
    unsafe { launch.launch(grid(n)) }
        .map_err(|e| fail(lease, "forming resident sink entries", e))?;
    if keep_failures && !mirror {
        return Ok(Records { values: output, n });
    }
    let prefix = scan_u64(lease, &flags, slots)?;
    let kept = usize::try_from(control(lease, &prefix, slots)?)
        .map_err(|_| parameter("resident kept count exceeds platform limits"))?;
    let compact = alloc::<Entry>(lease, kept)?;
    let (prefix_ptr, dst, total, source_base) = (
        ptr(&prefix, lease.stream()),
        ptr(&compact, lease.stream()),
        slots as u64,
        base as u64,
    );
    let mut launch = lease.execution().launch_builder(&kernels.compact);
    launch
        .arg(&out)
        .arg(&flag)
        .arg(&prefix_ptr)
        .arg(&total)
        .arg(&source_base)
        .arg(&dst);
    // SAFETY: exclusive prefix covers every source flag and the exact kept allocation.
    unsafe { launch.launch(grid(slots)) }
        .map_err(|e| fail(lease, "compacting resident sink entries", e))?;
    Ok(Records {
        values: compact,
        n: kept,
    })
}
fn sort(
    lease: &DeviceLease,
    kernels: &Kernels,
    mut records: Records,
    work: &ResidentWork,
    dtype: DeviceDType,
    partners: bool,
) -> Result<Records, EngineError> {
    if records.n < 2 {
        return Ok(records);
    }
    let mut scratch = alloc::<Entry>(lease, records.n)?;
    let launch_grid = exact_grid(records.n)?;
    let blocks = launch_grid.grid_dim.0 as usize;
    let hist_len = blocks
        .checked_mul(256)
        .ok_or_else(|| parameter("resident radix histogram size overflows"))?;
    let histogram = alloc::<u64>(lease, hist_len)?;
    let key_field = if dtype == DeviceDType::I32 { 3 } else { 1 };
    let bytes = |n: usize| {
        ((usize::BITS - (n.saturating_sub(1)).leading_zeros()).max(1) as usize).div_ceil(8)
    };
    let passes = [
        (0u64, if partners { bytes(work.n_b) } else { 0 }),
        (key_field, if dtype == DeviceDType::I32 { 4 } else { 8 }),
        (2u64, bytes(work.n_a)),
    ];
    for (field, count) in passes {
        for byte in 0..count {
            let (src, dst, n, shift, hist) = (
                ptr(&records.values, lease.stream()),
                ptr(&scratch, lease.stream()),
                records.n as u64,
                (byte * 8) as u64,
                ptr(&histogram, lease.stream()),
            );
            let mut launch = lease.execution().launch_builder(&kernels.radix_hist);
            launch.arg(&src).arg(&n).arg(&field).arg(&shift).arg(&hist);
            // SAFETY: fixed 256-thread histogram and complete initialized entry input.
            unsafe { launch.launch(launch_grid) }
                .map_err(|e| fail(lease, "counting resident radix bins", e))?;
            let prefix = scan_u64(lease, &histogram, hist_len)?;
            let pre = ptr(&prefix, lease.stream());
            let mut launch = lease.execution().launch_builder(&kernels.radix_scatter);
            launch
                .arg(&src)
                .arg(&n)
                .arg(&field)
                .arg(&shift)
                .arg(&pre)
                .arg(&dst);
            // SAFETY: exact exclusive bucket-major prefix gives a unique in-bounds destination.
            unsafe { launch.launch(launch_grid) }
                .map_err(|e| fail(lease, "ordering resident entries", e))?;
            std::mem::swap(&mut records.values, &mut scratch);
        }
    }
    Ok(records)
}
#[allow(clippy::too_many_arguments)]
fn gather_fields(
    lease: &DeviceLease,
    kernels: &Kernels,
    records: &Records,
    specs: &[ResidentFieldSpec],
    source: &[ResidentField],
    outputs: &[DeviceOutput],
    from_chunk: bool,
    start: u64,
    destination: usize,
    cutoff: Option<i64>,
) -> Result<(), EngineError> {
    let cols = columns(specs, source, outputs)?;
    let (entries, n, from, dest, has, limit) = (
        ptr(&records.values, lease.stream()),
        records.n as u64,
        u64::from(from_chunk),
        destination as u64,
        u64::from(cutoff.is_some()),
        cutoff.unwrap_or(0),
    );
    let mut launch = lease.execution().launch_builder(&kernels.fields);
    launch
        .arg(&entries)
        .arg(&n)
        .arg(&cols)
        .arg(&from)
        .arg(&start)
        .arg(&dest)
        .arg(&has)
        .arg(&limit);
    // SAFETY: entry sources index the matching producer chunk or concatenated
    // kept columns; every destination scalar is written exactly once.
    unsafe { launch.launch(grid(records.n)) }
        .map_err(|e| fail(lease, "gathering resident numeric fields", e))?;
    Ok(())
}
#[allow(clippy::too_many_arguments)]
fn unpack(
    lease: &DeviceLease,
    kernels: &Kernels,
    records: &Records,
    destination: usize,
    addresses: [u64; 5],
    dtype: DeviceDType,
    cutoff: Option<i64>,
) -> Result<(), EngineError> {
    let (src, n, dest, width, has, limit, mask) = (
        ptr(&records.values, lease.stream()),
        records.n as u64,
        destination as u64,
        dtype.width() as u64,
        u64::from(cutoff.is_some()),
        cutoff.unwrap_or(0),
        mask(dtype)?,
    );
    let mut launch = lease.execution().launch_builder(&kernels.unpack);
    launch
        .arg(&src)
        .arg(&n)
        .arg(&dest)
        .arg(&addresses[0])
        .arg(&addresses[1])
        .arg(&addresses[2])
        .arg(&addresses[3])
        .arg(&width)
        .arg(&addresses[4])
        .arg(&has)
        .arg(&limit)
        .arg(&mask);
    // SAFETY: each nonzero destination covers destination+n initialized scalar
    // positions, with exact integral widths and disjoint output columns.
    unsafe { launch.launch(grid(records.n)) }
        .map_err(|e| fail(lease, "publishing resident entry columns", e))?;
    Ok(())
}

pub(crate) type ChunkObserver<'a> =
    dyn FnMut(&DeviceLease, &ResidentChunk, u32) -> Result<(), EngineError> + 'a;
pub(crate) fn run(
    lease: &DeviceLease,
    kernels: &Kernels,
    work: &ResidentWork,
    producer: &mut dyn ResidentProducer,
    config: &ResidentSinkConfig,
) -> Result<ResidentSinkOutput, EngineError> {
    run_staged(lease, kernels, work, producer, config, None, Stage::Pairs)
}
pub(crate) fn run_observed(
    lease: &DeviceLease,
    kernels: &Kernels,
    work: &ResidentWork,
    producer: &mut NativeProducer<'_>,
    config: &ResidentSinkConfig,
    observer: Option<&mut ChunkObserver<'_>>,
) -> Result<ResidentSinkOutput, EngineError> {
    let halt = producer.replay_control();
    run_staged_with_halt(
        lease,
        kernels,
        work,
        producer,
        config,
        observer,
        Stage::Pairs,
        Some(&halt),
    )
}

/// The host chunk boundaries of one sink pass (cancellation-progress.md §3.5,
/// §4.2): the stage opens with the pass's work items; a boundary for each chunk
/// comes when the sink asks for the next one, after it has enqueued the chunk's
/// work. Streamed dense tiles (`stage` None) report candidates in the tile
/// producer itself.
struct Boundaries<'p, 'r> {
    inner: &'p mut (dyn ResidentProducer + 'r),
    stage: Option<Stage>,
    total: usize,
    pending: Option<usize>,
}
impl<'p, 'r> Boundaries<'p, 'r> {
    fn new(inner: &'p mut (dyn ResidentProducer + 'r), stage: Option<Stage>, total: usize) -> Self {
        if let Some(stage) = stage {
            cancel::stage(stage, total);
        }
        Self {
            inner,
            stage,
            total,
            pending: None,
        }
    }
    fn finish_chunk(&mut self) -> Result<(), EngineError> {
        if let (Some(stage), Some(items)) = (self.stage, self.pending.take()) {
            cancel::boundary(stage.site(), items)?;
        }
        Ok(())
    }
}
impl ResidentProducer for Boundaries<'_, '_> {
    fn key_dtype(&self) -> DeviceDType {
        self.inner.key_dtype()
    }
    fn fields(&self) -> &[ResidentFieldSpec] {
        self.inner.fields()
    }
    fn next_chunk(&mut self, lease: &DeviceLease) -> Result<Option<ResidentChunk>, EngineError> {
        self.finish_chunk()?;
        let chunk = self.inner.next_chunk(lease)?;
        if let Some(chunk) = &chunk {
            self.pending = Some(chunk.space.n as usize);
        }
        Ok(chunk)
    }
    fn passed(&self) -> Option<&DeviceColumn> {
        self.inner.passed()
    }
    fn replay_limits(&mut self, values: u64, active: u64, mirror: bool) {
        self.inner.replay_limits(values, active, mirror);
    }
    /// A replay sweep is a new stage row of the same kind and size.
    fn reset(&mut self) -> Result<(), EngineError> {
        self.inner.reset()?;
        self.pending = None;
        if let Some(stage) = self.stage {
            cancel::stage(stage, self.total);
        }
        Ok(())
    }
}

/// [`run_observed`] with the progress stage of the call's primary pass:
/// `pairs` for pairwise sinks, `sequences` for unary calls.
pub(crate) fn run_staged(
    lease: &DeviceLease,
    kernels: &Kernels,
    work: &ResidentWork,
    producer: &mut dyn ResidentProducer,
    config: &ResidentSinkConfig,
    observer: Option<&mut ChunkObserver<'_>>,
    primary: Stage,
) -> Result<ResidentSinkOutput, EngineError> {
    run_staged_with_halt(
        lease, kernels, work, producer, config, observer, primary, None,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_staged_with_halt(
    lease: &DeviceLease,
    kernels: &Kernels,
    work: &ResidentWork,
    producer: &mut dyn ResidentProducer,
    config: &ResidentSinkConfig,
    mut observer: Option<&mut ChunkObserver<'_>>,
    primary: Stage,
    halt: Option<&AtomicBool>,
) -> Result<ResidentSinkOutput, EngineError> {
    if !kernels.fixed_block_supported {
        return Err(EngineError::Device(
            "resident fixed scan/radix geometry requires 256 supported threads per block".into(),
        ));
    }
    let dtype = producer.key_dtype();
    let _ = mask(dtype)?;
    if work.n_a > i64::MAX as usize || work.n_b > i64::MAX as usize {
        return Err(EngineError::Overflow(
            "resident output indices exceed int64".into(),
        ));
    }
    let items = usize::try_from(work.space.n)
        .map_err(|_| parameter("resident work exceeds platform limits"))?;
    let primary = (config.tile.is_none()).then_some(primary);
    match config.sink {
        Sink::Each => {
            let producer = &mut Boundaries::new(producer, primary, items);
            each(lease, kernels, work, producer, config, &mut observer)
        }
        Sink::Hits => {
            let producer = &mut Boundaries::new(producer, primary, items);
            sparse(lease, kernels, work, producer, config, None, &mut observer)
        }
        Sink::RowStats => {
            let producer = &mut Boundaries::new(producer, primary, items);
            rowstats(lease, kernels, work, producer, config, &mut observer, halt)
        }
        Sink::Matrix => {
            let producer = &mut Boundaries::new(producer, primary, items);
            dense(lease, kernels, work, producer, config, &mut observer)
        }
        Sink::TopK => {
            if config.k == 0 {
                return Err(parameter("TopK requires a positive k"));
            }
            let thresholds = {
                let producer = &mut Boundaries::new(producer, Some(Stage::TopK1), items);
                topk_thresholds(lease, kernels, work, producer, config, &mut observer)?
            };
            // A bounded producer evaluates the replay against these row limits (§6.2).
            producer.replay_limits(
                ptr(&thresholds.values, lease.stream()),
                ptr(&thresholds.active, lease.stream()),
                work.mirror && config.mirror_symmetric,
            );
            producer.reset()?;
            let producer = &mut Boundaries::new(producer, Some(Stage::TopK2), items);
            sparse(
                lease,
                kernels,
                work,
                producer,
                config,
                Some(&thresholds),
                &mut observer,
            )
        }
    }
}
fn observe(
    observer: &mut Option<&mut ChunkObserver<'_>>,
    lease: &DeviceLease,
    chunk: &ResidentChunk,
    pass: u32,
) -> Result<(), EngineError> {
    if let Some(callback) = observer.as_deref_mut() {
        callback(lease, chunk, pass)?;
    }
    Ok(())
}

/// Positional Each with no cutoff or detail replay. pairs()/selection lineage
/// use the original zip domain (or the unary table), not sink entry columns.
/// Reuse the sole actual chunk directly; join only a genuinely chunked call.
pub(crate) fn each_fields(
    lease: &DeviceLease,
    work: &ResidentWork,
    producer: &mut dyn ResidentProducer,
    stage: Stage,
) -> Result<ResidentSinkOutput, EngineError> {
    // Keep the existing resident entry's preflight/error precedence even
    // though this path no longer launches the fixed-block sink kernels.
    if !lease.device().0.resident_sinks.fixed_block_supported {
        return Err(EngineError::Device(
            "resident fixed scan/radix geometry requires 256 supported threads per block".into(),
        ));
    }
    let dtype = producer.key_dtype();
    mask(dtype)?;
    if work.n_a > i64::MAX as usize || work.n_b > i64::MAX as usize {
        return Err(EngineError::Overflow(
            "resident output indices exceed int64".into(),
        ));
    }
    let n = usize::try_from(work.space.n)
        .map_err(|_| parameter("Each length exceeds platform limits"))?;
    let specs = producer.fields().to_vec();
    let mut producer = Boundaries::new(producer, Some(stage), n);
    let mut first = producer.next_chunk(lease)?;
    if let Some(chunk) = first.as_ref() {
        let len = validate_chunk(lease, chunk, dtype)?;
        if chunk.space.k0 != 0 || len > n {
            return Err(parameter(
                "Each producer must emit contiguous original work IDs",
            ));
        }
        if len == n {
            // Flush the final boundary, including cancellation, before reuse.
            if producer.next_chunk(lease)?.is_some() {
                return Err(parameter(
                    "Each producer must emit contiguous original work IDs",
                ));
            }
            let chunk = first.take().expect("complete chunk");
            let columns = chunk.fields;
            return Ok(ResidentSinkOutput {
                columns,
                entries: None,
                shape: [work.n_a, work.n_b],
                tie_rows: 0,
            });
        }
    }
    let fields = allocate_fields(lease, &specs, &[n])?;
    let mut consumed = 0usize;
    loop {
        let next = match first.take() {
            Some(chunk) => Some(chunk),
            None => producer.next_chunk(lease)?,
        };
        let Some(chunk) = next else {
            break;
        };
        let len = validate_chunk(lease, &chunk, dtype)?;
        if chunk.space.k0 != consumed as u64 || consumed.checked_add(len).is_none_or(|end| end > n)
        {
            return Err(parameter(
                "Each producer must emit contiguous original work IDs",
            ));
        }
        // Check the unchanged field schema before copying complete columns.
        let cols = columns(&specs, &chunk.fields, &fields)?;
        for at in 0..specs.len() {
            let width = cols.width[at] as usize;
            copy(
                lease,
                cols.dst[at] + (consumed * width) as u64,
                cols.src[at],
                len * width,
            )?;
        }
        consumed += len;
    }
    if consumed != n {
        return Err(parameter(
            "Each producer ended before the complete original work list",
        ));
    }
    let mut columns = Vec::with_capacity(specs.len());
    for (spec, field) in specs.iter().zip(fields) {
        columns.push(named(&spec.name, publish(field)?, spec.mask_bits));
    }
    Ok(ResidentSinkOutput {
        columns,
        entries: None,
        shape: [work.n_a, work.n_b],
        tie_rows: 0,
    })
}

fn each(
    lease: &DeviceLease,
    kernels: &Kernels,
    work: &ResidentWork,
    producer: &mut dyn ResidentProducer,
    config: &ResidentSinkConfig,
    observer: &mut Option<&mut ChunkObserver<'_>>,
) -> Result<ResidentSinkOutput, EngineError> {
    let dtype = producer.key_dtype();
    let n = usize::try_from(work.space.n)
        .map_err(|_| parameter("Each length exceeds platform limits"))?;
    let specs = producer.fields().to_vec();
    let fields = allocate_fields(lease, &specs, &[n])?;
    let ids = lease.output(DeviceDType::U64, vec![n])?;
    let i = lease.output(DeviceDType::I64, vec![n])?;
    let j = lease.output(DeviceDType::I64, vec![n])?;
    let keys = lease.output(dtype, vec![n])?;
    let passed = if config.cutoff.is_some() {
        Some(lease.output(DeviceDType::Bool, vec![n])?)
    } else {
        None
    };
    let addresses = [
        ids.address(),
        i.address(),
        j.address(),
        keys.address(),
        passed.as_ref().map_or(0, DeviceOutput::address),
    ];
    let mut consumed = 0usize;
    while let Some(chunk) = producer.next_chunk(lease)? {
        let len = validate_chunk(lease, &chunk, dtype)?;
        if chunk.space.k0 != consumed as u64 || consumed.checked_add(len).is_none_or(|end| end > n)
        {
            return Err(parameter(
                "Each producer must emit contiguous original work IDs",
            ));
        }
        observe(observer, lease, &chunk, 0)?;
        let records = build(
            lease,
            kernels,
            &chunk,
            config.cutoff,
            false,
            true,
            0,
            0,
            consumed,
            producer.passed(),
        )?;
        gather_fields(
            lease,
            kernels,
            &records,
            &specs,
            &chunk.fields,
            &fields,
            true,
            chunk.space.k0,
            consumed,
            config.cutoff,
        )?;
        unpack(
            lease,
            kernels,
            &records,
            consumed,
            addresses,
            dtype,
            config.cutoff,
        )?;
        if let (Some(source), Some(target)) = (producer.passed(), passed.as_ref()) {
            copy(
                lease,
                target.address() + consumed as u64,
                source.address(),
                len,
            )?;
        }
        consumed += len;
    }
    if consumed != n {
        return Err(parameter(
            "Each producer ended before the complete original work list",
        ));
    }
    let (ids, i, j, keys) = (publish(ids)?, publish(i)?, publish(j)?, publish(keys)?);
    let passed = passed.map(publish).transpose()?;
    let mut columns = Vec::with_capacity(specs.len() + 2);
    columns.push(named("__keys", keys.clone(), mask(dtype)? as u64));
    for (spec, value) in specs.iter().zip(fields) {
        columns.push(named(&spec.name, publish(value)?, spec.mask_bits));
    }
    if let Some(value) = passed.as_ref() {
        columns.push(named("passed", value.clone(), 0));
    }
    Ok(ResidentSinkOutput {
        columns,
        entries: Some(ResidentEntries {
            work_ids: ids,
            i,
            j,
            keys,
            passed,
            rank: None,
        }),
        shape: [work.n_a, work.n_b],
        tie_rows: 0,
    })
}
fn dense(
    lease: &DeviceLease,
    kernels: &Kernels,
    work: &ResidentWork,
    producer: &mut dyn ResidentProducer,
    config: &ResidentSinkConfig,
    observer: &mut Option<&mut ChunkObserver<'_>>,
) -> Result<ResidentSinkOutput, EngineError> {
    if config.cutoff.is_some() {
        return Err(parameter("Matrix and Tiles do not accept a cutoff"));
    }
    let [row0, col0, rows, cols] = config.tile.unwrap_or([0, 0, work.n_a, work.n_b]);
    if row0.checked_add(rows).is_none_or(|n| n > work.n_a)
        || col0.checked_add(cols).is_none_or(|n| n > work.n_b)
    {
        return Err(parameter("dense tile geometry exceeds pair-space shape"));
    }
    let n = rows
        .checked_mul(cols)
        .ok_or_else(|| parameter("dense result cell count overflows"))?;
    let dtype = producer.key_dtype();
    let specs = producer.fields().to_vec();
    let fields = allocate_fields(lease, &specs, &[rows, cols])?;
    let keys = lease.output(dtype, vec![rows, cols])?;
    let passed = lease.output(DeviceDType::Bool, vec![rows, cols])?;
    fill(
        lease,
        kernels,
        keys.address(),
        n,
        dtype.width(),
        mask(dtype)? as u64,
    )?;
    fill(lease, kernels, passed.address(), n, 1, 0)?;
    for (spec, field) in specs.iter().zip(&fields) {
        fill(
            lease,
            kernels,
            field.address(),
            n,
            spec.dtype.width(),
            spec.mask_bits,
        )?;
    }
    while let Some(chunk) = producer.next_chunk(lease)? {
        let len = validate_chunk(lease, &chunk, dtype)?;
        observe(observer, lease, &chunk, 0)?;
        let columns = columns(&specs, &chunk.fields, &fields)?;
        let (src, width, dst, pass, r0, c0, nr, nc, mirror) = (
            chunk.keys.address(),
            dtype.width() as u64,
            keys.address(),
            passed.address(),
            row0 as u64,
            col0 as u64,
            rows as u64,
            cols as u64,
            u64::from(work.mirror && config.mirror_symmetric),
        );
        let mut launch = lease.execution().launch_builder(&kernels.dense);
        launch
            .arg(&chunk.space)
            .arg(&src)
            .arg(&width)
            .arg(&columns)
            .arg(&dst)
            .arg(&pass)
            .arg(&r0)
            .arg(&c0)
            .arg(&nr)
            .arg(&nc)
            .arg(&mirror);
        // SAFETY: checked tile geometry bounds every scatter. Canonical work
        // is unique; diagonal mirrors are skipped, and holes remain initialized.
        unsafe { launch.launch(grid(len)) }
            .map_err(|e| fail(lease, "scattering resident dense cells", e))?;
    }
    let mut columns = Vec::with_capacity(specs.len() + 2);
    columns.push(named("__keys", publish(keys)?, mask(dtype)? as u64));
    for (spec, value) in specs.iter().zip(fields) {
        columns.push(named(&spec.name, publish(value)?, spec.mask_bits));
    }
    columns.push(named("passed", publish(passed)?, 0));
    Ok(ResidentSinkOutput {
        columns,
        entries: None,
        shape: [rows, cols],
        tie_rows: 0,
    })
}
fn rowstats(
    lease: &DeviceLease,
    kernels: &Kernels,
    work: &ResidentWork,
    producer: &mut Boundaries<'_, '_>,
    config: &ResidentSinkConfig,
    observer: &mut Option<&mut ChunkObserver<'_>>,
    halt: Option<&AtomicBool>,
) -> Result<ResidentSinkOutput, EngineError> {
    let n = work.n_a;
    let dtype = producer.key_dtype();
    let best = alloc::<i64>(lease, n)?;
    let counts = alloc::<u64>(lease, n)?;
    let ties = alloc::<u64>(lease, n)?;
    let arg = alloc::<u64>(lease, n)?;
    fill(
        lease,
        kernels,
        ptr(&best, lease.stream()),
        n,
        8,
        mask(dtype)? as u64,
    )?;
    fill(lease, kernels, ptr(&counts, lease.stream()), n, 8, 0)?;
    fill(lease, kernels, ptr(&ties, lease.stream()), n, 8, 0)?;
    fill(lease, kernels, ptr(&arg, lease.stream()), n, 8, u64::MAX)?;
    let (has, limit, mirror, b, c, t, a) = (
        u64::from(config.cutoff.is_some()),
        config.cutoff.unwrap_or(0),
        u64::from(work.mirror && config.mirror_symmetric),
        ptr(&best, lease.stream()),
        ptr(&counts, lease.stream()),
        ptr(&ties, lease.stream()),
        ptr(&arg, lease.stream()),
    );
    // An observer may stop a sweep without an error (the exact LSE rows need a
    // capacity replay). Reuse requires its explicit halt flag; an unknown
    // observer retains the ordinary two scoring sweeps.
    let can_reuse = observer.is_none() || halt.is_some();
    let mut reduce = |chunk: &ResidentChunk, sweep| -> Result<usize, EngineError> {
        let len = validate_chunk(lease, chunk, dtype)?;
        observe(observer, lease, chunk, sweep)?;
        let (keys, width) = (chunk.keys.address(), dtype.width() as u64);
        if sweep == 0 {
            let mut launch = lease.execution().launch_builder(&kernels.row_min);
            launch
                .arg(&chunk.space)
                .arg(&keys)
                .arg(&width)
                .arg(&has)
                .arg(&limit)
                .arg(&mirror)
                .arg(&b)
                .arg(&c);
            // SAFETY: one exact integer contribution per canonical distinct
            // partner, mirrored off-diagonal only. All row arrays cover n_a.
            // The keys and space are read-only in both reductions.
            unsafe { launch.launch(grid(len)) }
                .map_err(|e| fail(lease, "reducing resident row minima", e))?;
        } else {
            let mut launch = lease.execution().launch_builder(&kernels.row_arg);
            launch
                .arg(&chunk.space)
                .arg(&keys)
                .arg(&width)
                .arg(&has)
                .arg(&limit)
                .arg(&mirror)
                .arg(&b)
                .arg(&t)
                .arg(&a);
            // SAFETY: final minima are complete before replay, so equal-key
            // tie counts and smallest partner are order-independent integers.
            unsafe { launch.launch(grid(len)) }
                .map_err(|e| fail(lease, "reducing resident row ties", e))?;
        }
        Ok(len)
    };
    let mut single_chunk: Option<ResidentChunk> = None;
    for sweep in 0..2u32 {
        if sweep != 0 {
            producer.reset()?;
            if let Some(chunk) = single_chunk.take() {
                // As with next_chunk, count this chunk only after its sink work
                // is enqueued. Keep the immutable column owners alive through
                // row_arg and the boundary, then use their normal retirement.
                producer.pending = Some(chunk.space.n as usize);
                reduce(&chunk, sweep)?;
                producer.finish_chunk()?;
                break;
            }
        }
        let mut first = true;
        while let Some(chunk) = producer.next_chunk(lease)? {
            let len = reduce(&chunk, sweep)?;
            if sweep == 0 && len != 0 {
                single_chunk = if can_reuse
                    && first
                    && chunk.space.k0 == work.space.k0
                    && chunk.space.n == work.space.n
                {
                    Some(chunk)
                } else {
                    None
                };
                first = false;
            }
        }
        // Reaching None also flushed the last chunk's cancellation boundary.
        // Full actual coverage and normal exhaustion are both required: a halt
        // on the final (even sole) chunk is still an unsuccessful LSE sweep.
        if halt.is_some_and(|halt| halt.load(Ordering::Relaxed)) {
            single_chunk = None;
        }
    }
    let best_out = lease.output(dtype, vec![n])?;
    let ties_out = lease.output(DeviceDType::I32, vec![n])?;
    let arg_out = lease.output(DeviceDType::I64, vec![n])?;
    let count_out = if config.cutoff.is_some() {
        Some(lease.output(DeviceDType::I64, vec![n])?)
    } else {
        None
    };
    let error = alloc::<u64>(lease, 1)?;
    fill(lease, kernels, ptr(&error, lease.stream()), 1, 8, 0)?;
    let (n64, bo, width, to, ao, co, er) = (
        n as u64,
        best_out.address(),
        dtype.width() as u64,
        ties_out.address(),
        arg_out.address(),
        count_out.as_ref().map_or(0, DeviceOutput::address),
        ptr(&error, lease.stream()),
    );
    let mut launch = lease.execution().launch_builder(&kernels.row_finish);
    launch
        .arg(&b)
        .arg(&c)
        .arg(&t)
        .arg(&a)
        .arg(&n64)
        .arg(&bo)
        .arg(&width)
        .arg(&to)
        .arg(&ao)
        .arg(&co)
        .arg(&er);
    // SAFETY: fully initialized final row state, matching scalar output widths;
    // overflow is checked before any result is returned to Python.
    unsafe { launch.launch(grid(n)) }
        .map_err(|e| fail(lease, "finalizing resident row statistics", e))?;
    if control(lease, &error, 0)? != 0 {
        return Err(EngineError::Overflow(
            "resident row count or winning tie count exceeds its output dtype".into(),
        ));
    }
    let mut columns = vec![
        named("__best_key", publish(best_out)?, mask(dtype)? as u64),
        named("argbest", publish(arg_out)?, u64::MAX),
        named("n_best", publish(ties_out)?, 0),
    ];
    if let Some(count) = count_out {
        columns.push(named("count", publish(count)?, 0));
    }
    Ok(ResidentSinkOutput {
        columns,
        entries: None,
        shape: [work.n_a, work.n_b],
        tie_rows: 0,
    })
}
// Each row grows only after actual distinct passing partners arrive. The arena
// appends the new row workspace, so migration never aliases its old live runs.
// Input chunks and every radix scratch are bounded by the producer's chunk size.
struct TopKState {
    rows: crate::arena::Scratch<TopKRow>,
    seen: crate::arena::Scratch<u64>,
    pool: crate::arena::Scratch<i64>,
    tasks: crate::arena::Scratch<TopKTask>,
    jobs: crate::arena::Scratch<TopKJob>,
    job_count: crate::arena::Scratch<u64>,
    cursor: crate::arena::Scratch<u64>,
    status: crate::arena::Scratch<u64>,
    arena_used: crate::arena::Scratch<u64>,
    nrows: usize,
}
enum TopKPhase {
    ChunkLarge,
    ChunkLevel(u32),
    ChunkFlush,
    FinalLevel(u32),
    FinalFlush,
}
impl TopKState {
    fn new(lease: &DeviceLease, kernels: &Kernels, nrows: usize) -> Result<Self, EngineError> {
        let state = Self {
            rows: alloc(lease, nrows)?,
            seen: alloc(lease, nrows)?,
            pool: alloc(lease, 0)?,
            tasks: alloc(lease, nrows)?,
            jobs: alloc(lease, nrows)?,
            job_count: alloc(lease, 1)?,
            cursor: alloc(lease, 1)?,
            status: alloc(lease, 1)?,
            arena_used: alloc(lease, 1)?,
            nrows,
        };
        for counter in [
            &state.job_count,
            &state.cursor,
            &state.status,
            &state.arena_used,
        ] {
            fill(lease, kernels, ptr(counter, lease.stream()), 1, 8, 0)?;
        }
        let (rows, n, seen) = (
            ptr(&state.rows, lease.stream()),
            nrows as u64,
            ptr(&state.seen, lease.stream()),
        );
        let mut launch = lease.execution().launch_builder(&kernels.topk_init_empty);
        launch.arg(&rows).arg(&n).arg(&seen);
        // SAFETY: grid-stride initialization covers every row and seen counter.
        unsafe { launch.launch(grid(nrows)) }
            .map_err(|e| fail(lease, "initializing resident TopK rows", e))?;
        Ok(state)
    }
    fn persistent_grid(&self, lease: &DeviceLease) -> LaunchConfig {
        let blocks = lease
            .device()
            .0
            .resident_threads
            .div_ceil(THREADS as u64)
            .clamp(1, 65_535) as u32;
        LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    }
    fn queue_capacity(&self) -> Result<usize, EngineError> {
        self.nrows
            .checked_add(self.pool.len() / 1024)
            .map(|n| n.max(1))
            .ok_or_else(|| parameter("resident TopK job queue size overflows"))
    }
    fn resize_queue(&mut self, lease: &DeviceLease) -> Result<(), EngineError> {
        let capacity = self.queue_capacity()?;
        if self.jobs.len() < capacity {
            self.jobs = alloc(lease, capacity)?;
        }
        Ok(())
    }
    fn zero_jobs(&self, lease: &DeviceLease, kernels: &Kernels) -> Result<(), EngineError> {
        fill(
            lease,
            kernels,
            ptr(&self.job_count, lease.stream()),
            1,
            8,
            0,
        )
    }
    fn consume_jobs(
        &self,
        lease: &DeviceLease,
        kernels: &Kernels,
        entries: u64,
    ) -> Result<(), EngineError> {
        let (pool, tasks, jobs, count, cursor, status) = (
            ptr(&self.pool, lease.stream()),
            ptr(&self.tasks, lease.stream()),
            ptr(&self.jobs, lease.stream()),
            ptr(&self.job_count, lease.stream()),
            ptr(&self.cursor, lease.stream()),
            ptr(&self.status, lease.stream()),
        );
        fill(lease, kernels, cursor, 1, 8, 0)?;
        let mut launch = lease.execution().launch_builder(&kernels.topk_merge_jobs);
        launch
            .arg(&entries)
            .arg(&pool)
            .arg(&tasks)
            .arg(&jobs)
            .arg(&count)
            .arg(&cursor)
            .arg(&status);
        // SAFETY: preparation bounds every task/run/job; one persistent CTA
        // owns each queue ticket and writes its disjoint scratch tile.
        unsafe { launch.launch(self.persistent_grid(lease)) }
            .map_err(|e| fail(lease, "merging resident TopK runs", e))?;
        fill(lease, kernels, cursor, 1, 8, 0)?;
        let mut launch = lease.execution().launch_builder(&kernels.topk_commit_jobs);
        launch
            .arg(&pool)
            .arg(&tasks)
            .arg(&jobs)
            .arg(&count)
            .arg(&cursor)
            .arg(&status);
        // SAFETY: all scratch writes finish before this stream-ordered phase;
        // each destination tile is disjoint and its scratch prefix initialized.
        unsafe { launch.launch(self.persistent_grid(lease)) }
            .map_err(|e| fail(lease, "committing resident TopK runs", e))?;
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn grow(
        &mut self,
        lease: &DeviceLease,
        kernels: &Kernels,
        segments: &CudaSlice<TopKSegment>,
        count: &CudaSlice<u64>,
        capacity: usize,
        k: usize,
    ) -> Result<(), EngineError> {
        self.resize_queue(lease)?;
        self.zero_jobs(lease, kernels)?;
        let growth = alloc::<TopKGrowth>(lease, capacity)?;
        let (
            rows,
            nrows,
            old_pool,
            segment,
            count,
            cap,
            k,
            seen,
            arena,
            plans,
            growth_cap,
            jobs,
            job_cap,
            job_count,
            status,
        ) = (
            ptr(&self.rows, lease.stream()),
            self.nrows as u64,
            self.pool.len() as u64,
            ptr(segments, lease.stream()),
            ptr(count, lease.stream()),
            capacity as u64,
            k as u64,
            ptr(&self.seen, lease.stream()),
            ptr(&self.arena_used, lease.stream()),
            ptr(&growth, lease.stream()),
            capacity as u64,
            ptr(&self.jobs, lease.stream()),
            self.jobs.len() as u64,
            ptr(&self.job_count, lease.stream()),
            ptr(&self.status, lease.stream()),
        );
        let mut launch = lease
            .execution()
            .launch_builder(&kernels.topk_growth_prepare);
        launch
            .arg(&rows)
            .arg(&nrows)
            .arg(&old_pool)
            .arg(&segment)
            .arg(&count)
            .arg(&cap)
            .arg(&k)
            .arg(&seen)
            .arg(&arena)
            .arg(&plans)
            .arg(&growth_cap)
            .arg(&jobs)
            .arg(&job_cap)
            .arg(&job_count)
            .arg(&status);
        // SAFETY: at most one segment owns a row, with initialized old rows;
        // checked append reservations and a bounded queue precede every write.
        unsafe { launch.launch(exact_grid(capacity)?) }
            .map_err(|e| fail(lease, "planning resident TopK growth", e))?;
        // Only two scalar controls return to host, never keys, pair arrays or
        // row capacities. Read both before the single stream synchronization.
        let mut required = [0u64];
        let mut error = [0u64];
        lease
            .execution()
            .memcpy_dtoh(&self.arena_used.slice(0..1), &mut required[..])
            .map_err(|e| fail(lease, "reading resident TopK arena size", e))?;
        lease
            .execution()
            .memcpy_dtoh(&self.status.slice(0..1), &mut error[..])
            .map_err(|e| fail(lease, "reading resident TopK status", e))?;
        lease.synchronize()?;
        if error[0] != 0 {
            return Err(parameter(
                "resident TopK state or workspace capacity is invalid",
            ));
        }
        let required = usize::try_from(required[0])
            .map_err(|_| parameter("resident TopK arena exceeds platform limits"))?;
        if required > self.pool.len() {
            let doubled = self
                .pool
                .len()
                .checked_mul(2)
                .filter(|&n| n <= isize::MAX as usize / size_of::<i64>())
                .unwrap_or(required);
            let grown = alloc::<i64>(lease, required.max(doubled))?;
            let bytes = self
                .pool
                .len()
                .checked_mul(size_of::<i64>())
                .ok_or_else(|| parameter("resident TopK copy size overflows"))?;
            copy(
                lease,
                ptr(&grown, lease.stream()),
                ptr(&self.pool, lease.stream()),
                bytes,
            )?;
            self.pool = grown;
        }
        let (pool, cursor, pool_elements) = (
            ptr(&self.pool, lease.stream()),
            ptr(&self.cursor, lease.stream()),
            self.pool.len() as u64,
        );
        fill(lease, kernels, cursor, 1, 8, 0)?;
        let mut launch = lease
            .execution()
            .launch_builder(&kernels.topk_growth_copy_jobs);
        launch
            .arg(&pool)
            .arg(&plans)
            .arg(&jobs)
            .arg(&job_count)
            .arg(&cursor)
            .arg(&status);
        // SAFETY: old offsets remain valid after the preserving D2D copy;
        // only initialized old base/occupied runs migrate to appended regions.
        unsafe { launch.launch(self.persistent_grid(lease)) }
            .map_err(|e| fail(lease, "migrating resident TopK rows", e))?;
        let mut launch = lease
            .execution()
            .launch_builder(&kernels.topk_growth_commit);
        launch
            .arg(&rows)
            .arg(&nrows)
            .arg(&pool_elements)
            .arg(&segment)
            .arg(&count)
            .arg(&cap)
            .arg(&plans)
            .arg(&growth_cap)
            .arg(&status);
        // SAFETY: migration completes on this stream before row offsets change.
        unsafe { launch.launch(exact_grid(capacity)?) }
            .map_err(|e| fail(lease, "publishing resident TopK row growth", e))?;
        // The growth queue stays live through migration. Only now may its
        // allocation be replaced with the larger merge-phase queue.
        self.resize_queue(lease)?;
        Ok(())
    }
    fn phase(
        &self,
        lease: &DeviceLease,
        kernels: &Kernels,
        segments: Option<(&CudaSlice<TopKSegment>, &CudaSlice<u64>, usize)>,
        entries: u64,
        phase: TopKPhase,
    ) -> Result<(), EngineError> {
        self.zero_jobs(lease, kernels)?;
        let (function, level) = match phase {
            TopKPhase::ChunkLarge => (&kernels.topk_chunk_large, None),
            TopKPhase::ChunkLevel(l) => (&kernels.topk_chunk_level, Some(l)),
            TopKPhase::ChunkFlush => (&kernels.topk_chunk_flush, None),
            TopKPhase::FinalLevel(l) => (&kernels.topk_final_level, Some(l)),
            TopKPhase::FinalFlush => (&kernels.topk_final_flush, None),
        };
        let (rows, nrows, pool, tasks, task_cap, jobs, job_cap, count, status) = (
            ptr(&self.rows, lease.stream()),
            self.nrows as u64,
            self.pool.len() as u64,
            ptr(&self.tasks, lease.stream()),
            self.tasks.len() as u64,
            ptr(&self.jobs, lease.stream()),
            self.jobs.len() as u64,
            ptr(&self.job_count, lease.stream()),
            ptr(&self.status, lease.stream()),
        );
        let (segment, segment_count, segment_cap) = segments.map_or((0, 0, 0), |(s, c, n)| {
            (ptr(s, lease.stream()), ptr(c, lease.stream()), n as u64)
        });
        let mut launch = lease.execution().launch_builder(function);
        if let Some(ref level) = level {
            launch.arg(level);
        }
        launch.arg(&rows).arg(&nrows).arg(&pool);
        if segments.is_some() {
            launch.arg(&segment).arg(&segment_count).arg(&segment_cap);
        }
        launch
            .arg(&tasks)
            .arg(&task_cap)
            .arg(&jobs)
            .arg(&job_cap)
            .arg(&count)
            .arg(&status);
        // SAFETY: preparation reads initialized row state and bounded segment
        // geometry; task/queue capacities are checked in every enqueue.
        unsafe { launch.launch(exact_grid(segments.map_or(self.nrows, |(_, _, n)| n))?) }
            .map_err(|e| fail(lease, "preparing resident TopK merge phase", e))?;
        self.consume_jobs(lease, kernels, entries)
    }
}
fn topk_thresholds(
    lease: &DeviceLease,
    kernels: &Kernels,
    work: &ResidentWork,
    producer: &mut dyn ResidentProducer,
    config: &ResidentSinkConfig,
    observer: &mut Option<&mut ChunkObserver<'_>>,
) -> Result<Thresholds, EngineError> {
    let dtype = producer.key_dtype();
    let mut state = TopKState::new(lease, kernels, work.n_a)?;
    // Capacities may round up while below k. ceil(log2(min(k,n_b)))
    // still covers every possible occupied binary forest level.
    let max_capacity = config.k.min(work.n_b);
    let levels = if max_capacity == 0 {
        0
    } else {
        usize::BITS - (max_capacity - 1).leading_zeros()
    };
    while let Some(chunk) = producer.next_chunk(lease)? {
        validate_chunk(lease, &chunk, dtype)?;
        observe(observer, lease, &chunk, 0)?;
        let records = build(
            lease,
            kernels,
            &chunk,
            config.cutoff,
            work.mirror && config.mirror_symmetric,
            false,
            0,
            0,
            0,
            producer.passed(),
        )?;
        let records = sort(lease, kernels, records, work, dtype, false)?;
        if records.n == 0 {
            continue;
        }
        let capacity = records.n.min(work.n_a);
        let segments = alloc::<TopKSegment>(lease, capacity)?;
        let count = alloc::<u64>(lease, 1)?;
        fill(lease, kernels, ptr(&count, lease.stream()), 1, 8, 0)?;
        let (entries, n, nrows, segment, cap, segment_count, status) = (
            ptr(&records.values, lease.stream()),
            records.n as u64,
            work.n_a as u64,
            ptr(&segments, lease.stream()),
            capacity as u64,
            ptr(&count, lease.stream()),
            ptr(&state.status, lease.stream()),
        );
        let mut launch = lease.execution().launch_builder(&kernels.topk_segments);
        launch
            .arg(&entries)
            .arg(&n)
            .arg(&nrows)
            .arg(&segment)
            .arg(&cap)
            .arg(&segment_count)
            .arg(&status);
        // SAFETY: sorted records provide one disjoint segment per touched row.
        unsafe { launch.launch(exact_grid(records.n)?) }
            .map_err(|e| fail(lease, "segmenting resident TopK rows", e))?;
        state.grow(lease, kernels, &segments, &count, capacity, config.k)?;
        let args = Some((&*segments, &*count, capacity));
        state.phase(lease, kernels, args, entries, TopKPhase::ChunkLarge)?;
        for level in 0..levels {
            state.phase(lease, kernels, args, entries, TopKPhase::ChunkLevel(level))?;
        }
        state.phase(lease, kernels, args, entries, TopKPhase::ChunkFlush)?;
    }
    let (rows, nrows) = (ptr(&state.rows, lease.stream()), work.n_a as u64);
    let mut launch = lease.execution().launch_builder(&kernels.topk_final_begin);
    launch.arg(&rows).arg(&nrows);
    // SAFETY: initializes only the live row state on the same consumer stream.
    unsafe { launch.launch(exact_grid(work.n_a)?) }
        .map_err(|e| fail(lease, "finalizing resident TopK forest", e))?;
    for level in 0..levels {
        state.phase(lease, kernels, None, 0, TopKPhase::FinalLevel(level))?;
    }
    state.phase(lease, kernels, None, 0, TopKPhase::FinalFlush)?;
    let output = Thresholds {
        values: alloc(lease, work.n_a)?,
        active: alloc(lease, work.n_a)?,
    };
    let (pool, pool_elements, threshold, active, status) = (
        ptr(&state.pool, lease.stream()),
        state.pool.len() as u64,
        ptr(&output.values, lease.stream()),
        ptr(&output.active, lease.stream()),
        ptr(&state.status, lease.stream()),
    );
    let mut launch = lease.execution().launch_builder(&kernels.topk_thresholds);
    launch
        .arg(&rows)
        .arg(&nrows)
        .arg(&pool)
        .arg(&pool_elements)
        .arg(&threshold)
        .arg(&active)
        .arg(&status);
    // SAFETY: final runs and their actual counts are initialized; absence is
    // carried by active, never inferred from a signed key sentinel.
    unsafe { launch.launch(exact_grid(work.n_a)?) }
        .map_err(|e| fail(lease, "selecting resident TopK thresholds", e))?;
    if control(lease, &state.status, 0)? != 0 {
        return Err(parameter(
            "resident TopK state or workspace capacity is invalid",
        ));
    }
    Ok(output)
}

struct Kept {
    records: Records,
    fields: Vec<ResidentField>,
}
struct Thresholds {
    values: crate::arena::Scratch<i64>,
    active: crate::arena::Scratch<u8>,
}
// Hits are already canonical and sorted by the producer. Join their numeric
// columns and unpack coordinates directly into final arrays: an additional
// result-sized Entry table or field permutation would be redundant.
fn hits_output(
    lease: &DeviceLease,
    kernels: &Kernels,
    work: &ResidentWork,
    dtype: DeviceDType,
    specs: &[ResidentFieldSpec],
    kept: Vec<Kept>,
    total: usize,
) -> Result<ResidentSinkOutput, EngineError> {
    let ids = lease.output(DeviceDType::U64, vec![total])?;
    let i = lease.output(DeviceDType::I64, vec![total])?;
    let j = lease.output(DeviceDType::I64, vec![total])?;
    let keys = lease.output(dtype, vec![total])?;
    let fields = allocate_fields(lease, specs, &[total])?;
    let mut start = 0usize;
    for chunk in kept {
        unpack(
            lease,
            kernels,
            &chunk.records,
            start,
            [ids.address(), i.address(), j.address(), keys.address(), 0],
            dtype,
            None,
        )?;
        for (at, spec) in specs.iter().enumerate() {
            copy(
                lease,
                fields[at].address() + (start * spec.dtype.width()) as u64,
                chunk.fields[at].values.address(),
                chunk.records.n * spec.dtype.width(),
            )?;
        }
        start += chunk.records.n;
    }
    let (ids, i, j, keys) = (publish(ids)?, publish(i)?, publish(j)?, publish(keys)?);
    let mut columns = vec![
        named("i", i.clone(), 0),
        named("j", j.clone(), 0),
        named("__keys", keys.clone(), mask(dtype)? as u64),
    ];
    for (spec, field) in specs.iter().zip(fields) {
        columns.push(named(&spec.name, publish(field)?, spec.mask_bits));
    }
    Ok(ResidentSinkOutput {
        columns,
        entries: Some(ResidentEntries {
            work_ids: ids,
            i,
            j,
            keys,
            passed: None,
            rank: None,
        }),
        shape: [work.n_a, work.n_b],
        tie_rows: 0,
    })
}

fn sparse(
    lease: &DeviceLease,
    kernels: &Kernels,
    work: &ResidentWork,
    producer: &mut dyn ResidentProducer,
    config: &ResidentSinkConfig,
    thresholds: Option<&Thresholds>,
    observer: &mut Option<&mut ChunkObserver<'_>>,
) -> Result<ResidentSinkOutput, EngineError> {
    let dtype = producer.key_dtype();
    let specs = producer.fields().to_vec();
    let topk = thresholds.is_some();
    let (threshold, active) = thresholds.map_or((0, 0), |v| {
        (
            ptr(&v.values, lease.stream()),
            ptr(&v.active, lease.stream()),
        )
    });
    let mut kept = Vec::<Kept>::new();
    let mut total = 0usize;
    while let Some(chunk) = producer.next_chunk(lease)? {
        validate_chunk(lease, &chunk, dtype)?;
        observe(observer, lease, &chunk, u32::from(topk))?;
        let records = build(
            lease,
            kernels,
            &chunk,
            config.cutoff,
            topk && work.mirror && config.mirror_symmetric,
            false,
            threshold,
            active,
            total,
            producer.passed(),
        )?;
        let next = total
            .checked_add(records.n)
            .filter(|&n| n <= i64::MAX as usize)
            .ok_or_else(|| {
                EngineError::Overflow("resident sparse output offsets exceed int64".into())
            })?;
        if records.n != 0 {
            let outputs = allocate_fields(lease, &specs, &[records.n])?;
            gather_fields(
                lease,
                kernels,
                &records,
                &specs,
                &chunk.fields,
                &outputs,
                true,
                chunk.space.k0,
                0,
                None,
            )?;
            let mut fields = Vec::with_capacity(specs.len());
            for (spec, output) in specs.iter().zip(outputs) {
                fields.push(named(&spec.name, publish(output)?, spec.mask_bits));
            }
            seqblaze_core::detail::try_reserve(&mut kept, 1, "resident sparse chunks")?;
            kept.push(Kept { records, fields });
        }
        total = next;
    }
    if !topk {
        return hits_output(lease, kernels, work, dtype, &specs, kept, total);
    }
    let entries = alloc::<Entry>(lease, total)?;
    let concatenated = allocate_fields(lease, &specs, &[total])?;
    let mut start = 0usize;
    for chunk in kept {
        copy(
            lease,
            ptr(&entries, lease.stream()) + (start * size_of::<Entry>()) as u64,
            ptr(&chunk.records.values, lease.stream()),
            chunk.records.n * size_of::<Entry>(),
        )?;
        for (at, spec) in specs.iter().enumerate() {
            copy(
                lease,
                concatenated[at].address() + (start * spec.dtype.width()) as u64,
                chunk.fields[at].values.address(),
                chunk.records.n * spec.dtype.width(),
            )?;
        }
        start += chunk.records.n;
    }
    let mut sources = Vec::with_capacity(specs.len());
    for (spec, output) in specs.iter().zip(concatenated) {
        sources.push(named(&spec.name, publish(output)?, spec.mask_bits));
    }
    let records = Records {
        values: entries,
        n: total,
    };
    let records = if topk {
        sort(lease, kernels, records, work, dtype, true)?
    } else {
        records
    };
    let outputs = allocate_fields(lease, &specs, &[total])?;
    gather_fields(
        lease, kernels, &records, &specs, &sources, &outputs, false, 0, 0, None,
    )?;
    let ids = lease.output(DeviceDType::U64, vec![total])?;
    let i = lease.output(DeviceDType::I64, vec![total])?;
    let j = lease.output(DeviceDType::I64, vec![total])?;
    let keys = lease.output(dtype, vec![total])?;
    unpack(
        lease,
        kernels,
        &records,
        0,
        [ids.address(), i.address(), j.address(), keys.address(), 0],
        dtype,
        None,
    )?;
    let mut rank = None;
    let mut row_columns = Vec::new();
    let mut tie_rows = 0;
    if topk {
        let ranks = lease.output(DeviceDType::I32, vec![total])?;
        let offsets = lease.output(
            DeviceDType::I64,
            vec![work
                .n_a
                .checked_add(1)
                .ok_or_else(|| parameter("TopK offsets length overflows"))?],
        )?;
        let tied = lease.output(DeviceDType::Bool, vec![work.n_a])?;
        let count = alloc::<u64>(lease, 1)?;
        let error = alloc::<u64>(lease, 1)?;
        fill(lease, kernels, ptr(&count, lease.stream()), 1, 8, 0)?;
        fill(lease, kernels, ptr(&error, lease.stream()), 1, 8, 0)?;
        let (ent, n, rows, k, r, o, t, c, e) = (
            ptr(&records.values, lease.stream()),
            total as u64,
            work.n_a as u64,
            config.k as u64,
            ranks.address(),
            offsets.address(),
            tied.address(),
            ptr(&count, lease.stream()),
            ptr(&error, lease.stream()),
        );
        let mut launch = lease.execution().launch_builder(&kernels.topk_finish);
        launch
            .arg(&ent)
            .arg(&n)
            .arg(&rows)
            .arg(&k)
            .arg(&r)
            .arg(&o)
            .arg(&t)
            .arg(&c)
            .arg(&e);
        // SAFETY: final order is (row,key,partner); binary searches name exact
        // competition groups, and offsets retain every empty row.
        unsafe { launch.launch(grid(total.max(work.n_a.saturating_add(1)))) }
            .map_err(|e| fail(lease, "finalizing resident TopK ranks", e))?;
        if control(lease, &error, 0)? != 0 {
            return Err(EngineError::Overflow(
                "resident TopK competition rank exceeds int32".into(),
            ));
        }
        tie_rows = control(lease, &count, 0)?;
        let value = publish(ranks)?;
        rank = Some(value.clone());
        row_columns = vec![
            named("rank", value, 0),
            named("offsets", publish(offsets)?, 0),
            named("tie_at_k", publish(tied)?, 0),
        ];
    }
    let (ids, i, j, keys) = (publish(ids)?, publish(i)?, publish(j)?, publish(keys)?);
    let mut columns = vec![
        named("i", i.clone(), 0),
        named("j", j.clone(), 0),
        named("__keys", keys.clone(), mask(dtype)? as u64),
    ];
    for (spec, value) in specs.iter().zip(outputs) {
        columns.push(named(&spec.name, publish(value)?, spec.mask_bits));
    }
    columns.extend(row_columns);
    Ok(ResidentSinkOutput {
        columns,
        entries: Some(ResidentEntries {
            work_ids: ids,
            i,
            j,
            keys,
            passed: None,
            rank,
        }),
        shape: [work.n_a, work.n_b],
        tie_rows,
    })
}
