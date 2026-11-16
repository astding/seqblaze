//! The unary data path (docs/wiki/gpu/tm-cuda.md §5): upload the validated
//! table once, chunk over sequence indices, run the kernel's stage, and move
//! whole columns back. Each copies the requested columns (and `passed`)
//! device-to-host straight into the host output; Hits compacts passing rows on
//! the device (count, host scan, ordered write), so `i` is born ascending and
//! unique. There is no per-row host loop.
//!
//! Generic part: the table upload, the chunk planner (`plan::unary_chunk_end`),
//! the column copy-out, the Hits kernels with their column descriptors, and
//! core's `UnaryOutput`. Per kernel: the stage kernel, its pack and its masks
//! (device tm: `sb_tm`, kernels/tm.cuh; device hairpin with the primer3 preset:
//! the group kernels of kernels/primer3.cuh, launched by `p3.rs`, with the
//! depth bound of primer3-cuda.md §6.6 on each chunk; the tm compatibility
//! stages sb_tm_biopython and sb_tm_primer3 in kernels/tm_compat.cuh).
//! Testing builds refill tm and hairpin stage outputs (including `passed`)
//! before every chunk when poisoned; T2 output counters exclude allocations
//! and DP scratch.
//!
//! Also the testing kernel `sb_f64_eval` of the soft-binary64 routine (P5).
use std::sync::Arc;

use cudarc::driver::{
    sys, CudaSlice, CudaStream, DevicePtr, DeviceRepr, DriverError, LaunchConfig, PushKernelArg,
};
use seqblaze_core::{
    detail::{allocation_error, try_reserve, DetailColumns, DetailData},
    seq::SeqTable,
    sink::{Counts, Sink},
    unary::{UnaryOptions, UnaryOutput},
    validate::{EngineError, ValidationReport},
};

use crate::{
    device::{guard, Device},
    launch::{self, Settings},
    p3::{launch_buckets, P3Args, P3Buffers, P3Pack},
    plan::{p3_chunk_budget, p3_plan_t2, p3_slot_t2, unary_chunk_end, P3Bucket, MAX_GRID},
    session::{dev_alloc, failure, ptr, upload_unary_table},
    F64Op, TmBiopythonPack, TmPack, TmPrimer3Pack, UnaryCall, TM_INT32_SLOTS, TM_SLOTS,
};

/// Output slots of device hairpin, in the order of its fields (hairpin-primer3.md
/// §6): `found` (int32), `dg_kcal`, `dh_kcal`, `ds_cal_per_k`, `tm_c`.
const HAIRPIN_SLOTS: usize = 5;
const HAIRPIN_INT32_SLOTS: [bool; HAIRPIN_SLOTS] = [true, false, false, false, false];

/// Column descriptors per Hits write; must match SB_UNARY_MAX_COLS.
const MAX_COLS: usize = 8;

// #[repr(C)] mirrors of kernels/unary.cuh and kernels/tm.cuh; all fields are u64.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct SbUnaryTable {
    data: u64,
    off: u64,
    view: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct SbUnaryCols {
    count: u64,
    src: [u64; MAX_COLS],
    dst: [u64; MAX_COLS],
    width: [u64; MAX_COLS],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct SbTmOut {
    slots: [u64; TM_SLOTS],
    passed: u64,
}

// SAFETY: plain #[repr(C)] structs of u64/i64 fields with the same field order
// and widths as their CUDA counterparts; passed by value as kernel parameters.
unsafe impl DeviceRepr for SbUnaryTable {}
const _: () = {
    assert!(std::mem::size_of::<SbUnaryTable>() == 24);
    assert!(std::mem::offset_of!(SbUnaryTable, view) == 16);
};
unsafe impl DeviceRepr for SbUnaryCols {}
unsafe impl DeviceRepr for SbTmOut {}
unsafe impl DeviceRepr for TmPack {}
unsafe impl DeviceRepr for TmBiopythonPack {}
unsafe impl DeviceRepr for TmPrimer3Pack {}

const _: () = assert!(std::mem::size_of::<TmPack>() == 480);

/// A device column: float64 values as their bits, or int32.
enum DevColumn {
    F64(CudaSlice<f64>),
    I32(CudaSlice<i32>),
}

impl DevColumn {
    fn alloc(stream: &Arc<CudaStream>, int32: bool, len: usize) -> Result<Self, DriverError> {
        // SAFETY: every element a kernel or copy reads was written by a kernel first.
        Ok(if int32 {
            Self::I32(unsafe { dev_alloc::<i32>(stream, len.max(1)) }?)
        } else {
            Self::F64(unsafe { dev_alloc::<f64>(stream, len.max(1)) }?)
        })
    }

    fn len(&self) -> usize {
        match self {
            Self::F64(slice) => slice.len(),
            Self::I32(slice) => slice.len(),
        }
    }

    fn ptr(&self, stream: &CudaStream) -> u64 {
        match self {
            Self::F64(slice) => ptr(slice, stream),
            Self::I32(slice) => ptr(slice, stream),
        }
    }

    fn width(&self) -> u64 {
        match self {
            Self::F64(_) => 8,
            Self::I32(_) => 4,
        }
    }
}

/// The host column of one requested field.
enum HostColumn {
    F64(Vec<f64>),
    I32(Vec<i32>),
}

impl HostColumn {
    /// `rows` zeroed rows (Each: every row is overwritten by a copy).
    fn zeroed(int32: bool, rows: usize, what: &str) -> Result<Self, EngineError> {
        Ok(if int32 {
            Self::I32(zeroed(rows, what)?)
        } else {
            Self::F64(zeroed(rows, what)?)
        })
    }
}

/// A Vec of `rows` all-zero elements from one fallible zeroed allocation.
///
/// Large requests come from fresh, already-zero OS pages, so there is no
/// explicit fill pass over the output before the device-to-host copy
/// overwrites it (Vec::resize would write every row twice).
fn zeroed<T: Zeroable>(rows: usize, what: &str) -> Result<Vec<T>, EngineError> {
    if rows == 0 || std::mem::size_of::<T>() == 0 {
        return Ok(Vec::new());
    }
    let layout =
        std::alloc::Layout::array::<T>(rows).map_err(|error| allocation_error(what, error))?;
    // SAFETY: `layout` has a non-zero size (rows > 0, T not zero-sized).
    let raw = unsafe { std::alloc::alloc_zeroed(layout) }.cast::<T>();
    if raw.is_null() {
        return Err(allocation_error(
            what,
            format!("allocation of {} bytes failed", layout.size()),
        ));
    }
    // SAFETY: `raw` was allocated by the global allocator with the layout of
    // `[T; rows]`, so capacity `rows` matches; all-zero bytes are a valid `T`
    // (Zeroable), so all `rows` elements are initialized.
    Ok(unsafe { Vec::from_raw_parts(raw, rows, rows) })
}

/// Element types whose all-zero bit pattern is a valid value.
///
/// # Safety
/// Implement only for types for which every all-zero byte pattern is valid.
unsafe trait Zeroable: DeviceRepr {}
// SAFETY: zero bytes are 0.0, 0, 0 and 0 respectively.
unsafe impl Zeroable for f64 {}
unsafe impl Zeroable for i32 {}
unsafe impl Zeroable for u8 {}
unsafe impl Zeroable for i64 {}

/// One requested column: its kernel field, stage buffer, compacted Hits
/// buffer and host values.
struct Column {
    field: usize,
    stage: DevColumn,
    hits: Option<DevColumn>,
    host: HostColumn,
}

struct Unary<'d> {
    device: &'d Device,
    stream: Arc<CudaStream>,
    settings: Settings,
    /// Label of the call's output columns in reservation errors.
    output: &'static str,
}

impl Unary<'_> {
    fn fail(&self, what: &str, error: DriverError) -> EngineError {
        failure(&format!("cuda:{}", self.device.info().ordinal), what, error)
    }

    fn sync(&self) -> Result<(), EngineError> {
        self.stream
            .synchronize()
            .map_err(|error| self.fail("synchronization", error))
    }

    /// Testing builds: cuMemsetD8Async of `bytes` bytes at `address` to `byte`, on
    /// this call's stream (ordered after the previous chunk's copies).
    #[cfg(feature = "testing")]
    fn fill(&self, address: u64, byte: u8, bytes: usize) -> Result<(), EngineError> {
        if bytes == 0 {
            return Ok(());
        }
        self.stream
            .context()
            .bind_to_thread()
            .map_err(|error| self.fail("binding the context", error))?;
        // SAFETY: `address` points at a live stage buffer of at least `bytes` bytes
        // owned by this call; the memset is ordered on this stream.
        unsafe { sys::cuMemsetD8Async(address, byte, bytes, self.stream.cu_stream()) }
            .result()
            .map_err(|error| self.fail("memset", error))
    }

    /// A grid-stride launch over `items` (capped at `MAX_GRID` blocks).
    fn grid(&self, items: u64) -> LaunchConfig {
        let threads = u64::from(self.settings.threads_per_block);
        LaunchConfig {
            grid_dim: (items.div_ceil(threads).clamp(1, MAX_GRID) as u32, 1, 1),
            block_dim: (self.settings.threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// One block per tile of `threads_per_block` items (Hits).
    fn tiles(&self, items: usize) -> (usize, LaunchConfig) {
        let blocks = items
            .div_ceil(self.settings.threads_per_block as usize)
            .max(1);
        let config = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (self.settings.threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };
        (blocks, config)
    }

    fn download<T: DeviceRepr>(
        &self,
        source: &CudaSlice<T>,
        target: &mut [T],
    ) -> Result<(), EngineError> {
        if target.is_empty() {
            return Ok(());
        }
        crate::session::dtoh(&self.stream, &source.slice(0..target.len()), target)
            .map_err(|error| self.fail("device-to-host copy", error))
    }

    /// Copy the first `rows` elements of `source` into `host[base..base + rows]`.
    fn download_column(
        &self,
        source: &DevColumn,
        host: &mut HostColumn,
        base: usize,
        rows: usize,
    ) -> Result<(), EngineError> {
        match (source, host) {
            (DevColumn::F64(slice), HostColumn::F64(values)) => {
                self.download(slice, &mut values[base..base + rows])
            }
            (DevColumn::I32(slice), HostColumn::I32(values)) => {
                self.download(slice, &mut values[base..base + rows])
            }
            _ => unreachable!("a column's device and host kinds agree"),
        }
    }

    /// Append the first `rows` elements of `source` to `target`, copying
    /// straight into its spare capacity (no fill pass before the copy).
    fn append<T: DeviceRepr>(
        &self,
        source: &CudaSlice<T>,
        target: &mut Vec<T>,
        rows: usize,
        what: &str,
    ) -> Result<(), EngineError> {
        if rows == 0 {
            return Ok(());
        }
        assert!(rows <= source.len(), "append reads within the source");
        try_reserve(target, rows, what)?;
        self.stream
            .context()
            .bind_to_thread()
            .map_err(|error| self.fail("binding the context", error))?;
        let dst = target
            .spare_capacity_mut()
            .as_mut_ptr()
            .cast::<std::ffi::c_void>();
        {
            crate::session::check_stream(source.stream(), &self.stream);
            let (src, _record) = source.device_ptr(&self.stream);
            // SAFETY: `dst` has room for `rows` elements (reserved above) and
            // `src` holds at least `rows`; the copy is ordered on this stream.
            unsafe {
                sys::cuMemcpyDtoHAsync_v2(
                    dst,
                    src,
                    rows * std::mem::size_of::<T>(),
                    self.stream.cu_stream(),
                )
            }
            .result()
            .map_err(|error| self.fail("device-to-host copy", error))?;
        }
        self.sync()?;
        // SAFETY: the synchronized copy initialized `rows` elements past the
        // old length, within the reserved capacity.
        unsafe { target.set_len(target.len() + rows) };
        Ok(())
    }

    /// `append` for a column of either kind.
    fn append_column(
        &self,
        source: &DevColumn,
        host: &mut HostColumn,
        rows: usize,
    ) -> Result<(), EngineError> {
        match (source, host) {
            (DevColumn::F64(slice), HostColumn::F64(values)) => {
                self.append(slice, values, rows, self.output)
            }
            (DevColumn::I32(slice), HostColumn::I32(values)) => {
                self.append(slice, values, rows, self.output)
            }
            _ => unreachable!("a column's device and host kinds agree"),
        }
    }
}

/// The requested columns of `template` with their device slots (`int32`: the
/// kind of each slot of the call's kernel).
fn unary_columns(
    template: &DetailColumns,
    int32_slots: &[bool],
    name: &str,
) -> Result<Vec<(usize, bool)>, EngineError> {
    let mut result = Vec::new();
    for column in template.columns() {
        let int32 = matches!(column.data, DetailData::Int32(_));
        let numeric = int32 || matches!(column.data, DetailData::Float64(_));
        if column.field >= int32_slots.len() || !numeric || int32_slots[column.field] != int32 {
            return Err(EngineError::Parameter(format!(
                "internal error: device {name} has no column {:?} of this kind at slot {}",
                column.name, column.field
            )));
        }
        result.push((column.field, int32));
    }
    Ok(result)
}

pub(crate) fn run(
    device: &Device,
    call: &UnaryCall,
    table: &SeqTable,
    template: DetailColumns,
    options: &UnaryOptions<f64>,
    validation: ValidationReport,
) -> Result<UnaryOutput, EngineError> {
    guard(|| {
        let (int32_slots, name, output): (&[bool], &str, &'static str) = match call {
            UnaryCall::Tm(_) | UnaryCall::TmBiopython(_) | UnaryCall::TmPrimer3(_) => {
                (&TM_INT32_SLOTS, "tm", "tm output column")
            }
            UnaryCall::Hairpin(_) => (&HAIRPIN_INT32_SLOTS, "hairpin", "hairpin output column"),
        };
        let each = match options.sink {
            Sink::Each => true,
            Sink::Hits if options.cutoff.is_some() => false,
            _ => {
                return Err(EngineError::Parameter(
                    "a unary device call takes Each, or Hits with a cutoff".into(),
                ))
            }
        };
        let label = format!("cuda:{}", device.info().ordinal);
        let fail = |what: &str, error: DriverError| failure(&label, what, error);
        let stream = device
            .0
            .ctx
            .new_stream()
            .map_err(|error| fail("creating a stream", error))?;
        let mut settings = launch::settings();
        let cap = match call {
            UnaryCall::Tm(_) => Some(device.0.unary_block[0]),
            UnaryCall::TmBiopython(_) => Some(device.0.unary_block[1]),
            UnaryCall::TmPrimer3(_) => Some(device.0.unary_block[2]),
            UnaryCall::Hairpin(_) => None,
        };
        if let Some(cap) = cap {
            // Stage, Each and both Hits passes use this same effective block
            // size, including the per-block count and offset geometry.
            settings.threads_per_block = settings.threads_per_block.min(cap).max(32);
        }
        let run = Unary {
            device,
            stream,
            settings,
            output,
        };
        let n = table.len();
        let cutoff = options.cutoff;
        let dev_table = upload_unary_table(&run.stream, table, options.workers(), &fail)?;
        let off = dev_table.offsets();
        // Device hairpin: the handle's tables and the pack, once per call (§4).
        let p3 = match call {
            UnaryCall::Hairpin(pack) => Some(P3Pack::upload(&run.stream, device, pack)?),
            _ => None,
        };
        let mut p3_buffers = P3Buffers::default();
        let chunk_cap = n.min(run.settings.pairs_per_chunk as usize).max(1);
        let alloc = |error| fail("allocation", error);
        // Stage buffers for the largest chunk; host columns reserved up front
        // for Each (n rows), per chunk for Hits.
        let mut columns = Vec::new();
        for (field, int32) in unary_columns(&template, int32_slots, name)? {
            let stage = DevColumn::alloc(&run.stream, int32, chunk_cap).map_err(alloc)?;
            // Each: all n rows, zeroed without a fill pass, overwritten by the
            // per-chunk copies. Hits: empty, appended per chunk.
            let host = HostColumn::zeroed(int32, if each { n } else { 0 }, output)?;
            columns.push(Column {
                field,
                stage,
                hits: None,
                host,
            });
        }
        // SAFETY: written by sb_tm before any read.
        let passed_dev = match cutoff {
            Some(_) => Some(unsafe { dev_alloc::<u8>(&run.stream, chunk_cap) }.map_err(alloc)?),
            None => None,
        };
        let mut passed: Vec<u8> = if each && cutoff.is_some() {
            zeroed(n, "passed column")?
        } else {
            Vec::new()
        };
        let mut hit_i: Vec<i64> = Vec::new();
        let mut counts_dev: Option<CudaSlice<u64>> = None;
        let mut offsets_dev: Option<CudaSlice<u64>> = None;
        let mut out_i: Option<CudaSlice<i64>> = None;
        let table_arg = SbUnaryTable {
            data: ptr(&dev_table.data, &run.stream),
            off: ptr(&dev_table.off, &run.stream),
            view: 0,
        };
        let has_cutoff = i32::from(cutoff.is_some());
        let cutoff_bits = cutoff.map_or(0, f64::to_bits);
        let mask_each = i32::from(each && cutoff.is_some());
        let mut k0 = 0;
        seqblaze_core::cancel::stage(seqblaze_core::cancel::Stage::Sequences, n);
        while k0 < n {
            let mut k1 = unary_chunk_end(off, k0, &run.settings);
            // Device hairpin: each sequence's g and depth (§6.6); the chunk also
            // closes at the depth budget and holds at least one sequence.
            let mut buckets: [P3Bucket; 4] = Default::default();
            if p3.is_some() {
                let budget = p3_chunk_budget(&run.settings, device.0.resident_threads);
                let mut spent = 0u64;
                let mut end = k0;
                for k in k0..k1 {
                    let len = (off[k + 1] - off[k]) as u64;
                    let plan = p3_plan_t2(&run.settings, len);
                    let cost = plan.depth.saturating_mul(plan.g);
                    if end > k0 && spent.saturating_add(cost) > budget {
                        break;
                    }
                    spent = spent.saturating_add(cost);
                    // Amortized fallible growth: a refused host reservation raises
                    // DeviceError instead of aborting (primer3-cuda.md §9 step 6).
                    let bucket = &mut buckets[plan.variant()];
                    seqblaze_core::detail::try_reserve(
                        &mut bucket.items,
                        1,
                        "hairpin chunk items",
                    )?;
                    bucket.push(k - k0, &plan, p3_slot_t2(len));
                    end = k + 1;
                }
                k1 = end;
            }
            let rows = k1 - k0;
            let (start, count) = (k0 as u64, rows as u64);
            let passed_ptr = passed_dev.as_ref().map_or(0, |p| ptr(p, &run.stream));
            // Testing builds with a poison byte: refill every buffer the stage writes,
            // every chunk, so a row the stage misses shows the poison byte, never a
            // previous chunk's value (the allocation-time fill covers the first chunk
            // only; tm-compat-cuda.md §5.3, QTC8; primer3-cuda.md §11, QPC-16).
            #[cfg(feature = "testing")]
            if let Some(byte) = crate::launch::poison() {
                let hairpin = matches!(call, UnaryCall::Hairpin(_));
                let mut bytes = 0usize;
                for column in &columns {
                    let width = column.stage.width() as usize;
                    run.fill(column.stage.ptr(&run.stream), byte, rows * width)?;
                    if hairpin {
                        crate::launch::count_unary_refill(0, (rows * width) as u64, 0);
                    }
                    bytes += rows * width;
                }
                if passed_ptr != 0 {
                    run.fill(passed_ptr, byte, rows)?;
                    if hairpin {
                        crate::launch::count_unary_refill(0, 0, rows as u64);
                    }
                    bytes += rows;
                }
                if hairpin {
                    crate::launch::count_unary_refill(rows as u64, 0, 0);
                }
                crate::launch::count_poisoned(bytes as u64);
            }
            match (call, p3.as_ref()) {
                (
                    call @ (UnaryCall::Tm(_) | UnaryCall::TmBiopython(_) | UnaryCall::TmPrimer3(_)),
                    _,
                ) => {
                    let (stage, stage_name) = match call {
                        UnaryCall::Tm(_) => (&device.0.kernels.tm, "sb_tm"),
                        UnaryCall::TmBiopython(_) => {
                            (&device.0.kernels.tm_biopython, "sb_tm_biopython")
                        }
                        UnaryCall::TmPrimer3(_) => (&device.0.kernels.tm_primer3, "sb_tm_primer3"),
                        UnaryCall::Hairpin(_) => unreachable!("hairpin has a separate stage"),
                    };
                    // Stage: the kernel's stage writes the requested columns, `passed` and the masks.
                    let mut out = SbTmOut::default();
                    for column in &columns {
                        out.slots[column.field] = column.stage.ptr(&run.stream);
                    }
                    out.passed = passed_dev.as_ref().map_or(0, |p| ptr(p, &run.stream));
                    let mut launch = run.stream.launch_builder(stage);
                    launch.arg(&table_arg).arg(&start).arg(&count);
                    match call {
                        UnaryCall::Tm(pack) => launch.arg(pack),
                        UnaryCall::TmBiopython(pack) => launch.arg(pack),
                        UnaryCall::TmPrimer3(pack) => launch.arg(pack),
                        UnaryCall::Hairpin(_) => unreachable!("hairpin has a separate stage"),
                    };
                    launch
                        .arg(&out)
                        .arg(&has_cutoff)
                        .arg(&cutoff_bits)
                        .arg(&mask_each);
                    // SAFETY: argument types and order match the stage kernel (sb_tm,
                    // sb_tm_biopython and sb_tm_primer3 share their parameter list up to the
                    // pack's type); every buffer holds at least `rows` elements and the table
                    // holds rows k0..k1.
                    unsafe { launch.launch(run.grid(count)) }
                        .map_err(|error| run.fail(&format!("launching {stage_name}"), error))?;
                }
                (UnaryCall::Hairpin(_), Some(pack)) => {
                    // Stage: the group kernels write the requested columns, `passed`
                    // and the masks (§6.2 step 5).
                    let mut params = pack.params;
                    let mut slots = [0u64; HAIRPIN_SLOTS];
                    for column in &columns {
                        slots[column.field] = column.stage.ptr(&run.stream);
                    }
                    [params.found, params.dg, params.dh, params.ds, params.tm] = slots;
                    params.passed = passed_ptr;
                    params.has_cutoff = has_cutoff as u64;
                    params.cutoff = cutoff_bits;
                    params.mask_each = mask_each as u64;
                    launch_buckets(
                        (&run.stream, device),
                        &run.settings,
                        &buckets,
                        &params,
                        &P3Args::Hairpin(table_arg, start),
                        &mut p3_buffers,
                    )?;
                }
                (UnaryCall::Hairpin(_), None) => unreachable!("the hairpin pack is uploaded"),
            }
            if each {
                for column in &mut columns {
                    run.download_column(&column.stage, &mut column.host, k0, rows)?;
                }
                if let Some(device_passed) = passed_dev.as_ref() {
                    run.download(device_passed, &mut passed[k0..k1])?;
                }
            } else {
                let device_passed = passed_dev.as_ref().expect("Hits has a cutoff");
                let passed_ptr = ptr(device_passed, &run.stream);
                let (blocks, config) = run.tiles(rows);
                let counts = ensure(&run.stream, &mut counts_dev, blocks).map_err(alloc)?;
                let counts_ptr = ptr(counts, &run.stream);
                let mut launch = run
                    .stream
                    .launch_builder(&device.0.kernels.unary_hits_count);
                launch.arg(&passed_ptr).arg(&count).arg(&counts_ptr);
                // SAFETY: matches sb_unary_hits_count; one count slot per block.
                unsafe { launch.launch(config) }
                    .map_err(|error| run.fail("launching sb_unary_hits_count", error))?;
                let mut block_counts = vec![0u64; blocks];
                run.download(counts_dev.as_ref().expect("allocated"), &mut block_counts)?;
                run.sync()?;
                let mut offsets = block_counts;
                let mut total = 0u64;
                for value in &mut offsets {
                    let c = *value;
                    *value = total;
                    total += c;
                }
                if total > 0 {
                    let hits = total as usize;
                    let offsets_slice =
                        ensure(&run.stream, &mut offsets_dev, blocks).map_err(alloc)?;
                    crate::session::htod(&run.stream, &offsets, offsets_slice)
                        .map_err(|error| run.fail("uploading block offsets", error))?;
                    let offsets_ptr = ptr(offsets_slice, &run.stream);
                    let i_ptr = ptr(
                        ensure(&run.stream, &mut out_i, hits).map_err(alloc)?,
                        &run.stream,
                    );
                    let mut cols = SbUnaryCols {
                        count: columns.len() as u64,
                        ..SbUnaryCols::default()
                    };
                    for (c, column) in columns.iter_mut().enumerate() {
                        if column
                            .hits
                            .as_ref()
                            .is_none_or(|buffer| buffer.len() < hits)
                        {
                            column.hits = None;
                            let int32 = matches!(column.stage, DevColumn::I32(_));
                            column.hits =
                                Some(DevColumn::alloc(&run.stream, int32, hits).map_err(alloc)?);
                        }
                        cols.src[c] = column.stage.ptr(&run.stream);
                        cols.dst[c] = column.hits.as_ref().expect("allocated").ptr(&run.stream);
                        cols.width[c] = column.stage.width();
                    }
                    let mut launch = run
                        .stream
                        .launch_builder(&device.0.kernels.unary_hits_write);
                    launch
                        .arg(&passed_ptr)
                        .arg(&count)
                        .arg(&start)
                        .arg(&offsets_ptr)
                        .arg(&cols)
                        .arg(&i_ptr);
                    // SAFETY: matches sb_unary_hits_write; every destination holds
                    // `total` elements and every source `rows`.
                    unsafe { launch.launch(config) }
                        .map_err(|error| run.fail("launching sb_unary_hits_write", error))?;
                    let source_i = out_i.as_ref().expect("allocated");
                    run.append(source_i, &mut hit_i, hits, "hit indices")?;
                    for column in &mut columns {
                        let source = column.hits.as_ref().expect("allocated");
                        run.append_column(source, &mut column.host, hits)?;
                    }
                }
            }
            // The unary chunk loop's host boundary (cancellation-progress.md §3.5).
            seqblaze_core::cancel::boundary(seqblaze_core::cancel::Site::Sequences, k1 - k0)?;
            k0 = k1;
        }
        run.sync()?;
        let mut result = template;
        for column in columns {
            let data = match column.host {
                HostColumn::F64(values) => DetailData::Float64(values),
                HostColumn::I32(values) => DetailData::Int32(values),
            };
            result.set_column(column.field, data);
        }
        Ok(UnaryOutput {
            columns: result,
            passed: (each && cutoff.is_some()).then_some(passed),
            i: (!each).then_some(hit_i),
            n,
            counts: Counts {
                certified: 0,
                pruned: 0,
                full: n,
            },
            threads: options.threads,
            n_illegal_seqs: validation.n_illegal_seqs,
            validation_scan: validation.scans[0],
        })
    })
}

/// `slot`, grown to at least `len` elements (never zero bytes).
fn ensure<'s, T: DeviceRepr>(
    stream: &Arc<CudaStream>,
    slot: &'s mut Option<CudaSlice<T>>,
    len: usize,
) -> Result<&'s mut CudaSlice<T>, DriverError> {
    if slot.as_ref().is_none_or(|slice| slice.len() < len) {
        *slot = None;
        // SAFETY: written (upload or kernel) before any read.
        *slot = Some(unsafe { dev_alloc::<T>(stream, len.max(1)) }?);
    }
    Ok(slot.as_mut().expect("allocated above"))
}

pub(crate) fn f64_eval(
    device: &Device,
    op: F64Op,
    a: &[u64],
    b: Option<&[u64]>,
) -> Result<Vec<u64>, EngineError> {
    let code: i32 = match op {
        F64Op::Add => 0,
        F64Op::Sub => 1,
        F64Op::Mul => 2,
        F64Op::Div => 3,
        F64Op::Ge => 4,
        F64Op::FromI64 => 5,
        F64Op::Lt => 6,
        F64Op::Le => 7,
        F64Op::Gt => 8,
        F64Op::Eq => 9,
        F64Op::IsFinite => 10,
        F64Op::Abs => 11,
    };
    match (op.unary(), b) {
        (true, None) => {}
        (true, Some(_)) => {
            return Err(EngineError::Parameter(
                "a unary soft-binary64 operation takes b=None".into(),
            ))
        }
        (false, Some(b)) if b.len() == a.len() => {}
        _ => {
            return Err(EngineError::Parameter(
                "a binary soft-binary64 operation needs a and b of equal length".into(),
            ))
        }
    }
    if a.is_empty() {
        return Ok(Vec::new());
    }
    guard(|| {
        let label = format!("cuda:{}", device.info().ordinal);
        let fail = |what: &str, error: DriverError| failure(&label, what, error);
        let stream = device
            .0
            .ctx
            .new_stream()
            .map_err(|error| fail("creating a stream", error))?;
        let upload = |host: &[u64]| -> Result<CudaSlice<u64>, DriverError> {
            // SAFETY: fully written by the copy below.
            let mut slice = unsafe { dev_alloc::<u64>(&stream, host.len()) }?;
            crate::session::htod(&stream, host, &mut slice)?;
            Ok(slice)
        };
        let dev_a = upload(a).map_err(|error| fail("uploading operands", error))?;
        let dev_b = match b {
            Some(b) => Some(upload(b).map_err(|error| fail("uploading operands", error))?),
            None => None,
        };
        // SAFETY: written by sb_f64_eval before the copy reads it.
        let dev_out = unsafe { dev_alloc::<u64>(&stream, a.len()) }
            .map_err(|error| fail("allocation", error))?;
        let (a_ptr, out_ptr) = (ptr(&dev_a, &stream), ptr(&dev_out, &stream));
        let b_ptr = dev_b.as_ref().map_or(a_ptr, |slice| ptr(slice, &stream));
        let count = a.len() as u64;
        let threads = 256u64;
        let config = LaunchConfig {
            grid_dim: (count.div_ceil(threads).clamp(1, MAX_GRID) as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&device.0.kernels.f64_eval);
        launch
            .arg(&code)
            .arg(&a_ptr)
            .arg(&b_ptr)
            .arg(&out_ptr)
            .arg(&count);
        // SAFETY: matches sb_f64_eval; a, b and out hold `count` elements.
        unsafe { launch.launch(config) }.map_err(|error| fail("launching sb_f64_eval", error))?;
        let mut result = vec![0u64; a.len()];
        crate::session::dtoh(&stream, &dev_out, &mut result)
            .map_err(|error| fail("device-to-host copy", error))?;
        stream
            .synchronize()
            .map_err(|error| fail("synchronization", error))?;
        Ok(result)
    })
}
