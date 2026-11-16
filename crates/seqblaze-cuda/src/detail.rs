//! Device-only Levenshtein/alignment continuation and selected detail replay.
//! Planning downloads four capacity scalars; progress downloads seven integer
//! scalars. Rows, keys, per-entry lengths, traceback and payload never leave the
//! device. Construction buffers remain private until all launches complete.
use crate::{
    device::guard,
    interop::{
        execution_stream::DetailOperation, BoundDeviceRows, DeviceColumn, DeviceDType, DeviceLease,
        DeviceOutput,
    },
    resident::{
        ResidentDetails, ResidentEntries, ResidentField, ResidentRaggedField, ResidentRaggedKind,
        ResidentSpace,
    },
    resident_sinks::scan_u64,
    session::{failure_ordinal, ptr},
    AlignCall, AlignKind, EditMode,
};
use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, DriverError, LaunchConfig, PushKernelArg};
use seqblaze_core::{
    cancel::{self, Site, Stage},
    validate::EngineError,
};
use std::{mem::size_of, sync::Arc};

const STATE_WORDS: usize = 80;
const CHUNK_ENTRIES: usize = 1024;
const SCRATCH_TARGET: u64 = 128 * 1024 * 1024;
const QUANTUM: u64 = 4096;
/// Numeric output slots (kernels/detail.cuh SbDetailOutput.numeric): edit
/// b_end/n_ends/b_start/n_starts; align coords, stats, then n_ends (slot 8).
const NUMERIC_SLOTS: usize = 9;
/// Align's `n_ends` detail-mask bit (gotoh-align.md §3.1, align-cuda.md §3.1).
const ALIGN_N_ENDS: u64 = 2048;
const NUMERIC_NAMES: [[&str; NUMERIC_SLOTS]; 2] = [
    ["b_end", "n_ends", "b_start", "n_starts", "", "", "", "", ""],
    [
        "a_start",
        "a_end",
        "b_start",
        "b_end",
        "aln_len",
        "n_identity",
        "n_similarity",
        "n_gaps",
        "n_ends",
    ],
];

/// The detail-mask bit of a numeric slot: slot k is bit 2^k, except align's
/// slot 8 (`n_ends`), whose bit is 2048 (bit 256 is align's cigar).
fn slot_bit(align: bool, slot: usize) -> u64 {
    if align && slot == 8 {
        ALIGN_N_ENDS
    } else {
        1u64 << slot
    }
}

/// The dtype of a numeric slot: align coords and n_ends int64, stats int32;
/// edit n_ends and n_starts int32, b_end and b_start int64.
fn slot_dtype(align: bool, slot: usize) -> DeviceDType {
    let int64 = if align {
        slot < 4 || slot == 8
    } else {
        slot != 1 && slot != 3
    };
    if int64 {
        DeviceDType::I64
    } else {
        DeviceDType::I32
    }
}

/// A borrowed immutable call pack, not a new public dispatch or field policy.
pub(crate) enum DetailCall<'a> {
    Levenshtein(EditMode),
    Align(&'a AlignCall),
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Tables {
    a_data: u64,
    a_off: u64,
    b_data: u64,
    b_off: u64,
    a_view: u64,
    b_view: u64,
}
// SAFETY: six repr(C) words, matching kernels/hamming.cuh's SbTables.
unsafe impl DeviceRepr for Tables {}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Config {
    family: u64,
    kind: u64,
    fields: u64,
    go: u64,
    ge: u64,
    end_gaps: u64,
    mm: u64,
    matched: i64,
    mismatched: i64,
    n_classes: u64,
    class_map: u64,
    substitution: u64,
    identity_map: u64,
    display_map: u64,
    keys: u64,
    passed: u64,
    key_output: u64,
    output_base: u64,
}
// SAFETY: eighteen eight-byte scalar fields matching SbDetailConfig.
unsafe impl DeviceRepr for Config {}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Output {
    numeric: [u64; NUMERIC_SLOTS],
    counts: [u64; 3],
    offsets: [u64; 3],
    payload: [u64; 4],
    validity: u64,
    writing: u64,
}
// SAFETY: twenty-one repr(C) words matching SbDetailOutput.
unsafe impl DeviceRepr for Output {}
const _: () = {
    assert!(size_of::<Tables>() == 48);
    assert!(size_of::<Config>() == 144);
    assert!(size_of::<Output>() == 168);
};

fn parameter(message: &str) -> EngineError {
    EngineError::Parameter(message.into())
}
fn overflow(message: &str) -> EngineError {
    EngineError::Overflow(message.into())
}
fn failure(lease: &DeviceLease, operation: &str, e: DriverError) -> EngineError {
    failure_ordinal(lease.device(), operation, e)
}
fn one() -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Raw-address launches cannot register cudarc typed borrow events. Even an
/// error must finish outstanding consumers before freeing scratch; if the
/// driver can no longer synchronize, retain the allocation instead of freeing
/// memory that a device operation may still access.
struct Buffer<T: DeviceRepr> {
    data: Option<crate::arena::Scratch<T>>,
    stream: Arc<CudaStream>,
}
impl<T: DeviceRepr> Buffer<T> {
    fn allocate(lease: &DeviceLease, count: usize) -> Result<Self, EngineError> {
        count
            .max(1)
            .checked_mul(size_of::<T>())
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| overflow("resident detail allocation exceeds platform limits"))?;
        // SAFETY: each logical element is initialized by same-stream kernels
        // before any read. A zero-length logical buffer is never dereferenced.
        let data = unsafe { lease.execution().scratch(count.max(1)) }
            .map_err(|e| failure(lease, "allocating resident detail state", e))?;
        Ok(Self::adopt(lease, data))
    }
    fn adopt(lease: &DeviceLease, data: crate::arena::Scratch<T>) -> Self {
        Self {
            data: Some(data),
            stream: Arc::clone(lease.stream()),
        }
    }
    fn data(&self) -> &CudaSlice<T> {
        self.data.as_ref().expect("live detail workspace")
    }
    fn address(&self) -> u64 {
        ptr(self.data(), &self.stream)
    }
}
impl<T: DeviceRepr> Drop for Buffer<T> {
    fn drop(&mut self) {
        if self.data.is_some()
            && guard(|| {
                crate::interop::execution_stream::retire_in_drop(&self.stream)
                    .map_err(|_| EngineError::Device(String::new()))
            })
            .is_err()
        {
            if let Some(data) = self.data.take() {
                std::mem::forget(data);
            }
        }
    }
}

struct Pack {
    config: Config,
    // Constant class/score/identity/display metadata remains alive through the
    // final detail launch. These contain no sequence or per-pair data.
    _owners: Vec<DeviceColumn>,
}
fn pack(lease: &DeviceLease, call: &DetailCall<'_>, fields: u64) -> Result<Pack, EngineError> {
    let mut config = Config {
        fields,
        ..Config::default()
    };
    let mut owners = Vec::new();
    owners
        .try_reserve_exact(4)
        .map_err(|_| EngineError::Device(String::new()))?;
    match call {
        DetailCall::Levenshtein(mode) => {
            config.kind = match mode {
                EditMode::Global => 0,
                EditMode::Prefix => 1,
                EditMode::Infix => 2,
            };
        }
        DetailCall::Align(call) => {
            config.family = 1;
            config.kind = match call.kind {
                AlignKind::Local => 0,
                AlignKind::Global => 1,
                AlignKind::Water => 2,
                AlignKind::Needle => 3,
            };
            if call.go < 0 || call.ge < 0 || call.go > (1 << 30) || call.ge > (1 << 30) {
                return Err(parameter(
                    "resident alignment gap pack is outside the validated range",
                ));
            }
            config.go = call.go as u64;
            config.ge = call.ge as u64;
            config.end_gaps = u64::from(call.end_gaps);
            if let Some((matched, mismatched)) = call.matched {
                config.mm = 1;
                config.matched = i64::from(matched);
                config.mismatched = i64::from(mismatched);
            } else {
                if call.n_classes == 0
                    || call.n_classes > 256
                    || call.n_classes.checked_mul(call.n_classes) != Some(call.table.len())
                    || call.class.iter().any(|&x| usize::from(x) >= call.n_classes)
                {
                    return Err(parameter(
                        "resident alignment substitution pack is inconsistent",
                    ));
                }
                let class = lease.upload_bytes(DeviceDType::U8, vec![256], &call.class)?;
                config.n_classes = call.n_classes as u64;
                config.class_map = class.address();
                owners.push(class);
                let bytes = call
                    .table
                    .len()
                    .checked_mul(4)
                    .ok_or_else(|| overflow("alignment score pack length overflows"))?;
                let mut table = Vec::new();
                table
                    .try_reserve_exact(bytes)
                    .map_err(|_| EngineError::Device(String::new()))?;
                for &value in &call.table {
                    table.extend_from_slice(&value.to_le_bytes());
                }
                let table = lease.upload_bytes(DeviceDType::I32, vec![call.table.len()], &table)?;
                config.substitution = table.address();
                owners.push(table);
            }
            let compat = matches!(call.kind, AlignKind::Water | AlignKind::Needle);
            let (mut identity, mut display) = ([0u8; 256], [0u8; 256]);
            for (byte, (id, shown)) in identity.iter_mut().zip(&mut display).enumerate() {
                let raw = byte as u8;
                *id = if compat {
                    match raw.to_ascii_uppercase() {
                        b'X' => b'N',
                        x => x,
                    }
                } else {
                    raw
                };
                *shown = if compat {
                    match raw {
                        b'X' => b'N',
                        b'x' => b'n',
                        x => x,
                    }
                } else {
                    raw
                };
            }
            let identity = lease.upload_bytes(DeviceDType::U8, vec![256], &identity)?;
            config.identity_map = identity.address();
            owners.push(identity);
            let display = lease.upload_bytes(DeviceDType::U8, vec![256], &display)?;
            config.display_map = display.address();
            owners.push(display);
        }
    }
    Ok(Pack {
        config,
        _owners: owners,
    })
}

fn tables(a: &BoundDeviceRows, b: &BoundDeviceRows) -> Tables {
    Tables {
        a_view: a.address(),
        b_view: b.address(),
        ..Tables::default()
    }
}
fn validate_space(space: ResidentSpace) -> Result<usize, EngineError> {
    let n = usize::try_from(space.n)
        .map_err(|_| overflow("detail pair count exceeds platform limits"))?;
    space
        .k0
        .checked_add(space.n)
        .ok_or_else(|| overflow("detail work interval overflows"))?;
    if space.kind > 4
        || (n != 0
            && match space.kind {
                1 => space.li == 0 || space.lj == 0,
                2 | 4 => space.width == 0,
                3 => space.nseg == 0 || space.seg_k == 0 || space.seg_i == 0 || space.seg_j == 0,
                _ => false,
            })
    {
        return Err(parameter(
            "resident detail pair-space descriptor is inconsistent",
        ));
    }
    Ok(n)
}
fn control<const N: usize>(
    lease: &DeviceLease,
    input: &Buffer<u64>,
) -> Result<[u64; N], EngineError> {
    let mut out = [0u64; N];
    lease
        .execution()
        .detail_memcpy_dtoh(&input.data().slice(0..N), &mut out[..])
        .map_err(|e| failure(lease, "reading resident detail control scalars", e))?;
    lease.synchronize()?;
    Ok(out)
}
fn final_count(lease: &DeviceLease, input: &Buffer<u64>, n: usize) -> Result<u64, EngineError> {
    let mut out = [0u64];
    lease
        .execution()
        .detail_memcpy_dtoh(&input.data().slice(n..n + 1), &mut out[..])
        .map_err(|e| failure(lease, "reading resident detail allocation total", e))?;
    lease.synchronize()?;
    Ok(out[0])
}
fn copy(lease: &DeviceLease, dst: u64, src: u64, bytes: usize) -> Result<(), EngineError> {
    if bytes == 0 {
        return Ok(());
    }
    // SAFETY: private destination and immutable source are disjoint live
    // allocations with this validated span and the same consumer stream.
    unsafe { lease.execution().detail_copy_device(dst, src, bytes) }
        .map_err(|e| failure(lease, "publishing resident detail offsets", e))
}
fn publish(output: DeviceOutput) -> Result<DeviceColumn, EngineError> {
    // SAFETY: callers initialize all logical elements and finish all consumers
    // before relinquishing the construction-only writable buffer.
    unsafe { output.finish() }
}

/// Host boundaries of one replay (cancellation-progress.md §3.5).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Rounds {
    /// No boundary: a key replay inside the producer's pair stage, or a dense
    /// detail chunk whose caller reports its own `details` boundary.
    Silent,
    /// A `details` boundary after every progress readback: 0 items while a
    /// bounded chunk continues, its entry count once the chunk is finished.
    Details,
}

/// Run bounded batches. A single large pair uses its actual fallible scratch
/// reservation and continues on the device; it never falls back to a host DP.
/// Returned totals are checked before a global positive-integer prefix scan,
/// so scan arithmetic cannot wrap or produce unrepresentable int64 offsets.
fn run(
    lease: &DeviceLease,
    tables: Tables,
    space: ResidentSpace,
    config: Config,
    output: Output,
    rounds: Rounds,
) -> Result<[u64; 3], EngineError> {
    let n = validate_space(space)?;
    if config.output_base > space.k0 {
        return Err(parameter("detail output origin exceeds work origin"));
    }
    let mut totals = [0u64; 3];
    if n == 0 {
        return Ok(totals);
    }
    let kernels = &lease.device().0.kernels;
    let threads = kernels
        .detail_continue
        .max_threads_per_block()
        .map_err(|e| failure(lease, "querying detail continuation launch limit", e))?
        .min(128);
    if threads < 1 {
        return Err(EngineError::Device(
            "detail continuation has no supported block size".into(),
        ));
    }
    let threads = threads as u32;
    let control_buffer = Buffer::<u64>::allocate(lease, 7)?;
    let control_address = control_buffer.address();
    let mut at = 0usize;
    while at < n {
        let mut count = (n - at).min(CHUNK_ENTRIES);
        let states = Buffer::<u64>::allocate(
            lease,
            count
                .checked_mul(STATE_WORDS)
                .ok_or_else(|| overflow("detail state size overflows"))?,
        )?;
        let state_address = states.address();
        let mut chunk = space;
        chunk.k0 = space
            .k0
            .checked_add(at as u64)
            .ok_or_else(|| overflow("detail work index overflows"))?;
        let capacity = loop {
            chunk.n = count as u64;
            let mut launch = lease
                .execution()
                .detail_builder(&kernels.detail_plan, DetailOperation::Plan);
            launch
                .arg(&tables)
                .arg(&chunk)
                .arg(&config)
                .arg(&state_address)
                .arg(&control_address);
            // SAFETY: validated bound-row descriptors, scalar ABI mirrors and
            // states/control allocations span this bounded chunk completely.
            unsafe { launch.launch(one()) }
                .map_err(|e| failure(lease, "planning resident detail scratch", e))?;
            let planned = control::<4>(lease, &control_buffer)?;
            if planned[3] != 0 {
                return Err(parameter(
                    "resident detail plan exceeds validated field limits or row bounds",
                ));
            }
            let bytes = planned[0]
                .checked_mul(8)
                .and_then(|x| x.checked_add(planned[1]))
                .and_then(|x| x.checked_add(planned[2]))
                .ok_or_else(|| overflow("resident detail scratch size overflows"))?;
            if count == 1 || bytes <= SCRATCH_TARGET {
                break planned;
            }
            count = count.div_ceil(2);
        };
        let extent = |v: u64| {
            usize::try_from(v)
                .map_err(|_| overflow("resident detail scratch exceeds platform limits"))
        };
        let scratch = Buffer::<u64>::allocate(lease, extent(capacity[0])?)?;
        let codes = Buffer::<u8>::allocate(lease, extent(capacity[1])?)?;
        let moves = Buffer::<u8>::allocate(lease, extent(capacity[2])?)?;
        let (scratch_address, codes_address, moves_address) =
            (scratch.address(), codes.address(), moves.address());
        let grid = LaunchConfig {
            grid_dim: (count.div_ceil(threads as usize) as u32, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launches = 1usize;
        loop {
            for _ in 0..launches {
                let mut launch = lease
                    .execution()
                    .detail_builder(&kernels.detail_continue, DetailOperation::Continue);
                launch
                    .arg(&tables)
                    .arg(&chunk)
                    .arg(&config)
                    .arg(&output)
                    .arg(&state_address)
                    .arg(&scratch_address)
                    .arg(&codes_address)
                    .arg(&moves_address)
                    .arg(&QUANTUM);
                // SAFETY: the plan owns every scratch range; only this stream
                // advances its per-entry continuation state, with disjoint
                // output rows and fully retained source/pack owners.
                unsafe { launch.launch(grid) }
                    .map_err(|e| failure(lease, "continuing resident details", e))?;
            }
            let mut launch = lease
                .execution()
                .detail_builder(&kernels.detail_progress, DetailOperation::Progress);
            launch
                .arg(&state_address)
                .arg(&chunk.n)
                .arg(&output)
                .arg(&control_address);
            // SAFETY: bounded state is completely initialized by the planner;
            // a serial integer reduction writes seven control words.
            unsafe { launch.launch(one()) }
                .map_err(|e| failure(lease, "checking resident detail progress", e))?;
            let progress = control::<7>(lease, &control_buffer)?;
            if progress[1] != 0 {
                return Err(EngineError::Device(
                "device detail replay disagrees with a validated selecting key or output span; no result was published".into()));
            }
            if progress[6] != 0 {
                return Err(overflow("resident ragged detail total overflows"));
            }
            if progress[0] == 0 {
                for (sum, value) in totals.iter_mut().zip(&progress[3..6]) {
                    *sum = sum
                        .checked_add(*value)
                        .filter(|&x| x <= i64::MAX as u64)
                        .ok_or_else(|| overflow("resident ragged detail exceeds int64 offsets"))?;
                }
                break;
            }
            // A host boundary after the readback has synchronized the stream:
            // stopping here takes the error path of the checks above.
            if rounds == Rounds::Details {
                cancel::boundary(Site::Details, 0)?;
            }
            launches = (launches * 2).min(32);
        }
        at += count;
        if rounds == Rounds::Details {
            cancel::boundary(Site::Details, count)?;
        }
    }
    Ok(totals)
}

pub(crate) fn resident_keys(
    lease: &DeviceLease,
    a: &BoundDeviceRows,
    b: &BoundDeviceRows,
    space: ResidentSpace,
    call: &DetailCall<'_>,
) -> Result<DeviceColumn, EngineError> {
    guard(|| {
        let n = validate_space(space)?;
        let pack = pack(lease, call, 0)?;
        let output = lease.output(DeviceDType::I32, vec![n])?;
        let config = Config {
            key_output: output.address(),
            output_base: space.k0,
            ..pack.config
        };
        run(
            lease,
            tables(a, b),
            space,
            config,
            Output::default(),
            Rounds::Silent,
        )?;
        publish(output)
    })
}

fn detail_mask(call: &DetailCall<'_>, fields: &[&str]) -> Result<u64, EngineError> {
    let align = matches!(call, DetailCall::Align(_));
    let mut mask = 0u64;
    for &field in fields {
        let bit = if align {
            match field {
                "score" | "score_fx" => 0,
                "a_start" => 1,
                "a_end" => 2,
                "b_start" => 4,
                "b_end" => 8,
                "aln_len" => 16,
                "n_identity" => 32,
                "n_similarity" => 64,
                "n_gaps" => 128,
                "cigar" => 256,
                "aligned_a" => 512,
                "aligned_b" => 1024,
                "n_ends" => ALIGN_N_ENDS,
                _ => return Err(parameter("unknown resolved alignment detail field")),
            }
        } else {
            match field {
                "dist" => 0,
                "b_end" => 1,
                "n_ends" => 2,
                "b_start" => 4,
                "n_starts" => 8,
                "ends" => 16,
                "cigar" => 32,
                _ => return Err(parameter("unknown resolved Levenshtein detail field")),
            }
        };
        if bit != 0 && mask & bit != 0 {
            return Err(parameter("duplicate resolved device detail field"));
        }
        mask |= bit;
    }
    if let DetailCall::Levenshtein(EditMode::Global) = call {
        if mask & !32 != 0 {
            return Err(parameter("global Levenshtein offers only dist and cigar"));
        }
    }
    if let DetailCall::Align(call) = call {
        if mask & ALIGN_N_ENDS != 0 && !matches!(call.kind, AlignKind::Local) {
            return Err(parameter("n_ends is a native local alignment field"));
        }
    }
    Ok(mask)
}
fn validate_column(
    lease: &DeviceLease,
    column: &DeviceColumn,
    dtype: DeviceDType,
    n: usize,
) -> Result<(), EngineError> {
    if column.dtype() != dtype
        || column.shape() != [n]
        || column.ordinal() != lease.device().info().ordinal
    {
        return Err(parameter(
            "resident detail selected-column shape, dtype or device is inconsistent",
        ));
    }
    Ok(())
}

/// Detail fields of kept entries. With `stage` set (Each, Hits and TopK
/// entries), the replay opens a `details` progress stage per pass over the
/// entries and reports host boundaries at key `details`; a dense caller, which
/// reports its own chunk boundaries, passes false.
#[allow(clippy::too_many_arguments)]
pub(crate) fn resident_details(
    lease: &DeviceLease,
    a: &BoundDeviceRows,
    b: &BoundDeviceRows,
    entries: &ResidentEntries,
    call: &DetailCall<'_>,
    fields: &[&str],
    aligned_binary: bool,
    stage: bool,
) -> Result<ResidentDetails, EngineError> {
    guard(|| {
        let mask = detail_mask(call, fields)?;
        let align = matches!(call, DetailCall::Align(_));
        if entries.i.shape().len() != 1 {
            return Err(parameter("resident detail entries must be one-dimensional"));
        }
        let n = entries.i.shape()[0];
        validate_column(lease, &entries.i, DeviceDType::I64, n)?;
        validate_column(lease, &entries.j, DeviceDType::I64, n)?;
        validate_column(lease, &entries.keys, DeviceDType::I32, n)?;
        if let Some(passed) = &entries.passed {
            validate_column(lease, passed, DeviceDType::Bool, n)?;
        }
        let mut result = ResidentDetails {
            numeric: Vec::new(),
            ragged: Vec::new(),
        };
        result
            .numeric
            .try_reserve_exact(NUMERIC_SLOTS)
            .map_err(|_| EngineError::Device(String::new()))?;
        result
            .ragged
            .try_reserve_exact(4)
            .map_err(|_| EngineError::Device(String::new()))?;
        if mask == 0 {
            return Ok(result);
        }
        let pack = pack(lease, call, mask)?;
        let config = Config {
            keys: entries.keys.address(),
            passed: entries.passed.as_ref().map_or(0, DeviceColumn::address),
            ..pack.config
        };
        let space = ResidentSpace {
            kind: 1,
            n: n as u64,
            li: entries.i.address(),
            lj: entries.j.address(),
            ..ResidentSpace::default()
        };
        let mut output = Output::default();
        let mut numeric: [Option<DeviceOutput>; NUMERIC_SLOTS] = std::array::from_fn(|_| None);
        for (index, slot) in numeric.iter_mut().enumerate() {
            if mask & slot_bit(align, index) != 0 && (align || index < 4) {
                let values = lease.output(slot_dtype(align, index), vec![n])?;
                output.numeric[index] = values.address();
                *slot = Some(values);
            }
        }
        let needs = if align {
            [mask & 256 != 0, false, mask & (512 | 1024) != 0]
        } else {
            [mask & 32 != 0, mask & 16 != 0, false]
        };
        // CPU detail columns always carry explicit bool validity, including
        // all-valid and empty outputs. Reuse Each's mask when present; otherwise
        // let the detail pass initialize one shared bitmap for the ragged fields.
        let validity = if needs.iter().any(|&needed| needed) && entries.passed.is_none() {
            let values = lease.output(DeviceDType::Bool, vec![n])?;
            output.validity = values.address();
            Some(values)
        } else {
            None
        };
        let mut counts: [Option<Buffer<u64>>; 3] = std::array::from_fn(|_| None);
        for (index, slot) in counts.iter_mut().enumerate() {
            if needs[index] {
                let values = Buffer::<u64>::allocate(lease, n)?;
                output.counts[index] = values.address();
                *slot = Some(values);
            }
        }
        let replays = needs.iter().any(|&needed| needed);
        // One `details` stage row per pass over the entries; entries count once
        // per pass, so each row ends with done == total.
        let rounds = || {
            if stage {
                cancel::stage(Stage::Details, n);
                Rounds::Details
            } else {
                Rounds::Silent
            }
        };
        let totals = run(lease, tables(a, b), space, config, output, rounds())?;
        let mut offsets: [Option<DeviceColumn>; 3] = std::array::from_fn(|_| None);
        let mut payload: [Option<DeviceOutput>; 4] = std::array::from_fn(|_| None);
        for channel in 0..3 {
            if let Some(counts) = &counts[channel] {
                let prefix = Buffer::adopt(lease, scan_u64(lease, counts.data(), n)?);
                if final_count(lease, &prefix, n)? != totals[channel] {
                    return Err(EngineError::Device(
                        "resident detail prefix disagrees with checked allocation total".into(),
                    ));
                }
                let len = n
                    .checked_add(1)
                    .ok_or_else(|| overflow("resident offset length overflows"))?;
                let values = lease.output(DeviceDType::I64, vec![len])?;
                copy(lease, values.address(), prefix.address(), values.nbytes())?;
                let values = publish(values)?;
                output.offsets[channel] = values.address();
                offsets[channel] = Some(values);
            }
        }
        let mut allocate_payload =
            |slot: usize, channel: usize, dtype: DeviceDType| -> Result<(), EngineError> {
                let len = usize::try_from(totals[channel])
                    .map_err(|_| overflow("resident payload length exceeds platform limits"))?;
                let values = lease.output(dtype, vec![len])?;
                output.payload[slot] = values.address();
                payload[slot] = Some(values);
                Ok(())
            };
        if needs[0] {
            allocate_payload(0, 0, DeviceDType::U8)?;
        }
        if needs[1] {
            allocate_payload(1, 1, DeviceDType::I64)?;
        }
        if align && mask & 512 != 0 {
            allocate_payload(2, 2, DeviceDType::U8)?;
        }
        if align && mask & 1024 != 0 {
            allocate_payload(3, 2, DeviceDType::U8)?;
        }
        if replays {
            output.writing = 1;
            let repeated = run(lease, tables(a, b), space, config, output, rounds())?;
            if repeated != totals {
                return Err(EngineError::Device(
                    "resident detail input changed during payload replay; no result was published"
                        .into(),
                ));
            }
        }
        let validity = validity
            .map(publish)
            .transpose()?
            .or_else(|| entries.passed.clone());
        for (index, values) in numeric.into_iter().enumerate() {
            if let Some(values) = values {
                let dtype = slot_dtype(align, index);
                result.numeric.push(ResidentField {
                    name: NUMERIC_NAMES[usize::from(align)][index].into(),
                    values: publish(values)?,
                    mask_bits: if dtype == DeviceDType::I64 {
                        i64::MAX as u64
                    } else {
                        i32::MAX as u64
                    },
                });
            }
        }
        for (slot, values) in payload.into_iter().enumerate() {
            if let Some(values) = values {
                let (name, channel, kind) = match slot {
                    0 => ("cigar", 0, ResidentRaggedKind::Utf8),
                    1 => ("ends", 1, ResidentRaggedKind::ListI64),
                    2 => (
                        "aligned_a",
                        2,
                        if aligned_binary {
                            ResidentRaggedKind::Binary
                        } else {
                            ResidentRaggedKind::Utf8
                        },
                    ),
                    _ => (
                        "aligned_b",
                        2,
                        if aligned_binary {
                            ResidentRaggedKind::Binary
                        } else {
                            ResidentRaggedKind::Utf8
                        },
                    ),
                };
                result.ragged.push(ResidentRaggedField {
                    name: name.into(),
                    kind,
                    offsets: offsets[channel]
                        .as_ref()
                        .expect("constructed detail offsets")
                        .clone(),
                    values: publish(values)?,
                    validity: validity.clone(),
                });
            }
        }
        Ok(result)
    })
}
