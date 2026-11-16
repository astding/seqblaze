//! Resident score fast paths using the existing CUDA thread/warp families.
//! GPU classification produces only 22 host allocation-control integers.
//! Pair coordinates and results remain device-side; the former HOST domain
//! calls the device continuation in detail.rs and never reads sequence bytes.
use crate::{
    detail::{resident_keys, DetailCall},
    device::guard,
    interop::{BoundDeviceRows, DeviceColumn, DeviceDType, DeviceLease, DeviceOutput},
    launch::Settings,
    resident::ResidentSpace,
    session::{failure_ordinal, ptr},
    AlignKind, EditMode,
};
use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, DeviceRepr, DriverError, LaunchConfig, PushKernelArg,
};
use seqblaze_core::validate::EngineError;
use std::{mem::size_of, sync::Arc};

const SLOT_TARGET: usize = 64 * 1024 * 1024;
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
// SAFETY: six repr(C) words matching SbTables.
unsafe impl DeviceRepr for Tables {}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Lev {
    mode: u64,
    thread_words: u64,
    device_words: u64,
    pair_steps: u64,
    sigma: u64,
    map: u64,
    slots: u64,
    slot_words: u64,
    b_end: u64,
    n_ends: u64,
}
// SAFETY: ten repr(C) u64 fields matching SbLev.
unsafe impl DeviceRepr for Lev {}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Aln {
    kind: u64,
    end_gaps: u64,
    go: u64,
    ge: u64,
    mm: u64,
    matched: i64,
    mismatched: i64,
    n_classes: u64,
    cls: u64,
    table: u64,
    thread_rows: u64,
    lane_rows: u64,
    pair_cells: u64,
    slots: u64,
    slot_cols: u64,
}
// SAFETY: fifteen eight-byte integer fields matching SbAln.
unsafe impl DeviceRepr for Aln {}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Plan {
    family: u64,
    lev: Lev,
    aln: Aln,
}
// SAFETY: repr(C) mirrors SbResidentEditPlan, with integer-only nested packs.
unsafe impl DeviceRepr for Plan {}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Routes {
    bins: u64,
    cursors: u64,
    i: u64,
    j: u64,
    positions: u64,
    starts: [u64; 8],
}
// SAFETY: thirteen repr(C) u64 words matching SbResidentEditRoutes.
unsafe impl DeviceRepr for Routes {}
const _: () = {
    assert!(size_of::<Tables>() == 48);
    assert!(size_of::<Lev>() == 80);
    assert!(size_of::<Aln>() == 120);
    assert!(size_of::<Plan>() == 208);
    assert!(size_of::<Routes>() == 104);
};
fn parameter(text: &str) -> EngineError {
    EngineError::Parameter(text.into())
}
fn overflow(text: &str) -> EngineError {
    EngineError::Overflow(text.into())
}
fn fail(lease: &DeviceLease, operation: &str, error: DriverError) -> EngineError {
    failure_ordinal(lease.device(), operation, error)
}

/// Keep raw-pointer construction scratch alive until its stream consumers
/// retire, including every early error. Failure to synchronize leaks the
/// allocation rather than freeing memory a failed driver may still reference.
struct Buffer<T: DeviceRepr> {
    data: Option<CudaSlice<T>>,
    stream: Arc<CudaStream>,
}
impl<T: DeviceRepr> Buffer<T> {
    fn allocate(lease: &DeviceLease, n: usize) -> Result<Self, EngineError> {
        n.max(1)
            .checked_mul(size_of::<T>())
            .filter(|&b| b <= isize::MAX as usize)
            .ok_or_else(|| overflow("resident edit workspace exceeds platform limits"))?;
        // SAFETY: logical elements are initialized by copies, memset or
        // same-stream kernels before use; an empty span is not read.
        let data = unsafe { lease.execution().alloc(n.max(1)) }
            .map_err(|e| fail(lease, "allocating resident edit workspace", e))?;
        Ok(Self {
            data: Some(data),
            stream: Arc::clone(lease.stream()),
        })
    }
    fn data(&self) -> &CudaSlice<T> {
        self.data.as_ref().expect("live edit workspace")
    }
    fn address(&self) -> u64 {
        ptr(self.data(), &self.stream)
    }
}
impl Buffer<u64> {
    fn zeros(lease: &DeviceLease, n: usize) -> Result<Self, EngineError> {
        let out = Self::allocate(lease, n)?;
        // SAFETY: this private allocation covers n.max(1) u64 values. Routing
        // kernels consume these counters on the execution stream, so their
        // zero initialization must be ordered on that same stream.
        unsafe {
            lease
                .execution()
                .memset(out.address(), 0, n.max(1) * size_of::<u64>())
        }
        .map_err(|e| fail(lease, "initializing resident edit workspace", e))?;
        Ok(out)
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
    plan: Plan,
    _owners: Vec<DeviceColumn>,
}
fn pack(
    lease: &DeviceLease,
    call: &DetailCall<'_>,
    settings: &Settings,
    acgt: bool,
) -> Result<Pack, EngineError> {
    if settings.thread_words > 4
        || settings.device_words > 256
        || settings.thread_rows > 32
        || ![1, 2, 4, 8].contains(&settings.lane_rows)
        || settings.pair_steps > (1 << 40)
        || settings.pair_cells > (1 << 40)
        || !(32..=1024).contains(&settings.threads_per_block)
        || settings.threads_per_block % 32 != 0
    {
        return Err(parameter(
            "resident edit launch settings are outside compiled limits",
        ));
    }
    let mut plan = Plan::default();
    let mut owners = Vec::new();
    owners
        .try_reserve_exact(2)
        .map_err(|_| EngineError::Device(String::new()))?;
    match call {
        DetailCall::Levenshtein(mode) => {
            plan.lev = Lev {
                mode: match mode {
                    EditMode::Global => 0,
                    EditMode::Prefix => 1,
                    EditMode::Infix => 2,
                },
                thread_words: settings.thread_words,
                device_words: settings.device_words,
                pair_steps: settings.pair_steps,
                ..Lev::default()
            };
            // The raw kernels' identity map; the four-code kernels read no map
            // (levenshtein-cuda.md §13.4 step 2).
            if !acgt {
                let mut map = [0u8; 512];
                for (byte, slot) in map.chunks_exact_mut(2).enumerate() {
                    slot.copy_from_slice(&(byte as u16).to_le_bytes());
                }
                let map = lease.upload_bytes(DeviceDType::U16, vec![256], &map)?;
                plan.lev.sigma = 256;
                plan.lev.map = map.address();
                owners.push(map);
            }
        }
        DetailCall::Align(call) => {
            if call.go < 0 || call.ge < 0 || call.go > (1 << 30) || call.ge > (1 << 30) {
                return Err(parameter(
                    "resident alignment gap pack is outside the validated range",
                ));
            }
            let (matched, mismatched) = call.matched.unwrap_or((0, 0));
            plan.family = 1;
            plan.aln = Aln {
                kind: match call.kind {
                    AlignKind::Local => 0,
                    AlignKind::Global => 1,
                    AlignKind::Water => 2,
                    AlignKind::Needle => 3,
                },
                end_gaps: u64::from(call.end_gaps),
                go: call.go as u64,
                ge: call.ge as u64,
                mm: u64::from(call.matched.is_some()),
                matched: i64::from(matched),
                mismatched: i64::from(mismatched),
                n_classes: call.n_classes as u64,
                thread_rows: settings.thread_rows,
                lane_rows: settings.lane_rows,
                pair_cells: settings.pair_cells,
                ..Aln::default()
            };
            if call.matched.is_none() {
                if call.n_classes == 0
                    || call.n_classes > 256
                    || call.n_classes.checked_mul(call.n_classes) != Some(call.table.len())
                    || call.class.iter().any(|&x| usize::from(x) >= call.n_classes)
                {
                    return Err(parameter(
                        "resident alignment substitution pack is inconsistent",
                    ));
                }
                let cls = lease.upload_bytes(DeviceDType::U8, vec![256], &call.class)?;
                plan.aln.cls = cls.address();
                owners.push(cls);
                let mut bytes = Vec::new();
                let n = call
                    .table
                    .len()
                    .checked_mul(4)
                    .ok_or_else(|| overflow("alignment pack length overflows"))?;
                bytes
                    .try_reserve_exact(n)
                    .map_err(|_| EngineError::Device(String::new()))?;
                for &value in &call.table {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                let table = lease.upload_bytes(DeviceDType::I32, vec![call.table.len()], &bytes)?;
                plan.aln.table = table.address();
                owners.push(table);
            }
        }
    }
    Ok(Pack {
        plan,
        _owners: owners,
    })
}

fn block(lease: &DeviceLease, function: &CudaFunction, requested: u32) -> Result<u32, EngineError> {
    let supported = function
        .max_threads_per_block()
        .map_err(|e| fail(lease, "querying resident edit launch limit", e))?;
    let threads = requested.min(supported.max(0) as u32) / 32 * 32;
    if threads == 0 {
        return Err(EngineError::Device(
            "resident edit kernel cannot launch a full warp".into(),
        ));
    }
    Ok(threads)
}
fn grid(threads: u32, n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(threads as usize).clamp(1, 65535) as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}
fn stage_grid(
    lease: &DeviceLease,
    function: &CudaFunction,
    settings: &Settings,
    n: usize,
    warp: bool,
    slot_bytes: usize,
) -> Result<(LaunchConfig, usize), EngineError> {
    let mut threads = block(lease, function, settings.threads_per_block)?;
    let max_units = SLOT_TARGET
        .checked_div(slot_bytes)
        .map_or(usize::MAX, |n| n.max(1));
    let per_unit = if warp { 32 } else { 1 };
    if max_units < threads as usize / per_unit {
        threads = (max_units
            .saturating_mul(per_unit)
            .max(32)
            .min(threads as usize)
            / 32
            * 32) as u32;
    }
    let units = threads as usize / per_unit;
    let resident = usize::try_from(lease.device().0.resident_threads)
        .unwrap_or(usize::MAX)
        .div_ceil(threads as usize)
        .max(1);
    let blocks = n
        .div_ceil(units)
        .max(1)
        .min(resident)
        .min(65535)
        .min((max_units / units).max(1));
    let total = blocks
        .checked_mul(units)
        .ok_or_else(|| overflow("resident edit launch units overflow"))?;
    Ok((
        LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        },
        total,
    ))
}
fn offset(base: u64, index: u64, width: u64) -> Result<u64, EngineError> {
    index
        .checked_mul(width)
        .and_then(|v| base.checked_add(v))
        .ok_or_else(|| overflow("resident edit slice address overflows"))
}
fn publish(out: DeviceOutput) -> Result<DeviceColumn, EngineError> {
    // SAFETY: every logical element is initialized and this private writable
    // construction owner is relinquished after the final stream synchronize.
    unsafe { out.finish() }
}
fn finish(
    output: DeviceOutput,
    b_end: Option<DeviceOutput>,
    n_ends: Option<DeviceOutput>,
    continued: u64,
) -> Result<EditKeys, EngineError> {
    Ok(EditKeys {
        keys: publish(output)?,
        b_end: b_end.map(publish).transpose()?,
        n_ends: n_ends.map(publish).transpose()?,
        continued,
    })
}
fn scatter(
    lease: &DeviceLease,
    positions: u64,
    values: u64,
    output: u64,
    n: u64,
    threads: u32,
) -> Result<(), EngineError> {
    let mut launch = lease
        .execution()
        .launch_builder(&lease.device().0.kernels.resident_edit_scatter);
    launch.arg(&positions).arg(&values).arg(&output).arg(&n);
    // SAFETY: compact positions are a permutation of disjoint original chunk
    // positions; the input and destination spans cover every logical int32.
    unsafe { launch.launch(grid(threads, n as usize)) }
        .map_err(|e| fail(lease, "scattering resident edit keys", e))?;
    Ok(())
}

/// Levenshtein options of one resident chunk (levenshtein-cuda.md §13.4).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LevStage {
    /// Step 2: the four-code fast-tier kernels (a strict ACGT call).
    pub acgt: bool,
    /// Step 1: the fast tiers' in-pass `b_end` and `n_ends` per chunk item.
    pub b_end: bool,
    pub n_ends: bool,
    /// The validated longest rows of tables a and b (§14): when they prove
    /// that no pair needs the device continuation, the tier kernels run on the
    /// chunk's own work space, without classification or compaction.
    pub longest: Option<[u64; 2]>,
}

/// The fast-tier kernels that together cover every pair of rows no longer than
/// `longest` (levenshtein-cuda.md §14), or `None` when such a pair may need
/// the device continuation (bucket 6). Index 0 is the thread tier, 1–4 the
/// warp tiers Q = 1, 2, 4, 8. The predicate is `sb_resident_edit_bucket`'s,
/// evaluated at the largest W and n: both depth tests are monotone in W and n,
/// so every pair then lands in buckets 1–5, and each tier kernel computes
/// exactly the pairs of its own bucket and skips the rest.
fn direct_tiers(settings: &Settings, longest: [u64; 2]) -> Option<[bool; 5]> {
    let (words, n) = (longest[0].div_ceil(64), longest[1]);
    let (thread, steps) = (settings.thread_words, settings.pair_steps);
    if words > settings.device_words {
        return None;
    }
    let thread_words = words.min(thread);
    if thread != 0 && thread_words != 0 && n > steps / thread_words {
        return None;
    }
    let warp = thread == 0 || words > thread;
    if warp {
        let lanes = words.div_ceil(32);
        if lanes != 0 && (n > u64::MAX - 31 || n + 31 > steps / lanes) {
            return None;
        }
    }
    Some([
        thread != 0,
        warp,
        warp && words > 32,
        warp && words > 64,
        warp && words > 128,
    ])
}
/// One resident edit chunk: keys in work order, the requested in-pass fields
/// (written for every fast-tier item; a continued item's entry is never
/// written) and the number of items the device continuation served.
pub(crate) struct EditKeys {
    pub keys: DeviceColumn,
    pub b_end: Option<DeviceColumn>,
    pub n_ends: Option<DeviceColumn>,
    pub continued: u64,
}

/// Preserves the original work order (including explicit repeats). The only
/// host downloads are fixed-size controls. Existing fast tiers receive their
/// own compact safe domain; former HOST items execute device continuation.
pub(crate) fn resident_keys_fast(
    lease: &DeviceLease,
    a: &BoundDeviceRows,
    b: &BoundDeviceRows,
    space: ResidentSpace,
    call: &DetailCall<'_>,
    settings: &Settings,
    options: LevStage,
) -> Result<EditKeys, EngineError> {
    guard(|| {
        let n = usize::try_from(space.n)
            .map_err(|_| overflow("resident edit pair count exceeds platform limits"))?;
        space
            .k0
            .checked_add(space.n)
            .ok_or_else(|| overflow("resident edit work interval overflows"))?;
        if space.kind > 4
            || (n != 0
                && match space.kind {
                    1 => space.li == 0 || space.lj == 0,
                    2 | 4 => space.width == 0,
                    3 => {
                        space.nseg == 0 || space.seg_k == 0 || space.seg_i == 0 || space.seg_j == 0
                    }
                    _ => false,
                })
        {
            return Err(parameter(
                "resident edit pair-space descriptor is inconsistent",
            ));
        }
        let lev_family = matches!(call, DetailCall::Levenshtein(_));
        let options = if lev_family {
            options
        } else {
            LevStage::default()
        };
        let output = lease.output(DeviceDType::I32, vec![n])?;
        let b_end_out = if options.b_end {
            Some(lease.output(DeviceDType::I64, vec![n])?)
        } else {
            None
        };
        let n_ends_out = if options.n_ends {
            Some(lease.output(DeviceDType::I32, vec![n])?)
        } else {
            None
        };
        let in_pass = options.b_end || options.n_ends;
        if n == 0 {
            return finish(output, b_end_out, n_ends_out, 0);
        }
        let pack = pack(lease, call, settings, options.acgt)?;
        let tables = Tables {
            a_view: a.address(),
            b_view: b.address(),
            ..Tables::default()
        };
        let kernels = &lease.device().0.kernels;
        let direct = match options.longest {
            Some(longest) if pack.plan.family == 0 => {
                direct_tiers(settings, longest).map(|tiers| (tiers, longest[0].div_ceil(64)))
            }
            _ => None,
        };
        if let Some((tiers, words)) = direct {
            // §14: each tier kernel runs over every chunk item in the chunk's
            // own work space, computes the pairs of its bucket (the same word
            // steps as after compaction) and writes them, with the in-pass
            // fields, at their chunk positions; every item has one bucket in
            // 1–5, so every key is written exactly once. No host read, no wait.
            let mut lev = pack.plan.lev;
            lev.b_end = b_end_out.as_ref().map_or(0, DeviceOutput::address);
            lev.n_ends = n_ends_out.as_ref().map_or(0, DeviceOutput::address);
            let raw = !options.acgt;
            let key_address = output.address();
            for (tier, _) in tiers.iter().enumerate().filter(|(_, on)| **on) {
                let (function, tier_words) = if tier == 0 {
                    (
                        &kernels.lev_thread[usize::from(raw)],
                        words.min(settings.thread_words),
                    )
                } else {
                    (
                        &kernels.lev_warp[tier - 1][usize::from(raw)],
                        words.min(32 << (tier - 1)),
                    )
                };
                // Raw kernels: a zeroed slot of 256 words per Peq word of the
                // tier's widest pair, per unit, as for a compacted bucket.
                let slot_words = tier_words
                    .checked_mul(256)
                    .ok_or_else(|| overflow("resident edit Peq size overflows"))?;
                let slot_bytes = usize::try_from(slot_words)
                    .ok()
                    .and_then(|v| v.checked_mul(8))
                    .ok_or_else(|| overflow("resident edit Peq bytes exceed platform limits"))?;
                let (launch_config, units) = stage_grid(
                    lease,
                    function,
                    settings,
                    n,
                    tier != 0,
                    if raw { slot_bytes } else { 0 },
                )?;
                let _slots = if raw {
                    let len =
                        units
                            .checked_mul(usize::try_from(slot_words).map_err(|_| {
                                overflow("resident Peq words exceed platform limits")
                            })?)
                            .ok_or_else(|| overflow("resident edit Peq workspace overflows"))?;
                    let slots = Buffer::<u64>::zeros(lease, len)?;
                    lev.slots = slots.address();
                    lev.slot_words = slot_words;
                    Some(slots)
                } else {
                    None
                };
                let mut launch = lease.execution().launch_builder(function);
                launch.arg(&tables).arg(&space).arg(&lev).arg(&key_address);
                // SAFETY: direct_tiers proves every item's bucket is 1–5; this
                // kernel writes exactly its own tier's items (sb_lev_tier and the
                // Q range), each raw unit owns a zeroed slot of the tier's widest
                // pair, and the outputs cover every chunk position.
                unsafe { launch.launch(launch_config) }
                    .map_err(|e| fail(lease, "running resident Levenshtein fast tier", e))?;
            }
            return finish(output, b_end_out, n_ends_out, 0);
        }
        let classify_threads = block(
            lease,
            &kernels.resident_edit_classify,
            settings.threads_per_block,
        )?;
        let compact_threads = block(
            lease,
            &kernels.resident_edit_compact,
            settings.threads_per_block,
        )?;
        let scatter_threads = block(
            lease,
            &kernels.resident_edit_scatter,
            settings.threads_per_block,
        )?;
        let lev_scatter_threads = if in_pass {
            block(lease, &kernels.lev_scatter, settings.threads_per_block)?
        } else {
            0
        };
        let bins = Buffer::<u8>::allocate(lease, n)?;
        let control = Buffer::<u64>::zeros(lease, 22)?;
        let (bins_address, control_address) = (bins.address(), control.address());
        let mut launch = lease
            .execution()
            .launch_builder(&kernels.resident_edit_classify);
        launch
            .arg(&tables)
            .arg(&space)
            .arg(&pack.plan)
            .arg(&bins_address)
            .arg(&control_address);
        // SAFETY: source descriptors are already validated on this lease;
        // bins/control spans are initialized or completely written here.
        unsafe { launch.launch(grid(classify_threads, n)) }
            .map_err(|e| fail(lease, "classifying resident edit tiers", e))?;
        let mut summary = [0u64; 22];
        lease
            .execution()
            .memcpy_dtoh(control.data(), &mut summary[..])
            .map_err(|e| fail(lease, "reading resident edit allocation controls", e))?;
        lease.synchronize()?;
        if summary[21] != 0 {
            return Err(parameter(
                "resident edit route references an invalid row descriptor",
            ));
        }
        let mut starts = [0u64; 8];
        for bucket in 0..7 {
            starts[bucket + 1] = starts[bucket]
                .checked_add(summary[bucket])
                .ok_or_else(|| overflow("resident edit bucket counts overflow"))?;
        }
        if starts[7] != space.n {
            return Err(EngineError::Device(
                "resident edit routing lost a work item".into(),
            ));
        }
        let cursors = Buffer::<u64>::zeros(lease, 7)?;
        let i = Buffer::<i64>::allocate(lease, n)?;
        let j = Buffer::<i64>::allocate(lease, n)?;
        let positions = Buffer::<u64>::allocate(lease, n)?;
        let routes = Routes {
            bins: bins.address(),
            cursors: cursors.address(),
            i: i.address(),
            j: j.address(),
            positions: positions.address(),
            starts,
        };
        let mut launch = lease
            .execution()
            .launch_builder(&kernels.resident_edit_compact);
        launch.arg(&space).arg(&routes);
        // SAFETY: classified buckets sum to n, each integer cursor reserves a
        // disjoint initialized list slot, with no host pair expansion.
        unsafe { launch.launch(grid(compact_threads, n)) }
            .map_err(|e| fail(lease, "compacting resident edit tiers", e))?;
        for bucket in 0..7 {
            let count = summary[bucket];
            if count == 0 {
                continue;
            }
            let count_usize = usize::try_from(count)
                .map_err(|_| overflow("resident edit bucket exceeds platform limits"))?;
            let selected = ResidentSpace {
                kind: 1,
                n: count,
                li: offset(i.address(), starts[bucket], 8)?,
                lj: offset(j.address(), starts[bucket], 8)?,
                ..ResidentSpace::default()
            };
            let position = offset(positions.address(), starts[bucket], 8)?;
            if bucket == 6 {
                let keys = resident_keys(lease, a, b, selected, call)?;
                scatter(
                    lease,
                    position,
                    keys.address(),
                    output.address(),
                    count,
                    scatter_threads,
                )?;
                continue;
            }
            let keys = lease.output(DeviceDType::I32, vec![count_usize])?;
            if pack.plan.family == 0 {
                let mut lev = pack.plan.lev;
                let max_m = summary[7 + bucket];
                let words = max_m / 64 + u64::from(max_m % 64 != 0);
                let slot_words = words
                    .checked_mul(256)
                    .ok_or_else(|| overflow("resident edit Peq size overflows"))?;
                let slot_bytes = usize::try_from(slot_words)
                    .ok()
                    .and_then(|v| v.checked_mul(8))
                    .ok_or_else(|| overflow("resident edit Peq bytes exceed platform limits"))?;
                // The four-code kernels keep Peq in registers: no slot, no zero
                // fill, one unit per item up to the resident wave (§13.4 step 2).
                let raw = !options.acgt;
                let function = if bucket == 1 {
                    &kernels.lev_thread[usize::from(raw)]
                } else {
                    &kernels.lev_warp[bucket - 2][usize::from(raw)]
                };
                let (launch_config, units) = stage_grid(
                    lease,
                    function,
                    settings,
                    count_usize,
                    bucket != 1,
                    if raw { slot_bytes } else { 0 },
                )?;
                let _slots = if raw {
                    let len =
                        units
                            .checked_mul(usize::try_from(slot_words).map_err(|_| {
                                overflow("resident Peq words exceed platform limits")
                            })?)
                            .ok_or_else(|| overflow("resident edit Peq workspace overflows"))?;
                    let slots = Buffer::<u64>::zeros(lease, len)?;
                    lev.slots = slots.address();
                    lev.slot_words = slot_words;
                    Some(slots)
                } else {
                    None
                };
                // In-pass fields of this bucket's compact items (§13.4 step 1).
                let ends = if options.b_end {
                    Some(Buffer::<i64>::allocate(lease, count_usize)?)
                } else {
                    None
                };
                let counts = if options.n_ends {
                    Some(Buffer::<i32>::allocate(lease, count_usize)?)
                } else {
                    None
                };
                lev.b_end = ends.as_ref().map_or(0, Buffer::address);
                lev.n_ends = counts.as_ref().map_or(0, Buffer::address);
                let key_address = keys.address();
                let mut launch = lease.execution().launch_builder(function);
                launch
                    .arg(&tables)
                    .arg(&selected)
                    .arg(&lev)
                    .arg(&key_address);
                // SAFETY: bucket membership proves the original tier predicate
                // and Q domain, every active unit owns its zeroed raw Peq slot.
                unsafe { launch.launch(launch_config) }
                    .map_err(|e| fail(lease, "running resident Levenshtein fast tier", e))?;
                if in_pass {
                    let (values, ends_address, counts_address) = (
                        keys.address(),
                        ends.as_ref().map_or(0, Buffer::address),
                        counts.as_ref().map_or(0, Buffer::address),
                    );
                    let (dist, b_end, n_ends) = (
                        output.address(),
                        b_end_out.as_ref().map_or(0, DeviceOutput::address),
                        n_ends_out.as_ref().map_or(0, DeviceOutput::address),
                    );
                    let mut launch = lease.execution().launch_builder(&kernels.lev_scatter);
                    launch
                        .arg(&position)
                        .arg(&values)
                        .arg(&ends_address)
                        .arg(&counts_address)
                        .arg(&count)
                        .arg(&dist)
                        .arg(&b_end)
                        .arg(&n_ends);
                    // SAFETY: matches sb_lev_scatter; the compact positions are a
                    // permutation of disjoint chunk positions, every source holds
                    // `count` entries the tier kernel wrote, and a field's source
                    // and destination are both set or both 0.
                    unsafe { launch.launch(grid(lev_scatter_threads, count_usize)) }
                        .map_err(|e| fail(lease, "scattering resident Levenshtein fields", e))?;
                } else {
                    scatter(
                        lease,
                        position,
                        keys.address(),
                        output.address(),
                        count,
                        scatter_threads,
                    )?;
                }
            } else {
                let mut aln = pack.plan.aln;
                let kind = usize::try_from(aln.kind)
                    .map_err(|_| parameter("invalid resident alignment kind"))?;
                let function = if bucket == 0 {
                    &kernels.aln_closed
                } else if bucket == 1 {
                    &kernels.aln_thread[kind]
                } else {
                    &kernels.aln_warp[bucket - 2][kind]
                };
                let multistrip = bucket >= 2 && summary[7 + bucket] > (32u64 << (bucket - 2));
                let columns = if multistrip {
                    summary[14 + bucket]
                        .checked_add(1)
                        .ok_or_else(|| overflow("resident align strip width overflows"))?
                } else {
                    0
                };
                let slot_words = columns
                    .checked_mul(2)
                    .ok_or_else(|| overflow("resident align strip size overflows"))?;
                let slot_bytes = usize::try_from(slot_words)
                    .ok()
                    .and_then(|v| v.checked_mul(4))
                    .ok_or_else(|| overflow("resident align strip bytes exceed platform limits"))?;
                let (launch_config, units) = stage_grid(
                    lease,
                    function,
                    settings,
                    count_usize,
                    bucket >= 2,
                    slot_bytes,
                )?;
                let slots = if multistrip {
                    let len =
                        units
                            .checked_mul(usize::try_from(slot_words).map_err(|_| {
                                overflow("resident strip words exceed platform limits")
                            })?)
                            .ok_or_else(|| overflow("resident align strip workspace overflows"))?;
                    Some(Buffer::<i32>::allocate(lease, len)?)
                } else {
                    None
                };
                aln.slots = slots.as_ref().map_or(0, Buffer::address);
                aln.slot_cols = columns;
                let key_address = keys.address();
                let mut launch = lease.execution().launch_builder(function);
                launch
                    .arg(&tables)
                    .arg(&selected)
                    .arg(&aln)
                    .arg(&key_address);
                // SAFETY: compact membership proves CLOSED/THREAD/Q precisely;
                // multistrip carries have one disjoint full-width slot per
                // active warp and are initialized by the preceding strip.
                unsafe { launch.launch(launch_config) }
                    .map_err(|e| fail(lease, "running resident alignment fast tier", e))?;
                scatter(
                    lease,
                    position,
                    keys.address(),
                    output.address(),
                    count,
                    scatter_threads,
                )?;
            }
        }
        lease.synchronize()?;
        finish(output, b_end_out, n_ends_out, summary[6])
    })
}
