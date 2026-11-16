//! Device-resident primer3 T2/K2 stages, including the old K2 HOST tier.
//!
//! Only 25 integer scheduling scalars cross to the host. Work IDs, complete
//! cell tables, selection state, keys and reported fields remain on the lease's
//! stream. Inputs must first pass the public whole-input and pair preflight.
use std::sync::Arc;

use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
};
use seqblaze_core::validate::EngineError;

use crate::{
    device::{guard, Device},
    interop::{allocate, scan, BoundDeviceRows, DeviceColumn, DeviceDType, DeviceLease},
    launch::{settings, Settings},
    p3::{P3Pack, SbP3},
    plan::{p3_cap, p3_launches, P3_GROUPS},
    resident::{ResidentFieldSpec, ResidentSpace},
    session::{failure_ordinal, ptr, SbSpace, SbTables},
    Primer3Call,
};

/// Slot budget of the K2 device continuation (unchanged, primer3-cuda.md §22.4 step 4).
const SLOT_BUDGET: usize = 32 << 20;
/// Slot budget of one resident fast launch (§22.4 step 4, (P)).
const FAST_SLOT_BUDGET: u64 = 64 << 20;
/// Depth units per resident thread one resident fast launch may carry (§22.4 step 4,
/// D_res); it replaces `P3_LANE_STEPS` in the fast launch's depth bound only, and
/// enters no route or bucket decision. Basis: the §11 watchdog procedure on the
/// step 1-2 kernels (2026-11-06, §22.4 step 4): from 2^15 on, the launches at the
/// step's points are bound by the wave or the slot budget, not by depth, and the
/// longest measured fast launch took 0.27 s (P).
const D_RES: u64 = 1 << 15;
const FILL_GROUPS: u64 = 32;
const SELECTION_CELLS: u64 = 4096;
const MAX_SLOT: u64 = 10_080 + 10_001 * 61 * 16;
/// primer3-cuda.md §22.4 step 3: the base g of each width class (w at most 16, 32,
/// 64, and above 64) for a stage call of at most W₃₂ items, `[T2, K2]`; the plan
/// kernel takes the larger of it and the width rule, before the g floor. 0 keeps the
/// width rule. Chosen by the dated measurement recorded in §22.4 step 3 (P).
const G_SMALL: [[u8; 4]; 2] = [[32, 32, 32, 32], [32, 32, 32, 32]];
/// The same above W₃₂ (§22.4 step 4). 0 keeps the width rule: under one-wave
/// sizing no larger g beat it in any class (dated measurement in §22.4 step 4).
const G_BIG: [[u8; 4]; 2] = [[0; 4], [0; 4]];

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Plan {
    buckets: u64,
    stats: u64,
    cap: u64,
    group_lanes: u64,
    unary: u64,
    classes: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Resume {
    items: u64,
    first: u64,
    n: u64,
    state: u64,
    steps: u64,
}
// SAFETY: these repr(C) plain-u64 records mirror the two new CUDA ABIs.
unsafe impl DeviceRepr for Plan {}
unsafe impl DeviceRepr for Resume {}
const _: () = {
    assert!(std::mem::size_of::<Plan>() == 48);
    assert!(std::mem::size_of::<Resume>() == 40);
};

/// Raw-pointer ABI records cannot register every cudarc borrow event. Retain
/// packs and scratch through consumer completion, including any early error.
struct Pending<T> {
    value: Option<T>,
    stream: Arc<CudaStream>,
    device: Device,
}
impl<T> Pending<T> {
    fn new(lease: &DeviceLease, value: T) -> Self {
        Self {
            value: Some(value),
            stream: Arc::clone(lease.stream()),
            device: lease.device().clone(),
        }
    }
    fn get(&self) -> &T {
        self.value
            .as_ref()
            .expect("pending value is retained until drop")
    }
    fn get_mut(&mut self) -> &mut T {
        self.value
            .as_mut()
            .expect("pending value is retained until drop")
    }
}
impl<T> Drop for Pending<T> {
    fn drop(&mut self) {
        if guard(|| {
            crate::interop::execution_stream::retire_in_drop(&self.stream)
                .map_err(|e| failure_ordinal(&self.device, "retiring resident P3 raw owners", e))
        })
        .is_err()
        {
            if let Some(value) = self.value.take() {
                std::mem::forget(value);
            }
        }
    }
}
fn pending<T: DeviceRepr>(
    lease: &DeviceLease,
    len: usize,
) -> Result<Pending<crate::arena::Scratch<T>>, EngineError> {
    Ok(Pending::new(lease, allocate(lease, len)?))
}

fn bad(message: &str) -> EngineError {
    EngineError::Device(format!(
        "resident primer3 stage: {message}; no result was published"
    ))
}
fn failed(lease: &DeviceLease, what: &str, error: cudarc::driver::DriverError) -> EngineError {
    failure_ordinal(lease.device(), what, error)
}
fn require_warp(lease: &DeviceLease, function: &CudaFunction) -> Result<(), EngineError> {
    let cap = function
        .max_threads_per_block()
        .map_err(|e| failed(lease, "reading a resident P3 kernel's block limit", e))?;
    if cap < 32 {
        return Err(bad("the loaded kernel cannot launch one complete warp"));
    }
    Ok(())
}
fn scalar_grid(
    lease: &DeviceLease,
    function: &CudaFunction,
    n: u64,
) -> Result<LaunchConfig, EngineError> {
    let cap = function
        .max_threads_per_block()
        .map_err(|e| failed(lease, "reading a resident P3 kernel's block limit", e))?;
    let threads = u32::try_from(cap).unwrap_or(0).min(128) / 32 * 32;
    if threads == 0 {
        return Err(bad("the loaded scalar kernel cannot launch one warp"));
    }
    Ok(LaunchConfig {
        grid_dim: (n.div_ceil(threads as u64).clamp(1, 65535) as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    })
}
fn warp_grid(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// All numerical fields and the integer ordering key are independently owned
/// device arrays. `found` is byte-wide Bool, rather than the old stage's i32.
pub struct ResidentPrimer3Output {
    pub keys: DeviceColumn,
    pub found: DeviceColumn,
    pub dg_kcal: DeviceColumn,
    pub dh_kcal: DeviceColumn,
    pub ds_cal_per_k: DeviceColumn,
    pub tm_c: DeviceColumn,
    pub passed: Option<DeviceColumn>,
}

/// Own the exact existing P3 upload for every chunk and every TopK replay.
pub struct ResidentPrimer3Stage {
    pack: Pending<P3Pack>,
    settings: Settings,
    ordinal: usize,
    stream: u64,
}
impl ResidentPrimer3Stage {
    pub fn new(lease: &DeviceLease, call: &Primer3Call) -> Result<Self, EngineError> {
        guard(|| {
            Ok(Self {
                pack: Pending::new(
                    lease,
                    P3Pack::upload_on(
                        lease.stream(),
                        lease.device(),
                        call,
                        Some(lease.execution()),
                    )?,
                ),
                settings: settings(),
                ordinal: lease.device().info().ordinal,
                stream: lease.stream_handle(),
            })
        })
    }

    /// Pair preflight must validate the whole input, then every selected pair's
    /// existing (1..=10000, at least one <=60) domain before entering this stage.
    pub fn stage(
        &self,
        lease: &DeviceLease,
        a: &BoundDeviceRows,
        b: &BoundDeviceRows,
        space: ResidentSpace,
        max_lengths: [u64; 2],
    ) -> Result<ResidentPrimer3Output, EngineError> {
        if max_lengths.into_iter().any(|n| n > 10000) {
            return Err(bad("whole-input K2 length validation was not completed"));
        }
        self.run(lease, a, b, space, None)
    }

    /// T2's whole table must have passed its exact public 1..=60 validation.
    /// Its convenience ordering key is the K2 bit key of dg_kcal; the unary
    /// result assembler need not expose that internal column.
    pub fn hairpin(
        &self,
        lease: &DeviceLease,
        a: &BoundDeviceRows,
        first: u64,
        n: u64,
        max_len: u64,
        request: (&[ResidentFieldSpec], Option<i64>),
    ) -> Result<ResidentPrimer3Output, EngineError> {
        if max_len > 60 || first.checked_add(n).is_none_or(|end| end > a.rows.rows) {
            return Err(bad(
                "whole-input T2 length/range validation was not completed",
            ));
        }
        self.run(
            lease,
            a,
            a,
            ResidentSpace {
                kind: 0,
                k0: first,
                n,
                ..ResidentSpace::default()
            },
            Some(request),
        )
    }

    fn run(
        &self,
        lease: &DeviceLease,
        a: &BoundDeviceRows,
        b: &BoundDeviceRows,
        space: ResidentSpace,
        unary_request: Option<(&[ResidentFieldSpec], Option<i64>)>,
    ) -> Result<ResidentPrimer3Output, EngineError> {
        guard(|| {
            let unary = unary_request.is_some();
            if self.ordinal != lease.device().info().ordinal || self.stream != lease.stream_handle()
            {
                return Err(bad("a stage must use its original device and stream"));
            }
            space
                .k0
                .checked_add(space.n)
                .ok_or_else(|| bad("work range overflows"))?;
            let n = usize::try_from(space.n).map_err(|_| bad("work count exceeds the platform"))?;
            let keys = lease.output(DeviceDType::I64, vec![n])?;
            let found = lease.output(DeviceDType::Bool, vec![n])?;
            let dg = lease.output(DeviceDType::F64, vec![n])?;
            let dh = lease.output(DeviceDType::F64, vec![n])?;
            let ds = lease.output(DeviceDType::F64, vec![n])?;
            let tm = lease.output(DeviceDType::F64, vec![n])?;
            // T2's native cutoff decisions feed Each/Hits selection;
            // fields stay unmasked until the resident sink consumes them.
            let passed = if n != 0 && unary_request.is_some_and(|(_, cutoff)| cutoff.is_some()) {
                Some(lease.output(DeviceDType::Bool, vec![n])?)
            } else {
                None
            };
            if n != 0 {
                let found_i32 = pending::<i32>(lease, n)?;
                #[cfg(feature = "testing")]
                if let (Some((requested, _)), Some(byte)) = (unary_request, crate::launch::poison())
                {
                    for (name, address, width) in [
                        ("found", ptr(found_i32.get(), lease.stream()), 4usize),
                        ("dg_kcal", dg.address(), 8),
                        ("dh_kcal", dh.address(), 8),
                        ("ds_cal_per_k", ds.address(), 8),
                        ("tm_c", tm.address(), 8),
                    ] {
                        // Poison hidden primary fields as well; the dedicated
                        // counter counts only the requested public stage slots.
                        let bytes = n * width;
                        // SAFETY: each retained allocation covers n stage values.
                        unsafe { lease.execution().memset(address, byte, bytes) }
                            .map_err(|e| failed(lease, "refilling resident T2 output", e))?;
                        crate::launch::count_poisoned(bytes as u64);
                        if requested.iter().any(|field| field.name == name) {
                            crate::launch::count_unary_refill(0, bytes as u64, 0);
                        }
                    }
                    if let Some(passed) = &passed {
                        // SAFETY: the native T2 stage writes all n passed bytes.
                        unsafe { lease.execution().memset(passed.address(), byte, n) }
                            .map_err(|e| failed(lease, "refilling resident T2 passed", e))?;
                        crate::launch::count_poisoned(n as u64);
                        crate::launch::count_unary_refill(0, 0, n as u64);
                    }
                    crate::launch::count_unary_refill(n as u64, 0, 0);
                }
                let mut tables = SbTables::default();
                tables.a_view = a.address();
                tables.b_view = b.address();
                let space_abi = space.abi();
                let params = SbP3 {
                    key: keys.address(),
                    found: ptr(found_i32.get(), lease.stream()),
                    dg: dg.address(),
                    dh: dh.address(),
                    ds: ds.address(),
                    tm: tm.address(),
                    // The producer computes all values. Sinks apply their own
                    // cutoff/masking policy without changing the stage's bytes.
                    has_cutoff: 0,
                    mask_each: 0,
                    passed: 0,
                    ..self.pack.get().params
                };
                let params = if let Some(passed) = &passed {
                    let key = unary_request
                        .and_then(|(_, cutoff)| cutoff)
                        .expect("cutoff buffer");
                    let bits = if key < 0 {
                        (1u64 << 63) | key.unsigned_abs()
                    } else {
                        key as u64
                    };
                    SbP3 {
                        passed: passed.address(),
                        has_cutoff: 1,
                        cutoff: bits,
                        ..params
                    }
                } else {
                    params
                };
                self.compute(lease, tables, space_abi, params, unary)?;
                let function = &lease.device().0.kernels.resident_p3_columns;
                let out_found = found.address();
                let unary_arg = u64::from(unary);
                let mut launch = lease.execution().launch_builder(function);
                launch
                    .arg(&params)
                    .arg(&space.n)
                    .arg(&out_found)
                    .arg(&unary_arg);
                // SAFETY: every input field was initialized by exactly one
                // classified stage. This launch converts only the found dtype
                // and, for T2, initializes its internal dg ordering key.
                unsafe { launch.launch(scalar_grid(lease, function, space.n)?) }
                    .map_err(|e| failed(lease, "finishing resident P3 columns", e))?;
            }
            // SAFETY: all output writes are queued on the lease stream, and
            // no mutable aliases to these owned arrays are published.
            unsafe {
                Ok(ResidentPrimer3Output {
                    keys: keys.finish()?,
                    found: found.finish()?,
                    dg_kcal: dg.finish()?,
                    dh_kcal: dh.finish()?,
                    ds_cal_per_k: ds.finish()?,
                    tm_c: tm.finish()?,
                    passed: passed.map(|output| output.finish()).transpose()?,
                })
            }
        })
    }

    fn compute(
        &self,
        lease: &DeviceLease,
        tables: SbTables,
        space: SbSpace,
        params: SbP3,
        unary: bool,
    ) -> Result<(), EngineError> {
        let n = usize::try_from(space.n).map_err(|_| bad("work count exceeds the platform"))?;
        let buckets = pending::<u8>(lease, n)?;
        // Only fixed integer control data is uploaded/downloaded. No sequence,
        // index-list entry, computed field or key makes a host round trip.
        let stats = Pending::new(
            lease,
            lease
                .execution()
                .alloc_filled::<u64>(25, 0)
                .map_err(|e| failed(lease, "initializing resident P3 scheduling counters", e))?,
        );
        // §22.4 step 3: at most W₃₂ items each hold a whole resident warp at once, so
        // a larger g costs no extra wave. g is a launch parameter only.
        let family = usize::from(!unary);
        let table = if space.n <= lease.device().p3_wave()[family] {
            G_SMALL[family]
        } else {
            G_BIG[family]
        };
        let plan = Plan {
            buckets: ptr(buckets.get(), lease.stream()),
            stats: ptr(stats.get(), lease.stream()),
            cap: p3_cap(&self.settings),
            group_lanes: self.settings.group_lanes,
            unary: u64::from(unary),
            classes: u64::from_le_bytes([table[0], table[1], table[2], table[3], 0, 0, 0, 0]),
        };
        let kernels = &lease.device().0.kernels;
        let mut launch = lease.execution().launch_builder(&kernels.resident_p3_plan);
        launch.arg(&tables).arg(&space).arg(&params).arg(&plan);
        // SAFETY: validated descriptors cover all rows and the plan's two
        // allocations cover n bucket bytes and 25 initialized control integers.
        unsafe { launch.launch(scalar_grid(lease, &kernels.resident_p3_plan, space.n)?) }
            .map_err(|e| failed(lease, "classifying resident P3 items", e))?;
        let mut host = [0u64; 25];
        lease
            .execution()
            .memcpy_dtoh(stats.get(), &mut host[..])
            .map_err(|e| failed(lease, "reading resident P3 scheduling scalars", e))?;
        lease.synchronize()?;
        let total = host.chunks_exact(5).try_fold(0u64, |sum, row| {
            sum.checked_add(row[0])
                .ok_or_else(|| bad("device bucket counts overflow"))
        })?;
        if total != space.n || (unary && host[20] != 0) {
            return Err(bad("device bucket counts do not cover the requested work"));
        }
        for bucket in 0..5 {
            let info = &host[bucket * 5..bucket * 5 + 5];
            let count =
                usize::try_from(info[0]).map_err(|_| bad("bucket count exceeds the platform"))?;
            if count == 0 {
                continue;
            }
            let slot = info[1]
                .checked_add(15)
                .ok_or_else(|| bad("slot alignment overflows"))?
                / 16
                * 16;
            if slot == 0 || slot > MAX_SLOT || info[3] > 20000 || info[4] > 600000 {
                return Err(bad(
                    "validated pair dimensions exceed the existing P3 domain",
                ));
            }
            let items = compact(lease, buckets.get(), space.n, bucket as u64, count)?;
            if bucket < 4 {
                self.fast(
                    lease,
                    tables,
                    space,
                    params,
                    unary,
                    bucket,
                    items.get(),
                    count,
                    slot,
                    info[2],
                )?;
            } else {
                self.resume(
                    lease,
                    tables,
                    space,
                    params,
                    items.get(),
                    count,
                    slot,
                    info[3],
                    info[4],
                )?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn fast(
        &self,
        lease: &DeviceLease,
        tables: SbTables,
        space: SbSpace,
        params: SbP3,
        unary: bool,
        variant: usize,
        items: &CudaSlice<u64>,
        count: usize,
        slot: u64,
        depth: u64,
    ) -> Result<(), EngineError> {
        let kernels = &lease.device().0.kernels;
        let function = if unary {
            &kernels.resident_p3_hairpin[variant]
        } else {
            &kernels.resident_p3_dimer[variant]
        };
        require_warp(lease, function)?;
        // The existing slot/launch planner is reused after closing a bounded
        // subchunk. At most one item belongs to each live group in that chunk.
        // 32 threads also respects every loaded function's checked block cap.
        // §22.4 step 4: a launch never exceeds one resident wave of the chosen
        // kernel; the slot budget is FAST_SLOT_BUDGET and the depth bound uses D_RES.
        let g = P3_GROUPS[variant];
        let per_block = 32 / g;
        let slot_usize = usize::try_from(slot).map_err(|_| bad("slot exceeds the platform"))?;
        let inner = &lease.device().0;
        let by_bytes = (FAST_SLOT_BUDGET / slot / per_block).max(1) * per_block;
        let by_wave = inner.p3_sms * inner.p3_wave_blocks[usize::from(!unary)][variant] * per_block;
        let budget = self
            .settings
            .cells_per_launch
            .min(inner.resident_threads.max(1).saturating_mul(D_RES));
        let by_depth = (budget / depth.max(1) / g).max(per_block);
        let by_items = (count as u64).div_ceil(per_block) * per_block;
        let groups = by_bytes
            .min(by_wave)
            .min(by_depth)
            .min(by_items)
            .min(65535 * per_block)
            / per_block
            * per_block;
        let groups = usize::try_from(groups).map_err(|_| bad("slot count exceeds the platform"))?;
        let slots = pending::<u64>(
            lease,
            groups
                .checked_mul(slot_usize / 8)
                .ok_or_else(|| bad("fast slot bytes overflow"))?,
        )?;
        for first in (0..count).step_by(groups) {
            let take = (count - first).min(groups);
            for launch in p3_launches(&self.settings, take, g, 32, slot, depth) {
                let p = SbP3 {
                    items: ptr(items, lease.stream()) + (first + launch.start) as u64 * 8,
                    count: launch.len as u64,
                    slots: ptr(slots.get(), lease.stream()),
                    slot: launch.slot,
                    ..params
                };
                let mut builder = lease.execution().launch_builder(function);
                builder.arg(&tables).arg(&space).arg(&p);
                // SAFETY: all item IDs were compacted on the device; the
                // unchanged planner's block/group layout fits the retained slots.
                unsafe {
                    builder.launch(LaunchConfig {
                        grid_dim: (launch.blocks as u32, 1, 1),
                        block_dim: ((launch.per_block * g) as u32, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(|e| failed(lease, "evaluating a resident P3 fast bucket", e))?;
            }
        }
        // The bucket's kernels run while the call polls (T7); the slot guard's
        // Drop-path wait then finds the stream idle.
        crate::interop::execution_stream::idle_owned_stream(lease.stream());
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn resume(
        &self,
        lease: &DeviceLease,
        tables: SbTables,
        space: SbSpace,
        params: SbP3,
        items: &CudaSlice<u64>,
        count: usize,
        slot: u64,
        fill_groups: u64,
        selection_cells: u64,
    ) -> Result<(), EngineError> {
        let kernels = &lease.device().0.kernels;
        for function in [
            &kernels.resident_p3_init,
            &kernels.resident_p3_fill,
            &kernels.resident_p3_select,
            &kernels.resident_p3_finish,
        ] {
            require_warp(lease, function)?;
        }
        let slot_usize = usize::try_from(slot).map_err(|_| bad("slot exceeds the platform"))?;
        let live = (SLOT_BUDGET / slot_usize).max(1).min(count);
        let slots = pending::<u64>(
            lease,
            live.checked_mul(slot_usize / 8)
                .ok_or_else(|| bad("continuation slot bytes overflow"))?,
        )?;
        let state = pending::<u64>(
            lease,
            live.checked_mul(9)
                .ok_or_else(|| bad("continuation state bytes overflow"))?,
        )?;
        let p = SbP3 {
            slots: ptr(slots.get(), lease.stream()),
            slot,
            ..params
        };
        for first in (0..count).step_by(live) {
            let take = (count - first).min(live);
            let mut r = Resume {
                items: ptr(items, lease.stream()),
                first: first as u64,
                n: take as u64,
                state: ptr(state.get(), lease.stream()),
                steps: FILL_GROUPS,
            };
            let mut builder = lease.execution().launch_builder(&kernels.resident_p3_init);
            builder.arg(&tables).arg(&space).arg(&p).arg(&r);
            // SAFETY: one complete warp owns each disjoint retained slot/state.
            unsafe { builder.launch(warp_grid(take)) }
                .map_err(|e| failed(lease, "initializing resident P3 continuation", e))?;
            for _ in 0..fill_groups.div_ceil(FILL_GROUPS) {
                let mut builder = lease.execution().launch_builder(&kernels.resident_p3_fill);
                builder.arg(&p).arg(&r);
                // SAFETY: each launch advances at most 32 cell groups, with
                // every complete candidate fold kept inside the original helper.
                unsafe { builder.launch(warp_grid(take)) }
                    .map_err(|e| failed(lease, "advancing resident P3 cells", e))?;
            }
            r.steps = SELECTION_CELLS;
            for _ in 0..selection_cells.div_ceil(SELECTION_CELLS) {
                let mut builder = lease
                    .execution()
                    .launch_builder(&kernels.resident_p3_select);
                builder.arg(&p).arg(&r);
                // SAFETY: strict row-major selection carries score and first
                // winner between stream-ordered, bounded scans.
                unsafe { builder.launch(warp_grid(take)) }
                    .map_err(|e| failed(lease, "selecting resident P3 cells", e))?;
            }
            let mut builder = lease
                .execution()
                .launch_builder(&kernels.resident_p3_finish);
            builder.arg(&p).arg(&r);
            // SAFETY: the full table and chosen endpoint remain live; a legal
            // pair's unchanged backtrace has at most 60 predecessor steps.
            unsafe { builder.launch(warp_grid(take)) }
                .map_err(|e| failed(lease, "finishing resident P3 continuation", e))?;
        }
        // As in `fast`: wait for the continuation with polls before the slot
        // and state guards drop.
        crate::interop::execution_stream::idle_owned_stream(lease.stream());
        Ok(())
    }
}

fn compact(
    lease: &DeviceLease,
    buckets: &CudaSlice<u8>,
    n: u64,
    bucket: u64,
    count: usize,
) -> Result<Pending<crate::arena::Scratch<u64>>, EngineError> {
    let len = usize::try_from(n).map_err(|_| bad("work count exceeds the platform"))?;
    let mut flags = pending::<u64>(lease, len)?;
    let kernels = &lease.device().0.kernels;
    let mut builder = lease.execution().launch_builder(&kernels.resident_p3_flags);
    builder
        .slice_arg(buckets)
        .arg(&n)
        .arg(&bucket)
        .slice_arg(flags.get_mut());
    // SAFETY: one flag is initialized for every live bucket entry.
    unsafe { builder.launch(scalar_grid(lease, &kernels.resident_p3_flags, n)?) }
        .map_err(|e| failed(lease, "marking resident P3 bucket positions", e))?;
    let prefix = Pending::new(lease, scan(lease, flags.get(), len)?);
    let mut items = pending::<u64>(lease, count)?;
    let mut builder = lease
        .execution()
        .launch_builder(&kernels.resident_p3_positions);
    builder
        .slice_arg(flags.get())
        .slice_arg(prefix.get())
        .arg(&n)
        .slice_arg(items.get_mut());
    // SAFETY: exclusive ranks are distinct and the already downloaded bucket
    // count is their exact upper bound. No work ID reaches the host.
    unsafe { builder.launch(scalar_grid(lease, &kernels.resident_p3_positions, n)?) }
        .map_err(|e| failed(lease, "compacting resident P3 work IDs", e))?;
    Ok(items)
}
