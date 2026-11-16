//! Bounded native thermodynamic stages over validated resident inputs.
//!
//! Normal pairs use the existing full-key/F53 CUDA kernels. Work exceeding
//! their watchdog depth stays on the device: only its compacted work IDs and
//! a scalar count are produced, then eight global DP rings at most are resumed
//! one row per launch. No sequence, index list or computed term is downloaded.
use cudarc::driver::{CudaFunction, CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg};
use seqblaze_core::{detail::try_reserve_exact, validate::EngineError};

use std::sync::Arc;

use crate::{
    device::{guard, Device},
    interop::{allocate, scan, BoundDeviceRows, DeviceColumn, DeviceDType, DeviceLease},
    pack_cache::{self, Kind, Pack},
    resident::ResidentSpace,
    session::{failure_ordinal, ptr},
    DuplexCall, PfDeviceCall,
};

const MAX_LENGTH: usize = 200;
const RESUME_SLOTS: usize = 8;
/// The fast-ring budget B of duplex-cuda.md §15.4 step 1 **(P)**: fast slots of one
/// stage call stay within 256 MiB (plus at most three slots of rounding).
const FAST_BUDGET: u64 = 256 << 20;
/// Groups (one warp each) per fast block: 128-thread blocks.
const FAST_GROUPS: u64 = 4;
/// Bytes of the position-list area per group after the staged codes (§15.4 step 3,
/// `SB_SPARSE_LISTS`).
pub(crate) const LISTS: usize = 1616;
/// U_pass of §15.4 step 1 **(P)**: the per-unit depth budget of one fast launch, per pass
/// (separate from the routing cap), set by the watchdog procedure on the worst-case input
/// A^L against T^L (every cell and every predecessor pairs, so the sparse walk visits D
/// candidates): with the provisional 2^21, a narrow launch of three 200-nt pairs per group
/// took 521 ms and a detail launch of two 678 ms (limit 0.5 s), so every K1 pass uses 2^20.
const UNIT_DETAIL: u64 = 1 << 20;
const UNIT_SCORE: u64 = 1 << 20;
/// K3 (duplex-pf-cuda.md §19.4 step 1), by pack format: 2^20 for N_C = 1; format 2 (N_C = 3)
/// costs about twice as much per candidate (60 nt, 17 pairs per group: 783 ms), so 2^19.
const UNIT_PF: u64 = 1 << 20;
const UNIT_PF_CLASSES: u64 = 1 << 19;
/// Staged bytes per pair (q): the codes (400; K3 always), and with STATES also the K1 run
/// lengths (800; nndb2024-dna.md §5.4.4).
const CODES: usize = 400;
const CODES_STATES: usize = 800;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct Tables {
    a_data: u64,
    a_off: u64,
    b_data: u64,
    b_off: u64,
    a_view: u64,
    b_view: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct DupParams {
    score: u64,
    full: u64,
    lam: u64,
    end_l: u64,
    end_r: u64,
    stack: u64,
    bulge: u64,
    i11: u64,
    i12: u64,
    i21: u64,
    i22: u64,
    gen: u64,
    loop_out: u64,
    loop_in: u64,
    s: u64,
    kappa4: u64,
    kappa1: u64,
    group_lanes: u64,
    band_bytes: u64,
    pair_cells: u64,
    s_max: u64,
    s_block: u64,
    tier: u64,
    stride: u64,
    slots: u64,
    slot: u64,
    dh: u64,
    ds: u64,
    tm: u64,
    poison: u64,
    n_p: u64,
    n_c: u64,
    cls: u64,
    states: u64,
    has_states: u64,
    codes: u64,
    /// C1 bounded resident stage: u8* skip flags of the launch's items, or 0.
    skip: u64,
    #[cfg(feature = "testing")]
    resident_stats: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PfParams {
    w: u64,
    lam: u64,
    start: u64,
    end: u64,
    stack: u64,
    bulge: u64,
    i11: u64,
    i12: u64,
    i21: u64,
    i22: u64,
    gen: u64,
    loop_out: u64,
    loop_in: u64,
    group_lanes: u64,
    band_bytes: u64,
    pair_cells: u64,
    s_max: u64,
    s_block: u64,
    tier: u64,
    stride: u64,
    slots: u64,
    slot: u64,
    out: u64,
    poison: u64,
    n_p: u64,
    n_c: u64,
    cls: u64,
    /// C2 bounded resident stage: u8* skip flags of the launch's items, or 0.
    skip: u64,
    #[cfg(feature = "testing")]
    resident_stats: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Resume {
    positions: u64,
    first: u64,
    n: u64,
    codes: u64,
    band: u64,
    slot: u64,
    meta: u64,
    best: u64,
    row: u64,
}
// SAFETY: these repr(C) plain-u64 records mirror SbTables, SbDup, SbPf and
// SbResidentResume, respectively, without pointers to host memory.
unsafe impl DeviceRepr for Tables {}
unsafe impl DeviceRepr for DupParams {}
unsafe impl DeviceRepr for PfParams {}
unsafe impl DeviceRepr for Resume {}
const _: () = {
    assert!(std::mem::size_of::<Tables>() == 48);
    assert!(
        std::mem::size_of::<DupParams>() == 296 + if cfg!(feature = "testing") { 8 } else { 0 }
    );
    assert!(std::mem::size_of::<PfParams>() == 224 + if cfg!(feature = "testing") { 8 } else { 0 });
    assert!(std::mem::size_of::<Resume>() == 72);
    assert!(std::mem::offset_of!(DupParams, dh) == 208);
    assert!(std::mem::offset_of!(PfParams, out) == 176);
};

fn failed(lease: &DeviceLease, what: &str, cause: cudarc::driver::DriverError) -> EngineError {
    failure_ordinal(lease.device(), what, cause)
}

/// Device-written observations, separate from allocation poison. Testing builds
/// collect only these six scalars after a stage, never sequences or DP state.
#[cfg(feature = "testing")]
struct PoisonStats(CudaSlice<u64>);
#[cfg(feature = "testing")]
impl PoisonStats {
    fn new(lease: &DeviceLease) -> Result<Self, EngineError> {
        lease
            .execution()
            .alloc_filled::<u64>(6, 0)
            .map(Self)
            .map_err(|e| failed(lease, "initializing resident poison observations", e))
    }

    fn collect(&self, lease: &DeviceLease, family: usize) -> Result<(), EngineError> {
        let mut counts = [0u64; 6];
        lease
            .execution()
            .memcpy_dtoh(&self.0, &mut counts[..])
            .map_err(|e| failed(lease, "reading resident poison observations", e))?;
        lease.synchronize()?;
        crate::launch::count_resident_poison(family, counts);
        Ok(())
    }
}
fn bad(message: &str) -> EngineError {
    EngineError::Device(format!(
        "resident thermodynamic stage: {message}; no result was published"
    ))
}
/// Every resident kernel launches complete warps: `threads` (32 or 128) per block.
fn require_block(
    lease: &DeviceLease,
    function: &CudaFunction,
    threads: i32,
) -> Result<(), EngineError> {
    let cap = function
        .max_threads_per_block()
        .map_err(|e| failed(lease, "reading a resident kernel's block limit", e))?;
    if cap < threads {
        return Err(bad(if threads == 32 {
            "a loaded resident kernel cannot launch one complete warp"
        } else {
            "a loaded resident kernel cannot launch four complete warps"
        }));
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
        .map_err(|e| failed(lease, "reading a resident kernel's block limit", e))?;
    let threads = (u32::try_from(cap).unwrap_or(0).min(128) / 32) * 32;
    if threads == 0 {
        return Err(bad(
            "a loaded resident scalar kernel cannot launch one warp",
        ));
    }
    Ok(LaunchConfig {
        grid_dim: (n.div_ceil(threads as u64).clamp(1, 65535) as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    })
}
/// One warp per pair (continuation launches); `codes` dynamic shared bytes.
fn pair_grid(n: usize, codes: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.max(1) as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: codes as u32,
    }
}
fn table_views(a: &BoundDeviceRows, b: &BoundDeviceRows) -> Tables {
    Tables {
        a_view: a.address(),
        b_view: b.address(),
        ..Tables::default()
    }
}
fn chunk(space: ResidentSpace, first: usize, n: usize) -> Result<ResidentSpace, EngineError> {
    Ok(ResidentSpace {
        k0: space
            .k0
            .checked_add(first as u64)
            .ok_or_else(|| bad("work ID overflow"))?,
        n: n as u64,
        ..space
    })
}
fn dimensions(space: ResidentSpace, max_lengths: [u64; 2]) -> Result<(usize, usize), EngineError> {
    let n = usize::try_from(space.n).map_err(|_| bad("pair count exceeds the platform"))?;
    space
        .k0
        .checked_add(space.n)
        .ok_or_else(|| bad("work range overflows"))?;
    let longest = max_lengths[0].max(max_lengths[1]);
    if longest > MAX_LENGTH as u64 {
        return Err(bad(
            "native strands exceed the existing 200-byte validated domain",
        ));
    }
    Ok((n, longest as usize))
}
// A conservative monotone bound, using the actual legal lengths and fixed
// g=32, against the depth cap `cap` the stage schedules with (2^21, or the
// testing hook's value; nndb2024-dna.md §5.4.7). Only scheduling depends on
// this host scalar calculation; the device route flags decide.
fn may_route(lens: [u64; 2], lam: u64, pf: bool, cap: u64) -> bool {
    let [n, m] = lens;
    if n < 2 || m < 2 {
        return false;
    }
    let candidates: u64 = (0..=lam.min(n - 2)).map(|u| (lam - u).min(m - 2) + 1).sum();
    let forward = n * m.div_ceil(32);
    let steps = if pf {
        forward.max(m * n.div_ceil(32))
    } else {
        forward
    };
    steps * candidates > cap
}

/// The resident fast kernels (duplex-cuda.md §15.4, duplex-pf-cuda.md §19.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FastKernel {
    Narrow = 0,
    Wide = 1,
    Detail = 2,
    Pf = 3,
}
impl FastKernel {
    pub(crate) const ALL: [FastKernel; 4] = [Self::Narrow, Self::Wide, Self::Detail, Self::Pf];
}

pub(crate) fn fast_function(kernels: &crate::device::Kernels, kind: FastKernel) -> &CudaFunction {
    match kind {
        FastKernel::Narrow => &kernels.dup_res[0],
        FastKernel::Wide => &kernels.dup_res[1],
        FastKernel::Detail => &kernels.dup_res[2],
        FastKernel::Pf => &kernels.pf_res,
    }
}

/// Dynamic shared bytes of one group: the staged q bytes and the list area, rounded up
/// to 16 (duplex-cuda.md §15.4 step 1).
fn fast_stride(q: usize) -> usize {
    (q + LISTS).div_ceil(16) * 16
}

/// G_res of one fast kernel at q (§15.4 step 1): SMs × resident 128-thread blocks per SM
/// for 4 × stride dynamic shared bytes × 4. Read once per device (kernel loading).
pub(crate) fn fast_wave(
    kernels: &crate::device::Kernels,
    kind: FastKernel,
    q: usize,
    sms: u64,
) -> Result<u64, cudarc::driver::DriverError> {
    let blocks = fast_function(kernels, kind).occupancy_max_active_blocks_per_multiprocessor(
        (FAST_GROUPS * 32) as u32,
        FAST_GROUPS as usize * fast_stride(q),
        None,
    )?;
    Ok(sms * u64::from(blocks) * FAST_GROUPS)
}

/// N_sh of K3's shared tier (duplex-pf-cuda.md §19.4 step 3) for a band of `band` bytes:
/// SMs × resident one-warp blocks per SM of d = q + lists + band bytes (rounded up to
/// 16), from the driver's occupancy query, cached per d; 0 when d + s_block > S_max.
pub(crate) fn pf_shared_wave(device: &Device, q: u64, band: u64) -> Result<u64, EngineError> {
    let inner = &device.0;
    let d = (q + LISTS as u64 + band).div_ceil(16) * 16;
    if d + inner.pf_res_s_block > inner.dup_s_max {
        return Ok(0);
    }
    let mut cache = inner
        .pf_shared_wave
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    if let Some(&(_, wave)) = cache.iter().find(|(at, _)| *at == d) {
        return Ok(wave);
    }
    let blocks = fast_function(&inner.kernels, FastKernel::Pf)
        .occupancy_max_active_blocks_per_multiprocessor(32, d as usize, None)
        .map_err(|e| failure_ordinal(device, "reading a resident duplex kernel's occupancy", e))?;
    let wave = inner.p3_sms * u64::from(blocks);
    cache.push((d, wave));
    Ok(wave)
}

/// c(n, m, Λ) of duplex-cuda.md §5.6 for n, m ≥ 2: candidates of one cell at most.
fn candidates(n: u64, m: u64, lam: u64) -> u64 {
    (0..=lam.min(n - 2)).map(|u| (lam - u).min(m - 2) + 1).sum()
}

/// The launch plan of one stage call's fast pairs (duplex-cuda.md §15.4 step 1): S slots of
/// `slot` bytes, S/4 blocks of 128 threads, at most `per_launch` pairs per launch.
struct FastPlan {
    slots: usize,
    slot: usize,
    per_launch: usize,
    stride: usize,
}
impl FastPlan {
    /// `depth`: D at the stage's maximum lengths (0 when no pair reaches the DP);
    /// `unit`: U_pass.
    fn new(wave: u64, slot: usize, n: usize, depth: u64, unit: u64, q: usize) -> Self {
        let slot = slot.max(16).div_ceil(16) * 16;
        let (s_test, u_test) = crate::launch::resident_launch();
        let fit = (FAST_BUDGET / slot as u64).max(1);
        let lower = wave.max(1).min(fit).min(n.max(1) as u64).min(s_test);
        let slots = lower.div_ceil(FAST_GROUPS) * FAST_GROUPS;
        let unit = unit.min(u_test);
        let per_group = unit.checked_div(depth).map_or(n as u64, |v| v.max(1));
        let per_launch = slots.saturating_mul(per_group).min(n.max(1) as u64);
        Self {
            slots: slots as usize,
            slot,
            per_launch: per_launch as usize,
            stride: fast_stride(q),
        }
    }
    fn config(&self) -> LaunchConfig {
        LaunchConfig {
            grid_dim: ((self.slots as u64 / FAST_GROUPS) as u32, 1, 1),
            block_dim: ((FAST_GROUPS * 32) as u32, 1, 1),
            shared_mem_bytes: (FAST_GROUPS as usize * self.stride) as u32,
        }
    }
}

/// The continuation's rings (at most eight, independently sized as before, longest ×
/// longest), allocated only when the call has continued pairs.
struct Rings {
    band: crate::arena::Scratch<u64>,
    codes: crate::arena::Scratch<u8>,
    meta: crate::arena::Scratch<u64>,
    best: crate::arena::Scratch<i64>,
    slot: usize,
}
impl Rings {
    /// `pf`: K3's N_C (band ((1 + N_C)·R + 1)·m·16), or None for K1 (R·m·32); `codes`:
    /// the staged bytes per continuation slot (q).
    fn new(
        lease: &DeviceLease,
        resumed: usize,
        max_len: usize,
        lam: usize,
        pf: Option<usize>,
        codes: usize,
    ) -> Result<Self, EngineError> {
        let rows = max_len.min(lam + 2);
        let raw = match pf {
            Some(nc) => ((1 + nc) * rows + 1) * max_len * 16,
            None => rows * max_len * 32,
        };
        let slot = raw.max(16).div_ceil(16) * 16;
        Ok(Self {
            band: allocate(lease, slot * resumed / 8)?,
            codes: allocate(lease, resumed * codes)?,
            meta: allocate(lease, resumed * 6)?,
            best: allocate(lease, resumed * 32 * 4)?,
            slot,
        })
    }
    fn resume(
        &self,
        lease: &DeviceLease,
        positions: &CudaSlice<u64>,
        first: usize,
        n: usize,
    ) -> Resume {
        Resume {
            positions: ptr(positions, lease.stream()),
            first: first as u64,
            n: n as u64,
            codes: ptr(&self.codes, lease.stream()),
            band: ptr(&self.band, lease.stream()),
            slot: self.slot as u64,
            meta: ptr(&self.meta, lease.stream()),
            best: ptr(&self.best, lease.stream()),
            row: 0,
        }
    }
}

/// Compact the old HOST tier completely on the device. Only its total count
/// crosses to the host to bound the number of continuation launches.
fn routes<P: DeviceRepr>(
    lease: &DeviceLease,
    tables: Tables,
    space: ResidentSpace,
    params: P,
    pf: bool,
) -> Result<(crate::arena::Scratch<u64>, usize), EngineError> {
    let n = usize::try_from(space.n).map_err(|_| bad("route count exceeds the platform"))?;
    let mut flags = allocate::<u64>(lease, n)?;
    let kernels = &lease.device().0.kernels;
    let function = if pf {
        &kernels.resident_route_flags_pf
    } else {
        &kernels.resident_route_flags_dup
    };
    let mut launch = lease.execution().launch_builder(function);
    launch
        .arg(&tables)
        .arg(&space)
        .arg(&params)
        .slice_arg_mut(&mut flags);
    // SAFETY: the validated work decoder and row descriptors cover each item;
    // flags has exactly one initialized scalar per work item after this launch.
    unsafe { launch.launch(scalar_grid(lease, function, space.n)?) }
        .map_err(|e| failed(lease, "classifying resident routes", e))?;
    let prefix = scan(lease, &flags, n)?;
    let mut total = [0u64];
    lease
        .execution()
        .memcpy_dtoh(&prefix.slice(n..n + 1), &mut total[..])
        .map_err(|e| failed(lease, "reading the resident route count", e))?;
    lease.synchronize()?;
    let count = usize::try_from(total[0]).map_err(|_| bad("route count exceeds the platform"))?;
    if count > n {
        return Err(bad("invalid device route count"));
    }
    let mut positions = allocate::<u64>(lease, count)?;
    if count != 0 {
        let mut launch = lease
            .execution()
            .launch_builder(&kernels.resident_route_positions);
        launch
            .slice_arg(&flags)
            .slice_arg(&prefix)
            .arg(&space.n)
            .slice_arg_mut(&mut positions);
        // SAFETY: exclusive scan gives each flagged item a distinct slot below
        // count. Unflagged items never address the compacted allocation.
        unsafe {
            launch.launch(scalar_grid(
                lease,
                &kernels.resident_route_positions,
                space.n,
            )?)
        }
        .map_err(|e| failed(lease, "compacting resident routes", e))?;
    }
    Ok((positions, count))
}

pub struct ResidentDuplexOutput {
    pub keys: DeviceColumn,
    /// The detail fields; None when the stage runs the narrow or wide pass (no detail
    /// field requested, duplex-cuda.md §15.4 step 2).
    pub dh_kcal: Option<DeviceColumn>,
    pub ds_cal_per_k: Option<DeviceColumn>,
    pub tm_c: Option<DeviceColumn>,
}

/// The score keys of the narrow or wide fast pass beside CLS (S12 kinds `K1Narrow` and
/// `K1Wide`), held for the kernels' lifetime.
#[allow(dead_code)]
enum ScoreKeys {
    Narrow(Arc<Pack<i32, u8>>),
    Wide(Arc<Pack<i64, u8>>),
}

/// One K1 pack, `keys` beside CLS, through the device pack cache (S12) on the lease's
/// execution stream: a miss uploads both arrays and offers them to the lease; a hit
/// uploads nothing. Each uploaded content has its own kind (`pack_cache::Kind`).
fn k1_pack<T: DeviceRepr + Copy + Send + Sync + 'static>(
    lease: &DeviceLease,
    call: &DuplexCall,
    kind: Kind,
    keys: &[T],
    what: &str,
) -> Result<Arc<Pack<T, u8>>, EngineError> {
    Pack::get(
        lease.stream(),
        lease.device(),
        Some(lease.execution()),
        kind,
        [pack_cache::bytes(keys), &call.cls],
        || {
            // The keys retire under the host-proved guard if the CLS upload fails.
            let first = crate::interop::Retained::new(
                lease,
                lease
                    .execution()
                    .clone_htod(keys)
                    .map_err(|e| failed(lease, what, e))?,
            );
            // CLS keeps its existing failure text in every pass.
            let second = lease
                .execution()
                .clone_htod(&call.cls)
                .map_err(|e| failed(lease, "uploading resident K1 full keys", e))?;
            Ok::<_, EngineError>((first.into_inner(), second))
        },
    )
}

/// The full (D, H, G, Q) keys, flattened, beside CLS (S12 kind `K1`).
fn upload_full(lease: &DeviceLease, call: &DuplexCall) -> Result<Arc<Pack<i64, u8>>, EngineError> {
    let mut flat = Vec::new();
    try_reserve_exact(
        &mut flat,
        call.full
            .len()
            .checked_mul(4)
            .ok_or_else(|| bad("K1 pack size overflows"))?,
        "resident K1 full keys",
    )?;
    for key in &call.full {
        flat.extend_from_slice(key);
    }
    k1_pack(
        lease,
        call,
        Kind::K1,
        &flat,
        "uploading resident K1 full keys",
    )
}

/// The upload is immutable call control (published parameter pack), never
/// input sequences. Reuse this stage for every bounded chunk and TopK replay.
pub struct ResidentDuplexStage {
    call: Arc<DuplexCall>,
    /// Full keys beside CLS: uploaded with the stage for the detail pass, else on first
    /// use by the continuation (only when a chunk has continued pairs).
    full: Option<Arc<Pack<i64, u8>>>,
    _score: Option<ScoreKeys>,
    /// The fast pass: the detail kernel when a detail field is requested, else the
    /// narrow (int32) or wide (int64) score kernel of the call's width (§15.4 step 2).
    kind: FastKernel,
    params: DupParams,
    ordinal: usize,
    stream: u64,
    /// The C1 bound of a bounded call (duplex-mfe-bound.md §8), if any.
    bound: Option<crate::resident_bound::BoundStage>,
}
impl ResidentDuplexStage {
    /// `details`: the request contains `dh_kcal`, `ds_cal_per_k` or `tm_c`.
    pub fn new(
        lease: &DeviceLease,
        call: &Arc<DuplexCall>,
        details: bool,
    ) -> Result<Self, EngineError> {
        guard(|| {
            // Shape, CLS, STATES and every offset (nndb2024-dna.md §5.4.2).
            if !call.layout_ok()
                || call.full.len() != call.offsets[14]
                || call.score.len() != call.offsets[14]
            {
                return Err(bad("inconsistent K1 full-key pack"));
            }
            // The fast pass's pack: its keys' address (0 for the detail pass, which reads
            // `full`) and the CLS address beside them.
            let (kind, full, score, [score_at, cls]) = if details {
                let full = upload_full(lease, call)?;
                let [_, cls] = full.addresses();
                (FastKernel::Detail, Some(full), None, [0, cls])
            } else if call.wide {
                let keys = k1_pack(
                    lease,
                    call,
                    Kind::K1Wide,
                    &call.score,
                    "uploading resident K1 score keys",
                )?;
                let at = keys.addresses();
                (FastKernel::Wide, None, Some(ScoreKeys::Wide(keys)), at)
            } else {
                // The int32 narrowing of the score keys (§5.2): exact when the width is
                // int32, as the `Session` runner's.
                let narrow: Vec<i32> = call
                    .score
                    .iter()
                    .map(|&key| i32::try_from(key))
                    .collect::<Result<_, _>>()
                    .map_err(|_| bad("inconsistent K1 full-key pack"))?;
                let keys = k1_pack(
                    lease,
                    call,
                    Kind::K1Narrow,
                    &narrow,
                    "uploading resident K1 score keys",
                )?;
                let at = keys.addresses();
                (FastKernel::Narrow, None, Some(ScoreKeys::Narrow(keys)), at)
            };
            let o = &call.offsets;
            let has_states = call.has_states();
            let codes = if has_states { CODES_STATES } else { CODES };
            let params = DupParams {
                score: score_at,
                full: full.as_ref().map_or(0, |full| full.addresses()[0]),
                lam: call.loop_max as u64,
                end_l: o[2] as u64,
                end_r: o[3] as u64,
                stack: o[4] as u64,
                bulge: o[5] as u64,
                i11: o[6] as u64,
                i12: o[7] as u64,
                i21: o[8] as u64,
                i22: o[9] as u64,
                gen: o[10] as u64,
                loop_out: o[11] as u64,
                loop_in: o[12] as u64,
                s: call.s_bits,
                kappa4: call.kappa4_bits,
                kappa1: call.kappa1_bits,
                group_lanes: 32,
                pair_cells: crate::launch::RESIDENT_DEPTH_CAP,
                tier: 2,
                stride: codes as u64,
                poison: 256,
                n_p: call.n_p as u64,
                n_c: call.n_c as u64,
                cls,
                states: o[13] as u64,
                has_states: u64::from(has_states),
                codes: codes as u64,
                ..DupParams::default()
            };
            Ok(Self {
                call: Arc::clone(call),
                full,
                _score: score,
                kind,
                params,
                ordinal: lease.device().info().ordinal,
                stream: lease.stream_handle(),
                bound: None,
            })
        })
    }

    /// Attach the C1 bound of a bounded call (duplex-mfe-bound.md §8): every later
    /// chunk is classified first, and the DP skips the items the bound resolved.
    pub(crate) fn set_bound(&mut self, bound: crate::resident_bound::BoundStage) {
        self.bound = Some(bound);
    }

    /// The TopK replay's per-row limits (device addresses), set between the passes.
    pub(crate) fn replay_limits(&mut self, values: u64, active: u64, mirror: bool) {
        if let Some(bound) = self.bound.as_mut() {
            bound.replay_limits(values, active, mirror);
        }
    }

    /// A new pass of the same work begins (the TopK replay).
    pub(crate) fn reset_pass(&mut self) {
        if let Some(bound) = self.bound.as_mut() {
            bound.reset();
        }
    }

    /// The bound's outcome counts of pass 0 and of the replay (None without a bound).
    pub(crate) fn bound_counts(
        &self,
        lease: &DeviceLease,
    ) -> Result<Option<[[u64; 4]; 2]>, EngineError> {
        self.bound
            .as_ref()
            .map(|bound| bound.counts(lease))
            .transpose()
    }

    /// Inputs and work must have passed whole-input/pair validation on this
    /// lease; max_lengths are its two validation-summary maxima. No host table
    /// is consulted, even when the old route predicate selects a long pair.
    pub fn stage(
        &mut self,
        lease: &DeviceLease,
        a: &BoundDeviceRows,
        b: &BoundDeviceRows,
        space: ResidentSpace,
        max_lengths: [u64; 2],
    ) -> Result<ResidentDuplexOutput, EngineError> {
        guard(|| {
            if self.ordinal != lease.device().info().ordinal || self.stream != lease.stream_handle()
            {
                return Err(bad(
                    "a resident stage must use its original device and stream",
                ));
            }
            let (n, longest) = dimensions(space, max_lengths)?;
            let keys = lease.output(DeviceDType::I32, vec![n])?;
            // The detail columns exist only in the detail pass (§15.4 step 2).
            let details = self.kind == FastKernel::Detail;
            let field = || lease.output(DeviceDType::F64, vec![n]);
            let (dh, ds, tm) = if details {
                (Some(field()?), Some(field()?), Some(field()?))
            } else {
                (None, None, None)
            };
            if n != 0 {
                let tables = table_views(a, b);
                // The C1 bound first (duplex-mfe-bound.md §8): resolved items get their
                // key and fields here, and every DP kernel below skips them. A pass that
                // does not use the bound (outside the TopK replay) gives no skip flags.
                // The detail columns exist only in the detail pass (0: none to write).
                let skip = match self.bound.as_mut() {
                    Some(bound) => bound.classify(
                        lease,
                        tables,
                        space,
                        longest as u64,
                        [
                            keys.address(),
                            dh.as_ref().map_or(0, |c| c.address()),
                            ds.as_ref().map_or(0, |c| c.address()),
                            tm.as_ref().map_or(0, |c| c.address()),
                        ],
                    )?,
                    None => None,
                };
                let skip_base = skip.as_ref().map_or(0, |flags| ptr(flags, lease.stream()));
                let codes = self.params.codes as usize;
                let lam = self.params.lam;
                let kind = self.kind;
                let (width, unit) = match kind {
                    FastKernel::Narrow => (4u64, UNIT_SCORE),
                    FastKernel::Wide => (8, UNIT_SCORE),
                    _ => (32, UNIT_DETAIL),
                };
                // K1 keeps the pair's orientation: rows from a, columns from b (§15.4 step 1).
                let [a_max, b_max] = max_lengths;
                let depth = if a_max >= 2 && b_max >= 2 {
                    a_max * b_max.div_ceil(32) * candidates(a_max, b_max, lam)
                } else {
                    0
                };
                let wave =
                    lease.device().0.resident_wave[kind as usize][usize::from(codes != CODES)];
                let plan = FastPlan::new(
                    wave,
                    (a_max.min(lam + 2) * b_max * width) as usize,
                    n,
                    depth,
                    unit,
                    codes,
                );
                let band = allocate::<u64>(lease, plan.slots * plan.slot / 8)?;
                #[cfg(feature = "testing")]
                let stats = PoisonStats::new(lease)?;
                // The depth cap, read when the call is scheduled (nndb2024-dna.md §5.4.7):
                // the device tier predicate and the host's may_route use the same value.
                let cap = crate::launch::resident_depth_cap();
                let params = DupParams {
                    pair_cells: cap,
                    #[cfg(feature = "testing")]
                    resident_stats: ptr(&stats.0, lease.stream()),
                    // Sample on execution: a live Tiles stage may outlast a
                    // change to the process-wide testing poison setting.
                    #[cfg(feature = "testing")]
                    poison: crate::launch::poison().map_or(256, u64::from),
                    slots: ptr(&band, lease.stream()),
                    slot: plan.slot as u64,
                    stride: plan.stride as u64,
                    dh: dh.as_ref().map_or(0, |c| c.address()),
                    ds: ds.as_ref().map_or(0, |c| c.address()),
                    tm: tm.as_ref().map_or(0, |c| c.address()),
                    skip: skip_base,
                    ..self.params
                };
                let output = keys.address();
                let kernels = &lease.device().0.kernels;
                let fast = fast_function(kernels, kind);
                require_block(lease, fast, (FAST_GROUPS * 32) as i32)?;
                for function in [
                    &kernels.resident_dup_init,
                    &kernels.resident_dup_step,
                    &kernels.resident_dup_finish,
                ] {
                    require_block(lease, function, 32)?;
                }
                let mut launch = lease.execution().launch_builder(&kernels.dup_closed);
                launch.arg(&tables).arg(&space).arg(&params).arg(&output);
                // SAFETY: closed-form items initialize their complete output;
                // all other items are assigned to fast or continued DP below.
                unsafe { launch.launch(scalar_grid(lease, &kernels.dup_closed, space.n)?) }
                    .map_err(|e| failed(lease, "closing resident K1 pairs", e))?;
                for first in (0..n).step_by(plan.per_launch) {
                    let count = (n - first).min(plan.per_launch);
                    let sub = chunk(space, first, count)?;
                    let shift = |at: u64| if at == 0 { 0 } else { at + first as u64 * 8 };
                    let p = DupParams {
                        dh: shift(params.dh),
                        ds: shift(params.ds),
                        tm: shift(params.tm),
                        skip: if skip_base == 0 {
                            0
                        } else {
                            skip_base + first as u64
                        },
                        ..params
                    };
                    let out = output + first as u64 * 4;
                    let mut launch = lease.execution().launch_builder(fast);
                    launch.arg(&tables).arg(&sub).arg(&p).arg(&out);
                    // SAFETY: one warp per slot, grid-striding over the launch's items, global
                    // bands; the kernel skips exactly the closed and routed items.
                    unsafe { launch.launch(plan.config()) }
                        .map_err(|e| failed(lease, "evaluating resident K1 pairs", e))?;
                }
                if may_route(max_lengths, params.lam, false, cap) {
                    let (positions, total) = routes(lease, tables, space, params, false)?;
                    if total != 0 {
                        // Continued pairs keep the full-key continuation in every pass
                        // (§15.4 step 2); the score passes fetch the full keys here, once
                        // per stage, through the pack cache (a hit uploads nothing).
                        if self.full.is_none() {
                            self.full = Some(upload_full(lease, &self.call)?);
                        }
                        let params = DupParams {
                            full: self.full.as_ref().map_or(0, |full| full.addresses()[0]),
                            ..params
                        };
                        let rings = Rings::new(
                            lease,
                            total.min(RESUME_SLOTS),
                            longest,
                            lam as usize,
                            None,
                            codes,
                        )?;
                        for first in (0..total).step_by(RESUME_SLOTS) {
                            let count = (total - first).min(RESUME_SLOTS);
                            let mut r = rings.resume(lease, &positions, first, count);
                            let mut launch =
                                lease.execution().launch_builder(&kernels.resident_dup_init);
                            launch.arg(&tables).arg(&space).arg(&params).arg(&r);
                            // SAFETY: the compacted batch contains only validated
                            // non-closed pairs and has <=8 independently sized rings.
                            unsafe { launch.launch(pair_grid(count, 0)) }.map_err(|e| {
                                failed(lease, "initializing resident K1 continuation", e)
                            })?;
                            for row in 0..longest {
                                r.row = row as u64;
                                let mut launch =
                                    lease.execution().launch_builder(&kernels.resident_dup_step);
                                launch.arg(&params).arg(&r);
                                // SAFETY: the prior stream launch completed the
                                // preceding row. Every predecessor is retained in R rows.
                                unsafe { launch.launch(pair_grid(count, 0)) }
                                    .map_err(|e| failed(lease, "advancing a resident K1 row", e))?;
                            }
                            let mut launch = lease
                                .execution()
                                .launch_builder(&kernels.resident_dup_finish);
                            launch.arg(&params).arg(&r).arg(&output);
                            // SAFETY: final lexmin and existing soft-f64 finalize
                            // write every remaining original chunk output exactly once.
                            unsafe { launch.launch(pair_grid(count, 0)) }.map_err(|e| {
                                failed(lease, "finishing resident K1 continuation", e)
                            })?;
                        }
                    }
                }
                #[cfg(feature = "testing")]
                stats.collect(lease, 0)?;
            }
            // SAFETY: every item is initialized by closed/fast/continued kernels
            // queued on this lease, and no published allocation is written again.
            unsafe {
                Ok(ResidentDuplexOutput {
                    keys: keys.finish()?,
                    dh_kcal: dh.map(|c| c.finish()).transpose()?,
                    ds_cal_per_k: ds.map(|c| c.finish()).transpose()?,
                    tm_c: tm.map(|c| c.finish()).transpose()?,
                })
            }
        })
    }
}

pub struct ResidentPfStage {
    _pack: std::sync::Arc<crate::pack_cache::Pack<u64, u8>>,
    params: PfParams,
    ordinal: usize,
    stream: u64,
    /// The C2 bound of a bounded call (duplex-pf-bound.md §8), if any.
    bound: Option<crate::resident_bound::BoundStage>,
}
impl ResidentPfStage {
    pub fn new(lease: &DeviceLease, call: &PfDeviceCall) -> Result<Self, EngineError> {
        guard(|| {
            // Shape, CLS and every offset (nndb2024-dna.md §5.4.2).
            if !call.layout_ok() || call.m.len() != call.offsets[12] || call.e.len() != call.m.len()
            {
                return Err(bad("inconsistent K3 weight pack"));
            }
            let mut flat = Vec::new();
            try_reserve_exact(
                &mut flat,
                call.m
                    .len()
                    .checked_mul(2)
                    .ok_or_else(|| bad("K3 pack size overflows"))?,
                "resident K3 weights",
            )?;
            for (&m, &e) in call.m.iter().zip(&call.e) {
                flat.push(m);
                flat.push(e as u64);
            }
            let pack = crate::pack_cache::Pack::get(
                lease.stream(),
                lease.device(),
                Some(lease.execution()),
                crate::pack_cache::Kind::K3,
                [crate::pack_cache::bytes(&flat), &call.cls],
                || {
                    let first = crate::interop::Retained::new(
                        lease,
                        lease
                            .execution()
                            .clone_htod(&flat)
                            .map_err(|e| failed(lease, "uploading resident K3 weights", e))?,
                    );
                    let second = lease
                        .execution()
                        .clone_htod(&call.cls)
                        .map_err(|e| failed(lease, "uploading resident K3 weights", e))?;
                    Ok::<_, EngineError>((first.into_inner(), second))
                },
            )?;
            let [weights, cls] = pack.addresses();
            let o = &call.offsets;
            let params = PfParams {
                w: weights,
                lam: call.loop_max as u64,
                start: o[0] as u64,
                end: o[2] as u64,
                stack: o[3] as u64,
                bulge: o[4] as u64,
                i11: o[5] as u64,
                i12: o[6] as u64,
                i21: o[7] as u64,
                i22: o[8] as u64,
                gen: o[9] as u64,
                loop_out: o[10] as u64,
                loop_in: o[11] as u64,
                group_lanes: 32,
                pair_cells: crate::launch::RESIDENT_DEPTH_CAP,
                tier: 2,
                stride: CODES as u64,
                poison: 256,
                n_p: call.n_p as u64,
                n_c: call.n_c as u64,
                cls,
                ..PfParams::default()
            };
            Ok(Self {
                _pack: pack,
                params,
                ordinal: lease.device().info().ordinal,
                stream: lease.stream_handle(),
                bound: None,
            })
        })
    }

    /// Attach the C2 bound of a bounded call (duplex-pf-bound.md §8): every later chunk
    /// is classified first, and the DP skips the items the bound resolved.
    pub(crate) fn set_bound(&mut self, bound: crate::resident_bound::BoundStage) {
        self.bound = Some(bound);
    }

    /// The TopK replay's per-row limits (device addresses), set between the passes.
    pub(crate) fn replay_limits(&mut self, values: u64, active: u64, mirror: bool) {
        if let Some(bound) = self.bound.as_mut() {
            bound.replay_limits(values, active, mirror);
        }
    }

    /// A new pass or sweep of the same work begins.
    pub(crate) fn reset_pass(&mut self) {
        if let Some(bound) = self.bound.as_mut() {
            bound.reset();
        }
    }

    /// The bound's outcome counts of pass 0 and of the replay (None without a bound).
    pub(crate) fn bound_counts(
        &self,
        lease: &DeviceLease,
    ) -> Result<Option<[[u64; 4]; 2]>, EngineError> {
        self.bound
            .as_ref()
            .map(|bound| bound.counts(lease))
            .transpose()
    }

    /// Returns interleaved normalized (M,e) F53 values, U64[n,2]. Their device
    /// addresses feed PfTermView { mantissa:address, exponent:address+8,
    /// stride_bytes:16,n } directly; no host term/finalize channel is involved.
    pub fn stage(
        &mut self,
        lease: &DeviceLease,
        a: &BoundDeviceRows,
        b: &BoundDeviceRows,
        space: ResidentSpace,
        max_lengths: [u64; 2],
    ) -> Result<DeviceColumn, EngineError> {
        guard(|| {
            if self.ordinal != lease.device().info().ordinal || self.stream != lease.stream_handle()
            {
                return Err(bad(
                    "a resident stage must use its original device and stream",
                ));
            }
            let (n, longest) = dimensions(space, max_lengths)?;
            let terms = lease.output(DeviceDType::U64, vec![n, 2])?;
            if n != 0 {
                let tables = table_views(a, b);
                // The C2 bound first (duplex-pf-bound.md §8): resolved items get skip
                // flags, so the closed-form kernel writes their Z-hat = 0 and the DP and
                // routing kernels skip them. A pass that does not use the bound (outside
                // the TopK replay) gives no skip flags.
                let skip = match self.bound.as_mut() {
                    Some(bound) => bound.classify(lease, tables, space, longest as u64, [0; 4])?,
                    None => None,
                };
                let skip_base = skip.as_ref().map_or(0, |flags| ptr(flags, lease.stream()));
                let nc = self.params.n_c;
                let lam = self.params.lam;
                let [a_max, b_max] = max_lengths;
                // b_sym at the stage's maximum lengths (duplex-pf-cuda.md §19.4 step 1): the
                // band of either canonical orientation, ((1 + N_C)·min(n′, Λ + 2) + 1)·m′·16.
                let band = |rows: u64, cols: u64| ((1 + nc) * rows.min(lam + 2) + 1) * cols * 16;
                let b_sym = band(a_max, b_max).max(band(b_max, a_max));
                let depth = if a_max >= 2 && b_max >= 2 {
                    candidates(a_max, b_max, lam)
                        * (a_max * b_max.div_ceil(32)).max(b_max * a_max.div_ceil(32))
                } else {
                    0
                };
                let kind = FastKernel::Pf;
                let inner = &lease.device().0;
                // The shared tier for small chunks (duplex-pf-cuda.md §19.4 step 3): (a) at
                // most N_sh pairs (and, in testing builds, at most S_test), (b) one warp's
                // codes, lists and band b_sym within S_max − s_block.
                let d = (CODES as u64 + LISTS as u64 + b_sym).div_ceil(16) * 16;
                let fits = d + inner.pf_res_s_block <= inner.dup_s_max;
                let shared = fits && {
                    let n_sh = pf_shared_wave(lease.device(), CODES as u64, b_sym)?;
                    (n as u64) <= n_sh.min(crate::launch::resident_launch().0)
                };
                let wave = inner.resident_wave[kind as usize][0];
                let unit = if nc > 1 { UNIT_PF_CLASSES } else { UNIT_PF };
                let plan = FastPlan::new(wave, b_sym as usize, n, depth, unit, CODES);
                let ring = allocate::<u64>(
                    lease,
                    if shared {
                        0
                    } else {
                        plan.slots * plan.slot / 8
                    },
                )?;
                #[cfg(feature = "testing")]
                let stats = PoisonStats::new(lease)?;
                // The depth cap, read when the call is scheduled (nndb2024-dna.md §5.4.7).
                let cap = crate::launch::resident_depth_cap();
                let params = PfParams {
                    pair_cells: cap,
                    #[cfg(feature = "testing")]
                    resident_stats: ptr(&stats.0, lease.stream()),
                    #[cfg(feature = "testing")]
                    poison: crate::launch::poison().map_or(256, u64::from),
                    slots: ptr(&ring, lease.stream()),
                    slot: plan.slot as u64,
                    stride: plan.stride as u64,
                    out: terms.address(),
                    skip: skip_base,
                    ..self.params
                };
                let kernels = &inner.kernels;
                let fast = fast_function(kernels, kind);
                require_block(lease, fast, (FAST_GROUPS * 32) as i32)?;
                for function in [
                    &kernels.resident_pf_init,
                    &kernels.resident_pf_step,
                    &kernels.resident_pf_finish,
                ] {
                    require_block(lease, function, 32)?;
                }
                let mut launch = lease.execution().launch_builder(&kernels.pf_closed);
                launch.arg(&tables).arg(&space).arg(&params);
                // SAFETY: this initializes every closed-form item; remaining
                // terms are written by exactly one of fast/continued DP.
                unsafe { launch.launch(scalar_grid(lease, &kernels.pf_closed, space.n)?) }
                    .map_err(|e| failed(lease, "closing resident K3 pairs", e))?;
                if shared {
                    // Every fast pair passes the device predicate as shared: its band is at
                    // most b_sym = band_bytes and its fits term at most d + s_block <= S_max.
                    let p = PfParams {
                        tier: 1,
                        band_bytes: b_sym,
                        s_max: inner.dup_s_max,
                        s_block: inner.pf_res_s_block,
                        stride: d,
                        slots: 0,
                        slot: 0,
                        ..params
                    };
                    let mut launch = lease.execution().launch_builder(fast);
                    launch.arg(&tables).arg(&space).arg(&p);
                    // SAFETY: one one-warp block per item; each warp keeps its codes, lists
                    // and band in its own d bytes of dynamic shared memory.
                    unsafe {
                        launch.launch(LaunchConfig {
                            grid_dim: (n as u32, 1, 1),
                            block_dim: (32, 1, 1),
                            shared_mem_bytes: d as u32,
                        })
                    }
                    .map_err(|e| failed(lease, "evaluating resident K3 pairs", e))?;
                }
                // The global tier otherwise (§19.4 steps 1-2).
                let global = if shared { 0 } else { n };
                for first in (0..global).step_by(plan.per_launch) {
                    let count = (n - first).min(plan.per_launch);
                    let sub = chunk(space, first, count)?;
                    let p = PfParams {
                        out: params.out + first as u64 * 16,
                        skip: if skip_base == 0 {
                            0
                        } else {
                            skip_base + first as u64
                        },
                        ..params
                    };
                    let mut launch = lease.execution().launch_builder(fast);
                    launch.arg(&tables).arg(&sub).arg(&p);
                    // SAFETY: each warp owns its complete global band and grid-strides over
                    // the launch's items; every accepted pair's depth is <= the cap.
                    unsafe { launch.launch(plan.config()) }
                        .map_err(|e| failed(lease, "evaluating resident K3 pairs", e))?;
                }
                if may_route(max_lengths, params.lam, true, cap) {
                    let (positions, total) = routes(lease, tables, space, params, true)?;
                    if total != 0 {
                        let rings = Rings::new(
                            lease,
                            total.min(RESUME_SLOTS),
                            longest,
                            lam as usize,
                            Some(nc as usize),
                            CODES,
                        )?;
                        for first in (0..total).step_by(RESUME_SLOTS) {
                            let count = (total - first).min(RESUME_SLOTS);
                            let mut r = rings.resume(lease, &positions, first, count);
                            let mut launch =
                                lease.execution().launch_builder(&kernels.resident_pf_init);
                            launch.arg(&tables).arg(&space).arg(&params).arg(&r);
                            // SAFETY: initialized canonical codes, orientation,
                            // ring metadata and E(j), one disjoint slot per pair.
                            unsafe { launch.launch(pair_grid(count, 0)) }.map_err(|e| {
                                failed(lease, "initializing resident K3 continuation", e)
                            })?;
                            for row in 0..longest {
                                r.row = row as u64;
                                let mut launch =
                                    lease.execution().launch_builder(&kernels.resident_pf_step);
                                launch.arg(&params).arg(&r);
                                // SAFETY: same predecessor/fold order as sb_pf_body;
                                // row completion is a stream dependency before reuse.
                                unsafe { launch.launch(pair_grid(count, 0)) }
                                    .map_err(|e| failed(lease, "advancing a resident K3 row", e))?;
                            }
                            let mut launch = lease
                                .execution()
                                .launch_builder(&kernels.resident_pf_finish);
                            launch.arg(&params).arg(&r);
                            // SAFETY: E(0)..E(m-1) folds in its original order and
                            // writes the compacted item's original output position.
                            unsafe { launch.launch(pair_grid(count, 0)) }.map_err(|e| {
                                failed(lease, "finishing resident K3 continuation", e)
                            })?;
                        }
                    }
                }
                #[cfg(feature = "testing")]
                stats.collect(lease, 1)?;
            }
            // SAFETY: all logical term bytes are initialized on this stream;
            // immutable publication preserves the owners for device finalize.
            unsafe { terms.finish() }
        })
    }
}
